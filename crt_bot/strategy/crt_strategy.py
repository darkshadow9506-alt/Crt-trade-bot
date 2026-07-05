"""The multi-timeframe CRT strategy state machine.

Pipeline (HTF -> MTF -> LTF), mirroring the trader's 8-step playbook:

1.  **Bias** -- HTF market direction from swing structure (HH+HL / LH+LL).
2.  **Key levels** on the HTF (FVG / OB / prior-session liquidity) and wait
    for a touch.
3.  **CRT** manipulation candle on the key level -- fixes direction & range.
4.  **CHoCH** on the MTF in the CRT direction.
5.  **Pullback** into the 0.618-0.786 fib zone of the impulse leg.
6.  **CHoCH** on the LTF, then a pullback into a key level (iFVG/FVG zone or
    CISD confirmation).
7.  **BOS** on the LTF -- a close breaking the CHoCH leg's extreme.
8.  **Enter on the BOS** close (``entry.mode: "bos"``, the default). The
    legacy ``"retest"`` mode enters directly on the iFVG/CISD tag instead.

Risk: SL beyond the pullback extreme (+ ATR padding). Target = CRT extreme when
the HTF bias agrees with the trade, otherwise the CRT equilibrium (0.5); in
that counter-trend case the setup is cancelled if price already reached
equilibrium before the entry triggered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..core.indicators import last_atr
from ..core.models import CRTRange, Direction, Signal, SignalState
from ..core.session import SessionSet
from ..core.timeframes import TFSet
from ..smc import crt as crt_mod
from ..smc import fib as fib_mod
from ..smc.cisd import detect_cisd
from ..smc.fvg import find_fvgs, latest_entry_gap
from ..smc.liquidity import detect_sweep
from ..smc.order_block import find_order_blocks
from ..smc.structure import last_choch, structure_events, swing_points


@dataclass
class StrategyParams:
    symbol: str
    tf_set: TFSet
    sessions: SessionSet

    # POI
    use_fvg: bool = True
    use_order_block: bool = True
    use_liquidity_sweep: bool = True
    lookback_sessions: int = 3
    poi_validity_bars: int = 6          # POI must be tagged within this many HTF bars of the CRT

    # CRT
    require_close_inside: bool = True
    min_wick_sweep_atr: float = 0.05

    # structure
    swing_lookback: int = 2

    # fib
    pullback_min: float = 0.618
    pullback_max: float = 0.786
    equilibrium_level: float = 0.5
    skip_if_eq_already_touched: bool = True

    # entry
    #   "bos"    -> LTF CHoCH -> pullback to key level -> enter on the BOS close
    #   "retest" -> enter directly on the iFVG/CISD tag after the LTF CHoCH
    entry_mode: str = "bos"
    use_ifvg: bool = True
    use_cisd: bool = True
    sl_padding_atr: float = 0.1
    min_rr: float = 1.5

    # stage timeouts (in bars of the relevant TF)
    mtf_choch_timeout: int = 24
    pullback_timeout: int = 24
    ltf_choch_timeout: int = 40
    ltf_entry_timeout: int = 60

    @classmethod
    def from_config(cls, cfg: dict, tf_set: TFSet, sessions: SessionSet) -> "StrategyParams":
        s = cfg.get("strategy", {})
        poi = s.get("poi", {})
        crt = s.get("crt", {})
        struct = s.get("structure", {})
        fib = s.get("fib", {})
        entry = s.get("entry", {})
        return cls(
            symbol=cfg.get("symbol", "UNKNOWN"),
            tf_set=tf_set,
            sessions=sessions,
            use_fvg=poi.get("use_fvg", True),
            use_order_block=poi.get("use_order_block", True),
            use_liquidity_sweep=poi.get("use_liquidity_sweep", True),
            lookback_sessions=poi.get("lookback_sessions", 3),
            require_close_inside=crt.get("require_close_inside", True),
            min_wick_sweep_atr=crt.get("min_wick_sweep_atr", 0.05),
            swing_lookback=struct.get("swing_lookback", 2),
            pullback_min=fib.get("pullback_min", 0.618),
            pullback_max=fib.get("pullback_max", 0.786),
            equilibrium_level=fib.get("equilibrium_level", 0.5),
            skip_if_eq_already_touched=fib.get("skip_if_eq_already_touched", True),
            entry_mode=entry.get("mode", "bos"),
            use_ifvg=entry.get("use_ifvg", True),
            use_cisd=entry.get("use_cisd", True),
            sl_padding_atr=entry.get("sl_padding_atr", 0.1),
            min_rr=entry.get("min_rr", 1.5),
        )


@dataclass
class _Setup:
    """Mutable working state for one developing setup."""

    crt: CRTRange
    direction: Direction
    htf_trend_aligned: bool
    crt_time: pd.Timestamp
    htf_bias: Direction | None = None
    bias_basis: str = ""
    fib_zone: tuple[float, float] | None = None
    pullback_extreme: float | None = None   # deepest low (long) / high (short)
    choch_mtf_time: pd.Timestamp | None = None
    choch_ltf_time: pd.Timestamp | None = None
    eq_touched: bool = False                # did price reach the 50% before entry?
    # BOS entry mode: the CHoCH leg's extreme is the BOS reference level; a
    # pullback into a key level must happen before a close beyond it counts.
    ltf_leg_extreme: float | None = None
    ltf_pullback_done: bool = False
    bars_in_state: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def target(self) -> float:
        # trend-aligned -> full CRT range extreme; counter-trend -> 50% equilibrium
        if self.htf_trend_aligned:
            return self.crt.high if self.direction is Direction.LONG else self.crt.low
        return self.crt.equilibrium

    @property
    def tp_mode(self) -> str:
        if self.htf_trend_aligned:
            return "crt_high" if self.direction is Direction.LONG else "crt_low"
        return "equilibrium"


class CRTStrategy:
    """Stateful, look-ahead-free CRT setup detector.

    Call :meth:`update` once per closed LTF candle with the closed views of each
    timeframe. It returns a :class:`Signal` only on the bar the entry triggers.
    """

    def __init__(self, params: StrategyParams):
        self.p = params
        self.state = SignalState.WAIT_CRT
        self.setup: _Setup | None = None
        self._last_seen: dict[str, pd.Timestamp | None] = {
            "htf": None,
            "mtf": None,
            "ltf": None,
        }

    # -- helpers -----------------------------------------------------------
    def _is_new_bar(self, key: str, view: pd.DataFrame) -> bool:
        if view.empty:
            return False
        ts = view.index[-1]
        if self._last_seen[key] != ts:
            self._last_seen[key] = ts
            return True
        return False

    def reset(self) -> None:
        self.state = SignalState.WAIT_CRT
        self.setup = None

    def notify_trade_closed(self) -> None:
        self.reset()

    # -- main update -------------------------------------------------------
    def update(
        self,
        now: pd.Timestamp,
        htf: pd.DataFrame,
        mtf: pd.DataFrame,
        ltf: pd.DataFrame,
    ) -> Signal | None:
        if self.state is SignalState.IN_TRADE:
            return None

        new_htf = self._is_new_bar("htf", htf)
        new_mtf = self._is_new_bar("mtf", mtf)
        new_ltf = self._is_new_bar("ltf", ltf)

        # Global invalidation: price violated the CRT range against us.
        if self.setup is not None and self._invalidated(ltf):
            self.reset()

        # Track whether price has already reached the 50% equilibrium at any
        # point during the setup (used to cancel counter-trend trades).
        if self.setup is not None and new_ltf and not ltf.empty:
            self._track_eq(ltf.iloc[-1])

        if self.state is SignalState.WAIT_CRT:
            if new_htf:
                self._scan_htf(now, htf, ltf)
            return None

        if self.state is SignalState.WAIT_MTF_CHOCH:
            if new_mtf:
                self._scan_mtf_choch(mtf)
            return None

        if self.state is SignalState.WAIT_MTF_PULLBACK:
            if new_mtf:
                self._scan_pullback(mtf)
            return None

        if self.state is SignalState.WAIT_LTF_CHOCH:
            if new_ltf:
                self._scan_ltf_choch(ltf)
            return None

        if self.state is SignalState.WAIT_LTF_ENTRY:
            if new_ltf:
                return self._scan_entry(now, ltf)
            return None

        return None

    # -- stage 1: HTF POI + CRT -------------------------------------------
    def _scan_htf(self, now: pd.Timestamp, htf: pd.DataFrame, ltf: pd.DataFrame) -> None:
        if self.p.sessions.enabled and not self.p.sessions.contains(now):
            return
        if len(htf) < 3:
            return

        atr_v = last_atr(htf)
        crt = crt_mod.detect_crt(
            htf,
            atr_value=atr_v,
            min_wick_sweep_atr=self.p.min_wick_sweep_atr,
            require_close_inside=self.p.require_close_inside,
        )
        if crt is None:
            return

        if not self._poi_confluence(htf, crt.direction, now):
            return

        bias, basis = self._htf_trend(htf)
        aligned = bias is crt.direction
        self.setup = _Setup(
            crt=crt,
            direction=crt.direction,
            htf_trend_aligned=aligned,
            htf_bias=bias,
            bias_basis=basis,
            crt_time=crt.manip_candle_time,
            tags=["crt", "trend_aligned" if aligned else "counter_trend"],
        )
        # seed the 50% pre-touch flag from any LTF action since the CRT formed
        if not aligned and not ltf.empty:
            since = ltf[ltf.index >= self.setup.crt_time]
            for _, bar in since.iterrows():
                self._track_eq(bar)
        self.state = SignalState.WAIT_MTF_CHOCH

    def _poi_confluence(
        self, htf: pd.DataFrame, direction: Direction, now: pd.Timestamp
    ) -> bool:
        """A POI of ``direction`` tagged within the recent HTF window."""
        recent = htf.tail(self.p.poi_validity_bars + 3)
        last = htf.iloc[-1]

        if self.p.use_liquidity_sweep:
            sweep = detect_sweep(htf, self.p.sessions, now, self.p.lookback_sessions)
            if sweep is not None and sweep.direction is direction:
                return True

        if self.p.use_fvg:
            for g in find_fvgs(recent):
                if g.direction is direction and last["low"] <= g.top and last["high"] >= g.bottom:
                    return True

        if self.p.use_order_block:
            for ob in find_order_blocks(recent):
                if ob.direction is direction and last["low"] <= ob.top and last["high"] >= ob.bottom:
                    return True

        # If the user disabled every POI source, don't gate on confluence.
        if not (self.p.use_liquidity_sweep or self.p.use_fvg or self.p.use_order_block):
            return True
        return False

    def _htf_trend(self, htf: pd.DataFrame) -> tuple[Direction | None, str]:
        """HTF market bias from real swing structure -- the way a trader reads
        it by eye:

        * higher highs **and** higher lows  -> bullish  ("HH+HL")
        * lower highs  **and** lower lows    -> bearish  ("LH+LL")

        When structure is mixed or there aren't enough swings yet, fall back to
        a moving-average tiebreak ("MA"). Returns ``(direction, basis)``.
        """
        swings = swing_points(htf, self.p.swing_lookback)
        highs = [s.price for s in swings if s.is_high]
        lows = [s.price for s in swings if not s.is_high]
        if len(highs) >= 2 and len(lows) >= 2:
            hh, hl = highs[-1] > highs[-2], lows[-1] > lows[-2]
            lh, ll = highs[-1] < highs[-2], lows[-1] < lows[-2]
            if hh and hl:
                return Direction.LONG, "HH+HL"
            if lh and ll:
                return Direction.SHORT, "LH+LL"

        # mixed / not enough structure -> moving-average tiebreak
        closes = htf["close"]
        if len(closes) >= 3:
            sma = float(closes.tail(min(len(closes), 50)).mean())
            last = float(closes.iloc[-1])
            if last > sma:
                return Direction.LONG, "MA"
            if last < sma:
                return Direction.SHORT, "MA"
        return None, ""

    # -- stage 2: MTF CHoCH + pullback ------------------------------------
    def _scan_mtf_choch(self, mtf: pd.DataFrame) -> None:
        assert self.setup is not None
        self.setup.bars_in_state += 1
        if self.setup.bars_in_state > self.p.mtf_choch_timeout:
            self.reset()
            return

        ev = last_choch(
            mtf, self.setup.direction, self.p.swing_lookback, after=self.setup.crt_time
        )
        if ev is None:
            return

        # impulse leg = window from CRT to the CHoCH break
        leg = mtf[(mtf.index >= self.setup.crt_time) & (mtf.index <= ev.time)]
        if len(leg) < 2:
            leg = mtf.tail(10)
        leg_low = float(leg["low"].min())
        leg_high = float(leg["high"].max())
        self.setup.fib_zone = fib_mod.fib_pullback_zone(
            leg_low, leg_high, self.setup.direction, self.p.pullback_min, self.p.pullback_max
        )
        self.setup.choch_mtf_time = ev.time
        self.setup.bars_in_state = 0
        self.state = SignalState.WAIT_MTF_PULLBACK

    def _scan_pullback(self, mtf: pd.DataFrame) -> None:
        assert self.setup is not None and self.setup.fib_zone is not None
        self.setup.bars_in_state += 1
        if self.setup.bars_in_state > self.p.pullback_timeout:
            self.reset()
            return

        last = mtf.iloc[-1]
        zone_lo, zone_hi = self.setup.fib_zone
        # price wicks into the fib zone
        tagged = last["low"] <= zone_hi and last["high"] >= zone_lo
        if not tagged:
            return

        # record the pullback extreme (deepest point so far)
        if self.setup.direction is Direction.LONG:
            self.setup.pullback_extreme = float(last["low"])
        else:
            self.setup.pullback_extreme = float(last["high"])
        self.setup.bars_in_state = 0
        self.state = SignalState.WAIT_LTF_CHOCH

    # -- stage 3: LTF CHoCH + entry ---------------------------------------
    def _scan_ltf_choch(self, ltf: pd.DataFrame) -> None:
        assert self.setup is not None
        self.setup.bars_in_state += 1
        if self.setup.bars_in_state > self.p.ltf_choch_timeout:
            self.reset()
            return

        # keep tracking the deepest pullback extreme for the stop
        last = ltf.iloc[-1]
        self._extend_extreme(last)

        ev = last_choch(
            ltf, self.setup.direction, self.p.swing_lookback, after=self.setup.choch_mtf_time
        )
        if ev is None:
            return
        self.setup.choch_ltf_time = ev.time
        self.setup.bars_in_state = 0
        self.state = SignalState.WAIT_LTF_ENTRY

    def _scan_entry(self, now: pd.Timestamp, ltf: pd.DataFrame) -> Signal | None:
        assert self.setup is not None and self.setup.choch_ltf_time is not None
        self.setup.bars_in_state += 1
        if self.setup.bars_in_state > self.p.ltf_entry_timeout:
            self.reset()
            return None

        last = ltf.iloc[-1]
        self._extend_extreme(last)
        self._track_eq(last)
        direction = self.setup.direction

        # Counter-trend (signal against HTF bias): target is the 50% equilibrium.
        # If price already reached the 50% at any point before entry, there is no
        # room left -> cancel the whole setup. (Your: "اگه قبلش برخوردی داشت کنسل".)
        if (
            not self.setup.htf_trend_aligned
            and self.p.skip_if_eq_already_touched
            and self.setup.eq_touched
        ):
            self.reset()
            return None

        if self.p.entry_mode == "bos":
            trigger = self._bos_entry_check(ltf, direction)
        else:
            trigger = self._entry_trigger(ltf, direction)
        if trigger is None:
            return None

        entry = float(last["close"])
        atr_v = last_atr(ltf)
        pad = self.p.sl_padding_atr * atr_v
        extreme = self.setup.pullback_extreme
        if extreme is None:
            extreme = float(last["low"] if direction is Direction.LONG else last["high"])

        if direction is Direction.LONG:
            stop = extreme - pad
        else:
            stop = extreme + pad
        target = self.setup.target
        bias = (
            "bullish" if self.setup.htf_bias is Direction.LONG
            else "bearish" if self.setup.htf_bias is Direction.SHORT
            else "unclear"
        )

        signal = Signal(
            symbol=self.p.symbol,
            direction=direction,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            time=now,
            tf_set=self.p.tf_set.name,
            reason=f"CRT {direction.value} via {trigger}; "
            + ("trend-aligned -> CRT extreme" if self.setup.htf_trend_aligned else "counter-trend -> 50% equilibrium"),
            crt=self.setup.crt,
            entry_trigger=trigger,
            tp_mode=self.setup.tp_mode,
            market_bias=bias,
            bias_basis=self.setup.bias_basis,
            session=self.p.sessions.active_name(now) or "",
        )

        # sanity: correct side and acceptable reward:risk
        if signal.risk <= 0 or signal.reward <= 0:
            self.reset()
            return None
        if direction is Direction.LONG and not (stop < entry < target):
            self.reset()
            return None
        if direction is Direction.SHORT and not (target < entry < stop):
            self.reset()
            return None
        if signal.rr < self.p.min_rr:
            self.reset()
            return None

        self.state = SignalState.IN_TRADE
        return signal

    def _entry_trigger(self, ltf: pd.DataFrame, direction: Direction) -> str | None:
        """Legacy "retest" mode: enter directly on the iFVG/CISD tag."""
        tag = self._tags_key_level(ltf, direction)
        if tag is not None:
            return tag
        return None

    def _tags_key_level(self, ltf: pd.DataFrame, direction: Direction) -> str | None:
        """Is the LAST candle tagging a key level (iFVG/FVG retest or CISD)?

        The candle that *creates* a gap always touches its own edge, so the
        retest must come on a strictly later bar than the gap's creation (or
        inversion) time.
        """
        last = ltf.iloc[-1]
        last_time = ltf.index[-1]
        if self.p.use_ifvg:
            gap = latest_entry_gap(
                ltf, direction, after=self.setup.choch_mtf_time, use_ifvg=True
            )
            if gap is not None and last_time > (gap.inverted_time or gap.time):
                if last["low"] <= gap.top and last["high"] >= gap.bottom:
                    return "ifvg" if gap.inverted else "fvg"
        if self.p.use_cisd:
            if detect_cisd(ltf, direction) is not None:
                return "cisd"
        return None

    def _bos_entry_check(self, ltf: pd.DataFrame, direction: Direction) -> str | None:
        """Steps 6-8 of the playbook: after the LTF CHoCH, wait for a pullback
        into a key level, then enter when a close breaks the CHoCH leg's
        extreme (BOS). Returns "bos" on the entry bar, else None.
        """
        setup = self.setup
        assert setup is not None and setup.choch_ltf_time is not None
        last = ltf.iloc[-1]
        close = float(last["close"])

        # BOS reference = the CHoCH leg's extreme over bars BEFORE this one
        # (the breaking close must clear a level set by prior bars).
        if setup.ltf_leg_extreme is None:
            leg = ltf[(ltf.index >= setup.choch_ltf_time) & (ltf.index < ltf.index[-1])]
            if leg.empty:
                return None
            setup.ltf_leg_extreme = (
                float(leg["high"].max()) if direction is Direction.LONG
                else float(leg["low"].min())
            )

        if setup.ltf_pullback_done:
            if direction is Direction.LONG and close > setup.ltf_leg_extreme:
                return "bos"
            if direction is Direction.SHORT and close < setup.ltf_leg_extreme:
                return "bos"
            return None

        # still waiting for the pullback: a key-level tag completes it
        if self._tags_key_level(ltf, direction) is not None:
            setup.ltf_pullback_done = True
            return None

        # no pullback yet -> the leg is still running; extend the BOS reference
        if direction is Direction.LONG:
            setup.ltf_leg_extreme = max(setup.ltf_leg_extreme, float(last["high"]))
        else:
            setup.ltf_leg_extreme = min(setup.ltf_leg_extreme, float(last["low"]))
        return None

    # -- shared ------------------------------------------------------------
    def _extend_extreme(self, last: pd.Series) -> None:
        assert self.setup is not None
        if self.setup.pullback_extreme is None:
            return
        if self.setup.direction is Direction.LONG:
            self.setup.pullback_extreme = min(self.setup.pullback_extreme, float(last["low"]))
        else:
            self.setup.pullback_extreme = max(self.setup.pullback_extreme, float(last["high"]))

    def _track_eq(self, last: pd.Series) -> None:
        """Flag if price reached the CRT 50% equilibrium on this candle."""
        if self.setup is None or self.setup.eq_touched:
            return
        eq = self.setup.crt.equilibrium
        if self.setup.direction is Direction.LONG and float(last["high"]) >= eq:
            self.setup.eq_touched = True
        elif self.setup.direction is Direction.SHORT and float(last["low"]) <= eq:
            self.setup.eq_touched = True

    def _invalidated(self, ltf: pd.DataFrame) -> bool:
        """The CRT idea is dead if price closes beyond the swept extreme."""
        if self.setup is None or ltf.empty:
            return False
        last_close = float(ltf.iloc[-1]["close"])
        crt = self.setup.crt
        if self.setup.direction is Direction.LONG:
            return last_close < crt.low
        return last_close > crt.high
