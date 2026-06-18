"""The multi-timeframe CRT strategy state machine.

Pipeline (HTF -> MTF -> LTF), exactly mirroring the trader's playbook:

1.  **HTF** (after the session opens): wait for price to tag a POI (FVG / OB /
    prior-session liquidity sweep) **and** print a CRT manipulation candle in
    the same direction. This fixes the trade direction and the CRT range.
2.  **MTF**: wait for a CHoCH in the CRT direction, then a pullback into the
    0.618-0.79 fib zone of the impulse leg.
3.  **LTF**: wait for a CHoCH, then enter on an iFVG / CISD trigger.

Risk: SL beyond the pullback extreme (+ ATR padding). Target = CRT extreme when
the HTF trend agrees with the trade, otherwise the CRT equilibrium (0.5); in
that counter-trend case the setup is skipped if price already reached
equilibrium before the entry triggered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..core.indicators import last_atr
from ..core.models import CRTRange, Direction, Signal, SignalState
from ..core.session import Session
from ..core.timeframes import TFSet
from ..smc import crt as crt_mod
from ..smc import fib as fib_mod
from ..smc.cisd import detect_cisd
from ..smc.fvg import find_fvgs, latest_entry_gap
from ..smc.liquidity import detect_sweep
from ..smc.order_block import find_order_blocks
from ..smc.structure import last_choch, structure_events


@dataclass
class StrategyParams:
    symbol: str
    tf_set: TFSet
    session: Session

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
    pullback_max: float = 0.79
    equilibrium_level: float = 0.5
    skip_if_eq_already_touched: bool = True

    # entry
    use_ifvg: bool = True
    use_cisd: bool = True
    sl_padding_atr: float = 0.1
    min_rr: float = 1.5

    # stage timeouts (in bars of the relevant TF)
    mtf_choch_timeout: int = 24
    pullback_timeout: int = 24
    ltf_choch_timeout: int = 40
    ltf_entry_timeout: int = 40

    @classmethod
    def from_config(cls, cfg: dict, tf_set: TFSet, session: Session) -> "StrategyParams":
        s = cfg.get("strategy", {})
        poi = s.get("poi", {})
        crt = s.get("crt", {})
        struct = s.get("structure", {})
        fib = s.get("fib", {})
        entry = s.get("entry", {})
        return cls(
            symbol=cfg.get("symbol", "UNKNOWN"),
            tf_set=tf_set,
            session=session,
            use_fvg=poi.get("use_fvg", True),
            use_order_block=poi.get("use_order_block", True),
            use_liquidity_sweep=poi.get("use_liquidity_sweep", True),
            lookback_sessions=poi.get("lookback_sessions", 3),
            require_close_inside=crt.get("require_close_inside", True),
            min_wick_sweep_atr=crt.get("min_wick_sweep_atr", 0.05),
            swing_lookback=struct.get("swing_lookback", 2),
            pullback_min=fib.get("pullback_min", 0.618),
            pullback_max=fib.get("pullback_max", 0.79),
            equilibrium_level=fib.get("equilibrium_level", 0.5),
            skip_if_eq_already_touched=fib.get("skip_if_eq_already_touched", True),
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
    fib_zone: tuple[float, float] | None = None
    pullback_extreme: float | None = None   # deepest low (long) / high (short)
    choch_mtf_time: pd.Timestamp | None = None
    choch_ltf_time: pd.Timestamp | None = None
    bars_in_state: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def target(self) -> float:
        if self.htf_trend_aligned:
            return self.crt.high if self.direction is Direction.LONG else self.crt.low
        return self.crt.equilibrium


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

        if self.state is SignalState.WAIT_CRT:
            if new_htf:
                self._scan_htf(now, htf)
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
    def _scan_htf(self, now: pd.Timestamp, htf: pd.DataFrame) -> None:
        if self.p.session.enabled and not self.p.session.contains(now):
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

        aligned = self._htf_trend(htf) is crt.direction
        self.setup = _Setup(
            crt=crt,
            direction=crt.direction,
            htf_trend_aligned=aligned,
            crt_time=crt.manip_candle_time,
            tags=["crt", "trend_aligned" if aligned else "counter_trend"],
        )
        self.state = SignalState.WAIT_MTF_CHOCH

    def _poi_confluence(
        self, htf: pd.DataFrame, direction: Direction, now: pd.Timestamp
    ) -> bool:
        """A POI of ``direction`` tagged within the recent HTF window."""
        recent = htf.tail(self.p.poi_validity_bars + 3)
        last = htf.iloc[-1]

        if self.p.use_liquidity_sweep:
            sweep = detect_sweep(htf, self.p.session, now, self.p.lookback_sessions)
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

    def _htf_trend(self, htf: pd.DataFrame) -> Direction | None:
        events = structure_events(htf, self.p.swing_lookback)
        return events[-1].direction if events else None

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
        direction = self.setup.direction

        # counter-trend: abort if equilibrium target already reached
        if not self.setup.htf_trend_aligned and self.p.skip_if_eq_already_touched:
            eq = self.setup.crt.equilibrium
            if direction is Direction.LONG and last["high"] >= eq:
                self.reset()
                return None
            if direction is Direction.SHORT and last["low"] <= eq:
                self.reset()
                return None

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

        signal = Signal(
            symbol=self.p.symbol,
            direction=direction,
            entry=entry,
            stop_loss=stop,
            take_profit=target,
            time=now,
            tf_set=self.p.tf_set.name,
            reason=f"CRT {direction.value} via {trigger}; "
            + ("trend-aligned->extreme" if self.setup.htf_trend_aligned else "counter->equilibrium"),
            crt=self.setup.crt,
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
        if self.p.use_ifvg:
            gap = latest_entry_gap(
                ltf, direction, after=self.setup.choch_mtf_time, use_ifvg=True
            )
            if gap is not None:
                last = ltf.iloc[-1]
                # price retesting the gap zone
                if last["low"] <= gap.top and last["high"] >= gap.bottom:
                    return "ifvg" if gap.inverted else "fvg"
        if self.p.use_cisd:
            c = detect_cisd(ltf, direction)
            if c is not None:
                return "cisd"
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

    def _invalidated(self, ltf: pd.DataFrame) -> bool:
        """The CRT idea is dead if price closes beyond the swept extreme."""
        if self.setup is None or ltf.empty:
            return False
        last_close = float(ltf.iloc[-1]["close"])
        crt = self.setup.crt
        if self.setup.direction is Direction.LONG:
            return last_close < crt.low
        return last_close > crt.high
