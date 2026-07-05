"""BOS entry mode (playbook steps 6-8):

LTF CHoCH -> pullback into a key level -> enter on the close that breaks the
CHoCH leg's extreme (BOS).
"""

from __future__ import annotations

import pandas as pd

from crt_bot.core.models import CRTRange, Direction, SignalState
from crt_bot.core.session import SessionSet
from crt_bot.core.timeframes import TFSet
from crt_bot.strategy.crt_strategy import CRTStrategy, StrategyParams, _Setup

from .util import make_df

_TF = TFSet("1H", "5min", "1min")


def _bos_strat(**kw) -> CRTStrategy:
    base = dict(
        symbol="X", tf_set=_TF, sessions=SessionSet(enabled=False),
        entry_mode="bos", use_ifvg=True, use_cisd=False,
        min_rr=1.0, sl_padding_atr=0.0, swing_lookback=2,
    )
    base.update(kw)
    return CRTStrategy(StrategyParams(**base))


def _arm_long(strat: CRTStrategy, ltf: pd.DataFrame) -> None:
    """Put the strategy in WAIT_LTF_ENTRY with a trend-aligned LONG setup whose
    LTF CHoCH is the first bar of ``ltf``."""
    crt = CRTRange(
        direction=Direction.LONG, high=112.0, low=100.0, swept_level=100.0,
        range_candle_time=pd.Timestamp("2026-05-01 10:00", tz="UTC"),
        manip_candle_time=pd.Timestamp("2026-05-01 11:00", tz="UTC"),
    )
    strat.setup = _Setup(
        crt=crt, direction=Direction.LONG, htf_trend_aligned=True,
        htf_bias=Direction.LONG, crt_time=crt.manip_candle_time,
        fib_zone=(102.0, 104.0), pullback_extreme=103.0, mtf_fib_extreme=103.0,
        choch_mtf_time=ltf.index[0] - pd.Timedelta(minutes=1),
        choch_ltf_time=ltf.index[0],
    )
    strat.state = SignalState.WAIT_LTF_ENTRY


def _drive(strat: CRTStrategy, ltf: pd.DataFrame):
    """Feed the LTF frame bar by bar; return the first signal (or None)."""
    empty = ltf.iloc[0:0]
    for k in range(1, len(ltf) + 1):
        view = ltf.iloc[:k]
        now = view.index[-1] + pd.Timedelta(minutes=1)
        sig = strat.update(now, empty, empty, view)
        if sig is not None:
            return sig
    return None


# bar layout (1min):
#  b0 choch bar, b1 leg, b2 leg forms bullish FVG [105.0, 105.4] & leg high
#  b3 pullback tags the gap, b4 BOS close above the leg high -> entry
_LONG_ROWS = [
    (104.0, 105.0, 103.8, 104.9),   # b0: LTF CHoCH bar
    (104.9, 106.0, 104.9, 105.8),   # b1: impulse leg
    (105.8, 106.5, 105.4, 106.2),   # b2: FVG (low 105.4 > b0 high 105.0); leg high 106.5
    (106.0, 106.1, 105.2, 105.5),   # b3: pullback into the gap
    (105.6, 106.8, 105.5, 106.6),   # b4: BOS -- close 106.6 > leg high 106.5
]


def test_bos_long_full_sequence():
    strat = _bos_strat()
    ltf = make_df(_LONG_ROWS, start="2026-05-01 12:00", freq="1min")
    _arm_long(strat, ltf)

    sig = _drive(strat, ltf)

    assert sig is not None
    assert sig.entry_trigger == "bos"
    assert sig.direction is Direction.LONG
    assert sig.entry == 106.6                    # the BOS close
    assert sig.take_profit == 112.0              # trend-aligned -> CRT High
    # SL anchor "ltf_pullback": lowest low AFTER the LTF CHoCH bar (b1 104.9)
    assert sig.stop_loss == 104.9
    assert strat.state is SignalState.IN_TRADE


def test_bos_sl_anchor_mtf_fib():
    strat = _bos_strat(sl_anchor="mtf_fib")
    ltf = make_df(_LONG_ROWS, start="2026-05-01 12:00", freq="1min")
    _arm_long(strat, ltf)
    sig = _drive(strat, ltf)
    assert sig is not None
    # anchored at the MTF candle that tagged the fib zone
    assert sig.stop_loss == 103.0


def test_bos_sl_anchor_deepest():
    strat = _bos_strat(sl_anchor="deepest")
    ltf = make_df(_LONG_ROWS, start="2026-05-01 12:00", freq="1min")
    _arm_long(strat, ltf)
    sig = _drive(strat, ltf)
    assert sig is not None
    # deeper of (ltf pullback 104.9, mtf fib 103.0) -> 103.0
    assert sig.stop_loss == 103.0


def test_bos_requires_pullback_first():
    # same leg but NO pullback: price only grinds up, never tags the gap
    rows = [
        (104.0, 105.0, 103.8, 104.9),
        (104.9, 106.0, 104.9, 105.8),
        (105.8, 106.5, 105.4, 106.2),
        (106.2, 106.9, 106.0, 106.8),   # breaks the leg high WITHOUT a pullback
        (106.8, 107.5, 106.6, 107.4),
    ]
    strat = _bos_strat()
    ltf = make_df(rows, start="2026-05-01 12:00", freq="1min")
    _arm_long(strat, ltf)

    sig = _drive(strat, ltf)

    assert sig is None                           # step 7 without step 6 = no entry
    assert strat.state is SignalState.WAIT_LTF_ENTRY


def test_bos_no_entry_before_break():
    # pullback happens, but no close ever clears the leg high
    rows = [
        (104.0, 105.0, 103.8, 104.9),
        (104.9, 106.0, 104.9, 105.8),
        (105.8, 106.5, 105.4, 106.2),
        (106.0, 106.1, 105.2, 105.5),   # pullback tags the gap
        (105.5, 106.4, 105.4, 106.3),   # high stays below 106.5 -> no BOS
    ]
    strat = _bos_strat()
    ltf = make_df(rows, start="2026-05-01 12:00", freq="1min")
    _arm_long(strat, ltf)

    assert _drive(strat, ltf) is None
    assert strat.setup is not None and strat.setup.ltf_pullback_done is True


def test_bos_short_full_sequence():
    # mirror image of the long case
    rows = [
        (116.0, 116.2, 115.0, 115.1),   # b0: CHoCH bar
        (115.1, 115.1, 114.0, 114.2),   # b1: impulse leg down
        (114.2, 114.6, 113.5, 113.8),   # b2: bearish FVG (high 114.6 < b0 low 115.0); leg low 113.5
        (113.9, 114.8, 113.9, 114.5),   # b3: pullback into the gap [114.6, 115.0]
        (114.4, 114.5, 113.2, 113.4),   # b4: BOS -- close 113.4 < leg low 113.5
    ]
    strat = _bos_strat()
    ltf = make_df(rows, start="2026-05-01 12:00", freq="1min")

    crt = CRTRange(
        direction=Direction.SHORT, high=120.0, low=104.0, swept_level=120.0,
        range_candle_time=pd.Timestamp("2026-05-01 10:00", tz="UTC"),
        manip_candle_time=pd.Timestamp("2026-05-01 11:00", tz="UTC"),
    )
    strat.setup = _Setup(
        crt=crt, direction=Direction.SHORT, htf_trend_aligned=True,
        htf_bias=Direction.SHORT, crt_time=crt.manip_candle_time,
        fib_zone=(115.0, 117.0), pullback_extreme=116.5, mtf_fib_extreme=116.5,
        choch_mtf_time=ltf.index[0] - pd.Timedelta(minutes=1),
        choch_ltf_time=ltf.index[0],
    )
    strat.state = SignalState.WAIT_LTF_ENTRY

    sig = _drive(strat, ltf)

    assert sig is not None
    assert sig.entry_trigger == "bos"
    assert sig.direction is Direction.SHORT
    assert sig.entry == 113.4
    assert sig.take_profit == 104.0              # trend-aligned short -> CRT Low
    # SL anchor "ltf_pullback": highest high AFTER the CHoCH bar (b1 115.1)
    assert sig.stop_loss == 115.1
    assert strat.state is SignalState.IN_TRADE
