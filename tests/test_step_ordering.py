"""Playbook step ordering: the LTF CHoCH (step 6) must come AFTER the MTF fib
pullback (step 5), and the MT5 UTC-offset detection must reject stale values.
"""

from __future__ import annotations

import pandas as pd

from crt_bot.core.models import CRTRange, Direction, SignalState
from crt_bot.core.session import SessionSet
from crt_bot.core.timeframes import TFSet
from crt_bot.feeds.mt5 import MT5Feed
from crt_bot.strategy.crt_strategy import CRTStrategy, StrategyParams, _Setup

from .util import make_df

_TF = TFSet("1H", "5min", "1min")

# LTF frame whose only LONG CHoCH prints on the LAST bar (12:07)
_CHOCH_ROWS = [
    (110, 111, 109, 110),
    (110, 110.5, 104, 105),
    (105, 106, 103, 104),     # swing low 103
    (104, 107, 104, 106),     # swing high 107
    (106, 106, 102, 102.5),   # bos short (breaks 103)
    (102.5, 103, 101, 102),   # swing low 101
    (102, 104, 101.5, 103),
    (103, 109, 103, 108.5),   # close breaks 107 -> CHoCH long @ 12:07
]


def _armed_wait_ltf_choch(pullback_time: pd.Timestamp) -> tuple[CRTStrategy, pd.DataFrame]:
    ltf = make_df(_CHOCH_ROWS, start="2026-05-01 12:00", freq="1min")
    strat = CRTStrategy(StrategyParams(
        symbol="X", tf_set=_TF, sessions=SessionSet(enabled=False), swing_lookback=1,
    ))
    crt = CRTRange(
        direction=Direction.LONG, high=115.0, low=100.0, swept_level=100.0,
        range_candle_time=pd.Timestamp("2026-05-01 10:00", tz="UTC"),
        manip_candle_time=pd.Timestamp("2026-05-01 11:00", tz="UTC"),
    )
    strat.setup = _Setup(
        crt=crt, direction=Direction.LONG, htf_trend_aligned=True,
        htf_bias=Direction.LONG, crt_time=crt.manip_candle_time,
        fib_zone=(101.0, 103.0), pullback_extreme=101.0,
        choch_mtf_time=pd.Timestamp("2026-05-01 11:30", tz="UTC"),
        pullback_time=pullback_time,
    )
    strat.state = SignalState.WAIT_LTF_CHOCH
    return strat, ltf


def test_choch_before_pullback_is_rejected():
    # the only CHoCH is at 12:07; pullback tagged at 12:07 too -> not AFTER it
    strat, ltf = _armed_wait_ltf_choch(pd.Timestamp("2026-05-01 12:07", tz="UTC"))
    strat._scan_ltf_choch(ltf)
    assert strat.state is SignalState.WAIT_LTF_CHOCH   # must keep waiting


def test_choch_after_pullback_is_accepted():
    strat, ltf = _armed_wait_ltf_choch(pd.Timestamp("2026-05-01 12:06", tz="UTC"))
    strat._scan_ltf_choch(ltf)
    assert strat.state is SignalState.WAIT_LTF_ENTRY
    assert strat.setup.choch_ltf_time == pd.Timestamp("2026-05-01 12:07", tz="UTC")


def test_mt5_offset_clamp():
    assert MT5Feed._clamp_offset(3.0) == 3.0
    assert MT5Feed._clamp_offset(0.0) == 0.0
    assert MT5Feed._clamp_offset(-5.0) == -5.0
    assert MT5Feed._clamp_offset(14.0) == 14.0
    # stale weekend ticks produce absurd offsets -> fall back to UTC
    assert MT5Feed._clamp_offset(-34.0) == 0.0
    assert MT5Feed._clamp_offset(48.0) == 0.0
