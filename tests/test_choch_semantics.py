"""CHoCH semantics per the trader's definition.

For a SHORT setup (bearish 1H CRT): on the MTF, find the LAST bullish BOS
(the leg that swept the high); the bearish CHoCH is the break of THAT BOS's
higher-low -- not some unrelated structure break. Mirror for longs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from crt_bot.core.models import CRTRange, Direction, SignalState
from crt_bot.core.session import SessionSet
from crt_bot.core.timeframes import TFSet
from crt_bot.smc.structure import last_choch, structure_events
from crt_bot.strategy.crt_strategy import CRTStrategy, StrategyParams, _Setup

from .util import make_df

_TF = TFSet("1H", "5min", "1min")

# 5min bars: uptrend sweeps the 1H high. The LAST BULLISH BOS breaks 106 and
# its origin higher-low is 100. The bearish CHoCH must break exactly 100.
_SHORT_SCENARIO = [
    (100, 105, 99, 104),      # b0
    (104, 106, 103, 105),     # b1: swing high 106
    (105, 104.5, 101, 102),   # b2: dip
    (102, 103, 100, 101),     # b3: HIGHER LOW = 100 (origin of the bullish BOS)
    (101, 107, 100.8, 106.5), # b4: closes above 106 -> LAST BULLISH BOS
    (106.5, 108, 105, 106),   # b5: sweep top 108
    (106, 106, 103.5, 104),   # b6: decline
    (104, 104.5, 102, 102.5), # b7
    (102.5, 103, 99, 99.5),   # b8: closes below 100 -> CHoCH of THAT BOS
]


def test_choch_breaks_the_last_bullish_bos_higher_low():
    df = make_df(_SHORT_SCENARIO, start="2026-07-01 14:00", freq="5min")
    events = structure_events(df, lookback=1)
    kinds = [(e.kind, e.direction, e.level) for e in events]
    assert ("bos", Direction.LONG, 106.0) in kinds       # the last bullish BOS
    ch = last_choch(df, Direction.SHORT, lookback=1)
    assert ch is not None
    assert ch.level == 100.0                              # ITS higher-low, not another level


def test_no_choch_without_a_prior_opposite_bos():
    # pure downtrend: a bearish break with no bullish BOS before it is just a
    # BOS -- the strategy must never treat it as the CHoCH.
    rows = [
        (110, 111, 109, 110),
        (110, 110.5, 107, 108),
        (108, 109, 106, 107),      # swing low 106
        (107, 108, 106.5, 107.5),
        (107.5, 107.5, 104, 105),  # breaks 106 -- but no bullish structure before
    ]
    df = make_df(rows, start="2026-07-01 14:00", freq="5min")
    events = structure_events(df, lookback=1)
    assert all(e.kind == "bos" for e in events)
    assert last_choch(df, Direction.SHORT, lookback=1) is None


def test_bullish_choch_breaks_the_last_bearish_bos_lower_high():
    # mirror: downtrend sweeps the low; last bearish BOS origin lower-high=100
    rows = [
        (100, 101, 95, 96),
        (96, 97, 94, 95),         # swing low 94
        (95, 99, 95, 98),
        (98, 100, 97, 99),        # LOWER HIGH = 100 (origin of the bearish BOS)
        (99, 99.2, 93, 93.5),     # closes below 94 -> LAST BEARISH BOS
        (93.5, 95, 92, 94),       # sweep bottom 92
        (94, 96.5, 94, 96),
        (96, 97.5, 95.5, 97.5),
        (97.5, 101, 97, 100.5),   # closes above 100 -> CHoCH of THAT BOS
    ]
    df = make_df(rows, start="2026-07-01 14:00", freq="5min")
    ch = last_choch(df, Direction.LONG, lookback=1)
    assert ch is not None and ch.level == 100.0


def test_mtf_scan_catches_the_choch_of_that_bos_sequentially():
    """Drive the MTF stage bar by bar: the state machine must transition on
    the exact bar where the CHoCH of the last bullish BOS prints, and the fib
    leg must span the sweep high down to the CHoCH low (like the charts)."""
    df = make_df(_SHORT_SCENARIO, start="2026-07-01 14:00", freq="5min")
    strat = CRTStrategy(StrategyParams(
        symbol="X", tf_set=_TF, sessions=SessionSet(enabled=False), swing_lookback=1,
    ))
    crt = CRTRange(
        direction=Direction.SHORT, high=107.5, low=90.0, swept_level=107.5,
        range_candle_time=pd.Timestamp("2026-07-01 13:00", tz="UTC"),
        manip_candle_time=pd.Timestamp("2026-07-01 14:00", tz="UTC"),
    )
    strat.setup = _Setup(
        crt=crt, direction=Direction.SHORT, htf_trend_aligned=True,
        htf_bias=Direction.SHORT, bias_basis="LH+LL",
        crt_time=pd.Timestamp("2026-07-01 14:15", tz="UTC"),   # b3, during the sweep
    )
    strat.state = SignalState.WAIT_MTF_CHOCH

    for k in range(1, len(df) + 1):
        if strat.state is not SignalState.WAIT_MTF_CHOCH:
            break
        strat._scan_mtf_choch(df.iloc[:k])
        if k < len(df):   # before b8 closes there must be NO transition
            assert strat.state is SignalState.WAIT_MTF_CHOCH, f"early transition at bar {k}"

    assert strat.state is SignalState.WAIT_MTF_PULLBACK
    assert strat.setup.choch_mtf_time == pd.Timestamp("2026-07-01 14:40", tz="UTC")  # b8
    # fib leg = sweep high (108) down to the CHoCH low (99)
    zone_lo, zone_hi = strat.setup.fib_zone
    assert zone_lo == pytest.approx(99 + 0.618 * 9)
    assert zone_hi == pytest.approx(99 + 0.786 * 9)
