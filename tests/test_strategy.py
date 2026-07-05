"""Strategy state-machine tests (stage transitions + entry construction)."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from crt_bot.core.models import CRTRange, Direction, SignalState
from crt_bot.core.session import SessionSet
from crt_bot.core.timeframes import TFSet
from crt_bot.strategy.crt_strategy import CRTStrategy, StrategyParams, _Setup

from .util import make_df

_NO_SESSION = SessionSet(enabled=False)
_TF = TFSet("1H", "5min", "1min")


def _params(**kw) -> StrategyParams:
    base = dict(
        symbol="X",
        tf_set=_TF,
        sessions=_NO_SESSION,
        use_fvg=True,
        use_order_block=False,
        use_liquidity_sweep=False,
        min_wick_sweep_atr=0.0,
        swing_lookback=2,
        # these tests exercise the TP/entry construction directly on the tag
        # bar; BOS-mode sequencing has its own tests in test_bos_entry.py
        entry_mode="retest",
    )
    base.update(kw)
    return StrategyParams(**base)


def test_stage1_htf_crt_with_poi_transitions():
    # HTF: bullish FVG [102,106] @ c3, then a CRT manipulation candle c5
    rows = [
        (100, 101, 99, 100),
        (100, 102, 99.5, 101),
        (101, 108, 101, 107),
        (107, 110, 106, 109),    # bullish FVG forms here (low 106 > high[1] 102)
        (109, 112, 104, 105),    # range candle (high 112 / low 104)
        (105, 107, 101, 105.5),  # manip: sweeps 104, closes back inside, taps FVG
    ]
    htf = make_df(rows)
    empty = htf.iloc[0:0]

    strat = CRTStrategy(_params())
    now = htf.index[-1] + pd.Timedelta(hours=1)
    strat.update(now, htf, empty, empty)

    assert strat.state is SignalState.WAIT_MTF_CHOCH
    assert strat.setup is not None
    assert strat.setup.crt.direction is Direction.LONG
    assert strat.setup.crt.high == 112 and strat.setup.crt.low == 104


def test_entry_long_emits_signal_with_crt_target():
    strat = CRTStrategy(_params(min_rr=1.5))

    # LTF leg with a fresh bullish FVG [105.0, 105.2] @ l2, retested by l3
    rows = [
        (104.5, 105.0, 104.0, 104.8),
        (104.8, 106.0, 104.8, 105.8),
        (105.8, 106.5, 105.2, 106.0),   # bullish FVG (low 105.2 > high[0] 105.0)
        (105.6, 105.7, 105.0, 105.3),   # retests the gap -> entry trigger
    ]
    ltf = make_df(rows, start="2026-05-01 12:00", freq="1min")
    empty = ltf.iloc[0:0]

    crt = CRTRange(
        direction=Direction.LONG,
        high=112.0,
        low=104.0,
        swept_level=104.0,
        range_candle_time=pd.Timestamp("2026-05-01 10:00", tz="UTC"),
        manip_candle_time=pd.Timestamp("2026-05-01 11:00", tz="UTC"),
    )
    strat.setup = _Setup(
        crt=crt,
        direction=Direction.LONG,
        htf_trend_aligned=True,           # -> target = CRT high (112)
        crt_time=crt.manip_candle_time,
        fib_zone=(102.0, 104.0),
        pullback_extreme=103.0,
        choch_mtf_time=ltf.index[0] - pd.Timedelta(minutes=1),
        choch_ltf_time=ltf.index[0],
    )
    strat.state = SignalState.WAIT_LTF_ENTRY

    now = ltf.index[-1] + pd.Timedelta(minutes=1)
    signal = strat.update(now, empty, empty, ltf)

    assert signal is not None
    assert signal.direction is Direction.LONG
    assert signal.take_profit == 112.0           # trend-aligned -> CRT extreme
    assert signal.tp_mode == "crt_high"
    assert signal.stop_loss < 103.0              # below the pullback extreme
    assert signal.rr >= 1.5
    assert strat.state is SignalState.IN_TRADE


def _counter_trend_setup(strat, ltf, eq_touched):
    """Build a counter-trend LONG setup (bearish bias) targeting the 50%."""
    crt = CRTRange(
        direction=Direction.LONG,
        high=120.0, low=100.0, swept_level=100.0,   # equilibrium = 110
        range_candle_time=pd.Timestamp("2026-05-01 10:00", tz="UTC"),
        manip_candle_time=pd.Timestamp("2026-05-01 11:00", tz="UTC"),
    )
    strat.setup = _Setup(
        crt=crt,
        direction=Direction.LONG,
        htf_trend_aligned=False,        # bearish market, long signal
        htf_bias=Direction.SHORT,
        crt_time=crt.manip_candle_time,
        fib_zone=(102.0, 104.0),
        pullback_extreme=102.5,
        choch_mtf_time=ltf.index[0] - pd.Timedelta(minutes=1),
        choch_ltf_time=ltf.index[0],
        eq_touched=eq_touched,
    )
    strat.state = SignalState.WAIT_LTF_ENTRY


def _counter_trend_ltf():
    rows = [
        (102.5, 103.0, 102.0, 102.8),
        (102.8, 104.5, 102.8, 104.2),
        (104.2, 105.0, 103.5, 104.5),   # bullish FVG [103.0, 103.5]
        (104.0, 104.2, 103.2, 104.0),   # retest -> entry; high stays below 50%
    ]
    return make_df(rows, start="2026-05-01 12:00", freq="1min")


def test_counter_trend_targets_equilibrium_when_eq_not_touched():
    strat = CRTStrategy(_params(min_rr=1.5))
    ltf = _counter_trend_ltf()
    _counter_trend_setup(strat, ltf, eq_touched=False)
    now = ltf.index[-1] + pd.Timedelta(minutes=1)
    signal = strat.update(now, ltf.iloc[0:0], ltf.iloc[0:0], ltf)

    assert signal is not None
    assert signal.take_profit == 110.0          # 50% of the 100-120 CRT range
    assert signal.tp_mode == "equilibrium"
    assert signal.market_bias == "bearish"


def test_counter_trend_cancels_if_eq_already_touched():
    strat = CRTStrategy(_params(min_rr=1.5))
    ltf = _counter_trend_ltf()
    _counter_trend_setup(strat, ltf, eq_touched=True)   # price already tagged 50%
    now = ltf.index[-1] + pd.Timedelta(minutes=1)
    signal = strat.update(now, ltf.iloc[0:0], ltf.iloc[0:0], ltf)

    assert signal is None                        # whole setup cancelled
    assert strat.state is SignalState.WAIT_CRT
    assert strat.setup is None


def test_entry_short_emits_signal_with_crt_low_target():
    """Mirror of the LONG case: a trend-aligned SHORT targets the CRT Low."""
    strat = CRTStrategy(_params(min_rr=1.5))
    rows = [
        (118.0, 118.5, 117.0, 117.5),
        (117.5, 117.5, 115.0, 115.5),
        (115.5, 115.8, 114.0, 114.5),   # bearish FVG [115.8, 117.0]
        (115.9, 116.5, 115.5, 116.0),   # retest -> entry trigger
    ]
    ltf = make_df(rows, start="2026-05-01 12:00", freq="1min")
    empty = ltf.iloc[0:0]

    crt = CRTRange(
        direction=Direction.SHORT, high=120.0, low=100.0, swept_level=120.0,
        range_candle_time=pd.Timestamp("2026-05-01 10:00", tz="UTC"),
        manip_candle_time=pd.Timestamp("2026-05-01 11:00", tz="UTC"),
    )
    strat.setup = _Setup(
        crt=crt, direction=Direction.SHORT, htf_trend_aligned=True,   # bearish market
        htf_bias=Direction.SHORT, crt_time=crt.manip_candle_time,
        fib_zone=(116.0, 118.0), pullback_extreme=117.0,
        choch_mtf_time=ltf.index[0] - pd.Timedelta(minutes=1), choch_ltf_time=ltf.index[0],
    )
    strat.state = SignalState.WAIT_LTF_ENTRY

    now = ltf.index[-1] + pd.Timedelta(minutes=1)
    sig = strat.update(now, empty, empty, ltf)

    assert sig is not None
    assert sig.direction is Direction.SHORT
    assert sig.take_profit == 100.0              # trend-aligned short -> CRT Low
    assert sig.tp_mode == "crt_low"
    assert sig.market_bias == "bearish"
    assert sig.stop_loss > 117.0                 # above the pullback extreme
    assert sig.take_profit < sig.entry < sig.stop_loss
    assert sig.rr >= 1.5
    assert strat.state is SignalState.IN_TRADE
