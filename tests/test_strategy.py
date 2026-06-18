"""Strategy state-machine tests (stage transitions + entry construction)."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from crt_bot.core.models import CRTRange, Direction, SignalState
from crt_bot.core.session import Session
from crt_bot.core.timeframes import TFSet
from crt_bot.strategy.crt_strategy import CRTStrategy, StrategyParams, _Setup

from .util import make_df

_NO_SESSION = Session(False, "x", "UTC", dt.time(0), dt.time(23, 59))
_TF = TFSet("1H", "5min", "1min")


def _params(**kw) -> StrategyParams:
    base = dict(
        symbol="X",
        tf_set=_TF,
        session=_NO_SESSION,
        use_fvg=True,
        use_order_block=False,
        use_liquidity_sweep=False,
        min_wick_sweep_atr=0.0,
        swing_lookback=2,
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
    assert signal.stop_loss < 103.0              # below the pullback extreme
    assert signal.rr >= 1.5
    assert strat.state is SignalState.IN_TRADE
