"""HTF key-level confluence: breaker blocks and iFVGs count as POIs."""

from __future__ import annotations

import pandas as pd

from crt_bot.core.models import Direction
from crt_bot.core.session import SessionSet
from crt_bot.core.timeframes import TFSet
from crt_bot.strategy.crt_strategy import CRTStrategy, StrategyParams

from .util import make_df

_TF = TFSet("1H", "5min", "1min")


def _strat(**kw) -> CRTStrategy:
    base = dict(
        symbol="X", tf_set=_TF, sessions=SessionSet(enabled=False),
        use_fvg=False, poi_use_ifvg=False, use_order_block=False,
        use_breaker_block=False, use_liquidity_sweep=False,
    )
    base.update(kw)
    return CRTStrategy(StrategyParams(**base))


def _now(df: pd.DataFrame) -> pd.Timestamp:
    return df.index[-1] + pd.Timedelta(hours=1)


def test_breaker_block_counts_as_key_level():
    rows = [
        (100, 101, 99, 99.2),       # down candle -> bullish OB [99, 101]
        (99.2, 103, 99.1, 102.5),   # confirms the OB
        (102.5, 102.8, 97.5, 98),   # closes below 99 -> bearish breaker [99, 101]
        (98, 98.8, 97.6, 98.2),
        (98.2, 100.5, 98.0, 99.5),  # last candle wicks INTO the breaker zone
    ]
    htf = make_df(rows)
    strat = _strat(use_breaker_block=True)
    assert strat._poi_confluence(htf, Direction.SHORT, _now(htf)) is True
    # opposite direction does not match the breaker's polarity
    assert strat._poi_confluence(htf, Direction.LONG, _now(htf)) is False


def test_ifvg_counts_as_key_level():
    rows = [
        (10, 10.5, 9.5, 10),
        (10, 12, 10, 11.8),
        (11.8, 12.5, 11, 12),       # bullish FVG [10.5, 11]
        (10.6, 10.7, 9, 9.5),       # closes below 10.5 -> inverts to bearish iFVG
        (9.5, 10.8, 9.4, 10.2),     # last candle retests the iFVG zone
    ]
    htf = make_df(rows)
    strat = _strat(poi_use_ifvg=True)
    assert strat._poi_confluence(htf, Direction.SHORT, _now(htf)) is True


def test_no_key_level_touch_means_no_confluence():
    rows = [
        (100, 101, 99, 99.2),
        (99.2, 103, 99.1, 102.5),
        (102.5, 102.8, 97.5, 98),   # bearish breaker exists at [99, 101] ...
        (98, 98.5, 97.6, 98.2),
        (97.9, 98.4, 97.5, 98.0),   # ... but the last candle never reaches it
    ]
    htf = make_df(rows)
    strat = _strat(use_breaker_block=True)
    assert strat._poi_confluence(htf, Direction.SHORT, _now(htf)) is False
