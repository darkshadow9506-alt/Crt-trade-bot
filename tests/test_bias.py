"""HTF market-bias detection (real swing structure: HH/HL vs LH/LL)."""

from __future__ import annotations

from crt_bot.core.models import Direction
from crt_bot.core.session import SessionSet
from crt_bot.core.timeframes import TFSet
from crt_bot.strategy.crt_strategy import CRTStrategy, StrategyParams

from .util import make_df

_TF = TFSet("1H", "5min", "1min")
_SESS = SessionSet(enabled=False)


def _strat(lookback=1):
    return CRTStrategy(StrategyParams(symbol="X", tf_set=_TF, sessions=_SESS, swing_lookback=lookback))


def test_bias_bullish_hh_hl():
    # swing highs 108,112,116 (HH) ; swing lows 98,102 (HL)
    rows = [
        (101, 102, 100, 101),
        (102, 108, 101, 107),   # swing high 108
        (100, 104, 98, 100),    # swing low 98
        (104, 112, 103, 110),   # swing high 112
        (104, 106, 102, 103),   # swing low 102
        (108, 116, 107, 114),   # swing high 116
        (109, 110, 108, 109),
    ]
    direction, basis = _strat()._htf_trend(make_df(rows))
    assert direction is Direction.LONG and basis == "HH+HL"


def test_bias_bearish_lh_ll():
    # mirror: lower highs and lower lows
    rows = [
        (115, 116, 114, 115),
        (114, 115, 108, 109),   # swing low 108
        (110, 118, 110, 112),   # swing high 118
        (106, 113, 104, 105),   # swing low 104
        (110, 114, 110, 110),   # swing high 114
        (102, 109, 100, 101),   # swing low 100
        (101, 107, 100.5, 106),
    ]
    direction, basis = _strat()._htf_trend(make_df(rows))
    assert direction is Direction.SHORT and basis == "LH+LL"


def test_bias_ma_fallback_when_no_structure():
    # steadily rising, not enough confirmed 2+2 swings -> MA tiebreak -> LONG
    rows = [(100 + i, 100.5 + i, 99.5 + i, 100.4 + i) for i in range(6)]
    direction, basis = _strat()._htf_trend(make_df(rows))
    assert direction is Direction.LONG and basis == "MA"
