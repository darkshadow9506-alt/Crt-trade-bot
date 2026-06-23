"""Backtester fill model."""

from __future__ import annotations

import pandas as pd

from crt_bot.backtest.engine import Backtester, Costs
from crt_bot.core.models import Direction, Signal, Trade, TradeStatus
from crt_bot.core.session import Session
from crt_bot.core.timeframes import TFSet
from crt_bot.risk.risk_manager import RiskManager, RiskParams
from crt_bot.strategy.crt_strategy import CRTStrategy, StrategyParams


def _bt() -> Backtester:
    tf_set = TFSet("1H", "5min", "1min")
    session = Session(False, "x", "UTC", __import__("datetime").time(0),
                      __import__("datetime").time(23, 59))
    strat = CRTStrategy(StrategyParams(symbol="X", tf_set=tf_set, session=session))
    risk = RiskManager(RiskParams())
    return Backtester(strat, risk, tf_set, "1min", Costs())


def _trade(entry=100.0, sl=95.0, tp=110.0, size=10.0):
    sig = Signal("X", Direction.LONG, entry, sl, tp,
                 pd.Timestamp("2026-05-01", tz="UTC"), "t")
    return Trade(signal=sig, size=size, entry_time=sig.time, entry_price=entry)


def _bar(high, low):
    return pd.Series({"open": 100.0, "high": high, "low": low, "close": 100.0, "volume": 1.0})


def test_long_hits_take_profit():
    bt = _bt()
    tr = _trade()
    now = pd.Timestamp("2026-05-01 00:05", tz="UTC")
    assert bt._try_close(tr, _bar(high=111, low=99), now) is True
    assert tr.status is TradeStatus.WIN
    assert tr.pnl == 10 * (110 - 100)


def test_long_hits_stop():
    bt = _bt()
    tr = _trade()
    now = pd.Timestamp("2026-05-01 00:05", tz="UTC")
    assert bt._try_close(tr, _bar(high=101, low=94), now) is True
    assert tr.status is TradeStatus.LOSS
    assert tr.pnl == 10 * (95 - 100)


def test_stop_wins_ties():
    bt = _bt()
    tr = _trade()
    now = pd.Timestamp("2026-05-01 00:05", tz="UTC")
    # both SL and TP inside the bar -> conservative: stop first
    assert bt._try_close(tr, _bar(high=111, low=94), now) is True
    assert tr.status is TradeStatus.LOSS


def test_no_touch_keeps_open():
    bt = _bt()
    tr = _trade()
    now = pd.Timestamp("2026-05-01 00:05", tz="UTC")
    assert bt._try_close(tr, _bar(high=109, low=96), now) is False
    assert tr.status is TradeStatus.OPEN


def _short_trade(entry=100.0, sl=105.0, tp=90.0, size=10.0):
    sig = Signal("X", Direction.SHORT, entry, sl, tp,
                 pd.Timestamp("2026-05-01", tz="UTC"), "t")
    return Trade(signal=sig, size=size, entry_time=sig.time, entry_price=entry)


def test_short_hits_take_profit():
    bt = _bt()
    tr = _short_trade()
    now = pd.Timestamp("2026-05-01 00:05", tz="UTC")
    assert bt._try_close(tr, _bar(high=101, low=89), now) is True
    assert tr.status is TradeStatus.WIN
    assert tr.pnl == 10 * (90 - 100) * -1      # +100


def test_short_hits_stop():
    bt = _bt()
    tr = _short_trade()
    now = pd.Timestamp("2026-05-01 00:05", tz="UTC")
    assert bt._try_close(tr, _bar(high=106, low=99), now) is True
    assert tr.status is TradeStatus.LOSS
    assert tr.pnl == 10 * (105 - 100) * -1     # -50


def test_short_stop_wins_ties():
    bt = _bt()
    tr = _short_trade()
    now = pd.Timestamp("2026-05-01 00:05", tz="UTC")
    assert bt._try_close(tr, _bar(high=106, low=89), now) is True
    assert tr.status is TradeStatus.LOSS       # both touched -> stop first
