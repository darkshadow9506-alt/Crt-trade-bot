"""Risk manager: sizing and prop-firm guardrails."""

from __future__ import annotations

import pandas as pd

from crt_bot.core.models import Direction, Signal, Trade, TradeStatus
from crt_bot.risk.risk_manager import RiskManager, RiskParams


def _signal(entry=100.0, sl=90.0, tp=120.0):
    return Signal(
        symbol="X",
        direction=Direction.LONG,
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        time=pd.Timestamp("2026-05-01", tz="UTC"),
        tf_set="t",
    )


def _losing_trade(loss: float, t: pd.Timestamp) -> Trade:
    tr = Trade(signal=_signal(), size=1.0, entry_time=t, entry_price=100.0)
    tr.status = TradeStatus.LOSS
    tr.pnl = -loss
    return tr


def test_position_size():
    rm = RiskManager(RiskParams(account_balance=100_000, risk_per_trade_pct=1.0))
    size = rm.position_size(_signal(entry=100, sl=90))  # risk per unit = 10
    assert size == 100.0  # 1000 risk / 10


def test_daily_loss_gate_resets_next_day():
    rm = RiskManager(RiskParams(account_balance=100_000, max_daily_loss_pct=4.0))
    day1 = pd.Timestamp("2026-05-01 12:00", tz="UTC")
    assert rm.can_trade(day1) is True
    rm.register_open()
    rm.register_close(_losing_trade(5_000, day1), day1)  # -5% in one day
    assert rm.can_trade(day1) is False                    # daily limit hit
    day2 = pd.Timestamp("2026-05-02 12:00", tz="UTC")
    assert rm.can_trade(day2) is True                      # resets next day


def test_max_drawdown_halts():
    rm = RiskManager(RiskParams(account_balance=100_000, max_total_drawdown_pct=8.0,
                                max_daily_loss_pct=100.0))
    t = pd.Timestamp("2026-05-01 12:00", tz="UTC")
    rm.register_open()
    rm.register_close(_losing_trade(9_000, t), t)          # -9% drawdown
    assert rm.can_trade(t) is False
    assert rm.halted_reason == "max_drawdown"
