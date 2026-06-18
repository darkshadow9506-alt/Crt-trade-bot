"""Position sizing and prop-firm style guardrails.

Sizing risks a fixed % of the *current* balance per trade: the position size
is chosen so that hitting the stop loses exactly that amount. The guardrails
mirror typical funded-account rules and halt trading when breached:

* daily loss limit (resets each session/calendar day)
* overall max drawdown from the running peak balance
* optional profit target (stop once reached)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.models import Signal, Trade


@dataclass
class RiskParams:
    account_balance: float = 100_000.0
    risk_per_trade_pct: float = 0.5
    max_open_trades: int = 1
    max_daily_loss_pct: float = 4.0
    max_total_drawdown_pct: float = 8.0
    profit_target_pct: float = 8.0
    stop_trading_on_target: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> "RiskParams":
        r = cfg.get("risk", {})
        return cls(
            account_balance=r.get("account_balance", 100_000.0),
            risk_per_trade_pct=r.get("risk_per_trade_pct", 0.5),
            max_open_trades=r.get("max_open_trades", 1),
            max_daily_loss_pct=r.get("max_daily_loss_pct", 4.0),
            max_total_drawdown_pct=r.get("max_total_drawdown_pct", 8.0),
            profit_target_pct=r.get("profit_target_pct", 8.0),
            stop_trading_on_target=r.get("stop_trading_on_target", False),
        )


class RiskManager:
    def __init__(self, params: RiskParams):
        self.p = params
        self.balance = params.account_balance
        self.equity = params.account_balance
        self.peak_balance = params.account_balance
        self.start_balance = params.account_balance
        self.open_trades = 0
        self._day: pd.Timestamp | None = None
        self._day_start_balance = params.account_balance
        self.halted_reason: str | None = None

    # -- daily bookkeeping -------------------------------------------------
    def _roll_day(self, now: pd.Timestamp) -> None:
        day = now.normalize()
        if self._day != day:
            self._day = day
            self._day_start_balance = self.balance

    @property
    def daily_pnl_pct(self) -> float:
        if self._day_start_balance == 0:
            return 0.0
        return (self.balance - self._day_start_balance) / self._day_start_balance * 100.0

    @property
    def total_dd_pct(self) -> float:
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance * 100.0

    @property
    def total_return_pct(self) -> float:
        return (self.balance - self.start_balance) / self.start_balance * 100.0

    # -- gating ------------------------------------------------------------
    def can_trade(self, now: pd.Timestamp) -> bool:
        self._roll_day(now)
        if self.halted_reason is not None:
            return False
        if self.open_trades >= self.p.max_open_trades:
            return False
        if self.daily_pnl_pct <= -abs(self.p.max_daily_loss_pct):
            return False  # daily limit -- pauses until next day
        if self.total_dd_pct >= abs(self.p.max_total_drawdown_pct):
            self.halted_reason = "max_drawdown"
            return False
        if (
            self.p.stop_trading_on_target
            and self.total_return_pct >= self.p.profit_target_pct
        ):
            self.halted_reason = "profit_target"
            return False
        return True

    # -- sizing ------------------------------------------------------------
    def position_size(self, signal: Signal) -> float:
        risk_amount = self.balance * self.p.risk_per_trade_pct / 100.0
        risk_per_unit = signal.risk
        if risk_per_unit <= 0:
            return 0.0
        return risk_amount / risk_per_unit

    # -- accounting --------------------------------------------------------
    def register_open(self) -> None:
        self.open_trades += 1

    def register_close(self, trade: Trade, now: pd.Timestamp) -> None:
        self._roll_day(now)
        self.balance += trade.pnl
        self.open_trades = max(0, self.open_trades - 1)
        self.peak_balance = max(self.peak_balance, self.balance)
