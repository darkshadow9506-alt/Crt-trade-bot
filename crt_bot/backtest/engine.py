"""Event-driven backtester.

The engine drives bar-by-bar over the lowest timeframe. At each *closed* LTF
bar it builds look-ahead-free views of every timeframe (only candles that have
already closed), asks the strategy for a signal, opens at most one trade, and
manages open trades on *subsequent* bars so nothing is filled with information
from inside the entry bar.

Fill model per bar:
* LONG  -> stop if ``low <= SL``, target if ``high >= TP`` (SL wins ties)
* SHORT -> stop if ``high >= SL``, target if ``low <= TP`` (SL wins ties)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..core.models import Direction, Signal, Trade, TradeStatus
from ..core.timeframes import TFSet, closed_view, resample, tf_minutes
from ..risk.risk_manager import RiskManager
from ..strategy.crt_strategy import CRTStrategy


@dataclass
class Costs:
    spread_points: float = 0.0
    commission_per_trade: float = 0.0
    slippage_points: float = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "Costs":
        c = cfg.get("costs", {})
        return cls(
            spread_points=c.get("spread_points", 0.0),
            commission_per_trade=c.get("commission_per_trade", 0.0),
            slippage_points=c.get("slippage_points", 0.0),
        )


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    start_balance: float = 0.0
    final_balance: float = 0.0
    halted_reason: str | None = None
    symbol: str = ""
    tf_set: str = ""


class Backtester:
    def __init__(
        self,
        strategy: CRTStrategy,
        risk: RiskManager,
        tf_set: TFSet,
        base_timeframe: str,
        costs: Costs | None = None,
    ):
        self.strategy = strategy
        self.risk = risk
        self.tf_set = tf_set
        self.base_tf = base_timeframe
        self.costs = costs or Costs()

    def _view_for(self, full: dict[str, pd.DataFrame], tf: str, now: pd.Timestamp) -> pd.DataFrame:
        return closed_view(full[tf], now, tf)

    def run(self, base_df: pd.DataFrame) -> BacktestResult:
        htf, mtf, ltf = self.tf_set.all()
        full = {
            htf: base_df if htf == self.base_tf else resample(base_df, htf),
            mtf: base_df if mtf == self.base_tf else resample(base_df, mtf),
            ltf: base_df if ltf == self.base_tf else resample(base_df, ltf),
        }
        ltf_df = full[ltf]
        ltf_min = tf_minutes(ltf)

        result = BacktestResult(
            start_balance=self.risk.balance,
            symbol=self.strategy.p.symbol,
            tf_set=self.tf_set.name,
        )
        open_trade: Trade | None = None

        for open_time, bar in ltf_df.iterrows():
            now = open_time + pd.Timedelta(minutes=ltf_min)  # bar close time

            # 1) manage an existing trade against THIS bar
            if open_trade is not None:
                closed = self._try_close(open_trade, bar, now)
                if closed:
                    self.risk.register_close(open_trade, now)
                    open_trade.pnl_pct = (
                        open_trade.pnl / result.start_balance * 100.0
                        if result.start_balance
                        else 0.0
                    )
                    result.trades.append(open_trade)
                    self.strategy.notify_trade_closed()
                    open_trade = None

            # 2) look for a new entry (only when flat & allowed)
            if open_trade is None and self.risk.can_trade(now):
                hv = self._view_for(full, htf, now)
                mv = self._view_for(full, mtf, now)
                lv = self._view_for(full, ltf, now)
                signal = self.strategy.update(now, hv, mv, lv)
                if signal is not None:
                    open_trade = self._open(signal, now)
                    if open_trade is not None:
                        self.risk.register_open()
            elif open_trade is None:
                # keep the strategy's HTF/MTF clock advancing while paused
                self.strategy.update(
                    now,
                    self._view_for(full, htf, now),
                    self._view_for(full, mtf, now),
                    self._view_for(full, ltf, now),
                )

            result.equity_curve.append((now, self.risk.balance))

        result.final_balance = self.risk.balance
        result.halted_reason = self.risk.halted_reason
        return result

    # -- order handling ----------------------------------------------------
    def _open(self, signal: Signal, now: pd.Timestamp) -> Trade | None:
        size = self.risk.position_size(signal)
        if size <= 0:
            self.strategy.reset()
            return None
        slip = self.costs.slippage_points
        spread = self.costs.spread_points
        if signal.direction is Direction.LONG:
            entry = signal.entry + slip + spread / 2.0
        else:
            entry = signal.entry - slip - spread / 2.0
        return Trade(
            signal=signal,
            size=size,
            entry_time=now,
            entry_price=entry,
        )

    def _try_close(self, trade: Trade, bar: pd.Series, now: pd.Timestamp) -> bool:
        trade.bars_held += 1
        d = trade.direction
        sl = trade.signal.stop_loss
        tp = trade.signal.take_profit
        high = float(bar["high"])
        low = float(bar["low"])
        slip = self.costs.slippage_points

        hit_sl = low <= sl if d is Direction.LONG else high >= sl
        hit_tp = high >= tp if d is Direction.LONG else low <= tp

        if not hit_sl and not hit_tp:
            return False

        # conservative: if both touched in one bar, assume stop first
        if hit_sl:
            exit_price = sl - slip if d is Direction.LONG else sl + slip
            status = TradeStatus.LOSS
        else:
            exit_price = tp - slip if d is Direction.LONG else tp + slip
            status = TradeStatus.WIN

        pnl = trade.size * (exit_price - trade.entry_price) * d.sign
        pnl -= self.costs.commission_per_trade
        trade.exit_price = exit_price
        trade.exit_time = now
        trade.status = status
        trade.pnl = pnl
        return True
