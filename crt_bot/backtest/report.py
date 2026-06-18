"""Performance metrics and reporting for a backtest run."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ..core.models import TradeStatus
from .engine import BacktestResult


@dataclass
class Report:
    symbol: str
    tf_set: str
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    expectancy: float
    total_return_pct: float
    max_drawdown_pct: float
    avg_rr: float
    avg_win: float
    avg_loss: float
    start_balance: float
    final_balance: float
    halted_reason: str | None


def _max_drawdown_pct(equity: list[tuple[pd.Timestamp, float]]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for _, bal in equity:
        peak = max(peak, bal)
        if peak > 0:
            dd = (peak - bal) / peak * 100.0
            max_dd = max(max_dd, dd)
    return max_dd


def build_report(result: BacktestResult) -> Report:
    trades = result.trades
    wins = [t for t in trades if t.status is TradeStatus.WIN]
    losses = [t for t in trades if t.status is TradeStatus.LOSS]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)  # positive number

    n = len(trades)
    win_rate = len(wins) / n * 100.0 if n else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
    net = sum(t.pnl for t in trades)
    expectancy = net / n if n else 0.0
    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    avg_rr = sum(t.signal.rr for t in trades) / n if n else 0.0
    total_return = (
        (result.final_balance - result.start_balance) / result.start_balance * 100.0
        if result.start_balance
        else 0.0
    )

    return Report(
        symbol=result.symbol,
        tf_set=result.tf_set,
        trades=n,
        wins=len(wins),
        losses=len(losses),
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        total_return_pct=total_return,
        max_drawdown_pct=_max_drawdown_pct(result.equity_curve),
        avg_rr=avg_rr,
        avg_win=avg_win,
        avg_loss=avg_loss,
        start_balance=result.start_balance,
        final_balance=result.final_balance,
        halted_reason=result.halted_reason,
    )


def print_report(report: Report) -> None:
    r = report
    pf = "inf" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
    print("=" * 56)
    print(f"  CRT Backtest  |  {r.symbol}  |  TF set {r.tf_set}")
    print("=" * 56)
    print(f"  Trades            : {r.trades}  (W {r.wins} / L {r.losses})")
    print(f"  Win rate          : {r.win_rate:.1f}%")
    print(f"  Profit factor     : {pf}")
    print(f"  Avg R:R (planned) : {r.avg_rr:.2f}")
    print(f"  Expectancy/trade  : {r.expectancy:,.2f}")
    print(f"  Avg win / loss    : {r.avg_win:,.2f} / {r.avg_loss:,.2f}")
    print("-" * 56)
    print(f"  Start balance     : {r.start_balance:,.2f}")
    print(f"  Final balance     : {r.final_balance:,.2f}")
    print(f"  Total return      : {r.total_return_pct:+.2f}%")
    print(f"  Max drawdown      : {r.max_drawdown_pct:.2f}%")
    if r.halted_reason:
        print(f"  !! Halted         : {r.halted_reason}")
    print("=" * 56)


def report_to_dict(report: Report) -> dict:
    return asdict(report)
