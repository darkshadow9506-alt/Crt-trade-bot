"""Minimal end-to-end backtest example.

Run from the repo root:
    python examples/run_backtest.py
"""

from __future__ import annotations

import os
import sys

# allow running directly from the repo root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crt_bot.backtest.engine import Backtester, Costs
from crt_bot.backtest.report import build_report, print_report
from crt_bot.config import load_config
from crt_bot.core.session import SessionSet
from crt_bot.core.timeframes import TFSet
from crt_bot.data.loader import load_csv
from crt_bot.risk.risk_manager import RiskManager, RiskParams
from crt_bot.strategy.crt_strategy import CRTStrategy, StrategyParams


def main() -> None:
    cfg = load_config()
    df = load_csv(cfg["backtest"]["data_csv"])

    # Compare all three timeframe sets on the same data.
    for set_name in cfg["tf_sets"]:
        cfg["tf_set"] = set_name
        tf_set = TFSet.from_config(cfg["tf_sets"], set_name)
        sessions = SessionSet.from_config(cfg.get("session", {}))
        strategy = CRTStrategy(StrategyParams.from_config(cfg, tf_set, sessions))
        risk = RiskManager(RiskParams.from_config(cfg))
        bt = Backtester(strategy, risk, tf_set, cfg["base_timeframe"], Costs.from_config(cfg))
        result = bt.run(df)
        print_report(build_report(result))


if __name__ == "__main__":
    main()
