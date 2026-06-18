"""Command line interface: ``crt-bot backtest`` (and ``version``)."""

from __future__ import annotations

import argparse
import json
import os

from . import __version__
from .backtest.engine import Backtester, Costs
from .backtest.report import build_report, print_report, report_to_dict
from .config import load_config
from .core.session import Session
from .core.timeframes import TFSet
from .data.loader import load_csv
from .risk.risk_manager import RiskManager, RiskParams
from .strategy.crt_strategy import CRTStrategy, StrategyParams


def _build(cfg: dict):
    tf_set = TFSet.from_config(cfg["tf_sets"], cfg["tf_set"])
    session = Session.from_config(cfg.get("session", {}))
    strat_params = StrategyParams.from_config(cfg, tf_set, session)
    strategy = CRTStrategy(strat_params)
    risk = RiskManager(RiskParams.from_config(cfg))
    costs = Costs.from_config(cfg)
    bt = Backtester(strategy, risk, tf_set, cfg.get("base_timeframe", tf_set.ltf), costs)
    return bt


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    bt_cfg = cfg.get("backtest", {})
    csv_path = args.data or bt_cfg.get("data_csv")
    if not csv_path or not os.path.exists(csv_path):
        print(f"Data CSV not found: {csv_path!r}. Set backtest.data_csv or pass --data.")
        return 2

    df = load_csv(csv_path, start=bt_cfg.get("start"), end=bt_cfg.get("end"))
    if df.empty:
        print("Loaded 0 candles -- check the date range / file.")
        return 2

    bt = _build(cfg)
    print(f"Running backtest on {len(df):,} {bt.base_tf} candles "
          f"({df.index[0]} -> {df.index[-1]}) ...")
    result = bt.run(df)
    report = build_report(result)
    print_report(report)

    report_dir = bt_cfg.get("report_dir", "reports")
    if args.save:
        os.makedirs(report_dir, exist_ok=True)
        out = os.path.join(report_dir, f"report_{report.symbol}_{report.tf_set}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report_to_dict(report), fh, indent=2)
        print(f"Saved report -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crt-bot", description="CRT trade bot")
    parser.add_argument("--config", help="path to config.yaml", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="run a backtest")
    bt.add_argument("--data", help="override data CSV path", default=None)
    bt.add_argument("--save", action="store_true", help="save JSON report")
    bt.set_defaults(func=cmd_backtest)

    ver = sub.add_parser("version", help="print version")
    ver.set_defaults(func=lambda a: (print(__version__), 0)[1])

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
