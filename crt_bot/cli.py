"""Command line interface: ``crt-bot backtest`` (and ``version``)."""

from __future__ import annotations

import argparse
import json
import os

from . import __version__
from .backtest.engine import Backtester, Costs
from .backtest.report import build_report, print_report, report_to_dict
from .config import load_config
from .core.session import SessionSet
from .core.timeframes import TFSet
from .data.loader import load_csv
from .feeds.base import build_feed
from .feeds.cache import CachingFeed
from .live.multi_runner import MultiRunner
from .live.runner import LiveRunner
from .notify.telegram import TelegramNotifier
from .risk.risk_manager import RiskManager, RiskParams
from .strategy.crt_strategy import CRTStrategy, StrategyParams


def _build(cfg: dict):
    tf_set = TFSet.from_config(cfg["tf_sets"], cfg["tf_set"])
    sessions = SessionSet.from_config(cfg.get("session", {}))
    strat_params = StrategyParams.from_config(cfg, tf_set, sessions)
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


def _resolve_symbols(live: dict, cfg: dict) -> list[str]:
    symbols = live.get("symbols") or [live.get("symbol") or cfg.get("symbol", "BTCUSDT")]
    if isinstance(symbols, str):
        symbols = [symbols]
    # de-dupe, keep order
    seen, out = set(), []
    for s in symbols:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _resolve_tf_set_names(live: dict, cfg: dict) -> list[str]:
    defs = cfg["tf_sets"]
    tfs = live.get("tf_sets")
    if tfs is None:
        return [cfg.get("tf_set", next(iter(defs)))]
    if tfs == "all" or tfs == ["all"]:
        return list(defs.keys())
    if isinstance(tfs, str):
        return [tfs]
    return list(tfs)


def _build_live(cfg: dict) -> MultiRunner:
    live = cfg.get("live", {})
    symbols = _resolve_symbols(live, cfg)
    names = _resolve_tf_set_names(live, cfg)

    htf_l = live.get("htf_limit", 300)
    mtf_l = live.get("mtf_limit", 600)
    ltf_l = live.get("ltf_limit", 600)
    feed = CachingFeed(
        build_feed(cfg),
        min_fetch=max(htf_l, mtf_l, ltf_l),
        throttle=live.get("request_throttle", 0.0),
    )
    notifier = TelegramNotifier.from_config(cfg)
    sessions = SessionSet.from_config(cfg.get("session", {}))
    risk_cfg = cfg.get("risk", {})
    state_dir = live.get("state_dir", "reports")
    send_updates = live.get("send_trade_updates", True)
    # the signal carries the actual active session; this is just a fallback label
    session_name = None

    runners: list[LiveRunner] = []
    for sym in symbols:
        for name in names:
            tf_set = TFSet.from_config(cfg["tf_sets"], name)
            scfg = {**cfg, "symbol": sym}  # signals carry this symbol
            strategy = CRTStrategy(StrategyParams.from_config(scfg, tf_set, sessions))
            state_path = f"{state_dir}/live_state_{sym}_{name}.json" if state_dir else None
            runners.append(LiveRunner(
                symbol=sym,
                feed=feed,
                strategy=strategy,
                tf_set=tf_set,
                notifier=notifier,
                htf_limit=htf_l,
                mtf_limit=mtf_l,
                ltf_limit=ltf_l,
                send_trade_updates=send_updates,
                state_path=state_path,
                account_balance=risk_cfg.get("account_balance"),
                risk_pct=risk_cfg.get("risk_per_trade_pct"),
                session_name=session_name,
            ))
    return MultiRunner(runners, feed, notifier)


def cmd_live(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    runner = _build_live(cfg)
    if args.test_telegram:
        ok = runner.test_telegram()
        return 0 if ok else 1
    poll = cfg.get("live", {}).get("poll_seconds", 30)
    runner.run(poll_seconds=poll, max_steps=args.steps)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crt-bot", description="CRT trade bot")
    parser.add_argument("--config", help="path to config.yaml", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="run a backtest")
    bt.add_argument("--data", help="override data CSV path", default=None)
    bt.add_argument("--save", action="store_true", help="save JSON report")
    bt.set_defaults(func=cmd_backtest)

    lv = sub.add_parser("live", help="generate live signals and send to Telegram")
    lv.add_argument("--steps", type=int, default=None, help="stop after N poll cycles")
    lv.add_argument("--test-telegram", action="store_true",
                    help="send a test message and exit")
    lv.set_defaults(func=cmd_live)

    ver = sub.add_parser("version", help="print version")
    ver.set_defaults(func=lambda a: (print(__version__), 0)[1])

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
