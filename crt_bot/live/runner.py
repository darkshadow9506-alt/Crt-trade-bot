"""Live runner: poll a feed, run the CRT strategy, dispatch Telegram signals.

Signals-only mode: when the strategy emits a :class:`Signal` it is sent to
Telegram and tracked as a *virtual* trade. Subsequent candles are checked for
SL/TP so the bot can (a) message the outcome and (b) reset and hunt the next
setup. Nothing is executed on a broker.

The loop is split into :meth:`step` (one fetch + evaluate cycle) and
:meth:`run` (the polling/replay driver) so it is fully testable offline.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

import pandas as pd

from ..core.models import Direction, Signal
from ..core.timeframes import TFSet, tf_minutes
from ..feeds.base import DataFeed
from ..notify.telegram import TelegramNotifier
from ..strategy.crt_strategy import CRTStrategy


def _round(x: float) -> float:
    return round(float(x), 8)


class LiveRunner:
    def __init__(
        self,
        symbol: str,
        feed: DataFeed,
        strategy: CRTStrategy,
        tf_set: TFSet,
        notifier: TelegramNotifier,
        *,
        htf_limit: int = 300,
        mtf_limit: int = 600,
        ltf_limit: int = 600,
        send_trade_updates: bool = True,
        state_path: str | None = None,
        logger=print,
    ):
        self.symbol = symbol
        self.feed = feed
        self.strategy = strategy
        self.tf_set = tf_set
        self.notifier = notifier
        self.htf_limit = htf_limit
        self.mtf_limit = mtf_limit
        self.ltf_limit = ltf_limit
        self.send_trade_updates = send_trade_updates
        self.state_path = state_path
        self.log = logger

        self._open_signal: Signal | None = None
        self._last_sig_key: tuple | None = None
        self._load_state()

    # -- one evaluation cycle ---------------------------------------------
    def step(self) -> Signal | None:
        htf = self.feed.get_candles(self.symbol, self.tf_set.htf, self.htf_limit)
        mtf = self.feed.get_candles(self.symbol, self.tf_set.mtf, self.mtf_limit)
        ltf = self.feed.get_candles(self.symbol, self.tf_set.ltf, self.ltf_limit)
        if ltf.empty:
            return None
        now = ltf.index[-1] + pd.Timedelta(minutes=tf_minutes(self.tf_set.ltf))

        if self._open_signal is not None:
            self._manage_open(ltf)
            # keep the strategy clock moving while a virtual trade is open
            if self._open_signal is not None:
                self.strategy.update(now, htf, mtf, ltf)
                return None

        signal = self.strategy.update(now, htf, mtf, ltf)
        if signal is not None:
            self._emit(signal)
        return signal

    # -- driver ------------------------------------------------------------
    def run(self, poll_seconds: int = 30, max_steps: int | None = None) -> None:
        is_replay = getattr(self.feed, "is_replay", False)
        steps = 0
        self.log(f"Live runner started | {self.symbol} | TF set {self.tf_set.name} "
                 f"| feed={type(self.feed).__name__} | replay={is_replay}")
        while True:
            try:
                self.step()
            except Exception as exc:  # pragma: no cover - resilience in live loop
                self.log(f"[error] step failed: {exc!r}")
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            if not self.feed.advance():
                self.log("Feed exhausted -- stopping.")
                break
            if not is_replay:
                time.sleep(poll_seconds)

    # -- signal emit / dedup ----------------------------------------------
    def _emit(self, sig: Signal) -> None:
        key = (sig.direction.value, _round(sig.entry), _round(sig.stop_loss),
               _round(sig.take_profit), str(sig.time))
        if key == self._last_sig_key:
            return
        self._last_sig_key = key
        self._open_signal = sig
        sent = self.notifier.send_signal(sig)
        self.log(
            f"SIGNAL {sig.symbol} {sig.direction.value.upper()} "
            f"entry={sig.entry:.5f} sl={sig.stop_loss:.5f} tp={sig.take_profit:.5f} "
            f"rr={sig.rr:.2f} (telegram={'sent' if sent else 'off'})"
        )
        self._save_state()

    # -- virtual trade management -----------------------------------------
    def _manage_open(self, ltf: pd.DataFrame) -> None:
        sig = self._open_signal
        assert sig is not None
        d = sig.direction
        after = ltf[ltf.index >= sig.time]
        for ts, bar in after.iterrows():
            high, low = float(bar["high"]), float(bar["low"])
            hit_sl = low <= sig.stop_loss if d is Direction.LONG else high >= sig.stop_loss
            hit_tp = high >= sig.take_profit if d is Direction.LONG else low <= sig.take_profit
            if hit_sl:
                self._close_virtual("SL", sig.stop_loss, ts)
                return
            if hit_tp:
                self._close_virtual("TP", sig.take_profit, ts)
                return

    def _close_virtual(self, kind: str, price: float, ts) -> None:
        sig = self._open_signal
        assert sig is not None
        won = kind == "TP"
        rr = sig.rr if won else -1.0
        emoji = "✅" if won else "🛑"
        msg = (
            f"{emoji} <b>{kind} hit</b> — {sig.symbol} {sig.direction.value.upper()}\n"
            f"Exit: <code>{price:.5f}</code>  |  Result: <b>{'+' if won else ''}{rr:.2f}R</b>"
        )
        if self.send_trade_updates:
            self.notifier.send(msg)
        self.log(f"TRADE CLOSED {kind} @ {price:.5f} ({'+' if won else ''}{rr:.2f}R)")
        self._open_signal = None
        self.strategy.notify_trade_closed()
        self._save_state()

    def test_telegram(self) -> bool:
        ok = self.notifier.send(
            f"🤖 CRT bot connected — {self.symbol} ({self.tf_set.name}) "
            f"at {datetime.utcnow():%Y-%m-%d %H:%M} UTC"
        )
        self.log(f"Telegram test message: {'sent' if ok else 'NOT sent (disabled/empty config)'}")
        return ok

    # -- state persistence -------------------------------------------------
    def _save_state(self) -> None:
        if not self.state_path:
            return
        data = {
            "last_sig_key": list(self._last_sig_key) if self._last_sig_key else None,
            "open_signal": self._sig_to_dict(self._open_signal),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _load_state(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        with open(self.state_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        k = data.get("last_sig_key")
        self._last_sig_key = tuple(k) if k else None
        self._open_signal = self._sig_from_dict(data.get("open_signal"))

    def _sig_to_dict(self, sig: Signal | None) -> dict | None:
        if sig is None:
            return None
        return {
            "symbol": sig.symbol,
            "direction": sig.direction.value,
            "entry": sig.entry,
            "stop_loss": sig.stop_loss,
            "take_profit": sig.take_profit,
            "time": str(sig.time),
            "tf_set": sig.tf_set,
        }

    def _sig_from_dict(self, d: dict | None) -> Signal | None:
        if not d:
            return None
        return Signal(
            symbol=d["symbol"],
            direction=Direction(d["direction"]),
            entry=d["entry"],
            stop_loss=d["stop_loss"],
            take_profit=d["take_profit"],
            time=pd.Timestamp(d["time"]),
            tf_set=d["tf_set"],
        )
