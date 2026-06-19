"""Multi-symbol / multi-timeframe scanning + feed caching."""

from __future__ import annotations

import pandas as pd

from crt_bot.core.timeframes import TFSet
from crt_bot.feeds.base import DataFeed
from crt_bot.feeds.cache import CachingFeed
from crt_bot.live.multi_runner import MultiRunner
from crt_bot.live.runner import LiveRunner

_TF = TFSet("1H", "5min", "1min")


class CountingFeed(DataFeed):
    is_replay = True

    def __init__(self, bad: set[str] | None = None):
        self.calls = 0
        self.bad = bad or set()

    def get_candles(self, symbol, timeframe, limit):
        self.calls += 1
        if symbol in self.bad:
            raise RuntimeError("symbol not listed")
        idx = pd.date_range("2026-05-01", periods=3, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            index=idx,
        )
        return df.tail(limit)

    def advance(self):
        return True


class NeverStrategy:
    def update(self, *a):
        return None

    def notify_trade_closed(self):
        pass

    def reset(self):
        pass


class FakeNotifier:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)
        return True

    def send_signal(self, sig, **kwargs):
        return True


def _runner(symbol, feed, notifier):
    return LiveRunner(symbol, feed, NeverStrategy(), _TF, notifier)


def test_caching_dedups_fetches_within_a_cycle():
    inner = CountingFeed()
    feed = CachingFeed(inner, min_fetch=10)
    notifier = FakeNotifier()
    # two streams, SAME symbol -> their htf/mtf/ltf fetches must be shared
    runners = [_runner("BTCUSDT", feed, notifier), _runner("BTCUSDT", feed, notifier)]
    mr = MultiRunner(runners, feed, notifier, logger=lambda *a: None)
    mr.step_all()
    assert inner.calls == 3  # 1H, 5min, 1min fetched once each (not 6)


def test_new_cycle_refetches():
    inner = CountingFeed()
    feed = CachingFeed(inner, min_fetch=10)
    notifier = FakeNotifier()
    mr = MultiRunner([_runner("BTCUSDT", feed, notifier)], feed, notifier, logger=lambda *a: None)
    mr.step_all()
    mr.step_all()
    assert inner.calls == 6  # 3 per cycle, cache cleared each cycle


def test_bad_symbol_is_skipped_and_warned_once():
    inner = CountingFeed(bad={"BADUSDT"})
    feed = CachingFeed(inner)
    notifier = FakeNotifier()
    logs: list[str] = []
    runners = [
        _runner("BADUSDT", feed, notifier),
        _runner("BTCUSDT", feed, notifier),
    ]
    mr = MultiRunner(runners, feed, notifier, logger=logs.append)
    mr.step_all()  # must not raise
    mr.step_all()
    assert any("BADUSDT" in m for m in logs)
    assert sum("BADUSDT" in m for m in logs) == 1  # warned only once


def test_run_respects_max_steps():
    inner = CountingFeed()
    feed = CachingFeed(inner, min_fetch=10)
    notifier = FakeNotifier()
    mr = MultiRunner([_runner("BTCUSDT", feed, notifier)], feed, notifier, logger=lambda *a: None)
    mr.run(poll_seconds=0, max_steps=3)
    assert inner.calls == 9  # 3 cycles × 3 timeframes


def test_test_telegram_reports_stream_count():
    inner = CountingFeed()
    feed = CachingFeed(inner)
    notifier = FakeNotifier()
    runners = [_runner(s, feed, notifier) for s in ("BTCUSDT", "ETHUSDT")]
    mr = MultiRunner(runners, feed, notifier, logger=lambda *a: None)
    assert mr.test_telegram() is True
    assert "2 symbols" in notifier.messages[-1]
