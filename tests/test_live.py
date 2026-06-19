"""Live runner: signal dispatch, dedup, virtual-trade management, feeds."""

from __future__ import annotations

import os

import pandas as pd

from crt_bot.core.models import Direction, Signal
from crt_bot.core.timeframes import TFSet
from crt_bot.feeds.base import build_feed
from crt_bot.feeds.csv_replay import CsvReplayFeed
from crt_bot.live.runner import LiveRunner

SAMPLE = "examples/sample_BTCUSD_1min.csv"
_TF = TFSet("1H", "5min", "1min")


# -- fakes --------------------------------------------------------------
class FakeNotifier:
    def __init__(self):
        self.signals: list[Signal] = []
        self.messages: list[str] = []

    def send(self, text: str) -> bool:
        self.messages.append(text)
        return True

    def send_signal(self, sig: Signal, **kwargs) -> bool:
        self.signals.append(sig)
        return True


class FakeFeed:
    is_replay = True

    def __init__(self):
        self.frames: dict[str, pd.DataFrame] = {}

    def get_candles(self, symbol, timeframe, limit):
        return self.frames.get(timeframe, pd.DataFrame()).tail(limit)

    def advance(self):
        return False


class FakeStrategy:
    def __init__(self):
        self.calls = 0
        self.closed = 0

    def update(self, now, htf, mtf, ltf):
        self.calls += 1
        if self.calls == 1:
            return Signal("BTCUSDT", Direction.LONG, 100.0, 95.0, 110.0, now, _TF.name)
        return None

    def notify_trade_closed(self):
        self.closed += 1

    def reset(self):
        pass


def _ltf(times, rows):
    idx = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    return df


# -- tests --------------------------------------------------------------
def test_signal_is_sent_and_tracked():
    feed = FakeFeed()
    feed.frames["1H"] = _ltf(["2026-05-01 09:00"], [(100, 101, 99, 100)])
    feed.frames["5min"] = _ltf(["2026-05-01 09:50"], [(100, 101, 99, 100)])
    feed.frames["1min"] = _ltf(["2026-05-01 09:58", "2026-05-01 09:59"],
                               [(100, 100.5, 99.5, 100), (100, 100.5, 99.5, 100)])
    notifier = FakeNotifier()
    runner = LiveRunner("BTCUSDT", feed, FakeStrategy(), _TF, notifier)

    sig = runner.step()
    assert sig is not None
    assert len(notifier.signals) == 1
    assert runner._open_signal is not None


def test_dedup_does_not_resend():
    feed = FakeFeed()
    notifier = FakeNotifier()
    runner = LiveRunner("BTCUSDT", feed, FakeStrategy(), _TF, notifier)
    s = Signal("BTCUSDT", Direction.LONG, 100, 95, 110,
               pd.Timestamp("2026-05-01 10:00", tz="UTC"), _TF.name)
    runner._emit(s)
    runner._emit(s)  # identical -> ignored
    assert len(notifier.signals) == 1


def test_virtual_trade_closes_on_tp():
    feed = FakeFeed()
    notifier = FakeNotifier()
    strat = FakeStrategy()
    runner = LiveRunner("BTCUSDT", feed, strat, _TF, notifier)

    # step 1: emit the signal at now = 10:00
    feed.frames["1H"] = _ltf(["2026-05-01 09:00"], [(100, 101, 99, 100)])
    feed.frames["5min"] = _ltf(["2026-05-01 09:50"], [(100, 101, 99, 100)])
    feed.frames["1min"] = _ltf(["2026-05-01 09:59"], [(100, 100.5, 99.5, 100)])
    runner.step()
    assert runner._open_signal is not None
    open_time = runner._open_signal.time  # = 10:00 UTC

    # step 2: a later candle spikes through the TP (110)
    feed.frames["1min"] = _ltf(
        [pd.Timestamp("2026-05-01 09:59", tz="UTC"), open_time],
        [(100, 100.5, 99.5, 100), (100, 111, 100, 109)],
    )
    runner.step()
    assert runner._open_signal is None
    assert strat.closed == 1
    assert any("TP hit" in m for m in notifier.messages)


def test_test_telegram():
    runner = LiveRunner("BTCUSDT", FakeFeed(), FakeStrategy(), _TF, FakeNotifier())
    assert runner.test_telegram() is True


# -- feeds --------------------------------------------------------------
def test_csv_replay_feed_shape():
    if not os.path.exists(SAMPLE):
        import pytest

        pytest.skip("sample data not generated")
    feed = CsvReplayFeed(SAMPLE, start_index=5000)
    assert feed.is_replay is True
    df = feed.get_candles("BTCUSD", "1H", 10)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) <= 10
    assert df.index.tz is not None
    now_before = feed.now
    assert feed.advance() is True
    assert feed.now > now_before


def test_build_feed_csv():
    cfg = {"live": {"feed": "csv", "csv_path": SAMPLE}}
    if not os.path.exists(SAMPLE):
        import pytest

        pytest.skip("sample data not generated")
    feed = build_feed(cfg)
    assert isinstance(feed, CsvReplayFeed)
