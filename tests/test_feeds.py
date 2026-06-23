"""Exchange feed parsing (Binance / Toobit, Binance-compatible) and factory."""

from __future__ import annotations

import pandas as pd

from crt_bot.feeds.base import build_feed
from crt_bot.feeds.binance import BinancePublicFeed
from crt_bot.feeds.rest_common import klines_to_df
from crt_bot.feeds.toobit import ToobitFeed


def test_klines_to_df_binance_12col():
    rows = [
        [1700000000000, "100.0", "110.0", "90.0", "105.0", "12.0",
         1700000059999, "1260.0", 50, "6.0", "630.0", "0"],
        [1700000060000, "105.0", "112.0", "104.0", "108.0", "9.0",
         1700000119999, "972.0", 40, "5.0", "540.0", "0"],
    ]
    df = klines_to_df(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    assert df.iloc[0]["high"] == 110.0 and df.iloc[0]["low"] == 90.0
    assert df.iloc[1]["close"] == 108.0


def test_klines_to_df_toobit_6col():
    rows = [
        [1700000000000, "4360.0", "4362.0", "4358.0", "4361.0", "3.5"],
        [1700000300000, "4361.0", "4365.0", "4360.0", "4364.0", "2.1"],
    ]
    df = klines_to_df(rows)
    assert len(df) == 2
    assert df.iloc[0]["open"] == 4360.0 and df.iloc[1]["close"] == 4364.0


def test_klines_to_df_empty():
    assert klines_to_df([]).empty


def test_feed_endpoints_distinct():
    b = BinancePublicFeed()
    t = ToobitFeed()
    assert b.base_url == "https://api.binance.com" and b.klines_path == "/api/v3/klines"
    assert t.base_url == "https://api.toobit.com" and t.klines_path == "/quote/v1/klines"
    assert b._interval("15min") == "15m" and t._interval("4H") == "4h"


def test_build_feed_selects_exchange():
    assert isinstance(build_feed({"live": {"feed": "binance"}}), BinancePublicFeed)
    assert isinstance(build_feed({"live": {"feed": "toobit"}}), ToobitFeed)


def test_toobit_futures_symbol_conversion():
    assert ToobitFeed.to_contract_symbol("BTCUSDT") == "BTC-SWAP-USDT"
    assert ToobitFeed.to_contract_symbol("xauusdt") == "XAU-SWAP-USDT"
    assert ToobitFeed.to_contract_symbol("NAS100USDT") == "NAS100-SWAP-USDT"
    assert ToobitFeed.to_contract_symbol("SPX500USDT") == "SPX500-SWAP-USDT"
    # already-contract symbols pass through
    assert ToobitFeed.to_contract_symbol("BTC-SWAP-USDT") == "BTC-SWAP-USDT"


def test_build_feed_toobit_futures_market():
    feed = build_feed({"live": {"feed": "toobit", "market": "futures"}})
    assert isinstance(feed, ToobitFeed) and feed.market == "futures"


# -- retry / resilience -------------------------------------------------
class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self.reason = "x"
        self._payload = payload if payload is not None else []

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            import requests

            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


class _Session:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


_ROWS = [
    [1700000000000, "1", "2", "0.5", "1.5", "3"],
    [1700000060000, "1.5", "2", "1", "1.8", "2"],
]


def _feed():
    f = BinancePublicFeed()
    f.backoff = 0  # no sleeping in tests
    return f


def test_retry_recovers_from_connection_error():
    import requests

    f = _feed()
    f._session = _Session([requests.ConnectionError("boom"), _Resp(200, _ROWS)])
    df = f.get_candles("BTCUSDT", "1H", 5)
    assert len(df) == 1            # 2 rows minus the forming candle
    assert f._session.calls == 2  # retried once then succeeded


def test_retry_on_429_then_success():
    f = _feed()
    f._session = _Session([_Resp(429), _Resp(200, _ROWS)])
    df = f.get_candles("BTCUSDT", "1H", 5)
    assert len(df) == 1 and f._session.calls == 2


def test_retry_on_incomplete_read():
    import requests

    f = _feed()
    # mid-stream connection break ("IncompleteRead" / "Response ended prematurely")
    f._session = _Session([
        requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead"),
        _Resp(200, _ROWS),
    ])
    df = f.get_candles("BTCUSDT", "1H", 5)
    assert len(df) == 1 and f._session.calls == 2


def test_permanent_400_is_not_retried():
    import pytest

    f = _feed()
    f._session = _Session([_Resp(400)])
    with pytest.raises(Exception):
        f.get_candles("BADUSDT", "1H", 5)
    assert f._session.calls == 1   # bad symbol -> no pointless retries
