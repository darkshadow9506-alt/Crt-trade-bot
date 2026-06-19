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
