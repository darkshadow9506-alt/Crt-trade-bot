"""Binance public OHLCV feed (no API key required).

Provide the exact Binance symbol in config, e.g. ``BTCUSDT`` (not ``BTCUSD``).
Note: Binance geo-blocks some regions (e.g. sanctioned countries) -- if that
affects you, use the Toobit feed instead.
"""

from __future__ import annotations

from .rest_common import BinanceCompatFeed


class BinancePublicFeed(BinanceCompatFeed):
    base_url = "https://api.binance.com"
    klines_path = "/api/v3/klines"
    name = "binance"
