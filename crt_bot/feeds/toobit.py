"""Toobit public OHLCV feed (no API key required).

Toobit exposes a Binance-compatible public klines endpoint, which makes it a
good alternative where Binance is geo-blocked (e.g. sanctioned regions). Use
the exact Toobit symbol, e.g. ``BTCUSDT``.

Docs: https://api-docs.toobit.com/  (public market data, no auth)
"""

from __future__ import annotations

from .rest_common import BinanceCompatFeed


class ToobitFeed(BinanceCompatFeed):
    base_url = "https://api.toobit.com"
    klines_path = "/quote/v1/klines"
    name = "toobit"
