"""Toobit public OHLCV feed (no API key required).

Toobit exposes a Binance-compatible public klines endpoint, which makes it a
good alternative where Binance is geo-blocked (e.g. sanctioned regions).

Two markets share the same ``/quote/v1/klines`` endpoint but use different
symbol formats:

* **spot**    -> ``BTCUSDT``
* **futures** (USDT-M perpetual) -> ``BTC-SWAP-USDT``

Toobit's metals / indices / stocks (XAUUSDT, NAS100USDT, SPX500USDT, TSLAUSDT,
...) only exist on the USDT-M **futures** market. Set ``market="futures"`` and
just pass the symbol as shown in the app (e.g. ``XAUUSDT``); this feed converts
it to the ``XAU-SWAP-USDT`` contract format automatically.

Docs: https://api-docs.toobit.com/api/usdt-m-market-data
"""

from __future__ import annotations

import pandas as pd

from .rest_common import BinanceCompatFeed


class ToobitFeed(BinanceCompatFeed):
    base_url = "https://api.toobit.com"
    klines_path = "/quote/v1/klines"
    name = "toobit"

    def __init__(self, base_url: str | None = None, market: str = "spot"):
        super().__init__(base_url)
        self.market = (market or "spot").lower()

    @staticmethod
    def to_contract_symbol(symbol: str) -> str:
        """Convert an app/spot symbol to the USDT-M futures contract symbol.

        ``BTCUSDT -> BTC-SWAP-USDT``, ``NAS100USDT -> NAS100-SWAP-USDT``.
        Already-contract symbols are passed through unchanged.
        """
        s = symbol.upper()
        if "-SWAP-" in s:
            return s
        if s.endswith("USDT"):
            return f"{s[:-4]}-SWAP-USDT"
        return s

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if self.market == "futures":
            symbol = self.to_contract_symbol(symbol)
        return super().get_candles(symbol, timeframe, limit)
