"""Binance public OHLCV feed (no API key required).

Uses the public ``/api/v3/klines`` endpoint. Provide the exact Binance symbol
in config, e.g. ``BTCUSDT`` (not ``BTCUSD``).
"""

from __future__ import annotations

import pandas as pd

from ..core.models import OHLCV
from .base import DataFeed

_TF_TO_INTERVAL = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
}


class BinancePublicFeed(DataFeed):
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or "https://api.binance.com").rstrip("/")

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("requests is required for the Binance feed") from exc

        interval = _TF_TO_INTERVAL.get(timeframe)
        if interval is None:
            raise ValueError(f"Binance feed cannot serve timeframe {timeframe!r}")

        # +1 because we drop the last (forming) candle
        resp = requests.get(
            f"{self.base_url}/api/v3/klines",
            params={"symbol": symbol.upper(), "interval": interval, "limit": min(limit + 1, 1000)},
            timeout=15,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return pd.DataFrame(columns=OHLCV)

        df = pd.DataFrame(
            rows,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "trades", "tbav", "tqav", "ignore",
            ],
        )
        df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("time").sort_index()
        df = df[OHLCV].astype(float)
        # drop the currently-forming candle
        return df.iloc[:-1]
