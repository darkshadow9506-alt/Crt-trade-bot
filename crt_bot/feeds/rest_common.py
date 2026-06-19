"""Shared helpers for Binance-compatible REST kline feeds.

Both Binance and Toobit expose a public, key-less ``klines`` endpoint that
returns an array of arrays whose first six elements are
``[openTime(ms), open, high, low, close, volume]``. This base class handles the
request, parsing and dropping of the still-forming candle; subclasses only set
the host and path.
"""

from __future__ import annotations

import pandas as pd

from ..core.models import OHLCV
from .base import DataFeed

# our timeframe labels -> the interval string these exchanges expect
INTERVALS = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
}


def klines_to_df(rows: list) -> pd.DataFrame:
    """Convert a Binance-style klines array into a normalised OHLCV frame."""
    if not rows:
        return pd.DataFrame(columns=OHLCV)
    times, data = [], []
    for r in rows:
        times.append(int(r[0]))
        data.append([float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
    df = pd.DataFrame(data, columns=OHLCV)
    df.index = pd.to_datetime(times, unit="ms", utc=True)
    df.index.name = "time"
    return df.sort_index()


class BinanceCompatFeed(DataFeed):
    base_url = ""
    klines_path = ""
    name = "binance-compat"

    def __init__(self, base_url: str | None = None):
        if base_url:
            self.base_url = base_url.rstrip("/")

    def _interval(self, timeframe: str) -> str:
        try:
            return INTERVALS[timeframe]
        except KeyError as exc:
            raise ValueError(f"{self.name} feed cannot serve timeframe {timeframe!r}") from exc

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("requests is required for REST feeds") from exc

        interval = self._interval(timeframe)
        resp = requests.get(
            f"{self.base_url}{self.klines_path}",
            params={"symbol": symbol.upper(), "interval": interval,
                    "limit": min(limit + 1, 1000)},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        # some deployments wrap the array under data/result
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("result") or []
        df = klines_to_df(payload)
        # drop the currently-forming candle so we only act on closed bars
        return df.iloc[:-1] if len(df) else df
