"""Shared helpers for Binance-compatible REST kline feeds.

Both Binance and Toobit expose a public, key-less ``klines`` endpoint that
returns an array of arrays whose first six elements are
``[openTime(ms), open, high, low, close, volume]``. This base class handles the
request, retries, parsing and dropping of the still-forming candle; subclasses
only set the host and path.

Transient failures (connection drops, timeouts, HTTP 429/5xx) are retried with
exponential backoff so a brief VPN/network hiccup doesn't skip a symbol.
Permanent failures (e.g. HTTP 400 for an unknown symbol) are not retried.
"""

from __future__ import annotations

import time

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

    # transient HTTP statuses worth retrying
    _RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, base_url: str | None = None, retries: int = 4, backoff: float = 0.6):
        if base_url:
            self.base_url = base_url.rstrip("/")
        self.retries = max(1, retries)
        self.backoff = backoff
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def _interval(self, timeframe: str) -> str:
        try:
            return INTERVALS[timeframe]
        except KeyError as exc:
            raise ValueError(f"{self.name} feed cannot serve timeframe {timeframe!r}") from exc

    def _request(self, url: str, params: dict):
        """GET with retry/backoff on transient errors. Returns parsed JSON.

        Retries connection resets, timeouts, chunked/incomplete reads
        ("IncompleteRead", "Response ended prematurely"), truncated bodies and
        429/5xx. Does NOT retry a permanent 4xx (e.g. unknown symbol).
        """
        import requests

        session = self._get_session()
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = session.get(url, params=params, timeout=20)
                if resp.status_code in self._RETRY_STATUS:
                    last_exc = requests.HTTPError(f"{resp.status_code} {resp.reason}")
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                resp.raise_for_status()  # 4xx -> HTTPError (handled below, no retry)
                return resp.json()
            except requests.HTTPError:
                raise  # permanent client error (e.g. bad symbol) -> don't retry
            except (requests.RequestException, ValueError) as exc:
                # transient: connection reset, timeout, incomplete/chunked read,
                # truncated JSON body, etc.
                last_exc = exc
                time.sleep(self.backoff * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        interval = self._interval(timeframe)
        payload = self._request(
            f"{self.base_url}{self.klines_path}",
            {"symbol": symbol.upper(), "interval": interval, "limit": min(limit + 1, 1000)},
        )
        # some deployments wrap the array under data/result
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("result") or []
        df = klines_to_df(payload)
        # drop the currently-forming candle so we only act on closed bars
        return df.iloc[:-1] if len(df) else df
