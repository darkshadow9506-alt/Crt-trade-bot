"""A caching wrapper so a multi-symbol/multi-timeframe scan hits the exchange
once per (symbol, timeframe) each cycle instead of once per stream.

Many timeframe sets share timeframes (e.g. 15min is used by two sets), so
without caching a 20-symbol x 3-set scan would make hundreds of redundant API
calls every poll. The :class:`MultiRunner` calls :meth:`clear` at the start of
each cycle; within a cycle, repeated requests for the same (symbol, timeframe)
are served from memory. Failed fetches are cached too, so one missing symbol
does not get retried by every stream in the same cycle.
"""

from __future__ import annotations

import time

import pandas as pd

from .base import DataFeed


class CachingFeed(DataFeed):
    def __init__(self, inner: DataFeed, min_fetch: int = 600, throttle: float = 0.0):
        self.inner = inner
        self.min_fetch = min_fetch
        self.throttle = throttle
        self.is_replay = getattr(inner, "is_replay", False)
        self._cache: dict[tuple[str, str], pd.DataFrame | Exception] = {}

    def clear(self) -> None:
        self._cache.clear()

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        key = (symbol, timeframe)
        if key not in self._cache:
            try:
                self._cache[key] = self.inner.get_candles(
                    symbol, timeframe, max(limit, self.min_fetch)
                )
            except Exception as exc:  # cache the failure for this cycle
                self._cache[key] = exc
            if self.throttle:
                time.sleep(self.throttle)
        val = self._cache[key]
        if isinstance(val, Exception):
            raise val
        return val.tail(limit)

    def advance(self) -> bool:
        return self.inner.advance()
