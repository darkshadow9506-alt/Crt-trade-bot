"""Replay a historical CSV bar-by-bar to test the live pipeline offline.

Loads a base-timeframe CSV once, then serves higher timeframes by resampling
the data *up to a moving cursor*. Call :meth:`advance` to step one base bar
forward -- exactly what the live runner does each poll, but deterministic.
"""

from __future__ import annotations

import pandas as pd

from ..core.timeframes import closed_view, resample
from ..data.loader import load_csv
from .base import DataFeed


class CsvReplayFeed(DataFeed):
    is_replay = True

    def __init__(self, path: str, start_index: int = 0):
        self._base = load_csv(path)
        if self._base.empty:
            raise ValueError(f"CSV {path!r} loaded 0 rows")
        deltas = self._base.index.to_series().diff().dropna()
        self._base_step = deltas.median() if not deltas.empty else pd.Timedelta(minutes=1)
        self._cache: dict[str, pd.DataFrame] = {}
        self.cursor = max(0, min(start_index, len(self._base) - 1))

    @property
    def now(self) -> pd.Timestamp:
        """Close time of the most recent closed base bar."""
        return self._base.index[self.cursor] + self._base_step

    def _resampled(self, timeframe: str) -> pd.DataFrame:
        if timeframe not in self._cache:
            self._cache[timeframe] = resample(self._base, timeframe)
        return self._cache[timeframe]

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        full = self._resampled(timeframe)
        view = closed_view(full, self.now, timeframe)
        return view.tail(limit)

    def advance(self) -> bool:
        if self.cursor >= len(self._base) - 1:
            return False
        self.cursor += 1
        return True
