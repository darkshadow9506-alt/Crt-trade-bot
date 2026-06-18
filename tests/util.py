"""Shared test helpers."""

from __future__ import annotations

import pandas as pd


def make_df(rows, start="2026-05-01 00:00", freq="1h", tz="UTC") -> pd.DataFrame:
    """Build an OHLCV DataFrame from ``(open, high, low, close)`` tuples."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz=tz)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    df.index.name = "time"
    return df


def make_df_at(times, rows, tz="UTC") -> pd.DataFrame:
    """Build an OHLCV DataFrame from explicit timestamps."""
    idx = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    df.index.name = "time"
    return df
