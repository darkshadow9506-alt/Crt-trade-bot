"""Load OHLCV candle data from CSV into a normalised DataFrame.

Expected columns (case-insensitive, flexible names):
``time/timestamp/date``, ``open``, ``high``, ``low``, ``close`` and optional
``volume``. The result has a tz-aware (UTC) ``DatetimeIndex`` and the standard
``open, high, low, close, volume`` columns, sorted ascending.
"""

from __future__ import annotations

import pandas as pd

from ..core.models import OHLCV

_TIME_ALIASES = ["time", "timestamp", "date", "datetime", "open_time"]
_COL_ALIASES = {
    "open": ["open", "o"],
    "high": ["high", "h"],
    "low": ["low", "l"],
    "close": ["close", "c"],
    "volume": ["volume", "vol", "v"],
}


def _find(cols: list[str], aliases: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def load_csv(
    path: str,
    start: str | None = None,
    end: str | None = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)

    tcol = _find(cols, _TIME_ALIASES)
    if tcol is None:
        raise ValueError(f"No time column found in {path}; have {cols}")

    rename: dict[str, str] = {}
    for canonical, aliases in _COL_ALIASES.items():
        src = _find(cols, aliases)
        if src is None and canonical != "volume":
            raise ValueError(f"Missing {canonical!r} column in {path}; have {cols}")
        if src is not None:
            rename[src] = canonical
    df = df.rename(columns=rename)

    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.assign(_ts=ts).dropna(subset=["_ts"]).set_index("_ts").sort_index()
    df.index.name = "time"
    if tz != "UTC":
        df.index = df.index.tz_convert(tz)

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[OHLCV].astype(float)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz=df.index.tz)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz=df.index.tz)]
    return df
