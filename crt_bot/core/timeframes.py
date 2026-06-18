"""Timeframe parsing, resampling and multi-timeframe set helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .models import OHLCV

# Map our timeframe labels to pandas offset aliases and to minutes.
_TF_TO_PANDAS = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "1H": "1h",
    "4H": "4h",
    "1D": "1D",
}

_TF_TO_MINUTES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1H": 60,
    "4H": 240,
    "1D": 1440,
}


def tf_minutes(tf: str) -> int:
    try:
        return _TF_TO_MINUTES[tf]
    except KeyError as exc:  # pragma: no cover - guard
        raise ValueError(f"Unknown timeframe {tf!r}") from exc


def pandas_rule(tf: str) -> str:
    try:
        return _TF_TO_PANDAS[tf]
    except KeyError as exc:  # pragma: no cover - guard
        raise ValueError(f"Unknown timeframe {tf!r}") from exc


@dataclass(frozen=True)
class TFSet:
    """A high / mid / low timeframe triple used by the strategy."""

    htf: str
    mtf: str
    ltf: str

    @classmethod
    def from_config(cls, tf_sets: dict, name: str) -> "TFSet":
        if name not in tf_sets:
            raise ValueError(f"tf_set {name!r} not found in config tf_sets")
        s = tf_sets[name]
        return cls(htf=s["htf"], mtf=s["mtf"], ltf=s["ltf"])

    @property
    def name(self) -> str:
        return f"{self.htf}-{self.mtf}-{self.ltf}"

    def all(self) -> tuple[str, str, str]:
        return (self.htf, self.mtf, self.ltf)


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample a base OHLCV DataFrame to a higher timeframe.

    The base frame must have a sorted DatetimeIndex and the standard OHLCV
    columns. Empty buckets are dropped so the result only contains real bars.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample expects a DatetimeIndex")

    rule = pandas_rule(tf)
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # label/closed='left' so a bar is timestamped by its OPEN time, matching how
    # MT5/TradingView display candles.
    out = df.resample(rule, label="left", closed="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out[OHLCV]


def closed_view(df: pd.DataFrame, now: pd.Timestamp, tf: str) -> pd.DataFrame:
    """Return only the candles of ``df`` that are fully closed at ``now``.

    A candle opened at time ``t`` on timeframe ``tf`` closes at
    ``t + tf_minutes``. We include it only if it has closed at-or-before
    ``now`` -- this is what prevents look-ahead bias in the backtester.
    """

    if df.empty:
        return df
    minutes = tf_minutes(tf)
    close_times = df.index + pd.Timedelta(minutes=minutes)
    return df[close_times <= now]
