"""Small, dependency-light technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def last_atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR value on the most recent candle (0.0 if not enough data)."""
    if len(df) < period + 1:
        # fall back to a simple range estimate so the bot still functions early
        if len(df) < 2:
            return 0.0
        return float(true_range(df).tail(min(len(df), period)).mean())
    val = atr(df, period).iloc[-1]
    return 0.0 if np.isnan(val) else float(val)
