"""Liquidity: prior-session highs/lows and sweeps of them.

A sweep of a prior-session **low** takes sellside liquidity and is read as a
bullish POI; a sweep of a prior-session **high** takes buyside liquidity and is
a bearish POI.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.models import Direction
from ..core.session import Session


@dataclass
class LiquidityLevel:
    price: float
    is_high: bool
    session_date: pd.Timestamp


@dataclass
class Sweep:
    direction: Direction          # bias produced by the sweep
    level: float
    time: pd.Timestamp


def prior_session_levels(
    df: pd.DataFrame, session: Session, now: pd.Timestamp, lookback_sessions: int = 3
) -> list[LiquidityLevel]:
    """Highs/lows of the ``lookback_sessions`` sessions before ``now``'s session."""
    if df.empty:
        return []
    dates = df.index.map(session.session_date)
    cur = session.session_date(now)
    grouped = pd.DataFrame(
        {"high": df["high"].to_numpy(), "low": df["low"].to_numpy(), "date": dates}
    )
    grouped = grouped[grouped["date"] < cur]
    if grouped.empty:
        return []
    agg = grouped.groupby("date").agg(high=("high", "max"), low=("low", "min"))
    agg = agg.tail(lookback_sessions)
    out: list[LiquidityLevel] = []
    for d, row in agg.iterrows():
        out.append(LiquidityLevel(float(row["high"]), True, d))
        out.append(LiquidityLevel(float(row["low"]), False, d))
    return out


def detect_sweep(
    df: pd.DataFrame,
    session: Session,
    now: pd.Timestamp,
    lookback_sessions: int = 3,
) -> Sweep | None:
    """Did the most recent candle sweep a prior-session level and close back?"""
    if df.empty:
        return None
    levels = prior_session_levels(df, session, now, lookback_sessions)
    if not levels:
        return None
    last = df.iloc[-1]
    ts = df.index[-1]
    # bullish: wick under a prior low, close back above it
    for lv in levels:
        if not lv.is_high and last["low"] < lv.price <= last["close"]:
            return Sweep(Direction.LONG, lv.price, ts)
    # bearish: wick over a prior high, close back below it
    for lv in levels:
        if lv.is_high and last["high"] > lv.price >= last["close"]:
            return Sweep(Direction.SHORT, lv.price, ts)
    return None
