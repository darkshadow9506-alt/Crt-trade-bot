"""Market structure: swing points, BOS and CHoCH detection.

A *swing high* at bar ``i`` is a fractal whose high is the strict maximum over
``[i-n, i+n]`` (``n`` = ``lookback``); a *swing low* is the symmetric minimum.
A swing is only **confirmed** once ``n`` bars exist on its right side, so the
detector never relies on candles that have not formed yet.

Walking those confirmed swings forward we classify each break of structure:

* close above the most recent swing high  -> bullish break
* close below the most recent swing low    -> bearish break

A break that *continues* the prevailing trend is a **BOS** (break of
structure); a break that *reverses* it is a **CHoCH** (change of character),
which is the reversal confirmation the CRT strategy waits for on the MTF/LTF.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ..core.models import Direction


@dataclass
class SwingPoint:
    time: datetime
    price: float
    is_high: bool


@dataclass
class StructureEvent:
    time: datetime
    direction: Direction      # direction of the break
    kind: str                 # "bos" or "choch"
    level: float              # the swing level that was broken


def swing_points(df: pd.DataFrame, lookback: int = 2) -> list[SwingPoint]:
    """Return confirmed swing highs/lows in chronological order."""
    n = lookback
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df.index
    out: list[SwingPoint] = []
    for i in range(n, len(df) - n):
        win_h = highs[i - n : i + n + 1]
        win_l = lows[i - n : i + n + 1]
        if highs[i] == win_h.max() and (win_h == highs[i]).sum() == 1:
            out.append(SwingPoint(times[i], float(highs[i]), True))
        elif lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            out.append(SwingPoint(times[i], float(lows[i]), False))
    return out


def structure_events(df: pd.DataFrame, lookback: int = 2) -> list[StructureEvent]:
    """Classify every BOS/CHoCH break across ``df``.

    The break is timestamped at the *close* that breaches the swing level.
    """
    swings = swing_points(df, lookback)
    if not swings:
        return []

    closes = df["close"]
    events: list[StructureEvent] = []

    last_sh: float | None = None
    last_sl: float | None = None
    trend: Direction | None = None
    swing_idx = 0
    swings_sorted = swings

    for ts, close in closes.items():
        # Register all swings confirmed at-or-before this bar.
        while swing_idx < len(swings_sorted) and swings_sorted[swing_idx].time <= ts:
            sp = swings_sorted[swing_idx]
            if sp.is_high:
                last_sh = sp.price
            else:
                last_sl = sp.price
            swing_idx += 1

        if last_sh is not None and close > last_sh:
            kind = "choch" if trend is Direction.SHORT else "bos"
            events.append(StructureEvent(ts, Direction.LONG, kind, last_sh))
            trend = Direction.LONG
            last_sh = None  # consume; wait for a new swing high
        elif last_sl is not None and close < last_sl:
            kind = "choch" if trend is Direction.LONG else "bos"
            events.append(StructureEvent(ts, Direction.SHORT, kind, last_sl))
            trend = Direction.SHORT
            last_sl = None

    return events


def last_choch(
    df: pd.DataFrame,
    direction: Direction,
    lookback: int = 2,
    after: pd.Timestamp | None = None,
) -> StructureEvent | None:
    """Most recent CHoCH in ``direction`` (optionally after a timestamp)."""
    for ev in reversed(structure_events(df, lookback)):
        if ev.kind != "choch" or ev.direction is not direction:
            continue
        if after is not None and ev.time <= after:
            break
        return ev
    return None


def recent_swing_extreme(
    df: pd.DataFrame, is_high: bool, lookback: int = 2
) -> SwingPoint | None:
    """The latest confirmed swing high (or low)."""
    for sp in reversed(swing_points(df, lookback)):
        if sp.is_high == is_high:
            return sp
    return None
