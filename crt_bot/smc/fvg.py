"""Fair Value Gaps (FVG) and Inversion FVGs (iFVG).

Bullish FVG (3-candle): ``low[i] > high[i-2]`` -> unfilled gap that acts as
support. Bearish FVG: ``high[i] < low[i-2]`` -> gap that acts as resistance.

An **iFVG** is an FVG that price has closed *through*: its polarity inverts.
A bearish FVG that price closes above becomes bullish support (and vice
versa). The CRT entry model uses a bullish iFVG/FVG to enter longs and a
bearish one to enter shorts after the LTF CHoCH.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ..core.models import Direction


@dataclass
class FVG:
    direction: Direction          # bias the gap supports
    top: float
    bottom: float
    time: datetime                # time of the 3rd candle that created the gap
    inverted: bool = False        # True if this is an iFVG (polarity flipped)
    inverted_time: datetime | None = None

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2.0

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


def find_fvgs(df: pd.DataFrame, min_size: float = 0.0) -> list[FVG]:
    """All raw (non-inverted) FVGs in ``df`` in chronological order."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df.index
    out: list[FVG] = []
    for i in range(2, len(df)):
        # bullish gap between candle i-2 high and candle i low
        if lows[i] > highs[i - 2] and (lows[i] - highs[i - 2]) >= min_size:
            out.append(FVG(Direction.LONG, lows[i], highs[i - 2], times[i]))
        # bearish gap between candle i-2 low and candle i high
        elif highs[i] < lows[i - 2] and (lows[i - 2] - highs[i]) >= min_size:
            out.append(FVG(Direction.SHORT, lows[i - 2], highs[i], times[i]))
    return out


def find_ifvgs(df: pd.DataFrame, min_size: float = 0.0) -> list[FVG]:
    """FVGs that have since been inverted (closed through), returned as iFVGs.

    For each raw FVG we scan the candles after it; the first close beyond the
    far edge inverts it. The returned zone keeps the original price band but
    flips ``direction`` and records ``inverted_time``.
    """
    closes = df["close"]
    raw = find_fvgs(df, min_size)
    out: list[FVG] = []
    for gap in raw:
        after = closes[closes.index > gap.time]
        for ts, close in after.items():
            if gap.direction is Direction.LONG and close < gap.bottom:
                out.append(
                    FVG(Direction.SHORT, gap.top, gap.bottom, gap.time, True, ts)
                )
                break
            if gap.direction is Direction.SHORT and close > gap.top:
                out.append(
                    FVG(Direction.LONG, gap.top, gap.bottom, gap.time, True, ts)
                )
                break
    return out


def latest_entry_gap(
    df: pd.DataFrame,
    direction: Direction,
    after: pd.Timestamp,
    use_ifvg: bool = True,
    min_size: float = 0.0,
) -> FVG | None:
    """Best entry gap in ``direction`` formed after ``after``.

    Prefers the most recent qualifying gap (raw FVG or iFVG) so the entry uses
    the freshest imbalance left behind by the CHoCH leg.
    """
    candidates: list[FVG] = [
        g for g in find_fvgs(df, min_size) if g.direction is direction and g.time > after
    ]
    if use_ifvg:
        candidates += [
            g
            for g in find_ifvgs(df, min_size)
            if g.direction is direction
            and g.inverted_time is not None
            and g.inverted_time > after
        ]
    if not candidates:
        return None
    # sort by the time the gap became actionable
    candidates.sort(key=lambda g: g.inverted_time or g.time)
    return candidates[-1]
