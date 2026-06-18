"""Order blocks (OB).

A *bullish* order block is the last down-candle before an up-displacement that
closes above that candle's high. A *bearish* OB is the last up-candle before a
down-displacement that closes below its low. The candle's full range is the
zone price is expected to react from.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from ..core.models import Direction


@dataclass
class OrderBlock:
    direction: Direction
    top: float
    bottom: float
    time: datetime

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


def find_order_blocks(df: pd.DataFrame) -> list[OrderBlock]:
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    low = df["low"].to_numpy()
    c = df["close"].to_numpy()
    times = df.index
    out: list[OrderBlock] = []
    for i in range(1, len(df)):
        # bullish OB: candle i-1 is down, candle i closes above its high
        if c[i - 1] < o[i - 1] and c[i] > h[i - 1]:
            out.append(OrderBlock(Direction.LONG, h[i - 1], low[i - 1], times[i - 1]))
        # bearish OB: candle i-1 is up, candle i closes below its low
        elif c[i - 1] > o[i - 1] and c[i] < low[i - 1]:
            out.append(OrderBlock(Direction.SHORT, h[i - 1], low[i - 1], times[i - 1]))
    return out
