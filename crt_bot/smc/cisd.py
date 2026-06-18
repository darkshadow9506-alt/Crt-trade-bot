"""CISD - Change in State of Delivery.

The market is "delivering" in one direction while it prints a run of
same-colour candles. Delivery flips when price closes back beyond the *open*
of that run:

* bullish CISD: a close above the open of the most recent bearish run
* bearish CISD: a close below the open of the most recent bullish run

We only confirm a CISD on the **last** candle so it can be used as a live
entry trigger.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.models import Direction


@dataclass
class CISD:
    direction: Direction
    level: float                  # the opening price of the delivery run
    time: pd.Timestamp


def detect_cisd(df: pd.DataFrame, direction: Direction) -> CISD | None:
    """Did the last candle confirm a CISD in ``direction``?"""
    n = len(df)
    if n < 2:
        return None

    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    last_close = c[-1]

    if direction is Direction.LONG:
        # find the most recent run of bearish candles ending at index -2
        j = n - 2
        if c[j] >= o[j]:  # candle before last must be bearish to have a down run
            return None
        while j - 1 >= 0 and c[j - 1] < o[j - 1]:
            j -= 1
        run_open = float(o[j])  # open of the first bearish candle in the run
        if last_close > run_open:
            return CISD(Direction.LONG, run_open, df.index[-1])
        return None

    # SHORT: most recent run of bullish candles
    j = n - 2
    if c[j] <= o[j]:
        return None
    while j - 1 >= 0 and c[j - 1] > o[j - 1]:
        j -= 1
    run_open = float(o[j])
    if last_close < run_open:
        return CISD(Direction.SHORT, run_open, df.index[-1])
    return None
