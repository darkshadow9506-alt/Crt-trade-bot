"""CRT (Candlestick Range Theory) manipulation detection.

Three-candle Power-of-3 logic applied to two closed candles:

* *range candle*       -- defines the range high/low (the candle before)
* *manipulation candle*-- sweeps one extreme of the range candle and closes
                          back **inside** the range

Bullish CRT: manipulation wick takes the range low and closes back above it ->
expect distribution up to the range high. Bearish CRT is the mirror image.
"""

from __future__ import annotations

import pandas as pd

from ..core.models import CRTRange, Direction


def detect_crt(
    df: pd.DataFrame,
    atr_value: float = 0.0,
    min_wick_sweep_atr: float = 0.0,
    require_close_inside: bool = True,
) -> CRTRange | None:
    """Test whether the **last** closed candle is a CRT manipulation candle.

    Returns the resulting :class:`CRTRange` (with target = range extreme) or
    ``None``.
    """
    if len(df) < 2:
        return None

    rng = df.iloc[-2]      # range candle
    manip = df.iloc[-1]    # manipulation candle
    rng_high, rng_low = float(rng["high"]), float(rng["low"])
    min_sweep = min_wick_sweep_atr * atr_value

    # --- bullish: sweep the range low, close back inside -------------------
    swept_low = rng_low - manip["low"]
    if (
        manip["low"] < rng_low
        and manip["close"] > rng_low
        and swept_low >= min_sweep
        and (not require_close_inside or manip["close"] <= rng_high)
    ):
        return CRTRange(
            direction=Direction.LONG,
            high=rng_high,
            low=rng_low,
            swept_level=rng_low,
            range_candle_time=df.index[-2],
            manip_candle_time=df.index[-1],
        )

    # --- bearish: sweep the range high, close back inside -----------------
    swept_high = manip["high"] - rng_high
    if (
        manip["high"] > rng_high
        and manip["close"] < rng_high
        and swept_high >= min_sweep
        and (not require_close_inside or manip["close"] >= rng_low)
    ):
        return CRTRange(
            direction=Direction.SHORT,
            high=rng_high,
            low=rng_low,
            swept_level=rng_high,
            range_candle_time=df.index[-2],
            manip_candle_time=df.index[-1],
        )

    return None
