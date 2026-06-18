"""Generate a deterministic synthetic 1-minute OHLCV CSV for demos/tests.

This is *not* real market data -- it is a seeded random walk with intrabar
high/low, just enough structure (sweeps, gaps, pullbacks) for the backtester
to exercise the full pipeline end to end.

Usage:
    python scripts/generate_sample_data.py [out.csv] [days]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def generate(days: int = 15, start_price: float = 65_000.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = days * 24 * 60
    index = pd.date_range("2026-05-01", periods=n, freq="1min", tz="UTC")

    # minute log-returns with mild intraday seasonality (more vol around the
    # NY open) so liquidity sweeps cluster realistically.
    minutes_of_day = index.hour * 60 + index.minute
    ny_open = 13 * 60  # 13:00 UTC ~ 08:00 New York (DST-naive)
    season = 1.0 + 0.8 * np.exp(-((minutes_of_day - ny_open) ** 2) / (2 * 90 ** 2))
    base_sigma = 0.0006
    rets = rng.normal(0, base_sigma, n) * season
    # gentle mean reversion to keep price in a sane band
    close = np.empty(n)
    price = start_price
    for i in range(n):
        price *= np.exp(rets[i])
        price += (start_price - price) * 0.0002
        close[i] = price

    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # intrabar wicks proportional to the bar's move + noise
    spread = np.abs(close - open_) + close * base_sigma * rng.uniform(0.5, 2.5, n)
    high = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.0, n)
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, n)
    volume = rng.uniform(1, 100, n) * season

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).rename_axis("time")


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_BTCUSD_1min.csv"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    df = generate(days=days)
    df.to_csv(out, float_format="%.2f")
    print(f"Wrote {len(df):,} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
