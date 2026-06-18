"""Live market-data feeds for signal generation.

Every feed returns **closed** OHLCV candles for an exact timeframe (the forming
candle is dropped) with a tz-aware UTC DatetimeIndex and the standard
``open, high, low, close, volume`` columns -- the same shape the backtester and
strategy already consume.
"""

from .base import DataFeed, build_feed

__all__ = ["DataFeed", "build_feed"]
