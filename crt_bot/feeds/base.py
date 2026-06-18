"""DataFeed interface and a factory to build one from config."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class DataFeed(ABC):
    """A source of closed OHLCV candles for a single symbol."""

    #: replay feeds set this True so the runner doesn't sleep between steps
    is_replay: bool = False

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Return the most recent ``limit`` **closed** candles for ``timeframe``.

        The result has a tz-aware UTC DatetimeIndex (candle *open* time) and the
        columns ``open, high, low, close, volume``, sorted ascending. The
        currently-forming candle must not be included.
        """

    def advance(self) -> bool:  # pragma: no cover - default no-op
        """Step a replay feed forward one bar. Returns False when exhausted.

        Live feeds ignore this (they always serve current data).
        """
        return True


def build_feed(cfg: dict) -> DataFeed:
    """Instantiate the feed named in ``cfg['live']['feed']``."""
    live = cfg.get("live", {})
    name = (live.get("feed") or "binance").lower()

    if name in ("binance", "crypto"):
        from .binance import BinancePublicFeed

        return BinancePublicFeed(base_url=live.get("binance_base_url"))
    if name == "mt5":
        from .mt5 import MT5Feed

        return MT5Feed(
            login=live.get("mt5_login"),
            password=live.get("mt5_password"),
            server=live.get("mt5_server"),
        )
    if name in ("csv", "replay"):
        from .csv_replay import CsvReplayFeed

        return CsvReplayFeed(
            path=live["csv_path"],
            start_index=live.get("csv_start_index", 0),
        )
    raise ValueError(f"Unknown live.feed {name!r} (use binance / mt5 / csv)")
