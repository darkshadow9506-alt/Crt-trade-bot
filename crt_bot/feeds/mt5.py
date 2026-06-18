"""MetaTrader 5 OHLCV feed (forex / metals / indices, for prop accounts).

Requires the ``MetaTrader5`` package and a running MT5 terminal (Windows / VPS):
``pip install MetaTrader5``. Imported lazily so the rest of the bot runs on any
platform.
"""

from __future__ import annotations

import pandas as pd

from ..core.models import OHLCV
from .base import DataFeed

# resolved lazily to MT5 timeframe constants inside _mt5_tf()
_TF_NAMES = {
    "1min": "TIMEFRAME_M1",
    "5min": "TIMEFRAME_M5",
    "15min": "TIMEFRAME_M15",
    "30min": "TIMEFRAME_M30",
    "1H": "TIMEFRAME_H1",
    "4H": "TIMEFRAME_H4",
    "1D": "TIMEFRAME_D1",
}


class MT5Feed(DataFeed):
    def __init__(self, login=None, password=None, server=None):
        self.login = login
        self.password = password
        self.server = server
        self._mt5 = None

    def _ensure(self):
        if self._mt5 is not None:
            return self._mt5
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "MetaTrader5 package not installed (pip install MetaTrader5). "
                "MT5 feed only works on Windows/VPS with a running terminal."
            ) from exc
        kwargs = {}
        if self.login:
            kwargs = {"login": int(self.login), "password": self.password, "server": self.server}
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
        self._mt5 = mt5
        return mt5

    def _mt5_tf(self, timeframe: str):
        mt5 = self._ensure()
        name = _TF_NAMES.get(timeframe)
        if name is None:
            raise ValueError(f"MT5 feed cannot serve timeframe {timeframe!r}")
        return getattr(mt5, name)

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        mt5 = self._ensure()
        tf = self._mt5_tf(timeframe)
        # start_pos=1 skips the currently-forming bar
        rates = mt5.copy_rates_from_pos(symbol, tf, 1, limit)
        if rates is None or len(rates) == 0:
            return pd.DataFrame(columns=OHLCV)
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()
        df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns:
            df["volume"] = df.get("real_volume", 0.0)
        return df[OHLCV].astype(float)
