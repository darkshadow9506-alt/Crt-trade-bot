"""MetaTrader 5 OHLCV feed (forex / metals / indices, for prop accounts).

This is the *exact* chart your broker / prop firm judges you on -- the same
candles you see in the MT5 terminal. Requires the ``MetaTrader5`` package and
a running, logged-in MT5 terminal on this Windows machine:
``pip install MetaTrader5``. Imported lazily so the rest of the bot runs on
any platform.

Symbol names vary by broker: gold is usually ``XAUUSD``; the Nasdaq CFD (NQ)
is ``USTEC`` / ``US100`` / ``NAS100``; the S&P CFD (ES) is ``US500`` /
``SPX500``. Check the exact spelling in the terminal's Market Watch.

Timestamps: MT5 returns candles in the broker's *server* time (often UTC+2/+3),
which would shift session windows if taken as UTC. By default the feed
auto-detects the server's UTC offset from a live tick and normalises all
candle times to real UTC; set ``live.mt5_utc_offset`` to a number to override.
"""

from __future__ import annotations

import time as _time

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

_SYMBOL_HINT = (
    "check the exact name in the MT5 Market Watch "
    "(gold: XAUUSD; Nasdaq/NQ: USTEC, US100 or NAS100; S&P/ES: US500 or SPX500)"
)


class MT5Feed(DataFeed):
    def __init__(self, login=None, password=None, server=None, utc_offset="auto"):
        self.login = login
        self.password = password
        self.server = server
        self.utc_offset = utc_offset          # "auto" or hours (e.g. 2, 3, 0)
        self._detected_offset: float | None = None
        self._mt5 = None

    def _ensure(self):
        if self._mt5 is not None:
            return self._mt5
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "MetaTrader5 package not installed (pip install MetaTrader5). "
                "MT5 feed only works on Windows with a running, logged-in terminal."
            ) from exc
        kwargs = {}
        if self.login:
            kwargs = {"login": int(self.login), "password": self.password, "server": self.server}
        if not mt5.initialize(**kwargs):
            raise RuntimeError(
                f"MT5 initialize() failed: {mt5.last_error()}. "
                "Is the MT5 terminal installed, running and logged in on this machine?"
            )
        self._mt5 = mt5
        return mt5

    def _mt5_tf(self, timeframe: str):
        mt5 = self._ensure()
        name = _TF_NAMES.get(timeframe)
        if name is None:
            raise ValueError(f"MT5 feed cannot serve timeframe {timeframe!r}")
        return getattr(mt5, name)

    def _offset_hours(self, mt5, symbol: str) -> float:
        """Broker server-time offset from UTC, in hours."""
        if isinstance(self.utc_offset, (int, float)):
            return float(self.utc_offset)
        if self._detected_offset is None:
            tick = mt5.symbol_info_tick(symbol)
            ts = float(getattr(tick, "time", 0) or 0)
            if ts <= 0:
                self._detected_offset = 0.0
            else:
                # tick.time is the server wall-clock encoded as an epoch;
                # its distance from real UTC now = the server's UTC offset.
                self._detected_offset = float(round((ts - _time.time()) / 3600.0))
        return self._detected_offset

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        mt5 = self._ensure()
        tf = self._mt5_tf(timeframe)
        # make sure the symbol is visible in Market Watch, else no data comes back
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"MT5: symbol {symbol!r} not found on this broker -- {_SYMBOL_HINT}")
        # start_pos=1 skips the currently-forming bar
        rates = mt5.copy_rates_from_pos(symbol, tf, 1, limit)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"MT5: no {timeframe} data for {symbol!r} ({mt5.last_error()}) -- {_SYMBOL_HINT}"
            )
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        offset = self._offset_hours(mt5, symbol)
        if offset:
            df["time"] = df["time"] - pd.Timedelta(hours=offset)
        df = df.set_index("time").sort_index()
        df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns:
            df["volume"] = df.get("real_volume", 0.0)
        return df[OHLCV].astype(float)
