"""Trading-session windows (e.g. the New York killzone).

Setups are only hunted while price is inside the configured session. Times are
interpreted in the session timezone and compared against tz-aware candle
timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import pandas as pd


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


@dataclass
class Session:
    enabled: bool
    name: str
    timezone: str
    start: time
    end: time

    @classmethod
    def from_config(cls, cfg: dict) -> "Session":
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            name=cfg.get("name", "new_york"),
            timezone=cfg.get("timezone", "America/New_York"),
            start=_parse_hhmm(cfg.get("start", "08:00")),
            end=_parse_hhmm(cfg.get("end", "16:00")),
        )

    def contains(self, ts: pd.Timestamp) -> bool:
        """Is timestamp ``ts`` inside the session window (local session time)?"""
        if not self.enabled:
            return True
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        local = ts.tz_convert(self.timezone)
        t = local.time()
        if self.start <= self.end:
            return self.start <= t <= self.end
        # window crosses midnight
        return t >= self.start or t <= self.end

    def session_date(self, ts: pd.Timestamp) -> pd.Timestamp:
        """The local calendar date of ``ts`` -- used to group prior sessions."""
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(self.timezone).normalize()
