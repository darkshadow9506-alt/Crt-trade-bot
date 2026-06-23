"""Trading-session windows (Asia / London / New York ...).

Setups are hunted while price is inside **any** enabled session. A
:class:`Session` is one window (name + timezone + start/end); a
:class:`SessionSet` groups several and is what the strategy consults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import pandas as pd


def _parse_hhmm(s: str) -> time:
    hh, mm = str(s).split(":")
    return time(int(hh), int(mm))


# Sensible defaults if the user configures nothing: the major FX sessions.
_DEFAULT_SESSIONS = [
    {"name": "asia", "timezone": "Asia/Tokyo", "start": "09:00", "end": "15:00"},
    {"name": "london", "timezone": "Europe/London", "start": "07:00", "end": "16:00"},
    {"name": "new_york", "timezone": "America/New_York", "start": "08:00", "end": "17:00"},
]


@dataclass
class Session:
    """A single named trading window in its own timezone."""

    name: str
    timezone: str
    start: time
    end: time

    def contains(self, ts: pd.Timestamp) -> bool:
        """Is timestamp ``ts`` inside this window (local session time)?"""
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        local = ts.tz_convert(self.timezone)
        t = local.time()
        if self.start <= self.end:
            return self.start <= t <= self.end
        return t >= self.start or t <= self.end  # window crosses midnight

    def session_date(self, ts: pd.Timestamp) -> pd.Timestamp:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(self.timezone).normalize()


@dataclass
class SessionSet:
    """A collection of sessions. Active if price is inside *any* of them."""

    enabled: bool = True
    sessions: list[Session] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict) -> "SessionSet":
        enabled = bool(cfg.get("enabled", True))
        raw = cfg.get("sessions")
        if raw:  # new multi-session format
            sessions = [
                Session(
                    s.get("name", "session"),
                    s.get("timezone", "UTC"),
                    _parse_hhmm(s.get("start", "00:00")),
                    _parse_hhmm(s.get("end", "23:59")),
                )
                for s in raw
            ]
        elif cfg.get("timezone") or cfg.get("start") or cfg.get("name"):
            # legacy single-session block (backward compatible)
            sessions = [
                Session(
                    cfg.get("name", "session"),
                    cfg.get("timezone", "UTC"),
                    _parse_hhmm(cfg.get("start", "00:00")),
                    _parse_hhmm(cfg.get("end", "23:59")),
                )
            ]
        else:
            sessions = [
                Session(d["name"], d["timezone"], _parse_hhmm(d["start"]), _parse_hhmm(d["end"]))
                for d in _DEFAULT_SESSIONS
            ]
        return cls(enabled=enabled, sessions=sessions)

    def contains(self, ts: pd.Timestamp) -> bool:
        if not self.enabled:
            return True
        return any(s.contains(ts) for s in self.sessions)

    def active_name(self, ts: pd.Timestamp) -> str | None:
        """Name of the first session containing ``ts`` (for display)."""
        for s in self.sessions:
            if s.contains(ts):
                return s.name
        return None

    def session_date(self, ts: pd.Timestamp) -> pd.Timestamp:
        """Calendar date used to group prior sessions' liquidity."""
        if self.sessions:
            return self.sessions[0].session_date(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.normalize()
