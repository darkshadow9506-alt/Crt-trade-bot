"""Multi-session windows (Asia / London / New York)."""

from __future__ import annotations

import pandas as pd

from crt_bot.core.session import SessionSet


def _cfg():
    return {
        "enabled": True,
        "sessions": [
            {"name": "asia", "timezone": "Asia/Tokyo", "start": "09:00", "end": "15:00"},
            {"name": "london", "timezone": "Europe/London", "start": "07:00", "end": "16:00"},
            {"name": "new_york", "timezone": "America/New_York", "start": "08:00", "end": "17:00"},
        ],
    }


def test_london_active_at_london_morning():
    s = SessionSet.from_config(_cfg())
    # 09:00 UTC = 10:00 London (BST) -> inside London; NY is 05:00 (closed)
    ts = pd.Timestamp("2026-06-19 09:00", tz="UTC")
    assert s.contains(ts) is True
    assert s.active_name(ts) == "london"


def test_new_york_active_in_afternoon():
    s = SessionSet.from_config(_cfg())
    # 20:00 UTC = 16:00 NY (EDT) inside NY; London is 21:00 (closed)
    ts = pd.Timestamp("2026-06-19 20:00", tz="UTC")
    assert s.contains(ts) is True
    assert s.active_name(ts) == "new_york"


def test_dead_zone_is_inactive():
    s = SessionSet.from_config(_cfg())
    # 04:00 UTC: Tokyo 13:00 -> actually inside Asia. Use 22:30 UTC instead:
    # 22:30 UTC -> NY 18:30 (closed), London 23:30 (closed), Tokyo 07:30 (closed)
    ts = pd.Timestamp("2026-06-19 22:30", tz="UTC")
    assert s.contains(ts) is False
    assert s.active_name(ts) is None


def test_disabled_is_always_active():
    s = SessionSet.from_config({"enabled": False, "sessions": _cfg()["sessions"]})
    assert s.contains(pd.Timestamp("2026-06-19 22:30", tz="UTC")) is True


def test_legacy_single_session_config():
    # old flat format still works (backward compatible)
    s = SessionSet.from_config({
        "enabled": True, "name": "new_york",
        "timezone": "America/New_York", "start": "08:00", "end": "16:00",
    })
    assert len(s.sessions) == 1 and s.sessions[0].name == "new_york"


def test_default_sessions_when_unconfigured():
    s = SessionSet.from_config({})
    assert {x.name for x in s.sessions} == {"asia", "london", "new_york"}
