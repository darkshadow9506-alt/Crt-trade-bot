"""Telegram signal-card formatting."""

from __future__ import annotations

import pandas as pd

from crt_bot.core.models import CRTRange, Direction, Signal
from crt_bot.notify.telegram import format_signal


def _signal():
    crt = CRTRange(
        direction=Direction.LONG, high=65942.5, low=65180.0, swept_level=65180.0,
        range_candle_time=pd.Timestamp("2026-06-19 12:00", tz="UTC"),
        manip_candle_time=pd.Timestamp("2026-06-19 13:00", tz="UTC"),
    )
    return Signal(
        symbol="BTCUSDT", direction=Direction.LONG,
        entry=65420.5, stop_loss=65180.0, take_profit=65942.5,
        time=pd.Timestamp("2026-06-19 13:45", tz="UTC"), tf_set="4H-15min-5min",
        crt=crt, entry_trigger="ifvg", tp_mode="crt_high", market_bias="bullish",
    )


def test_card_contains_all_fields():
    msg = format_signal(_signal(), account_balance=100_000, risk_pct=0.5,
                        session_name="new_york")
    for token in ["LONG", "BTCUSDT", "Entry", "SL", "TP", "R:R",
                  "CRT range", "50%", "BULLISH", "IFVG", "size", "new_york"]:
        assert token in msg, f"missing {token!r}"
    # TP equals CRT high for a trend-aligned long
    assert "65,942" in msg


def test_card_without_account_has_no_size_line():
    msg = format_signal(_signal())
    assert "size" not in msg
    assert "Entry" in msg and "TP" in msg
