"""Timeframe resampling and the look-ahead-free closed-candle view."""

from __future__ import annotations

import pandas as pd

from crt_bot.core.timeframes import TFSet, closed_view, resample

from .util import make_df


def test_closed_view_excludes_forming_candle():
    # 1H candles opening at 00:00, 01:00, 02:00 (each closes 1h later)
    df = make_df([(1, 2, 0, 1)] * 3, start="2026-05-01 00:00", freq="1h")
    # at 02:30 the 02:00 candle (closes 03:00) is still forming -> excluded
    view = closed_view(df, pd.Timestamp("2026-05-01 02:30", tz="UTC"), "1H")
    assert len(view) == 2
    assert view.index[-1] == pd.Timestamp("2026-05-01 01:00", tz="UTC")


def test_closed_view_includes_exactly_closed_candle():
    df = make_df([(1, 2, 0, 1)] * 3, start="2026-05-01 00:00", freq="1h")
    # exactly at 02:00 the 01:00 candle has just closed and is included
    view = closed_view(df, pd.Timestamp("2026-05-01 02:00", tz="UTC"), "1H")
    assert view.index[-1] == pd.Timestamp("2026-05-01 01:00", tz="UTC")


def test_resample_aggregates_ohlc():
    # 10 one-minute bars -> one 5-min bar uses first open, max high, min low, last close
    rows = [(10 + i, 12 + i, 8 + i, 11 + i) for i in range(5)]
    df = make_df(rows, start="2026-05-01 00:00", freq="1min")
    out = resample(df, "5min")
    assert len(out) == 1
    bar = out.iloc[0]
    assert bar["open"] == 10 and bar["close"] == 15
    assert bar["high"] == 16 and bar["low"] == 8


def test_tfset_from_config():
    defs = {"1H-5min-1min": {"htf": "1H", "mtf": "5min", "ltf": "1min"}}
    tf = TFSet.from_config(defs, "1H-5min-1min")
    assert tf.htf == "1H" and tf.ltf == "1min" and tf.name == "1H-5min-1min"
