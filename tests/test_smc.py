"""Unit tests for the Smart Money Concepts detectors."""

from __future__ import annotations

from crt_bot.core.models import Direction
from crt_bot.core.session import Session, SessionSet
from crt_bot.smc.cisd import detect_cisd
from crt_bot.smc.crt import detect_crt
from crt_bot.smc.fib import fib_pullback_zone
from crt_bot.smc.fvg import find_fvgs, find_ifvgs
from crt_bot.smc.liquidity import detect_sweep
from crt_bot.smc.structure import last_choch, structure_events

from .util import make_df, make_df_at


# -- structure ----------------------------------------------------------
def test_choch_after_short_trend():
    rows = [
        (110, 111, 109, 110),
        (110, 110.5, 104, 105),
        (105, 106, 103, 104),     # swing low 103
        (104, 107, 104, 106),     # swing high 107
        (106, 106, 102, 102.5),   # close breaks below 103 -> bos short
        (102.5, 103, 101, 102),   # swing low 101
        (102, 104, 101.5, 103),
        (103, 109, 103, 108.5),   # close breaks above 107 -> choch long
    ]
    df = make_df(rows)
    events = structure_events(df, lookback=1)
    kinds = [(e.kind, e.direction) for e in events]
    assert ("bos", Direction.SHORT) in kinds
    assert ("choch", Direction.LONG) in kinds
    ev = last_choch(df, Direction.LONG, lookback=1)
    assert ev is not None and ev.direction is Direction.LONG


# -- FVG / iFVG ---------------------------------------------------------
def test_bullish_fvg():
    rows = [
        (10, 10.5, 9.5, 10),
        (10, 12, 10, 11.8),
        (11.8, 12.5, 11, 12),   # low 11 > high[0] 10.5 -> bullish gap
    ]
    gaps = find_fvgs(make_df(rows))
    assert len(gaps) == 1
    g = gaps[0]
    assert g.direction is Direction.LONG
    assert g.top == 11 and g.bottom == 10.5


def test_bearish_fvg():
    rows = [
        (20, 20.5, 19.5, 20),
        (18, 18.5, 17, 17.5),
        (17.5, 17.8, 16, 16.5),  # high 17.8 < low[0] 19.5 -> bearish gap
    ]
    gaps = find_fvgs(make_df(rows))
    assert len(gaps) == 1 and gaps[0].direction is Direction.SHORT


def test_ifvg_inversion():
    rows = [
        (10, 10.5, 9.5, 10),
        (10, 12, 10, 11.8),
        (11.8, 12.5, 11, 12),    # bullish FVG [10.5, 11]
        (10.6, 10.7, 9, 9.5),    # closes below 10.5 -> inverts to bearish
    ]
    inv = find_ifvgs(make_df(rows))
    assert len(inv) == 1
    assert inv[0].direction is Direction.SHORT and inv[0].inverted


# -- CRT ----------------------------------------------------------------
def test_crt_bullish():
    rows = [
        (100, 101, 99, 100),
        (100, 110, 90, 105),     # range candle
        (100, 104, 85, 102),     # manip: sweep 90, close back inside
    ]
    crt = detect_crt(make_df(rows))
    assert crt is not None
    assert crt.direction is Direction.LONG
    assert crt.high == 110 and crt.low == 90


def test_crt_bearish():
    rows = [
        (100, 101, 99, 100),
        (100, 110, 90, 95),
        (100, 115, 98, 105),     # manip: sweep 110, close back inside
    ]
    crt = detect_crt(make_df(rows))
    assert crt is not None and crt.direction is Direction.SHORT


def test_crt_rejects_close_outside():
    rows = [
        (100, 101, 99, 100),
        (100, 110, 90, 105),
        (100, 104, 85, 88),      # closes below the range low -> not a CRT
    ]
    assert detect_crt(make_df(rows)) is None


# -- CISD ---------------------------------------------------------------
def test_cisd_long():
    rows = [
        (10, 10, 9, 9),          # bearish, run open 10
        (9, 9, 8, 8),            # bearish
        (8, 11, 8, 10.5),        # closes above run open -> CISD long
    ]
    c = detect_cisd(make_df(rows), Direction.LONG)
    assert c is not None and c.level == 10


def test_cisd_short():
    rows = [
        (10, 11, 10, 11),
        (11, 12, 11, 12),
        (12, 12, 9, 9.5),        # closes below run open -> CISD short
    ]
    c = detect_cisd(make_df(rows), Direction.SHORT)
    assert c is not None and c.direction is Direction.SHORT


# -- fib ----------------------------------------------------------------
def test_fib_pullback_zone_long():
    lo, hi = fib_pullback_zone(100, 110, Direction.LONG, 0.618, 0.79)
    assert abs(hi - 103.82) < 1e-6
    assert abs(lo - 102.1) < 1e-6


# -- liquidity sweep ----------------------------------------------------
def test_liquidity_sweep_long():
    times = [
        "2026-05-01 14:00", "2026-05-01 15:00", "2026-05-01 16:00",
        "2026-05-02 14:00", "2026-05-02 15:00",
    ]
    rows = [
        (100, 110, 90, 105),     # prior session establishes low 90 / high 110
        (105, 108, 95, 100),
        (100, 109, 92, 104),
        (104, 106, 100, 103),
        (103, 105, 85, 95),      # current candle sweeps 90 and closes back above
    ]
    df = make_df_at(times, rows)
    session = SessionSet(True, [Session("ny", "America/New_York",
                                        __import__("datetime").time(8), __import__("datetime").time(16))])
    sweep = detect_sweep(df, session, df.index[-1], lookback_sessions=3)
    assert sweep is not None
    assert sweep.direction is Direction.LONG and sweep.level == 90
