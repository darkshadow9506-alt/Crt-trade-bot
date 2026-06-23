"""Typed data structures used across the bot.

Candle *series* are kept as pandas DataFrames (columns:
``open, high, low, close, volume`` with a tz-aware DatetimeIndex) because every
detector works on vectorised columns. The dataclasses below describe the
higher-level objects the strategy reasons about (POIs, the CRT range, signals
and trades).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# Canonical OHLCV column names expected on every candle DataFrame.
OHLCV = ["open", "high", "low", "close", "volume"]


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1

    @property
    def opposite(self) -> "Direction":
        return Direction.SHORT if self is Direction.LONG else Direction.LONG


class POIType(str, Enum):
    FVG = "fvg"
    ORDER_BLOCK = "order_block"
    LIQUIDITY_SWEEP = "liquidity_sweep"


@dataclass
class POI:
    """A high-timeframe point of interest that price has tagged."""

    poi_type: POIType
    direction: Direction          # bias the POI supports (bullish/bearish)
    top: float
    bottom: float
    time: datetime

    def contains(self, price: float) -> bool:
        return self.bottom <= price <= self.top


@dataclass
class CRTRange:
    """The CRT (Candlestick Range Theory) range defined by the manipulation.

    ``high``/``low`` are the range-candle extremes. ``direction`` is the
    expected distribution direction (LONG after a low sweep, SHORT after a
    high sweep). ``swept_level`` is the liquidity level that was taken.
    """

    direction: Direction
    high: float
    low: float
    swept_level: float
    range_candle_time: datetime
    manip_candle_time: datetime

    @property
    def size(self) -> float:
        return self.high - self.low

    @property
    def equilibrium(self) -> float:
        return (self.high + self.low) / 2.0

    def fib(self, level: float) -> float:
        """Price at a fib level measured from the manipulation extreme toward
        the opposite side (the direction we expect price to travel)."""
        if self.direction is Direction.LONG:
            # low -> high; level 0 at low, 1 at high
            return self.low + level * self.size
        return self.high - level * self.size


class SignalState(str, Enum):
    WAIT_POI = "wait_htf_poi"
    WAIT_CRT = "wait_htf_crt"
    WAIT_MTF_CHOCH = "wait_mtf_choch"
    WAIT_MTF_PULLBACK = "wait_mtf_pullback"
    WAIT_LTF_CHOCH = "wait_ltf_choch"
    WAIT_LTF_ENTRY = "wait_ltf_entry"
    IN_TRADE = "in_trade"


@dataclass
class Signal:
    """A ready-to-execute trade signal produced by the strategy."""

    symbol: str
    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    time: datetime
    tf_set: str
    reason: str = ""
    crt: Optional[CRTRange] = None
    entry_trigger: str = ""        # "ifvg" | "fvg" | "cisd"
    tp_mode: str = ""              # "crt_high" | "crt_low" | "equilibrium"
    market_bias: str = ""          # "bullish" | "bearish" | "unclear"
    bias_basis: str = ""           # "HH+HL" | "LH+LL" | "MA"
    session: str = ""              # active session at entry (london/new_york/asia)

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)

    @property
    def reward(self) -> float:
        return abs(self.take_profit - self.entry)

    @property
    def rr(self) -> float:
        r = self.risk
        return self.reward / r if r > 0 else 0.0


class TradeStatus(str, Enum):
    OPEN = "open"
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class Trade:
    """An executed (or simulated) trade and its outcome."""

    signal: Signal
    size: float                       # units / lots
    entry_time: datetime
    entry_price: float
    status: TradeStatus = TradeStatus.OPEN
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def direction(self) -> Direction:
        return self.signal.direction

    @property
    def is_open(self) -> bool:
        return self.status is TradeStatus.OPEN
