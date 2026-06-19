"""Fibonacci retracement helpers for the pullback zone."""

from __future__ import annotations

from ..core.models import Direction


def fib_pullback_zone(
    leg_low: float,
    leg_high: float,
    direction: Direction,
    min_level: float = 0.618,
    max_level: float = 0.786,
) -> tuple[float, float]:
    """Price band of the ``[min_level, max_level]`` retracement of an impulse.

    For a LONG the impulse runs ``leg_low -> leg_high`` and the pullback is
    measured downward from the high; for a SHORT it runs ``leg_high -> leg_low``
    and is measured upward from the low. Returns ``(zone_bottom, zone_top)``.
    """
    rng = leg_high - leg_low
    if direction is Direction.LONG:
        top = leg_high - min_level * rng
        bottom = leg_high - max_level * rng
    else:
        bottom = leg_low + min_level * rng
        top = leg_low + max_level * rng
    return (min(bottom, top), max(bottom, top))


def in_zone(price: float, zone: tuple[float, float]) -> bool:
    return zone[0] <= price <= zone[1]
