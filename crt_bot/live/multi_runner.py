"""Scan many symbols across many timeframe sets at once.

Holds one :class:`LiveRunner` per (symbol, timeframe-set) stream -- each with
its own strategy, virtual-trade state and dedup -- and drives them all from a
single shared (caching) feed and a single Telegram notifier. A failing symbol
(e.g. one your feed doesn't list) is skipped with a one-time warning instead of
killing the loop.
"""

from __future__ import annotations

import time

from .runner import LiveRunner


class MultiRunner:
    def __init__(self, runners: list[LiveRunner], feed, notifier, logger=print):
        self.runners = runners
        self.feed = feed
        self.notifier = notifier
        self.log = logger
        self._warned: set[str] = set()

    @property
    def symbols(self) -> list[str]:
        return sorted({r.symbol for r in self.runners})

    @property
    def sets_per_symbol(self) -> int:
        n = len(self.symbols)
        return len(self.runners) // n if n else 0

    # -- one cycle across every stream ------------------------------------
    def step_all(self) -> None:
        if hasattr(self.feed, "clear"):
            self.feed.clear()
        errored: set[str] = set()
        for r in self.runners:
            if r.symbol in errored:
                continue  # this symbol already failed this cycle
            try:
                r.step()
            except Exception as exc:
                errored.add(r.symbol)
                if r.symbol not in self._warned:
                    self._warned.add(r.symbol)
                    self.log(f"[skip] {r.symbol}: {exc}")
            else:
                self._warned.discard(r.symbol)

    # -- driver ------------------------------------------------------------
    def run(self, poll_seconds: int = 60, max_steps: int | None = None) -> None:
        is_replay = getattr(self.feed, "is_replay", False)
        self.log(
            f"Live runner started | {len(self.runners)} streams "
            f"({len(self.symbols)} symbols × {self.sets_per_symbol} TF sets) "
            f"| feed={type(self.feed).__name__} | replay={is_replay}"
        )
        steps = 0
        while True:
            self.step_all()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            advanced = self.feed.advance()
            if is_replay and not advanced:
                self.log("Feed exhausted -- stopping.")
                break
            if not is_replay:
                time.sleep(poll_seconds)

    def test_telegram(self) -> bool:
        ok = self.notifier.send(
            f"🤖 CRT bot connected — watching {len(self.symbols)} symbols × "
            f"{self.sets_per_symbol} timeframe sets ({len(self.runners)} streams)."
        )
        self.log(
            f"Telegram test message: "
            f"{'sent' if ok else 'NOT sent (disabled/empty config)'}"
        )
        return ok
