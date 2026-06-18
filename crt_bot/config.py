"""Config loading helpers."""

from __future__ import annotations

import os

import yaml

_DEFAULT = os.path.join("config", "config.yaml")
_EXAMPLE = os.path.join("config", "config.example.yaml")


def load_config(path: str | None = None) -> dict:
    """Load a YAML config. Falls back to the example config if no user config
    exists yet, so the bot is runnable straight after cloning."""
    if path is None:
        path = _DEFAULT if os.path.exists(_DEFAULT) else _EXAMPLE
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
