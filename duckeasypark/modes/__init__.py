"""Mode registry.

Each mode is a runner ``callable(config, **overrides) -> RunStats``. Only
``test`` is implemented for now; future modes (live camera, car-only, ...) plug
in here.
"""

from __future__ import annotations

from typing import Callable

from ..config import ConfigError
from .test import run_test_mode

MODES: dict[str, Callable] = {
    "test": run_test_mode,
}


def get_mode_runner(name: str) -> Callable:
    try:
        return MODES[name]
    except KeyError:
        available = ", ".join(sorted(MODES))
        raise ConfigError(f"unknown mode {name!r}; available modes: {available}") from None
