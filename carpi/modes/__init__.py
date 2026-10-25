"""Mode registry.

Each mode is a runner ``callable(config, **overrides) -> RunStats``. Two exist:
``test`` (frames from a video file) and ``motion_capture`` (frames from the live
camera). Future modes (car-only, ...) plug in here.
"""

from __future__ import annotations

from typing import Callable

from ..config import ConfigError
from .motion_capture import run_motion_capture_mode
from .test import run_test_mode

MODES: dict[str, Callable] = {
    "test": run_test_mode,
    "motion_capture": run_motion_capture_mode,
}


def get_mode_runner(name: str) -> Callable:
    try:
        return MODES[name]
    except KeyError:
        available = ", ".join(sorted(MODES))
        raise ConfigError(f"unknown mode {name!r}; available modes: {available}") from None
