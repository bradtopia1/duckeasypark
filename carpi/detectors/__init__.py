"""Car-detector backends and the factory that selects one from config."""

from __future__ import annotations

from typing import Optional

from ..config import BackendConfig, ConfigError
from .base import BBox, CarDetector, Detection
from .mock import MockCarDetector

__all__ = ["BBox", "CarDetector", "Detection", "MockCarDetector", "build_car_detector"]

# Backends that are planned but not implemented yet.
_PLANNED = {
    "opencv_dnn": "OpenCV DNN (e.g. a MobileNet-SSD ONNX/TF model)",
    "yolo": "Ultralytics YOLO",
}


def build_car_detector(config: BackendConfig) -> Optional[CarDetector]:
    """Return a car detector, or ``None`` when the stage is disabled.

    ``backend: none`` disables car detection entirely, so the pipeline just
    records motion clips and stops.
    """
    backend = config.backend
    if backend == "none":
        return None
    if backend == "mock":
        return MockCarDetector(**config.options)
    if backend in _PLANNED:
        raise ConfigError(
            f"car detector backend '{backend}' is not implemented yet "
            f"({_PLANNED[backend]}). Use backend: mock, or none to disable - "
            f"see README.md."
        )
    raise ConfigError(
        f"unknown car detector backend {backend!r}; "
        f"available: none, mock, {', '.join(sorted(_PLANNED))}"
    )
