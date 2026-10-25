"""Inspector-classifier backends and the factory that selects one from config."""

from __future__ import annotations

from typing import Optional

from ..config import BackendConfig, ConfigError
from .base import ClassificationResult, InspectorClassifier
from .mock import MockInspectorClassifier

__all__ = [
    "ClassificationResult",
    "InspectorClassifier",
    "MockInspectorClassifier",
    "build_inspector_classifier",
]


def build_inspector_classifier(config: BackendConfig) -> Optional[InspectorClassifier]:
    """Return an inspector classifier, or ``None`` when the stage is disabled."""
    backend = config.backend
    if backend == "none":
        return None
    if backend == "mock":
        return MockInspectorClassifier(**config.options)
    if backend == "tflite":
        from .tflite import TFLiteInspectorClassifier

        options = dict(config.options)
        if "model_path" not in options:
            raise ConfigError(
                "detectors.inspector.options.model_path is required for the "
                "'tflite' backend"
            )
        return TFLiteInspectorClassifier(**options)
    raise ConfigError(
        f"unknown inspector backend {backend!r}; available: none, mock, tflite"
    )
