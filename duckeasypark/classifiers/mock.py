"""A dependency-free inspector classifier used until a real model is wired in.

It flips a seeded coin: with probability ``positive_probability`` a car is
flagged as a parking inspector. Setting it to 1.0 makes every car an inspector
(handy for demoing the alert path); 0.0 never flags one.
"""

from __future__ import annotations

import random

import numpy as np

from .base import ClassificationResult, InspectorClassifier


class MockInspectorClassifier(InspectorClassifier):
    name = "mock-inspector"

    def __init__(
        self,
        positive_probability: float = 0.0,
        seed: int = 1234,
        positive_label: str = "inspector",
        positive_confidence: float = 0.95,
    ) -> None:
        self.positive_probability = max(0.0, min(1.0, float(positive_probability)))
        self.positive_label = positive_label
        self.positive_confidence = float(positive_confidence)
        self._rng = random.Random(seed)

    def classify(self, image: np.ndarray) -> ClassificationResult:
        if self._rng.random() < self.positive_probability:
            conf = self.positive_confidence
            return ClassificationResult(
                label=self.positive_label,
                confidence=conf,
                scores={self.positive_label: conf, "not_inspector": 1.0 - conf},
            )
        conf = 1.0 - self.positive_confidence
        return ClassificationResult(
            label="not_inspector",
            confidence=conf,
            scores={self.positive_label: 1.0 - conf, "not_inspector": conf},
        )
