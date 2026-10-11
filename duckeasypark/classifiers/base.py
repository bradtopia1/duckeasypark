"""Parking-inspector classifier interface and shared data types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class ClassificationResult:
    """The winning label plus, optionally, the full score distribution."""

    label: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"label": self.label, "confidence": self.confidence, "scores": dict(self.scores)}


class InspectorClassifier(ABC):
    """Decides whether a cropped car image shows a parking inspector.

    ``positive_label`` names the class that means "this is an inspector".
    """

    name: str = "inspector-classifier"
    positive_label: str = "inspector"

    @abstractmethod
    def classify(self, image: np.ndarray) -> ClassificationResult:
        ...

    def is_inspector(self, result: ClassificationResult) -> bool:
        return result.label == self.positive_label
