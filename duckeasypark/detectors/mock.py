"""A dependency-free car detector used until a real model is wired in.

It simply promotes the motion region to a car-sized bounding box. That is
enough to exercise the whole pipeline (clip -> crop -> classify) without any
model weights or heavy dependencies.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..crop import clamp_box
from .base import BBox, CarDetector, Detection


class MockCarDetector(CarDetector):
    name = "mock-car"

    def __init__(
        self,
        confidence: float = 0.9,
        expand: float = 0.15,
        label: str = "car",
    ) -> None:
        self.confidence = float(confidence)
        self.expand = float(expand)
        self.label = label

    def detect(self, frame: np.ndarray, region: Optional[BBox] = None) -> List[Detection]:
        height, width = frame.shape[:2]

        if region is not None:
            x, y, w, h = region
        else:
            # No hint: fall back to a box in the middle of the frame.
            w, h = width // 2, height // 2
            x, y = (width - w) // 2, (height - h) // 2

        if self.expand:
            dx = int(round(w * self.expand))
            dy = int(round(h * self.expand))
            x, y, w, h = x - dx, y - dy, w + 2 * dx, h + 2 * dy

        box = clamp_box((x, y, w, h), width, height)
        return [Detection(label=self.label, confidence=self.confidence, bbox=box)]
