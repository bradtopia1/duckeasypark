"""Car-detector interface and shared data types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # x, y, width, height


@dataclass
class Detection:
    """A single detected object."""

    label: str
    confidence: float
    bbox: BBox
    extra: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CarDetector(ABC):
    """Turns a frame into zero or more car detections.

    ``region`` is an optional hint (typically the motion bounding box) that a
    detector may use to narrow its search. Detectors are free to ignore it.
    """

    name: str = "car-detector"

    @abstractmethod
    def detect(self, frame: np.ndarray, region: Optional[BBox] = None) -> List[Detection]:
        ...
