"""Motion detection using OpenCV background subtraction.

Frames are compared against a learned background model (MOG2). The result is
masked to the configured region of interest, cleaned up morphologically, and
reduced to bounding boxes whose area exceeds a sensitivity-derived minimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .config import MotionConfig

BBox = Tuple[int, int, int, int]  # x, y, width, height


@dataclass
class MotionResult:
    """Outcome of analysing a single frame."""

    motion: bool
    boxes: List[BBox]
    area_pixels: int
    mask: np.ndarray

    @property
    def union_box(self) -> BBox | None:
        """Single bounding box covering every motion blob, or ``None``."""
        if not self.boxes:
            return None
        xs = [b[0] for b in self.boxes]
        ys = [b[1] for b in self.boxes]
        x1 = max(b[0] + b[2] for b in self.boxes)
        y1 = max(b[1] + b[3] for b in self.boxes)
        return min(xs), min(ys), x1 - min(xs), y1 - min(ys)


class MotionDetector:
    """Stateful background-subtraction motion detector for a fixed frame size."""

    def __init__(
        self,
        config: MotionConfig,
        frame_size: Tuple[int, int],
        *,
        warmup_frames: int = 30,
    ) -> None:
        self.config = config
        self.width, self.height = frame_size
        self._warmup = max(0, warmup_frames)
        self._seen = 0

        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=config.shadow_detection,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._roi_mask = self._build_roi_mask()
        roi_area = int(cv2.countNonZero(self._roi_mask)) or 1
        self._min_area_px = max(
            1,
            int(roi_area * (config.min_area_percent / 100.0) * config.min_area_scale),
        )

    # -- public API --------------------------------------------------------- #
    @property
    def warming_up(self) -> bool:
        """True while the background model is still being learned."""
        return self._seen <= self._warmup

    def detect(self, frame: np.ndarray) -> MotionResult:
        self._seen += 1

        foreground = self._subtractor.apply(frame)
        # Drop the soft shadow value (127) MOG2 produces.
        _, foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(foreground, self._roi_mask)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, self._kernel)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, self._kernel, iterations=2)

        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: List[BBox] = []
        total_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_area_px:
                continue
            boxes.append(cv2.boundingRect(contour))
            total_area += int(area)

        motion = bool(boxes) and not self.warming_up
        return MotionResult(motion=motion, boxes=boxes, area_pixels=total_area, mask=foreground)

    # -- internals ---------------------------------------------------------- #
    def _build_roi_mask(self) -> np.ndarray:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        x0, y0, x1, y1 = self.config.area.to_pixels(self.width, self.height)
        mask[y0:y1, x0:x1] = 255
        return mask
