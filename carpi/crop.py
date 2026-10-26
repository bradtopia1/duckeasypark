"""Cropping helpers for isolating a detected object from the frame."""

from __future__ import annotations

from typing import Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # x, y, width, height


def scale_box(box: BBox, from_shape: Tuple[int, int], to_shape: Tuple[int, int]) -> BBox:
    """Scale ``box`` from one frame size to another.

    ``from_shape`` / ``to_shape`` are ``(height, width)``, as returned by
    ``image.shape[:2]``. Boxes are detected in one stage's resolution and used
    in another's (motion, detection, full-resolution), so they need carrying
    between the two.
    """
    from_h, from_w = from_shape[:2]
    to_h, to_w = to_shape[:2]
    if (from_w, from_h) == (to_w, to_h):
        return box
    sx, sy = to_w / from_w, to_h / from_h
    x, y, w, h = box
    return (
        int(round(x * sx)), int(round(y * sy)),
        int(round(w * sx)), int(round(h * sy)),
    )


def clamp_box(box: BBox, width: int, height: int) -> BBox:
    """Clamp a bounding box so it lies fully inside a ``width`` x ``height`` frame."""
    x, y, w, h = box
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return x, y, w, h


def crop_bbox(image: np.ndarray, box: BBox, *, pad: float = 0.0) -> np.ndarray:
    """Return the region of ``image`` covered by ``box``.

    ``pad`` grows the box by that fraction of its size on every side (e.g.
    ``pad=0.1`` adds 10% padding) before clamping to the image bounds.
    """
    height, width = image.shape[:2]
    x, y, w, h = clamp_box(box, width, height)

    if pad:
        dx = int(round(w * pad))
        dy = int(round(h * pad))
        x, y, w, h = clamp_box((x - dx, y - dy, w + 2 * dx, h + 2 * dy), width, height)

    return image[y : y + h, x : x + w].copy()
