"""Frame sources.

A :class:`FrameSource` abstracts *where frames come from* so the rest of the
pipeline never needs to know whether it is reading a file (test mode) or a live
camera (future modes).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


def make_fourcc(code: str) -> int:
    """Return an OpenCV fourcc integer for a 4-character code."""
    if len(code) != 4:
        raise ValueError(f"fourcc must be 4 characters, got {code!r}")
    if hasattr(cv2.VideoWriter, "fourcc"):
        return cv2.VideoWriter.fourcc(*code)
    return cv2.VideoWriter_fourcc(*code)  # type: ignore[attr-defined]


class FrameSource(ABC):
    """Base class for anything that yields video frames."""

    @property
    @abstractmethod
    def fps(self) -> float:
        ...

    @property
    @abstractmethod
    def size(self) -> Tuple[int, int]:
        """Frame size as (width, height)."""

    @abstractmethod
    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(frame_index, frame)`` pairs until the source is exhausted."""

    def close(self) -> None:  # pragma: no cover - overridden where needed
        pass


class FileSource(FrameSource):
    """Reads frames from a video file. Used by test mode."""

    def __init__(
        self,
        path: str | Path,
        *,
        loop: bool = False,
        realtime: bool = False,
        max_seconds: Optional[float] = None,
    ) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(f"video file not found: {self.path}")

        self.loop = loop
        self.realtime = realtime
        self.max_seconds = max_seconds

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"could not open video file: {self.path}")

        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps and fps > 1e-3 else 25.0

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            ok, frame = self._cap.read()
            if not ok:
                raise IOError(f"could not read any frames from: {self.path}")
            height, width = frame.shape[:2]
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._size = (width, height)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        index = 0
        start = time.perf_counter()
        while True:
            ok, frame = self._cap.read()
            if not ok:
                if self.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            yield index, frame
            index += 1

            if self.max_seconds is not None and index / self._fps >= self.max_seconds:
                break
            if self.realtime:
                target = start + index / self._fps
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

        self.close()

    def close(self) -> None:
        self._cap.release()


class CameraSource(FrameSource):
    """Placeholder for a live Raspberry Pi camera.

    Not implemented yet - test mode is the only mode for now. On a Pi this will
    wrap ``picamera2`` (``picamera2.Picamera2`` / ``start_and_capture``), and on
    a laptop it could wrap ``cv2.VideoCapture(0)``.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Live camera mode is not implemented yet. Only 'test' mode is "
            "available for now - see README.md for the planned picamera2 wiring."
        )

    @property
    def fps(self) -> float:  # pragma: no cover
        raise NotImplementedError

    @property
    def size(self) -> Tuple[int, int]:  # pragma: no cover
        raise NotImplementedError

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:  # pragma: no cover
        raise NotImplementedError
        yield  # pragma: no cover
