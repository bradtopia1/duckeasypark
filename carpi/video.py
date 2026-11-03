"""Frame sources.

A :class:`FrameSource` abstracts *where frames come from* so the rest of the
pipeline never needs to know whether it is reading a file (test mode) or a live
camera (motion_capture mode).
"""

from __future__ import annotations

import importlib.util
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import cv2
import numpy as np


def make_fourcc(code: str) -> int:
    """Return an OpenCV fourcc integer for a 4-character code."""
    if len(code) != 4:
        raise ValueError(f"fourcc must be 4 characters, got {code!r}")
    if hasattr(cv2.VideoWriter, "fourcc"):
        return cv2.VideoWriter.fourcc(*code)
    return cv2.VideoWriter_fourcc(*code)  # type: ignore[attr-defined]


def _fourcc_name(value: int) -> Optional[str]:
    """Decode a ``CAP_PROP_FOURCC`` value back into its 4-character code."""
    if value <= 0:
        return None
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


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


def _picamera2_available() -> bool:
    return importlib.util.find_spec("picamera2") is not None


class _OpenCVCamera:
    """A USB / V4L2 camera, read through OpenCV."""

    def __init__(
        self,
        device: Any,
        width: Optional[int],
        height: Optional[int],
        fps: float,
    ) -> None:
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            self._cap.release()
            raise IOError(
                f"could not open camera {device!r} through OpenCV - is it "
                f"connected, and not already in use by another program? "
                f"(list the devices with: ls /dev/video*)"
            )
        # Ask for MJPEG *before* the size. Over USB 2.0 the raw formats (YUYV,
        # NV12) only sustain a few fps above 640x480, so a camera asked for a
        # size in the format it happens to default to answers with 640x480 -
        # the highest its raw mode manages at 30 fps. MJPEG is what lets a USB
        # webcam deliver 1080p/1440p at 30 fps. A camera that does not offer it
        # ignores the request, so this is safe to always do.
        self._cap.set(cv2.CAP_PROP_FOURCC, make_fourcc("MJPG"))
        if width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, fps)

        reported = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.fps = reported if reported > 1e-3 else (fps if fps > 0 else 30.0)
        self.fourcc = _fourcc_name(int(self._cap.get(cv2.CAP_PROP_FOURCC)))

        # Some drivers only report a size once a frame has been read.
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            frame = self.read()
            if frame is None:
                self._cap.release()
                raise IOError(f"camera {device!r} opened but produced no frames")
            h, w = frame.shape[:2]
        self.size: Tuple[int, int] = (w, h)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class _Picamera2Camera:
    """The Raspberry Pi's CSI camera, through picamera2 / libcamera."""

    def __init__(
        self,
        device: Any,
        width: Optional[int],
        height: Optional[int],
        fps: float,
    ) -> None:
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise ImportError(
                "the 'picamera2' camera backend needs the picamera2 package, "
                "which ships with Raspberry Pi OS: "
                "sudo apt install -y python3-picamera2. On another machine, use "
                "backend: opencv with a USB camera instead."
            ) from exc

        if not isinstance(device, int):
            raise ValueError(
                f"the 'picamera2' backend takes a camera index (0, 1, ...), "
                f"not {device!r} - use backend: opencv for a device path"
            )

        main: dict = {"format": "RGB888"}
        if width is not None and height is not None:
            main["size"] = (width, height)
        controls = {"FrameRate": fps} if fps > 0 else {}

        try:
            self._picam2 = Picamera2(int(device))
            self._picam2.configure(
                self._picam2.create_video_configuration(main=main, controls=controls)
            )
            self._picam2.start()
        except Exception as exc:  # libcamera reports failures as RuntimeError
            raise IOError(f"picamera2 could not start the camera: {exc}") from exc

        self.fps = fps if fps > 0 else 30.0
        # libcamera has no fourcc of its own; the property exists so callers
        # can report one codec name for both backends.
        self.fourcc: Optional[str] = None
        # Reading one frame both confirms the camera is streaming and tells us
        # the size the driver actually negotiated (it may differ from the ask).
        frame = self.read()
        if frame is None:
            self.close()
            raise IOError("picamera2 started but produced no frames")
        h, w = frame.shape[:2]
        self.size: Tuple[int, int] = (w, h)

    def read(self) -> Optional[np.ndarray]:
        try:
            frame = self._picam2.capture_array()
        except RuntimeError:  # a dropped/failed capture - the caller retries
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        try:
            self._picam2.stop()
        except RuntimeError:  # pragma: no cover - already stopped
            pass


# How many consecutive failed reads to ride out before giving up on a camera.
# A camera can drop a frame or two; a long run means it is gone.
MAX_READ_FAILURES = 30


class CameraSource(FrameSource):
    """Live frames from a camera: the Pi's CSI camera, or a USB / V4L2 one.

    ``backend: auto`` uses picamera2 when it is installed (that is, on
    Raspberry Pi OS) and OpenCV everywhere else, so one config works both on
    the rig and on a dev machine. Force one explicitly with ``opencv`` or
    ``picamera2`` - e.g. to use a USB webcam on a Pi, where ``auto`` would
    otherwise pick picamera2 and the CSI camera.

    Unlike :class:`FileSource`, this never ends on its own: it keeps yielding
    frames until the run is stopped (Ctrl-C, 'q' in the preview, or systemd).
    """

    def __init__(
        self,
        *,
        device: Any = 0,
        backend: str = "auto",
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: float = 30.0,
    ) -> None:
        backend = str(backend).strip().lower()
        if backend == "auto":
            backend = "picamera2" if _picamera2_available() else "opencv"
        self.backend = backend

        if backend == "opencv":
            self._camera: Any = _OpenCVCamera(device, width, height, fps)
        elif backend == "picamera2":
            self._camera = _Picamera2Camera(device, width, height, fps)
        else:
            raise ValueError(f"unknown camera backend {backend!r}")

    @property
    def fps(self) -> float:
        return self._camera.fps

    @property
    def fourcc(self) -> Optional[str]:
        """The pixel format the driver negotiated (None when it has no name)."""
        return self._camera.fourcc

    @property
    def size(self) -> Tuple[int, int]:
        return self._camera.size

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        index = 0
        failures = 0
        while True:
            frame = self._camera.read()
            if frame is None:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    raise IOError(
                        f"camera stopped producing frames after {index} frame(s) - "
                        f"check the cable and that nothing else is using the device"
                    )
                time.sleep(0.05)
                continue
            failures = 0
            yield index, frame
            index += 1

    def close(self) -> None:
        self._camera.close()
