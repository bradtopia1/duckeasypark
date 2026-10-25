#!/usr/bin/env python3
"""Generate a synthetic test clip for carpi's test mode.

Draws a couple of "cars" (coloured rectangles) crossing a textured background so
the motion -> car -> inspector pipeline has something to chew on without needing
a real recording.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _fourcc(code: str) -> int:
    if hasattr(cv2.VideoWriter, "fourcc"):
        return cv2.VideoWriter.fourcc(*code)
    return cv2.VideoWriter_fourcc(*code)  # type: ignore[attr-defined]


def _open_writer(path: Path, fps: float, size: tuple[int, int]):
    """Open an mp4 writer, falling back to avi/MJPG if the codec is missing."""
    for candidate, code in ((path, "mp4v"), (path.with_suffix(".avi"), "MJPG")):
        writer = cv2.VideoWriter(str(candidate), _fourcc(code), fps, size)
        if writer.isOpened():
            return writer, candidate
        writer.release()
    raise SystemExit("could not open a video writer (tried mp4v and MJPG)")


def make_video(
    output: str | Path,
    width: int = 640,
    height: int = 480,
    fps: float = 25.0,
    seconds: float = 8.0,
    seed: int = 0,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer, output = _open_writer(output, fps, (width, height))

    rng = np.random.default_rng(seed)

    # Static, slightly textured background (an asphalt-ish scene).
    background = rng.integers(35, 70, size=(height, width, 3), dtype=np.uint8)
    cv2.rectangle(background, (0, int(height * 0.9)), (width, height), (55, 55, 60), -1)
    for _ in range(250):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        cv2.circle(background, (x, y), int(rng.integers(1, 3)), (80, 80, 85), -1)

    def sweep(t: float, start: float, end: float, left_to_right: bool) -> int | None:
        """Centre-x of a car crossing during [start, end] of the clip, else None."""
        if t < start or t > end:
            return None
        u = (t - start) / (end - start)
        if left_to_right:
            return int(-120 + u * (width + 240))
        return int(width + 120 - u * (width + 240))

    frames = max(1, int(round(fps * seconds)))
    for i in range(frames):
        frame = background.copy()
        t = i / max(1, frames - 1)

        # First "car": crosses left -> right, then a clear gap with no motion.
        cx = sweep(t, 0.02, 0.28, left_to_right=True)
        if cx is not None:
            cy = int(height * 0.55)
            cv2.rectangle(frame, (cx - 70, cy - 38), (cx + 70, cy + 38), (30, 30, 185), -1)
            cv2.rectangle(frame, (cx - 34, cy - 24), (cx + 14, cy + 10), (205, 205, 215), -1)
            cv2.circle(frame, (cx - 44, cy + 38), 12, (20, 20, 20), -1)
            cv2.circle(frame, (cx + 44, cy + 38), 12, (20, 20, 20), -1)

        # Second "car": later, crosses right -> left through the lower band.
        cx2 = sweep(t, 0.45, 0.72, left_to_right=False)
        if cx2 is not None:
            cy2 = int(height * 0.8)
            cv2.rectangle(frame, (cx2 - 60, cy2 - 30), (cx2 + 60, cy2 + 30), (50, 140, 60), -1)

        noise = rng.integers(0, 7, size=frame.shape, dtype=np.uint8)
        writer.write(cv2.add(frame, noise))

    writer.release()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic test clip.")
    parser.add_argument("-o", "--output", default=str(PROJECT_ROOT / "samples" / "test.mp4"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = make_video(args.output, args.width, args.height, args.fps, args.seconds, args.seed)
    print(f"wrote {path} ({args.width}x{args.height} @ {args.fps}fps, {args.seconds}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
