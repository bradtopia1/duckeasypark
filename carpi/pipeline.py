"""The detection pipeline: motion -> clip -> car -> crop -> inspector -> action.

The pipeline is deliberately backend-agnostic: it only talks to the
:class:`~carpi.detectors.base.CarDetector` and
:class:`~carpi.classifiers.base.InspectorClassifier` interfaces, so the
mocks used today can be swapped for real models via config.

Speed/memory notes
------------------
Each stage runs at its own resolution (``processing.motion_width`` /
``detect_width`` / ``clip_width``; ``None`` means the source resolution). Motion
detection usually runs smallest because it only needs to trigger a clip; the car
detector can run larger so distant cars keep enough pixels; and clips are worth
keeping at full resolution when they double as training data. The car crop can
still be taken from the original full-resolution frame via
``processing.full_res_crops``. Clips are streamed to disk as an event unfolds,
so memory stays bounded regardless of how long an event lasts - only the
pre-roll ring buffer and a single "best" frame are held. Note the ring buffer
holds frames at the *clip* resolution, so full-res clips mean a much larger
buffer.

The pipeline can additionally write a crop of the moving region to ``crops/``
(the top-level ``crops`` section) as raw material for a training set; both modes
write the same crops. Those crops show whatever moved, not cars - no car
detector is involved in that path.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .classifiers.base import ClassificationResult, InspectorClassifier
from .config import AppConfig, MotionCropConfig
from .crop import crop_bbox, scale_box
from .detectors.base import CarDetector, Detection
from .metadata import ManifestWriter, build_location_provider
from .motion import BBox, MotionDetector, MotionResult
from .thermal import cpu_temperature_c
from .video import FrameSource, make_fourcc

logger = logging.getLogger("carpi.pipeline")

CAR_CROP_PAD = 0.05


def _iou(a: BBox, b: BBox) -> float:
    """Intersection over union of two boxes, ``0.0`` when they do not overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    overlap = (x1 - x0) * (y1 - y0)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


@dataclass
class InspectorAlert:
    """Everything known about a car that was flagged as a parking inspector."""

    timestamp: str
    clip_path: Path
    car_image_path: Path
    inspector_image_path: Path
    car_detection: Detection
    classification: ClassificationResult


ActionHook = Callable[[InspectorAlert], None]


@dataclass
class RunStats:
    frames_processed: int = 0
    motion_events: int = 0
    cars_detected: int = 0
    inspectors_flagged: int = 0
    crops_saved: int = 0
    clips: List[Path] = field(default_factory=list)
    car_images: List[Path] = field(default_factory=list)
    inspector_images: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.frames_processed} frames, {self.motion_events} motion event(s), "
            f"{self.cars_detected} car(s), {self.inspectors_flagged} inspector(s), "
            f"{self.crops_saved} crop(s)"
        )


def default_action_hook(alert: InspectorAlert) -> None:
    """Placeholder for what should happen when an inspector is detected.

    This is the single extension point for the "further actions" that are not
    decided yet - e.g. send a notification, start recording, sound an alarm,
    upload the image, log to a database, ...
    """
    logger.warning(
        "TODO: parking inspector detected - no action configured yet "
        "(see default_action_hook in pipeline.py). Image: %s",
        alert.inspector_image_path,
    )


class _ClipRecorder:
    """Streams frames to a video file, trying mp4 first and falling back to avi.

    Writing incrementally keeps memory flat: only the pre-roll ring buffer and
    one best frame are ever held, no matter how long the event runs.
    """

    def __init__(self, directory: Path, stem: str, fps: float, size: Tuple[int, int]) -> None:
        self._writer: Optional[cv2.VideoWriter] = None
        self.path: Optional[Path] = None
        # The rate this clip is written at, kept so the event can report it -
        # on a live feed it is the achieved capture rate, not the camera's
        # nominal fps (see Pipeline._clip_fps).
        self.fps = fps
        # What we tried, so a failure can say why rather than just "skipped".
        self.attempts: List[str] = []
        for extension, code in ((".mp4", "mp4v"), (".avi", "MJPG")):
            candidate = directory / f"{stem}{extension}"
            self.attempts.append(f"{code} -> {candidate.name}")
            writer = cv2.VideoWriter(str(candidate), make_fourcc(code), fps, size)
            if writer.isOpened():
                self._writer = writer
                self.path = candidate
                break
            writer.release()

    @property
    def ok(self) -> bool:
        return self._writer is not None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def close(self) -> Optional[Path]:
        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        if self.path is not None and self.path.exists() and self.path.stat().st_size > 0:
            return self.path
        return None


class _MotionCropSampler:
    """Writes crops of the moving region, for use as training-set material.

    A crop is only kept when the region has moved on since the last kept one, so
    a slow car yields a handful of distinct poses instead of one image per
    frame. ``min_gap_frames`` caps the rate and ``iou_threshold`` the
    repetitiveness; both live in the top-level ``crops`` section.
    """

    def __init__(
        self,
        config: MotionCropConfig,
        directory: Path,
        manifest: Optional[ManifestWriter] = None,
        *,
        full_res: bool,
    ) -> None:
        self.config = config
        self.directory = directory
        self.manifest = manifest
        self.full_res = full_res
        self._box: Optional[BBox] = None
        self._index: Optional[int] = None

    def reset(self) -> None:
        """Forget the last crop, so the next motion event starts from scratch."""
        self._box = None
        self._index = None

    def consider(
        self,
        frame: np.ndarray,
        clip_frame: np.ndarray,
        result: MotionResult,
        event_index: int,
        index: int,
        stats: RunStats,
        stamp: str,
        timestamp: datetime,
    ) -> None:
        """Keep a crop of each motion blob, unless the region hasn't moved on."""
        box = result.union_box
        if box is None:
            return
        if self._index is not None and index - self._index < self.config.min_gap_frames:
            return
        if self._box is not None and _iou(box, self._box) >= self.config.iou_threshold:
            return
        self._write(frame, clip_frame, result, event_index, index, stats, stamp, timestamp)
        self._box, self._index = box, index

    def _write(
        self,
        frame: np.ndarray,
        clip_frame: np.ndarray,
        result: MotionResult,
        event_index: int,
        index: int,
        stats: RunStats,
        stamp: str,
        timestamp: datetime,
    ) -> None:
        # Boxes come back at the motion resolution; crop from the original frame
        # unless the config asks for the (smaller) clip resolution.
        source = frame if self.full_res else clip_frame
        motion_shape = result.mask.shape[:2]
        for crop_index, box in enumerate(result.boxes):
            crop = crop_bbox(source, scale_box(box, motion_shape, source.shape[:2]), pad=self.config.pad)
            name = f"{stamp}_event{event_index:03d}_f{index:05d}_c{crop_index}.jpg"
            path = self.directory / name
            cv2.imwrite(str(path), crop)
            stats.crops_saved += 1
            if self.manifest is not None:
                self.manifest.record(
                    kind="crop", file=path, timestamp=timestamp,
                    event=event_index, frame=index,
                )


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        car_detector: Optional[CarDetector],
        inspector_classifier: Optional[InspectorClassifier],
        *,
        action_hook: Optional[ActionHook] = None,
        motion_crops: Optional[MotionCropConfig] = None,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.car_detector = car_detector
        self.inspector = inspector_classifier
        self.action_hook: ActionHook = action_hook or default_action_hook
        self.logger = log or logger

        self.output_dir = config.output_dir
        self.clips_dir = self.output_dir / "clips"
        self.cars_dir = self.output_dir / "cars"
        self.inspectors_dir = self.output_dir / "inspectors"
        self.crops_dir = self.output_dir / "crops"

        self.motion_width = config.processing.motion_width
        self.detect_width = config.processing.detect_width
        self.clip_width = config.processing.clip_width
        self.full_res_crops = config.processing.full_res_crops

        self.fps = 25.0
        self._stop = False
        self._motion_size: Tuple[int, int] = (0, 0)
        self._detect_size: Tuple[int, int] = (0, 0)
        self._clip_size: Tuple[int, int] = (0, 0)
        self._run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Set when a motion event begins, so a run stamped once and left to run
        # for days still names each clip (and its crops) by when it happened.
        self._event_stamp: Optional[str] = None
        self._event_time: Optional[datetime] = None
        self.motion: Optional[MotionDetector] = None

        # Machine-readable provenance (when/where) for every clip and crop.
        self.manifest: Optional[ManifestWriter] = None
        if config.metadata.enabled:
            manifest_path = config.metadata.manifest_path or (self.output_dir / "manifest.jsonl")
            self.manifest = ManifestWriter(
                manifest_path, build_location_provider(config.metadata.location)
            )

        # Both modes pass the shared crop config; the sampler is simply absent
        # when it is disabled.
        self.crops: Optional[_MotionCropSampler] = None
        if motion_crops is not None and motion_crops.enabled:
            self.crops = _MotionCropSampler(
                motion_crops,
                self.crops_dir,
                self.manifest,
                full_res=self.full_res_crops,
            )

    # -- public API --------------------------------------------------------- #
    def run(self, source: FrameSource, *, preview: bool = False) -> RunStats:
        self._prepare_dirs()
        self.fps = source.fps or 25.0
        self._motion_size = self._scaled_size(source.size, self.motion_width)
        self._detect_size = self._scaled_size(source.size, self.detect_width)
        self._clip_size = self._scaled_size(source.size, self.clip_width)
        self.motion = MotionDetector(self.config.motion, self._motion_size)

        motion_cfg = self.config.motion
        pre_frames = max(1, int(round(motion_cfg.pre_seconds * self.fps)))
        post_frames = max(1, int(round(motion_cfg.post_seconds * self.fps)))
        trigger_frames = motion_cfg.trigger_frames

        self.logger.info(
            "pipeline: source %dx%d -> motion %dx%d, detect %dx%d, clip %dx%d, "
            "fps=%.2f, sensitivity=%.2f, full_res_crops=%s",
            source.size[0], source.size[1],
            self._motion_size[0], self._motion_size[1],
            self._detect_size[0], self._detect_size[1],
            self._clip_size[0], self._clip_size[1],
            self.fps, motion_cfg.sensitivity, self.full_res_crops,
        )
        self.logger.info("motion area: %s", motion_cfg.area)
        if self.crops is not None:
            self.logger.info(
                "motion crops: keeping a crop when the box moves (at least %d frame(s) "
                "apart, IoU < %.2f) -> %s",
                self.crops.config.min_gap_frames,
                self.crops.config.iou_threshold,
                self.crops_dir,
            )
        if self.manifest is not None:
            self.logger.info(
                "provenance: writing when/where for each clip and crop -> %s "
                "(location source: %s)",
                self.manifest.path,
                self.manifest.location.name,
            )

        # Periodic health reporting: on a headless rig this is the only sign
        # that the pipeline is still moving and keeping up with the source.
        status_interval = self.config.logging.status_interval_seconds
        run_started_at = time.perf_counter()
        status_at = run_started_at
        status_frames = 0

        # Seconds spent in each stage over the current status window, so a run
        # that is slow can be diagnosed from the log (the timing line) rather
        # than guessed at. Whatever the window has left over is "other".
        timing = {"source": 0.0, "motion": 0.0, "clip": 0.0, "crops": 0.0}
        frame_ended_at = run_started_at

        ring: deque = deque(maxlen=pre_frames)
        # Arrival times of the last second of frames. On a live feed this is how
        # fast the source is really delivering them, which is what a clip's fps
        # should follow when the pipeline is behind (see _clip_fps).
        recent_arrivals: deque = deque()
        recorder: Optional[_ClipRecorder] = None
        capturing = False
        last_motion_index: Optional[int] = None
        consecutive = 0
        best: Optional[tuple] = None  # (area, proc_frame, full_frame, union_box)
        event_index = 0
        stats = RunStats()
        preview_ok = bool(preview)

        for index, frame in source.frames():
            now = time.perf_counter()
            # Waiting for and decoding the frame is the source's share of the
            # per-frame time.
            timing["source"] += now - frame_ended_at
            recent_arrivals.append(now)
            while len(recent_arrivals) >= 2 and now - recent_arrivals[0] > 1.0:
                recent_arrivals.popleft()

            stats.frames_processed += 1
            stage_at = time.perf_counter()
            motion_frame = self._to_size(frame, self._motion_size)
            result = self.motion.detect(motion_frame)
            timing["motion"] += time.perf_counter() - stage_at
            stage_at = time.perf_counter()
            clip_frame = self._to_size(frame, self._clip_size)
            timing["clip"] += time.perf_counter() - stage_at

            # The first moving frame of an event stamps it, so the clip, its
            # crops and the manifest all agree on when it happened.
            if result.motion and self._event_stamp is None:
                self._event_time = datetime.now().astimezone()
                self._event_stamp = self._event_time.strftime("%Y%m%d_%H%M%S")

            if self.crops is not None and result.motion:
                stage_at = time.perf_counter()
                self.crops.consider(
                    frame, clip_frame, result, event_index, index, stats,
                    self._event_stamp or self._run_stamp,
                    self._event_time or datetime.now().astimezone(),
                )
                timing["crops"] += time.perf_counter() - stage_at

            if not capturing:
                ring.append(clip_frame.copy())
                consecutive = consecutive + 1 if result.motion else 0
                if not result.motion:
                    # Idle again; the next movement starts a fresh event.
                    self._event_stamp = None
                    self._event_time = None
                if result.motion and consecutive >= trigger_frames:
                    capturing = True
                    clip_fps = self._clip_fps(source, self._recent_frame_rate(recent_arrivals))
                    recorder = _ClipRecorder(
                        self.clips_dir, self._stem(event_index), clip_fps, self._clip_size
                    )
                    if not recorder.ok:
                        self.logger.error(
                            "no usable video codec: motion clips will be skipped - tried %s "
                            "in %s. OpenCV probably has no encoder built in; check with: "
                            "python -c \"import cv2; print(cv2.getBuildInformation())\"",
                            ", ".join(recorder.attempts), self.clips_dir,
                        )
                    stage_at = time.perf_counter()
                    for buffered in ring:
                        recorder.write(buffered)
                    timing["clip"] += time.perf_counter() - stage_at
                    last_motion_index = index
                    best = self._consider(best, frame, motion_frame, result)
            else:
                stage_at = time.perf_counter()
                recorder.write(clip_frame)
                timing["clip"] += time.perf_counter() - stage_at
                if result.motion:
                    last_motion_index = index
                    best = self._consider(best, frame, motion_frame, result)
                if last_motion_index is not None and index - last_motion_index >= post_frames:
                    self._finish_event(recorder, best, event_index, stats)
                    event_index += 1
                    capturing = False
                    recorder = None
                    consecutive = 0
                    best = None
                    last_motion_index = None
                    ring.clear()
                    self._event_stamp = None
                    self._event_time = None
                    if self.crops is not None:
                        self.crops.reset()

            if preview_ok:
                preview_ok = self._show_preview(motion_frame, result)

            if status_interval > 0:
                now = time.perf_counter()
                if now - status_at >= status_interval:
                    processed = stats.frames_processed
                    self._log_status(
                        stats, processed - status_frames, processed,
                        status_at, now, run_started_at, timing,
                    )
                    status_at = now
                    status_frames = processed
                    for key in timing:
                        timing[key] = 0.0

            if self._stop:
                break

            frame_ended_at = time.perf_counter()

        if capturing and recorder is not None:
            self._finish_event(recorder, best, event_index, stats)

        if stats.frames_processed == 0:
            self.logger.warning(
                "source produced no frames - nothing was processed. Check that %s "
                "is a readable video, or that the camera is producing frames",
                getattr(source, "path", type(source).__name__),
            )

        self._close_preview()
        return stats

    # -- internals ---------------------------------------------------------- #
    def _log_status(
        self,
        stats: RunStats,
        window_frames: int,
        processed: int,
        window_start: float,
        now: float,
        run_start: float,
        timing: Dict[str, float],
    ) -> None:
        """Log one periodic "still alive and keeping up" line."""
        elapsed = now - run_start
        current_fps = window_frames / max(1e-6, now - window_start)
        average_fps = processed / max(1e-6, elapsed)
        pace = current_fps / self.fps if self.fps > 0 else 0.0
        # The pace is measured over one interval, so at genuine real time it
        # jitters a little either side of 1.0 (a live camera sits right on the
        # boundary, since capture is itself wall-clock paced). Tolerate a small
        # margin so a displayed "1.00x" is never called "slower than real time".
        verdict = "keeping up" if pace >= 0.995 else "slower than real time"
        # A rig that throttles looks exactly like a slow one, so show the
        # temperature next to the pace when the machine has a CPU sensor.
        temperature = cpu_temperature_c()
        thermal = f" | cpu {temperature:.1f}C" if temperature is not None else ""
        self.logger.info(
            "status: %d frames in %.1fs | now %.1f fps, avg %.1f fps | "
            "source %.2f fps -> %.2fx, %s | %d motion event(s), %d car(s), "
            "%d inspector(s)%s",
            processed, elapsed, current_fps, average_fps, self.fps, pace, verdict,
            stats.motion_events, stats.cars_detected, stats.inspectors_flagged,
            thermal,
        )
        if window_frames > 0:
            # Where the per-frame time went, averaged over the same window as
            # the fps above. "other" is whatever the window has left over once
            # the measured stages are subtracted - anything not instrumented.
            per_ms = 1000.0 / window_frames
            accounted = timing["source"] + timing["motion"] + timing["clip"] + timing["crops"]
            other = max(0.0, (now - window_start) - accounted)
            self.logger.info(
                "timing/frame: source %.1fms, motion %.1fms, clip %.1fms, "
                "crops %.1fms, other %.1fms",
                timing["source"] * per_ms, timing["motion"] * per_ms,
                timing["clip"] * per_ms, timing["crops"] * per_ms, other * per_ms,
            )

    @staticmethod
    def _recent_frame_rate(arrivals: "deque[float]") -> Optional[float]:
        """Frames per second the source delivered over a recent window.

        ``arrivals`` holds the timestamps of roughly the last second of frames.
        ``None`` until there are two to measure between.
        """
        if len(arrivals) < 2:
            return None
        span = arrivals[-1] - arrivals[0]
        return (len(arrivals) - 1) / span if span > 0 else None

    def _clip_fps(self, source: FrameSource, recent_rate: Optional[float]) -> float:
        """The frame rate to write the current clip at.

        A live camera is paced to real time, so when the pipeline cannot keep up
        the driver drops frames and the source really delivers them at less than
        its nominal fps; writing the clip at that achieved rate keeps it the
        right length instead of playing back fast-forward. A file - even one
        replayed faster than real time - drops nothing and keeps its own fps, as
        does a live feed that is keeping up (the small tolerance absorbs the
        normal jitter of a real-time feed).
        """
        if recent_rate is None or not source.realtime_paced:
            return self.fps
        if recent_rate >= self.fps * 0.95:
            return self.fps
        return max(1.0, recent_rate)

    def _prepare_dirs(self) -> None:
        # Only create the folders for the stages that are actually enabled.
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        if self.crops is not None:
            self.crops_dir.mkdir(parents=True, exist_ok=True)
        if self.car_detector is not None:
            self.cars_dir.mkdir(parents=True, exist_ok=True)
        if self.inspector is not None and self.car_detector is not None:
            self.inspectors_dir.mkdir(parents=True, exist_ok=True)

    def _stem(self, event_index: int) -> str:
        stamp = self._event_stamp or self._run_stamp
        return f"{stamp}_event{event_index:03d}"

    @staticmethod
    def _scaled_size(size: Tuple[int, int], width: Optional[int]) -> Tuple[int, int]:
        """Downscale ``size`` to ``width``, keeping the aspect ratio.

        ``width`` of ``None`` (or one that is not smaller) keeps the source size.
        """
        w, h = size
        if width is None or w <= width:
            return (w, h)
        scale = width / w
        return (width, max(1, int(round(h * scale))))

    @staticmethod
    def _to_size(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Return ``frame`` resized to ``size``, or the frame itself if it fits."""
        h, w = frame.shape[:2]
        if (w, h) == size:
            return frame
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    def _consider(self, best, full: np.ndarray, motion: np.ndarray, result: MotionResult):
        box = result.union_box
        if box is None:
            return best
        if best is not None and result.area_pixels <= best[0]:
            return best
        if self._detect_size == self._motion_size:
            detect = motion
        else:
            detect = self._to_size(full, self._detect_size)
        box_detect = scale_box(box, motion.shape[:2], detect.shape[:2])
        return (result.area_pixels, detect, full.copy(), box_detect)

    def _finish_event(self, recorder: Optional[_ClipRecorder], best, event_index: int, stats: RunStats) -> None:
        stats.motion_events += 1
        stem = self._stem(event_index)

        clip_path = recorder.close() if recorder is not None else None
        if clip_path is not None:
            stats.clips.append(clip_path)
            if self.manifest is not None:
                self.manifest.record(
                    kind="clip",
                    file=clip_path,
                    timestamp=self._event_time or datetime.now().astimezone(),
                    event=event_index,
                )
            if recorder is not None and recorder.fps < self.fps * 0.95:
                self.logger.info(
                    "motion event %d -> %s (%0.1f fps - the source was delivering "
                    "frames below its %.1f fps, so the clip stays real speed)",
                    event_index, clip_path.name, recorder.fps, self.fps,
                )
            else:
                self.logger.info("motion event %d -> %s", event_index, clip_path.name)
        elif recorder is not None and not recorder.ok:
            self.logger.info("motion event %d (clip skipped: no usable codec)", event_index)

        if best is None or self.car_detector is None:
            return

        _, detect, full, box_detect = best
        detections = self.car_detector.detect(detect, region=box_detect)
        if not detections:
            self.logger.info("  no car detected in %s", stem)
            return

        # Detect on the (smaller) detection frame, crop from full resolution.
        crop_source = full if self.full_res_crops else detect
        for car_index, detection in enumerate(detections):
            box = scale_box(detection.bbox, detect.shape[:2], crop_source.shape[:2])
            self._handle_car(crop_source, replace(detection, bbox=box), car_index, stem, clip_path, stats)

    def _handle_car(
        self,
        source_frame: np.ndarray,
        detection: Detection,
        car_index: int,
        stem: str,
        clip_path: Optional[Path],
        stats: RunStats,
    ) -> None:
        stats.cars_detected += 1
        crop = crop_bbox(source_frame, detection.bbox, pad=CAR_CROP_PAD)
        car_path = self.cars_dir / f"{stem}_car{car_index}.jpg"
        cv2.imwrite(str(car_path), crop)
        stats.car_images.append(car_path)
        self.logger.info(
            "  car %d (%s %.2f) bbox=%s -> %s",
            car_index, detection.label, detection.confidence, detection.bbox, car_path.name,
        )

        if self.inspector is None:
            return

        classification = self.inspector.classify(crop)
        if not self.inspector.is_inspector(classification):
            return

        stats.inspectors_flagged += 1
        inspector_path = self.inspectors_dir / f"{stem}_car{car_index}_inspector.jpg"
        cv2.imwrite(str(inspector_path), crop)
        stats.inspector_images.append(inspector_path)
        self.logger.warning(
            "  INSPECTOR flagged (%s %.2f) -> %s",
            classification.label, classification.confidence, inspector_path.name,
        )

        alert = InspectorAlert(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            clip_path=clip_path or Path(),
            car_image_path=car_path,
            inspector_image_path=inspector_path,
            car_detection=detection,
            classification=classification,
        )
        self._run_action(alert)

    def _run_action(self, alert: InspectorAlert) -> None:
        try:
            self.action_hook(alert)
        except Exception:  # noqa: BLE001 - an action must never kill the run
            self.logger.exception("action hook failed for %s", alert.inspector_image_path)

    def _show_preview(self, frame: np.ndarray, result: MotionResult) -> bool:
        canvas = frame.copy()
        if self.motion is not None:
            x0, y0, x1, y1 = self.config.motion.area.to_pixels(self.motion.width, self.motion.height)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 200, 0), 1)
        for x, y, w, h in result.boxes:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        try:
            cv2.imshow("carpi", canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                self._stop = True
                return False
        except cv2.error as exc:
            self.logger.warning("preview disabled (no GUI available): %s", exc)
            return False
        return True

    def _close_preview(self) -> None:
        try:
            cv2.destroyAllWindows()
        except cv2.error:  # pragma: no cover - headless
            pass
