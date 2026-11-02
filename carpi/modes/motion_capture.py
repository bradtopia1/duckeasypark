"""Motion-capture mode: watch the live camera and record every motion event.

This is test mode's pipeline with the camera in place of a video file, so
everything that decides *what* is captured is shared with it: the motion
settings (``motion.sensitivity``, ``motion.min_area_percent``,
``motion.trigger_frames``, ``motion.area``, the clip's ``pre_seconds`` /
``post_seconds``), the per-stage resolutions and the AI backends all come from
the top-level sections of the config. Only ``modes.motion_capture`` is specific
to this mode, and it only describes the camera.

Unlike test mode it never finishes on its own - it watches until it is stopped
(Ctrl-C, 'q' in the preview window, or systemd), which is what you want in the
autostart unit on the rig.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..classifiers import build_inspector_classifier
from ..config import AppConfig
from ..detectors import build_car_detector
from ..pipeline import ActionHook, Pipeline, RunStats
from ..video import CameraSource

logger = logging.getLogger("carpi.modes.motion_capture")


def run_motion_capture_mode(
    config: AppConfig,
    *,
    preview: Optional[bool] = None,
    action_hook: Optional[ActionHook] = None,
) -> RunStats:
    mode_cfg = config.motion_capture

    car_detector = build_car_detector(config.car)
    inspector = build_inspector_classifier(config.inspector)

    source = CameraSource(
        device=mode_cfg.device,
        backend=mode_cfg.backend,
        width=mode_cfg.width,
        height=mode_cfg.height,
        fps=mode_cfg.fps,
    )

    logger.info(
        "motion_capture mode: camera=%s backend=%s -> %dx%d @ %.1f fps",
        mode_cfg.device, source.backend, source.size[0], source.size[1], source.fps,
    )
    logger.info(
        "backends: car=%s | inspector=%s",
        car_detector.name if car_detector is not None else "none",
        inspector.name if inspector is not None else "none",
    )
    if car_detector is None and inspector is None:
        logger.info("motion-only: AI stages disabled, recording motion clips only")
    elif car_detector is None:
        logger.warning(
            "inspector classifier is enabled but car detection is disabled - "
            "no cars will ever reach it"
        )
    elif inspector is None:
        logger.info("inspector classification disabled; saving car crops only")

    pipeline = Pipeline(
        config,
        car_detector,
        inspector,
        action_hook=action_hook,
        motion_crops=config.crops,
    )
    show_preview = mode_cfg.show_preview if preview is None else preview
    try:
        return pipeline.run(source, preview=show_preview)
    finally:
        source.close()
