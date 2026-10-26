"""Test mode: run the pipeline against a video file instead of a live camera.

There is no camera and no real feed here - the frames come from
``modes.test.video_path`` in the config. Everything downstream (motion
detection, car detection, cropping, inspector classification) behaves exactly as
it will in a live mode, which makes this the place to tune sensitivity, the
region of interest, and the AI backends.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..classifiers import build_inspector_classifier
from ..config import AppConfig, ConfigError
from ..detectors import build_car_detector
from ..pipeline import ActionHook, Pipeline, RunStats
from ..video import FileSource

logger = logging.getLogger("carpi.modes.test")


def run_test_mode(
    config: AppConfig,
    *,
    preview: Optional[bool] = None,
    action_hook: Optional[ActionHook] = None,
) -> RunStats:
    mode_cfg = config.test

    if not mode_cfg.video_path.exists():
        raise ConfigError(
            f"test mode video not found: {mode_cfg.video_path}\n"
            f"hint: set modes.test.video_path in the config to your own video, "
            f"or generate a synthetic one for testing with "
            f"`python scripts/make_sample_video.py`"
        )

    source = FileSource(
        mode_cfg.video_path,
        loop=mode_cfg.loop,
        realtime=mode_cfg.realtime,
        max_seconds=mode_cfg.max_seconds,
    )

    car_detector = build_car_detector(config.car)
    inspector = build_inspector_classifier(config.inspector)

    logger.info(
        "test mode: video=%s loop=%s realtime=%s max_seconds=%s",
        mode_cfg.video_path, mode_cfg.loop, mode_cfg.realtime, mode_cfg.max_seconds,
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
    return pipeline.run(source, preview=show_preview)
