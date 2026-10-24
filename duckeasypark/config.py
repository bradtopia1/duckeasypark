"""Typed configuration loading and validation.

The whole application is driven by a single YAML file (see ``config.yaml``).
This module turns that loosely-typed document into frozen dataclasses so the
rest of the code can rely on validated values with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


class ConfigError(ValueError):
    """Raised when the config file is missing, malformed, or inconsistent."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_float(value: Any, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{ctx}: expected a number, got {value!r}")
    return float(value)


def _as_int(value: Any, ctx: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{ctx}: expected an integer, got {value!r}")
    return int(value)


def _as_bool(value: Any, ctx: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{ctx}: expected true/false, got {value!r}")
    return value


def _opt_width(value: Any, ctx: str) -> Optional[int]:
    """Parse an optional target width; ``None`` means 'keep the source size'."""
    if value is None:
        return None
    return max(1, _as_int(value, ctx))


def _section(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping, got {type(value).__name__}")
    return value


def _resolve(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (base_dir / p)


# --------------------------------------------------------------------------- #
# motion detection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MotionArea:
    """Region of interest expressed as fractions of the frame (0.0 .. 1.0)."""

    vertical: Tuple[float, float]
    horizontal: Tuple[float, float]

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Return (x0, y0, x1, y1) clamped to the frame."""
        x0 = int(round(self.horizontal[0] * width))
        x1 = int(round(self.horizontal[1] * width))
        y0 = int(round(self.vertical[0] * height))
        y1 = int(round(self.vertical[1] * height))
        x0 = _clamp_int(x0, 0, width - 1)
        y0 = _clamp_int(y0, 0, height - 1)
        x1 = _clamp_int(x1, x0 + 1, width)
        y1 = _clamp_int(y1, y0 + 1, height)
        return x0, y0, x1, y1


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _parse_span(value: Any, ctx: str) -> Tuple[float, float]:
    """Parse an axis definition into (start_fraction, end_fraction).

    Accepts either a single percentage (interpreted as a *centred* band of that
    size) or an explicit ``[start, end]`` pair of percentages.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        span = _clamp(_as_float(value, ctx), 0.0, 100.0)
        start = (100.0 - span) / 2.0
        return start / 100.0, (start + span) / 100.0
    if isinstance(value, (list, tuple)) and len(value) == 2:
        a = _clamp(_as_float(value[0], ctx), 0.0, 100.0)
        b = _clamp(_as_float(value[1], ctx), 0.0, 100.0)
        if a > b:
            a, b = b, a
        return a / 100.0, b / 100.0
    raise ConfigError(
        f"{ctx}: expected a percentage (e.g. 60) or [start, end] percentages "
        f"(e.g. [10, 90]), got {value!r}"
    )


@dataclass(frozen=True)
class MotionConfig:
    sensitivity: float
    min_area_percent: float
    trigger_frames: int
    area: MotionArea
    pre_seconds: float
    post_seconds: float
    history: int = 500

    @property
    def var_threshold(self) -> float:
        """MOG2 variance threshold. Higher sensitivity -> lower threshold."""
        return 4.0 + (1.0 - self.sensitivity) * 32.0

    @property
    def min_area_scale(self) -> float:
        """Higher sensitivity shrinks the minimum blob size."""
        return 1.5 - 0.5 * self.sensitivity


# --------------------------------------------------------------------------- #
# AI backends
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BackendConfig:
    backend: str
    options: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TestModeConfig:
    video_path: Path
    loop: bool = False
    realtime: bool = False
    show_preview: bool = False
    max_seconds: Optional[float] = None


@dataclass(frozen=True)
class ProcessingConfig:
    """Per-stage resolution, for speed/memory.

    Each stage runs at its own size because they want different things: motion
    detection only needs a coarse trigger, the car detector wants more detail,
    and clips are worth keeping at full resolution as training data. ``None``
    means "keep the source resolution" for that stage.
    """

    motion_width: Optional[int] = None
    detect_width: Optional[int] = None
    clip_width: Optional[int] = None
    full_res_crops: bool = True


# --------------------------------------------------------------------------- #
# top-level config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppConfig:
    mode: str
    motion: MotionConfig
    car: BackendConfig
    inspector: BackendConfig
    output_dir: Path
    config_path: Path
    test: TestModeConfig
    processing: ProcessingConfig
    raw: Dict[str, Any] = field(default_factory=dict)

    def with_mode(self, mode: str) -> "AppConfig":
        return replace(self, mode=mode)


def _parse_backend(data: Dict[str, Any], ctx: str, default_backend: str) -> BackendConfig:
    backend = data.get("backend", default_backend)
    if not isinstance(backend, str) or not backend.strip():
        raise ConfigError(f"{ctx}.backend: expected a non-empty string")
    options = data.get("options", {}) or {}
    if not isinstance(options, dict):
        raise ConfigError(f"{ctx}.options: expected a mapping")
    return BackendConfig(backend=backend.strip().lower(), options=dict(options))


def load_config(path: str | Path) -> AppConfig:
    """Load, validate, and return the application configuration."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - passthrough
        raise ConfigError(f"could not parse {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("top level of the config file must be a mapping")

    base_dir = config_path.parent

    mode = raw.get("mode", "test")
    if not isinstance(mode, str) or not mode.strip():
        raise ConfigError("'mode' must be a non-empty string")
    mode = mode.strip()

    # -- motion ------------------------------------------------------------- #
    m = _section(raw, "motion")
    sensitivity = _clamp(_as_float(m.get("sensitivity", 0.5), "motion.sensitivity"), 0.0, 1.0)
    min_area_percent = max(0.0, _as_float(m.get("min_area_percent", 0.4), "motion.min_area_percent"))
    trigger_frames = max(1, _as_int(m.get("trigger_frames", 2), "motion.trigger_frames"))
    history = max(1, _as_int(m.get("history", 500), "motion.history"))

    area = _section(m, "area")
    motion_area = MotionArea(
        vertical=_parse_span(area.get("vertical", [0, 100]), "motion.area.vertical"),
        horizontal=_parse_span(area.get("horizontal", [0, 100]), "motion.area.horizontal"),
    )

    clip = _section(m, "clip")
    pre_seconds = max(0.0, _as_float(clip.get("pre_seconds", 1.0), "motion.clip.pre_seconds"))
    post_seconds = max(0.0, _as_float(clip.get("post_seconds", 1.5), "motion.clip.post_seconds"))

    motion = MotionConfig(
        sensitivity=sensitivity,
        min_area_percent=min_area_percent,
        trigger_frames=trigger_frames,
        area=motion_area,
        pre_seconds=pre_seconds,
        post_seconds=post_seconds,
        history=history,
    )

    # -- AI backends -------------------------------------------------------- #
    detectors = _section(raw, "detectors")
    car = _parse_backend(_section(detectors, "car"), "detectors.car", "mock")
    inspector = _parse_backend(_section(detectors, "inspector"), "detectors.inspector", "mock")
    # Resolve a model file relative to the config file, like other paths.
    if inspector.backend == "tflite" and inspector.options.get("model_path"):
        inspector = replace(
            inspector,
            options={
                **inspector.options,
                "model_path": str(_resolve(str(inspector.options["model_path"]), base_dir)),
            },
        )

    # -- output ------------------------------------------------------------- #
    output_dir = _resolve(str(raw.get("output_dir", "output")), base_dir)

    # -- processing --------------------------------------------------------- #
    proc = _section(raw, "processing")
    # ``width`` is the pre-split single knob. Keep honouring it as the default
    # for the motion and detection stages so existing configs still work.
    legacy_width = proc.get("width", None)
    processing = ProcessingConfig(
        motion_width=_opt_width(proc.get("motion_width", legacy_width), "processing.motion_width"),
        detect_width=_opt_width(proc.get("detect_width", legacy_width), "processing.detect_width"),
        clip_width=_opt_width(proc.get("clip_width", None), "processing.clip_width"),
        full_res_crops=_as_bool(proc.get("full_res_crops", True), "processing.full_res_crops"),
    )

    # -- modes -------------------------------------------------------------- #
    modes = _section(raw, "modes")
    t = _section(modes, "test")
    video_path = _resolve(str(t.get("video_path", "samples/test.mp4")), base_dir)
    max_seconds_raw = t.get("max_seconds", None)
    max_seconds = None if max_seconds_raw is None else max(0.0, _as_float(max_seconds_raw, "modes.test.max_seconds"))

    test = TestModeConfig(
        video_path=video_path,
        loop=_as_bool(t.get("loop", False), "modes.test.loop"),
        realtime=_as_bool(t.get("realtime", False), "modes.test.realtime"),
        show_preview=_as_bool(t.get("show_preview", False), "modes.test.show_preview"),
        max_seconds=max_seconds,
    )

    return AppConfig(
        mode=mode,
        motion=motion,
        car=car,
        inspector=inspector,
        output_dir=output_dir,
        config_path=config_path,
        test=test,
        processing=processing,
        raw=raw,
    )
