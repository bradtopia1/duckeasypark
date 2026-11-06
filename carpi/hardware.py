"""Which board carpi is running on, so the camera can be sized to match.

The rig moves between Raspberry Pi models with very different decoding power,
and a frame costs roughly with its pixel count: a Pi 4 or 5 can take the
camera's full-resolution 30 fps mode, while a Pi 3B cannot, so asking it for the
same drops frames. Rather than editing the config whenever the code lands on
another board, the capture size and rate both default to a per-board profile
(``resolution: auto`` and ``fps: auto``) and can still be pinned explicitly.

The trade the weak profile makes is deliberate: it keeps 720p because a crop can
only be as sharp as the captured frame (the training stills are cropped from
these frames), and gives up frame rate instead, which costs the dataset very
little - a car does not change appearance between samples.

Detection is the kernel's own device-tree model string, a plain file under
``/proc`` - the same dependency-free approach as ``thermal.py``. A machine that
is not a Raspberry Pi (a dev box) has no such file, and carpi falls back to the
full profile.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Where the kernel names the board. Absent on anything that is not a
# device-tree machine, which is how a dev box is recognised.
_MODEL_PATH = "/proc/device-tree/model"

# What a capable board (Pi 4/5, or a machine that is not a Pi at all) asks for:
# the camera's full-resolution 30 fps MJPEG mode.
FULL_CAPTURE_SIZE = (2560, 1440)
FULL_CAPTURE_FPS = 30.0

# The weaker boards (Pi 3B and older) cannot decode the full size at 30 fps, so
# they capture 720p at 10 fps. 720p keeps a car crop near the classifier's input
# size without upscaling and 10 fps is still plenty of distinct poses.
LIGHT_CAPTURE_SIZE = (1280, 720)
LIGHT_CAPTURE_FPS = 10.0

# The width motion detection runs at. It is resized from every frame, so on a
# slow board it is the biggest per-frame knob: a weak board runs it smaller.
FULL_MOTION_WIDTH = 640
LIGHT_MOTION_WIDTH = 320

# Pi generations at or above this one get the full profile.
_FULL_PROFILE_GENERATION = 4


def board_model() -> Optional[str]:
    """The device-tree model string, e.g. ``'Raspberry Pi 4 Model B Rev 1.5'``.

    ``None`` when the machine has no device tree at all - on a desktop that is
    the first sign this is not a Pi.
    """
    try:
        with open(_MODEL_PATH, encoding="ascii", errors="ignore") as handle:
            model = handle.read().strip("\x00 \t\r\n")
    except OSError:
        return None
    return model or None


def pi_generation(model: Optional[str] = None) -> Optional[int]:
    """The Raspberry Pi generation (5, 4, 3, 2, 1), or ``None`` if not a Pi.

    Handles the naming the device tree actually uses, including the boards that
    do not put a bare number first: ``Raspberry Pi 400`` (a Pi 4), ``Pi 500``
    (a Pi 5), ``Pi Zero 2 W``, ``Pi Compute Module 4`` and the original
    ``Raspberry Pi Model B``.
    """
    if model is None:
        model = board_model()
    if not model or "Raspberry Pi" not in model:
        return None

    rest = model.split("Raspberry Pi", 1)[1].strip()
    tokens = rest.split()
    if not tokens:
        return None

    first = tokens[0]
    if first == "Compute" and len(tokens) > 2:   # 'Compute Module 4 ...'
        first = tokens[2]
    if first.startswith("Zero"):                 # 'Zero W' / 'Zero 2 W'
        return 2 if rest.startswith("Zero 2") else 1
    if first == "Model":                         # 'Raspberry Pi Model B ...'
        return 1
    if first.isdigit():
        number = int(first)
        return number // 100 if number >= 100 else number   # 400 -> 4, 500 -> 5
    return None


def _needs_light_profile(model: Optional[str]) -> bool:
    """True for a Pi 3B and older boards, which cannot take the full profile."""
    generation = pi_generation(model)
    return generation is not None and generation < _FULL_PROFILE_GENERATION


def default_capture_size(model: Optional[str] = None) -> Tuple[int, int]:
    """The capture size to ask for on this board, as ``(width, height)``.

    Full resolution on a Pi 4/5 - and on anything that is not a Raspberry Pi,
    such as the dev machine; 720p on a Pi 3B and older boards.
    """
    return LIGHT_CAPTURE_SIZE if _needs_light_profile(model) else FULL_CAPTURE_SIZE


def default_capture_fps(model: Optional[str] = None) -> float:
    """The capture frame rate to ask for on this board.

    30 fps on a capable board; 10 fps on a Pi 3B and older, which is enough for
    the training stills and is what keeps the pipeline at real time.
    """
    return LIGHT_CAPTURE_FPS if _needs_light_profile(model) else FULL_CAPTURE_FPS


def default_motion_width(model: Optional[str] = None) -> int:
    """The width motion detection runs at on this board.

    Motion detection runs on every frame, so on a weak board a smaller width is
    the cheapest way to buy back real time.
    """
    return LIGHT_MOTION_WIDTH if _needs_light_profile(model) else FULL_MOTION_WIDTH


def resolve_capture_size(spec: str, model: Optional[str] = None) -> Tuple[int, int]:
    """Turn a config ``resolution`` value into ``(width, height)``.

    ``"auto"`` is the per-board default; anything else is an explicit
    ``"WIDTHxHEIGHT"`` already validated by the config loader.
    """
    text = str(spec).strip().lower()
    if text == "auto":
        return default_capture_size(model)
    width, height = text.split("x", 1)
    return int(width), int(height)
