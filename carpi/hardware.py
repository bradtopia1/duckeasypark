"""Which board carpi is running on, so the camera can be sized to match.

The rig moves between Raspberry Pi models with very different decoding power,
and a frame costs roughly with its pixel count: a Pi 4 or 5 can take the
camera's full-resolution 30 fps mode, while a Pi 3B can only manage 1080p -
asking it for more just drops frames. Rather than editing the config whenever
the code lands on another board, the capture size defaults to a per-board value
(``resolution: auto``) and can still be pinned explicitly.

Detection is the kernel's own device-tree model string, a plain file under
``/proc`` - the same dependency-free approach as ``thermal.py``. A machine that
is not a Raspberry Pi (a dev box) has no such file, and carpi falls back to the
full size.
"""

from __future__ import annotations

from typing import Optional, Tuple

# Where the kernel names the board. Absent on anything that is not a
# device-tree machine, which is how a dev box is recognised.
_MODEL_PATH = "/proc/device-tree/model"

# The camera's full-resolution 30 fps mode (MJPEG) - what the rig asks for on a
# capable board. This is the size the config asked for before it was automatic.
FULL_CAPTURE_SIZE = (2560, 1440)

# The step down for the weaker boards (Pi 3B and older): 1080p is about as much
# as they can decode at 30 fps before dropping frames.
LIGHT_CAPTURE_SIZE = (1920, 1080)

# Pi generations at or above this one are given the full capture size.
_FULL_SIZE_GENERATION = 4


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


def default_capture_size(model: Optional[str] = None) -> Tuple[int, int]:
    """The capture size to ask for on this board, as ``(width, height)``.

    Full resolution on a Pi 4/5 - and on anything that is not a Raspberry Pi,
    such as the dev machine; 1080p on a Pi 3B and older boards, which cannot
    decode the full size at 30 fps.
    """
    generation = pi_generation(model)
    if generation is not None and generation < _FULL_SIZE_GENERATION:
        return LIGHT_CAPTURE_SIZE
    return FULL_CAPTURE_SIZE


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
