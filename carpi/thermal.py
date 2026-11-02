"""Read the CPU/SoC temperature from the kernel's sensor interfaces.

The rig runs headless, so a temperature reading on the periodic status line is
one more thing visible in the log without running a separate monitoring daemon.
Linux exposes this as plain files under ``/sys``: ``/sys/class/thermal`` on a
Raspberry Pi and ``/sys/class/hwmon`` on x86, so this needs no dependency.

Returns ``None`` when no CPU sensor is found (a container, or a machine whose
only sensors are a GPU, NVMe or WiFi chip), so callers can simply omit it.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

# thermal_zone types that are the CPU/SoC, not a GPU, disk or WiFi chip.
_THERMAL_TYPES = ("cpu", "soc", "x86_pkg_temp")
# hwmon drivers whose temp1_input is the CPU package/SoC temperature.
_HWMON_NAMES = ("k10temp", "coretemp", "zenpower", "cpu_thermal")

_THERMAL_ZONES = "/sys/class/thermal/thermal_zone*"
_HWMON_DEVICES = "/sys/class/hwmon/hwmon*"


def _read_file(path: str) -> Optional[str]:
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _read_celsius(path: str) -> Optional[float]:
    """Read a millidegree sensor file as degrees Celsius, or ``None``."""
    raw = _read_file(path)
    if raw is None:
        return None
    try:
        celsius = float(raw) / 1000.0
    except ValueError:
        return None
    # Guard against a virtual sensor reporting 0 or a nonsense value; real
    # silicon never sits outside this range.
    return celsius if -50.0 <= celsius <= 150.0 else None


def cpu_temperature_c() -> Optional[float]:
    """Current CPU/SoC temperature in degrees Celsius, or ``None``."""
    for zone in sorted(glob.glob(_THERMAL_ZONES)):
        kind = (_read_file(os.path.join(zone, "type")) or "").lower()
        if any(kind.startswith(name) for name in _THERMAL_TYPES):
            celsius = _read_celsius(os.path.join(zone, "temp"))
            if celsius is not None:
                return celsius

    for device in sorted(glob.glob(_HWMON_DEVICES)):
        name = (_read_file(os.path.join(device, "name")) or "").lower()
        if name in _HWMON_NAMES:
            celsius = _read_celsius(os.path.join(device, "temp1_input"))
            if celsius is not None:
                return celsius

    return None
