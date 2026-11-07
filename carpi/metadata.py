"""Provenance for clips and crops: when and where each one was recorded.

Two facts are recorded for every file, and both are placeholders for hardware
the rig does not have yet:

* **Time** comes from a :class:`ClockProvider`. With no trusted clock module
  yet the default is a *placeholder* - the machine clock - which on a Pi with no
  RTC is only right once NTP (or a GPS fix) has set it. The clock module on the
  rig supplies this instead once it has a battery and can be believed.
* **Location** comes from a :class:`LocationProvider`. With no GPS module wired
  up the default is a *placeholder* fix - one configured point (0, 0 by
  default) - so the records are always populated and the output shape is
  complete; ``none`` reports no fix, and a real GPS becomes one more backend
  later.

Neither belongs in the filename. Coordinates are long and lose precision once
rounded, and a name that changes length as the rig moves is awkward to
reference; so the filename carries a compact, sortable timestamp and the exact
time and coordinates go in the sidecar index - one JSON object per line, one
line per clip and per crop.

Wiring up a real GPS or clock later means writing one more provider and naming
it in ``metadata.location.backend`` / ``metadata.clock.backend``; nothing else
changes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

from .config import ClockConfig, ConfigError, LocationConfig

logger = logging.getLogger("carpi.metadata")


@dataclass(frozen=True)
class Fix:
    """A geographical position, in decimal degrees (WGS84)."""

    latitude: float
    longitude: float


class LocationProvider(Protocol):
    """Where the rig is, right now. ``fix()`` returns ``None`` with no fix."""

    name: str

    def fix(self) -> Optional[Fix]:
        ...


class NoFixProvider:
    """Reports no fix at all - the record's coordinates come out ``null``."""

    name = "none"

    def fix(self) -> Optional[Fix]:
        return None


class PlaceholderProvider:
    """A stand-in fix for when there is no GPS module yet.

    Reports one configured point (0, 0 by default) so every record carries a
    latitude and longitude and the shape of the output is complete. The
    ``location_source`` field marks these as placeholders, not measurements.
    """

    name = "placeholder"

    def __init__(self, latitude: float = 0.0, longitude: float = 0.0) -> None:
        self._fix = Fix(float(latitude), float(longitude))

    def fix(self) -> Optional[Fix]:
        return self._fix


def build_location_provider(config: LocationConfig) -> LocationProvider:
    """Choose a location source from ``metadata.location`` in the config."""
    if config.backend == "placeholder":
        try:
            return PlaceholderProvider(
                config.options.get("latitude", 0.0),
                config.options.get("longitude", 0.0),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "metadata.location: 'latitude' and 'longitude' must be numbers"
            ) from exc
    return NoFixProvider()


class ClockProvider(Protocol):
    """What time it is. ``name`` marks the source in the manifest."""

    name: str

    def now(self) -> datetime:
        ...


class PlaceholderClock:
    """The machine clock, standing in for the rig's clock module.

    Asked once when a motion event begins, so the clip and every crop from that
    event share one timestamp. On the dev machine, and on a Pi that has synced
    (NTP, or a GPS fix), this is the real time; on a Pi with no RTC and no
    network it is only right once something has set it - which is why the
    records mark it ``time_source: "placeholder"`` rather than passing it off as
    the clock module's own reading.
    """

    name = "placeholder"

    def now(self) -> datetime:
        return datetime.now().astimezone()


def build_clock_provider(config: ClockConfig) -> ClockProvider:
    """Choose a time source from ``metadata.clock`` in the config.

    Only the machine clock exists so far; the clock module becomes one more
    branch here once it has a battery and can be trusted.
    """
    return PlaceholderClock()


class ManifestWriter:
    """Appends one JSON object per file to a sidecar index.

    Written line by line and flushed per entry, so a run that is killed
    mid-event still leaves a readable record of everything written so far.
    """

    def __init__(self, path: Path, location: LocationProvider, clock: ClockProvider) -> None:
        self.path = path
        self.location = location
        self.clock = clock

    def record(
        self,
        *,
        kind: str,
        file: Path,
        timestamp: datetime,
        event: Optional[int] = None,
        frame: Optional[int] = None,
    ) -> None:
        fix = self.location.fix()
        row = {
            "file": file.name,
            "kind": kind,
            "timestamp": timestamp.isoformat(),
            "time_source": self.clock.name,
            "latitude": fix.latitude if fix else None,
            "longitude": fix.longitude if fix else None,
            "location_source": self.location.name,
        }
        if event is not None:
            row["event"] = event
        if frame is not None:
            row["frame"] = frame
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
