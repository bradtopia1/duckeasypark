#!/usr/bin/env python3
"""carpi - entry point.

Reads a YAML config, picks a mode, and runs the motion -> car -> inspector
pipeline. See README.md and config.yaml for details.
"""

from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, Sequence

from carpi.config import ConfigError, load_config
from carpi.modes import get_mode_runner

# Log file, written next to wherever the program is run from (on the Pi that is
# the project folder). Rotated so a 24/7 rig cannot fill the disk.
LOG_FILE = "carpi.log"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carpi",
        description="Detect cars (and parking inspectors) from a camera or video file.",
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml",
        help="path to the YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "-m", "--mode", default=None,
        help="override the mode set in the config file (e.g. 'test')",
    )
    parser.add_argument(
        "--preview", dest="preview", action="store_true", default=None,
        help="force the debug preview window on (needs a GUI build of OpenCV)",
    )
    parser.add_argument(
        "--no-preview", dest="preview", action="store_false",
        help="force the debug preview window off",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="-v for info logs, -vv for debug logs",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    level = max(logging.DEBUG, logging.WARNING - 10 * args.verbose)
    # Log to the terminal/journal as usual, and to a rotating file so there is a
    # lasting record on the rig.
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8",
            ),
        ],
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.mode:
        config = config.with_mode(args.mode)

    try:
        runner = get_mode_runner(config.mode)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        stats = runner(config, preview=args.preview)
    except (ConfigError, FileNotFoundError, NotImplementedError, IOError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    print(f"\ndone: {stats.summary()}")
    print(f"output: {config.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
