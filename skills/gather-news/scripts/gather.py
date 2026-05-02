#!/usr/bin/env python3
"""Thin wrapper invoked by the gather-news skill.

In v0.1.0 there is no dedicated `radio gather` command — the curator
stage of the pipeline covers the wire-desk fetch. This wrapper invokes
`radio run pipeline --no-distribute --no-music` so an autonomous
station can produce raw items + script artifacts without rendering.

Usage:
    python gather.py [extra args passed to `radio run`]
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [
        "uv",
        "run",
        "radio",
        "run",
        "--no-distribute",
        "--no-music",
        *sys.argv[1:],
    ]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
