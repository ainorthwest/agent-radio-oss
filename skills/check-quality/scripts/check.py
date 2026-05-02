#!/usr/bin/env python3
"""Thin wrapper around the standalone quality module.

The OSS CLI does not yet expose `radio quality` as a top-level command
in v0.1.0; quality runs as a stage inside the pipeline. For agent-driven
ad-hoc scoring (a single mp3, a single segment, a manifest) the
standalone module entrypoint is the canonical surface.

Usage:
    python check.py <audio> [--manifest manifest.json] [--script script.json]
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: check.py <audio> [--manifest M] [--script S] [--reference R]",
            file=sys.stderr,
        )
        return 2
    cmd = ["uv", "run", "python", "-m", "src.quality", *sys.argv[1:]]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
