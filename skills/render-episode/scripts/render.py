#!/usr/bin/env python3
"""Thin wrapper around `radio render episode`.

Usage:
    python render.py <script.json> [extra args]
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: render.py <script.json> [extra args]", file=sys.stderr)
        return 2
    cmd = ["uv", "run", "radio", "render", "episode", *sys.argv[1:]]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
