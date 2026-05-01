#!/usr/bin/env python3
"""Thin wrapper around `radio publish episode`.

Usage:
    python publish.py <episode_dir>
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: publish.py <episode_dir>", file=sys.stderr)
        return 2
    cmd = ["uv", "run", "radio", "publish", "episode", *sys.argv[1:]]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
