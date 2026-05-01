#!/usr/bin/env python3
"""Thin wrapper around `radio edit script`.

Forwards every arg to the CLI. Exists so the agent has a stable
script-name entry point that's harness-portable; nothing here is logic.

Usage:
    python edit-segment.py <script.json> --delete 2
    python edit-segment.py <script.json> --replace 0 --text "Fixed!"
    python edit-segment.py <script.json> --reorder "2,0,1"
    python edit-segment.py <script.json> --change-voice 0 --speaker host_b
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = ["uv", "run", "radio", "edit", "script", *sys.argv[1:]]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
