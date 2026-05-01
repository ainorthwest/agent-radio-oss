#!/usr/bin/env python3
"""Thin wrapper around `radio edit anomalies`.

Reads anomalies.json after running and prints just the count, so the
agent's harness can decide whether to loop or ship. The full report
lands at <episode_dir>/anomalies.json regardless.

Usage:
    python check-anomalies.py <manifest.json>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check-anomalies.py <manifest.json>", file=sys.stderr)
        return 2
    manifest_path = Path(sys.argv[1])
    rc = subprocess.run(
        ["uv", "run", "radio", "edit", "anomalies", str(manifest_path)],
        check=False,
    ).returncode
    if rc != 0:
        return rc
    anomalies_path = manifest_path.parent / "anomalies.json"
    if not anomalies_path.exists():
        return 0
    data = json.loads(anomalies_path.read_text())
    print(f"anomaly_count={len(data.get('anomalies', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
