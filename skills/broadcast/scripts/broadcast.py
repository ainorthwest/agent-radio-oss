#!/usr/bin/env python3
"""Thin orchestrator for the broadcast skill.

Invokes the three distribution branches sequentially, treating each as
best-effort. Returns 0 if at least one branch succeeded, non-zero if
all branches failed (which usually means none was configured).

Each branch is feature-flagged in the underlying `radio` CLI by
checking env vars / config; this wrapper does not pre-validate
credentials. Read the per-branch stderr output to see warnings.

Usage:
    python broadcast.py <episode_date> [--program <slug>]
"""

from __future__ import annotations

import subprocess
import sys


def _run(label: str, cmd: list[str]) -> bool:
    print(f"\n=== broadcast: {label} ===", file=sys.stderr)
    rc = subprocess.run(cmd, check=False).returncode
    if rc == 0:
        print(f"=== broadcast: {label} OK ===", file=sys.stderr)
        return True
    print(f"=== broadcast: {label} FAILED (rc={rc}) ===", file=sys.stderr)
    return False


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: broadcast.py <episode_date> [--program <slug>]",
            file=sys.stderr,
        )
        return 2

    base_args = sys.argv[1:]

    distribute_cmd = ["uv", "run", "radio", "distribute", "episode", *base_args]
    feed_cmd = ["uv", "run", "radio", "distribute", "feed", *base_args]
    stream_cmd = ["uv", "run", "radio", "stream", "update", *base_args]

    successes = 0
    successes += int(_run("R2 + Discourse (radio distribute episode)", distribute_cmd))
    successes += int(_run("podcast RSS (radio distribute feed)", feed_cmd))
    successes += int(_run("AzuraCast rolling update (radio stream update)", stream_cmd))

    if successes == 0:
        print(
            "\n[broadcast] all branches failed. Check env vars (AGENT_RADIO_R2_*,"
            " AGENT_RADIO_DISCOURSE_API_KEY, AGENT_RADIO_AZURACAST_API_KEY) and"
            " endpoint config in config/radio.yaml.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
