#!/usr/bin/env python3
"""Sequential orchestrator for the run-station meta-skill.

Invokes each skill's wrapper in order, with verdict-based branching at
the quality stage. Designed to be called from cron / systemd / an
agent harness loop. Exits non-zero on terminal failure; surfaces
intermediate decisions as JSON lines on stderr.

This is a *reference* implementation. Operators with richer agent
harnesses should re-implement the orchestration in the harness's
preferred form (Hermes routine, Claude Code subagent, Gaia bundle).
The OSS contract is the *shape* of the loop — see
``skills/run-station/SKILL.md``.

Usage:
    python loop.py [--program <slug>] [--no-broadcast]
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime


def _emit(event: str, **fields: object) -> None:
    """Emit a single-line JSON event to stderr for harness consumption."""
    payload = {"ts": datetime.now(UTC).isoformat(), "event": event, **fields}
    print(json.dumps(payload), file=sys.stderr)


def _run(label: str, cmd: list[str]) -> int:
    _emit("phase.start", phase=label, cmd=" ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode
    _emit("phase.end", phase=label, rc=rc)
    return rc


def main() -> int:
    args = sys.argv[1:]
    no_broadcast = "--no-broadcast" in args
    if no_broadcast:
        args.remove("--no-broadcast")

    today = datetime.now(UTC).strftime("%Y-%m-%d")

    # Phase 1+2+3: gather + script + render + quality, all wired into
    # the pipeline. This is one CLI invocation that spans gather →
    # check-quality. The pipeline gates ship/review/reject internally.
    pipeline_args = ["--no-distribute"]
    if "--program" in args:
        idx = args.index("--program")
        pipeline_args.extend(["--program", args[idx + 1]])
    rc = _run("pipeline (gather→render→quality)", ["uv", "run", "radio", "run", *pipeline_args])
    if rc != 0:
        _emit("station.halt", reason="pipeline_failed", rc=rc)
        return rc

    # Phase 4: publish-episode (deterministic derivative content)
    # The pipeline already runs the publisher non-blockingly, but we
    # invoke it explicitly here so an agent harness can observe the
    # phase boundary and re-invoke independently if the artifacts go
    # stale (e.g., after a manual `edit-script`).
    program_slug = None
    if "--program" in args:
        idx = args.index("--program")
        program_slug = args[idx + 1]

    if program_slug:
        episode_dir = f"library/programs/{program_slug}/episodes/{today}"
    else:
        episode_dir = f"output/episodes/{today}"

    rc = _run(
        "publish",
        ["uv", "run", "radio", "publish", "episode", episode_dir],
    )
    if rc != 0:
        _emit("station.halt", reason="publish_failed", rc=rc)
        return rc

    # Phase 5: broadcast (or skip)
    if no_broadcast:
        _emit("station.skip_broadcast", reason="--no-broadcast")
        _emit("station.complete", broadcast=False)
        return 0

    rc = _run(
        "broadcast",
        [
            "python",
            __file__.replace("/run-station/scripts/loop.py", "/broadcast/scripts/broadcast.py"),
            today,
            *(["--program", program_slug] if program_slug else []),
        ],
    )
    if rc != 0:
        # Broadcast partial-failure is non-terminal — the local
        # artifacts are still valid. Surface the failure and return
        # non-zero so the operator's harness knows to investigate.
        _emit("station.partial", reason="broadcast_failed", rc=rc)
        return rc

    _emit("station.complete", broadcast=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
