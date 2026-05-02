"""Pytest configuration — ensure src/ is importable, plus the
``shell_runner`` fixture for testing scripts under ``scripts/``.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# Add project root to sys.path so tests can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class ShellRun:
    """Result of running a shell script under ``shell_runner``."""

    returncode: int
    stdout: str
    stderr: str
    calls: list[list[str]] = field(default_factory=list)


class ShellRunner:
    """Path-stubbing harness for shell-script tests.

    Each ``stub(name, ...)`` call writes a tiny bash script into a
    fresh ``$STUB_BIN`` directory that, when invoked, logs its own
    ``$0 $@`` (one call per line) into a call log file and exits with
    the configured return code. The bin directory is then prepended to
    ``$PATH`` for the subprocess that runs the target script — bash
    runs FOR REAL but every external command we declared is
    intercepted.

    The fixture is per-test (a fresh tmpdir each invocation), so state
    can't leak between tests. ``$PATH`` modifications happen only in
    the child process; the parent's environment is untouched.
    """

    def __init__(self, tmp_path: Path) -> None:
        self._stub_bin = tmp_path / "_stubs_bin"
        self._stub_bin.mkdir(parents=True, exist_ok=True)
        self._call_log = tmp_path / "_stubs_calls.log"
        self._call_log.write_text("")
        self._stubs: dict[str, tuple[int, str, str]] = {}

    def stub(self, name: str, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        """Register a stub for ``name``. Logs invocations, then exits."""
        self._stubs[name] = (returncode, stdout, stderr)
        path = self._stub_bin / name
        # The stub itself is bash. It prints the configured stdout/stderr,
        # then appends a single line of "name|arg1|arg2|..." to the call
        # log — pipe-delimited so spaces inside args survive a round-trip.
        # Self-locating call-log path baked into the stub via an env var
        # avoids hard-coding the tmp path into shell text.
        stdout_literal = shlex.quote(stdout)
        stderr_literal = shlex.quote(stderr)
        contents = (
            "#!/usr/bin/env bash\n"
            f"printf '%s' {stdout_literal}\n"
            f"printf '%s' {stderr_literal} >&2\n"
            '{ printf "%s" "$0"; for arg in "$@"; do printf "|%s" "$arg"; done; printf "\\n"; }'
            ' >> "${RADIO_TEST_CALL_LOG}"\n'
            f"exit {returncode}\n"
        )
        path.write_text(contents)
        path.chmod(0o755)

    def run(
        self,
        script: str | Path,
        *,
        env: dict[str, str] | None = None,
        args: list[str] | None = None,
        cwd: str | Path | None = None,
        timeout: float = 30.0,
    ) -> ShellRun:
        """Run ``script`` with stubs in ``$PATH``. Returns a ``ShellRun``."""
        merged_env = os.environ.copy()
        # Stub bin first so stubs win over any real binary on PATH.
        merged_env["PATH"] = f"{self._stub_bin}:{merged_env.get('PATH', '')}"
        merged_env["RADIO_TEST_CALL_LOG"] = str(self._call_log)
        if env:
            merged_env.update(env)

        cmd = ["bash", str(script)]
        if args:
            cmd.extend(args)

        proc = subprocess.run(
            cmd,
            env=merged_env,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        calls: list[list[str]] = []
        if self._call_log.exists():
            for line in self._call_log.read_text().splitlines():
                if line:
                    calls.append(line.split("|"))

        return ShellRun(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            calls=calls,
        )


@pytest.fixture
def shell_runner(tmp_path: Path) -> ShellRunner:
    """Per-test ``ShellRunner`` instance scoped to the test's tmp_path."""
    return ShellRunner(tmp_path)
