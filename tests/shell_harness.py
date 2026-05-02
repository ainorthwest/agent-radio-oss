"""Shell-script test harness — path-stubbing for `scripts/*.sh`.

Used by ``tests/conftest.py`` to expose a ``shell_runner`` fixture, and
imported directly by Tier-2 tests that want to type-annotate their
``shell_runner`` parameter.

The stubbing trick: each ``ShellRunner.stub(name, ...)`` writes a tiny
bash script into ``$STUB_BIN`` that, when invoked, prints the configured
stdout/stderr and appends a NUL-separated record (one record per line)
to a call log. The stub bin is then prepended to ``$PATH`` for the
subprocess that runs the target script — bash runs FOR REAL but every
external command we declared is intercepted.

The NUL separator is important: arguments may contain pipes, slashes,
URLs, paths, even spaces — none of those are NUL-safe in normal shell
output, but NUL is. We write each record on its own line (newlines
inside arguments would still corrupt this; bash arguments may not
contain literal newlines via the standard CLI surface so this is
acceptable in practice).
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Record separator inside a single call log line.
ARG_SEP = "\x00"


@dataclass
class ShellRun:
    """Result of running a shell script under :class:`ShellRunner`."""

    returncode: int
    stdout: str
    stderr: str
    calls: list[list[str]] = field(default_factory=list)


class ShellRunner:
    """Path-stubbing harness for shell-script tests.

    Per-test scoped — instantiated by the ``shell_runner`` fixture with
    a fresh ``tmp_path``. Stubs go into ``tmp_path/_stubs_bin``; calls
    are logged to ``tmp_path/_stubs_calls.log``. The parent process's
    environment is never mutated — ``$PATH`` is overridden only inside
    the subprocess invocation via ``env=``.
    """

    def __init__(self, tmp_path: Path) -> None:
        self._stub_bin = tmp_path / "_stubs_bin"
        self._stub_bin.mkdir(parents=True, exist_ok=True)
        self._call_log = tmp_path / "_stubs_calls.log"
        self._call_log.write_text("")
        self._stubs: dict[str, tuple[int, str, str]] = {}

    def stub(
        self,
        name: str,
        *,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        """Register a stub for ``name``. Logs invocations, then exits."""
        self._stubs[name] = (returncode, stdout, stderr)
        path = self._stub_bin / name
        stdout_literal = shlex.quote(stdout)
        stderr_literal = shlex.quote(stderr)
        # The stub uses ${RADIO_TEST_CALL_LOG:?...} so it fails loudly
        # if the harness ever forgets to set the env var (e.g. someone
        # invokes the stub from a hand-rolled subprocess outside of
        # ShellRunner.run). The append loop emits ARG_SEP-delimited
        # fields; each call is one newline-terminated line.
        contents = (
            "#!/usr/bin/env bash\n"
            'log="${RADIO_TEST_CALL_LOG:?RADIO_TEST_CALL_LOG not set; '
            'stub invoked outside ShellRunner.run}"\n'
            f"printf '%s' {stdout_literal}\n"
            f"printf '%s' {stderr_literal} >&2\n"
            '{ printf "%s" "$0";'
            ' for arg in "$@"; do printf "\\x00%s" "$arg"; done;'
            ' printf "\\n"; }'
            ' >> "$log"\n'
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
        """Run ``script`` with stubs in ``$PATH``. Returns a :class:`ShellRun`."""
        merged_env = os.environ.copy()
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
                    calls.append(line.split(ARG_SEP))

        return ShellRun(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            calls=calls,
        )
