"""Tests for `scripts/*.sh` — the Day 5 shell-script foundation.

This file establishes the three-tier test pattern for shell scripts in the
OSS repo:

- **Tier 1 (static):** every script under ``scripts/`` has a bash shebang,
  ``set -euo pipefail`` near the top, and passes shellcheck. The discovery
  test auto-enrolls every new script — no allowlist.
- **Tier 2 (mock):** the ``shell_runner`` fixture from ``conftest.py``
  builds a stub-PATH so the script runs *for real* with bash but external
  commands (``uv``, ``cmake``, ``apt-get``, ``brew``, ``rocminfo``,
  ``nvidia-smi``, ``curl``, ``sha256sum`` …) are intercepted and logged.
  Tests assert the right commands fired in the right order. CI runs Tier
  2.
- **Tier 3 (real, env-gated):** ``@pytest.mark.skipif`` on
  ``RADIO_RUN_REAL_SETUP=1``. Operator runs manually on each target host.
  CI never runs Tier 3.

PR 1 establishes only Tier 1 and the Tier 2 fixture. Later PRs add per-script
Tier 2 + Tier 3 tests as their scripts land.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _discover_scripts() -> list[Path]:
    """Return every ``scripts/*.sh`` (excluding ``scripts/lib/*``).

    Empty list when ``scripts/`` doesn't exist yet — that's the bootstrap
    case in PR 1 where no setup scripts have landed.
    """
    if not SCRIPTS_DIR.exists():
        return []
    return sorted(p for p in SCRIPTS_DIR.glob("*.sh") if p.is_file())


def _discover_lib_scripts() -> list[Path]:
    """Return every ``scripts/lib/*.sh`` (sourced helpers, not executables)."""
    lib_dir = SCRIPTS_DIR / "lib"
    if not lib_dir.exists():
        return []
    return sorted(p for p in lib_dir.glob("*.sh") if p.is_file())


SCRIPTS = _discover_scripts()
LIB_SCRIPTS = _discover_lib_scripts()
ALL_SHELL = SCRIPTS + LIB_SCRIPTS

SHELLCHECK = shutil.which("shellcheck")


# ---------------------------------------------------------------------------
# Tier 1 — Static checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_executable_script_has_bash_shebang(script: Path) -> None:
    """Every executable script under ``scripts/`` starts with bash shebang."""
    first_line = script.read_text().splitlines()[0]
    assert first_line == "#!/usr/bin/env bash", (
        f"{script.name} must start with '#!/usr/bin/env bash', got {first_line!r}"
    )


@pytest.mark.parametrize("script", ALL_SHELL, ids=lambda p: p.name)
def test_script_has_strict_mode(script: Path) -> None:
    """Every shell file (executable + lib) sets ``-euo pipefail`` near the top.

    Lib files are sourced, so they don't need their own shebang — but they
    must still set strict mode if they manipulate state, otherwise an error
    in a sourced helper silently fails.
    """
    text = script.read_text()
    head = "\n".join(text.splitlines()[:10])
    assert "set -euo pipefail" in head, (
        f"{script.relative_to(REPO_ROOT)} must contain 'set -euo pipefail' in the first 10 lines"
    )


@pytest.mark.skipif(SHELLCHECK is None, reason="shellcheck not installed")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_executable_script_passes_shellcheck(script: Path) -> None:
    """Every script under ``scripts/`` passes ``shellcheck -x``."""
    result = subprocess.run(
        [SHELLCHECK, "-x", "-S", "warning", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"shellcheck failed for {script.name}:\n{result.stdout}\n{result.stderr}"
    )


def test_discovery_finds_no_scripts_yet_or_some() -> None:
    """Bootstrap-friendly: this test passes whether scripts/ is empty or not.

    PR 1 ships no scripts. Later PRs ship scripts and the parametrized tests
    above start running. Both states are valid.
    """
    assert isinstance(SCRIPTS, list)
    assert isinstance(LIB_SCRIPTS, list)


# ---------------------------------------------------------------------------
# Tier 2 — shell_runner fixture sanity
# ---------------------------------------------------------------------------


def test_shell_runner_fixture_is_provided(shell_runner) -> None:  # type: ignore[no-untyped-def]
    """The shell_runner fixture from conftest.py is wired into pytest."""
    assert shell_runner is not None
    assert hasattr(shell_runner, "stub")
    assert hasattr(shell_runner, "run")


def test_shell_runner_stubs_intercept_external_commands(shell_runner, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A stubbed command logs invocations and returns the configured exit code."""
    shell_runner.stub("fakecmd", returncode=0, stdout="hello\n")

    script = tmp_path / "probe.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nfakecmd one two\n")
    script.chmod(0o755)

    result = shell_runner.run(str(script))

    assert result.returncode == 0
    assert any(call[0].endswith("fakecmd") and call[1:] == ["one", "two"] for call in result.calls)


def test_shell_runner_propagates_nonzero_exit(shell_runner, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A stub that returns nonzero should propagate up through the script."""
    shell_runner.stub("breakme", returncode=2)

    script = tmp_path / "probe.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nbreakme\n")
    script.chmod(0o755)

    result = shell_runner.run(str(script))

    assert result.returncode != 0


def test_shell_runner_restores_path_after_run(shell_runner, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The fixture must not leak ``$PATH`` modifications across tests."""
    original_path = os.environ.get("PATH", "")
    shell_runner.stub("ephemeral", returncode=0)

    script = tmp_path / "probe.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\nephemeral\n")
    script.chmod(0o755)
    shell_runner.run(str(script))

    # After the run, the parent process's PATH is unchanged.
    assert os.environ.get("PATH", "") == original_path


def test_shell_runner_passes_env_through(shell_runner, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Caller-supplied env variables reach the script."""
    script = tmp_path / "probe.sh"
    script.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\necho "GOT=${RADIO_TEST_VAR:-unset}"\n'
    )
    script.chmod(0o755)

    result = shell_runner.run(str(script), env={"RADIO_TEST_VAR": "hello-world"})

    assert "GOT=hello-world" in result.stdout
