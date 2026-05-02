"""Pytest configuration — ensure src/ is importable, plus the
``shell_runner`` fixture for testing scripts under ``scripts/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add project root to sys.path so tests can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.shell_harness import ShellRunner  # noqa: E402


@pytest.fixture
def shell_runner(tmp_path: Path) -> ShellRunner:
    """Per-test ``ShellRunner`` instance scoped to the test's tmp_path.

    See ``tests/shell_harness.py`` for the path-stubbing mechanics.
    """
    return ShellRunner(tmp_path)
