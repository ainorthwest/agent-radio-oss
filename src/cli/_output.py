"""Shared CLI output helpers.

Centralizes JSON/human output formatting, error handling,
and optional-dependency guards so every command group stays thin.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.cli import State


def output(state: State, data: Any, human_fmt: str | None = None) -> None:
    """Print *data* as JSON (when ``--json``) or human-readable text.

    Parameters
    ----------
    state:
        Global CLI state (carries the ``json_output`` flag).
    data:
        Payload — dataclass, dict, list, Path, str, or int.
    human_fmt:
        Optional pre-formatted string for human output.  When *None*,
        ``str(data)`` is used.
    """
    if state.json_output:
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            blob = dataclasses.asdict(data)  # type: ignore[arg-type]
        elif isinstance(data, (dict, list)):
            blob = data
        else:
            blob = {"result": str(data)}
        print(json.dumps(blob, indent=2, default=str))
    elif human_fmt is not None:
        print(human_fmt)
    else:
        print(data)


def err(msg: str) -> None:
    """Print an error to stderr and exit with code 1."""
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def require_extra(extra_name: str, test_import: str) -> None:
    """Guard for optional dependencies.

    Call at the *top* of a command body — before importing the heavy module.
    If the import fails, prints a helpful install hint and exits.
    """
    try:
        __import__(test_import)
    except ImportError:
        err(f"Missing dependency. Install with: uv sync --extra {extra_name}")
