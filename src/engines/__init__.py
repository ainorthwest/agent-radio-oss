"""Engine surface for agent-radio-oss.

The renderer imports each supported engine module directly (e.g.
``from src.engines.kokoro import get_kokoro``). This module exposes:

- ``Engine`` — a Protocol describing the shape an engine must satisfy
  for the long-plan abstraction refactor (Phase 2). Nothing currently
  uses this Protocol at runtime; it is documentation that future
  contributors can typecheck against.
- ``SUPPORTED_ENGINES`` / ``available_engines()`` — the list of engine
  names this distribution supports. The CLI's ``radio config engines``
  and ``radio soundbooth engines`` commands read from here.

Adding a new engine: implement a module under ``src/engines/<name>.py``
that exposes a ``get_<name>()`` lazy loader, add ``"<name>"`` to
``SUPPORTED_ENGINES`` below, and wire dispatch in ``src/renderer.py``.
The Protocol below is the contract — at minimum your engine needs a
name, a sample rate, and a ``render(text, voice_profile, register)``
method returning a mono float32 numpy array.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class Engine(Protocol):
    """A TTS engine that the renderer can dispatch to.

    Not yet enforced at runtime — see the module docstring. The Engine
    Protocol exists so the long-plan engine abstraction has a target
    shape it can converge on without rewriting the renderer.
    """

    name: str
    sample_rate: int

    def render(
        self,
        text: str,
        voice_profile: dict[str, Any],
        register: str = "baseline",
    ) -> np.ndarray:
        """Synthesize ``text`` using the given voice profile.

        ``voice_profile`` is the parsed YAML dict (see
        ``voices/kokoro-*.yaml`` for the canonical Kokoro shape).
        ``register`` selects per-segment delivery overrides defined
        under ``profile["registers"][<register>]``.

        Returns a mono float32 numpy array at ``self.sample_rate``.
        """
        ...


# Engines this distribution supports. The renderer imports each engine
# module directly; this list is what the CLI surfaces to operators.
SUPPORTED_ENGINES: list[str] = ["kokoro"]


def available_engines() -> list[str]:
    """Return the sorted list of engines this distribution supports."""
    return sorted(SUPPORTED_ENGINES)
