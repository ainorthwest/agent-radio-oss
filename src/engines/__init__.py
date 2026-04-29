"""Engine registry for agent-radio-oss.

A deliberately minimal Protocol + registry. The goal is to keep Kokoro
pluggable without pre-deciding the long-plan engine abstraction (Phase 2).

Adding a new engine: subclass Engine, register it in
``src/engines/__init__.py`` (or import it from your module so the
registration side-effect runs), and the renderer will dispatch to it
when a voice profile sets ``engine: <your_name>``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class Engine(Protocol):
    """A TTS engine that the renderer can dispatch to."""

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


REGISTRY: dict[str, type[Engine]] = {}


def register(engine_cls: type[Engine]) -> type[Engine]:
    """Decorator to register an Engine subclass under its ``name`` attribute."""
    REGISTRY[engine_cls.name] = engine_cls
    return engine_cls


def get_engine(name: str) -> Engine:
    """Resolve an engine by name and return an instance.

    Raises:
        ValueError: if no engine is registered under ``name``.
    """
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY)) or "<none>"
        raise ValueError(f"Unknown engine {name!r}. Registered: {available}")
    return REGISTRY[name]()


def available_engines() -> list[str]:
    """Return the sorted list of registered engine names."""
    return sorted(REGISTRY)


# Trigger registration side-effects.
from src.engines import kokoro  # noqa: E402, F401
