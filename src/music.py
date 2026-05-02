"""Music generation surface — v0.1.0 stub, real engine in v0.1.1.

Why this module exists in v0.1.0
--------------------------------

`src/mixer.py` imports `MusicParams` and `generate` from this module when
a script segment carries a music cue with a `generate:` prefix
(e.g. ``generate:sting``, ``generate:musicgen:<prompt>``). For OSS
v0.1.0, all music for the only shipped show (Haystack News) is
pre-rendered on disk under ``library/programs/haystack-news/assets/``
and overlaid by the mixer's static-asset path. No script in v0.1.0
emits a ``generate:`` cue, so this module is import-safe but never
actually invoked.

If a future script *does* request generation, ``generate()`` raises
``NotImplementedError`` with a message that names the issue tracking
v0.1.1 work and tells the operator what to do instead. That is the
agent-experience contract: tools that can't do something must tell the
agent the next move.

The plan for v0.1.1
-------------------

Replace ``generate()`` with a real call to Stable Audio Open
(``stabilityai/stable-audio-open-1.0``) backed by ``stable-audio-tools``
on PyTorch. See https://github.com/ainorthwest/agent-radio-oss/issues/9
for the full v0.1.1 contract — required pre-flight (PyTorch ROCm wheel
index in ``pyproject.toml``), cross-hardware bring-up matrix, license
audit gate, and acceptance criteria.

Keep the dataclass shapes below stable so the mixer call site at
``src/mixer.py:658-680`` does not need to change when the engine ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# v0.1.1 GitHub issue tracking the real implementation. Surfaced in the
# NotImplementedError message so the operator knows where to follow up.
_V0_1_1_ISSUE_URL = "https://github.com/ainorthwest/agent-radio-oss/issues/9"


@dataclass
class MusicParams:
    """Parameters for a music-generation request.

    Field set covers both call shapes the mixer uses today:
      - MusicGen-style: ``MusicParams(engine="musicgen", prompt=...)``
      - MIDI/sting-style: ``MusicParams(type="sting", key="C")``

    Defaults are conservative so an under-specified call still
    produces a defensible-shape request when v0.1.1 lands.
    """

    engine: str = "none"
    prompt: str = ""
    type: str = "sting"
    key: str = "C"
    duration_s: float = 5.0
    seed: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class MusicAsset:
    """A generated music asset on disk."""

    path: Path
    type: str
    duration_s: float
    params: MusicParams


def generate(
    params: MusicParams,
    output_dir: Path,
    **kwargs: Any,
) -> MusicAsset:
    """Generate a music asset to ``output_dir``.

    v0.1.0: not implemented. Raises ``NotImplementedError`` with a
    message naming the v0.1.1 tracking issue and the recommended
    workaround. The mixer at ``src/mixer.py:679-681`` catches this
    exception type specifically and surfaces an AX-friendly warning
    so the operator knows to substitute a pre-rendered asset path.

    v0.1.1+: dispatches to Stable Audio Open via stable-audio-tools.
    """
    raise NotImplementedError(
        "Music generation is deferred to v0.1.1. "
        f"See {_V0_1_1_ISSUE_URL} for the implementation plan. "
        "For v0.1.0, replace any 'generate:*' music cue with a "
        "pre-rendered asset path in your show's program.yaml music block "
        "(see library/programs/haystack-news/program.yaml for an example)."
    )
