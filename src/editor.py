"""Script-segment editor — pure JSON ops on script.json.

The editor is the agent's (and operator's) tool for fixing a bad render
without re-running the full pipeline. Each operation:

1. Takes a script dict + arguments.
2. Returns ``(new_script, ScriptDiff)``. The script is a fresh deep
   copy; the input is never mutated.
3. Records what changed in a :class:`ScriptDiff` so the renderer can
   re-render only the affected segments and pull the rest from the
   per-segment cache.

The five operations are the minimum viable surface for an autonomous
station agent:

* :func:`delete_segment` — drop a segment that didn't land
* :func:`replace_text` — fix a misread / mispronunciation
* :func:`reorder_segments` — change the running order
* :func:`insert_segment` — add a missing transition or correction
* :func:`change_voice` — re-cast a segment to a different host

Larger refactors (rewriting the whole script, splitting an episode)
are out of scope — those are curator-stage changes, not editor-stage.

Segment shape (matches what curator emits, see
``library/programs/<slug>/episodes/<date>/script.json``):

    {
        "speaker": "host_a",       # required
        "text": "...",              # required
        "topic": "intro",           # optional, defaulted by renderer
        "register": "baseline",     # optional, defaulted by renderer
    }

Index convention: positional, 0-based, into the ``segments`` list.
Negative indices are intentionally rejected (Python's wrap-around
semantics make agent prompts ambiguous).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScriptDiff:
    """What changed between the input and output script.

    The renderer reads this to decide which segments to re-render and
    which to load from the per-segment cache. Empty lists mean nothing
    of that kind changed.

    Attributes:
        added: indices of newly-inserted segments in the output script
        removed: indices of segments removed (relative to the input)
        modified: indices of segments whose text or speaker changed
        reordered: True if the segment order changed (cache hits still
            valid; only mix-time assembly changes)
    """

    added: list[int] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)
    modified: list[int] = field(default_factory=list)
    reordered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "modified": list(self.modified),
            "reordered": self.reordered,
        }


def _check_segment_index(script: dict[str, Any], index: int) -> None:
    if index < 0:
        raise IndexError(f"negative segment index not supported: {index}")
    if index >= len(script.get("segments", [])):
        raise IndexError(
            f"segment index {index} out of range for script with "
            f"{len(script.get('segments', []))} segments"
        )


def delete_segment(script: dict[str, Any], index: int) -> tuple[dict[str, Any], ScriptDiff]:
    """Remove the segment at ``index``.

    Returns a new script (deep-copied) and a :class:`ScriptDiff` with
    ``removed=[index]``. Subsequent segments shift up by one in the
    output, but the diff records the **input** index so callers can
    correlate against the input script's segment list.
    """
    _check_segment_index(script, index)
    new_script = copy.deepcopy(script)
    del new_script["segments"][index]
    return new_script, ScriptDiff(removed=[index])


def replace_text(
    script: dict[str, Any], index: int, new_text: str
) -> tuple[dict[str, Any], ScriptDiff]:
    """Replace a segment's ``text`` field; preserve other fields.

    Empty or whitespace-only text is rejected — silent segments are not
    a valid editorial intent in the autonomous-station loop.
    """
    if not new_text or not new_text.strip():
        raise ValueError("replacement text cannot be empty or whitespace-only")
    _check_segment_index(script, index)
    new_script = copy.deepcopy(script)
    new_script["segments"][index]["text"] = new_text
    return new_script, ScriptDiff(modified=[index])


def reorder_segments(
    script: dict[str, Any], new_order: list[int]
) -> tuple[dict[str, Any], ScriptDiff]:
    """Rearrange segments by a permutation of input indices.

    ``new_order`` must be a permutation of ``range(len(segments))`` —
    same length, every index appears exactly once. The output's
    ``segments[i]`` is the input's ``segments[new_order[i]]``.

    If ``new_order`` is the identity, the diff records
    ``reordered=False`` so the renderer treats it as a no-op.
    """
    n = len(script.get("segments", []))
    if len(new_order) != n:
        raise ValueError(f"new_order length {len(new_order)} != segment count {n}")
    if sorted(new_order) != list(range(n)):
        raise ValueError(f"new_order must be a permutation of 0..{n - 1}; got {new_order!r}")
    new_script = copy.deepcopy(script)
    new_script["segments"] = [copy.deepcopy(script["segments"][i]) for i in new_order]
    is_reordered = new_order != list(range(n))
    return new_script, ScriptDiff(reordered=is_reordered)


def insert_segment(
    script: dict[str, Any], index: int, segment: dict[str, Any]
) -> tuple[dict[str, Any], ScriptDiff]:
    """Insert a new segment at ``index``; existing segments shift right.

    The new segment must include at minimum ``speaker`` and ``text``.
    ``topic`` and ``register`` are optional (renderer defaults apply).

    ``index`` may equal ``len(segments)`` to append.
    """
    if "speaker" not in segment or not segment.get("speaker"):
        raise ValueError("segment must include a non-empty 'speaker' field")
    if "text" not in segment or not str(segment.get("text", "")).strip():
        raise ValueError("segment must include a non-empty 'text' field")
    n = len(script.get("segments", []))
    if index < 0 or index > n:
        raise IndexError(f"insert index {index} out of range for script with {n} segments")
    new_script = copy.deepcopy(script)
    new_script["segments"].insert(index, copy.deepcopy(segment))
    return new_script, ScriptDiff(added=[index])


def change_voice(
    script: dict[str, Any], index: int, new_speaker: str
) -> tuple[dict[str, Any], ScriptDiff]:
    """Reassign a segment to a different cast voice.

    Only ``speaker`` changes; text, topic, and register are preserved.
    If the new speaker matches the current one, returns an unchanged
    script with an empty diff.
    """
    _check_segment_index(script, index)
    current = script["segments"][index].get("speaker")
    new_script = copy.deepcopy(script)
    if current == new_speaker:
        return new_script, ScriptDiff()
    new_script["segments"][index]["speaker"] = new_speaker
    return new_script, ScriptDiff(modified=[index])
