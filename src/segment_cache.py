"""Per-segment WAV cache, content-addressed.

The renderer is the slow part of the pipeline — Kokoro takes seconds
per segment on CPU. When the editor changes one line in a 20-segment
script, re-rendering all 20 is wasteful. This cache keys rendered WAVs
by ``sha256(text + speaker + register + voice_profile + engine)``, so
unchanged segments are loaded from disk instead of re-synthesized.

Cache layout::

    library/programs/<slug>/cache/segments/<hash>.wav

Hash is the first 16 hex chars of the SHA-256 — short enough for
filesystem hygiene, long enough that collisions on a single station
are negligible (16^16 = 1.8e19, dwarfs the segment count of any
plausible station's lifetime).

The cache is best-effort: a corrupted or missing entry simply triggers
a re-render. Operators can ``rm -rf library/programs/<slug>/cache/`` at
any time to force a clean rebuild.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

_HASH_RE = re.compile(r"^[0-9a-f]{16}$")


def compute_segment_hash(
    *,
    text: str,
    speaker: str,
    register: str,
    voice_profile: dict[str, Any],
    engine: str,
) -> str:
    """Compute the content-addressed cache key for one segment.

    Inputs that affect rendered audio are hashed; inputs that don't
    (like ``topic``, which is metadata) are excluded so unrelated
    edits don't invalidate the cache.

    The voice_profile dict is canonicalized via ``json.dumps(sort_keys=True)``
    so YAML's free key ordering doesn't change the hash.
    """
    payload = json.dumps(
        {
            "text": text,
            "speaker": speaker,
            "register": register,
            "voice_profile": voice_profile,
            "engine": engine,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class SegmentCache:
    """File-backed cache of rendered segment WAVs.

    Use ``get(hash)`` for direct lookup, ``copy_to(hash, dest)`` to
    populate an episode segments directory, ``put(hash, wav_path)`` to
    add a freshly-rendered segment.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._hits = 0
        self._misses = 0

    def _path(self, segment_hash: str) -> Path:
        if not _HASH_RE.match(segment_hash):
            raise ValueError(
                f"invalid segment hash {segment_hash!r}; expected 16 lowercase hex characters"
            )
        return self.cache_dir / f"{segment_hash}.wav"

    def get(self, segment_hash: str) -> Path | None:
        """Return the cached WAV path, or ``None`` on miss."""
        path = self._path(segment_hash)
        if path.exists():
            self._hits += 1
            return path
        self._misses += 1
        return None

    def put(self, segment_hash: str, wav_path: Path) -> None:
        """Store a rendered WAV under ``segment_hash``.

        Writes to a sibling ``.tmp`` file then atomically renames into
        place, so an interrupted ``put()`` (OOM-kill, power loss) cannot
        leave a corrupt entry that a later ``get()`` would happily
        return as a "hit." Same-filesystem rename is atomic on POSIX.
        """
        dest = self._path(segment_hash)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copyfile(wav_path, tmp)
        tmp.replace(dest)

    def copy_to(self, segment_hash: str, dest: Path) -> bool:
        """Copy the cached WAV to ``dest``. Returns True on hit, False on miss.

        Creates ``dest.parent`` if it doesn't exist. Increments hit/miss
        counters identically to :meth:`get` so the caller doesn't need to
        track them separately.
        """
        cached = self.get(segment_hash)
        if cached is None:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, dest)
        return True

    def stats(self) -> dict[str, int]:
        """Hit/miss counts since this cache instance was created."""
        return {"hits": self._hits, "misses": self._misses}
