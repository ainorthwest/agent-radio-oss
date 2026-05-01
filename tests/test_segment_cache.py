"""Tests for src/segment_cache.py — content-addressed per-segment WAV cache.

The cache is keyed by ``sha256(text + speaker + register + voice_profile_yaml
+ engine)`` truncated to 16 chars. Cache hits skip TTS render entirely; the
WAV is copied from cache to the episode segments directory.

Tests use only soundfile + numpy — no Kokoro init required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_wav(path: Path, sr: int = 24000, duration: float = 0.5) -> None:
    import soundfile as sf

    samples = np.random.RandomState(42).randn(int(sr * duration)).astype(np.float32) * 0.1
    sf.write(str(path), samples, sr)


# Top-level helper for renderer-integration test (top-level so the SHA-keyed
# fake_create can find the same instance across calls).


class _FakeKokoro:
    """Stand-in for kokoro_onnx.Kokoro that records calls and returns a
    short stub WAV. Used by TestRendererIntegration."""

    def __init__(self, sr: int = 24000) -> None:
        self.sr = sr
        self.calls: list[dict] = []

    def create(self, text, voice, speed, lang):
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        # Generate a short deterministic stub
        n = int(0.2 * self.sr)
        return np.linspace(-0.1, 0.1, n, dtype=np.float32), None


# ── compute_segment_hash ─────────────────────────────────────────────────────


class TestComputeSegmentHash:
    def test_deterministic(self):
        from src.segment_cache import compute_segment_hash

        h1 = compute_segment_hash(
            text="Hello world",
            speaker="host_a",
            register="baseline",
            voice_profile={"engine": "kokoro", "kokoro": {"voice_id": "am_michael"}},
            engine="kokoro",
        )
        h2 = compute_segment_hash(
            text="Hello world",
            speaker="host_a",
            register="baseline",
            voice_profile={"engine": "kokoro", "kokoro": {"voice_id": "am_michael"}},
            engine="kokoro",
        )
        assert h1 == h2
        assert len(h1) == 16
        assert all(c in "0123456789abcdef" for c in h1)

    def test_different_text_different_hash(self):
        from src.segment_cache import compute_segment_hash

        h1 = compute_segment_hash(
            text="Hello",
            speaker="host_a",
            register="baseline",
            voice_profile={},
            engine="kokoro",
        )
        h2 = compute_segment_hash(
            text="World",
            speaker="host_a",
            register="baseline",
            voice_profile={},
            engine="kokoro",
        )
        assert h1 != h2

    def test_different_speaker_different_hash(self):
        from src.segment_cache import compute_segment_hash

        kw = dict(text="X", register="baseline", voice_profile={}, engine="kokoro")
        assert compute_segment_hash(speaker="host_a", **kw) != compute_segment_hash(
            speaker="host_b", **kw
        )

    def test_different_voice_profile_different_hash(self):
        from src.segment_cache import compute_segment_hash

        kw = dict(text="X", speaker="host_a", register="baseline", engine="kokoro")
        h1 = compute_segment_hash(voice_profile={"kokoro": {"voice_id": "am_michael"}}, **kw)
        h2 = compute_segment_hash(voice_profile={"kokoro": {"voice_id": "af_bella"}}, **kw)
        assert h1 != h2

    def test_different_engine_different_hash(self):
        from src.segment_cache import compute_segment_hash

        kw = dict(text="X", speaker="host_a", register="baseline", voice_profile={})
        assert compute_segment_hash(engine="kokoro", **kw) != compute_segment_hash(
            engine="chatterbox", **kw
        )

    def test_voice_profile_dict_order_does_not_matter(self):
        """Dict key insertion order must not affect the hash — YAML may emit
        keys in any order."""
        from src.segment_cache import compute_segment_hash

        kw = dict(text="X", speaker="host_a", register="baseline", engine="kokoro")
        a = {"engine": "kokoro", "kokoro": {"voice_id": "am_michael", "speed": 1.0}}
        b = {"kokoro": {"speed": 1.0, "voice_id": "am_michael"}, "engine": "kokoro"}
        assert compute_segment_hash(voice_profile=a, **kw) == compute_segment_hash(
            voice_profile=b, **kw
        )


# ── SegmentCache (file-backed) ───────────────────────────────────────────────


class TestSegmentCache:
    def test_miss_then_hit(self, tmp_path: Path):
        from src.segment_cache import SegmentCache

        cache = SegmentCache(tmp_path / "cache")
        # Miss: nothing stored
        assert cache.get("abc123abc123abc1") is None

        # Store
        wav = tmp_path / "input.wav"
        _write_wav(wav)
        cache.put("abc123abc123abc1", wav)

        # Hit
        cached = cache.get("abc123abc123abc1")
        assert cached is not None
        assert cached.exists()
        assert cached.read_bytes() == wav.read_bytes()

    def test_copy_to_writes_destination(self, tmp_path: Path):
        from src.segment_cache import SegmentCache

        cache = SegmentCache(tmp_path / "cache")
        wav = tmp_path / "input.wav"
        _write_wav(wav)
        cache.put("abcdef0123456789", wav)

        dest = tmp_path / "out" / "seg-000-host_a.wav"
        result = cache.copy_to("abcdef0123456789", dest)
        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == wav.read_bytes()

    def test_copy_to_returns_false_on_miss(self, tmp_path: Path):
        from src.segment_cache import SegmentCache

        cache = SegmentCache(tmp_path / "cache")
        dest = tmp_path / "out" / "seg.wav"
        result = cache.copy_to("f00fbabe12345678", dest)
        assert result is False
        assert not dest.exists()

    def test_creates_cache_dir_lazily(self, tmp_path: Path):
        from src.segment_cache import SegmentCache

        cache_dir = tmp_path / "does-not-exist-yet"
        cache = SegmentCache(cache_dir)
        assert not cache_dir.exists()  # not created on init

        wav = tmp_path / "input.wav"
        _write_wav(wav)
        cache.put("abcdef0123456789", wav)
        assert cache_dir.exists()

    def test_put_is_idempotent(self, tmp_path: Path):
        """Putting the same hash twice should overwrite cleanly."""
        from src.segment_cache import SegmentCache

        cache = SegmentCache(tmp_path / "cache")
        wav1 = tmp_path / "v1.wav"
        wav2 = tmp_path / "v2.wav"
        _write_wav(wav1, duration=0.3)
        _write_wav(wav2, duration=0.6)

        cache.put("deadbeef01234567", wav1)
        cache.put("deadbeef01234567", wav2)

        cached = cache.get("deadbeef01234567")
        assert cached is not None
        assert cached.read_bytes() == wav2.read_bytes()

    def test_invalid_hash_format_rejected(self, tmp_path: Path):
        """Hash must be 16 lowercase hex chars to avoid path traversal."""
        from src.segment_cache import SegmentCache

        cache = SegmentCache(tmp_path / "cache")
        wav = tmp_path / "input.wav"
        _write_wav(wav)
        with pytest.raises(ValueError, match="hash"):
            cache.put("../../../evil", wav)
        with pytest.raises(ValueError, match="hash"):
            cache.get("not-hex-chars!!!")

    def test_stats_tracks_hits_and_misses(self, tmp_path: Path):
        from src.segment_cache import SegmentCache

        cache = SegmentCache(tmp_path / "cache")
        wav = tmp_path / "input.wav"
        _write_wav(wav)
        cache.put("abcdef0123456789", wav)

        # Two hits + one miss
        cache.copy_to("abcdef0123456789", tmp_path / "a.wav")
        cache.copy_to("abcdef0123456789", tmp_path / "b.wav")
        cache.copy_to("f00fbabe12345678", tmp_path / "c.wav")

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1


# ── Renderer integration: cache miss -> render -> populate -> hit ────────────


class TestRendererIntegration:
    """Drive _render_segments_kokoro with a fake Kokoro and assert the cache
    short-circuits the second call."""

    def _make_config(self, sr: int = 24000):
        from types import SimpleNamespace

        return SimpleNamespace(renderer=SimpleNamespace(sample_rate=sr))

    def test_cache_miss_then_hit_skips_kokoro(self, tmp_path: Path, monkeypatch):
        from src import renderer
        from src.segment_cache import SegmentCache

        fake = _FakeKokoro()
        monkeypatch.setattr(renderer, "get_kokoro", lambda: (fake, None))
        monkeypatch.setattr(renderer, "_resolve_voice", lambda k, p: "am_michael")
        monkeypatch.setattr(renderer, "apply_dsp", lambda a, p, sr: a)

        config = self._make_config()
        segments = [{"speaker": "host_a", "text": "Hello world."}]
        voice_profiles = {"host_a": {"engine": "kokoro", "kokoro": {"voice_id": "am_michael"}}}
        seg_dir = tmp_path / "segments"
        seg_dir.mkdir()
        cache = SegmentCache(tmp_path / "cache")

        # First render — cache miss, Kokoro called, segment cached
        manifest1 = renderer._render_segments_kokoro(
            config, segments, voice_profiles, seg_dir, cache=cache
        )
        assert len(fake.calls) == 1
        assert manifest1[0]["cache_hit"] is False
        assert "segment_hash" in manifest1[0]
        assert cache.stats()["hits"] == 0
        assert cache.stats()["misses"] == 1

        # Second render — same text/profile/speaker => same hash => cache hit
        seg_dir2 = tmp_path / "segments2"
        seg_dir2.mkdir()
        manifest2 = renderer._render_segments_kokoro(
            config, segments, voice_profiles, seg_dir2, cache=cache
        )
        assert len(fake.calls) == 1, "Kokoro should not have been called again"
        assert manifest2[0]["cache_hit"] is True
        assert manifest2[0]["segment_hash"] == manifest1[0]["segment_hash"]
        assert (seg_dir2 / manifest2[0]["file"]).exists()
        assert cache.stats()["hits"] == 1

    def test_changed_text_invalidates_cache(self, tmp_path: Path, monkeypatch):
        from src import renderer
        from src.segment_cache import SegmentCache

        fake = _FakeKokoro()
        monkeypatch.setattr(renderer, "get_kokoro", lambda: (fake, None))
        monkeypatch.setattr(renderer, "_resolve_voice", lambda k, p: "am_michael")
        monkeypatch.setattr(renderer, "apply_dsp", lambda a, p, sr: a)

        config = self._make_config()
        voice_profiles = {"host_a": {"engine": "kokoro", "kokoro": {"voice_id": "am_michael"}}}
        cache = SegmentCache(tmp_path / "cache")

        # First render
        seg_dir = tmp_path / "v1"
        seg_dir.mkdir()
        renderer._render_segments_kokoro(
            config,
            [{"speaker": "host_a", "text": "First."}],
            voice_profiles,
            seg_dir,
            cache=cache,
        )

        # Different text -> different hash -> miss again
        seg_dir2 = tmp_path / "v2"
        seg_dir2.mkdir()
        renderer._render_segments_kokoro(
            config,
            [{"speaker": "host_a", "text": "Different."}],
            voice_profiles,
            seg_dir2,
            cache=cache,
        )

        assert len(fake.calls) == 2, "different text should require another Kokoro call"

    def test_no_cache_skips_hashing_and_logs(self, tmp_path: Path, monkeypatch):
        """Backward compat: passing cache=None preserves the old manifest shape
        (no segment_hash, no cache_hit fields)."""
        from src import renderer

        fake = _FakeKokoro()
        monkeypatch.setattr(renderer, "get_kokoro", lambda: (fake, None))
        monkeypatch.setattr(renderer, "_resolve_voice", lambda k, p: "am_michael")
        monkeypatch.setattr(renderer, "apply_dsp", lambda a, p, sr: a)

        config = self._make_config()
        seg_dir = tmp_path / "segments"
        seg_dir.mkdir()
        manifest = renderer._render_segments_kokoro(
            config,
            [{"speaker": "host_a", "text": "x"}],
            {"host_a": {"engine": "kokoro"}},
            seg_dir,
            cache=None,
        )
        assert "segment_hash" not in manifest[0]
        assert "cache_hit" not in manifest[0]
