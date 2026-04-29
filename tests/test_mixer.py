"""Tests for mixer — gap timing, ducking, and assembly."""

import json
from pathlib import Path

import numpy as np

from src.mixer import (
    GAP_AFTER_EMPHASIS,
    GAP_AFTER_REACTIVE,
    GAP_AFTER_REFLECTIVE,
    GAP_DEFAULT_CROSS,
    GAP_SAME_SPEAKER,
    GAP_SHORT_ACK,
    GAP_TOPIC_CHANGE,
    _apply_ducking,
    _assemble_voice_track,
    _build_duck_envelope,
    _compute_gap_seconds,
)


class TestGapTiming:
    """Tests for context-aware gap computation."""

    def test_short_acknowledgment_fast_pickup(self):
        """Short cross-speaker responses should have near-instant gaps."""
        current = {"speaker": "host_a", "register": "baseline", "topic": "A"}
        next_seg = {"speaker": "host_b", "register": "baseline", "topic": "A", "text": "Right."}
        gap, ctx = _compute_gap_seconds(current, next_seg)
        assert 0.04 <= gap <= 0.13
        assert ctx == GAP_SHORT_ACK

    def test_topic_change_longer_gap(self):
        """Topic transitions should have longer gaps."""
        current = {"speaker": "host_a", "register": "baseline", "topic": "Intro"}
        next_seg = {
            "speaker": "host_b",
            "register": "baseline",
            "topic": "Main",
            "text": "OK so here's the thing about that.",
        }
        gap, ctx = _compute_gap_seconds(current, next_seg)
        assert 0.23 <= gap <= 0.42
        assert ctx == GAP_TOPIC_CHANGE

    def test_same_speaker_breath_pause(self):
        """Same speaker continuing gets a breath pause."""
        current = {"speaker": "host_a", "register": "baseline", "topic": "A"}
        next_seg = {
            "speaker": "host_a",
            "register": "baseline",
            "topic": "A",
            "text": "And furthermore we should consider the implications.",
        }
        gap, ctx = _compute_gap_seconds(current, next_seg)
        assert 0.05 <= gap <= 0.13
        assert ctx == GAP_SAME_SPEAKER

    def test_after_reflective_longer_gap(self):
        """After reflective register, let the moment breathe."""
        current = {"speaker": "host_a", "register": "reflective", "topic": "A"}
        next_seg = {
            "speaker": "host_b",
            "register": "baseline",
            "topic": "A",
            "text": "That is a really important point to consider.",
        }
        gap, ctx = _compute_gap_seconds(current, next_seg)
        assert 0.18 <= gap <= 0.37
        assert ctx == GAP_AFTER_REFLECTIVE

    def test_after_reactive_fast_energy(self):
        """After reactive register, energy carries forward."""
        current = {"speaker": "host_a", "register": "reactive", "topic": "A"}
        next_seg = {
            "speaker": "host_b",
            "register": "baseline",
            "topic": "A",
            "text": "That is a really interesting observation about the system.",
        }
        gap, ctx = _compute_gap_seconds(current, next_seg)
        assert 0.03 <= gap <= 0.12
        assert ctx == GAP_AFTER_REACTIVE

    def test_word_count_from_manifest(self):
        """Gap timing should use word_count field if present (manifest format)."""
        current = {"speaker": "host_a", "register": "baseline", "topic": "A"}
        next_seg = {"speaker": "host_b", "register": "baseline", "topic": "A", "word_count": 2}
        gap, ctx = _compute_gap_seconds(current, next_seg)
        assert 0.04 <= gap <= 0.13  # short ack range
        assert ctx == GAP_SHORT_ACK

    def test_after_emphasis_gap(self):
        """After emphasis register, moderate beat before response."""
        current = {"speaker": "host_a", "register": "emphasis", "topic": "A"}
        next_seg = {
            "speaker": "host_b",
            "register": "baseline",
            "topic": "A",
            "text": "That is a really great observation about things.",
        }
        gap, ctx = _compute_gap_seconds(current, next_seg)
        assert 0.07 <= gap <= 0.18
        assert ctx == GAP_AFTER_EMPHASIS

    def test_default_cross_speaker(self):
        """Default cross-speaker same-topic gap."""
        current = {"speaker": "host_a", "register": "baseline", "topic": "A"}
        next_seg = {
            "speaker": "host_b",
            "register": "baseline",
            "topic": "A",
            "text": "That is a very interesting point to consider.",
        }
        gap, ctx = _compute_gap_seconds(current, next_seg)
        assert 0.05 <= gap <= 0.16
        assert ctx == GAP_DEFAULT_CROSS


class TestDuckEnvelope:
    """Tests for voice activity envelope generation."""

    def test_envelope_shape(self):
        """Envelope should be 1.0 during voice and 0.0 during silence."""
        total = 48000  # 1 second at 48kHz
        regions = [(0, 12000), (24000, 36000)]
        envelope = _build_duck_envelope(total, regions, fade_samples=0)
        assert envelope.shape == (total,)
        assert np.allclose(envelope[:12000], 1.0)
        assert np.allclose(envelope[12000:24000], 0.0)
        assert np.allclose(envelope[24000:36000], 1.0)
        assert np.allclose(envelope[36000:], 0.0)

    def test_envelope_with_fade(self):
        """Envelope should smooth transitions with fade samples."""
        total = 48000
        regions = [(10000, 20000)]
        envelope = _build_duck_envelope(total, regions, fade_samples=500)
        assert envelope.shape == (total,)
        # Core voice region should be near 1.0
        assert envelope[15000] > 0.8
        # Well outside voice region should be near 0.0
        assert envelope[0] < 0.1
        assert envelope[40000] < 0.1

    def test_empty_regions(self):
        """No voice regions should produce all-zero envelope."""
        envelope = _build_duck_envelope(48000, [], fade_samples=0)
        assert np.allclose(envelope, 0.0)


class TestDucking:
    """Tests for music ducking math."""

    def test_ducking_reduces_volume(self):
        """Music should be quieter where voice is active."""
        music = np.ones(100, dtype=np.float32)
        envelope = np.ones(100, dtype=np.float32)  # all voice
        ducked = _apply_ducking(music, envelope, duck_db=-18.0, bed_level_db=-6.0)
        expected_gain = 10 ** (-18.0 / 20)  # ~0.126
        assert np.allclose(ducked, expected_gain, atol=0.01)

    def test_no_ducking_in_silence(self):
        """Music should play at bed level where no voice."""
        music = np.ones(100, dtype=np.float32)
        envelope = np.zeros(100, dtype=np.float32)  # no voice
        ducked = _apply_ducking(music, envelope, duck_db=-18.0, bed_level_db=-6.0)
        expected_gain = 10 ** (-6.0 / 20)  # ~0.501
        assert np.allclose(ducked, expected_gain, atol=0.01)

    def test_ducking_interpolates(self):
        """Ducking should interpolate between duck and bed levels."""
        music = np.ones(100, dtype=np.float32)
        envelope = np.full(100, 0.5, dtype=np.float32)  # half voice
        ducked = _apply_ducking(music, envelope, duck_db=-18.0, bed_level_db=-6.0)
        duck_gain = 10 ** (-18.0 / 20)
        bed_gain = 10 ** (-6.0 / 20)
        expected = bed_gain * 0.5 + duck_gain * 0.5
        assert np.allclose(ducked, expected, atol=0.01)


class TestAssemblyGapRecords:
    """Tests for gap records emitted during voice track assembly."""

    def _make_manifest(self, tmp_path: Path, n_segments: int = 3) -> dict:
        """Build a minimal manifest with synthetic WAVs."""
        import soundfile as sf

        seg_dir = tmp_path / "segments"
        seg_dir.mkdir()
        segments = []
        for i in range(n_segments):
            speaker = "host_a" if i % 2 == 0 else "host_b"
            fname = f"seg-{i:03d}-{speaker}.wav"
            t = np.linspace(0, 0.5, 12000, endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            sf.write(str(seg_dir / fname), audio, 24000)
            segments.append(
                {
                    "index": i,
                    "speaker": speaker,
                    "register": "baseline",
                    "topic": "A",
                    "text": "This is a test segment with some words.",
                    "file": fname,
                }
            )
        return {
            "segments_dir": str(seg_dir),
            "segments": segments,
            "sample_rate": 24000,
        }

    def test_gap_records_in_assembly(self, tmp_path: Path) -> None:
        """3-segment manifest should produce 2 gap records."""
        manifest = self._make_manifest(tmp_path, n_segments=3)
        _, _, gap_records = _assemble_voice_track(manifest, 24000)
        assert len(gap_records) == 2
        for rec in gap_records:
            assert "context_type" in rec
            assert "duration_seconds" in rec
            assert "after_segment" in rec
            assert "before_segment" in rec
            assert "gap_start_seconds" in rec
            assert rec["duration_seconds"] > 0

    def test_single_segment_no_gaps(self, tmp_path: Path) -> None:
        """Single segment should produce zero gap records."""
        manifest = self._make_manifest(tmp_path, n_segments=1)
        _, _, gap_records = _assemble_voice_track(manifest, 24000)
        assert len(gap_records) == 0

    def test_gap_context_types_complete(self) -> None:
        """Verify all 7 context types are reachable via _compute_gap_seconds."""
        cases = [
            # short_ack: short cross-speaker response
            (
                {"speaker": "A", "register": "baseline", "topic": "T"},
                {"speaker": "B", "register": "baseline", "topic": "T", "text": "Yes."},
                GAP_SHORT_ACK,
            ),
            # topic_change: different topics, different speakers
            (
                {"speaker": "A", "register": "baseline", "topic": "T1"},
                {
                    "speaker": "B",
                    "register": "baseline",
                    "topic": "T2",
                    "text": "Moving on to the next topic now.",
                },
                GAP_TOPIC_CHANGE,
            ),
            # same_speaker: same speaker continues
            (
                {"speaker": "A", "register": "baseline", "topic": "T"},
                {
                    "speaker": "A",
                    "register": "baseline",
                    "topic": "T",
                    "text": "And furthermore we should consider this point.",
                },
                GAP_SAME_SPEAKER,
            ),
            # after_reflective
            (
                {"speaker": "A", "register": "reflective", "topic": "T"},
                {
                    "speaker": "B",
                    "register": "baseline",
                    "topic": "T",
                    "text": "That is a really important point to consider now.",
                },
                GAP_AFTER_REFLECTIVE,
            ),
            # after_emphasis
            (
                {"speaker": "A", "register": "emphasis", "topic": "T"},
                {
                    "speaker": "B",
                    "register": "baseline",
                    "topic": "T",
                    "text": "That is a really great observation about things today.",
                },
                GAP_AFTER_EMPHASIS,
            ),
            # after_reactive
            (
                {"speaker": "A", "register": "reactive", "topic": "T"},
                {
                    "speaker": "B",
                    "register": "baseline",
                    "topic": "T",
                    "text": "That is a really interesting observation about the system.",
                },
                GAP_AFTER_REACTIVE,
            ),
            # default_cross: baseline cross-speaker, same topic, > 4 words
            (
                {"speaker": "A", "register": "baseline", "topic": "T"},
                {
                    "speaker": "B",
                    "register": "baseline",
                    "topic": "T",
                    "text": "That is a very interesting point to consider.",
                },
                GAP_DEFAULT_CROSS,
            ),
        ]
        seen_types = set()
        for current, next_seg, expected_type in cases:
            _, ctx = _compute_gap_seconds(current, next_seg)
            assert ctx == expected_type, f"Expected {expected_type}, got {ctx}"
            seen_types.add(ctx)

        # Verify all 7 types were produced
        all_types = {
            GAP_SHORT_ACK,
            GAP_TOPIC_CHANGE,
            GAP_SAME_SPEAKER,
            GAP_AFTER_REFLECTIVE,
            GAP_AFTER_EMPHASIS,
            GAP_AFTER_REACTIVE,
            GAP_DEFAULT_CROSS,
        }
        assert seen_types == all_types


# ── Editorial integration tests (Phase 4C) ─────────────────────────────────


class TestEditorialIntegration:
    """Tests for editorial manifest integration in the mixer."""

    def _make_manifest(self, tmp_path: Path, n_segments: int = 3) -> dict:
        """Build a minimal manifest with synthetic WAVs."""
        import soundfile as sf

        seg_dir = tmp_path / "segments"
        seg_dir.mkdir()
        segments = []
        for i in range(n_segments):
            speaker = "host_a" if i % 2 == 0 else "host_b"
            fname = f"seg-{i:03d}-{speaker}.wav"
            t = np.linspace(0, 0.5, 12000, endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            sf.write(str(seg_dir / fname), audio, 24000)
            segments.append(
                {
                    "index": i,
                    "speaker": speaker,
                    "register": "baseline",
                    "topic": "A",
                    "text": "This is a test segment with some words.",
                    "file": fname,
                }
            )
        return {
            "segments_dir": str(seg_dir),
            "segments": segments,
            "sample_rate": 24000,
        }

    def test_no_editorial_backward_compat(self, tmp_path: Path):
        """V1 manifest (no editorial block) should work unchanged."""
        manifest = self._make_manifest(tmp_path, n_segments=3)
        audio, regions, gaps = _assemble_voice_track(manifest, 24000)
        assert len(regions) == 3
        assert len(gaps) == 2
        assert len(audio) > 0

    def test_skip_segment(self, tmp_path: Path):
        """Skipping a segment should reduce voice regions."""
        from src.editorial import EditorialManifest, PacingConfig, SegmentOverride

        manifest = self._make_manifest(tmp_path, n_segments=3)
        editorial = EditorialManifest(
            segment_overrides={1: SegmentOverride(skip=True)},
            music_cues=[],
            pacing=PacingConfig(),
        )
        audio, regions, gaps = _assemble_voice_track(manifest, 24000, editorial)
        # Skipped segment 1 → only 2 voice regions
        assert len(regions) == 2

    def test_volume_adjustment(self, tmp_path: Path):
        """Volume adjustment should change audio amplitude."""
        from src.editorial import EditorialManifest, PacingConfig, SegmentOverride

        manifest = self._make_manifest(tmp_path, n_segments=2)

        # No editorial — get baseline amplitude
        audio_base, _, _ = _assemble_voice_track(manifest, 24000)
        max_base = np.max(np.abs(audio_base))

        # With +6dB on segment 0
        editorial = EditorialManifest(
            segment_overrides={0: SegmentOverride(volume_db=6.0)},
            music_cues=[],
            pacing=PacingConfig(),
        )
        audio_loud, _, _ = _assemble_voice_track(manifest, 24000, editorial)
        max_loud = np.max(np.abs(audio_loud))

        # +6dB ≈ 2x amplitude
        assert max_loud > max_base * 1.5

    def test_gap_override(self, tmp_path: Path):
        """Gap override should change gap duration."""
        from src.editorial import EditorialManifest, PacingConfig, SegmentOverride

        manifest = self._make_manifest(tmp_path, n_segments=2)
        editorial = EditorialManifest(
            segment_overrides={0: SegmentOverride(gap_after_seconds=1.0)},
            music_cues=[],
            pacing=PacingConfig(),
        )
        _, _, gaps = _assemble_voice_track(manifest, 24000, editorial)
        assert len(gaps) == 1
        assert gaps[0]["context_type"] == "editorial_override"
        assert abs(gaps[0]["duration_seconds"] - 1.0) < 0.01

    def test_global_gap_multiplier(self, tmp_path: Path):
        """Global gap multiplier should scale computed gaps."""
        import random as rng_mod

        from src.editorial import EditorialManifest, PacingConfig

        manifest = self._make_manifest(tmp_path, n_segments=3)

        # Use fixed seed so both calls get the same base gap values
        rng_mod.seed(12345)
        _, _, gaps_base = _assemble_voice_track(manifest, 24000)

        # 2x gaps — same seed to get same base random values before multiplier
        rng_mod.seed(12345)
        editorial = EditorialManifest(
            segment_overrides={},
            music_cues=[],
            pacing=PacingConfig(global_gap_multiplier=2.0),
        )
        _, _, gaps_2x = _assemble_voice_track(manifest, 24000, editorial)

        for base, scaled in zip(gaps_base, gaps_2x):
            assert abs(scaled["duration_seconds"] - base["duration_seconds"] * 2.0) < 0.01

    def test_global_gap_multiplier_skips_overrides(self, tmp_path: Path):
        """Global gap multiplier must NOT scale explicit editorial gap overrides."""
        from src.editorial import EditorialManifest, PacingConfig, SegmentOverride

        manifest = self._make_manifest(tmp_path, n_segments=3)

        editorial = EditorialManifest(
            segment_overrides={0: SegmentOverride(gap_after_seconds=1.0)},
            music_cues=[],
            pacing=PacingConfig(global_gap_multiplier=2.0),
        )
        _, _, gaps = _assemble_voice_track(manifest, 24000, editorial)

        # First gap has an explicit override of 1.0s — should NOT be scaled
        assert gaps[0]["context_type"] == "editorial_override"
        assert abs(gaps[0]["duration_seconds"] - 1.0) < 0.01

    def test_editorial_none_handled(self, tmp_path: Path):
        """editorial=None should be handled gracefully."""
        manifest = self._make_manifest(tmp_path, n_segments=2)
        audio, regions, gaps = _assemble_voice_track(manifest, 24000, editorial=None)
        assert len(regions) == 2


class TestEditorialSerialization:
    """Tests for editorial manifest serialization (Phase 4C bug fix)."""

    def test_volume_db_preserved_in_serialization(self, tmp_path: Path):
        """Bug fix: volume_db was missing from write_editorial_manifest."""
        from src.editorial import (
            EditorialManifest,
            MusicCue,
            PacingConfig,
            write_editorial_manifest,
        )

        editorial = EditorialManifest(
            segment_overrides={},
            music_cues=[
                MusicCue(
                    type="sting",
                    after_segment=2,
                    asset="test.wav",
                    volume_db=-3.0,
                ),
            ],
            pacing=PacingConfig(),
        )

        # Create a minimal manifest file
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "segments": [],
                    "segments_dir": str(tmp_path),
                }
            )
        )

        write_editorial_manifest(manifest_path, editorial)

        # Read back and verify volume_db is preserved
        import json as json_mod

        data = json_mod.loads(manifest_path.read_text())
        cues = data["editorial"]["music_cues"]
        assert len(cues) == 1
        assert cues[0]["volume_db"] == -3.0

    def test_round_trip_empty_editorial(self, tmp_path: Path):
        """Empty editorial should serialize/deserialize cleanly."""
        from src.editorial import EditorialManifest, load_editorial, write_editorial_manifest

        editorial = EditorialManifest()
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "segments": [],
                    "segments_dir": str(tmp_path),
                }
            )
        )

        write_editorial_manifest(manifest_path, editorial)
        import json as json_mod

        data = json_mod.loads(manifest_path.read_text())
        loaded = load_editorial(data)
        assert not loaded.has_overrides()

    def test_round_trip_full_editorial(self, tmp_path: Path):
        """Full editorial manifest should survive serialization round-trip."""
        from src.editorial import (
            EditorialManifest,
            MusicCue,
            PacingConfig,
            SegmentOverride,
            load_editorial,
            write_editorial_manifest,
        )

        editorial = EditorialManifest(
            segment_overrides={
                0: SegmentOverride(volume_db=-2.0, note="too loud"),
                3: SegmentOverride(skip=True),
            },
            music_cues=[
                MusicCue(
                    type="transition",
                    after_segment=1,
                    asset="assets/music/sting.wav",
                    fade_in_s=0.5,
                    fade_out_s=1.0,
                    volume_db=-6.0,
                ),
            ],
            pacing=PacingConfig(global_gap_multiplier=1.2),
        )

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "segments": [],
                    "segments_dir": str(tmp_path),
                }
            )
        )

        write_editorial_manifest(manifest_path, editorial)
        import json as json_mod

        data = json_mod.loads(manifest_path.read_text())
        loaded = load_editorial(data)

        assert loaded.has_overrides()
        assert loaded.segment_overrides[0].volume_db == -2.0
        assert loaded.segment_overrides[3].skip is True
        assert len(loaded.music_cues) == 1
        assert loaded.music_cues[0].volume_db == -6.0
        assert loaded.pacing.global_gap_multiplier == 1.2
