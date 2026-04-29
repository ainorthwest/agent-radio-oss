"""Tests for cross-episode learning module (Phase 4D)."""

import json
from dataclasses import dataclass
from pathlib import Path

from src.episode_history import (
    EpisodeSummary,
    append_episode,
    detect_voice_drift,
    extract_summary,
    find_effective_patterns,
    load_history,
    render_quality_trend,
    score_trend,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_summary(
    date: str = "2026-03-13",
    score: float = 0.75,
    speakers: dict[str, float] | None = None,
    chemistry: float = 0.8,
    fingerprints: dict[str, dict[str, float]] | None = None,
    structure: dict[str, float] | None = None,
) -> EpisodeSummary:
    """Create a test EpisodeSummary."""
    return EpisodeSummary(
        date=date,
        overall_score=score,
        speaker_scores=speakers or {"host_a": 0.7, "host_b": 0.8},
        chemistry_score=chemistry,
        segment_count=12,
        total_duration_s=180.0,
        voice_fingerprints=fingerprints or {},
        script_structure=structure or {},
    )


# ── Storage tests ────────────────────────────────────────────────────────────


class TestStorage:
    """Tests for JSONL append/load."""

    def test_append_creates_file(self, tmp_path: Path):
        path = tmp_path / "history.jsonl"
        summary = _make_summary()
        append_episode(summary, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_append_and_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "history.jsonl"
        s1 = _make_summary(date="2026-03-13", score=0.7)
        s2 = _make_summary(date="2026-03-14", score=0.8)
        append_episode(s1, path)
        append_episode(s2, path)

        history = load_history(path)
        assert len(history) == 2
        assert history[0].date == "2026-03-13"
        assert history[1].date == "2026-03-14"
        assert history[0].overall_score == 0.7
        assert history[1].overall_score == 0.8

    def test_load_sorted_by_date(self, tmp_path: Path):
        path = tmp_path / "history.jsonl"
        # Append out of order
        append_episode(_make_summary(date="2026-03-15"), path)
        append_episode(_make_summary(date="2026-03-13"), path)
        append_episode(_make_summary(date="2026-03-14"), path)

        history = load_history(path)
        dates = [ep.date for ep in history]
        assert dates == ["2026-03-13", "2026-03-14", "2026-03-15"]

    def test_load_creates_dir(self, tmp_path: Path):
        path = tmp_path / "nested" / "dir" / "history.jsonl"
        summary = _make_summary()
        append_episode(summary, path)
        assert path.exists()

    def test_load_empty_file(self, tmp_path: Path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        history = load_history(path)
        assert history == []

    def test_load_nonexistent_file(self, tmp_path: Path):
        path = tmp_path / "nonexistent.jsonl"
        history = load_history(path)
        assert history == []

    def test_corrupt_line_skipped(self, tmp_path: Path):
        path = tmp_path / "history.jsonl"
        good = _make_summary(date="2026-03-13")
        path.write_text(
            good.to_json()
            + "\n"
            + "THIS IS NOT JSON\n"
            + _make_summary(date="2026-03-14").to_json()
            + "\n"
        )
        history = load_history(path)
        assert len(history) == 2


# ── Extract summary tests ───────────────────────────────────────────────────


class TestExtractSummary:
    """Tests for extract_summary from mock reports."""

    def test_no_reports(self):
        summary = extract_summary()
        assert summary.date == "unknown"
        assert summary.overall_score == 0.0

    def test_from_episode_report(self):
        """Mock an EpisodeReport-like object."""

        @dataclass
        class MockSegment:
            duration_seconds: float = 5.0
            features: dict = None  # type: ignore[assignment]

            def __post_init__(self):
                if self.features is None:
                    self.features = {"dnsmos_ovr": 4.0, "wer": 0.05}

        @dataclass
        class MockSpeaker:
            score: float = 0.75
            mean_features: dict = None  # type: ignore[assignment]

            def __post_init__(self):
                if self.mean_features is None:
                    self.mean_features = {
                        "spectral_centroid_mean": 1200.0,
                        "pitch_variance": 0.1,
                        "pitch_range_normalized": 0.3,
                        "lufs_approx": -18.0,
                        "speech_rate_variation": 0.2,
                    }

        @dataclass
        class MockChemistry:
            overall_chemistry: float = 0.85

        @dataclass
        class MockReport:
            episode_date: str = "2026-03-13"
            overall_score: float = 0.78
            speaker_reports: dict = None  # type: ignore[assignment]
            chemistry: MockChemistry = None  # type: ignore[assignment]
            segment_reports: list = None  # type: ignore[assignment]

            def __post_init__(self):
                if self.speaker_reports is None:
                    self.speaker_reports = {
                        "host_a": MockSpeaker(),
                        "host_b": MockSpeaker(score=0.80),
                    }
                if self.chemistry is None:
                    self.chemistry = MockChemistry()
                if self.segment_reports is None:
                    self.segment_reports = [MockSegment(), MockSegment(), MockSegment()]

        report = MockReport()
        summary = extract_summary(episode_report=report)

        assert summary.date == "2026-03-13"
        assert summary.overall_score == 0.78
        assert summary.speaker_scores["host_a"] == 0.75
        assert summary.chemistry_score == 0.85
        assert summary.segment_count == 3
        assert summary.total_duration_s == 15.0
        assert abs(summary.mean_dnsmos - 4.0) < 0.01
        assert abs(summary.mean_wer - 0.05) < 0.01
        assert "spectral_centroid_mean" in summary.voice_fingerprints["host_a"]

    def test_from_script_report(self):
        @dataclass
        class MockScriptReport:
            overall_score: float = 0.82
            dimension_scores: dict = None  # type: ignore[assignment]

            def __post_init__(self):
                if self.dimension_scores is None:
                    self.dimension_scores = {"register_balance": 0.9, "speaker_balance": 0.85}

        summary = extract_summary(script_report=MockScriptReport())
        assert summary.script_score == 0.82
        assert summary.script_structure["register_balance"] == 0.9


# ── Voice drift tests ────────────────────────────────────────────────────────


class TestVoiceDrift:
    """Tests for voice drift detection."""

    def test_stable_voice_low_drift(self):
        """Consistent features across episodes → low CV."""
        fp = {"spectral_centroid_mean": 1200.0, "pitch_variance": 0.1}
        history = [
            _make_summary(date=f"2026-03-{10 + i}", fingerprints={"host_a": fp}) for i in range(5)
        ]
        drift = detect_voice_drift(history, "host_a")
        assert all(cv < 0.01 for cv in drift.values())

    def test_varying_voice_high_drift(self):
        """Varying features → high CV."""
        history = []
        for i in range(5):
            fp = {
                "spectral_centroid_mean": 800.0 + i * 200,  # 800, 1000, 1200, 1400, 1600
                "pitch_variance": 0.05 + i * 0.05,
            }
            history.append(_make_summary(date=f"2026-03-{10 + i}", fingerprints={"host_a": fp}))
        drift = detect_voice_drift(history, "host_a")
        assert drift["spectral_centroid_mean"] > 0.1

    def test_single_episode_returns_empty(self):
        history = [_make_summary(fingerprints={"host_a": {"centroid": 1000.0}})]
        drift = detect_voice_drift(history, "host_a")
        assert drift == {}

    def test_unknown_speaker_returns_empty(self):
        history = [_make_summary(fingerprints={"host_a": {"centroid": 1000.0}})]
        drift = detect_voice_drift(history, "host_z")
        assert drift == {}

    def test_window_limits_data(self):
        """Window should only use the last N episodes."""
        fp_stable = {"centroid": 1000.0}
        fp_wild = {"centroid": 5000.0}
        history = [
            _make_summary(date="2026-03-01", fingerprints={"host_a": fp_wild}),
            _make_summary(date="2026-03-02", fingerprints={"host_a": fp_wild}),
            # Last 3 are stable
            _make_summary(date="2026-03-10", fingerprints={"host_a": fp_stable}),
            _make_summary(date="2026-03-11", fingerprints={"host_a": fp_stable}),
            _make_summary(date="2026-03-12", fingerprints={"host_a": fp_stable}),
        ]
        drift = detect_voice_drift(history, "host_a", window=3)
        assert drift["centroid"] < 0.01


# ── Score trend tests ────────────────────────────────────────────────────────


class TestScoreTrend:
    """Tests for score_trend analysis."""

    def test_rising_scores_positive_slope(self):
        history = [_make_summary(date=f"2026-03-{10 + i}", score=0.5 + i * 0.1) for i in range(5)]
        trend = score_trend(history)
        assert trend["slope"] > 0
        assert trend["latest"] == 0.9
        assert trend["best"] == 0.9
        assert trend["worst"] == 0.5

    def test_flat_scores_zero_slope(self):
        history = [_make_summary(date=f"2026-03-{10 + i}", score=0.7) for i in range(5)]
        trend = score_trend(history)
        assert abs(trend["slope"]) < 0.001
        assert trend["std"] < 0.001

    def test_single_episode(self):
        history = [_make_summary(score=0.8)]
        trend = score_trend(history)
        assert trend["mean"] == 0.8
        assert trend["slope"] == 0.0

    def test_empty_history(self):
        trend = score_trend([])
        assert trend["mean"] == 0.0


# ── Pattern analysis tests ───────────────────────────────────────────────────


class TestPatternAnalysis:
    """Tests for find_effective_patterns."""

    def test_insufficient_data(self):
        history = [_make_summary() for _ in range(3)]
        result = find_effective_patterns(history, min_episodes=5)
        assert result["status"] == "insufficient_data"

    def test_correlated_feature(self):
        """Feature that increases with quality should have positive correlation."""
        history = []
        for i in range(10):
            history.append(
                _make_summary(
                    date=f"2026-03-{10 + i}",
                    score=0.5 + i * 0.05,
                    structure={"register_balance": 0.5 + i * 0.05},
                )
            )
        result = find_effective_patterns(history)
        assert "correlations" in result
        assert result["correlations"]["register_balance"] > 0.9

    def test_uncorrelated_feature(self):
        """Random structure values should have low correlation."""
        import random

        random.seed(42)
        history = []
        for i in range(20):
            history.append(
                _make_summary(
                    date=f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}",
                    score=0.7 + random.uniform(-0.1, 0.1),
                    structure={"random_dim": random.uniform(0.0, 1.0)},
                )
            )
        result = find_effective_patterns(history)
        if "random_dim" in result.get("correlations", {}):
            assert abs(result["correlations"]["random_dim"]) < 0.8


# ── Visualization tests ─────────────────────────────────────────────────────


class TestVisualization:
    """Tests for quality trend visualization."""

    def test_renders_png(self, tmp_path: Path):
        history = [
            _make_summary(date="2026-03-10", score=0.6),
            _make_summary(date="2026-03-11", score=0.7),
            _make_summary(date="2026-03-12", score=0.75),
        ]
        result = render_quality_trend(history, tmp_path)
        assert result is not None
        assert result.exists()
        assert result.suffix == ".png"
        assert result.stat().st_size > 1000

    def test_single_episode_returns_none(self, tmp_path: Path):
        history = [_make_summary()]
        result = render_quality_trend(history, tmp_path)
        assert result is None

    def test_empty_history_returns_none(self, tmp_path: Path):
        result = render_quality_trend([], tmp_path)
        assert result is None


# ── Serialization tests ──────────────────────────────────────────────────────


class TestSerialization:
    """Tests for EpisodeSummary serialization."""

    def test_to_json_parseable(self):
        summary = _make_summary()
        data = json.loads(summary.to_json())
        assert data["date"] == "2026-03-13"
        assert data["overall_score"] == 0.75

    def test_to_dict_has_all_fields(self):
        summary = _make_summary()
        d = summary.to_dict()
        expected_keys = {
            "date",
            "overall_score",
            "speaker_scores",
            "chemistry_score",
            "production_score",
            "script_score",
            "mean_dnsmos",
            "mean_wer",
            "segment_count",
            "total_duration_s",
            "voice_fingerprints",
            "script_structure",
        }
        assert set(d.keys()) == expected_keys
