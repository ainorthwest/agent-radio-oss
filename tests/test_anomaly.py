"""Tests for src/anomaly.py — post-render segment-level anomaly detector.

The anomaly detector runs after render, before the editorial decision
(ship/review/reject). It scans per-segment WAVs + the whisper transcript
for three classes of failure:

1. Silence-rate spike (TTS dropout)
2. WER outlier (mispronunciation, hallucinated word)
3. Duration anomaly (truncation, runaway generation)

Output is an AnomalyReport with a list of flagged segments and one of
three suggested actions: regenerate, replace_text, manual_review.

Tests use synthetic audio + mocked whisper transcription so they pass
without TTS or whisper.cpp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_wav(path: Path, duration_s: float, sr: int = 24000, silence: bool = False) -> None:
    """Helper: write a WAV with either speech-like noise or silence."""
    import soundfile as sf

    n = int(duration_s * sr)
    if silence:
        samples = np.zeros(n, dtype=np.float32)
    else:
        rng = np.random.RandomState(42)
        samples = (rng.randn(n) * 0.1).astype(np.float32)
    sf.write(str(path), samples, sr)


# ── silence-rate check ──────────────────────────────────────────────────────


class TestSilenceCheck:
    def test_clean_segment_no_anomaly(self, tmp_path: Path):
        from src.anomaly import check_silence

        wav = tmp_path / "seg.wav"
        _write_wav(wav, 3.0)
        result = check_silence(wav)
        assert result is None  # no anomaly

    def test_silent_segment_flagged(self, tmp_path: Path):
        from src.anomaly import check_silence

        wav = tmp_path / "seg.wav"
        _write_wav(wav, 3.0, silence=True)
        result = check_silence(wav)
        assert result is not None
        assert result["check"] == "silence"
        assert result["severity"] in {"high", "medium"}
        assert "suggestion" in result

    def test_partial_silence_above_threshold(self, tmp_path: Path):
        """A segment that is >40% silence (default threshold) should flag."""
        from src.anomaly import check_silence

        # Build: 1s of silence + 1s of speech, 50% silent.
        sr = 24000
        speech = np.random.RandomState(0).randn(sr).astype(np.float32) * 0.1
        signal = np.concatenate([np.zeros(sr, dtype=np.float32), speech])
        wav = tmp_path / "seg.wav"
        import soundfile as sf

        sf.write(str(wav), signal, sr)

        result = check_silence(wav, threshold=0.4)
        assert result is not None
        assert result["check"] == "silence"


# ── WER outlier check ───────────────────────────────────────────────────────


class TestWerOutlierCheck:
    def test_no_outliers_returns_empty(self):
        from src.anomaly import check_wer_outliers

        per_segment_wer = [
            {"index": 0, "wer": 0.0},
            {"index": 1, "wer": 0.05},
            {"index": 2, "wer": 0.03},
            {"index": 3, "wer": 0.02},
        ]
        result = check_wer_outliers(per_segment_wer)
        assert result == []

    def test_one_high_wer_segment_flagged(self):
        from src.anomaly import check_wer_outliers

        per_segment_wer = [
            {"index": 0, "wer": 0.0},
            {"index": 1, "wer": 0.0},
            {"index": 2, "wer": 0.0},
            {"index": 3, "wer": 0.5},  # outlier: median is 0, this is way above
        ]
        result = check_wer_outliers(per_segment_wer)
        assert len(result) == 1
        assert result[0]["index"] == 3
        assert result[0]["check"] == "wer_outlier"
        assert result[0]["wer"] == 0.5

    def test_threshold_floors_tiny_medians(self):
        """A median of 0.0 alone should not flag every segment with WER>0 —
        we require WER > max(0.05, 2*median) to avoid noise."""
        from src.anomaly import check_wer_outliers

        per_segment_wer = [
            {"index": 0, "wer": 0.0},
            {"index": 1, "wer": 0.04},  # >0 but below 0.05 floor
        ]
        result = check_wer_outliers(per_segment_wer)
        assert result == []

    def test_empty_input(self):
        from src.anomaly import check_wer_outliers

        assert check_wer_outliers([]) == []

    def test_handles_sentinel_wer(self):
        """When WER is -1 (not computed), skip it from the median calculation."""
        from src.anomaly import check_wer_outliers

        per_segment_wer = [
            {"index": 0, "wer": -1.0},  # not computed
            {"index": 1, "wer": -1.0},
            {"index": 2, "wer": 0.5},
        ]
        # With only one valid datapoint, no comparison is possible — no flag
        result = check_wer_outliers(per_segment_wer)
        assert result == []


# ── duration anomaly check ──────────────────────────────────────────────────


class TestDurationCheck:
    def test_in_range_duration_no_anomaly(self):
        from src.anomaly import check_duration

        # ~10 words, expected ~3-5s; actual 3.5s => fine
        result = check_duration(
            text="one two three four five six seven eight nine ten", actual_duration=3.5
        )
        assert result is None

    def test_too_short_flagged(self):
        from src.anomaly import check_duration

        # 10 words; 0.3s is way too short
        result = check_duration(
            text="one two three four five six seven eight nine ten", actual_duration=0.3
        )
        assert result is not None
        assert result["check"] == "duration"
        assert "short" in result["severity"].lower() or "short" in str(result).lower()

    def test_too_long_flagged(self):
        from src.anomaly import check_duration

        # 10 words; 30s is runaway generation
        result = check_duration(
            text="one two three four five six seven eight nine ten", actual_duration=30.0
        )
        assert result is not None
        assert result["check"] == "duration"

    def test_empty_text_handled(self):
        from src.anomaly import check_duration

        result = check_duration(text="", actual_duration=0.0)
        assert result is None  # nothing to compare


# ── orchestrator: detect_anomalies ───────────────────────────────────────────


class TestDetectAnomalies:
    def test_clean_episode_returns_empty_report(self, tmp_path: Path):
        from src.anomaly import AnomalyReport, detect_anomalies

        # Build manifest with one clean segment
        seg_path = tmp_path / "seg-000-host_a.wav"
        _write_wav(seg_path, duration_s=3.0)

        manifest = {
            "segments": [
                {
                    "index": 0,
                    "file": "seg-000-host_a.wav",
                    "speaker": "host_a",
                    "text": "one two three four five six seven eight nine ten",
                    "duration_seconds": 3.0,
                }
            ],
            "segments_dir": str(tmp_path),
        }

        report = detect_anomalies(manifest, per_segment_wer=[])
        assert isinstance(report, AnomalyReport)
        assert report.anomalies == []

    def test_silent_segment_in_episode_flagged(self, tmp_path: Path):
        from src.anomaly import detect_anomalies

        seg_path = tmp_path / "seg-000-host_a.wav"
        _write_wav(seg_path, duration_s=3.0, silence=True)

        manifest = {
            "segments": [
                {
                    "index": 0,
                    "file": "seg-000-host_a.wav",
                    "speaker": "host_a",
                    "text": "one two three four five",
                    "duration_seconds": 3.0,
                }
            ],
            "segments_dir": str(tmp_path),
        }

        report = detect_anomalies(manifest, per_segment_wer=[])
        assert len(report.anomalies) >= 1
        assert any(a["check"] == "silence" for a in report.anomalies)
        assert all(a["index"] == 0 for a in report.anomalies)

    def test_wer_outlier_with_real_wer_input(self, tmp_path: Path):
        from src.anomaly import detect_anomalies

        # Two clean segments + one with bad WER
        for i in range(3):
            _write_wav(tmp_path / f"seg-{i:03d}-host_a.wav", duration_s=2.0)

        manifest = {
            "segments": [
                {
                    "index": i,
                    "file": f"seg-{i:03d}-host_a.wav",
                    "speaker": "host_a",
                    "text": "one two three",
                    "duration_seconds": 2.0,
                }
                for i in range(3)
            ],
            "segments_dir": str(tmp_path),
        }
        per_seg_wer = [
            {"index": 0, "wer": 0.0},
            {"index": 1, "wer": 0.0},
            {"index": 2, "wer": 0.6},
        ]

        report = detect_anomalies(manifest, per_segment_wer=per_seg_wer)
        wer_anomalies = [a for a in report.anomalies if a["check"] == "wer_outlier"]
        assert len(wer_anomalies) == 1
        assert wer_anomalies[0]["index"] == 2

    def test_to_json_serializable(self, tmp_path: Path):
        import json

        from src.anomaly import AnomalyReport

        report = AnomalyReport(
            anomalies=[
                {
                    "index": 0,
                    "check": "silence",
                    "severity": "high",
                    "suggestion": "regenerate",
                    "ratio": 0.85,
                },
                {
                    "index": 2,
                    "check": "wer_outlier",
                    "severity": "medium",
                    "suggestion": "replace_text",
                    "wer": 0.6,
                },
            ]
        )
        s = json.dumps(report.to_dict())
        parsed = json.loads(s)
        assert len(parsed["anomalies"]) == 2

    def test_handles_missing_wav_gracefully(self, tmp_path: Path):
        """If a WAV file referenced in the manifest is missing, log and skip
        rather than crash the whole report."""
        from src.anomaly import detect_anomalies

        manifest = {
            "segments": [
                {
                    "index": 0,
                    "file": "missing.wav",
                    "speaker": "host_a",
                    "text": "x",
                    "duration_seconds": 1.0,
                }
            ],
            "segments_dir": str(tmp_path),
        }
        # Should not raise; should produce no silence anomaly (file missing)
        report = detect_anomalies(manifest, per_segment_wer=[])
        # Either skip silently or include a manual_review entry — accept either
        assert isinstance(report.anomalies, list)
