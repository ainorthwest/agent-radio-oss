"""Tests for src/stt.py — whisper.cpp wrapper.

Tests are mock-based by default (patch subprocess.run) so they pass
without the whisper.cpp binary present. Integration tests that exercise
the real binary are gated on env var RADIO_WHISPER_BIN pointing to a
real build, and use pytest.mark.skipif when it's missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── transcribe() ─────────────────────────────────────────────────────────────


class TestTranscribe:
    """Tests for src.stt.transcribe()."""

    def test_returns_text_from_whisper_output(self, tmp_path: Path):
        """transcribe() should run whisper.cpp and return the transcribed text."""
        from src.stt import transcribe

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")  # placeholder; mocked subprocess won't read it

        # whisper.cpp -otxt writes a sibling file: audio.wav.txt
        expected_text = "hello world this is a test"

        def fake_run(cmd, **kwargs):
            # Find -of (output prefix) or fall back to writing alongside input
            of_prefix = None
            for i, arg in enumerate(cmd):
                if arg == "-of" and i + 1 < len(cmd):
                    of_prefix = Path(cmd[i + 1])
                    break
            assert of_prefix is not None, "stt.transcribe must use -of to control output path"
            (of_prefix.parent / f"{of_prefix.name}.txt").write_text(expected_text + "\n")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("src.stt.subprocess.run", side_effect=fake_run) as mock_run:
            text = transcribe(audio)

        assert text == expected_text
        assert mock_run.call_count == 1
        cmd = mock_run.call_args[0][0]
        assert any("whisper" in str(c).lower() for c in cmd), "command must invoke whisper binary"
        assert str(audio) in cmd, "command must include input audio path"

    def test_uses_env_binary_when_set(self, tmp_path: Path, monkeypatch):
        """RADIO_WHISPER_BIN env var should override the default binary path."""
        from src.stt import transcribe

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")
        custom_bin = "/opt/whisper/bin/main"
        monkeypatch.setenv("RADIO_WHISPER_BIN", custom_bin)

        def fake_run(cmd, **kwargs):
            of_prefix = next(Path(cmd[i + 1]) for i, a in enumerate(cmd) if a == "-of")
            (of_prefix.parent / f"{of_prefix.name}.txt").write_text("ok")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("src.stt.subprocess.run", side_effect=fake_run) as mock_run:
            transcribe(audio)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == custom_bin

    def test_uses_env_model_when_set(self, tmp_path: Path, monkeypatch):
        """RADIO_WHISPER_MODEL env var should override the default model path."""
        from src.stt import transcribe

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")
        custom_model = "/opt/models/ggml-medium.en.bin"
        monkeypatch.setenv("RADIO_WHISPER_MODEL", custom_model)

        def fake_run(cmd, **kwargs):
            of_prefix = next(Path(cmd[i + 1]) for i, a in enumerate(cmd) if a == "-of")
            (of_prefix.parent / f"{of_prefix.name}.txt").write_text("ok")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("src.stt.subprocess.run", side_effect=fake_run) as mock_run:
            transcribe(audio)

        cmd = mock_run.call_args[0][0]
        # whisper.cpp uses -m for model
        m_idx = cmd.index("-m")
        assert cmd[m_idx + 1] == custom_model

    def test_raises_when_binary_missing(self, tmp_path: Path, monkeypatch):
        """If the whisper binary is not available, raise WhisperUnavailableError."""
        from src.stt import WhisperUnavailableError, transcribe

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")
        monkeypatch.setenv("RADIO_WHISPER_BIN", "/nonexistent/whisper-cli")

        def fake_run(cmd, **kwargs):
            raise FileNotFoundError(f"No such file: {cmd[0]}")

        with patch("src.stt.subprocess.run", side_effect=fake_run):
            with pytest.raises(WhisperUnavailableError):
                transcribe(audio)

    def test_raises_when_audio_missing(self, tmp_path: Path):
        """Missing audio file should raise FileNotFoundError before invoking whisper."""
        from src.stt import transcribe

        with pytest.raises(FileNotFoundError):
            transcribe(tmp_path / "does-not-exist.wav")

    def test_propagates_nonzero_exit(self, tmp_path: Path):
        """Nonzero exit from whisper.cpp should raise WhisperError with stderr."""
        from src.stt import WhisperError, transcribe

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")

        def fake_run(cmd, **kwargs):
            return MagicMock(returncode=1, stdout="", stderr="model load failed")

        with patch("src.stt.subprocess.run", side_effect=fake_run):
            with pytest.raises(WhisperError, match="model load failed"):
                transcribe(audio)


# ── transcribe_with_timing() ─────────────────────────────────────────────────


SAMPLE_WHISPER_JSON = {
    "transcription": [
        {"timestamps": {"from": "00:00:00,000", "to": "00:00:00,420"}, "text": "Hello"},
        {"timestamps": {"from": "00:00:00,420", "to": "00:00:00,830"}, "text": " world"},
        {"timestamps": {"from": "00:00:00,830", "to": "00:00:01,200"}, "text": " this"},
        {"timestamps": {"from": "00:00:01,200", "to": "00:00:01,580"}, "text": " is"},
        {"timestamps": {"from": "00:00:01,580", "to": "00:00:02,100"}, "text": " a"},
        {"timestamps": {"from": "00:00:02,100", "to": "00:00:02,750"}, "text": " test"},
    ]
}


def _fake_run_with_json(payload: dict):
    """Build a fake subprocess.run that drops `<prefix>.json` next to -of."""

    def fake_run(cmd, **kwargs):
        of_prefix = next(Path(cmd[i + 1]) for i, a in enumerate(cmd) if a == "-of")
        (of_prefix.parent / f"{of_prefix.name}.json").write_text(json.dumps(payload))
        return MagicMock(returncode=0, stdout="", stderr="")

    return fake_run


class TestTranscribeWithTiming:
    """Tests for src.stt.transcribe_with_timing()."""

    def test_returns_word_segments(self, tmp_path: Path):
        """transcribe_with_timing returns a list of WordSegment with start/end seconds."""
        from src.stt import WordSegment, transcribe_with_timing

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")

        with patch("src.stt.subprocess.run", side_effect=_fake_run_with_json(SAMPLE_WHISPER_JSON)):
            segments = transcribe_with_timing(audio)

        assert len(segments) == 6
        assert all(isinstance(s, WordSegment) for s in segments)
        assert segments[0].text == "Hello"
        assert segments[0].start == 0.0
        assert abs(segments[0].end - 0.420) < 1e-6
        assert segments[5].text == "test"  # leading whitespace stripped
        assert abs(segments[5].end - 2.750) < 1e-6

    def test_uses_max_len_one_for_word_level(self, tmp_path: Path):
        """The whisper command must include --max-len 1 to force word-level output."""
        from src.stt import transcribe_with_timing

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")

        with patch(
            "src.stt.subprocess.run", side_effect=_fake_run_with_json(SAMPLE_WHISPER_JSON)
        ) as mock_run:
            transcribe_with_timing(audio)

        cmd = mock_run.call_args[0][0]
        assert "-ml" in cmd or "--max-len" in cmd
        ml_flag = "-ml" if "-ml" in cmd else "--max-len"
        ml_idx = cmd.index(ml_flag)
        assert cmd[ml_idx + 1] == "1"

    def test_emits_json_output(self, tmp_path: Path):
        """The whisper command must request JSON output (-oj / --output-json)."""
        from src.stt import transcribe_with_timing

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")

        with patch(
            "src.stt.subprocess.run", side_effect=_fake_run_with_json(SAMPLE_WHISPER_JSON)
        ) as mock_run:
            transcribe_with_timing(audio)

        cmd = mock_run.call_args[0][0]
        assert "-oj" in cmd or "--output-json" in cmd

    def test_empty_transcription_returns_empty_list(self, tmp_path: Path):
        """Whisper output with no segments should return []."""
        from src.stt import transcribe_with_timing

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")

        with patch(
            "src.stt.subprocess.run",
            side_effect=_fake_run_with_json({"transcription": []}),
        ):
            segments = transcribe_with_timing(audio)

        assert segments == []


class TestSrtExport:
    """Tests for src.stt.write_srt()."""

    def test_writes_well_formed_srt(self, tmp_path: Path):
        """write_srt should produce a sequentially-numbered SRT block per segment."""
        from src.stt import WordSegment, write_srt

        segments = [
            WordSegment(text="Hello", start=0.0, end=0.5),
            WordSegment(text="world", start=0.5, end=1.0),
        ]
        out = tmp_path / "audio.srt"
        write_srt(segments, out)

        body = out.read_text()
        # SRT format: index, time range, text, blank line
        assert body.startswith("1\n")
        assert "00:00:00,000 --> 00:00:00,500" in body
        assert "Hello" in body
        assert "\n2\n" in body
        assert "00:00:00,500 --> 00:00:01,000" in body
        assert "world" in body

    def test_empty_list_writes_empty_file(self, tmp_path: Path):
        """No segments → empty SRT file (still created)."""
        from src.stt import write_srt

        out = tmp_path / "empty.srt"
        write_srt([], out)

        assert out.exists()
        assert out.read_text() == ""


# ── wer() ────────────────────────────────────────────────────────────────────


class TestWer:
    """Tests for src.stt.wer() — word error rate."""

    def test_identical_strings_score_zero(self):
        from src.stt import wer

        assert wer("hello world this is a test", "hello world this is a test") == 0.0

    def test_completely_different_scores_one(self):
        from src.stt import wer

        # Reference has 4 words; hypothesis replaces all 4 → WER 1.0
        assert wer("alpha bravo charlie delta", "one two three four") == 1.0

    def test_one_substitution_in_four_words(self):
        from src.stt import wer

        # 1 substitution / 4 reference words = 0.25
        result = wer("the quick brown fox", "the quick red fox")
        assert abs(result - 0.25) < 1e-6

    def test_one_deletion(self):
        from src.stt import wer

        # Reference 5 words, hypothesis missing one → 1 deletion / 5 = 0.2
        result = wer("the quick brown fox jumps", "the quick brown fox")
        assert abs(result - 0.2) < 1e-6

    def test_one_insertion(self):
        from src.stt import wer

        # Reference 4 words, hypothesis adds one → 1 insertion / 4 = 0.25
        result = wer("the quick brown fox", "the very quick brown fox")
        assert abs(result - 0.25) < 1e-6

    def test_case_insensitive(self):
        from src.stt import wer

        assert wer("Hello World", "hello world") == 0.0

    def test_punctuation_ignored(self):
        from src.stt import wer

        assert wer("hello, world.", "hello world") == 0.0

    def test_empty_reference_returns_zero_when_hyp_also_empty(self):
        from src.stt import wer

        assert wer("", "") == 0.0

    def test_empty_reference_with_nonempty_hyp_returns_one(self):
        """Convention: when reference is empty but hypothesis has words,
        report WER as 1.0 (everything in hypothesis is an error)."""
        from src.stt import wer

        assert wer("", "extra noise") == 1.0

    def test_strips_bracket_tags(self):
        """Non-speech tags like [laugh] (sigh) should be ignored before WER."""
        from src.stt import wer

        assert wer("[laugh] hello (sighs) world", "hello world") == 0.0


# ── cer() ────────────────────────────────────────────────────────────────────


class TestCer:
    """Character Error Rate."""

    def test_identical_returns_zero(self):
        from src.stt import cer

        assert cer("hello world", "hello world") == 0.0

    def test_one_substitution_in_five_chars(self):
        from src.stt import cer

        # "hello" vs "hellp" → 1 sub / 5 chars = 0.2
        result = cer("hello", "hellp")
        assert abs(result - 0.2) < 1e-6

    def test_case_insensitive_and_strips_tags(self):
        from src.stt import cer

        assert cer("[laugh] Hello", "hello") == 0.0


# ── round_trip_score() ───────────────────────────────────────────────────────


class TestRoundTripScore:
    """Tests for src.stt.round_trip_score() — per-segment WER report."""

    def test_returns_per_segment_wer(self, tmp_path: Path):
        """round_trip_score should produce a per-segment WER list and an overall WER."""
        from src.stt import RoundTripReport, round_trip_score

        # Mock: each segment audio file gets transcribed exactly to its expected
        # text. Identity ⇒ WER = 0 for all segments.
        seg_a = tmp_path / "seg-000-host_a.wav"
        seg_b = tmp_path / "seg-001-host_b.wav"
        seg_a.write_bytes(b"\x00")
        seg_b.write_bytes(b"\x00")

        segments = [
            {"index": 0, "speaker": "host_a", "text": "the quick brown fox", "audio_path": seg_a},
            {
                "index": 1,
                "speaker": "host_b",
                "text": "jumps over the lazy dog",
                "audio_path": seg_b,
            },
        ]

        # Mock subprocess to drop a `.txt` file with the segment's expected text.
        def fake_run(cmd, **kwargs):
            of_prefix = next(Path(cmd[i + 1]) for i, a in enumerate(cmd) if a == "-of")
            audio_arg = next(cmd[i + 1] for i, a in enumerate(cmd) if a == "-f")
            seg = next(s for s in segments if str(s["audio_path"]) == audio_arg)
            (of_prefix.parent / f"{of_prefix.name}.txt").write_text(seg["text"])
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("src.stt.subprocess.run", side_effect=fake_run):
            report = round_trip_score(segments)

        assert isinstance(report, RoundTripReport)
        assert len(report.per_segment) == 2
        assert all(s["wer"] == 0.0 for s in report.per_segment)
        assert report.overall_wer == 0.0
        assert report.outliers == []

    def test_flags_outlier_segments(self, tmp_path: Path):
        """Segments with WER >2× median should be flagged as outliers."""
        from src.stt import round_trip_score

        segments = []
        for i, (text, transcript) in enumerate(
            [
                ("the quick brown fox", "the quick brown fox"),  # 0.0
                ("hello world test", "hello world test"),  # 0.0
                ("this is fine", "this is fine"),  # 0.0
                # Outlier: total mismatch
                ("alpha bravo charlie", "completely different words here"),  # 1.0
            ]
        ):
            audio = tmp_path / f"seg-{i:03d}.wav"
            audio.write_bytes(b"\x00")
            segments.append(
                {
                    "index": i,
                    "speaker": "host_a",
                    "text": text,
                    "audio_path": audio,
                    "transcript": transcript,
                }
            )

        def fake_run(cmd, **kwargs):
            of_prefix = next(Path(cmd[i + 1]) for i, a in enumerate(cmd) if a == "-of")
            audio_arg = next(cmd[i + 1] for i, a in enumerate(cmd) if a == "-f")
            seg = next(s for s in segments if str(s["audio_path"]) == audio_arg)
            (of_prefix.parent / f"{of_prefix.name}.txt").write_text(seg["transcript"])
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("src.stt.subprocess.run", side_effect=fake_run):
            report = round_trip_score(segments)

        assert len(report.outliers) == 1
        assert report.outliers[0] == 3  # the bad segment

    def test_to_json_serializable(self, tmp_path: Path):
        """RoundTripReport.to_dict() should be JSON-serializable."""
        from src.stt import round_trip_score

        seg = tmp_path / "seg.wav"
        seg.write_bytes(b"\x00")
        segments = [{"index": 0, "speaker": "host_a", "text": "hello world", "audio_path": seg}]

        def fake_run(cmd, **kwargs):
            of_prefix = next(Path(cmd[i + 1]) for i, a in enumerate(cmd) if a == "-of")
            (of_prefix.parent / f"{of_prefix.name}.txt").write_text("hello world")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("src.stt.subprocess.run", side_effect=fake_run):
            report = round_trip_score(segments)

        # Should round-trip through JSON without TypeError
        json.dumps(report.to_dict())


# ── transcribe_for_corpus() ──────────────────────────────────────────────────


class TestTranscribeForCorpus:
    """Tests for src.stt.transcribe_for_corpus() — corpus-clean transcript."""

    def test_strips_punctuation_and_lowercases(self, tmp_path: Path):
        """Corpus output: lowercase, no punctuation, single-spaced."""
        from src.stt import transcribe_for_corpus

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")

        def fake_run(cmd, **kwargs):
            of_prefix = next(Path(cmd[i + 1]) for i, a in enumerate(cmd) if a == "-of")
            (of_prefix.parent / f"{of_prefix.name}.txt").write_text(
                "Hello, World! This is a TEST.  Multiple   spaces."
            )
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("src.stt.subprocess.run", side_effect=fake_run):
            corpus = transcribe_for_corpus(audio)

        assert corpus == "hello world this is a test multiple spaces"


# ── integration smoke test (only when binary is built) ───────────────────────


WHISPER_BIN = os.environ.get("RADIO_WHISPER_BIN")
WHISPER_MODEL = os.environ.get("RADIO_WHISPER_MODEL")


@pytest.mark.skipif(
    not (WHISPER_BIN and WHISPER_MODEL and Path(WHISPER_BIN).exists()),
    reason="whisper.cpp binary not configured (set RADIO_WHISPER_BIN + RADIO_WHISPER_MODEL)",
)
class TestTranscribeIntegration:
    """Real whisper.cpp smoke test — gated on env vars."""

    def test_transcribes_silence_to_empty_or_short(self, tmp_path: Path):
        """A silent WAV should transcribe to empty string or a very short result."""
        import numpy as np
        import soundfile as sf

        from src.stt import transcribe

        sr = 16000
        audio_path = tmp_path / "silence.wav"
        sf.write(str(audio_path), np.zeros(sr * 2, dtype=np.float32), sr)
        text = transcribe(audio_path)
        # whisper sometimes emits "[BLANK_AUDIO]" or punctuation on silence
        assert isinstance(text, str)
        assert len(text) < 100
