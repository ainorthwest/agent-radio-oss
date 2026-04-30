"""whisper.cpp wrapper — speech-to-text for Agent Radio OSS.

The OSS distribution uses whisper.cpp (MIT, hardware-portable via compile
flags) instead of mlx-whisper. The binary is built per platform:

    Apple Silicon:  make GGML_METAL=1
    AMD ROCm:       make GGML_HIPBLAS=1   (or GGML_VULKAN=1 fallback)
    NVIDIA CUDA:    make GGML_CUDA=1
    CPU:            make

Configuration via env vars (no hard-coded paths):

    RADIO_WHISPER_BIN     — path to whisper.cpp ``main``/``whisper-cli`` binary
                            (default: ``./whisper.cpp/main``)
    RADIO_WHISPER_MODEL   — path to ggml model
                            (default: ``./models/ggml-base.en.bin``)
    RADIO_WHISPER_THREADS — thread count (default: 4)

This module is intentionally boring — ``subprocess.run`` against the
binary, parse the output files. No fancy bindings, no shell-injection
surface (``shell=False``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_BIN = "./whisper.cpp/build/bin/whisper-cli"
DEFAULT_MODEL = "./models/ggml-base.en.bin"
DEFAULT_THREADS = "4"


class WhisperUnavailableError(RuntimeError):
    """Raised when the whisper.cpp binary is not on disk or not executable."""


class WhisperError(RuntimeError):
    """Raised when whisper.cpp exits non-zero. Wraps stderr."""


@dataclass(frozen=True)
class WordSegment:
    """One word (or short token) with start/end seconds."""

    text: str
    start: float
    end: float


@dataclass
class RoundTripReport:
    """Per-segment round-trip WER report.

    ``per_segment`` is a list of dicts: ``{index, speaker, wer, expected, transcribed}``.
    ``outliers`` lists segment indices whose WER is >2× the episode median.
    ``overall_wer`` is the WER computed over the concatenated reference vs.
    the concatenated hypothesis (single number for the whole episode).
    """

    per_segment: list[dict[str, Any]]
    overall_wer: float
    outliers: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_segment": self.per_segment,
            "overall_wer": self.overall_wer,
            "outliers": self.outliers,
        }


def _binary() -> str:
    return os.environ.get("RADIO_WHISPER_BIN", DEFAULT_BIN)


def _model() -> str:
    return os.environ.get("RADIO_WHISPER_MODEL", DEFAULT_MODEL)


def _threads() -> str:
    return os.environ.get("RADIO_WHISPER_THREADS", DEFAULT_THREADS)


def transcribe(audio_path: Path) -> str:
    """Transcribe an audio file to plain text.

    Shells out to whisper.cpp with ``-otxt`` (plain-text output). Returns
    the transcript as a single string with surrounding whitespace stripped.

    Raises:
        FileNotFoundError: if ``audio_path`` does not exist.
        WhisperUnavailableError: if the whisper binary is missing.
        WhisperError: if whisper exits non-zero.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    # whisper.cpp writes alongside `-of <prefix>` as `<prefix>.txt` (or .srt, .json).
    # Use the audio path's stem in its own directory for the prefix.
    of_prefix = audio_path.with_suffix("")
    cmd = [
        _binary(),
        "-m",
        _model(),
        "-f",
        str(audio_path),
        "-otxt",
        "-of",
        str(of_prefix),
        "-t",
        _threads(),
        "-l",
        "en",
        "--no-prints",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise WhisperUnavailableError(
            f"whisper.cpp binary not found at {cmd[0]}; set RADIO_WHISPER_BIN"
        ) from exc

    if proc.returncode != 0:
        raise WhisperError(
            f"whisper.cpp failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    txt_path = of_prefix.parent / f"{of_prefix.name}.txt"
    if not txt_path.exists():
        raise WhisperError(f"whisper.cpp did not produce expected output: {txt_path}")
    return txt_path.read_text().strip()


def transcribe_with_timing(audio_path: Path) -> list[WordSegment]:
    """Transcribe with word-level timestamps.

    Uses whisper.cpp ``--max-len 1`` (one token per segment) plus
    ``--output-json`` to produce a list of ``WordSegment`` with start/end
    in seconds. Suitable for SRT export and per-word alignment.

    Raises the same exceptions as :func:`transcribe`.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    of_prefix = audio_path.with_suffix("")
    cmd = [
        _binary(),
        "-m",
        _model(),
        "-f",
        str(audio_path),
        "-oj",
        "-of",
        str(of_prefix),
        "-ml",
        "1",
        "-t",
        _threads(),
        "-l",
        "en",
        "--no-prints",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise WhisperUnavailableError(
            f"whisper.cpp binary not found at {cmd[0]}; set RADIO_WHISPER_BIN"
        ) from exc

    if proc.returncode != 0:
        raise WhisperError(
            f"whisper.cpp failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    json_path = of_prefix.parent / f"{of_prefix.name}.json"
    if not json_path.exists():
        raise WhisperError(f"whisper.cpp did not produce expected output: {json_path}")

    payload = json.loads(json_path.read_text())
    items = payload.get("transcription", [])
    return [
        WordSegment(
            text=item.get("text", "").strip(),
            start=_parse_timestamp(item["timestamps"]["from"]),
            end=_parse_timestamp(item["timestamps"]["to"]),
        )
        for item in items
    ]


def _parse_timestamp(ts: str) -> float:
    """Parse whisper.cpp ``HH:MM:SS,mmm`` into seconds."""
    hms, ms = ts.split(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _format_timestamp(seconds: float) -> str:
    """Format seconds back to ``HH:MM:SS,mmm`` for SRT output."""
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


_TAG_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_PUNCT_RE = re.compile(r"[^\w\s']")


def _normalize(text: str) -> list[str]:
    """Lowercase, strip non-speech tags + punctuation, split on whitespace."""
    text = _TAG_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return text.lower().split()


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate between reference and hypothesis.

    Computes Levenshtein distance over normalized word tokens, divided
    by the reference word count. Conventions:

    * Both empty → 0.0
    * Reference empty, hypothesis non-empty → 1.0 (every hypothesis
      word is an insertion error against an empty reference; we cap at
      1.0 to keep the metric bounded for downstream gates)
    * Otherwise → ``edit_distance / len(ref_tokens)`` (capped at 1.0
      for severe over-insertion cases)
    """
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    distance = _levenshtein(ref, hyp)
    return min(1.0, distance / len(ref))


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate.

    Same shape as :func:`wer` but operates over characters of the
    normalized text. Whitespace between tokens is preserved as a single
    space; bracket tags and punctuation are stripped first.
    """
    ref = " ".join(_normalize(reference))
    hyp = " ".join(_normalize(hypothesis))
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    distance = _levenshtein(list(ref), list(hyp))
    return min(1.0, distance / len(ref))


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance over token sequences (substitution/insertion/deletion cost 1).

    Works for both lists of words and lists of single characters.
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Two-row DP table, rolling.
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            curr[j] = min(
                curr[j - 1] + 1,  # insertion
                prev[j] + 1,  # deletion
                prev[j - 1] + cost,  # substitution / match
            )
        prev = curr
    return prev[-1]


def write_srt(segments: list[WordSegment], out_path: Path) -> None:
    """Write a SubRip (SRT) file from word segments.

    Each segment becomes one numbered cue. An empty list produces an
    empty file (still created, so callers can rely on file existence).
    """
    out_path = Path(out_path)
    if not segments:
        out_path.write_text("")
        return
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{_format_timestamp(seg.start)} --> {_format_timestamp(seg.end)}\n{seg.text}\n"
        )
    out_path.write_text("\n".join(blocks))


def transcribe_for_corpus(audio_path: Path) -> str:
    """Transcribe and emit a corpus-clean string.

    Lowercase, punctuation stripped, single-spaced. Useful for downstream
    voice-training corpora where mixed-case/punct introduces noise.
    """
    raw = transcribe(audio_path)
    return " ".join(_normalize(raw))


def round_trip_score(segments: list[dict[str, Any]]) -> RoundTripReport:
    """Per-segment round-trip WER for an episode.

    Each segment dict must include ``index``, ``speaker``, ``text``, and
    ``audio_path``. Each audio file is transcribed via whisper.cpp; the
    transcription is compared to the segment's text via :func:`wer`.

    Outliers are segments whose WER is more than 2× the episode median
    (and >0.05 absolute) — they're candidates for re-render or text
    rewrite.
    """
    per_segment: list[dict[str, Any]] = []
    ref_concat: list[str] = []
    hyp_concat: list[str] = []
    for seg in segments:
        text = str(seg.get("text", ""))
        audio = Path(seg["audio_path"])
        transcribed = transcribe(audio)
        score = wer(text, transcribed)
        per_segment.append(
            {
                "index": seg.get("index"),
                "speaker": seg.get("speaker", ""),
                "wer": score,
                "expected": text,
                "transcribed": transcribed,
            }
        )
        ref_concat.append(text)
        hyp_concat.append(transcribed)

    overall = wer(" ".join(ref_concat), " ".join(hyp_concat))

    # Outliers: WER > 2× median AND > 0.05 absolute (otherwise tiny
    # medians make trivial differences look like outliers)
    wers = sorted(s["wer"] for s in per_segment)
    if wers:
        mid = len(wers) // 2
        median = wers[mid] if len(wers) % 2 else (wers[mid - 1] + wers[mid]) / 2
    else:
        median = 0.0
    outliers = [s["index"] for s in per_segment if s["wer"] > max(0.05, 2 * median)]
    return RoundTripReport(per_segment=per_segment, overall_wer=overall, outliers=outliers)
