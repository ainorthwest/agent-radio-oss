"""Quality evaluation: spectral + perceived quality + intelligibility analysis.

Three-pillar evaluation:
  Pillar 1 — Signal analysis via librosa (spectral features, prosody metrics)
  Pillar 2 — Perceived quality via torchmetrics[audio] (DNSMOS, SRMR, PESQ, STOI)
  Pillar 3 — Intelligibility via mlx-whisper + jiwer (WER/CER round-trip)

OSS scope: Pillars 1 and 2 ship in v0.1.0-mvp. Pillar 3 is stubbed —
``_compute_intelligibility`` gracefully returns sentinel ``-1.0`` values
when ``mlx_whisper`` isn't installed (it isn't in the OSS extras). The
real OSS WER pillar lands on Day 3 of the MVP sprint, wired through
whisper.cpp via subprocess. Until then, ``wer`` and ``cer`` fields in
quality reports stay at ``-1.0``.

DNSMOS and SRMR are reference-free (no clean target needed). PESQ and STOI
require reference audio and are only computed when available.

Requires: uv sync --extra quality

Standalone:
    uv run python -m src.quality output/episode.mp3
    uv run python -m src.quality output/audition.wav --engine kokoro
    uv run python -m src.quality --manifest manifest.json --script script.json
    uv run python -m src.quality --build-reference samples/*.wav --engine kokoro
    uv run python -m src.quality --build-reference samples/*.wav -o config/quality-reference.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Known TTS engines for reference profile resolution.
# OSS distribution ships only Kokoro; the broader engine set lives in
# the proprietary agent-radio repo.
KNOWN_ENGINES = frozenset({"kokoro"})

# Default directory for engine-specific reference profiles
REFERENCE_DIR = Path("config")


def _resolve_engine_reference(engine: str) -> Path:
    """Resolve engine name to its reference profile path.

    Returns config/quality-reference-{engine}.json.
    Raises ValueError for unknown engines.
    """
    if engine not in KNOWN_ENGINES:
        raise ValueError(f"Unknown engine '{engine}' — expected one of {sorted(KNOWN_ENGINES)}")
    return REFERENCE_DIR / f"quality-reference-{engine}.json"


@dataclass
class QualityReport:
    """Structured quality report for a rendered audio file."""

    overall_score: float = 0.0
    dynamic_range_lufs: float = 0.0
    silence_ratio: float = 0.0
    spectral_centroid_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0
    zcr_mean: float = 0.0
    mfcc_distance: float = 0.0
    pitch_variance: float = 0.0
    pitch_range_normalized: float = 0.0
    pitch_contour_smoothness: float = 0.0
    speech_rate_variation: float = 0.0
    pause_naturalness: float = 0.0
    syllable_duration_variance: float = 0.0
    duration_seconds: float = 0.0
    # Perceived quality (torchmetrics) — Pillar 2
    dnsmos_ovr: float = 0.0  # DNSMOS overall MOS (1-5 scale)
    dnsmos_sig: float = 0.0  # DNSMOS signal quality (1-5)
    dnsmos_bak: float = 0.0  # DNSMOS background quality (1-5)
    dnsmos_p808: float = 0.0  # DNSMOS P.808 MOS prediction (1-5)
    srmr: float = 0.0  # Speech Reverberation Modulation Energy Ratio (higher = drier)
    pesq: float = 0.0  # PESQ score (1-5, reference-based, 0 if unavailable)
    stoi: float = 0.0  # STOI score (0-1, reference-based, 0 if unavailable)
    # Intelligibility (mlx-whisper + jiwer) — Pillar 3
    wer: float = -1.0  # Word Error Rate (0=perfect, 1=all wrong, -1=not computed)
    cer: float = -1.0  # Character Error Rate (0=perfect, 1=all wrong, -1=not computed)
    # Artifact detection — TTS failure modes
    artifact_count: int = 0  # Total detected artifacts (clipping + spikes + repetitions)
    clipping_frames: int = 0  # Consecutive-sample clipping events
    spectral_spikes: int = 0  # Energy spikes > 3σ from local mean (clicks/pops)
    repetition_score: float = 0.0  # MFCC self-similarity (0=none, 1=exact loop)
    snr_db: float = 0.0  # Signal-to-noise ratio (voiced / unvoiced energy)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _compute_prosody(sr: int, voiced_f0: Any, rms: Any, rms_db: Any) -> dict[str, float]:
    """Extract prosody-specific metrics from audio.

    These metrics measure what the human ear detects as "natural" vs "robotic":
    pitch movement, speech rate variation, pause structure, and syllable timing.
    All computed from already-extracted arrays to avoid redundant processing.
    """
    import numpy as np

    # ── Pitch range normalized ───────────────────────────────────────────────
    # (max - min) / median — dimensionless, comparable across voice registers.
    # Natural radio speech: 0.3–1.2. Below = flat. Above = erratic.
    if len(voiced_f0) > 2:
        median_f0 = float(np.median(voiced_f0))
        if median_f0 > 0:
            pitch_range_normalized = float((np.max(voiced_f0) - np.min(voiced_f0)) / median_f0)
        else:
            pitch_range_normalized = 0.0
    else:
        pitch_range_normalized = 0.0

    # ── Pitch contour smoothness ─────────────────────────────────────────────
    # Mean absolute F0 derivative (Hz per frame). Lower = smoother transitions.
    # Abrupt jumps between frames indicate robotic pitch stepping.
    if len(voiced_f0) > 2:
        f0_diff = np.abs(np.diff(voiced_f0))
        pitch_contour_smoothness = float(np.mean(f0_diff))
    else:
        pitch_contour_smoothness = 0.0

    # ── Speech rate variation ────────────────────────────────────────────────
    # Variance of energy onset density across 500ms windows.
    # Natural speech speeds up and slows down. Robotic TTS is metronomic.
    hop_length = 512
    frame_duration = hop_length / sr
    window_frames = max(1, int(0.5 / frame_duration))  # 500ms windows

    # Use RMS energy onsets as proxy for syllable boundaries
    rms_flat = rms.flatten() if rms.ndim > 1 else rms
    rms_diff = np.diff(rms_flat)
    onsets = (rms_diff > 0).astype(float)  # rising energy = onset

    # Count onsets per window
    n_windows = max(1, len(onsets) // window_frames)
    if n_windows > 1:
        onset_counts = []
        for w in range(n_windows):
            start = w * window_frames
            end = min(start + window_frames, len(onsets))
            onset_counts.append(float(np.sum(onsets[start:end])))
        speech_rate_variation = float(np.var(onset_counts))
    else:
        speech_rate_variation = 0.0

    # ── Pause naturalness ────────────────────────────────────────────────────
    # Score the distribution of silence runs. Natural speech has mostly short
    # pauses (breaths, <200ms) with occasional long pauses (clause boundaries,
    # >400ms). Ratio of short to long should be 3:1 to 8:1.
    silence_threshold_db = -40.0
    rms_db_flat = rms_db.flatten() if rms_db.ndim > 1 else rms_db
    is_silent = rms_db_flat < silence_threshold_db

    # Find silence runs
    short_pauses = 0  # < 200ms
    long_pauses = 0  # > 400ms
    run_length = 0
    short_threshold_frames = max(1, int(0.2 / frame_duration))  # 200ms
    long_threshold_frames = max(1, int(0.4 / frame_duration))  # 400ms

    for silent in is_silent:
        if silent:
            run_length += 1
        else:
            if run_length > 0:
                if run_length < short_threshold_frames:
                    short_pauses += 1
                elif run_length > long_threshold_frames:
                    long_pauses += 1
            run_length = 0
    # Handle trailing silence run
    if run_length > 0:
        if run_length < short_threshold_frames:
            short_pauses += 1
        elif run_length > long_threshold_frames:
            long_pauses += 1

    # Score: ratio of short to long pauses. Target 3:1 to 8:1.
    if long_pauses > 0:
        ratio = short_pauses / long_pauses
        if 3.0 <= ratio <= 8.0:
            pause_naturalness = 1.0
        elif 1.5 <= ratio <= 12.0:
            # Linear decay outside ideal range
            if ratio < 3.0:
                pause_naturalness = (ratio - 1.5) / 1.5
            else:
                pause_naturalness = (12.0 - ratio) / 4.0
            pause_naturalness = max(0.0, pause_naturalness)
        else:
            pause_naturalness = 0.0
    elif short_pauses > 0:
        # All short pauses, no long ones — slightly unnatural but not terrible
        pause_naturalness = 0.4
    else:
        # No pauses at all
        pause_naturalness = 0.0

    # ── Syllable duration variance ───────────────────────────────────────────
    # Variance of inter-onset intervals. Natural speech has variable syllable
    # durations. TTS tends to over-regularize them.
    onset_indices = np.where(rms_diff > np.percentile(rms_diff, 75))[0]
    if len(onset_indices) > 2:
        intervals = np.diff(onset_indices) * frame_duration  # in seconds
        syllable_duration_variance = float(np.var(intervals))
    else:
        syllable_duration_variance = 0.0

    return {
        "pitch_range_normalized": pitch_range_normalized,
        "pitch_contour_smoothness": pitch_contour_smoothness,
        "speech_rate_variation": speech_rate_variation,
        "pause_naturalness": pause_naturalness,
        "syllable_duration_variance": syllable_duration_variance,
    }


def _compute_features(y: Any, sr: int) -> dict[str, float]:
    """Extract spectral and prosody features from audio array."""
    import numpy as np

    librosa = _import_librosa()

    duration = float(len(y) / sr)

    # Spectral centroid — brightness indicator
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))

    # Spectral rolloff — high-frequency energy boundary
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = float(np.mean(rolloff))

    # Zero-crossing rate — breathiness/noise characteristics
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    # MFCCs — timbral shape
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = [float(x) for x in np.mean(mfccs, axis=1)]
    mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

    # RMS energy for dynamic range
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms)
    above_floor = rms_db[rms_db > -80]
    dynamic_range = float(np.max(rms_db) - np.min(above_floor)) if len(above_floor) > 0 else 0.0

    # Approximate LUFS (simplified — true LUFS needs ITU-R BS.1770)
    rms_mean = float(np.mean(rms))
    lufs_approx = float(20 * np.log10(rms_mean + 1e-10) - 0.691)

    # Silence ratio — frames below -40 dB
    silence_threshold = -40.0
    silent_frames = int(np.sum(rms_db < silence_threshold))
    total_frames = len(rms_db)
    silence_ratio = float(silent_frames / total_frames) if total_frames > 0 else 0.0

    # Pitch contour (via pyin)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced_f0 = f0[voiced_flag.astype(bool) & ~np.isnan(f0)]
    pitch_variance = float(np.var(voiced_f0)) if len(voiced_f0) > 1 else 0.0

    # Prosody metrics — reuse voiced_f0 and rms arrays (no redundant pyin call)
    prosody = _compute_prosody(sr, voiced_f0, rms, rms_db)

    features = {
        "duration": duration,
        "spectral_centroid_mean": centroid_mean,
        "spectral_rolloff_mean": rolloff_mean,
        "zcr_mean": zcr_mean,
        "mfcc_mean": mfcc_mean,
        "mfcc_var": mfcc_var,
        "dynamic_range_db": dynamic_range,
        "lufs_approx": lufs_approx,
        "silence_ratio": silence_ratio,
        "pitch_variance": pitch_variance,
    }
    features.update(prosody)
    return features


# ── Artifact detection ────────────────────────────────────────────────────


def _compute_artifacts(y: Any, sr: int) -> dict[str, Any]:
    """Detect TTS failure modes: clicks, repetition loops, clipping, noise.

    Returns dict with:
      artifact_count: total number of detected artifacts
      clipping_frames: number of consecutive-sample clipping events
      clipping_locations: list of sample indices where clipping starts
      spectral_spikes: number of energy spikes > 3σ from local mean
      spike_locations: list of frame indices with spikes
      repetition_score: 0-1 MFCC self-similarity (0=no repetition, 1=exact loop)
      repetition_locations: list of frame indices where repetition detected
      snr_db: signal-to-noise ratio (voiced energy / unvoiced noise floor)
    """
    import numpy as np

    librosa = _import_librosa()

    result: dict[str, Any] = {
        "artifact_count": 0,
        "clipping_frames": 0,
        "clipping_locations": [],
        "spectral_spikes": 0,
        "spike_locations": [],
        "repetition_score": 0.0,
        "repetition_locations": [],
        "snr_db": 0.0,
    }

    if len(y) == 0:
        return result

    # ── Clipping detection ────────────────────────────────────────────────
    # Consecutive samples at ±0.99 (near-clipping threshold)
    clip_threshold = 0.99
    clipped = np.abs(y) >= clip_threshold
    # Find runs of consecutive clipped samples (3+ frames = clipping event)
    clip_locations: list[int] = []
    run_start = -1
    run_len = 0
    for i in range(len(clipped)):
        if clipped[i]:
            if run_start < 0:
                run_start = i
            run_len += 1
        else:
            if run_len >= 3:
                clip_locations.append(int(run_start))
            run_start = -1
            run_len = 0
    if run_len >= 3:
        clip_locations.append(int(run_start))

    result["clipping_frames"] = len(clip_locations)
    result["clipping_locations"] = clip_locations

    # ── Spectral discontinuity (clicks/pops) ──────────────────────────────
    # Non-overlapping RMS frames so a click lands in exactly 1 frame.
    # frame_length=hop_length prevents the default 2048-sample window from
    # spreading a single-sample spike across 4 overlapping frames.
    # Zero out already-detected clipping regions to avoid double-counting.
    hop_length = 512
    y_for_rms = y.copy()
    for clip_start in clip_locations:
        # Zero out the clipped run (conservative: 3 samples minimum)
        y_for_rms[clip_start : clip_start + max(3, int(0.01 * sr))] = 0.0
    rms = librosa.feature.rms(y=y_for_rms, frame_length=hop_length, hop_length=hop_length)[0]
    if len(rms) > 10:
        # Local mean/std in sliding window of 21 frames (~0.45s at 24kHz)
        window = 21
        half_w = window // 2
        spike_locations: list[int] = []
        for i in range(half_w, len(rms) - half_w):
            local = rms[max(0, i - half_w) : i + half_w + 1]
            local_mean = float(np.mean(local))
            local_std = float(np.std(local))
            if local_std > 0 and rms[i] > local_mean + 3 * local_std:
                spike_locations.append(int(i))
        result["spectral_spikes"] = len(spike_locations)
        result["spike_locations"] = spike_locations

    # ── Repetition detection (TTS loops) ──────────────────────────────────
    # Compute MFCCs, then sliding-window cosine self-similarity.
    # High similarity between distant windows = repetition loop.
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    n_frames = mfccs.shape[1]
    # Compare windows of ~2s, step by ~1s
    win_frames = max(1, int(2.0 * sr / hop_length))
    step_frames = max(1, win_frames // 2)

    rep_location_set: set[int] = set()
    max_sim = 0.0

    if n_frames > 2 * win_frames:
        for i in range(0, n_frames - 2 * win_frames, step_frames):
            window_a = mfccs[:, i : i + win_frames].flatten()
            # Compare with windows at least 1 window-length ahead
            for j in range(i + win_frames, n_frames - win_frames, step_frames):
                window_b = mfccs[:, j : j + win_frames].flatten()
                # Cosine similarity
                norm_a = float(np.linalg.norm(window_a))
                norm_b = float(np.linalg.norm(window_b))
                if norm_a > 0 and norm_b > 0:
                    sim = float(np.dot(window_a, window_b) / (norm_a * norm_b))
                    if sim > max_sim:
                        max_sim = sim
                    if sim > 0.95:  # very high similarity = likely repetition
                        rep_location_set.add(int(j))

    rep_locations = sorted(rep_location_set)
    result["repetition_score"] = round(max_sim, 4)
    result["repetition_locations"] = rep_locations

    # ── SNR estimation ────────────────────────────────────────────────────
    # Compare energy in voiced regions vs unvoiced (noise floor)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    voiced = voiced_flag.astype(bool) if voiced_flag is not None else np.zeros(len(rms), dtype=bool)

    # Align voiced flag length with RMS length
    min_len = min(len(voiced), len(rms))
    voiced = voiced[:min_len]
    rms_aligned = rms[:min_len]

    voiced_energy = rms_aligned[voiced]
    unvoiced_energy = rms_aligned[~voiced]

    if len(voiced_energy) > 0 and len(unvoiced_energy) > 0:
        mean_voiced = float(np.mean(voiced_energy))
        mean_unvoiced = float(np.mean(unvoiced_energy))
        if mean_unvoiced > 1e-10:
            snr = 20 * np.log10(mean_voiced / mean_unvoiced)
            result["snr_db"] = round(float(snr), 2)

    # ── Total artifact count ──────────────────────────────────────────────
    result["artifact_count"] = (
        result["clipping_frames"] + result["spectral_spikes"] + len(rep_locations)
    )

    return result


# ── Perceived quality (Pillar 2 — torchmetrics[audio]) ────────────────────


def _compute_perceived_quality(
    y: Any, sr: int, y_ref: Any | None = None, sr_ref: int | None = None
) -> dict[str, float]:
    """Compute perceived quality metrics via torchmetrics[audio].

    Reference-free metrics (always computed):
        dnsmos_sig, dnsmos_bak, dnsmos_ovr, dnsmos_p808: DNSMOS P.835 sub-scores (1-5)
        srmr: Speech Reverberation Modulation Energy Ratio (higher = drier/cleaner)

    Reference-based metrics (only when y_ref provided):
        pesq: Perceptual Evaluation of Speech Quality (1-5)
        stoi: Short-Time Objective Intelligibility (0-1)

    Audio is resampled to 16kHz for DNSMOS (required by the model).
    """
    import numpy as np

    results: dict[str, float] = {
        "dnsmos_sig": 0.0,
        "dnsmos_bak": 0.0,
        "dnsmos_ovr": 0.0,
        "dnsmos_p808": 0.0,
        "srmr": 0.0,
        "pesq": 0.0,
        "stoi": 0.0,
    }

    try:
        import torch
    except ImportError:
        print(
            "[quality] torch not installed — DNSMOS/SRMR/PESQ/STOI zeroed (~20% score impact). "
            "Install with: uv sync --extra quality",
            file=sys.stderr,
        )
        return results

    # Resample to 16kHz for DNSMOS (use librosa — already guaranteed by quality extra)
    import librosa as _librosa

    target_sr = 16000
    if sr != target_sr:
        y_16k = _librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=target_sr)
    else:
        y_16k = np.asarray(y, dtype=np.float32)

    preds = torch.from_numpy(y_16k).unsqueeze(0)

    # DNSMOS — reference-free MOS prediction (Microsoft DNS Challenge model)
    try:
        from torchmetrics.functional.audio import deep_noise_suppression_mean_opinion_score

        dnsmos = deep_noise_suppression_mean_opinion_score(preds, target_sr, personalized=False)
        results["dnsmos_sig"] = float(dnsmos[0][0])
        results["dnsmos_bak"] = float(dnsmos[0][1])
        results["dnsmos_ovr"] = float(dnsmos[0][2])
        results["dnsmos_p808"] = float(dnsmos[0][3])
    except Exception as e:
        print(f"[quality] DNSMOS failed: {e}", file=sys.stderr)

    # SRMR — reference-free reverberation quality
    try:
        from torchmetrics.functional.audio.srmr import (
            speech_reverberation_modulation_energy_ratio,
        )

        srmr_score = speech_reverberation_modulation_energy_ratio(preds, target_sr)
        results["srmr"] = float(srmr_score[0])
    except Exception as e:
        print(f"[quality] SRMR failed: {e}", file=sys.stderr)

    # Reference-based metrics (PESQ + STOI) — only when reference audio provided
    if y_ref is not None:
        ref_sr = sr_ref if sr_ref is not None else sr
        if ref_sr != target_sr:
            y_ref_16k = _librosa.resample(
                np.asarray(y_ref, dtype=np.float32), orig_sr=ref_sr, target_sr=target_sr
            )
        else:
            y_ref_16k = np.asarray(y_ref, dtype=np.float32)

        # Match lengths (truncate to shorter)
        min_len = min(len(y_16k), len(y_ref_16k))
        preds_matched = torch.from_numpy(y_16k[:min_len]).unsqueeze(0)
        target_matched = torch.from_numpy(y_ref_16k[:min_len]).unsqueeze(0)

        # PESQ — perceptual speech quality (1-5 scale)
        try:
            from torchmetrics.functional.audio import perceptual_evaluation_speech_quality

            pesq_score = perceptual_evaluation_speech_quality(
                preds_matched, target_matched, target_sr, "wb"
            )
            results["pesq"] = float(pesq_score)
        except Exception as e:
            print(f"[quality] PESQ failed: {e}", file=sys.stderr)

        # STOI — short-time objective intelligibility (0-1)
        try:
            from torchmetrics.functional.audio import short_time_objective_intelligibility

            stoi_score = short_time_objective_intelligibility(
                preds_matched, target_matched, target_sr
            )
            results["stoi"] = float(stoi_score)
        except Exception as e:
            print(f"[quality] STOI failed: {e}", file=sys.stderr)

    return results


def _compute_intelligibility(audio_path: Path, script_text: str) -> dict[str, float]:
    """Pillar 3: Whisper round-trip intelligibility scoring.

    Transcribes audio with mlx-whisper, then compares against the original
    script text using jiwer to compute WER (Word Error Rate) and CER
    (Character Error Rate).

    Requires: uv sync --extra asr (mlx-whisper + jiwer)

    Returns dict with "wer" and "cer" keys. Values are -1.0 if ASR
    dependencies are unavailable.
    """
    results: dict[str, float] = {"wer": -1.0, "cer": -1.0}

    if not script_text or not script_text.strip():
        return results

    try:
        import mlx_whisper
    except ImportError:
        print(
            "[quality] mlx-whisper not installed — WER/CER disabled (intelligibility not scored). "
            "Install with: uv sync --extra asr",
            file=sys.stderr,
        )
        return results

    try:
        import jiwer
    except ImportError:
        print("[quality] jiwer not installed — skipping intelligibility", file=sys.stderr)
        return results

    # Transcribe with Whisper
    try:
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
            language="en",
        )
        hypothesis = result.get("text", "").strip()
    except Exception as e:
        print(f"[quality] Whisper transcription failed: {e}", file=sys.stderr)
        return results

    if not hypothesis:
        # Whisper returned empty — treat as total failure
        results["wer"] = 1.0
        results["cer"] = 1.0
        return results

    # Strip non-speech tags from script text before comparison
    # (e.g., [laugh], (sighs), etc. — these aren't spoken words)
    import re

    reference_text = re.sub(r"\[.*?\]|\(.*?\)", "", script_text).strip()
    reference_text = re.sub(r"\s+", " ", reference_text)

    if not reference_text:
        return results

    # Compute WER and CER using jiwer's default text normalization
    try:
        wer_result = jiwer.wer(reference_text, hypothesis)
        cer_result = jiwer.cer(reference_text, hypothesis)
        results["wer"] = round(float(wer_result), 4)
        results["cer"] = round(float(cer_result), 4)
    except Exception as e:
        print(f"[quality] jiwer scoring failed: {e}", file=sys.stderr)

    return results


def _score_against_reference(features: dict[str, Any], reference: dict[str, Any]) -> float:
    """Compare features to reference profile, return 0-1 composite score."""
    import numpy as np

    raw_score = 0.0
    total_weight = 0.0

    metric_keys = [
        ("spectral_centroid_mean", 0.12),
        ("zcr_mean", 0.10),
        ("silence_ratio", 0.12),
        ("pitch_variance", 0.10),
        ("lufs_approx", 0.12),
        ("pitch_range_normalized", 0.15),
        ("pitch_contour_smoothness", 0.10),
        ("pause_naturalness", 0.12),
        ("speech_rate_variation", 0.07),
    ]

    for key, weight in metric_keys:
        if key not in reference:
            continue
        ref = reference[key]
        ref_mean = ref.get("mean", 0.0)
        ref_std = ref.get("std", 1.0)
        if ref_std == 0:
            ref_std = 1.0

        val = features.get(key, 0.0)
        if isinstance(val, list):
            continue

        # Score: 1.0 within 1 std, linear decay to 0.0 at 2+ std
        distance = abs(val - ref_mean) / ref_std
        component_score = max(0.0, 1.0 - max(0.0, distance - 1.0))
        raw_score += component_score * weight
        total_weight += weight

    # Normalize so the score reaches 1.0 even when some reference keys are missing
    score = raw_score / total_weight if total_weight > 0 else 0.0

    # MFCC distance if available — blended as 20% of final score
    if "mfcc_mean" in reference and "mfcc_mean" in features:
        ref_mfcc = np.array(reference["mfcc_mean"]["values"])
        feat_mfcc = np.array(features["mfcc_mean"])
        if len(ref_mfcc) == len(feat_mfcc):
            mfcc_dist = float(np.sqrt(np.sum((ref_mfcc - feat_mfcc) ** 2)))
            ref_std = reference["mfcc_mean"].get("std", 10.0)
            distance = mfcc_dist / ref_std if ref_std > 0 else mfcc_dist
            component_score = max(0.0, 1.0 - max(0.0, distance - 1.0))
            score = score * 0.8 + component_score * 0.2

    return min(1.0, max(0.0, score))


def _score_standalone(features: dict[str, Any]) -> tuple[float, list[str]]:
    """Score without a reference profile — heuristic quality checks.

    Two-pillar scoring:
      Pillar 1 (librosa): spectral basics (5×0.10 = 0.50) + prosody (5 metrics = 0.50)
      Pillar 2 (DNSMOS): when available, final score = 80% Pillar 1 + 20% DNSMOS normalized

    When DNSMOS is unavailable, score is 100% Pillar 1.
    """
    score = 0.0
    notes: list[str] = []

    # ── Spectral basics (0.10 each, 0.50 total) ─────────────────────────────

    # LUFS in podcast range (-18 to -14)
    lufs = features.get("lufs_approx", -30.0)
    if -18.0 <= lufs <= -14.0:
        score += 0.10
    elif -22.0 <= lufs <= -10.0:
        score += 0.05
        notes.append(f"Loudness {lufs:.1f} LUFS — slightly outside podcast range (-18 to -14)")
    else:
        notes.append(f"Loudness {lufs:.1f} LUFS — well outside podcast range (-18 to -14)")

    # Silence ratio between 5-20%
    silence = features.get("silence_ratio", 0.0)
    if 0.05 <= silence <= 0.20:
        score += 0.10
    elif 0.02 <= silence <= 0.30:
        score += 0.05
        notes.append(f"Silence ratio {silence:.1%} — outside ideal range (5-20%)")
    else:
        notes.append(f"Silence ratio {silence:.1%} — significantly outside ideal range (5-20%)")

    # Spectral centroid — speech typically 1000-4000 Hz
    centroid = features.get("spectral_centroid_mean", 0.0)
    if 1000 <= centroid <= 4000:
        score += 0.10
    elif 500 <= centroid <= 6000:
        score += 0.05
        notes.append(f"Spectral centroid {centroid:.0f} Hz — outside typical speech range")
    else:
        notes.append(f"Spectral centroid {centroid:.0f} Hz — atypical for speech")

    # Pitch variance — should be non-zero (not monotone) but not erratic
    pitch_var = features.get("pitch_variance", 0.0)
    if pitch_var > 100:
        score += 0.10
        notes.append("Good pitch variation — not monotone")
    elif pitch_var > 10:
        score += 0.05
        notes.append("Low pitch variation — may sound somewhat flat")
    else:
        notes.append("Very low pitch variation — likely monotone delivery")

    # Dynamic range > 10 dB
    dr = features.get("dynamic_range_db", 0.0)
    if dr > 15:
        score += 0.10
    elif dr > 8:
        score += 0.05
        notes.append(f"Dynamic range {dr:.1f} dB — could be more expressive")
    else:
        notes.append(f"Dynamic range {dr:.1f} dB — very compressed/flat")

    # ── Prosody metrics (0.50 total) ─────────────────────────────────────────

    # Pitch range normalized (0.13) — target 0.3–1.2 for natural radio speech
    prn = features.get("pitch_range_normalized", 0.0)
    if 0.3 <= prn <= 1.2:
        score += 0.13
        notes.append(f"Good pitch range ({prn:.2f}) — expressive but controlled")
    elif 0.15 <= prn <= 2.0:
        score += 0.07
        if prn < 0.3:
            notes.append(f"Pitch range {prn:.2f} — narrow, may sound flat")
        else:
            notes.append(f"Pitch range {prn:.2f} — wide, may sound erratic")
    else:
        if prn < 0.15:
            notes.append(f"Pitch range {prn:.2f} — very narrow, likely monotone")
        else:
            notes.append(f"Pitch range {prn:.2f} — extremely wide, likely unstable")

    # Pause naturalness (0.12) — already scored 0-1 in _compute_prosody
    pn = features.get("pause_naturalness", 0.0)
    score += 0.12 * pn
    if pn >= 0.7:
        notes.append(f"Natural pause distribution (score {pn:.2f})")
    elif pn >= 0.4:
        notes.append(f"Pause structure could be more natural (score {pn:.2f})")
    else:
        notes.append(f"Unnatural pause distribution (score {pn:.2f})")

    # Speech rate variation (0.10) — higher variance = more natural pacing
    srv = features.get("speech_rate_variation", 0.0)
    if srv > 5.0:
        score += 0.10
        notes.append("Good speech rate variation — natural pacing")
    elif srv > 1.0:
        score += 0.05
        notes.append("Low speech rate variation — somewhat metronomic")
    else:
        notes.append("Very low speech rate variation — robotic pacing")

    # Pitch contour smoothness (0.08) — lower = smoother transitions
    # Natural speech: 2-15 Hz/frame. Above 20 = jerky robotic stepping.
    pcs = features.get("pitch_contour_smoothness", 0.0)
    if 0 < pcs <= 15.0:
        score += 0.08
        notes.append(f"Smooth pitch transitions ({pcs:.1f} Hz/frame)")
    elif 0 < pcs <= 25.0:
        score += 0.04
        notes.append(f"Moderately smooth pitch ({pcs:.1f} Hz/frame)")
    else:
        if pcs > 25.0:
            notes.append(f"Jerky pitch transitions ({pcs:.1f} Hz/frame) — robotic stepping")
        else:
            notes.append("No pitch contour data — insufficient voiced frames")

    # Syllable duration variance (0.07) — higher = more natural timing
    # TTS tends to over-regularize syllable durations.
    sdv = features.get("syllable_duration_variance", 0.0)
    if sdv > 0.01:
        score += 0.07
        notes.append(f"Good syllable timing variation ({sdv:.4f})")
    elif sdv > 0.003:
        score += 0.035
        notes.append(f"Low syllable timing variation ({sdv:.4f}) — slightly mechanical")
    else:
        notes.append(f"Very low syllable timing variation ({sdv:.4f}) — robotic rhythm")

    # ── Perceived quality (Pillar 2) ────────────────────────────────────────
    # DNSMOS overall MOS (1-5 scale) blended into final score.
    # When available, perceived quality contributes 20% of the composite
    # and librosa score is weighted down to 80%.
    dnsmos_ovr = features.get("dnsmos_ovr", 0.0)
    if dnsmos_ovr > 0:
        # Normalize DNSMOS from 1-5 scale to 0-1 (1=terrible, 5=excellent)
        dnsmos_norm = max(0.0, min(1.0, (dnsmos_ovr - 1.0) / 4.0))
        score = score * 0.80 + dnsmos_norm * 0.20
        notes.append(f"DNSMOS overall: {dnsmos_ovr:.2f}/5 (perceived quality)")
    else:
        notes.append("DNSMOS unavailable — using librosa-only scoring")

    # SRMR note (informational — not scored, but flagged if reverberant)
    srmr_val = features.get("srmr", 0.0)
    if srmr_val > 0:
        if srmr_val < 3.0:
            notes.append(f"SRMR {srmr_val:.1f} — reverberant (podcast should be dry)")
        elif srmr_val > 8.0:
            notes.append(f"SRMR {srmr_val:.1f} — very dry (good for podcast)")

    # ── Intelligibility (Pillar 3) ─────────────────────────────────────────
    # WER flagging — high error rate means TTS garbled the text
    wer_val = features.get("wer", -1.0)
    if wer_val >= 0:
        if wer_val > 0.30:
            notes.append(f"WER {wer_val:.2f} — severe: TTS garbled the text, flag for re-render")
        elif wer_val > 0.15:
            notes.append(f"WER {wer_val:.2f} — high: review segment for intelligibility issues")
        elif wer_val > 0.05:
            notes.append(f"WER {wer_val:.2f} — minor word differences detected")
        else:
            notes.append(f"WER {wer_val:.2f} — excellent intelligibility")

    # ── Artifact detection (quality gate) ──────────────────────────────────
    # Severe artifacts override score to 0 — these are TTS failures that
    # must trigger re-rendering, not just lower the score.
    artifact_count = features.get("artifact_count", 0)
    clipping = features.get("clipping_frames", 0)
    spikes = features.get("spectral_spikes", 0)
    rep_score = features.get("repetition_score", 0.0)
    snr = features.get("snr_db", 0.0)

    if clipping > 0:
        notes.append(f"Clipping detected: {clipping} event(s) — hard distortion")
    if spikes > 5:
        notes.append(f"Spectral spikes: {spikes} click/pop events detected")
    elif spikes > 0:
        notes.append(f"Spectral spikes: {spikes} minor click/pop event(s)")
    if rep_score > 0.95:
        notes.append(f"Repetition loop detected (similarity {rep_score:.3f}) — TTS stuck in loop")
    elif rep_score > 0.85:
        notes.append(f"High self-similarity ({rep_score:.3f}) — possible repetition")
    if snr > 0:
        if snr < 10:
            notes.append(f"Low SNR ({snr:.1f} dB) — noisy output")
        elif snr > 30:
            notes.append(f"Good SNR ({snr:.1f} dB)")

    # Hard fail: clipping, many spikes, or repetition loop → score 0
    if clipping > 2 or spikes > 10 or rep_score > 0.95:
        notes.append(
            f"ARTIFACT GATE: score overridden to 0 (artifacts={artifact_count}, "
            f"clipping={clipping}, spikes={spikes}, rep={rep_score:.3f})"
        )
        return 0.0, notes

    # Moderate penalty: some artifacts degrade the score
    if artifact_count > 0:
        penalty = min(0.3, artifact_count * 0.02)
        score = max(0.0, score - penalty)
        notes.append(f"Artifact penalty: -{penalty:.2f} ({artifact_count} artifact(s))")

    return min(1.0, max(0.0, score)), notes


def _import_librosa() -> Any:
    """Lazy import librosa to keep it optional."""
    try:
        import librosa

        return librosa
    except ImportError:
        print("librosa is required: uv sync --extra quality", file=sys.stderr)
        sys.exit(1)


def evaluate(
    audio_path: Path,
    reference_path: Path | None = None,
    script_text: str | None = None,
    engine: str | None = None,
) -> QualityReport:
    """Run three-pillar quality evaluation on an audio file.

    Pillar 1: Signal analysis (librosa) — spectral features + prosody
    Pillar 2: Perceived quality (torchmetrics) — DNSMOS, SRMR, PESQ, STOI
    Pillar 3: Intelligibility (mlx-whisper + jiwer) — WER/CER round-trip

    If engine is specified and no explicit reference_path is given,
    auto-resolves to config/quality-reference-{engine}.json.
    Explicit reference_path always takes precedence over engine.
    """
    import numpy as np

    librosa = _import_librosa()

    y, sr = librosa.load(str(audio_path), sr=None)
    features = _compute_features(y, sr)

    # Pillar 2: Perceived quality metrics
    perceived = _compute_perceived_quality(y, sr)
    features.update(perceived)

    # Artifact detection
    artifacts = _compute_artifacts(y, sr)
    features.update(artifacts)

    # Pillar 3: Intelligibility (when script text is available)
    if script_text:
        intelligibility = _compute_intelligibility(audio_path, script_text)
        features.update(intelligibility)

    # Resolve reference: explicit path > engine-specific > none
    if reference_path is None and engine:
        reference_path = _resolve_engine_reference(engine)

    reference: dict[str, Any] | None = None
    if reference_path and reference_path.exists():
        with reference_path.open() as f:
            reference = json.load(f)

    if reference:
        overall = _score_against_reference(features, reference)
        notes = []
        if overall < 0.5:
            notes.append("Significant deviation from reference audio profile")
        elif overall < 0.7:
            notes.append("Moderate deviation from reference audio profile")
        else:
            notes.append("Close to reference audio profile")
    else:
        overall, notes = _score_standalone(features)
        notes.insert(0, "No reference profile — using standalone heuristics")

    # MFCC distance (for report, even without reference)
    mfcc_dist = 0.0
    if reference and "mfcc_mean" in reference and "mfcc_mean" in features:
        ref_mfcc = np.array(reference["mfcc_mean"]["values"])
        feat_mfcc = np.array(features["mfcc_mean"])
        if len(ref_mfcc) == len(feat_mfcc):
            mfcc_dist = float(np.sqrt(np.sum((ref_mfcc - feat_mfcc) ** 2)))

    return QualityReport(
        overall_score=round(overall, 4),
        dynamic_range_lufs=round(features["lufs_approx"], 2),
        silence_ratio=round(features["silence_ratio"], 4),
        spectral_centroid_mean=round(features["spectral_centroid_mean"], 2),
        spectral_rolloff_mean=round(features["spectral_rolloff_mean"], 2),
        zcr_mean=round(features["zcr_mean"], 6),
        mfcc_distance=round(mfcc_dist, 4),
        pitch_variance=round(features["pitch_variance"], 2),
        pitch_range_normalized=round(features.get("pitch_range_normalized", 0.0), 4),
        pitch_contour_smoothness=round(features.get("pitch_contour_smoothness", 0.0), 4),
        speech_rate_variation=round(features.get("speech_rate_variation", 0.0), 4),
        pause_naturalness=round(features.get("pause_naturalness", 0.0), 4),
        syllable_duration_variance=round(features.get("syllable_duration_variance", 0.0), 6),
        duration_seconds=round(features["duration"], 2),
        dnsmos_ovr=round(features.get("dnsmos_ovr", 0.0), 4),
        dnsmos_sig=round(features.get("dnsmos_sig", 0.0), 4),
        dnsmos_bak=round(features.get("dnsmos_bak", 0.0), 4),
        dnsmos_p808=round(features.get("dnsmos_p808", 0.0), 4),
        srmr=round(features.get("srmr", 0.0), 4),
        pesq=round(features.get("pesq", 0.0), 4),
        stoi=round(features.get("stoi", 0.0), 4),
        wer=round(features.get("wer", -1.0), 4),
        cer=round(features.get("cer", -1.0), 4),
        artifact_count=int(features.get("artifact_count", 0)),
        clipping_frames=int(features.get("clipping_frames", 0)),
        spectral_spikes=int(features.get("spectral_spikes", 0)),
        repetition_score=round(features.get("repetition_score", 0.0), 4),
        snr_db=round(features.get("snr_db", 0.0), 2),
        notes=notes,
    )


def build_reference(audio_paths: list[Path], output_path: Path, engine: str | None = None) -> None:
    """Build a reference fingerprint from a collection of audio files.

    If engine is specified, it is stored in the reference profile metadata.
    """
    import numpy as np

    librosa = _import_librosa()

    all_features: list[dict[str, Any]] = []

    for i, path in enumerate(audio_paths):
        print(f"  [{i + 1}/{len(audio_paths)}] Processing {path.name}...")
        y, sr = librosa.load(str(path), sr=None)
        features = _compute_features(y, sr)
        # Merge perceived quality metrics (DNSMOS, SRMR) into features
        perceived = _compute_perceived_quality(y, sr)
        features.update(perceived)
        all_features.append(features)

    if len(all_features) == 1:
        print(
            "Warning: single-sample reference — std estimates are unreliable. "
            "Use 3+ samples for meaningful scoring.",
            file=sys.stderr,
        )

    if not all_features:
        print("No audio files processed.", file=sys.stderr)
        sys.exit(1)

    # Compute mean and std for each scalar metric
    reference: dict[str, Any] = {}
    scalar_keys = [
        "spectral_centroid_mean",
        "spectral_rolloff_mean",
        "zcr_mean",
        "lufs_approx",
        "silence_ratio",
        "pitch_variance",
        "dynamic_range_db",
        "pitch_range_normalized",
        "pitch_contour_smoothness",
        "speech_rate_variation",
        "pause_naturalness",
        "syllable_duration_variance",
        "dnsmos_ovr",
        "srmr",
    ]

    for key in scalar_keys:
        values = [f[key] for f in all_features if key in f and f[key] > 0]
        if values:
            reference[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
                if len(values) > 1
                else float(abs(np.mean(values))) * 0.1,
            }

    # MFCC reference
    mfcc_arrays = [np.array(f["mfcc_mean"]) for f in all_features if "mfcc_mean" in f]
    if mfcc_arrays:
        stacked = np.stack(mfcc_arrays)
        reference["mfcc_mean"] = {
            "values": [float(x) for x in np.mean(stacked, axis=0)],
            "std": float(np.mean(np.std(stacked, axis=0))),
        }

    reference["sample_count"] = len(all_features)
    if engine:
        reference["engine"] = engine

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(reference, f, indent=2)

    print(f"Reference profile saved: {output_path} ({len(all_features)} samples)")


# ── Per-segment / per-speaker / episode evaluation (multi-track) ─────────


@dataclass
class SegmentReport:
    """Quality report for a single rendered segment."""

    index: int
    speaker: str
    register: str
    topic: str
    duration_seconds: float
    features: dict[str, float]
    score: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpeakerReport:
    """Aggregated quality report for one speaker across all their segments."""

    speaker: str
    segment_count: int
    total_duration: float
    mean_features: dict[str, float]
    feature_consistency: dict[str, float]  # std per metric
    register_coverage: dict[str, int]
    register_effectiveness: float  # 0=all registers identical, 1=max differentiation
    register_deltas: dict[str, float]  # per-metric mean delta across registers
    score: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CastChemistry:
    """Cross-speaker metrics measuring ensemble quality."""

    pitch_separation: dict[str, float]  # "host_a_vs_host_b" -> Hz
    energy_contrast: dict[str, float]  # transition energy ratios
    spectral_distinctness: dict[str, float]  # MFCC distance between pairs
    register_diversity: dict[str, float]  # per speaker
    transition_scores: dict[str, float]  # composite per pair
    overall_chemistry: float
    # Phase 4A: enhanced chemistry fields
    transition_coherence_by_pair: dict[str, float] = field(default_factory=dict)
    register_compatibility: dict[str, dict[str, float]] = field(default_factory=dict)
    energy_matching: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransitionCoherence:
    """Transition coherence between adjacent segments."""

    per_transition: list[dict[str, Any]]  # per-boundary scores + deltas
    mean_score: float  # episode-level mean (0=jarring, 1=seamless)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeReport:
    """Full episode quality report with per-segment, per-speaker, and chemistry."""

    episode_date: str
    segment_reports: list[SegmentReport]
    speaker_reports: dict[str, SpeakerReport]
    chemistry: CastChemistry
    transition_coherence: TransitionCoherence = field(
        default_factory=lambda: TransitionCoherence(per_transition=[], mean_score=1.0)
    )
    overall_score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_date": self.episode_date,
            "overall_score": self.overall_score,
            "notes": self.notes,
            "segment_reports": [s.to_dict() for s in self.segment_reports],
            "speaker_reports": {k: v.to_dict() for k, v in self.speaker_reports.items()},
            "chemistry": self.chemistry.to_dict(),
            "transition_coherence": self.transition_coherence.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def evaluate_segment(
    segment_path: Path,
    script_text: str | None = None,
) -> tuple[dict[str, Any], float, list[str]]:
    """Evaluate a single segment WAV. Returns (features, score, notes).

    The atomic unit of multi-track evaluation. Reuses _compute_features()
    and _score_standalone() from whole-file evaluation.

    If script_text is provided, Pillar 3 intelligibility (WER/CER) is computed.
    """
    import soundfile as sf

    y, sr = sf.read(str(segment_path), dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)

    features = _compute_features(y, sr)

    # Pillar 2: Perceived quality metrics
    perceived = _compute_perceived_quality(y, sr)
    features.update(perceived)

    # Artifact detection
    artifacts = _compute_artifacts(y, sr)
    features.update(artifacts)

    # Pillar 3: Intelligibility (when script text available)
    if script_text:
        intelligibility = _compute_intelligibility(segment_path, script_text)
        features.update(intelligibility)

    # Extract F0 contour and RMS contour for visualization
    librosa = _import_librosa()
    import numpy as np

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )
    features["f0_contour"] = [float(x) if not np.isnan(x) else 0.0 for x in f0]

    rms = librosa.feature.rms(y=y)[0]
    features["rms_contour"] = [float(x) for x in rms]

    score, notes = _score_standalone(features)
    return features, score, notes


def _compute_register_effectiveness(
    segment_reports: list[SegmentReport],
    speaker: str,
) -> tuple[float, dict[str, float]]:
    """Measure feature differentiation across registers for one speaker.

    Groups segments by register, computes mean features per register,
    then measures pairwise deltas across all register pairs. Higher
    deltas mean the register system produces audibly different delivery.

    Returns (score, deltas) where:
      score: 0 = all registers identical, 1 = maximum differentiation
      deltas: per-metric mean absolute delta across register pairs

    Target range: 0.3–0.7 (differentiated but not cartoonish).
    """
    import numpy as np

    speaker_segs = [s for s in segment_reports if s.speaker == speaker]
    if len(speaker_segs) < 2:
        return 0.0, {}

    # Group segments by register
    by_register: dict[str, list[dict[str, float]]] = {}
    for seg in speaker_segs:
        reg = seg.register
        if reg not in by_register:
            by_register[reg] = []
        by_register[reg].append(seg.features)

    if len(by_register) < 2:
        return 0.0, {}

    # Feature dimensions that should vary across registers
    delta_keys = [
        "pitch_range_normalized",
        "lufs_approx",
        "speech_rate_variation",
        "spectral_centroid_mean",
    ]

    # Compute mean features per register
    register_means: dict[str, dict[str, float]] = {}
    for reg, feat_list in by_register.items():
        means: dict[str, float] = {}
        for key in delta_keys:
            vals = [f.get(key, 0.0) for f in feat_list if key in f]
            if vals:
                means[key] = float(np.mean(vals))
        register_means[reg] = means

    # Pairwise deltas across all register pairs
    registers = list(register_means.keys())
    pair_deltas: dict[str, list[float]] = {k: [] for k in delta_keys}

    for i, reg_a in enumerate(registers):
        for reg_b in registers[i + 1 :]:
            for key in delta_keys:
                va = register_means[reg_a].get(key, 0.0)
                vb = register_means[reg_b].get(key, 0.0)
                pair_deltas[key].append(abs(va - vb))

    # Mean delta per metric
    mean_deltas: dict[str, float] = {}
    for key in delta_keys:
        if pair_deltas[key]:
            mean_deltas[key] = round(float(np.mean(pair_deltas[key])), 4)

    if not mean_deltas:
        return 0.0, {}

    # Normalize each metric delta to 0-1 using empirically reasonable ranges.
    # These represent "expected maximum differentiation" for natural speech.
    normalization = {
        "pitch_range_normalized": 0.5,  # 0.5 octave difference = max
        "lufs_approx": 4.0,  # 4 dB energy difference = max (mean of all pairs)
        "speech_rate_variation": 5.0,  # 5 syllables/sec variation = max
        "spectral_centroid_mean": 500.0,  # 500 Hz centroid shift = max
    }

    normalized_deltas: list[float] = []
    for key in delta_keys:
        delta = mean_deltas.get(key, 0.0)
        max_range = normalization.get(key, 1.0)
        normalized_deltas.append(min(1.0, delta / max_range))

    # Score = mean of normalized deltas
    score = float(np.mean(normalized_deltas))
    return round(min(1.0, max(0.0, score)), 4), mean_deltas


def _compute_transition_coherence(
    segment_paths: list[Path],
    boundary_seconds: float = 0.5,
) -> TransitionCoherence:
    """Measure spectral continuity at segment boundaries.

    For each adjacent pair of segments, extracts the last `boundary_seconds`
    of segment N and the first `boundary_seconds` of segment N+1, then
    compares spectral centroid, energy (RMS), and noise floor (ZCR).

    Score per transition: 0 = jarring room change, 1 = seamless.
    Episode-level: mean across all boundaries.

    Note: trailing silence in segments will lower coherence scores at
    boundaries even when voices are acoustically matched. This is by
    design — post-DSP (compression, EQ) should minimize boundary gaps.
    """
    import numpy as np

    librosa = _import_librosa()

    if len(segment_paths) < 2:
        return TransitionCoherence(per_transition=[], mean_score=1.0)

    per_transition: list[dict[str, Any]] = []

    # Cache loaded audio to avoid re-reading the same file when it appears
    # on both sides of consecutive transitions (path_b at i becomes path_a at i+1).
    audio_cache: dict[str, tuple[Any, int]] = {}

    def _load_cached(path: Path) -> tuple[Any, int]:
        key = str(path)
        if key not in audio_cache:
            audio_cache[key] = librosa.load(key, sr=None)
        return audio_cache[key]

    for i in range(len(segment_paths) - 1):
        path_a = segment_paths[i]
        path_b = segment_paths[i + 1]

        if not path_a.exists() or not path_b.exists():
            continue

        y_a, sr_a = _load_cached(path_a)
        y_b, sr_b = _load_cached(path_b)

        # Extract boundary regions
        tail_samples = int(boundary_seconds * sr_a)
        head_samples = int(boundary_seconds * sr_b)

        tail = y_a[-tail_samples:] if len(y_a) > tail_samples else y_a
        head = y_b[:head_samples] if len(y_b) > head_samples else y_b

        if len(tail) == 0 or len(head) == 0:
            continue

        # Spectral centroid at boundary
        centroid_tail = float(np.mean(librosa.feature.spectral_centroid(y=tail, sr=sr_a)))
        centroid_head = float(np.mean(librosa.feature.spectral_centroid(y=head, sr=sr_b)))
        centroid_delta = abs(centroid_tail - centroid_head)

        # Energy (RMS) at boundary
        rms_tail = float(np.mean(librosa.feature.rms(y=tail)))
        rms_head = float(np.mean(librosa.feature.rms(y=head)))
        # Convert to dB difference
        rms_tail_db = 20 * np.log10(max(rms_tail, 1e-10))
        rms_head_db = 20 * np.log10(max(rms_head, 1e-10))
        energy_delta_db = abs(float(rms_tail_db - rms_head_db))

        # Noise floor proxy via ZCR (zero crossing rate) — high ZCR = noisy/sibilant
        zcr_tail = float(np.mean(librosa.feature.zero_crossing_rate(y=tail)))
        zcr_head = float(np.mean(librosa.feature.zero_crossing_rate(y=head)))
        zcr_delta = abs(zcr_tail - zcr_head)

        # Score: lower deltas = more coherent
        # Normalize each dimension, then average
        centroid_score = max(0.0, 1.0 - centroid_delta / 500.0)  # 500 Hz = max tolerable
        energy_score = max(0.0, 1.0 - energy_delta_db / 6.0)  # 6 dB = max tolerable
        zcr_score = max(0.0, 1.0 - zcr_delta / 0.1)  # 0.1 ZCR diff = max tolerable

        transition_score = round(0.4 * centroid_score + 0.4 * energy_score + 0.2 * zcr_score, 4)

        per_transition.append(
            {
                "from_segment": i,
                "to_segment": i + 1,
                "score": transition_score,
                "centroid_delta": round(centroid_delta, 2),
                "energy_delta_db": round(energy_delta_db, 2),
                "zcr_delta": round(zcr_delta, 4),
            }
        )

    mean_score = (
        round(float(np.mean([t["score"] for t in per_transition])), 4) if per_transition else 1.0
    )

    return TransitionCoherence(per_transition=per_transition, mean_score=mean_score)


def _aggregate_speaker_features(
    segment_reports: list[SegmentReport],
    speaker: str,
) -> SpeakerReport:
    """Aggregate features across all segments from one speaker."""
    import numpy as np

    speaker_segs = [s for s in segment_reports if s.speaker == speaker]
    if not speaker_segs:
        return SpeakerReport(
            speaker=speaker,
            segment_count=0,
            total_duration=0.0,
            mean_features={},
            feature_consistency={},
            register_coverage={},
            register_effectiveness=0.0,
            register_deltas={},
            score=0.0,
            notes=["No segments found"],
        )

    # Scalar feature keys to aggregate
    scalar_keys = [
        "spectral_centroid_mean",
        "spectral_rolloff_mean",
        "zcr_mean",
        "pitch_variance",
        "lufs_approx",
        "silence_ratio",
        "dynamic_range_db",
        "pitch_range_normalized",
        "pitch_contour_smoothness",
        "speech_rate_variation",
        "pause_naturalness",
        "syllable_duration_variance",
        "dnsmos_ovr",
        "srmr",
    ]

    mean_features: dict[str, float] = {}
    feature_consistency: dict[str, float] = {}

    for key in scalar_keys:
        values = [s.features.get(key, 0.0) for s in speaker_segs if key in s.features]
        if values:
            mean_features[key] = float(np.mean(values))
            feature_consistency[key] = float(np.std(values))

    # WER/CER aggregation (uses -1 sentinel, so separate from scalar_keys)
    wer_values = [s.features["wer"] for s in speaker_segs if s.features.get("wer", -1.0) >= 0]
    if wer_values:
        mean_features["wer"] = float(np.mean(wer_values))
    cer_values = [s.features["cer"] for s in speaker_segs if s.features.get("cer", -1.0) >= 0]
    if cer_values:
        mean_features["cer"] = float(np.mean(cer_values))

    # Register coverage
    register_coverage: dict[str, int] = {}
    for s in speaker_segs:
        reg = s.register
        register_coverage[reg] = register_coverage.get(reg, 0) + 1

    total_duration = sum(s.duration_seconds for s in speaker_segs)
    mean_score = float(np.mean([s.score for s in speaker_segs]))

    notes: list[str] = []
    # Check consistency
    high_variance_keys = [
        k for k, v in feature_consistency.items() if v > 0.3 * abs(mean_features.get(k, 1.0))
    ]
    if high_variance_keys:
        notes.append(f"High variance in: {', '.join(high_variance_keys)}")

    if len(register_coverage) == 1:
        notes.append(f"Only uses {list(register_coverage.keys())[0]} register")

    # Register effectiveness — how differentiated are the registers?
    reg_eff, reg_deltas = _compute_register_effectiveness(segment_reports, speaker)
    if reg_eff > 0:
        if 0.3 <= reg_eff <= 0.7:
            notes.append(f"Register effectiveness: {reg_eff:.2f} — good differentiation")
        elif reg_eff < 0.3:
            notes.append(f"Register effectiveness: {reg_eff:.2f} — registers sound too similar")
        else:
            notes.append(
                f"Register effectiveness: {reg_eff:.2f} — registers may be over-differentiated"
            )

    return SpeakerReport(
        speaker=speaker,
        segment_count=len(speaker_segs),
        total_duration=round(total_duration, 2),
        mean_features={k: round(v, 4) for k, v in mean_features.items()},
        feature_consistency={k: round(v, 4) for k, v in feature_consistency.items()},
        register_coverage=register_coverage,
        register_effectiveness=reg_eff,
        register_deltas=reg_deltas,
        score=round(mean_score, 4),
        notes=notes,
    )


def _compute_cast_chemistry(
    speaker_reports: dict[str, SpeakerReport],
    segment_reports: list[SegmentReport],
    coherence: TransitionCoherence | None = None,
) -> CastChemistry:
    """Compute cross-speaker chemistry metrics.

    When coherence data is provided (from Phase 1G), computes additional
    per-pair metrics: transition coherence by pair, register compatibility,
    and energy matching at handoff boundaries.
    """
    import numpy as np

    speakers = list(speaker_reports.keys())
    pitch_separation: dict[str, float] = {}
    energy_contrast: dict[str, float] = {}
    spectral_distinctness: dict[str, float] = {}
    transition_scores: dict[str, float] = {}

    # Pairwise comparisons
    for i, sa in enumerate(speakers):
        for sb in speakers[i + 1 :]:
            pair_key = f"{sa}_vs_{sb}"
            feat_a = speaker_reports[sa].mean_features
            feat_b = speaker_reports[sb].mean_features

            # Pitch separation — |median F0 proxy via pitch_variance|
            # Use spectral centroid difference as a proxy for pitch register separation
            pitch_a = feat_a.get("spectral_centroid_mean", 0.0)
            pitch_b = feat_b.get("spectral_centroid_mean", 0.0)
            pitch_sep = abs(pitch_a - pitch_b)
            pitch_separation[pair_key] = round(pitch_sep, 2)

            # Energy contrast — RMS/LUFS difference
            lufs_a = feat_a.get("lufs_approx", -20.0)
            lufs_b = feat_b.get("lufs_approx", -20.0)
            lufs_diff = abs(lufs_a - lufs_b)
            energy_contrast[pair_key] = round(lufs_diff, 2)

            # Spectral distinctness — aggregate feature distance
            shared_keys = [
                "spectral_centroid_mean",
                "zcr_mean",
                "pitch_variance",
                "pitch_range_normalized",
                "pitch_contour_smoothness",
            ]
            diffs_sq = []
            for key in shared_keys:
                va = feat_a.get(key, 0.0)
                vb = feat_b.get(key, 0.0)
                # Normalize by max to make dimensions comparable
                max_val = max(abs(va), abs(vb), 1e-6)
                diffs_sq.append(((va - vb) / max_val) ** 2)
            spectral_dist = float(np.sqrt(sum(diffs_sq)))
            spectral_distinctness[pair_key] = round(spectral_dist, 4)

            # Transition score (composite)
            # Higher pitch separation = better (normalize to 0-1, cap at 500 Hz)
            pitch_score = min(1.0, pitch_sep / 500.0)
            # Moderate energy contrast = good (0-3 dB ideal, penalize >6 dB)
            if lufs_diff <= 3.0:
                energy_score = 1.0
            elif lufs_diff <= 6.0:
                energy_score = 1.0 - (lufs_diff - 3.0) / 6.0
            else:
                energy_score = max(0.0, 0.5 - (lufs_diff - 6.0) / 12.0)
            # Spectral distinctness — higher = better (cap at 2.0)
            spectral_score = min(1.0, spectral_dist / 2.0)

            transition_scores[pair_key] = round(
                0.4 * pitch_score + 0.3 * energy_score + 0.3 * spectral_score, 4
            )

    # Register diversity per speaker
    available_registers = {"baseline", "emphasis", "reflective", "reactive"}
    register_diversity: dict[str, float] = {}
    for speaker, report in speaker_reports.items():
        used = len(report.register_coverage)
        register_diversity[speaker] = round(used / len(available_registers), 2)

    # ── Phase 4A: enhanced chemistry from coherence data ─────────────
    transition_coherence_by_pair: dict[str, float] = {}
    register_compatibility: dict[str, dict[str, float]] = {}
    energy_matching: dict[str, float] = {}

    if coherence and coherence.per_transition and len(segment_reports) > 1:
        # Build lookup: segment index → speaker, register
        seg_lookup: dict[int, tuple[str, str]] = {}
        for seg in segment_reports:
            seg_lookup[seg.index] = (seg.speaker, seg.register)

        # Group boundary data by speaker pair
        pair_coherence_scores: dict[str, list[float]] = {}
        pair_energy_deltas: dict[str, list[float]] = {}
        pair_register_combos: dict[str, dict[str, list[float]]] = {}

        for t_item in coherence.per_transition:
            from_idx = t_item["from_segment"]
            to_idx = t_item["to_segment"]
            if from_idx not in seg_lookup or to_idx not in seg_lookup:
                continue

            from_speaker, from_reg = seg_lookup[from_idx]
            to_speaker, to_reg = seg_lookup[to_idx]

            # Only count speaker-change boundaries for pair metrics
            if from_speaker == to_speaker:
                continue

            # Canonical pair key (alphabetical)
            pair = tuple(sorted([from_speaker, to_speaker]))
            pair_key = f"{pair[0]}_vs_{pair[1]}"

            # Coherence by pair
            pair_coherence_scores.setdefault(pair_key, []).append(t_item["score"])

            # Energy matching: |LUFS delta| at handoff
            energy_delta = t_item.get("energy_delta_db", 0.0)
            pair_energy_deltas.setdefault(pair_key, []).append(energy_delta)

            # Register compatibility: score per register combo
            reg_combo = f"{from_reg}\u2192{to_reg}"
            pair_register_combos.setdefault(pair_key, {}).setdefault(reg_combo, []).append(
                t_item["score"]
            )

        # Aggregate per-pair coherence
        for pair_key, scores in pair_coherence_scores.items():
            transition_coherence_by_pair[pair_key] = round(float(np.mean(scores)), 4)

        # Energy matching: 1.0 when delta < 1dB, linear decay to 0.0 at >= 6dB
        for pair_key, deltas in pair_energy_deltas.items():
            mean_delta = float(np.mean(deltas))
            if mean_delta < 1.0:
                energy_matching[pair_key] = 1.0
            elif mean_delta >= 6.0:
                energy_matching[pair_key] = 0.0
            else:
                energy_matching[pair_key] = round(1.0 - (mean_delta - 1.0) / 5.0, 4)

        # Register compatibility: mean score per register combo per pair
        for pair_key, combos in pair_register_combos.items():
            register_compatibility[pair_key] = {
                combo: round(float(np.mean(scores)), 4) for combo, scores in combos.items()
            }

    # Overall chemistry: blend existing transition_scores with coherence if available
    if transition_scores:
        base_overall = float(np.mean(list(transition_scores.values())))
        if transition_coherence_by_pair:
            coherence_overall = float(np.mean(list(transition_coherence_by_pair.values())))
            overall = 0.7 * base_overall + 0.3 * coherence_overall
        else:
            overall = base_overall
    else:
        overall = 0.0

    return CastChemistry(
        pitch_separation=pitch_separation,
        energy_contrast=energy_contrast,
        spectral_distinctness=spectral_distinctness,
        register_diversity=register_diversity,
        transition_scores=transition_scores,
        overall_chemistry=round(overall, 4),
        transition_coherence_by_pair=transition_coherence_by_pair,
        register_compatibility=register_compatibility,
        energy_matching=energy_matching,
    )


def _score_episode_composite(
    speaker_reports: dict[str, SpeakerReport],
    chemistry: CastChemistry,
    segment_reports: list[SegmentReport] | None = None,
    coherence: TransitionCoherence | None = None,
) -> tuple[float, list[str]]:
    """Compute episode-level composite score.

    50% mean speaker scores + 25% chemistry + 15% register util + 10% consistency
    Transition coherence is reported but doesn't alter the composite (diagnostic).
    """
    import numpy as np

    notes: list[str] = []

    # 50% — mean speaker quality
    speaker_scores = [r.score for r in speaker_reports.values()]
    mean_speaker = float(np.mean(speaker_scores)) if speaker_scores else 0.0

    # 25% — cast chemistry
    chem_score = chemistry.overall_chemistry

    # 15% — register utilization (breadth × effectiveness)
    # Breadth: fraction of available registers used (from chemistry)
    reg_diversity_scores = list(chemistry.register_diversity.values())
    reg_breadth = float(np.mean(reg_diversity_scores)) if reg_diversity_scores else 0.0
    # Effectiveness: how differentiated the registers sound (from speaker reports)
    reg_eff_scores = [
        r.register_effectiveness for r in speaker_reports.values() if r.register_effectiveness > 0
    ]
    reg_effectiveness = float(np.mean(reg_eff_scores)) if reg_eff_scores else 0.0
    # Blend: 50% breadth + 50% effectiveness
    reg_util = 0.5 * reg_breadth + 0.5 * reg_effectiveness

    # 10% — speaker consistency (lower std across segments = better)
    consistency_scores = []
    for report in speaker_reports.values():
        if report.feature_consistency:
            # Normalize: low variance = high score
            variances = list(report.feature_consistency.values())
            # Cap variance contribution — below 0.1 std is great
            avg_std = float(np.mean(variances))
            consistency_scores.append(max(0.0, 1.0 - min(1.0, avg_std)))
    consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.5

    overall = 0.50 * mean_speaker + 0.25 * chem_score + 0.15 * reg_util + 0.10 * consistency

    notes.append(f"Speaker quality: {mean_speaker:.2f} (50%)")
    notes.append(f"Cast chemistry: {chem_score:.2f} (25%)")
    notes.append(f"Register utilization: {reg_util:.2f} (15%)")
    notes.append(f"Speaker consistency: {consistency:.2f} (10%)")

    # Report transition coherence (informational — diagnostic, not in composite)
    if coherence and coherence.per_transition:
        notes.append(
            f"Transition coherence: {coherence.mean_score:.2f} (mean across {len(coherence.per_transition)} boundaries)"
        )
        jarring = [t for t in coherence.per_transition if t["score"] < 0.5]
        if jarring:
            notes.append(f"  {len(jarring)} jarring transition(s) — check DSP chain")

    # Report mean DNSMOS across speakers (informational)
    dnsmos_values = [
        r.mean_features.get("dnsmos_ovr", 0.0)
        for r in speaker_reports.values()
        if r.mean_features.get("dnsmos_ovr", 0.0) > 0
    ]
    if dnsmos_values:
        mean_dnsmos = float(np.mean(dnsmos_values))
        notes.append(f"Mean DNSMOS: {mean_dnsmos:.2f}/5 (perceived quality)")

    # Report mean WER across segments (informational)
    if segment_reports:
        wer_values = [
            s.features.get("wer", -1.0) for s in segment_reports if s.features.get("wer", -1.0) >= 0
        ]
        if wer_values:
            mean_wer = float(np.mean(wer_values))
            notes.append(f"Mean WER: {mean_wer:.3f} (intelligibility)")
            high_wer = [s for s in segment_reports if s.features.get("wer", -1.0) > 0.15]
            if high_wer:
                notes.append(f"  {len(high_wer)} segment(s) with WER > 0.15 — review for re-render")

    return round(min(1.0, max(0.0, overall)), 4), notes


def evaluate_manifest(
    manifest_path: Path,
    script_path: Path | None = None,
) -> EpisodeReport:
    """Evaluate a multi-track episode from its manifest.

    Main entry point for production episodes. Evaluates per-segment,
    aggregates per-speaker, computes cast chemistry, and produces
    a composite episode score.

    If script_path is provided, segment text is joined by index for
    Pillar 3 intelligibility scoring (WER/CER).
    """
    manifest = json.loads(manifest_path.read_text())
    segments_dir = Path(manifest["segments_dir"])
    date_str = manifest.get("date", "unknown")

    # Load script text if available (for intelligibility scoring)
    script_texts: dict[int, str] = {}
    if script_path and script_path.exists():
        script = json.loads(script_path.read_text())
        for i, seg in enumerate(script.get("segments", [])):
            script_texts[i] = seg.get("text", "")

    # Evaluate each segment
    segment_reports: list[SegmentReport] = []
    features_by_segment: list[dict[str, Any]] = []
    segment_paths: list[Path] = []

    for seg in manifest["segments"]:
        wav_path = segments_dir / seg["file"]
        if not wav_path.exists():
            print(f"  Warning: {wav_path} not found, skipping", file=sys.stderr)
            continue

        segment_paths.append(wav_path)
        seg_text = script_texts.get(seg["index"], None)
        features, score, notes = evaluate_segment(wav_path, script_text=seg_text)
        features_by_segment.append(features)

        segment_reports.append(
            SegmentReport(
                index=seg["index"],
                speaker=seg["speaker"],
                register=seg.get("register", "baseline"),
                topic=seg.get("topic", ""),
                duration_seconds=seg.get("duration_seconds", features.get("duration", 0.0)),
                features=features,
                score=score,
                notes=notes,
            )
        )

    # Aggregate by speaker
    speakers = sorted(set(s.speaker for s in segment_reports))
    speaker_reports: dict[str, SpeakerReport] = {}
    for speaker in speakers:
        speaker_reports[speaker] = _aggregate_speaker_features(segment_reports, speaker)

    # Transition coherence (compute first — used by cast chemistry)
    coherence = _compute_transition_coherence(segment_paths)

    # Cast chemistry (enhanced with coherence data in Phase 4A)
    chemistry = _compute_cast_chemistry(speaker_reports, segment_reports, coherence)

    # Episode composite
    overall, episode_notes = _score_episode_composite(
        speaker_reports, chemistry, segment_reports, coherence
    )

    return EpisodeReport(
        episode_date=date_str,
        segment_reports=segment_reports,
        speaker_reports=speaker_reports,
        chemistry=chemistry,
        transition_coherence=coherence,
        overall_score=overall,
        notes=episode_notes,
    )


def log_experiment(
    voice_id: str,
    report: QualityReport,
    exaggeration: float = 0.0,
    cfg_weight: float = 0.0,
    temperature: float = 0.0,
    status: str = "completed",
    description: str = "",
    results_path: Path = Path("autoresearch/results.tsv"),
) -> None:
    """Append an experiment result to the autoresearch results.tsv log.

    Creates the file with headers if it doesn't exist. Append-only.
    """
    from datetime import UTC, datetime

    headers = (
        "timestamp\tvoice_id\texaggeration\tcfg_weight\ttemperature\t"
        "overall_score\tpitch_range\tpause_naturalness\tstatus\tdescription"
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists() or results_path.stat().st_size == 0

    with results_path.open("a") as f:
        if write_header:
            f.write(headers + "\n")
        row = (
            f"{datetime.now(UTC).isoformat()}\t"
            f"{voice_id}\t"
            f"{exaggeration}\t"
            f"{cfg_weight}\t"
            f"{temperature}\t"
            f"{report.overall_score}\t"
            f"{report.pitch_range_normalized}\t"
            f"{report.pause_naturalness}\t"
            f"{status}\t"
            f"{description}"
        )
        f.write(row + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Radio quality evaluation")
    parser.add_argument("audio", nargs="?", help="Audio file to evaluate (single-file mode)")
    parser.add_argument("--build-reference", nargs="+", help="Audio files to build reference from")
    parser.add_argument(
        "-o", "--output", default="config/quality-reference.json", help="Reference output path"
    )
    parser.add_argument("--reference", default=None, help="Reference profile for scoring")
    parser.add_argument(
        "--manifest", default=None, help="Manifest JSON for per-segment episode evaluation"
    )
    parser.add_argument(
        "--script",
        default=None,
        help="Script JSON for intelligibility scoring (requires --manifest)",
    )
    parser.add_argument(
        "--viz", action="store_true", help="Generate visualization PNGs (requires --manifest)"
    )
    parser.add_argument(
        "--engine",
        default=None,
        choices=sorted(KNOWN_ENGINES),
        help="TTS engine — auto-selects engine-specific reference profile",
    )
    args = parser.parse_args()

    if args.build_reference:
        paths = [Path(p) for p in args.build_reference]
        output = Path(args.output)
        # Auto-name output when engine is specified and output is the default
        if args.engine and args.output == "config/quality-reference.json":
            output = REFERENCE_DIR / f"quality-reference-{args.engine}.json"
        build_reference(paths, output, engine=args.engine)
    elif args.manifest:
        manifest_path = Path(args.manifest)
        script_path = Path(args.script) if args.script else None
        print(f"Evaluating episode: {manifest_path}")
        if script_path:
            print(f"Script for intelligibility: {script_path}")
        report = evaluate_manifest(manifest_path, script_path=script_path)
        print(report.to_json())

        # Write report JSON next to manifest
        report_path = manifest_path.parent / "episode-report.json"
        report_path.write_text(report.to_json())
        print(f"\nReport saved: {report_path}")

        if args.viz:
            from src.visualize import render_all

            manifest = json.loads(manifest_path.read_text())

            # Build features_by_segment from segment reports
            features_list = [s.features for s in report.segment_reports]

            # Build speaker_features from speaker reports
            speaker_features = {k: v.mean_features for k, v in report.speaker_reports.items()}

            # Chemistry metrics
            chemistry_metrics = report.chemistry.to_dict()

            viz_paths = render_all(
                manifest,
                features_list,
                speaker_features,
                chemistry_metrics,
                manifest_path.parent,
            )
            print("\nVisualizations generated:")
            for key, paths in viz_paths.items():
                if isinstance(paths, list):
                    for p in paths:
                        print(f"  {p}")
                else:
                    print(f"  {paths}")
    elif args.audio:
        ref_path: Path | None = None
        if args.reference:
            ref_path = Path(args.reference)
            if not ref_path.exists():
                ref_path = None
        elif not args.engine:
            # No explicit reference and no engine — try generic fallback
            default_ref = Path("config/quality-reference.json")
            if default_ref.exists():
                ref_path = default_ref
        report = evaluate(Path(args.audio), reference_path=ref_path, engine=args.engine)
        print(report.to_json())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
