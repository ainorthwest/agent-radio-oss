"""DSP post-processing for rendered TTS audio.

Applies broadcast-standard signal chain via pedalboard.
Shared by Sound Booth previews and the pipeline renderer.

Chain order: HPF → pitch → warmth EQ → de-box cut → compressor → presence EQ → de-esser → limiter → reverb

Final pass (episode-level): loudness normalization via pyloudnorm (ITU-R BS.1770).
Target: -16 LUFS (standard podcast loudness). Applied in _write_output before export.
"""

from __future__ import annotations

from typing import Any

DSP_DEFAULTS: dict[str, float] = {
    "speed_factor": 1.0,
    "pitch_semitones": 0.0,
    "warmth_db": 0.0,
    "presence_db": 0.0,
    "comp_threshold_db": 0.0,
    "comp_ratio": 1.0,
    "comp_attack_ms": 3.0,
    "comp_release_ms": 100.0,
    "reverb_room_size": 0.0,
    "hpf_hz": 80.0,
    "debox_db": 0.0,
    "deesser_db": 0.0,
    "limiter_db": -1.0,
}


def _is_default(post: dict[str, float]) -> bool:
    """Return True if all DSP params match defaults (no processing needed).

    Checks all trigger params including debox and deesser — any non-default
    value causes the full DSP chain to run. HPF, limiter, compressor attack/
    release are always active when the chain runs, so they are not checked here.
    """
    trigger_keys = {
        "speed_factor": 1.0,
        "pitch_semitones": 0.0,
        "warmth_db": 0.0,
        "presence_db": 0.0,
        "comp_threshold_db": 0.0,
        "comp_ratio": 1.0,
        "reverb_room_size": 0.0,
        "debox_db": 0.0,
        "deesser_db": 0.0,
    }
    for key, default in trigger_keys.items():
        if abs(post.get(key, default) - default) > 1e-6:
            return False
    return True


def apply_dsp(audio: Any, profile: dict[str, Any], sample_rate: int) -> Any:
    """Apply broadcast-standard DSP chain from profile's ``post:`` block.

    Chain: speed → HPF → pitch → warmth EQ (180Hz) → de-box cut (350Hz) →
           compressor (explicit attack/release) → presence EQ (4kHz) →
           de-esser (7.5kHz) → limiter (-1dBTP) → reverb

    Args:
        audio: 1-D numpy float32 array (mono).
        profile: Voice profile dict. Reads ``profile["post"]``.
        sample_rate: Sample rate in Hz (typically 24000).

    Returns:
        Processed 1-D numpy float32 array, same length/rate.
        Returns input unchanged if no ``post:`` block or all defaults.
    """
    post = profile.get("post")
    if not post or _is_default(post):
        return audio

    import numpy as np

    # ── 0. Speed adjustment (time-stretch via resampling) ─────────────
    speed = float(post.get("speed_factor", 1.0))
    if abs(speed - 1.0) > 0.001:
        from scipy.signal import resample_poly

        # speed > 1.0 = faster (shorter), speed < 1.0 = slower (longer)
        # Resample: treat as if recorded at sr*speed, play at sr
        up = 100
        down = int(100 * speed)
        audio = resample_poly(audio, up, down).astype(np.float32)
    from pedalboard import (  # type: ignore[attr-defined]
        Compressor,
        HighpassFilter,
        HighShelfFilter,
        Limiter,
        LowShelfFilter,
        PeakFilter,
        Pedalboard,
        PitchShift,
        Reverb,
    )

    effects: list[Any] = []

    # ── 1. High-pass filter (remove rumble below speech) ──────────────────
    hpf_hz = float(post.get("hpf_hz", 80.0))
    if hpf_hz > 0:
        effects.append(HighpassFilter(cutoff_frequency_hz=hpf_hz))

    # ── 2. Pitch shift ────────────────────────────────────────────────────
    pitch = float(post.get("pitch_semitones", 0.0))
    if abs(pitch) > 1e-6:
        effects.append(PitchShift(semitones=pitch))

    # ── 3. Warmth EQ (low shelf at 180Hz, not 300Hz) ─────────────────────
    warmth = float(post.get("warmth_db", 0.0))
    if abs(warmth) > 1e-6:
        effects.append(LowShelfFilter(cutoff_frequency_hz=180.0, gain_db=warmth))

    # ── 4. De-box cut (narrow notch at 350Hz to remove TTS boxiness) ─────
    debox_db = float(post.get("debox_db", 0.0))
    if abs(debox_db) > 1e-6:
        effects.append(
            PeakFilter(
                cutoff_frequency_hz=350.0,
                gain_db=debox_db,
                q=2.0,
            )
        )

    # ── 5. Compressor (explicit attack/release) ──────────────────────────
    threshold = float(post.get("comp_threshold_db", 0.0))
    ratio = float(post.get("comp_ratio", 1.0))
    if ratio > 1.0 + 1e-6 or threshold < -1e-6:
        attack_ms = float(post.get("comp_attack_ms", 3.0))
        release_ms = float(post.get("comp_release_ms", 100.0))
        effects.append(
            Compressor(
                threshold_db=threshold,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
        )

    # ── 6. Presence EQ (high shelf at 4kHz) ──────────────────────────────
    presence = float(post.get("presence_db", 0.0))
    if abs(presence) > 1e-6:
        effects.append(HighShelfFilter(cutoff_frequency_hz=4000.0, gain_db=presence))

    # ── 7. De-esser (narrow cut at 7.5kHz to tame sibilance) ─────────────
    deesser_db = float(post.get("deesser_db", 0.0))
    if abs(deesser_db) > 1e-6:
        effects.append(
            PeakFilter(
                cutoff_frequency_hz=7500.0,
                gain_db=deesser_db,
                q=3.0,
            )
        )

    # ── 8. Limiter (true peak protection) ─────────────────────────────────
    limiter_db = float(post.get("limiter_db", -1.0))
    if limiter_db < 0.0:
        effects.append(Limiter(threshold_db=limiter_db))

    # ── 9. Reverb (optional room character) ──────────────────────────────
    room = float(post.get("reverb_room_size", 0.0))
    if room > 1e-6:
        effects.append(Reverb(room_size=room, wet_level=0.15))

    if not effects:
        return audio

    board = Pedalboard(effects)
    # pedalboard expects (channels, samples) — reshape mono to (1, N)
    shaped = audio.reshape(1, -1).astype(np.float32)
    processed = board(shaped, sample_rate)
    return processed.flatten().astype(np.float32)


# ── Episode-level loudness normalization ─────────────────────────────────────

TARGET_LUFS: float = -16.0  # EBU R128 / standard podcast loudness


def normalize_loudness(
    audio: Any,
    sample_rate: int,
    target_lufs: float = TARGET_LUFS,
) -> Any:
    """Normalize integrated loudness of assembled episode audio to target LUFS.

    Uses ITU-R BS.1770 metering (pyloudnorm). Applied once on the full episode
    array before export — not per-segment, so relative dynamics are preserved.
    Includes a true peak limiter pass at -1dBTP after gain adjustment.

    Args:
        audio:       1-D numpy float32 array (mono, full episode).
        sample_rate: Sample rate in Hz.
        target_lufs: Target integrated loudness (default -16 LUFS).

    Returns:
        Gain-adjusted 1-D numpy float32 array.
        Returns input unchanged if measured loudness is already within 0.5 LU
        of target, or if pyloudnorm is unavailable.
    """
    try:
        import numpy as np
        import pyloudnorm as pyln
    except ImportError:
        return audio

    # pyloudnorm expects float64 and shape (samples,) for mono
    audio_f64 = audio.astype(np.float64)

    meter = pyln.Meter(sample_rate)  # ITU-R BS.1770
    measured = meter.integrated_loudness(audio_f64)

    # Guard against silence or invalid measurement
    if not (measured == measured) or measured < -70.0:  # NaN or effectively silent
        return audio

    delta = abs(measured - target_lufs)
    if delta < 0.5:
        return audio  # already close enough

    normalized = pyln.normalize.loudness(audio_f64, measured, target_lufs)
    normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)

    # True peak limiter pass — prevents inter-sample peaks above -1dBTP
    try:
        from pedalboard import Limiter, Pedalboard  # type: ignore[attr-defined]

        board = Pedalboard([Limiter(threshold_db=-1.0)])
        shaped = normalized.reshape(1, -1)
        normalized = board(shaped, sample_rate).flatten().astype(np.float32)
    except ImportError:
        pass  # pedalboard unavailable — rely on np.clip above

    print(
        f"  Loudness: {measured:.1f} LUFS → {target_lufs:.1f} LUFS (gain {target_lufs - measured:+.1f} LU)"
    )
    return normalized
