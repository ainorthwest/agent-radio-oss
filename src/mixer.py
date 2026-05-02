"""Mixer: assembles per-segment WAVs + music beds into a final episode.

The mixer is the editor — it reads a manifest produced by the renderer,
loads individual segment WAVs and music beds, assembles them on a timeline
with context-aware gap timing, applies music ducking, and produces the
final mastered episode. Output lands in the episode bundle directory
(output/episodes/{date}/).

Usage:
    uv run python -m src.mixer output/episodes/date/manifest.json
    uv run python -m src.mixer manifest.json --no-music
    uv run python -m src.mixer manifest.json --output custom.mp3
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from src.dsp import normalize_loudness

# ── Gap timing (assembly decisions) ──────────────────────────────────────────

# Gap context types — classify why a particular gap duration was chosen
GAP_SHORT_ACK = "short_ack"
GAP_TOPIC_CHANGE = "topic_change"
GAP_SAME_SPEAKER = "same_speaker"
GAP_AFTER_REFLECTIVE = "after_reflective"
GAP_AFTER_EMPHASIS = "after_emphasis"
GAP_AFTER_REACTIVE = "after_reactive"
GAP_DEFAULT_CROSS = "default_cross"


def _compute_gap_seconds(
    current_seg: dict[str, Any],
    next_seg: dict[str, Any],
) -> tuple[float, str]:
    """Compute natural gap duration between two segments.

    Calibrated against real broadcast radio (Creative Commons samples,
    2026-03-16 analysis). Real radio median gap is ~100ms, mean ~180ms.
    Most cross-speaker gaps are 60-160ms. Only topic changes and
    reflective beats go above 300ms.

    Rules (applied in priority order):
    - Short acknowledgments (< 4 words): 0.04-0.08s (near-instant pickup)
    - Same speaker continuation: 0.06-0.12s (breath pause)
    - Topic change between speakers: 0.25-0.40s (beat)
    - After reflective register: 0.20-0.35s (let it land)
    - After emphasis register: 0.08-0.16s (energy carries forward)
    - After reactive register: 0.04-0.10s (momentum)
    - Default same-topic cross-speaker: 0.06-0.14s

    Returns:
        Tuple of (gap_seconds, context_type) where context_type is one of
        the GAP_* constants explaining why this duration was chosen.
    """
    import random

    text = str(next_seg.get("text", ""))
    word_count = next_seg.get("word_count", len(text.split()))
    current_speaker = str(current_seg.get("speaker", ""))
    next_speaker = str(next_seg.get("speaker", ""))
    current_register = str(current_seg.get("register", "baseline"))
    current_topic = str(current_seg.get("topic", ""))
    next_topic = str(next_seg.get("topic", ""))
    same_speaker = current_speaker == next_speaker
    topic_change = current_topic != next_topic

    if word_count <= 4 and not same_speaker:
        return random.uniform(0.04, 0.08), GAP_SHORT_ACK
    elif topic_change and not same_speaker:
        return random.uniform(0.25, 0.40), GAP_TOPIC_CHANGE
    elif same_speaker:
        return random.uniform(0.06, 0.12), GAP_SAME_SPEAKER
    elif current_register == "reflective":
        return random.uniform(0.20, 0.35), GAP_AFTER_REFLECTIVE
    elif current_register == "emphasis":
        return random.uniform(0.08, 0.16), GAP_AFTER_EMPHASIS
    elif current_register == "reactive":
        return random.uniform(0.04, 0.10), GAP_AFTER_REACTIVE
    else:
        return random.uniform(0.06, 0.14), GAP_DEFAULT_CROSS


# ── Audio loading + resampling ───────────────────────────────────────────────


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy polyphase filter."""
    if orig_sr == target_sr:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    """Load a WAV file, resample to target_sr, convert to mono float32."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")

    # Stereo to mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        data = resample_audio(data, sr, target_sr)

    return data.astype(np.float32)


# ── Naturalness utilities ────────────────────────────────────────────────────


def _trim_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -40.0) -> np.ndarray:
    """Trim leading and trailing silence from a rendered segment.

    TTS models often produce dead air at the start and end of renders.
    Real radio producers trim these milliseconds tightly — the gap timing
    system handles inter-segment spacing, not the segment audio itself.

    Uses -40dB threshold (conservative) to avoid cutting trailing speech.
    Keeps a 30ms pad on each end to prevent clipping the first/last
    phoneme. Only trims silence, never speech.
    """
    threshold = 10 ** (threshold_db / 20)
    pad_samples = int(sample_rate * 0.03)  # 30ms safety pad

    # RMS in 10ms windows
    window = int(sample_rate * 0.01)
    n_windows = len(audio) // window
    if n_windows < 3:
        return audio

    rms = np.array(
        [np.sqrt(np.mean(audio[i * window : (i + 1) * window] ** 2)) for i in range(n_windows)]
    )

    # Find first and last window above threshold
    voiced = np.where(rms > threshold)[0]
    if len(voiced) == 0:
        return audio

    start = max(0, voiced[0] * window - pad_samples)
    end = min(len(audio), (voiced[-1] + 1) * window + pad_samples)

    return audio[start:end]


# Crossfade duration at segment boundaries (prevents clicks and micro-cuts)
_CROSSFADE_MS = 5  # 5ms equal-power crossfade

# Pink noise floor amplitude (-65 dBFS — below conscious detection,
# above digital floor, eliminates dead-air uncanny valley)
_NOISE_FLOOR_DB = -65.0
_NOISE_FLOOR_AMP = 10 ** (_NOISE_FLOOR_DB / 20)  # ≈ 0.000562

# Per-segment loudness target (EBU R128 dialogue) — applied before assembly
# so all segments enter the mix at consistent perceived loudness
_SEGMENT_TARGET_LUFS = -23.0


def _generate_noise_floor(n_samples: int, amplitude: float = _NOISE_FLOOR_AMP) -> np.ndarray:
    """Generate shaped noise floor to fill inter-segment gaps.

    Uses white noise with a gentle low-pass to approximate pink noise character.
    This replaces np.zeros() gaps — the psychoacoustic effect of faint room
    ambience vs dead digital silence is significant for perceived naturalness.
    """
    from scipy.signal import butter, lfilter

    white = np.random.normal(0, amplitude, n_samples).astype(np.float32)
    # Gentle 1st-order low-pass at 4kHz to shape toward pink spectrum
    # (full pink noise generation is overkill for sub-audible fill)
    b, a = butter(1, 4000.0, btype="low", fs=MIX_SAMPLE_RATE)
    return lfilter(b, a, white).astype(np.float32)


def _apply_segment_fades(audio: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply equal-power fade-out on tail and fade-in on head of a segment.

    Equal-power (sqrt curve) prevents the perceived loudness dip that
    linear crossfades create at the midpoint.
    """
    if len(audio) < fade_samples * 2:
        return audio  # too short to fade safely
    fade_out = np.sqrt(np.linspace(1.0, 0.0, fade_samples, dtype=np.float32))
    fade_in = np.sqrt(np.linspace(0.0, 1.0, fade_samples, dtype=np.float32))
    audio = audio.copy()
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    return audio


def _normalize_segment_loudness(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Normalize individual segment to -23 LUFS before assembly.

    This ensures segments from different parts of the episode have
    consistent perceived loudness before the mix. Episode-level
    normalization to -16 LUFS happens after full assembly.
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        return audio

    # pyloudnorm requires audio longer than the block size (400ms at default)
    min_samples = int(sample_rate * 0.5)  # 500ms minimum
    if len(audio) < min_samples:
        return audio

    audio_f64 = audio.astype(np.float64)
    meter = pyln.Meter(sample_rate)
    measured = meter.integrated_loudness(audio_f64)

    if not (measured == measured) or measured < -70.0:  # NaN or silence
        return audio

    if abs(measured - _SEGMENT_TARGET_LUFS) < 1.0:
        return audio  # already close enough

    normalized = pyln.normalize.loudness(audio_f64, measured, _SEGMENT_TARGET_LUFS)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


# ── Timeline assembly ────────────────────────────────────────────────────────


def _assemble_voice_track(
    manifest: dict[str, Any],
    target_sr: int,
    editorial: Any | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]], list[dict[str, Any]]]:
    """Place segments on a timeline with context-aware gaps.

    When an editorial manifest is provided, applies segment-level overrides:
    skip, volume adjustment, gap override, and global gap multiplier.

    Naturalness features:
    - Per-segment LUFS normalization (-23 LUFS) before assembly
    - Equal-power crossfades (5ms) at segment boundaries
    - Shaped noise floor in gaps instead of digital silence

    Returns:
        audio: 1-D float32 array of the full voice track at target_sr
        voice_regions: list of (start_sample, end_sample) tuples marking
                       where voice is active (used for music ducking)
        gap_records: list of gap metadata dicts with keys:
            after_segment, before_segment, context_type,
            duration_seconds, gap_start_seconds
    """
    from src.editorial import get_gap_override, get_volume_adjustment, should_skip_segment

    segments = manifest["segments"]
    segments_dir = Path(manifest["segments_dir"])

    chunks: list[np.ndarray] = []
    voice_regions: list[tuple[int, int]] = []
    gap_records: list[dict[str, Any]] = []
    cursor = 0  # current position in samples (at target_sr)
    fade_samples = int(target_sr * _CROSSFADE_MS / 1000)

    for i, seg in enumerate(segments):
        # Editorial: skip segment if marked
        if editorial and should_skip_segment(i, editorial):
            continue

        seg_audio = _load_audio(segments_dir / seg["file"], target_sr)

        # Trim leading/trailing silence from TTS render
        seg_audio = _trim_silence(seg_audio, target_sr)

        # Per-segment loudness normalization
        seg_audio = _normalize_segment_loudness(seg_audio, target_sr)

        # Editorial: apply volume adjustment
        if editorial:
            vol_db = get_volume_adjustment(i, editorial)
            if vol_db != 0.0:
                gain = 10 ** (vol_db / 20)
                seg_audio = seg_audio * gain

        # Apply equal-power crossfades at segment boundaries
        seg_audio = _apply_segment_fades(seg_audio, fade_samples)

        # Record voice region
        start = cursor
        end = cursor + len(seg_audio)
        voice_regions.append((start, end))

        chunks.append(seg_audio)
        cursor = end

        # Add gap before next segment
        if i < len(segments) - 1:
            # Editorial: gap override takes priority
            gap_override = get_gap_override(i, editorial) if editorial else None
            if gap_override is not None:
                gap_seconds = gap_override
                context_type = "editorial_override"
            else:
                gap_seconds, context_type = _compute_gap_seconds(seg, segments[i + 1])

                # Editorial: apply global gap multiplier (only to computed gaps,
                # not explicit editorial overrides which are absolute)
                if editorial and editorial.pacing.global_gap_multiplier != 1.0:
                    gap_seconds *= editorial.pacing.global_gap_multiplier

            gap_samples = int(target_sr * gap_seconds)
            chunks.append(_generate_noise_floor(gap_samples))

            gap_records.append(
                {
                    "after_segment": i,
                    "before_segment": i + 1,
                    "context_type": context_type,
                    "duration_seconds": round(gap_seconds, 4),
                    "gap_start_seconds": round(cursor / target_sr, 4),
                }
            )

            cursor += gap_samples

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return audio, voice_regions, gap_records


# ── Music ducking ────────────────────────────────────────────────────────────

# Defaults — overridable via cast.yaml music: block
DUCK_DB = -18.0  # Music level when voice is active (dB below full)
BED_LEVEL_DB = -6.0  # Music level when NO voice (dB below original)
FADE_MS = 50  # Duck envelope fade time


def _build_duck_envelope(
    total_samples: int,
    voice_regions: list[tuple[int, int]],
    fade_samples: int = 0,
    attack_samples: int | None = None,
    release_samples: int | None = None,
) -> np.ndarray:
    """Create ducking envelope: 1.0 = voice active, 0.0 = silence/gap.

    Supports asymmetric fade: fast attack (music ducks instantly when voice
    starts) and slow release (music fades back up gradually after voice ends).

    Args:
        total_samples: Length of the envelope.
        voice_regions: List of (start, end) sample positions where voice is active.
        fade_samples: Legacy symmetric fade (preserved for backward compat).
                      Used as fallback when attack/release are not specified.
        attack_samples: Duck attack time (voice starts → music ducks). Fast = crisp pickup.
        release_samples: Duck release time (voice ends → music returns). Slow = smooth.
    """
    if attack_samples is None:
        attack_samples = fade_samples
    if release_samples is None:
        release_samples = fade_samples

    envelope = np.zeros(total_samples, dtype=np.float32)

    for start, end in voice_regions:
        s = max(0, min(start, total_samples))
        e = max(0, min(end, total_samples))
        envelope[s:e] = 1.0

    # Asymmetric smoothing: fast attack, slow release
    if attack_samples > 0 or release_samples > 0:
        smoothed = np.zeros_like(envelope)
        current = 0.0
        for i in range(total_samples):
            target = envelope[i]
            if target > current:
                # Attack: ramp up (voice starting → duck music fast)
                rate = 1.0 / max(attack_samples, 1)
                current = min(current + rate, target)
            else:
                # Release: ramp down (voice ending → bring music back slowly)
                rate = 1.0 / max(release_samples, 1)
                current = max(current - rate, target)
            smoothed[i] = current
        envelope = smoothed.astype(np.float32)
        np.clip(envelope, 0.0, 1.0, out=envelope)

    return envelope


def _apply_ducking(
    music: np.ndarray,
    envelope: np.ndarray,
    duck_db: float = DUCK_DB,
    bed_level_db: float = BED_LEVEL_DB,
) -> np.ndarray:
    """Apply ducking to music based on voice activity envelope.

    Where voice is active (envelope=1.0): music plays at duck_db level
    Where voice is silent (envelope=0.0): music plays at bed_level_db level
    """
    duck_gain = 10 ** (duck_db / 20)  # e.g. -18dB = 0.126
    bed_gain = 10 ** (bed_level_db / 20)  # e.g. -6dB = 0.501

    # Interpolate between bed_gain (no voice) and duck_gain (voice active)
    gain = bed_gain * (1.0 - envelope) + duck_gain * envelope

    return music * gain


def _overlay_music_bed(
    voice: np.ndarray,
    voice_regions: list[tuple[int, int]],
    music_path: str,
    target_sr: int,
    start_sample: int,
    duck_db: float = DUCK_DB,
    bed_level_db: float = BED_LEVEL_DB,
    fade_ms: int = FADE_MS,
) -> np.ndarray:
    """Overlay a music bed onto the voice track with ducking.

    The music is placed starting at start_sample and runs for its full
    duration (or until the voice track ends, whichever comes first).
    """
    music = _load_audio(Path(music_path), target_sr)

    # Trim music to fit within voice track
    available = len(voice) - start_sample
    if available <= 0:
        return voice
    if len(music) > available:
        music = music[:available]

    # Build envelope with asymmetric ducking: fast attack (10ms), slow release (200ms)
    attack_samples = int(target_sr * 0.01)  # 10ms — music ducks instantly
    release_samples = int(target_sr * 0.2)  # 200ms — music fades back gradually
    region_envelope = _build_duck_envelope(
        len(music),
        # Offset voice regions to be relative to music start
        [(max(0, s - start_sample), max(0, e - start_sample)) for s, e in voice_regions],
        attack_samples=attack_samples,
        release_samples=release_samples,
    )

    # Apply ducking
    ducked_music = _apply_ducking(music, region_envelope, duck_db, bed_level_db)

    # Apply fade-in at start and fade-out at end of music
    fade_in_samples = min(int(target_sr * 0.5), len(ducked_music) // 4)  # 0.5s or 25% of music
    fade_out_samples = min(int(target_sr * 1.0), len(ducked_music) // 4)  # 1.0s or 25%
    if fade_in_samples > 0:
        ducked_music[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples, dtype=np.float32)
    if fade_out_samples > 0:
        ducked_music[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples, dtype=np.float32)

    # Mix into voice track
    result = voice.copy()
    result[start_sample : start_sample + len(ducked_music)] += ducked_music
    return result


# ── Final export ─────────────────────────────────────────────────────────────


def _wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    """Convert WAV to MP3 via ffmpeg. Removes the WAV on success."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-ab", "192k", str(mp3_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    wav_path.unlink()


def _next_episode_number(episode_dir: Path, ext: str) -> int:
    """Find the next available episode number (episode_000, episode_001, ...).

    Checks both .wav and .mp3 at each slot to avoid cross-format collisions.
    """
    n = 0
    while (episode_dir / f"episode_{n:03d}.wav").exists() or (
        episode_dir / f"episode_{n:03d}.mp3"
    ).exists():
        n += 1
    return n


def _write_episode(
    audio: np.ndarray,
    sample_rate: int,
    episode_dir: Path,
    output_format: str = "mp3",
) -> Path:
    """Normalize loudness, write WAV, optionally convert to MP3.

    Writes into the episode bundle directory (e.g. output/episodes/{date}/).
    Files are numbered episode_000, episode_001, etc. to prevent overwrites.
    """
    import soundfile as sf

    duration = len(audio) / sample_rate
    print(f"  Total audio: {duration:.1f}s")

    audio = normalize_loudness(audio, sample_rate)

    if output_format == "mp3":
        n = _next_episode_number(episode_dir, "mp3")
        wav_path = episode_dir / f"episode_{n:03d}.wav"
        sf.write(str(wav_path), audio, sample_rate)
        mp3_path = episode_dir / f"episode_{n:03d}.mp3"
        _wav_to_mp3(wav_path, mp3_path)
        print(f"  Episode saved: {mp3_path}")
        return mp3_path
    else:
        n = _next_episode_number(episode_dir, "wav")
        wav_path = episode_dir / f"episode_{n:03d}.wav"
        sf.write(str(wav_path), audio, sample_rate)
        print(f"  Episode saved: {wav_path}")
        return wav_path


# ── Public API ───────────────────────────────────────────────────────────────

# Mix target sample rate — upsample voice to match music beds
MIX_SAMPLE_RATE = 48000


def mix(
    manifest_path: Path,
    no_music: bool = False,
    output_format: str = "mp3",
) -> Path:
    """Assemble segments + music beds into a final episode.

    Reads manifest.json, loads segment WAVs and music beds,
    applies gap timing, music ducking, and loudness normalization.
    Output writes into the episode bundle directory (manifest's parent).

    Args:
        manifest_path: Path to manifest.json from render_segments().
        no_music:      If True, skip music bed overlay (voice-only).
        output_format: "mp3" or "wav".

    Returns:
        Path to the final episode file.
    """
    # Episode dir is the manifest's parent (the episode bundle)
    episode_dir = manifest_path.parent
    episode_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    music = manifest.get("music", {})
    target_sr = MIX_SAMPLE_RATE

    print(f"Mixing: {manifest_path.name} ({len(manifest['segments'])} segments)")

    # Load editorial manifest (v2 format — graceful no-op for v1)
    from src.editorial import EditorialManifest, load_editorial

    editorial: EditorialManifest = load_editorial(manifest)
    if editorial.has_overrides():
        n_overrides = len(editorial.segment_overrides)
        n_cues = len(editorial.music_cues)
        print(f"  Editorial: {n_overrides} segment override(s), {n_cues} music cue(s)")

    # Step 1: Assemble voice track with context-aware gaps
    voice, voice_regions, gap_records = _assemble_voice_track(manifest, target_sr, editorial)
    print(f"  Voice track: {len(voice) / target_sr:.1f}s, {len(voice_regions)} segments")

    # Step 2: Overlay music beds (if available and not disabled)
    preroll_s = 0.0
    if not no_music:
        # Read music config from manifest (v2+) — falls back to cast.yaml for older manifests
        manifest_music_config = manifest.get("music_config", {})
        if manifest_music_config:
            duck_db = float(manifest_music_config.get("duck_db", DUCK_DB))
            bed_level_db = float(manifest_music_config.get("bed_level_db", BED_LEVEL_DB))
            fade_ms = int(manifest_music_config.get("fade_ms", FADE_MS))
            intro_preroll_s_val = float(manifest_music_config.get("intro_preroll_s", 0.0))
        else:
            from src.renderer import _load_cast

            program_slug = manifest.get("program")
            cast = _load_cast(program_slug)
            cast_music = cast.get("music", {})
            duck_db = float(cast_music.get("duck_db", DUCK_DB))
            bed_level_db = float(cast_music.get("bed_level_db", BED_LEVEL_DB))
            fade_ms = int(cast_music.get("fade_ms", FADE_MS))
            intro_preroll_s_val = float(cast_music.get("intro_preroll_s", 0.0))

        # Intro music — optionally starts before voice (pre-roll)
        intro_path = music.get("intro")
        if intro_path and Path(intro_path).exists():
            intro_preroll_s = intro_preroll_s_val
            if intro_preroll_s > 0:
                preroll_s = intro_preroll_s
                # Prepend silence so theme plays before first voice segment
                preroll_samples = int(intro_preroll_s * target_sr)
                voice = np.concatenate([np.zeros(preroll_samples, dtype=voice.dtype), voice])
                # Shift all voice regions forward by the preroll
                voice_regions = [
                    (start + preroll_samples, end + preroll_samples) for start, end in voice_regions
                ]
            print(f"  Overlaying intro: {intro_path}")
            voice = _overlay_music_bed(
                voice,
                voice_regions,
                intro_path,
                target_sr,
                start_sample=0,
                duck_db=duck_db,
                bed_level_db=bed_level_db,
                fade_ms=fade_ms,
            )

        # Outro music — aligned so music ends when (or near) voice ends
        outro_path = music.get("outro")
        if outro_path and Path(outro_path).exists():
            outro_audio = _load_audio(Path(outro_path), target_sr)
            outro_start = max(0, len(voice) - len(outro_audio))
            print(f"  Overlaying outro: {outro_path} (starts at {outro_start / target_sr:.1f}s)")
            voice = _overlay_music_bed(
                voice,
                voice_regions,
                outro_path,
                target_sr,
                start_sample=outro_start,
                duck_db=duck_db,
                bed_level_db=bed_level_db,
                fade_ms=fade_ms,
            )

        # Step 2.5: Process editorial music cues (stings, beds)
        if editorial.music_cues and not no_music:
            for cue in editorial.music_cues:
                cue_asset_path: str | None = None

                # Resolve asset path — generate on-the-fly or use file
                if cue.asset.startswith("generate:"):
                    # Dynamic generation:
                    #   "generate:sting" or "generate:bumper:Am" (MIDI engine)
                    #   "generate:musicgen:<prompt>" (MusicGen engine)
                    try:
                        from src.music import MusicParams
                        from src.music import generate as gen_music

                        parts = cue.asset.split(":")
                        if len(parts) > 1 and parts[1] == "musicgen":
                            # MusicGen: join remaining parts as prompt
                            prompt = ":".join(parts[2:]) if len(parts) > 2 else ""
                            asset = gen_music(
                                MusicParams(engine="musicgen", prompt=prompt),
                                output_dir=episode_dir / "music",
                            )
                        else:
                            # MIDI engine: type and optional key
                            gen_type = parts[1] if len(parts) > 1 else cue.type
                            gen_key = parts[2] if len(parts) > 2 else "C"
                            asset = gen_music(
                                MusicParams(type=gen_type, key=gen_key),
                                output_dir=episode_dir / "music",
                                midi_only=False,
                            )
                        cue_asset_path = str(asset.path)
                    except NotImplementedError as exc:
                        print(
                            f"  WARNING: Music cue '{cue.asset}' requested generation, "
                            f"but generation is not available in this build: {exc}"
                        )
                        continue
                    except Exception as exc:
                        print(f"  WARNING: Music generation failed for cue: {exc}")
                        continue
                elif (
                    cue.asset in music
                    and isinstance(music[cue.asset], str)
                    and music[cue.asset]
                    and Path(music[cue.asset]).exists()
                ):
                    # Palette key (e.g. "transition", "sting") — resolve via manifest music block
                    cue_asset_path = music[cue.asset]
                elif Path(cue.asset).exists():
                    cue_asset_path = cue.asset

                if not cue_asset_path:
                    print(f"  WARNING: Music cue asset not found: {cue.asset}")
                    continue

                # Find the gap start position for this cue
                gap_start_sample = 0
                for rec in gap_records:
                    if rec["after_segment"] == cue.after_segment:
                        gap_start_sample = int(rec["gap_start_seconds"] * target_sr)
                        break
                else:
                    # If no gap record, place at end of the segment's voice region
                    if cue.after_segment < len(voice_regions):
                        gap_start_sample = voice_regions[cue.after_segment][1]

                if cue.type in ("sting", "transition"):
                    # Insert sting audio into the gap
                    sting_audio = _load_audio(Path(cue_asset_path), target_sr)
                    # Apply cue volume
                    if cue.volume_db != 0.0:
                        sting_audio = sting_audio * (10 ** (cue.volume_db / 20))
                    # Apply fades
                    fade_in = int(target_sr * cue.fade_in_s)
                    fade_out = int(target_sr * cue.fade_out_s)
                    if fade_in > 0 and fade_in < len(sting_audio):
                        sting_audio[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
                    if fade_out > 0 and fade_out < len(sting_audio):
                        sting_audio[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
                    # Mix into gap region (additive)
                    end_sample = min(gap_start_sample + len(sting_audio), len(voice))
                    sting_len = end_sample - gap_start_sample
                    if sting_len > 0:
                        voice[gap_start_sample:end_sample] += sting_audio[:sting_len]
                        print(
                            f"  Sting cue after seg {cue.after_segment}: "
                            f"{sting_len / target_sr:.2f}s"
                        )

                elif cue.type == "bed":
                    # Overlay as a ducked music bed starting at the gap
                    voice = _overlay_music_bed(
                        voice,
                        voice_regions,
                        cue_asset_path,
                        target_sr,
                        start_sample=gap_start_sample,
                        duck_db=duck_db,
                        bed_level_db=bed_level_db + cue.volume_db,
                        fade_ms=fade_ms,
                    )
                    print(
                        f"  Bed cue after seg {cue.after_segment}: "
                        f"starts at {gap_start_sample / target_sr:.1f}s"
                    )

    # Write gap timing sidecar (after preroll offset is applied)
    if gap_records:
        if preroll_s > 0:
            for rec in gap_records:
                rec["gap_start_seconds"] = rec.get("gap_start_seconds", 0) + preroll_s
        gap_path = episode_dir / "gap-timing.json"
        gap_path.write_text(json.dumps(gap_records, indent=2))
        print(f"  Gap timing: {len(gap_records)} gaps → {gap_path.name}")

    # Step 3: Final mastering + export
    return _write_episode(voice, target_sr, episode_dir, output_format)


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agent Radio mixer — assemble episodes from segments"
    )
    parser.add_argument("manifest", help="Path to manifest.json from render_segments()")
    parser.add_argument("--no-music", action="store_true", help="Skip music bed overlay")
    parser.add_argument("--format", choices=["mp3", "wav"], default="mp3", help="Output format")
    args = parser.parse_args()

    mix(
        manifest_path=Path(args.manifest),
        no_music=args.no_music,
        output_format=args.format,
    )
