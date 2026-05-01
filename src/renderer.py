"""Renderer: episode script JSON -> per-segment WAVs + manifest.

OSS scope: Kokoro ONNX is the only supported engine. Voice profiles
in voices/*.yaml must have ``engine: kokoro``. The MLX engines (CSM,
Dia, Orpheus, Qwen3, Chatterbox-MLX) and PyTorch Chatterbox live in
the proprietary `agent-radio` repo.

Hardware backend is selected by the ``KOKORO_PROVIDER`` env var (see
``src/engines/kokoro.py``) — defaults to CPU.

Extras required:
    Kokoro:    uv sync --extra tts

Output structure:
    Episodes:  output/episodes/{date}/segments/, output/episodes/{date}/manifest.json
    Auditions: output/auditions/{profile}-{engine}-{timestamp}/audition.wav

Modes:
    Episode:       uv run python -m src.renderer output/episodes/{date}/script.json
    Segments only: uv run python -m src.renderer output/episodes/{date}/script.json --segments-only
    Audition:      uv run python -m src.renderer --audition --voice voices/foo.yaml script.json
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

from src.config import RadioConfig, load_config
from src.dsp import apply_dsp, normalize_loudness
from src.engines.kokoro import get_kokoro, resolve_kokoro_voice


def _engine_unavailable(name: str) -> Any:
    """Return a callable that raises if a non-OSS engine is invoked.

    The dispatch branches for MLX / Chatterbox engines stay in this
    file so the existing test suite keeps verifying dispatch decisions,
    profile validation, and tag-conversion logic. They are unreachable
    at runtime for OSS users because voice profiles can only legally
    set ``engine: kokoro``.
    """

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            f"Engine {name!r} is not available in agent-radio-oss. "
            "OSS supports only 'kokoro'. The proprietary agent-radio repo "
            f"holds the {name!r} implementation."
        )

    return _raise


# Non-OSS engines kept as raise-on-call stubs so the renderer source
# stays close to the upstream and tests can still exercise dispatch
# branches without importing MLX or Chatterbox.
get_chatterbox = _engine_unavailable("chatterbox")
get_chatterbox_conds = _engine_unavailable("chatterbox")
get_chatterbox_mlx = _engine_unavailable("chatterbox-mlx")
get_csm = _engine_unavailable("csm")
get_dia = _engine_unavailable("dia")
get_orpheus = _engine_unavailable("orpheus")
get_qwen3 = _engine_unavailable("qwen3")
get_qwen3_custom = _engine_unavailable("qwen3-custom")
load_ref_audio_mx = _engine_unavailable("mlx-ref-audio")

# MLX engine identifiers — used for dispatch.
MLX_ENGINES = {"csm", "dia", "chatterbox-mlx", "orpheus", "qwen3", "qwen3-custom"}
ALL_ENGINES = {"kokoro", "chatterbox"} | MLX_ENGINES


def _load_voice_profile(yaml_path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(yaml_path).read_text())  # type: ignore[no-any-return]


def _apply_register(profile: dict[str, Any], register: str) -> dict[str, Any]:
    """Deep-merge register-specific overrides into a copy of the voice profile."""
    if register == "baseline" or not register:
        return profile
    registers = profile.get("registers", {})
    overrides = registers.get(register, {})
    if not overrides:
        return profile
    merged = copy.deepcopy(profile)
    for key, val in overrides.items():
        if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
            merged[key].update(val)
        else:
            merged[key] = val
    return merged


def _resolve_voice(kokoro: Any, profile: dict[str, Any]) -> Any:
    """Return voice string or blended numpy embedding."""
    kok = profile.get("kokoro", {})
    voice_a = str(kok.get("voice_id", "af_heart"))
    blend = profile.get("blend", {})
    voice_b = str(blend.get("voice_b", "")) if blend else ""
    ratio = float(blend.get("ratio", 0.0)) if blend else 0.0
    return resolve_kokoro_voice(kokoro, voice_a, voice_b, ratio)


def _load_cast(program_slug: str | None = None) -> dict[str, Any]:
    """Load program config for voice/music resolution.

    Priority: program.yaml (when slug provided) > cast.yaml > empty dict.
    Program.yaml cast/music blocks are mapped into the same structure as
    cast.yaml for backward compatibility with existing render/mix code.
    """
    # Try program.yaml first when a slug is provided
    if program_slug:
        from src.paths import LibraryPaths

        prog_path = LibraryPaths().program_config(program_slug)
        if not prog_path.exists():
            print(
                f"  WARNING: program.yaml not found for '{program_slug}' "
                f"at {prog_path} — falling back to cast.yaml"
            )
        if prog_path.exists():
            data = yaml.safe_load(prog_path.read_text())
            if isinstance(data, dict):
                # Map program.yaml structure into cast.yaml-compatible shape
                cast: dict[str, Any] = {}
                # Slots from cast block
                prog_cast = data.get("cast", {})
                if prog_cast:
                    cast["slots"] = prog_cast
                # Music config
                if "music" in data:
                    cast["music"] = data["music"]
                # Timing
                if "timing" in data:
                    cast["timing"] = data["timing"]
                return cast

    # Fallback: cast.yaml from CWD
    cast_path = Path("cast.yaml")
    if cast_path.exists():
        data = yaml.safe_load(cast_path.read_text())
        return data if isinstance(data, dict) else {}
    return {}


def _apply_overrides(profile: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge experiment overrides onto a voice profile."""
    merged = copy.deepcopy(profile)
    for key, val in overrides.items():
        if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
            merged[key].update(val)
        else:
            merged[key] = val
    return merged


# ── Tag conversion helpers ───────────────────────────────────────────────────

# Chatterbox bracket tags → Dia parenthetical tags
_CHATTERBOX_TO_DIA: dict[str, str] = {
    "[laugh]": "(laughs)",
    "[chuckle]": "(chuckle)",
    "[cough]": "(coughs)",
    "[sigh]": "(sighs)",
    "[gasp]": "(gasps)",
    "[groan]": "(groans)",
    "[sniff]": "(sniffs)",
    "[shush]": "(mumbles)",
    "[clear throat]": "(clears throat)",
}

# Regex matching any Chatterbox bracket tag
_BRACKET_TAG_RE = re.compile(
    r"\[(?:laugh|chuckle|cough|sigh|gasp|sniff|groan|clear throat|shush)\]"
)

# Regex matching any Dia parenthetical tag
_PAREN_TAG_RE = re.compile(
    r"\((?:laughs|chuckle|sighs|gasps|coughs|groans|sniffs|screams|inhales|exhales|"
    r"clears throat|singing|sings|mumbles|beep|claps|applause|burps|humming|sneezes|whistles)\)"
)


def _convert_tags_to_dia(text: str) -> str:
    """Convert Chatterbox [bracket] tags to Dia (parenthetical) tags."""

    def _replace(match: re.Match[str]) -> str:
        return _CHATTERBOX_TO_DIA.get(match.group(0), "")

    return _BRACKET_TAG_RE.sub(_replace, text)


def _strip_all_tags(text: str) -> str:
    """Remove all engine-specific non-speech tags (bracket and parenthetical)."""
    text = _BRACKET_TAG_RE.sub("", text)
    text = _PAREN_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# Orpheus emotion/paralinguistic tags — angle bracket format
_ORPHEUS_TAG_RE = re.compile(
    r"<(laugh|chuckle|sigh|cough|sniffle|groan|yawn|gasp|"
    r"excited|fearful|angry|sad|surprised|disgusted|happy|neutral)>"
)


def _extract_orpheus_tags(text: str) -> tuple[str, str, int]:
    """Extract leading Orpheus tags from text.

    Returns (text_with_tags, text_without_tags, tag_count). The text_with_tags
    is sent to the model (tags affect delivery). The text_without_tags is used
    to estimate how much audio to trim — the tag portion at the start of the
    render gets spoken aloud and needs to be cut.
    """
    # Find all tags at the start of the text
    clean = text.strip()
    tags_found = []
    while True:
        m = _ORPHEUS_TAG_RE.match(clean)
        if m:
            tags_found.append(m.group(0))
            clean = clean[m.end() :].strip()
        else:
            break
    return text.strip(), clean, len(tags_found)


def _trim_orpheus_tag_audio(
    audio: Any,
    sample_rate: int,
    tag_count: int,
) -> Any:
    """Trim spoken tag words from the start of Orpheus audio.

    Orpheus reads emotion tags aloud before the content. The tag word
    creates a burst of energy followed by a brief dip before the real
    speech starts. This function detects that energy dip and trims
    everything before it.

    Strategy: find the first energy dip after initial speech, which
    marks the boundary between the spoken tag and the actual content.
    """
    import numpy as np

    if tag_count == 0:
        return audio

    # RMS in 20ms windows
    window = int(sample_rate * 0.02)
    n_windows = len(audio) // window
    if n_windows < 10:
        return audio

    rms = np.array(
        [np.sqrt(np.mean(audio[i * window : (i + 1) * window] ** 2)) for i in range(n_windows)]
    )

    threshold = np.max(rms) * 0.05  # 5% of peak
    above = rms > threshold

    # Scan first 1.5s for the pattern: speech → dip → speech
    # The dip after the tag word is our trim point
    max_scan = min(int(1.5 / 0.02), n_windows)  # 75 windows = 1.5s
    in_speech = False
    dip_found = False
    trim_window = 0

    for i in range(max_scan):
        if above[i] and not in_speech:
            in_speech = True
        elif not above[i] and in_speech:
            # Found a dip after speech — check if speech resumes
            for j in range(i, min(i + 15, n_windows)):  # look ahead 300ms
                if above[j]:
                    gap_windows = j - i
                    if gap_windows >= 2:  # at least 40ms gap
                        trim_window = i  # trim up to the dip
                        dip_found = True
                    break
            if dip_found:
                break

    if not dip_found:
        # Fallback: trim a fixed amount (~500ms per tag)
        trim_samples = int(sample_rate * 0.5 * tag_count)
        if trim_samples < len(audio) * 0.4:
            return audio[trim_samples:]
        return audio

    trim_samples = trim_window * window
    if trim_samples < len(audio) * 0.4:  # safety: never trim > 40%
        return audio[trim_samples:]
    return audio


# ── MLX engine helpers ───────────────────────────────────────────────────────


def _generate_mlx(model: Any, custom_voice: bool = False, **kwargs: Any) -> tuple[Any, int]:
    """Run mlx-audio model.generate() and return (numpy_audio, sample_rate).

    When custom_voice=True, uses model.generate_custom_voice() instead of
    model.generate() — for Qwen3-TTS 1.7B CustomVoice engine.
    """
    import numpy as np

    if custom_voice:
        results = list(model.generate_custom_voice(**kwargs))
    else:
        results = list(model.generate(**kwargs))
    if not results:
        raise RuntimeError("No audio generated by MLX engine")
    audio = np.array(results[0].audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    sample_rate = results[0].sample_rate
    return audio, sample_rate


def _resolve_ref_audio(profile: dict[str, Any], engine: str) -> str | None:
    """Find valid ref_audio path from engine-specific or top-level profile blocks."""
    # Check engine-specific block first
    engine_key = engine.replace("-", "_")  # chatterbox-mlx → chatterbox_mlx
    engine_block = profile.get(engine_key, {})
    ref = str(engine_block.get("ref_audio", "")) if engine_block else ""
    if ref and Path(ref).exists():
        return ref

    # Check csm/dia/chatterbox blocks
    for block_name in ("csm", "dia", "chatterbox_mlx", "chatterbox"):
        block = profile.get(block_name, {})
        ref = str(block.get("ref_audio", "")) if block else ""
        if ref and Path(ref).exists():
            return ref

    # Check top-level ref_audio
    ref = str(profile.get("ref_audio", ""))
    if ref and Path(ref).exists():
        return ref

    return None


def _build_mlx_kwargs(
    engine: str,
    text: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Build kwargs dict for _generate_mlx() based on engine type.

    CSM:            text, speaker (int), ref_audio (mx.array), ref_text, sampler
    Dia:            text with [S1]/[S2] prefix, ref_audio (mx.array), ref_text, temperature, top_p
    Chatterbox-MLX: text, exaggeration, cfg_weight, temperature, ref_audio (file path)
    Orpheus:        text, voice (named), temperature, top_p, max_tokens
    Qwen3:          text, ref_audio (file path), ref_text
    Qwen3-Custom:   text, speaker, language, instruct (no ref_audio — uses named voices)
    """
    ref_audio_path = _resolve_ref_audio(profile, engine)

    if engine == "csm":
        kwargs: dict[str, Any] = {
            "text": _strip_all_tags(text),
            "max_audio_length_ms": 15000,  # 15s cap — trust natural EOS over silence fill
            "split_pattern": None,  # prevent text splitting that breaks context
        }
        csm_block = profile.get("csm", {})
        kwargs["speaker"] = int(csm_block.get("speaker", 0))
        if ref_audio_path:
            kwargs["ref_audio"] = load_ref_audio_mx(ref_audio_path)
            ref_text = str(profile.get("ref_text", ""))
            if ref_text:
                kwargs["ref_text"] = ref_text

        # CSM sampling control — temperature and top_k per register
        temperature = csm_block.get("temperature")
        top_k = csm_block.get("top_k")
        if temperature is not None or top_k is not None:
            try:
                from mlx_audio.tts.models.sesame.sesame import (
                    make_sampler as csm_make_sampler,
                )

                sampler_kwargs: dict[str, Any] = {}
                if temperature is not None:
                    sampler_kwargs["temp"] = float(temperature)
                if top_k is not None:
                    sampler_kwargs["top_k"] = int(top_k)
                kwargs["sampler"] = csm_make_sampler(**sampler_kwargs)
            except ImportError:
                pass  # mlx-audio CSM not available

        return kwargs

    elif engine == "dia":
        dia_block = profile.get("dia", {})
        speaker_tag = str(dia_block.get("speaker_tag", "S1"))
        if speaker_tag not in ("S1", "S2"):
            speaker_tag = "S1"
        dia_text = _convert_tags_to_dia(text)
        # Strip leftover bracket tags (not converted), keep paren tags
        dia_text = _BRACKET_TAG_RE.sub("", dia_text)
        dia_text = re.sub(r"\s+", " ", dia_text).strip()
        kwargs = {"text": f"[{speaker_tag}] {dia_text}"}
        # Dia supports ref_audio and ref_text for voice conditioning.
        # Note: Dia's native sample rate is 44.1kHz — ref_audio must be
        # loaded at 44100, not 24000 (the CSM default).
        if ref_audio_path:
            kwargs["ref_audio"] = load_ref_audio_mx(ref_audio_path, target_sr=44100)
            ref_text = str(profile.get("ref_text", ""))
            if ref_text:
                kwargs["ref_text"] = ref_text
        # Note: Dia's generate() accepts temperature and top_p but does NOT
        # forward them to _generate() in the current mlx-audio version.
        # These are included for forward-compat when the upstream bug is fixed.
        temperature = dia_block.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        top_p = dia_block.get("top_p")
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        return kwargs

    elif engine == "chatterbox-mlx":
        cb_block = profile.get("chatterbox_mlx", profile.get("chatterbox", {}))
        kwargs = {
            "text": text,  # Chatterbox-MLX handles its own tags
            "exaggeration": float(cb_block.get("exaggeration", 0.5)),
            "cfg_weight": float(cb_block.get("cfg_weight", 0.5)),
            "temperature": float(cb_block.get("temperature", 0.8)),
        }
        if ref_audio_path:
            kwargs["ref_audio"] = ref_audio_path  # file path string
        return kwargs

    elif engine == "orpheus":
        orpheus_block = profile.get("orpheus", {})
        clean_text = _strip_all_tags(text)
        # Dynamic max_tokens based on word count — ~86 audio tokens per
        # second of speech, ~3 words per second = ~29 tokens per word.
        # Add 50% headroom for pauses and varied pacing.
        word_count = len(clean_text.split())
        tokens_per_word = 29
        headroom = 2.5
        dynamic_max = int(word_count * tokens_per_word * headroom)
        min_tokens = 400  # floor for very short segments
        max_tokens = max(min_tokens, dynamic_max)
        kwargs = {
            "text": clean_text,
            "voice": str(orpheus_block.get("voice", "tara")),
            "temperature": float(orpheus_block.get("temperature", 0.6)),
            "top_p": float(orpheus_block.get("top_p", 0.8)),
            "max_tokens": max_tokens,
        }
        split = orpheus_block.get("split_pattern")
        if split is not None:
            kwargs["split_pattern"] = split
        return kwargs

    elif engine == "qwen3":
        qwen3_block = profile.get("qwen3", {})  # noqa: F841 — preserved for upstream sync
        kwargs = {
            "text": _strip_all_tags(text),
        }
        if ref_audio_path:
            kwargs["ref_audio"] = ref_audio_path  # file path string
            ref_text = str(profile.get("ref_text", ""))
            if ref_text:
                kwargs["ref_text"] = ref_text
        return kwargs

    elif engine == "qwen3-custom":
        qc_block = profile.get("qwen3_custom", {})
        kwargs = {
            "text": _strip_all_tags(text),
            "speaker": str(qc_block.get("speaker", "vivian")),
            "language": str(qc_block.get("language", "English")),
            "instruct": str(qc_block.get("instruct", "Natural conversation.")),
        }
        return kwargs

    else:
        raise ValueError(f"Not an MLX engine: {engine!r}")


# ── Mixed-engine rendering ───────────────────────────────────────────────────


def _get_engine_model(engine: str) -> tuple[Any, int]:
    """Load and return (model, native_sample_rate) for any supported engine."""
    if engine == "csm":
        return get_csm()
    elif engine == "dia":
        return get_dia()
    elif engine == "chatterbox-mlx":
        return get_chatterbox_mlx()
    elif engine == "orpheus":
        return get_orpheus()
    elif engine == "qwen3":
        return get_qwen3()
    elif engine == "qwen3-custom":
        return get_qwen3_custom()
    else:
        raise ValueError(f"Unknown MLX engine: {engine!r}")


def _render_segments_mixed(
    config: RadioConfig,
    default_engine: str,
    segments: list[dict[str, Any]],
    voice_profiles: dict[str, dict[str, Any]],
    segments_dir: Path,
    indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Render segments across multiple engines, dispatching per voice profile.

    Each segment is rendered with the engine specified in its voice profile.
    Models are loaded on demand and cached as singletons.
    """
    from src.mixer import resample_audio

    target_sr = config.renderer.sample_rate
    manifest_segments: list[dict[str, Any]] = []
    total = len(segments)

    # CSM context chaining: track rendered segments across engine boundaries
    # so that CSM voices in a mixed-engine episode maintain voice consistency.
    csm_context_segments: list[Any] = []

    for i, seg in enumerate(segments):
        if indices is not None and i not in indices:
            continue

        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        speaker_key = str(seg.get("speaker", "host_a"))
        register = str(seg.get("register", "baseline"))
        topic = str(seg.get("topic", ""))
        profile = _apply_register(voice_profiles.get(speaker_key, {}), register)

        engine = str(profile.get("engine", default_engine))
        ref_audio = _resolve_ref_audio(profile, engine)
        reg_label = f" [{register}]" if register != "baseline" else ""
        char_name = str(profile.get("character_name", speaker_key))
        print(f"  [{i + 1}/{total}] {char_name}{reg_label} ({engine}): {text[:55]}...")

        # Build kwargs and render
        if engine in MLX_ENGINES:
            # Orpheus tag handling
            tag_count = 0
            if engine == "orpheus":
                _text_with_tags, _text_clean, tag_count = _extract_orpheus_tags(text)

            kwargs = _build_mlx_kwargs(engine, text, profile)

            # CSM context chaining: inject sliding window of prior segments
            if engine == "csm" and csm_context_segments:
                kwargs["context"] = csm_context_segments[-2:]

            model, native_sr = _get_engine_model(engine)
            is_custom = engine == "qwen3-custom"
            audio_segment, gen_sr = _generate_mlx(model, custom_voice=is_custom, **kwargs)

            # CSM context chaining: capture rendered audio as Segment for next iteration
            if engine == "csm":
                try:
                    import mlx.core as mx
                    from mlx_audio.tts.models.sesame.sesame import Segment as CsmSegment

                    csm_block = profile.get("csm", {})
                    speaker_id = int(csm_block.get("speaker", 0))
                    audio_mx = mx.array(audio_segment)
                    csm_context_segments.append(
                        CsmSegment(speaker=speaker_id, text=_strip_all_tags(text), audio=audio_mx)
                    )
                except ImportError:
                    pass

            if engine == "orpheus" and tag_count > 0:
                audio_segment = _trim_orpheus_tag_audio(audio_segment, gen_sr, tag_count)

            if gen_sr != target_sr:
                audio_segment = resample_audio(audio_segment, gen_sr, target_sr)
        elif engine == "kokoro":
            import numpy as np

            kokoro, _ = get_kokoro()
            clean_text = _strip_all_tags(text)
            kok = profile.get("kokoro", {})
            voice = _resolve_voice(kokoro, profile)
            speed = float(kok.get("speed", 1.0))
            lang = str(kok.get("lang", "en-us"))
            raw_samples, _ = kokoro.create(clean_text, voice=voice, speed=speed, lang=lang)
            audio_segment = np.array(raw_samples, dtype=np.float32)
        elif engine == "chatterbox":
            import numpy as np

            model, sample_rate_cb = get_chatterbox()
            cb = profile.get("chatterbox", {})
            conds = get_chatterbox_conds(ref_audio) if ref_audio else None
            if conds is not None:
                model.conds = conds
            wav = model.generate(
                text,
                exaggeration=float(cb.get("exaggeration", 0.5)),
                cfg_weight=float(cb.get("cfg_weight", 0.5)),
                temperature=float(cb.get("temperature", 0.8)),
                top_p=float(cb.get("top_p", 0.95)),
                min_p=float(cb.get("min_p", 0.05)),
                repetition_penalty=float(cb.get("repetition_penalty", 1.2)),
            )
            audio_segment = wav.cpu().numpy().squeeze().astype(np.float32)
        else:
            raise ValueError(f"Unknown engine: {engine!r}")

        audio_segment = apply_dsp(audio_segment, profile, target_sr)

        filename = _write_segment(audio_segment, i, speaker_key, segments_dir, target_sr)
        duration = len(audio_segment) / target_sr

        manifest_segments.append(
            {
                "index": i,
                "file": filename,
                "speaker": speaker_key,
                "register": register,
                "topic": topic,
                "word_count": len(text.split()),
                "duration_seconds": round(duration, 3),
                "engine": engine,
            }
        )

    return manifest_segments


# ── Per-segment rendering ────────────────────────────────────────────────────


def _write_segment(
    audio: Any, index: int, speaker: str, segments_dir: Path, sample_rate: int
) -> str:
    """Write a single segment WAV and return the filename."""
    import soundfile as sf

    filename = f"seg-{index:03d}-{speaker}.wav"
    sf.write(str(segments_dir / filename), audio, sample_rate)
    return filename


def _render_segments_kokoro(
    config: RadioConfig,
    segments: list[dict[str, Any]],
    voice_profiles: dict[str, dict[str, Any]],
    segments_dir: Path,
    indices: set[int] | None = None,
    cache: Any = None,
) -> list[dict[str, Any]]:
    """Render each segment as an individual WAV using Kokoro. Returns manifest entries.

    If ``cache`` is a :class:`src.segment_cache.SegmentCache`, segment
    hashes are computed and the cache is consulted before invoking Kokoro.
    Cache hits skip the TTS call entirely. Cache misses render and store.
    """
    import numpy as np
    import soundfile as sf

    from src.segment_cache import compute_segment_hash

    sample_rate = config.renderer.sample_rate
    manifest_segments: list[dict[str, Any]] = []
    total = len(segments)
    kokoro: Any = None

    for i, seg in enumerate(segments):
        if indices is not None and i not in indices:
            continue

        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        # Strip all non-speech tags — Kokoro doesn't understand them
        text = _strip_all_tags(text)
        if not text:
            continue

        speaker_key = str(seg.get("speaker", "host_a"))
        register = str(seg.get("register", "baseline"))
        topic = str(seg.get("topic", ""))
        profile = _apply_register(voice_profiles.get(speaker_key, {}), register)

        # Cache lookup (text post-tag-strip, profile post-register-merge —
        # whatever Kokoro will actually consume).
        segment_hash: str | None = None
        cache_hit = False
        if cache is not None:
            segment_hash = compute_segment_hash(
                text=text,
                speaker=speaker_key,
                register=register,
                voice_profile=profile,
                engine="kokoro",
            )
            dest = segments_dir / f"seg-{i:03d}-{speaker_key}.wav"
            if cache.copy_to(segment_hash, dest):
                cache_hit = True
                cached_audio, _sr = sf.read(str(dest), dtype="float32")
                duration = len(cached_audio) / sample_rate
                print(f"  [{i + 1}/{total}] {speaker_key} (cached {segment_hash}): {text[:60]}...")
                manifest_segments.append(
                    {
                        "index": i,
                        "file": dest.name,
                        "speaker": speaker_key,
                        "register": register,
                        "topic": topic,
                        "word_count": len(text.split()),
                        "duration_seconds": round(duration, 3),
                        "segment_hash": segment_hash,
                        "cache_hit": True,
                    }
                )
                continue

        # Cache miss (or no cache) — render through Kokoro.
        if kokoro is None:
            kokoro, _ = get_kokoro()

        kok = profile.get("kokoro", {})
        voice = _resolve_voice(kokoro, profile)
        speed = float(kok.get("speed", 1.0))
        lang = str(kok.get("lang", "en-us"))

        voice_label = str(kok.get("voice_id", "af_heart"))
        print(f"  [{i + 1}/{total}] {speaker_key} ({voice_label}): {text[:60]}...")

        raw_samples, _phonemes = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        audio_segment = np.array(raw_samples, dtype=np.float32)
        audio_segment = apply_dsp(audio_segment, profile, sample_rate)

        filename = _write_segment(audio_segment, i, speaker_key, segments_dir, sample_rate)
        duration = len(audio_segment) / sample_rate

        # Populate cache after successful render.
        if cache is not None and segment_hash is not None:
            cache.put(segment_hash, segments_dir / filename)

        manifest_entry: dict[str, Any] = {
            "index": i,
            "file": filename,
            "speaker": speaker_key,
            "register": register,
            "topic": topic,
            "word_count": len(text.split()),
            "duration_seconds": round(duration, 3),
        }
        if segment_hash is not None:
            manifest_entry["segment_hash"] = segment_hash
            manifest_entry["cache_hit"] = cache_hit  # always False here
        manifest_segments.append(manifest_entry)

    return manifest_segments


def _render_segments_chatterbox(
    config: RadioConfig,
    segments: list[dict[str, Any]],
    voice_profiles: dict[str, dict[str, Any]],
    segments_dir: Path,
    indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Render each segment as an individual WAV using Chatterbox. Returns manifest entries."""
    import numpy as np

    model, sample_rate = get_chatterbox()

    # Pre-load all voice conditionals
    speaker_conds: dict[str, Any] = {}
    for key, profile in voice_profiles.items():
        cb_block = profile.get("chatterbox", {})
        ref_audio = str(cb_block.get("ref_audio", ""))
        if ref_audio:
            ref_path = Path(ref_audio)
            if ref_path.exists():
                speaker_conds[key] = get_chatterbox_conds(str(ref_path))
                print(f"  Loaded voice conditionals for {key}: {ref_audio}")

    manifest_segments: list[dict[str, Any]] = []
    total = len(segments)

    for i, seg in enumerate(segments):
        if indices is not None and i not in indices:
            continue

        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        speaker_key = str(seg.get("speaker", "host_a"))
        register = str(seg.get("register", "baseline"))
        topic = str(seg.get("topic", ""))
        profile = _apply_register(voice_profiles.get(speaker_key, {}), register)
        cb_block = profile.get("chatterbox", {})
        ref_audio = str(cb_block.get("ref_audio", ""))

        reg_label = f" [{register}]" if register != "baseline" else ""
        print(
            f"  [{i + 1}/{total}] {speaker_key}{reg_label} ({ref_audio or 'no ref'}): {text[:60]}..."
        )

        if speaker_key in speaker_conds:
            model.conds = speaker_conds[speaker_key]

        wav = model.generate(
            text,
            exaggeration=float(cb_block.get("exaggeration", 0.5)),
            cfg_weight=float(cb_block.get("cfg_weight", 0.5)),
            temperature=float(cb_block.get("temperature", 0.8)),
            top_p=float(cb_block.get("top_p", 0.95)),
            min_p=float(cb_block.get("min_p", 0.05)),
            repetition_penalty=float(cb_block.get("repetition_penalty", 1.2)),
        )
        audio_segment = wav.cpu().numpy().squeeze().astype(np.float32)
        audio_segment = apply_dsp(audio_segment, profile, sample_rate)

        filename = _write_segment(audio_segment, i, speaker_key, segments_dir, sample_rate)
        duration = len(audio_segment) / sample_rate

        manifest_segments.append(
            {
                "index": i,
                "file": filename,
                "speaker": speaker_key,
                "register": register,
                "topic": topic,
                "word_count": len(text.split()),
                "duration_seconds": round(duration, 3),
            }
        )

    return manifest_segments


def _render_segments_mlx(
    config: RadioConfig,
    engine: str,
    segments: list[dict[str, Any]],
    voice_profiles: dict[str, dict[str, Any]],
    segments_dir: Path,
    indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Render each segment using an MLX-audio engine (CSM, Dia, Chatterbox-MLX).

    Single function for all 3 MLX engines — loads the correct model once,
    iterates segments with the same pattern. Resamples Dia output from
    44.1kHz to 24kHz before writing.

    CSM context chaining: for the CSM engine, previously rendered segments
    are fed back as context to maintain voice consistency and conversational
    flow across the episode. A sliding window of the last 3 segments keeps
    the context within the model's 2048-token budget.
    """
    from src.mixer import resample_audio

    # Load the correct model
    if engine == "csm":
        model, native_sr = get_csm()
    elif engine == "dia":
        model, native_sr = get_dia()
    elif engine == "chatterbox-mlx":
        model, native_sr = get_chatterbox_mlx()
    elif engine == "orpheus":
        model, native_sr = get_orpheus()
    elif engine == "qwen3":
        model, native_sr = get_qwen3()
    elif engine == "qwen3-custom":
        model, native_sr = get_qwen3_custom()
    else:
        raise ValueError(f"Unknown MLX engine: {engine!r}")

    target_sr = config.renderer.sample_rate
    manifest_segments: list[dict[str, Any]] = []
    total = len(segments)

    # CSM context chaining: track rendered segments to feed as context
    # to subsequent generate() calls for voice/prosody continuity.
    csm_context_segments: list[Any] = []  # list of Segment objects

    for i, seg in enumerate(segments):
        if indices is not None and i not in indices:
            continue

        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        speaker_key = str(seg.get("speaker", "host_a"))
        register = str(seg.get("register", "baseline"))
        topic = str(seg.get("topic", ""))
        profile = _apply_register(voice_profiles.get(speaker_key, {}), register)

        ref_audio = _resolve_ref_audio(profile, engine)
        reg_label = f" [{register}]" if register != "baseline" else ""
        print(
            f"  [{i + 1}/{total}] {speaker_key}{reg_label} "
            f"({engine}, {ref_audio or 'no ref'}): {text[:60]}..."
        )

        # For Orpheus: detect emotion tags, keep them in prompt for delivery,
        # then trim the spoken tag audio from the start of the render.
        tag_count = 0
        if engine == "orpheus":
            _text_with_tags, _text_clean, tag_count = _extract_orpheus_tags(text)
            # Build kwargs with tags preserved (affects delivery)
            kwargs = _build_mlx_kwargs(engine, text, profile)
        else:
            kwargs = _build_mlx_kwargs(engine, text, profile)

        # CSM context chaining: inject sliding window of prior segments
        if engine == "csm" and csm_context_segments:
            kwargs["context"] = csm_context_segments[-2:]  # last 2 segments
            # Keep ref_audio — CSM's generate() elif falls through to
            # default_speaker_prompt() when ref_audio is None, even with context.

        is_custom = engine == "qwen3-custom"
        audio_segment, gen_sr = _generate_mlx(model, custom_voice=is_custom, **kwargs)

        # CSM context chaining: capture rendered audio as a Segment for next iteration
        if engine == "csm":
            try:
                import mlx.core as mx
                from mlx_audio.tts.models.sesame.sesame import Segment as CsmSegment

                csm_block = profile.get("csm", {})
                speaker_id = int(csm_block.get("speaker", 0))
                # Store raw audio at native sample rate (before DSP)
                audio_mx = mx.array(audio_segment)
                csm_context_segments.append(
                    CsmSegment(
                        speaker=speaker_id,
                        text=_strip_all_tags(text),
                        audio=audio_mx,
                    )
                )
            except ImportError:
                pass  # CSM Segment not available

        # Trim spoken tag words from Orpheus output using energy detection
        if engine == "orpheus" and tag_count > 0:
            audio_segment = _trim_orpheus_tag_audio(audio_segment, gen_sr, tag_count)

        # Resample if engine native rate differs from target (Dia: 44.1kHz → 24kHz)
        if gen_sr != target_sr:
            audio_segment = resample_audio(audio_segment, gen_sr, target_sr)

        audio_segment = apply_dsp(audio_segment, profile, target_sr)

        filename = _write_segment(audio_segment, i, speaker_key, segments_dir, target_sr)
        duration = len(audio_segment) / target_sr

        manifest_segments.append(
            {
                "index": i,
                "file": filename,
                "speaker": speaker_key,
                "register": register,
                "topic": topic,
                "word_count": len(text.split()),
                "duration_seconds": round(duration, 3),
            }
        )

    return manifest_segments


# ── Public API ───────────────────────────────────────────────────────────────


def render_segments(
    config: RadioConfig,
    script_path: Path,
    output_dir: Path = Path("output"),
    episode_dir: Path | None = None,
    program_slug: str | None = None,
    indices: set[int] | None = None,
) -> Path:
    """Render each segment to an individual WAV file + manifest JSON.

    When program_slug is provided, voice profiles and music assets are loaded
    from library/programs/{slug}/program.yaml instead of cast.yaml/radio.yaml.

    When indices is provided, only the specified segment indices are re-rendered.
    Existing WAVs and manifest entries for other segments are preserved.

    Returns path to the manifest.json file.
    """
    with script_path.open() as f:
        script: dict[str, Any] = json.load(f)

    # Infer program_slug from script if not provided
    if program_slug is None:
        program_slug = script.get("program")

    date_str = str(script.get("date", script_path.stem.replace("-script", "")))
    title = str(script.get("title", ""))
    segments: list[dict[str, Any]] = script.get("segments", [])

    # Create segments directory inside episode bundle
    if episode_dir is None:
        episode_dir = output_dir / "episodes" / date_str
    segments_dir = episode_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Load voice profiles — program.yaml > cast.yaml > radio.yaml
    cast = _load_cast(program_slug)
    voices: dict[str, str] = dict(config.voices)  # start with radio.yaml
    for slot_key, slot_val in cast.get("slots", {}).items():
        profile_path = slot_val.get("profile") if isinstance(slot_val, dict) else None
        if profile_path:
            voices[slot_key] = profile_path

    voice_profiles: dict[str, dict[str, Any]] = {}
    for key, path in voices.items():
        voice_profiles[key] = _load_voice_profile(path)

    # Resolve engines: per-voice profile engine field, falling back to config default
    default_engine = config.renderer.engine
    engines_in_use: set[str] = set()
    for profile in voice_profiles.values():
        engines_in_use.add(str(profile.get("engine", default_engine)))
    if not engines_in_use:
        engines_in_use.add(default_engine)

    sample_rate = config.renderer.sample_rate

    # Load existing manifest for surgical re-rendering (merge preserved segments)
    existing_manifest_segments: list[dict[str, Any]] = []
    if indices is not None:
        manifest_path = episode_dir / "manifest.json"
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text())
            existing_manifest_segments = existing_manifest.get("segments", [])
        label = ", ".join(str(i) for i in sorted(indices))
        print(f"Surgical re-render: segments [{label}] of {len(segments)} total")

    if len(engines_in_use) == 1:
        # Single-engine path — use the optimized per-engine renderers
        engine = engines_in_use.pop()
        print(f"Rendering segments with engine: {engine}")

        if engine == "chatterbox":
            manifest_segments = _render_segments_chatterbox(
                config, segments, voice_profiles, segments_dir, indices=indices
            )
        elif engine == "kokoro":
            manifest_segments = _render_segments_kokoro(
                config, segments, voice_profiles, segments_dir, indices=indices
            )
        elif engine in MLX_ENGINES:
            manifest_segments = _render_segments_mlx(
                config, engine, segments, voice_profiles, segments_dir, indices=indices
            )
        else:
            raise ValueError(
                f"Unknown renderer engine: {engine!r}. Valid: {', '.join(sorted(ALL_ENGINES))}"
            )
    else:
        # Mixed-engine path — dispatch per segment based on voice profile engine
        engine = "mixed"
        print(f"Rendering segments with mixed engines: {sorted(engines_in_use)}")
        manifest_segments = _render_segments_mixed(
            config, default_engine, segments, voice_profiles, segments_dir, indices=indices
        )

    # Merge: for surgical re-render, keep existing entries for untouched segments
    if indices is not None and existing_manifest_segments:
        re_rendered_indices = {seg["index"] for seg in manifest_segments}
        merged = [
            seg for seg in existing_manifest_segments if seg.get("index") not in re_rendered_indices
        ]
        merged.extend(manifest_segments)
        merged.sort(key=lambda s: s.get("index", 0))
        manifest_segments = merged

    if not manifest_segments:
        raise ValueError("No segments rendered — script may be empty.")

    # Resolve music paths — show palette first, then cast.yaml direct paths
    # (cast already loaded above for voice slot resolution)
    music_config = cast.get("music", {})
    music = {}

    palette_path = music_config.get("show")
    if palette_path and Path(palette_path).exists():
        try:
            from src.show_palette import load_palette

            palette = load_palette(palette_path)
            for cue_type in ("intro", "outro", "sting", "transition"):
                asset_path = palette.assets.get(cue_type)
                if asset_path and Path(asset_path).exists():
                    music[cue_type] = asset_path
        except Exception as exc:
            print(f"  WARNING: Failed to load show palette: {exc}")

    # Fall back to direct paths in music config, then program assets directory
    for key in ("intro", "outro", "sting", "transition"):
        if key not in music:
            # Check music config explicit path
            path = music_config.get(key, "")
            if path and Path(path).exists():
                music[key] = path
            # Check program assets directory
            elif program_slug:
                from src.paths import LibraryPaths

                prog_asset = LibraryPaths().program_assets(program_slug) / f"{key}.wav"
                if prog_asset.exists():
                    music[key] = str(prog_asset)

    # Build cast metadata — maps slot keys to character names from profiles
    cast_meta: dict[str, dict[str, str]] = {}
    for slot_key, profile in voice_profiles.items():
        cast_meta[slot_key] = {
            "character_name": str(profile.get("character_name", slot_key)),
            "profile": str(voices.get(slot_key, "")),
            "engine": str(profile.get("engine", engine)),
        }

    # Write manifest — include music_config so mixer reads ducking params from here
    manifest = {
        "version": 2,
        "date": date_str,
        "title": title,
        "engine": engine,
        "sample_rate": sample_rate,
        "segments_dir": str(segments_dir),
        "cast": cast_meta,
        "segments": manifest_segments,
        "music": music,
        "music_config": {
            "duck_db": float(music_config.get("duck_db", -18)),
            "bed_level_db": float(music_config.get("bed_level_db", -6)),
            "fade_ms": int(music_config.get("fade_ms", 50)),
            "intro_preroll_s": float(music_config.get("intro_preroll_s", 0.0)),
        },
    }
    if program_slug:
        manifest["program"] = program_slug

    # Artwork by convention — program artwork → station fallback → omit
    # episode_dir = library/programs/{slug}/episodes/{date}/ so parents[3] = library root
    if program_slug:
        try:
            _lib_root = episode_dir.parents[3]
            _paths = LibraryPaths(_lib_root)
            _art = _paths.program_artwork(program_slug)
            if not _art.exists():
                _art = _paths.station_artwork()
            if _art.exists():
                manifest["artwork_path"] = str(_art)
        except IndexError:
            pass  # episode_dir too shallow — skip artwork

    manifest_path = episode_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Manifest: {manifest_path} ({len(manifest_segments)} segments)")
    return manifest_path


def render(
    config: RadioConfig,
    script_path: Path,
    output_dir: Path = Path("output"),
    segments_only: bool = False,
    episode_dir: Path | None = None,
    program_slug: str | None = None,
    no_music: bool = False,
) -> Path:
    """Render episode script to audio. Returns path to output file.

    With segments_only=True, returns path to manifest.json (no mixing).
    If episode_dir is set, outputs go there instead of output/episodes/{date}/.
    If program_slug is set, loads cast/music from program.yaml.
    If no_music is True, skip all music overlays (voice-only output).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = render_segments(
        config,
        script_path,
        output_dir,
        episode_dir=episode_dir,
        program_slug=program_slug,
    )

    if segments_only:
        return manifest_path

    from src.mixer import mix

    return mix(manifest_path, no_music=no_music)


# ── Voice audition path (unchanged) ─────────────────────────────────────────


def render_voice_audition(
    voice_profile_path: str,
    script_path: Path,
    experiment_path: Path | None = None,
    output_dir: Path = Path("output/auditions"),
) -> Path:
    """Render a single voice through an audition script for evaluation.

    This is the tight loop for voice fingerprinting: load one voice profile,
    optionally apply experiment.yaml overrides, render all segments (typically
    4 registers, ~40s total), normalize, and write WAV for quality evaluation.

    Args:
        voice_profile_path: Path to voice profile YAML.
        script_path:        Path to audition script JSON (single-speaker).
        experiment_path:    Optional experiment.yaml with voice_overrides to
                            apply on top of the base profile.
        output_dir:         Output directory for audition WAVs.

    Returns:
        Path to the rendered WAV file.
    """
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load voice profile
    profile = _load_voice_profile(voice_profile_path)
    profile_name = Path(voice_profile_path).stem

    # Apply experiment overrides if provided
    if experiment_path and experiment_path.exists():
        exp_data = yaml.safe_load(experiment_path.read_text()) or {}
        overrides = exp_data.get("voice_overrides", {})
        if overrides:
            profile = _apply_overrides(profile, overrides)
            print(f"  Applied experiment overrides from {experiment_path}")

    # Load audition script
    with script_path.open() as f:
        script: dict[str, Any] = json.load(f)
    segments: list[dict[str, Any]] = script.get("segments", [])

    engine = profile.get("engine", "kokoro")
    print(f"Audition: {profile_name} ({engine}), {len(segments)} segments")

    if engine == "chatterbox":
        model, sample_rate = get_chatterbox()

        # Load voice conditionals
        cb_block = profile.get("chatterbox", {})
        ref_audio = str(cb_block.get("ref_audio", ""))
        if ref_audio and Path(ref_audio).exists():
            conds = get_chatterbox_conds(ref_audio)
            print(f"  Loaded voice conditionals: {ref_audio}")
        else:
            conds = None

        chunks: list[Any] = []
        total = len(segments)
        gap_samples = np.zeros(int(sample_rate * 0.3), dtype=np.float32)

        for i, seg in enumerate(segments):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue

            register = str(seg.get("register", "baseline"))
            seg_profile = _apply_register(profile, register)
            seg_cb = seg_profile.get("chatterbox", {})

            reg_label = f" [{register}]" if register != "baseline" else ""
            print(f"  [{i + 1}/{total}]{reg_label}: {text[:60]}...")

            if conds is not None:
                model.conds = conds

            wav = model.generate(
                text,
                exaggeration=float(seg_cb.get("exaggeration", 0.5)),
                cfg_weight=float(seg_cb.get("cfg_weight", 0.5)),
                temperature=float(seg_cb.get("temperature", 0.8)),
                top_p=float(seg_cb.get("top_p", 0.95)),
                min_p=float(seg_cb.get("min_p", 0.05)),
                repetition_penalty=float(seg_cb.get("repetition_penalty", 1.2)),
            )
            audio_segment = wav.cpu().numpy().squeeze().astype(np.float32)
            audio_segment = apply_dsp(audio_segment, seg_profile, sample_rate)
            chunks.append(audio_segment)

            if i < total - 1:
                chunks.append(gap_samples)

        if not chunks:
            raise ValueError("No segments rendered — audition script may be empty.")

        combined = np.concatenate(chunks)

    elif engine in MLX_ENGINES:
        from src.mixer import resample_audio

        # Load the correct MLX model
        if engine == "csm":
            model, native_sr = get_csm()
        elif engine == "dia":
            model, native_sr = get_dia()
        elif engine == "orpheus":
            model, native_sr = get_orpheus()
        elif engine == "qwen3":
            model, native_sr = get_qwen3()
        elif engine == "qwen3-custom":
            model, native_sr = get_qwen3_custom()
        else:
            model, native_sr = get_chatterbox_mlx()

        sample_rate = 24000  # target output rate
        chunks: list[Any] = []
        total = len(segments)
        gap_samples = np.zeros(int(sample_rate * 0.3), dtype=np.float32)

        # CSM context chaining for auditions
        csm_audition_context: list[Any] = []

        for i, seg in enumerate(segments):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue

            register = str(seg.get("register", "baseline"))
            seg_profile = _apply_register(profile, register)

            reg_label = f" [{register}]" if register != "baseline" else ""
            print(f"  [{i + 1}/{total}]{reg_label}: {text[:60]}...")

            kwargs = _build_mlx_kwargs(engine, text, seg_profile)

            # CSM context chaining: inject prior segments.
            # Keep ref_audio in kwargs — CSM's generate() has an elif that
            # falls through to default_speaker_prompt() when ref_audio is None,
            # even if context is non-empty.
            if engine == "csm" and csm_audition_context:
                kwargs["context"] = csm_audition_context[-2:]

            is_custom = engine == "qwen3-custom"
            audio_segment, gen_sr = _generate_mlx(model, custom_voice=is_custom, **kwargs)

            # CSM context chaining: capture for next iteration
            if engine == "csm":
                try:
                    import mlx.core as mx
                    from mlx_audio.tts.models.sesame.sesame import (
                        Segment as CsmSegment,
                    )

                    csm_block = seg_profile.get("csm", {})
                    speaker_id = int(csm_block.get("speaker", 0))
                    csm_audition_context.append(
                        CsmSegment(
                            speaker=speaker_id,
                            text=_strip_all_tags(text),
                            audio=mx.array(audio_segment),
                        )
                    )
                except ImportError:
                    pass

            # Resample if needed (Dia: 44.1kHz → 24kHz)
            if gen_sr != sample_rate:
                audio_segment = resample_audio(audio_segment, gen_sr, sample_rate)

            audio_segment = apply_dsp(audio_segment, seg_profile, sample_rate)
            chunks.append(audio_segment)

            if i < total - 1:
                chunks.append(gap_samples)

        if not chunks:
            raise ValueError("No segments rendered — audition script may be empty.")

        combined = np.concatenate(chunks)

    elif engine == "kokoro":
        kokoro, _ = get_kokoro()
        sample_rate = 24000
        chunks = []
        total = len(segments)
        gap_samples = np.zeros(int(sample_rate * 0.3), dtype=np.float32)

        for i, seg in enumerate(segments):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            text = _strip_all_tags(text)
            if not text:
                continue

            register = str(seg.get("register", "baseline"))
            seg_profile = _apply_register(profile, register)
            kok = seg_profile.get("kokoro", {})
            voice = _resolve_voice(kokoro, seg_profile)
            speed = float(kok.get("speed", 1.0))
            lang = str(kok.get("lang", "en-us"))

            reg_label = f" [{register}]" if register != "baseline" else ""
            print(f"  [{i + 1}/{total}]{reg_label}: {text[:60]}...")

            raw_samples, _ = kokoro.create(text, voice=voice, speed=speed, lang=lang)
            audio_segment = np.array(raw_samples, dtype=np.float32)
            audio_segment = apply_dsp(audio_segment, seg_profile, sample_rate)
            chunks.append(audio_segment)

            if i < total - 1:
                chunks.append(gap_samples)

        if not chunks:
            raise ValueError("No segments rendered — audition script may be empty.")

        combined = np.concatenate(chunks)
    else:
        raise ValueError(
            f"Unknown engine in voice profile: {engine!r}. Valid: {', '.join(sorted(ALL_ENGINES))}"
        )

    # Normalize and write WAV (auditions always output WAV for quality eval)
    from datetime import UTC, datetime

    import soundfile as sf

    combined = normalize_loudness(combined, sample_rate)
    duration = len(combined) / sample_rate

    # Output structure: output/{date}/auditions/{NNN}-{profile}-{engine}/audition.wav
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    auditions_dir = output_dir / date_str / "auditions"
    auditions_dir.mkdir(parents=True, exist_ok=True)

    # Sequential numbering — count existing dirs for today
    existing = sorted(d for d in auditions_dir.iterdir() if d.is_dir())
    n = len(existing) + 1
    audition_dir = auditions_dir / f"{n:03d}-{profile_name}"
    audition_dir.mkdir(parents=True, exist_ok=True)

    out_path = audition_dir / "audition.wav"
    sf.write(str(out_path), combined, sample_rate)
    print(f"  Audition saved: {out_path} ({duration:.1f}s)")
    return out_path


def generate_reference_clip(
    kokoro_profile_path: str,
    script_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Render a reference clip using Kokoro for Chatterbox to clone from.

    Uses the full Kokoro profile (voice blend, speed, DSP) to produce
    a high-quality, prosodically controlled reference. The result can replace
    the short Common Voice clips currently used as Chatterbox references.

    Args:
        kokoro_profile_path: Path to Kokoro voice profile YAML.
        script_path: Path to reference script JSON (autoresearch/reference-script.json).
        output_path: Where to save the WAV. Defaults to voices/ref-{name}.wav.

    Returns:
        Path to the generated reference WAV.
    """
    import numpy as np
    import soundfile as sf

    profile = _load_voice_profile(kokoro_profile_path)
    profile_name = Path(kokoro_profile_path).stem

    if output_path is None:
        output_path = Path(f"voices/ref-{profile_name}.wav")

    # Load reference script
    with script_path.open() as f:
        script: dict[str, Any] = json.load(f)
    segments: list[dict[str, Any]] = script.get("segments", [])

    kokoro, _ = get_kokoro()
    sample_rate = 24000
    chunks: list[Any] = []
    gap_samples = np.zeros(int(sample_rate * 0.3), dtype=np.float32)
    total = len(segments)

    print(f"Generating reference clip: {profile_name} ({total} segments)")

    for i, seg in enumerate(segments):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        # Strip all non-speech tags
        text = _strip_all_tags(text)
        if not text:
            continue

        register = str(seg.get("register", "baseline"))
        seg_profile = _apply_register(profile, register)
        kok = seg_profile.get("kokoro", {})
        voice = _resolve_voice(kokoro, seg_profile)
        speed = float(kok.get("speed", 1.0))
        lang = str(kok.get("lang", "en-us"))

        reg_label = f" [{register}]" if register != "baseline" else ""
        print(f"  [{i + 1}/{total}]{reg_label}: {text[:60]}...")

        raw_samples, _ = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        audio_segment = np.array(raw_samples, dtype=np.float32)
        audio_segment = apply_dsp(audio_segment, seg_profile, sample_rate)
        chunks.append(audio_segment)

        if i < total - 1:
            chunks.append(gap_samples)

    if not chunks:
        raise ValueError("No segments rendered — reference script may be empty.")

    combined = np.concatenate(chunks)
    combined = normalize_loudness(combined, sample_rate)
    duration = len(combined) / sample_rate

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), combined, sample_rate)
    print(f"  Reference clip saved: {output_path} ({duration:.1f}s)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent Radio renderer")
    parser.add_argument("script", help="Script JSON file to render")
    parser.add_argument(
        "--audition",
        action="store_true",
        help="Voice audition mode — render single voice through audition script",
    )
    parser.add_argument(
        "--voice",
        help="Voice profile YAML path (required for --audition)",
    )
    parser.add_argument(
        "--experiment",
        default="autoresearch/experiment.yaml",
        help="Experiment overrides YAML (default: autoresearch/experiment.yaml)",
    )
    parser.add_argument(
        "--segments-only",
        action="store_true",
        help="Output per-segment WAVs + manifest only (no mixing)",
    )
    parser.add_argument(
        "--gen-reference",
        action="store_true",
        help="Generate a Kokoro reference clip for Chatterbox cloning (requires --voice)",
    )
    parser.add_argument(
        "--program",
        help="Program slug (e.g. floating-point) — loads cast and music from program.yaml",
    )
    parser.add_argument(
        "--remix",
        action="store_true",
        help="Re-mix from existing manifest + segment WAVs (no re-rendering). "
        "Pass manifest.json as the script argument.",
    )
    parser.add_argument(
        "--re-render-segment",
        help="Re-render specific segment(s) by index, then re-mix. "
        "Comma-separated: --re-render-segment 5,8,12. "
        "Pass manifest.json as the script argument.",
    )
    parser.add_argument(
        "--no-music",
        action="store_true",
        help="Skip all music overlays — voice-only output.",
    )
    args = parser.parse_args()

    if args.remix:
        from src.mixer import mix

        manifest_path = Path(args.script)
        if not manifest_path.exists():
            parser.error(f"Manifest not found: {manifest_path}")
        result = mix(manifest_path, no_music=args.no_music)
        print(f"  Output: {result}")

    elif args.re_render_segment is not None:
        # Re-render specific segments, overwrite their WAVs, then re-mix
        manifest_path = Path(args.script)
        if not manifest_path.exists():
            parser.error(f"Manifest not found: {manifest_path}")

        indices = [int(x.strip()) for x in args.re_render_segment.split(",")]
        manifest = json.loads(manifest_path.read_text())
        segments_dir = Path(manifest["segments_dir"])
        script_path_from_manifest = manifest_path.parent / "script.json"

        # Load script to get original text
        # Try to find the script — check common locations
        script_data = None
        for candidate in [
            script_path_from_manifest,
            manifest_path.parent / "bard-draft.json",
        ]:
            if candidate.exists():
                script_data = json.loads(candidate.read_text())
                break

        if script_data is None:
            # Try to reconstruct from manifest — we need original text
            parser.error(
                "Cannot find original script. Place script.json or bard-draft.json "
                "in the same directory as the manifest."
            )

        all_segments = script_data.get("segments", [])
        program_slug = args.program or manifest.get("program")

        # Load voice profiles
        cfg = load_config()
        cast = _load_cast(program_slug)
        voices: dict[str, str] = dict(cfg.voices)
        for slot_key, slot_val in cast.get("slots", {}).items():
            profile_path = slot_val.get("profile") if isinstance(slot_val, dict) else None
            if profile_path:
                voices[slot_key] = profile_path

        voice_profiles: dict[str, dict[str, Any]] = {}
        for key, path in voices.items():
            voice_profiles[key] = _load_voice_profile(path)

        default_engine = cfg.renderer.engine
        target_sr = cfg.renderer.sample_rate

        print(f"Re-rendering segments: {indices}")

        for idx in indices:
            if idx >= len(all_segments):
                print(f"  WARNING: segment {idx} out of range, skipping")
                continue

            seg = all_segments[idx]
            text = str(seg.get("text", "")).strip()
            speaker_key = str(seg.get("speaker", "host_a"))
            register = str(seg.get("register", "baseline"))
            profile = _apply_register(voice_profiles.get(speaker_key, {}), register)
            engine = str(profile.get("engine", default_engine))
            char_name = str(profile.get("character_name", speaker_key))

            print(f"  [{idx}] {char_name} ({engine}): {text[:55]}...")

            if engine in MLX_ENGINES:
                kwargs = _build_mlx_kwargs(engine, text, profile)
                model, native_sr = _get_engine_model(engine)
                is_custom = engine == "qwen3-custom"
                audio_segment, gen_sr = _generate_mlx(model, custom_voice=is_custom, **kwargs)
                if gen_sr != target_sr:
                    from src.mixer import resample_audio

                    audio_segment = resample_audio(audio_segment, gen_sr, target_sr)
            elif engine == "kokoro":
                import numpy as np

                kokoro, _ = get_kokoro()
                clean_text = _strip_all_tags(text)
                kok = profile.get("kokoro", {})
                voice = _resolve_voice(kokoro, profile)
                speed = float(kok.get("speed", 1.0))
                lang = str(kok.get("lang", "en-us"))
                raw_samples, _ = kokoro.create(clean_text, voice=voice, speed=speed, lang=lang)
                audio_segment = np.array(raw_samples, dtype=np.float32)
            else:
                print(f"  WARNING: engine {engine} not supported for re-render, skipping")
                continue

            audio_segment = apply_dsp(audio_segment, profile, target_sr)
            filename = _write_segment(audio_segment, idx, speaker_key, segments_dir, target_sr)
            duration = len(audio_segment) / target_sr

            # Update manifest entry
            for m_seg in manifest["segments"]:
                if m_seg["index"] == idx:
                    m_seg["file"] = filename
                    m_seg["duration_seconds"] = round(duration, 3)
                    break

            print(f"    Saved: {filename} ({duration:.1f}s)")

        # Write updated manifest
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print("\nManifest updated. Re-mixing...")

        from src.mixer import mix

        result = mix(manifest_path, no_music=args.no_music)
        print(f"  Output: {result}")

    elif args.gen_reference:
        if not args.voice:
            parser.error("--gen-reference requires --voice <kokoro-profile.yaml>")
        generate_reference_clip(
            kokoro_profile_path=args.voice,
            script_path=Path(args.script),
        )
    elif args.audition:
        if not args.voice:
            parser.error("--audition requires --voice <profile.yaml>")
        render_voice_audition(
            voice_profile_path=args.voice,
            script_path=Path(args.script),
            experiment_path=Path(args.experiment),
        )
    else:
        cfg = load_config()
        result = render(
            cfg,
            Path(args.script),
            segments_only=args.segments_only,
            program_slug=args.program,
            no_music=args.no_music,
        )
        print(f"  Output: {result}")
