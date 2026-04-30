"""Kokoro ONNX TTS engine — the OSS default.

Kokoro is 82M params, runs on CPU / CUDA / ROCm / CoreML via ONNX
Runtime providers. The provider is selected at load time via the
``KOKORO_PROVIDER`` env var; falls back to ``CPUExecutionProvider``
if unset.

Internally, ``KOKORO_PROVIDER`` is translated to ``ONNX_PROVIDER``
(the env var ``kokoro-onnx`` 0.5.0+ actually reads) and also passed
via the ``providers=`` constructor kwarg for ``kokoro-onnx`` 0.4.x.
The actual provider used is read back from the ONNX Runtime session
after init — what we log is ground truth, not what was requested.

Voice IDs are named presets (e.g. ``am_michael``, ``af_bella``). Two
voices can be blended via the ``blend:`` block in a voice profile.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

MODEL_DIR = Path("models")
KOKORO_ONNX = MODEL_DIR / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODEL_DIR / "voices-v1.0.bin"
DEFAULT_PROVIDER = "CPUExecutionProvider"

# Providers ONNX Runtime knows about. Used to validate the env var.
_VALID_PROVIDERS = {
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
}

_kokoro: Any = None
_kokoro_sample_rate: int = 24000
_active_provider: str | None = None


def _resolve_provider() -> str:
    """Pick the ONNX Runtime provider from env, with validation."""
    requested = os.environ.get("KOKORO_PROVIDER", DEFAULT_PROVIDER)
    if requested not in _VALID_PROVIDERS:
        print(
            f"[kokoro] WARNING: KOKORO_PROVIDER={requested!r} not recognized; "
            f"falling back to {DEFAULT_PROVIDER}.",
            file=sys.stderr,
        )
        return DEFAULT_PROVIDER
    return requested


def get_kokoro() -> tuple[Any, int]:
    """Return (kokoro_model, sample_rate). Lazy-loads on first call.

    Honors ``KOKORO_PROVIDER`` env var to select the ONNX Runtime
    execution provider. Logs the *actual* provider after init so
    operators can confirm their hardware was picked up — not what
    was requested.
    """
    global _kokoro, _kokoro_sample_rate, _active_provider
    if _kokoro is None:
        try:
            from kokoro_onnx import Kokoro as KokoroOnnx
        except ImportError as exc:
            raise RuntimeError("Kokoro not installed. Run: uv sync --extra tts") from exc

        if not KOKORO_ONNX.exists() or not KOKORO_VOICES.exists():
            raise RuntimeError(
                f"Kokoro model files missing in {MODEL_DIR}/.\n"
                "Download with: bash scripts/download-models.sh"
            )

        provider = _resolve_provider()
        # @gotcha[claude/2026-04-30]: kokoro-onnx 0.5.0+ silently ignores the
        # `providers=` kwarg AND reads `ONNX_PROVIDER` (not `KOKORO_PROVIDER`)
        # from env. We translate to `ONNX_PROVIDER` here and *also* pass
        # `providers=` for 0.4.x backcompat, then read providers back from
        # the session to verify what actually loaded. The env var is scoped
        # to Kokoro init only — restored afterwards so future engines that
        # also touch ONNX Runtime aren't surprised by a stale value.
        prev_onnx_provider = os.environ.get("ONNX_PROVIDER")
        os.environ["ONNX_PROVIDER"] = provider
        try:
            try:
                _kokoro = KokoroOnnx(
                    str(KOKORO_ONNX),
                    str(KOKORO_VOICES),
                    providers=[provider],
                )
            except TypeError:
                _kokoro = KokoroOnnx(str(KOKORO_ONNX), str(KOKORO_VOICES))
        finally:
            if prev_onnx_provider is None:
                os.environ.pop("ONNX_PROVIDER", None)
            else:
                os.environ["ONNX_PROVIDER"] = prev_onnx_provider
        _kokoro_sample_rate = 24000

        actual = _read_session_providers(_kokoro)
        if actual is None:
            print(
                "[kokoro] WARNING: could not read providers from ONNX Runtime "
                "session; logging the requested provider only — actual provider "
                "may differ.",
                file=sys.stderr,
            )
        _active_provider = actual or provider
        if actual and actual != provider:
            print(
                f"[kokoro] WARNING: requested provider={provider} but ONNX Runtime "
                f"loaded {actual}. Verify your onnxruntime install supports the "
                f"requested provider.",
                file=sys.stderr,
            )
        print(f"[kokoro] loaded with provider={_active_provider}", file=sys.stderr)
    return _kokoro, _kokoro_sample_rate


def _read_session_providers(kokoro_model: Any) -> str | None:
    """Read the active provider from the underlying ONNX Runtime session.

    kokoro-onnx exposes the session as ``.sess``; ONNX Runtime sessions
    expose ``get_providers()`` which returns the prioritized list of
    providers actually attached. The first entry is what real inference
    will use. Returns None if we cannot introspect (defensive — never
    block rendering on a log-line nicety).
    """
    sess = getattr(kokoro_model, "sess", None)
    if sess is None:
        return None
    get_providers = getattr(sess, "get_providers", None)
    if get_providers is None:
        return None
    try:
        providers = get_providers()
    except Exception:
        return None
    if not providers:
        return None
    return str(providers[0])


def active_provider() -> str | None:
    """Return the provider in use, or None if Kokoro hasn't loaded yet."""
    return _active_provider


def resolve_kokoro_voice(kokoro: Any, voice_a: str, voice_b: str, blend_ratio: float) -> Any:
    """Return voice string or blended embedding numpy array.

    Two named voices can be linearly mixed in their embedding space
    via the ``blend:`` block in a voice profile. Ratio of 0 returns
    voice_a, 1 returns voice_b, anything in between returns a fresh
    float32 array.
    """
    if not voice_b or blend_ratio <= 0.0:
        return voice_a
    if blend_ratio >= 1.0:
        return voice_b
    style_a = kokoro.get_voice_style(voice_a)
    style_b = kokoro.get_voice_style(voice_b)
    return (style_a * (1.0 - blend_ratio) + style_b * blend_ratio).astype(np.float32)
