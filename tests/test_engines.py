"""Tests for engine provider resolution.

Day 2 of the OSS sprint surfaced that ``kokoro-onnx`` 0.5.0 silently
broke our provider plumbing — the ``providers=`` constructor kwarg was
removed, and the underlying lib reads ``ONNX_PROVIDER`` from env (not
``KOKORO_PROVIDER``). These tests pin the public contract (``KOKORO_PROVIDER``
env var, validated against a whitelist, fallback warning on invalid) so
future kokoro-onnx version drift doesn't silently break cross-hardware
deployments again.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.engines import kokoro


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts with KOKORO_PROVIDER and ONNX_PROVIDER unset."""
    monkeypatch.delenv("KOKORO_PROVIDER", raising=False)
    monkeypatch.delenv("ONNX_PROVIDER", raising=False)


def test_resolve_provider_default_is_cpu() -> None:
    assert kokoro._resolve_provider() == "CPUExecutionProvider"


def test_resolve_provider_honors_valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KOKORO_PROVIDER", "CoreMLExecutionProvider")
    assert kokoro._resolve_provider() == "CoreMLExecutionProvider"


def test_resolve_provider_warns_on_invalid_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("KOKORO_PROVIDER", "BogusExecutionProvider")
    result = kokoro._resolve_provider()
    assert result == "CPUExecutionProvider"
    captured = capsys.readouterr()
    assert "BogusExecutionProvider" in captured.err
    assert "falling back" in captured.err.lower()


@pytest.mark.parametrize(
    "provider",
    [
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
        "CoreMLExecutionProvider",
        "DmlExecutionProvider",
    ],
)
def test_all_documented_providers_pass_validation(
    provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KOKORO_PROVIDER", provider)
    assert kokoro._resolve_provider() == provider


def _patch_kokoro_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the kokoro module's MODEL_DIR / KOKORO_ONNX / KOKORO_VOICES
    at real (empty) files in a temp dir, so the existence guard passes
    without needing the real 350MB models.
    """
    onnx_path = tmp_path / "kokoro-v1.0.onnx"
    voices_path = tmp_path / "voices-v1.0.bin"
    onnx_path.touch()
    voices_path.touch()
    monkeypatch.setattr(kokoro, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(kokoro, "KOKORO_ONNX", onnx_path)
    monkeypatch.setattr(kokoro, "KOKORO_VOICES", voices_path)


def test_get_kokoro_translates_to_onnx_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fix: KOKORO_PROVIDER must be propagated to ONNX_PROVIDER before
    Kokoro init, because kokoro-onnx 0.5.0+ reads ONNX_PROVIDER (not our
    public env var name) when picking its ONNX Runtime backend.

    We verify the translation without actually instantiating Kokoro
    (which would need 350MB of model files). Stub KokoroOnnx to record
    the env at construction time, then assert.
    """
    captured: dict[str, str | None] = {}

    class _StubKokoro:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            captured["onnx_provider"] = os.environ.get("ONNX_PROVIDER")
            self.sess = _StubSession()

    class _StubSession:
        def get_providers(self) -> list[str]:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    import kokoro_onnx

    monkeypatch.setenv("KOKORO_PROVIDER", "CoreMLExecutionProvider")
    monkeypatch.setattr(kokoro_onnx, "Kokoro", _StubKokoro)
    _patch_kokoro_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(kokoro, "_kokoro", None)
    monkeypatch.setattr(kokoro, "_active_provider", None)

    model, sample_rate = kokoro.get_kokoro()
    assert sample_rate == 24000
    assert isinstance(model, _StubKokoro)
    assert captured["onnx_provider"] == "CoreMLExecutionProvider"
    # active_provider() must return ground truth from sess.get_providers(),
    # not what we asked for (they happen to match here, but the contract
    # is "what actually loaded", read from the session).
    assert kokoro.active_provider() == "CoreMLExecutionProvider"


def test_active_provider_reflects_session_when_request_was_overridden(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """If ONNX Runtime didn't honor our request (e.g. user installed plain
    onnxruntime instead of onnxruntime-gpu and asked for CUDA), the session
    falls back to CPU. active_provider() must reflect that, and we must
    warn — operators need to know their hardware claim is unmet.
    """

    class _StubKokoro:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.sess = _StubSession()

    class _StubSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    import kokoro_onnx

    monkeypatch.setenv("KOKORO_PROVIDER", "CUDAExecutionProvider")
    monkeypatch.setattr(kokoro_onnx, "Kokoro", _StubKokoro)
    _patch_kokoro_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(kokoro, "_kokoro", None)
    monkeypatch.setattr(kokoro, "_active_provider", None)

    kokoro.get_kokoro()
    assert kokoro.active_provider() == "CPUExecutionProvider"
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "CUDAExecutionProvider" in err
    assert "CPUExecutionProvider" in err


def test_get_kokoro_restores_prior_onnx_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ONNX_PROVIDER is a process-wide env var; we set it to translate
    KOKORO_PROVIDER for kokoro-onnx 0.5.0+, but we must not leave it
    set after init — future engines that also read ONNX_PROVIDER
    should not be surprised by a stale Kokoro value.
    """

    class _StubKokoro:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.sess = _StubSession()

    class _StubSession:
        def get_providers(self) -> list[str]:
            return ["CoreMLExecutionProvider"]

    import kokoro_onnx

    monkeypatch.setenv("KOKORO_PROVIDER", "CoreMLExecutionProvider")
    monkeypatch.setenv("ONNX_PROVIDER", "PriorValueShouldBeRestored")
    monkeypatch.setattr(kokoro_onnx, "Kokoro", _StubKokoro)
    _patch_kokoro_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(kokoro, "_kokoro", None)
    monkeypatch.setattr(kokoro, "_active_provider", None)

    kokoro.get_kokoro()
    assert os.environ.get("ONNX_PROVIDER") == "PriorValueShouldBeRestored"


def test_get_kokoro_clears_onnx_provider_when_no_prior_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If ONNX_PROVIDER wasn't set before Kokoro init, it should not be
    set after init either — the env should be returned to its original
    state, not left polluted with our translation.
    """

    class _StubKokoro:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.sess = _StubSession()

    class _StubSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    import kokoro_onnx

    monkeypatch.setenv("KOKORO_PROVIDER", "CPUExecutionProvider")
    # _clean_env autouse fixture already deletes ONNX_PROVIDER
    monkeypatch.setattr(kokoro_onnx, "Kokoro", _StubKokoro)
    _patch_kokoro_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(kokoro, "_kokoro", None)
    monkeypatch.setattr(kokoro, "_active_provider", None)

    kokoro.get_kokoro()
    assert "ONNX_PROVIDER" not in os.environ


def test_warns_when_session_introspection_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """If we can't read providers back from the session (sess attr missing,
    get_providers raises, or returns empty), we must explicitly warn —
    silently logging the requested provider as if it were ground truth
    re-introduces the lying log this whole patch exists to fix.
    """

    class _StubKokoroNoSess:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass  # no .sess attribute

    import kokoro_onnx

    monkeypatch.setenv("KOKORO_PROVIDER", "CoreMLExecutionProvider")
    monkeypatch.setattr(kokoro_onnx, "Kokoro", _StubKokoroNoSess)
    _patch_kokoro_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(kokoro, "_kokoro", None)
    monkeypatch.setattr(kokoro, "_active_provider", None)

    kokoro.get_kokoro()
    err = capsys.readouterr().err
    assert "could not read providers" in err
    # Falls back to logging the requested provider, but only after the warning
    assert kokoro.active_provider() == "CoreMLExecutionProvider"
