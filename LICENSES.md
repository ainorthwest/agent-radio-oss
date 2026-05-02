# Third-party licenses

`agent-radio-oss` is Apache 2.0 (see [LICENSE](./LICENSE)). This document
records the license of every component bundled with or required by
`v0.1.0-mvp`, so a commercial operator can confirm deployability without
reading `uv.lock`.

**Last audit:** 2026-05-02 against commit `8a13b70`
(branch `docs/sprint-day-6-readme-and-licenses`, post-parselmouth-removal).

## Summary

| Layer | Verdict |
|---|---|
| First-party code | Apache-2.0 ✓ commercial-OK |
| Bundled model weights (v0.1.0) | All permissive (Apache-2.0 / MIT) ✓ commercial-OK |
| Required runtime dependencies | All permissive ✓ commercial-OK |
| Optional `[tts]` extras | All permissive **except `pedalboard` (GPL-v3)** — see [The GPL-v3 question](#the-gpl-v3-question) |
| Optional `[tts]` transitive | Includes **`phonemizer-fork` (GPL-v3-or-later)** via `kokoro-onnx` — see [The GPL-v3 question](#the-gpl-v3-question) |
| Optional `[quality]` extras | All permissive ✓ |
| Optional `[distribute]` / `[dev]` | All permissive ✓ |
| Deferred to v0.1.1 | Stable Audio Open (Stability AI Community License) — re-audited when [GH#9](https://github.com/ainorthwest/agent-radio-oss/issues/9) lands |

**No AGPL, SSPL, CPL/CPLv1, CC-BY-NC, or Stability AI Community License
packages are present in v0.1.0.**

## First-party

| Component | License | File |
|---|---|---|
| `agent-radio-oss` source | Apache-2.0 | [LICENSE](./LICENSE) |

Copyright 2026 Lightcone Studios LLC.

## Bundled model weights

Downloaded by [`scripts/download-models.sh`](./scripts/download-models.sh)
(SHA-pinned) on first install.

| Asset | License | Source | SHA-pinned |
|---|---|---|---|
| Kokoro v1.0 ONNX (`kokoro-v1.0.onnx`) | Apache-2.0 | `github.com/thewh1teagle/kokoro-onnx` releases (`model-files-v1.0`, Jan 28 2025) | yes |
| Kokoro voices bundle (`voices-v1.0.bin`) | Apache-2.0 | (same release) | yes |
| Whisper ggml `base.en` (default) | MIT | `huggingface.co/ggerganov/whisper.cpp` | yes |
| Whisper ggml `small.en` (opt-in) | MIT | (same) | no — requires `--allow-unpinned` |
| Whisper ggml `medium.en` (opt-in) | MIT | (same) | no — requires `--allow-unpinned` |

The Whisper model weights are the upstream OpenAI Whisper checkpoints
(MIT, [`openai/whisper`](https://github.com/openai/whisper)) repackaged
in ggerganov's GGML format.

## Required runtime dependencies (`pyproject [project]`)

| Package | Version | License |
|---|---|---|
| `httpx` | `>=0.27.0` | BSD-3-Clause |
| `pyyaml` | `>=6.0` | MIT |
| `openai` | `>=1.0.0` | Apache-2.0 |
| `setuptools` | `>=82.0.0` | MIT |
| `typer` | `>=0.15.0` | MIT |

## Optional extras

### `[tts]`

| Package | Version | License | Notes |
|---|---|---|---|
| `kokoro-onnx` | `>=0.4.0` | MIT | Wrapper. Weights are Apache-2.0 (above). Pulls `phonemizer-fork` transitively (GPL-v3-or-later — see below). |
| `soundfile` | `>=0.12.0` | BSD-3-Clause | |
| `numpy` | `>=1.24.0` | BSD-3-Clause + (transitive: 0BSD, MIT, Zlib, CC0-1.0) | |
| `pedalboard` | `>=0.9.0` | **GPL-v3** | Used by [`src/dsp.py`](./src/dsp.py) for the mastering chain (limiter, compressor, EQ). See [The GPL-v3 question](#the-gpl-v3-question). |

### `[quality]`

| Package | Version | License |
|---|---|---|
| `librosa` | `>=0.10.0` | ISC |
| `soundfile` | `>=0.12.0` | BSD-3-Clause (same as `[tts]`) |
| `numpy` | `>=1.24.0` | BSD-3-Clause (same as `[tts]`) |
| `matplotlib` | `>=3.8.0` | Matplotlib License (PSF-derived, permissive) |
| `torchmetrics[audio]` | `>=1.6.0` | Apache-2.0; transitive: `torch` (BSD-3), `torchaudio` (BSD), `pesq` (MIT), `pystoi` (MIT) |

### `[distribute]`

| Package | Version | License |
|---|---|---|
| `boto3` | `>=1.34.0` | Apache-2.0 (transitive: `botocore` Apache-2.0, `s3transfer` Apache-2.0, `jmespath` MIT) |

### `[dev]`

| Package | Version | License |
|---|---|---|
| `pytest` | `>=8.0.0` | MIT |
| `ruff` | `>=0.3.0` | MIT |
| `mypy` | `>=1.8.0` | MIT |
| `types-PyYAML` | `>=6.0.0` | Apache-2.0 |

## Transitive scan

Generated from `uv pip list --format json` against
`uv sync --extra tts --extra quality --extra distribute --extra dev`
on 2026-05-02 against commit `8a13b70`. The full venv resolved to **108
packages**.

License distribution (verified by reading each `<pkg>.dist-info/METADATA`
`License:`, `License-Expression:`, and `Classifier: License :: OSI
Approved` fields):

- **Permissive (commercial-OK):** MIT, BSD-2-Clause, BSD-3-Clause,
  Apache-2.0, ISC, PSF-2.0, MPL-2.0, MIT-CMU, Matplotlib License (PSF-derived),
  CC0-1.0, 0BSD, Zlib.
- **Weak copyleft (commercial-OK with attribution; dynamic linking
  uncontested):** LGPL-2.1-or-later (`soxr`).
- **Strong copyleft (combined-work contagion at distribution time):**
  GPL-v3 (`pedalboard`), GPL-v3-or-later (`phonemizer-fork`).
- **None of the following were found:** AGPL (any version), SSPL,
  CPL/CPLv1, CC-BY-NC (any version), Stability AI Community License,
  proprietary EULAs.

The transitive list is reproducible from `uv.lock` via the same command.

## The GPL-v3 question

Two GPL-v3 packages are present in `[tts]`:

1. **`pedalboard`** (Spotify, GPL-v3) is a *first-order* dependency.
   It is imported by [`src/dsp.py`](./src/dsp.py) at
   [line 89](./src/dsp.py#L89) (full mastering chain) and
   [line 231](./src/dsp.py#L231) (soft-import fallback to `np.clip`).
2. **`phonemizer-fork`** (GPL-v3-or-later) is a *transitive* dependency
   pulled unconditionally by `kokoro-onnx`. It is hard-imported by
   `kokoro_onnx/tokenizer.py:7-8` for grapheme-to-phoneme conversion.

Python imports of GPL-v3 modules at runtime create a *combined work*
under GPL-v3's terms when **distributed**. The `agent-radio-oss` source
code itself remains Apache-2.0; the GPL-v3 obligations attach to the
distribution of the combination.

### What this means for operators

| Use case | GPL-v3 trigger? |
|---|---|
| Internal SaaS / hosted radio station (no binary distributed) | **No.** GPL-v3 has no network-use clause (that is AGPL, which is not present). |
| Self-host the source repo as-is | **No.** The repo declares the dep; it does not bundle it. |
| Distribute a Docker image / installer / forked repo with `[tts]` installed | **Yes.** This is GPL-v3 distribution. Source must be made available on request per GPL-v3 §4. The Apache-2.0 source code stays Apache-2.0; the *combined* binary inherits GPL-v3 obligations for the bundled GPL-v3 packages. |
| Distribute the source repo + a build script (no installed `.venv`) | **No.** The source on its own is Apache-2.0. |

### Why this is tractable

- GPL-v3 (unlike AGPL) does not have a network-use clause. Cloud and
  SaaS operators are unaffected as long as they do not *distribute*
  binaries to end users.
- The `kokoro-onnx` Python package itself is MIT; the GPL-v3 contagion
  comes from `phonemizer-fork`, which exists primarily as an espeak-ng
  bridge. Permissive replacements exist (e.g., direct espeak-ng calls,
  alternative G2P libraries) and an upstream change to `kokoro-onnx`
  could remove the contagion entirely.
- `pedalboard` covers a well-bounded surface (DSP / mastering) that has
  permissive substitutes (`pyloudnorm` BSD-3, `scipy.signal`, custom
  limiter) suitable for a `v0.1.1` swap.

### `v0.1.1` plan

Two follow-up issues will be opened with this PR:

- **Replace `pedalboard` with permissive equivalents.** Target: drop
  GPL-v3 from `[tts]` first-order. Likely components: `pyloudnorm`
  (BSD-3) for loudness, `scipy.signal` for filters, custom limiter.
- **Audit `kokoro-onnx` upstream for permissive G2P alternatives.**
  Either a downstream patch, an upstream PR to `kokoro-onnx`, or a
  fork that uses a permissive phonemizer.

After both land, the entire `[tts]` install path becomes permissive end
to end.

## Deferred (`v0.1.1`)

- **Stable Audio Open** — Stability AI Community License. Commercial use
  permitted up to a published revenue threshold (verify the current
  threshold at <https://stability.ai/community-license-agreement> at
  install time). Will be re-audited and added to this document when
  [GH#9](https://github.com/ainorthwest/agent-radio-oss/issues/9) lands.
  `v0.1.0` does **not** generate music — it overlays pre-rendered assets.

## How this audit was done

- **Codebase license:** [`pyproject.toml`](./pyproject.toml) `license = "Apache-2.0"` cross-checked with [`LICENSE`](./LICENSE) (Apache 2.0 with `Copyright 2026 Lightcone Studios LLC`).
- **Bundled model weights:** SHA-pinned URLs in [`scripts/download-models.sh`](./scripts/download-models.sh); license confirmed at upstream source on 2026-05-02.
- **Python deps (108 packages):** `uv pip list --format json` against the venv at commit `8a13b70`, then for each package: read `.venv/lib/python3.12/site-packages/<pkg>.dist-info/METADATA` `License:`, `License-Expression:`, and `Classifier: License :: OSI Approved` fields. Cross-checked any `License: UNKNOWN` rows against PyPI metadata.
- **Last audit:** 2026-05-02 against commit `8a13b70`.

To reproduce:

```bash
uv sync --extra tts --extra quality --extra distribute --extra dev
uv pip list --format json
# Then iterate over .venv/lib/python3.12/site-packages/*.dist-info/METADATA
# extracting License / License-Expression / Classifier: License lines.
```
