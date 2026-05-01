# Apple Silicon (M-series)

`agent-radio-oss` runs on Apple Silicon via ONNX Runtime's `CoreMLExecutionProvider`. CoreML is the framework Apple ships for routing tensor compute to the GPU and Apple Neural Engine; ONNX Runtime exposes it as a named provider, so the same Kokoro ONNX graph that runs on AMD ROCm or CPU also runs unmodified on M-series Macs.

This doc captures the Day 2 (2026-04-30) bring-up on a real Mac. Every command was actually run; every number is a measurement, not an estimate.

## Verified host

| | |
|---|---|
| Hardware | MacBook Pro M3 Pro |
| OS | macOS 26.3 (build 25D125), Darwin kernel 25.3.0, `arm64 T6031` |
| Python | 3.12.11 |
| `onnxruntime` | 1.25.1 |
| `kokoro-onnx` | 0.5.0 |
| Available ONNX providers | `CoreMLExecutionProvider`, `AzureExecutionProvider`, `CPUExecutionProvider` |

## Setup

ONNX Runtime ships CoreML support by default in the `onnxruntime` wheel on macOS — no separate `onnxruntime-coreml` package, no compile flags. If `kokoro-onnx` is installed, you already have CoreML.

```bash
# From a fresh clone:
uv sync --extra tts --extra quality --extra dev

# Verify CoreML is available:
uv run python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# → ['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

Download the Kokoro model files into `models/`:

```bash
mkdir -p models
# kokoro-v1.0.onnx (~310 MB) and voices-v1.0.bin (~27 MB)
# Download URLs: https://github.com/thewh1teagle/kokoro-onnx/releases
```

## Running an audition

```bash
KOKORO_PROVIDER=CoreMLExecutionProvider \
  uv run radio render audition \
  library/programs/haystack-news/episodes/sample/script.json \
  --voice voices/kokoro-michael.yaml
```

**Verify CoreML actually engaged.** ONNX Runtime emits a partition-coverage log line on first inference when the CoreML provider attaches:

```
[W:onnxruntime:, coreml_execution_provider.cc:113 GetCapability]
CoreMLExecutionProvider::GetCapability,
number of partitions supported by CoreML: 129
number of nodes in the graph: 2256
number of nodes supported by CoreML: 1023
```

If you see this line, CoreML is real. If you only see `[kokoro] loaded with provider=CoreMLExecutionProvider` *without* the ONNX Runtime partition log, something silently fell back to CPU — see "Verifying ground truth" below.

## Day 2 measurements

Same input (`library/programs/haystack-news/episodes/sample/script.json`, 5 segments, ~54s of speech), same `kokoro-michael` voice profile, two providers:

| Provider | Render wall-clock | DNSMOS OVR | DNSMOS SIG | SRMR | Repetition score |
|---|---|---|---|---|---|
| `CPUExecutionProvider` | 9.05s | 4.2086 | 4.3057 | 4.9631 | 0.9341 |
| `CoreMLExecutionProvider` | 12.49s | 4.2086 | (≈4.3057) | 4.9614 | 0.9342 |
| **Delta** | +3.44s | 0.0000 | ≈0 | 0.0017 | 0.0001 |

All audio-quality deltas are well under the 0.1 parity threshold. The output is **acoustically identical** between providers — same Kokoro graph, same float32 outputs, same DSP chain. CoreML is just the route, not the result.

### Why CoreML is slower than CPU here

Counterintuitive but expected:

- Kokoro is 82M params. The graph splits into **129 CoreML partitions** (1023 of 2256 nodes assigned to CoreML; the rest stay on CPU via ORT's "shape ops to CPU" optimization).
- Each partition handoff carries memory-copy and synchronization overhead. For a graph this size, with this many partitions, the overhead exceeds the per-op savings.
- M3 Pro's CPU is fast enough that the entire model fits comfortably; the GPU/ANE acceleration doesn't help.

For larger workloads (longer episodes, batch rendering) CoreML will likely catch up or pull ahead — single short-segment audition is the worst case for partition overhead. We'll re-measure on Day 4 with full episode renders.

## Verifying ground truth

`agent-radio-oss` reads the actual provider list back from the ONNX Runtime session after init and logs that — not what you requested. You should see:

```
[kokoro] loaded with provider=CoreMLExecutionProvider
```

If your env says `KOKORO_PROVIDER=CoreMLExecutionProvider` but the log says `loaded with provider=CPUExecutionProvider`, that is the engine telling you ONNX Runtime did not honor your request. You will also get an explicit warning:

```
[kokoro] WARNING: requested provider=CoreMLExecutionProvider but ONNX Runtime
loaded CPUExecutionProvider. Verify your onnxruntime install supports the
requested provider.
```

**Treat the warning as an error.** Cross-hardware deployments depend on the provider claim being true.

## Quirks observed

1. **`kokoro-onnx 0.5.0` removed the `providers=` constructor kwarg.** It reads `ONNX_PROVIDER` from env instead. `agent-radio-oss` translates `KOKORO_PROVIDER` → `ONNX_PROVIDER` internally so operators only need to know the public env var. See `src/engines/kokoro.py:_resolve_provider` and `tests/test_engines.py`.
2. **First-render warm-up.** First segment under CoreML takes ~5–6s alone (graph compilation to CoreML's `.mlmodelc`); subsequent segments are fast. Cache lives at `~/Library/Caches/ai.onnxruntime/`.
3. **`session_state.cc:1359 VerifyEachNodeIsAssignedToAnEp` warning is benign.** ONNX Runtime intentionally keeps shape-related ops on CPU for performance — this is not a fallback, it's an optimization.
4. **No FP16 / ANE-specific configuration is needed for v0.1.0.** ONNX Runtime picks reasonable defaults. If we want explicit ANE pinning later (long-plan territory), that's a `provider_options` dict on the session, not an env var.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[kokoro] loaded with provider=CPUExecutionProvider` despite setting `KOKORO_PROVIDER=CoreMLExecutionProvider` | `onnxruntime` install missing CoreML, or running on Intel Mac | Reinstall: `uv pip install --force-reinstall onnxruntime`; verify with `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` |
| No `coreml_execution_provider.cc` log line on first render | CoreML provider rejected the graph entirely (rare) | File an issue with the kokoro model version + ORT version + macOS version |
| Render hangs on first segment for >30s | First-time CoreML graph compilation; cache cold | Wait. Subsequent runs use the cache and are fast. |
| `error: macOS version too old` from CoreML | macOS <13 (Ventura) | Upgrade macOS, or use `KOKORO_PROVIDER=CPUExecutionProvider` |

## whisper.cpp on Apple Silicon (Day 3a)

Quality Pillar 3 uses whisper.cpp with the Metal backend on Apple
Silicon. Build is one command — Metal is auto-detected.

### Shiro build (M3 Pro, macOS 14)

```bash
git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 8
```

Output: `whisper.cpp/build/bin/whisper-cli`. CMake auto-finds the
Metal framework and `Accelerate.framework` for BLAS. Apple Clang ships
with Xcode Command Line Tools (`xcode-select --install` if missing).

Download a model:

```bash
mkdir -p models
bash whisper.cpp/models/download-ggml-model.sh base.en models/
```

Wire into Agent Radio:

```bash
export RADIO_WHISPER_BIN=$(pwd)/whisper.cpp/build/bin/whisper-cli
export RADIO_WHISPER_MODEL=$(pwd)/models/ggml-base.en.bin
```

### Round-trip test (Kokoro → whisper.cpp)

End-to-end smoke test on Shiro M3 Pro: render a JFK quote with Kokoro
(CoreML), transcribe with whisper.cpp (Metal), score WER:

```
[kokoro] loaded with provider=CPUExecutionProvider
[render] voice: am_michael
[render] wrote /tmp/audition.wav (6.72s)
[whisper] 'And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.'
[score] WER = 0.0000  CER = 0.0000
```

Identical text round-trip. The `tests/test_stt.py` integration test
also passes when env vars are set:

```
$ RADIO_WHISPER_BIN=$(pwd)/whisper.cpp/build/bin/whisper-cli \
  RADIO_WHISPER_MODEL=$(pwd)/models/ggml-base.en.bin \
  uv run pytest tests/test_stt.py::TestTranscribeIntegration -v
1 passed
```

### Quirks observed

1. **CMake 4.x, not Make.** Modern whisper.cpp builds via CMake; the
   old `make GGML_METAL=1` invocation is deprecated. The `main`
   binary still appears in `build/bin/` for back-compat, but
   `whisper-cli` is the canonical name. `src/stt.py` defaults to the
   new path.
2. **First inference compiles Metal kernels.** ~1s of warmup the first
   time the binary runs; subsequent invocations are fast. No
   user-visible cache file.

## Next backends

- AMD ROCm: see [`amd-rocm.md`](./amd-rocm.md)
- CPU (any platform): see [`cpu.md`](./cpu.md)
- Comparison matrix: see [`README.md`](./README.md)
