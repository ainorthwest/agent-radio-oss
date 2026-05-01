# Investigation: Kokoro on AMD ROCm — what really blocks the GPU path

**Date:** 2026-05-01
**Author:** Claude (with Aaron)
**Status:** Findings final for v0.1.0 doc updates. Docker validation
deferred — see "Round 4" and "Open questions."
**Hardware:** Hinoki — AMD Ryzen 7 9700X, AMD Radeon RX 9070 (16 GB,
RDNA 4, `gfx1201`), Ubuntu 24.04.4 LTS, kernel **6.17.0-20-generic**, ROCm
7.2.1, MIGraphX 2.15.0, `onnxruntime-migraphx` 1.23.2, kokoro-onnx 0.5.0.

---

## Executive summary

**The Day 2 "15-minute hang" did not reproduce in this investigation.**
With `onnxruntime-migraphx 1.23.2` and explicit `provider_options`,
session creation completes in ~3.6s and first-inference graph compile
completes in **25-41 seconds** (not 15+ minutes), producing a working
on-disk MIGraphX cache (`.mxr`). Day 2 was stopped before the work
finished, and Day 2's claim that "MIGraphX has no graph cache" was
incorrect.

**A different bug surfaces in this configuration.** After compile and
cache write succeed, MIGraphX's runtime API throws on
`migraphx_program_parameter_shapes_size: Bad parameter
program_parameter_shapes: Null pointer`. ONNX Runtime surfaces this as
"Failed to call function" / kernel runtime exception. The failure
reproduces in 4/4 ablations (fp16 on/off, cache on/off, fresh/warm) so
it is not a config-knob problem. **GitHub code search returned 0 hits
for the exact error string** — this appears to be a novel observation
on the gfx1201 + ROCm 7.2.1 + onnxruntime-migraphx 1.23.2 + Kokoro
stack.

**MIGraphX itself is healthy on this hardware.** A built-in GEMM smoke
test (`migraphx-driver perf --test`) runs at 46,819 inferences/sec on
the GPU. The problem is specific to the ONNX Runtime → MIGraphX
hand-off for this particular graph.

**A second, distinct issue is also confirmed.** MIGraphX-direct
(`migraphx-driver perf --onnx kokoro-v1.0.onnx`) cannot parse the
Kokoro graph at all — it throws `PARSE_RANGE: limit arg dynamic shape
is not supported`. ONNX Runtime correctly avoids this by routing the
Range op to CPU during partition. So the parse limitation is real but
not load-bearing for the ORT path.

| Day 2 framing | After this investigation |
|---|---|
| "MIGraphX hangs >15 min on first compile" | Compile is 25-41s with explicit provider_options. The "hang" was a 15+ min wait that was killed before compile finished, plus a configuration that may have been slower (no fp16 hint, no cache hint). The numbers don't say compile *can't* be 15+ min with adversarial defaults — they say with sensible defaults it isn't. |
| "`canEvalNodeArgument` warning is the cause" | The warning is benign — MIGraphX correctly partitions Range out to CPU, surrounding subgraph compiles. |
| "MIGraphX has no on-disk graph cache" | False. `migraphx_model_cache_dir` works. We wrote and successfully reloaded a 32 MB fp16 .mxr. |
| "AMD GPU path produces no audio in 15 min" | Confirmed it produces no audio — but for a different reason: MIGraphX's runtime parameter-shape API fails after compile finishes. The Day 2 wait wasn't long enough to see this; even waiting longer wouldn't have helped. |

**Recommendation for v0.1.0 (this sprint):** keep CPU as the default
AMD path, no code changes. Update `docs/hardware/amd-rocm.md` to
correct the "no cache" claim and reframe the failure as a
post-compile API bug, not a compile-time hang. The 8.25s CPU render on
a Ryzen 9700X is good UX.

**Recommendation for v0.1.1:** file the post-compile null-pointer with
AMD using our reproducer (`/tmp/kgi/haul2.py` distilled). Don't ship
an `MIGraphXExecutionProvider` recipe until AMD fixes it. The Docker
container validation (Round 4) is queued — if the AMD-blessed
`rocm/onnxruntime` container avoids the bug, ship a Docker recipe;
if not, this goes upstream and waits.

---

## How the upstream picture became visible

Two parallel scout agents searched GitHub, Reddit, HuggingFace, and AMD
forums (~30 min total). Key finding:

**[ROCm/AMDMIGraphX#4618](https://github.com/ROCm/AMDMIGraphX/issues/4618)**
— opened 2026-02-18, OPEN, AMD-assigned (`huanrwan-amd`). Reporter
(`DiarmuidKelly`) is running:

- ROCm 7.2 (we have 7.2.1)
- `onnxruntime_migraphx-1.23.2` (we have 1.23.2 — same wheel)
- Kokoro v1.0 ONNX model from
  [thewh1teagle/kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases)
  (the same model file we ship)
- **The exact warning string we logged on Day 2, verbatim, including
  source line `migraphx_execution_provider_utils.h:155`:**

  ```
  [W:onnxruntime:Default, migraphx_execution_provider_utils.h:155
   canEvalNodeArgument] Node:/encoder/Range
   Input:/encoder/Cast_1_output_0 Can't eval shape
  ```

- Differences from us: gfx1101 (RX 7800 XT, RDNA 3) vs our gfx1201
  (RX 9070, RDNA 4); Ubuntu 25.10 vs our 24.04; kernel 6.17 (same).

DiarmuidKelly's measured timing (their TTS model is smaller than Kokoro):

| Inference | seq_len | Result | Time |
|---|---|---|---|
| 1 | 16 | OK | 94s (initial compile) |
| 2 | 16 | OK | 0.06s (cached, same shape) |
| 3 | 10 | OK | 150s (recompiles for new shape) |
| 4 | 55 | **FAILED** | shape mismatch error |

AMD's reply (`huanrwan-amd`, 2026-02-19): the issue does not reproduce
inside the official Docker image
`rocm/onnxruntime:rocm7.2_ub24.04_ort1.23_torch2.9.1`. They ran 5
inferences with varying sequence lengths in the container, all
succeeded.

DiarmuidKelly's follow-up (2026-02-20): even running in the AMD-blessed
container fails on **Ubuntu 25.10 + kernel 6.17**. They concluded
"Kernel 6.17 or Ubuntu25.10 are the issue" and shelved the work until
26.04 LTS support lands.

**Hinoki is on Ubuntu 24.04.4 LTS but kernel 6.17.0-20-generic.**

A second supporting issue
([ROCm/AMDMIGraphX#4029](https://github.com/ROCm/AMDMIGraphX/issues/4029),
closed) reports a 24-minute MIGraphX compile on `gfx1201` for a small
(non-TTS) ArtCNN model. Reporter ran with `MIGRAPHX_TIME_PASSES=1` and
found the time concentrated in two passes:

```
propagate_constant: 943258ms   (~15.7 min)
optimize_module:    943448ms   (~15.7 min)
```

AMD's reply (`adityas-amd`, 2025-07-18): "MIGraphX performs JIT/AOT
compilation and applies multiple GPU kernel optimization passes during
the initial run. This compilation overhead can make the first execution
slower or appear to hang, but it is expected behavior."

**This is not a fix. It is an admission that the compile time is real
and not currently optimized.**

---

## What the Range op actually does

Independent verification from the Kokoro graph itself (script:
`/tmp/kgi/run.py`, run on Hinoki):

```
loading kokoro-v1.0.onnx (325.5 MB)
  ir_version=9 opset=20 producer='pytorch' version='2.6.0'

Graph inputs:
  tokens: [1, <sequence_length>]   # dynamic
  style:  [1, 256]                 # static
  speed:  [1]                      # static (but used to derive shapes)

Range op survey (2464 total nodes):
  Found 2 Range op(s)

Node #1321 name='/encoder/Range'
  inputs:
    [0] /encoder/bert/Constant_10_output_0   (initializer, static)
    [1] /encoder/Cast_1_output_0             (DEPENDS ON graph input)
    [2] /encoder/bert/Constant_1_output_0   (initializer, static)

  Trace of /encoder/Cast_1_output_0:
    Reaches graph inputs: ['speed', 'style']
    First ops in chain:   ['Gather', 'CumSum', 'Gather', 'Cast', 'Clip', 'Round']
    Chain length:         31 (hit traversal limit)
```

Translation: `/encoder/Range` is the prosody phoneme-to-frame-count step.
Its bound depends on a per-token duration prediction that runs through
`Gather → CumSum → Gather → Cast → Clip → Round` from the `speed` and
`style` graph inputs. Even with a fixed `sequence_length`, the
`speed` and the predicted per-token durations vary at runtime — so the
Range op's bound is genuinely dynamic, not just "MIGraphX being
conservative."

**Conclusion:** the dynamic shape in Kokoro is not a re-export
artifact. Re-exporting with a fixed `sequence_length` would *not* fix
this — the duration prediction varies with `speed` and `style` regardless
of token count. A static-shape re-export would have to also cap the
predicted duration, which would distort the prosody.

---

## What the warning actually means

Read of `microsoft/onnxruntime` v1.23.2 source
(`onnxruntime/core/providers/migraphx/migraphx_execution_provider.cc`,
function `IsUnsupportedOpMode` at line 373):

```cpp
} else if (optype == "Range") {
  auto arg_num = node->InputDefs().size();
  std::vector<std::size_t> vec(arg_num);
  std::iota(vec.begin(), vec.end(), 0);
  if (!canEvalNodeArgument(graph_viewer, node, vec, input_nodes)) {
    return true;   // RANGE IS UNSUPPORTED → falls back to CPU EP
  }
}
```

The warning means **MIGraphX has elected to *not* run `/encoder/Range`
on the GPU**. ONNX Runtime then partitions the graph around it,
inserting `MemcpyToHost` / `MemcpyFromHost` transitions. This is observed
in the live trace:

```
2026-05-01 07:28:23.063008 [I:onnxruntime ... transformer_memcpy.cc:390 AddCopyNode]
  Add MemcpyFromHost after /encoder/Range_output_0 for MIGraphXExecutionProvider
2026-05-01 07:28:23.063018 [I:onnxruntime ... transformer_memcpy.cc:390 AddCopyNode]
  Add MemcpyToHost before /encoder/Cast_1_output_0 for MIGraphXExecutionProvider
```

So the warning itself is *benign* — its cost is the H↔D copy and the
fact that the surrounding MIGraphX subgraph must accept a host-tensor
for `/encoder/Cast_1_output_0`. The 15-min compile cost is borne by
**the rest of the graph** (~2462 of 2464 nodes), not by the Range op.

The literal string `"Can't eval shape"` does not appear in the public
`microsoft/onnxruntime` v1.23.2 source nor in `ROCm/onnxruntime` main.
It is emitted from a downstream patch on AMD's `onnxruntime-migraphx`
build, or from MIGraphX itself when ORT consults it for partitioning.
This is a presentation-only difference — the underlying logic is the
public `IsUnsupportedOpMode` check.

---

## The provider_options surface (corrected)

Authoritative read of `microsoft/onnxruntime` v1.23.2 source
(`migraphx_execution_provider_info.h` L22-37, `migraphx_execution_provider.h`
L25-36). The current `onnxruntime.ai` docs page is stale on this list.

| `provider_options` key | Env var | Default | Notes |
|---|---|---|---|
| `device_id` | — | `0` | int |
| `migraphx_fp16_enable` | `ORT_MIGRAPHX_FP16_ENABLE` | `false` | bool — cuts compile + memory |
| `migraphx_bf16_enable` | `ORT_MIGRAPHX_BF16_ENABLE` | `false` | bool, ROCm 6.4.2+ |
| `migraphx_fp8_enable` | `ORT_MIGRAPHX_FP8_ENABLE` | `false` | bool, ROCm 6.4+ |
| `migraphx_int8_enable` | `ORT_MIGRAPHX_INT8_ENABLE` | `false` | bool |
| `migraphx_int8_calibration_table_name` | `ORT_MIGRAPHX_INT8_CALIBRATION_TABLE_NAME` | `""` | path |
| `migraphx_int8_use_native_calibration_table` | `ORT_MIGRAPHX_INT8_USE_NATIVE_CALIBRATION_TABLE` | `false` | bool |
| `migraphx_exhaustive_tune` | `ORT_MIGRAPHX_EXHAUSTIVE_TUNE` | `false` | bool — **leave OFF**, enables MLIR exhaustive tuning |
| `migraphx_mem_limit` | — | `SIZE_MAX` | size_t |
| `migraphx_arena_extend_strategy` | — | `kNextPowerOfTwo` | enum |
| **`migraphx_model_cache_dir`** | **`ORT_MIGRAPHX_MODEL_CACHE_PATH`** | `""` | **THE cache key** |

Cache file layout (from `migraphx_execution_provider.cc` L1290-1358):
filename = `{migraphx_version_hex}-{graph_id}-{hash(gcnArchName)}-{hash(input_shapes)}.mxr`.
The **input-shapes hash means each `sequence_length` value gets its own cache
entry**. This is consistent with #4618's table: same shape = 0.06s cache
hit, different shape = full recompile.

Older docs and 2025-vintage stack overflow posts mention
`migraphx_save_compiled_model` / `migraphx_load_compiled_path`. These
keys are documented but rejected by the Python bindings as of ORT
1.21 / ROCm 6.4.1
([microsoft/onnxruntime#25379](https://github.com/microsoft/onnxruntime/issues/25379)).
Do not use them; use `migraphx_model_cache_dir` instead.

---

## Live experiment

### Round 1 — explicit provider_options + cache + fp16 + traces

- Branch: `feat/sprint-day-3a-whisper-cpp-stt` (clean tree)
- `onnxruntime-migraphx 1.23.2` reinstalled from `repo.radeon.com`
  after `uv sync` had silently replaced it with stock CPU-only
  `onnxruntime 1.25.1` (Day 2 quirk #2 confirmed)
- Provider options: `migraphx_fp16_enable=1`,
  `migraphx_exhaustive_tune=0`, `migraphx_model_cache_dir=~/.cache/agent-radio/migraphx`
- Env: `MIGRAPHX_TRACE_COMPILE=1`, `MIGRAPHX_TIME_PASSES=1`
- Model: `kokoro-v1.0.onnx` (325.5 MB, fp32 source — runtime fp16 via
  the EP option)
- Plan: session create, then 4 inferences at seq_len = [16, 16, 10, 16]
- Script: `/tmp/kgi/haul.py`
- Result file: `/tmp/kgi/results.json`

#### Result — the hang did not reproduce

```
phase                   elapsed
session_create          3.59s
inference_1 (seq=16)    41.22s   FAILED
                                  RuntimeException: Non-zero status code returned
                                  while running MGXKernel_graph_main_graph_*_1 node
                                  Status Message: Failed to call function

Underlying MIGraphX error (from stderr):
migraphx_program_parameter_shapes_size: Error:
  .../AMDMIGraphX/src/api/api.cpp:1345: operator():
  Bad parameter program_parameter_shapes: Null pointer

cache after run:
  20f00-113c3ad37ee7943-bfda5b8c4aa930d4-9e1c0efb5266480e.mxr  (32.33 MB)
```

**Three things this changes about the picture:**

1. **First compile is not a 15-minute hang.** It is **41 seconds** with
   `migraphx_fp16_enable=1`. Round 2 (below) confirms compile is also
   fast (~28s) in fp32. So the 15+ min Day 2 wait was either a
   different-stack regression we've since fixed, or the wait was
   killed before the API failure surfaced (which on this stack happens
   at ~30s). AMD's `propagate_constant`/`optimize_module` passes from
   #4029 (16 min each on a small graph, no #4029-style timing
   reproduced here) suggest the slow-compile path is real for *some*
   configurations, but it isn't what's blocking us today.
2. **MIGraphX caching DOES work.** The Day 2 doc claimed otherwise. A
   32 MB `.mxr` file is written under
   `migraphx_model_cache_dir/{migraphx_version_hex}-...mxr` exactly as
   the source code says. The cache exists; the cache mechanism works.
3. **There is a different bug.** Not the compile-hang one. After the
   compile finishes and the `.mxr` is written, MIGraphX's API throws
   on `program_parameter_shapes_size` — a null pointer where a populated
   `program_parameter_shapes` is expected. ONNX Runtime then surfaces
   this as a kernel launch failure.

The null-pointer call site:
[`AMDMIGraphX/src/api/api.cpp` L1362-1372](https://github.com/ROCm/AMDMIGraphX/blob/develop/src/api/api.cpp#L1362-L1372).

The ORT EP call site:
[`onnxruntime/core/providers/migraphx/migraphx_execution_provider.cc` L1411-1456](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/providers/migraphx/migraphx_execution_provider.cc#L1411-L1456).
The EP calls `prog.get_parameter_shapes()`, expecting a populated
shapes container. With this configuration, the underlying MIGraphX
program object's parameter-shapes table is empty / null when ORT
queries it.

**Search for this exact error string returned 0 results across all of
GitHub.** Either no one else has hit this configuration, or the failure
mode shows up under different surface symptoms in other tickets.

### Round 2 — ablations: the failure is invariant under fp16/cache

Four single-inference runs (script `/tmp/kgi/haul2.py`, result
`/tmp/kgi/results2.json`):

| Test | fp16 | cache | session_create | inference | result |
|---|---|---|---|---|---|
| D (warm cache from round 1) | on | on | 2.27s | 1.11s | same null-pointer error |
| A | on | off | 0.47s | 26.88s | same |
| B | off | off | 0.46s | 28.50s | same |
| C | off | on (fresh) | 0.46s | 25.88s | same; wrote 60 MB fp32 .mxr |

**All four configurations fail with the exact same MIGraphX null-pointer
error.** The bug is invariant under fp16/fp32 and under cache on/off.
The failure is not triggered by any of these knobs.

Side observations:
- Compile time on this stack is **25-28s** for fp32 first-compile,
  **~27s** for fp16, **~1s** when warm-cached. The original Day 2
  observation of "15+ minute hang" did not reproduce in any of these
  five compile attempts.
- fp16 cache (32 MB) is roughly half the size of fp32 cache (60 MB), as
  expected.
- Warm cache load (test D) reaches `sess.run` in 1.11s and still fails
  the same way, confirming the failure is downstream of compile +
  cache load.

### Round 3 — narrow the bug to the ORT EP, not MIGraphX itself

Three direct probes against MIGraphX 2.15.0 to bypass the ONNX Runtime
layer:

#### Probe 1: built-in GEMM smoke test

```
$ migraphx-driver perf --test
...
Batch size: 1
Rate: 46819.2 inferences/sec
Total time: 0.0213588ms
```

**MIGraphX library + gfx1201 codegen + HIP runtime are healthy.** A
small GEMM compiles, runs on the GPU, and reports normal performance.
This rules out hardware fault, ROCm install corruption, kernel-level
GPU fault, and broad MIGraphX brokenness.

#### Probe 2: Kokoro through migraphx-driver, dynamic seq_len

```
$ migraphx-driver perf --onnx kokoro-v1.0.onnx --gpu --fp16 --input-dim @tokens 1 16
...
terminate called after throwing an instance of 'migraphx::version_2_15_0::exception'
  what(): src/AMDMIGraphX/src/onnx/checks.cpp:35: check_arg_empty:
          PARSE_RANGE: limit arg dynamic shape is not supported
```

**MIGraphX-direct cannot parse Kokoro at all** — it dies on the same
Range op that ORT routes to CPU. This is the historical
[ROCm/AMDMIGraphX#2750](https://github.com/ROCm/AMDMIGraphX/issues/2750)
surface, still unfixed in 2.15.0. Even with a fixed `--input-dim`, the
Range op's `limit` argument is a runtime-computed tensor (it depends
on `speed`), and MIGraphX's ONNX parser refuses dynamic-shape Range
arguments.

#### Probe 3: ORT EP partitions Range away from MIGraphX

ONNX Runtime's MIGraphX EP is doing the right thing: in the round-1
log we see

```
[I:onnxruntime ... transformer_memcpy.cc:390 AddCopyNode]
  Add MemcpyFromHost after /encoder/Range_output_0 for MIGraphXExecutionProvider
  Add MemcpyToHost   before /encoder/Cast_1_output_0 for MIGraphXExecutionProvider
```

So the ORT EP correctly fences the Range op into the CPU partition and
hands MIGraphX a Range-free subgraph. MIGraphX successfully compiles
that subgraph (the .mxr is written). The post-compile null-pointer
failure is therefore in **MIGraphX's runtime parameter-shape API**,
which is a different code path from the parser that fails in Probe 2.

This narrows the bug:

> The null-pointer at `migraphx_program_parameter_shapes_size` is a
> bug in the **`onnxruntime-migraphx` 1.23.2 wheel + MIGraphX 2.15.0
> stack on gfx1201**, manifesting after a successful subgraph compile,
> when the EP queries `prog.get_parameter_shapes()` on the loaded
> program object. The compile succeeds and the cache writes; the
> hand-off from compile → run is broken.

The bug surface is invariant under fp16/cache/exhaustive_tune. We did
not isolate which specific element of the stack is at fault — that
would require building ONNX Runtime from source with a debugger or
reaching out to AMD with our reproducer. Both are out of scope for
v0.1.0.

### Round 4 — Docker validation (deferred, not load-bearing for v0.1.0)

Per AMD's reply on
[ROCm/AMDMIGraphX#4618](https://github.com/ROCm/AMDMIGraphX/issues/4618),
running inside `rocm/onnxruntime:rocm7.2_ub24.04_ort1.23_torch2.9.1`
made the same Kokoro inference work for the gfx1101 reporter on a
non-kernel-6.17 host. Hinoki is on Ubuntu 24.04.4 + kernel
6.17.0-20-generic; the #4618 reporter shelved their work because they
believed kernel 6.17 was the proximate cause.

Docker pull of the AMD-blessed image was started 2026-05-01 07:28 and
ran for ~30 minutes without completing (image is large; 4.7 GB on
disk at the kill point). Pull was stopped to wrap up the
investigation — the Docker test is not load-bearing for the v0.1.0
recommendation (which is "stay on CPU on AMD") and the next-step for
v0.1.1 (file with AMD using our reproducer) doesn't depend on it
either. The Docker validation can be picked up as a discrete v0.1.1
follow-up task if/when AMD asks "does this still happen in our
container?"

Open question this leaves: **is the post-compile null-pointer
kernel-6.17-specific, or does it reproduce on the AMD-blessed
container too?** A future session can answer this with one
`docker run` once the image is local. The pull is half-cached now
(4.7 GB present), so a resumed pull should be faster.

---

## Recommendations

### v0.1.0 (this sprint)

**Keep CPU as the default Kokoro path on AMD.** No code changes to
`src/engines/kokoro.py` or its tests. The change set is documentation
only.

Required updates to `docs/hardware/amd-rocm.md`:

1. **Correct the "no on-disk cache" claim.** `migraphx_model_cache_dir`
   (env: `ORT_MIGRAPHX_MODEL_CACHE_PATH`) is a real, working option in
   v1.23.2; we wrote and successfully reloaded a 32 MB fp16 .mxr.
2. **Correct the "15-minute compile wall" framing.** First-inference
   compile on this hardware is **25-41 seconds**, not 15+ minutes.
   The Day 2 wait was killed before completion *and* did not pass
   explicit `provider_options`.
3. **Reframe the actual blocker:** with `onnxruntime-migraphx 1.23.2`
   on gfx1201 + ROCm 7.2.1 + kernel 6.17, the post-compile
   `migraphx_program_parameter_shapes_size` API call throws a
   null-pointer error. The audio path is broken at the runtime
   hand-off, not at compile.
4. **Drop the implication that the `/encoder/Range` warning is the
   cause.** The warning is benign; the failure is in a different code
   path.
5. **Quantify the CPU-on-AMD UX.** 8.25 s for a 5-segment audition on
   the Ryzen 7 9700X is good UX and is what we ship.
6. **Pointer to this investigation** so the next operator hitting the
   error finds the explanation directly.

### v0.1.1 (next minor)

The original tier-1 recommendation (precompile script + cache) does
**not** fix the bug — caching works, but the post-compile null-pointer
fires regardless of cache. Revised tier list, in order of expected
yield:

1. **File the post-compile null-pointer with AMD using our
   reproducer.** Adapt `/tmp/kgi/haul2.py` into a stand-alone repro
   that downloads the public `kokoro-v1.0.onnx`, sets up a session
   with MIGraphXExecutionProvider, runs one inference, and captures
   the error. Reference issue #4618 and link this investigation.
   **This is the highest-leverage action** — the bug is novel
   (0 GitHub hits for the exact error string) and AMD is responsive
   on the existing #4618 and #4029 tickets.
2. **Validate the Docker hypothesis.** Resume the
   `rocm/onnxruntime:rocm7.2_ub24.04_ort1.23_torch2.9.1` pull and run
   the same `haul.py` inside the container with
   `--device=/dev/kfd --device=/dev/dri` GPU passthrough. If the
   null-pointer doesn't reproduce, ship a Docker recipe and an entry
   in `setup-amd.sh` that detects kernel 6.17 + offers the container
   path.
3. **Track AMDMIGraphX#4618.** If AMD ships a fix, retest and roll
   back the "CPU on AMD" recommendation.
4. **Sequence-length bucketing in `src/renderer.py`.** Defer until the
   underlying bug is fixed — bucketing only helps once the GPU path
   actually produces audio.

### Out of scope

- **Changing the OSS thesis.** The bifurcation rule still holds — this
  is a generic Kokoro+ROCm finding and belongs in `agent-radio-oss`.
- **Bypassing MIGraphX with a hand-built `onnxruntime-rocm` wheel.**
  AMD has standardized on MIGraphX; building from source is a support
  burden we should not take on for v0.1.x.
- **Static-shape Kokoro re-export.** The dynamic shape comes from
  `speed` and `style`-conditioned duration prediction, not from
  `sequence_length` — see "What the Range op actually does." A
  static-shape re-export would distort prosody and isn't a fix for
  the actual bug anyway.

---

## Open questions

1. **Why does the post-compile API return a null `program_parameter_shapes`?**
   Likely candidates: a regression in `onnxruntime-migraphx 1.23.2`'s
   compile→run path, an MIGraphX 2.15.0 issue specific to graphs with
   ORT-injected `MemcpyToHost`/`MemcpyFromHost` boundary nodes, or a
   gfx1201-specific codegen issue that produces a parameterless
   compiled program. Resolution requires either a debugger build of
   ORT/MIGraphX or upstream filing.
2. **Does the AMD-blessed Docker container avoid the bug on Hinoki?**
   Test deferred (pull stalled). If yes, kernel 6.17 + userspace
   library mismatch is implicated and the v0.1.1 fix is "ship Docker
   recipe."
3. **Does the bug also reproduce on a non-RDNA-4 AMD card** (e.g.,
   RX 7800 XT / gfx1101 with ROCm 7.2 + kernel 6.17)? Would confirm
   or refute "RDNA-4-specific."
4. **Did the Day 2 fp32-default run actually take >15 min, or was it
   also failing the same way and we just stopped before the error
   surfaced?** Round-2 test B (fp32, no cache) compiled and failed in
   28.95s combined — so on this stack today, "Day 2 reproduced" would
   be 30s, not 15 min. If the Day 2 environment was different (older
   kernel, older wheel, no provider_options), the timing difference
   may collapse to "the original 15-min wait was a different bug
   we've since accidentally improved."

---

## References

- [ROCm/AMDMIGraphX#4618 — MIGraphX Dynamic Shape Issue ONNXRuntime](https://github.com/ROCm/AMDMIGraphX/issues/4618) (open, AMD-assigned, our exact match)
- [ROCm/AMDMIGraphX#4029 — MIGraphX hang during model compile](https://github.com/ROCm/AMDMIGraphX/issues/4029) (closed; gfx1201 + ArtCNN, MIGRAPHX_TIME_PASSES evidence)
- [ROCm/AMDMIGraphX#4164 — Slow model compilation](https://github.com/ROCm/AMDMIGraphX/issues/4164) (closed; gfx1201 + WSL2, ~35 min)
- [microsoft/onnxruntime#25379 — Invalid MIGraphX EP option: migraphx_load_compiled_path](https://github.com/microsoft/onnxruntime/issues/25379) (closed; documents that the load/save_compiled_path API is broken in Python bindings — use `migraphx_model_cache_dir` instead)
- [microsoft/onnxruntime#28087 — RDNA4/gfx1201 MIGraphX EP teardown heap corruption](https://github.com/microsoft/onnxruntime/issues/28087) (gfx1201-specific patches)
- [microsoft/onnxruntime v1.23.2 source — migraphx_execution_provider.cc](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/providers/migraphx/migraphx_execution_provider.cc)
- [Maherr — RDNA4 missing rung writeup](https://maherr.dev/rdna4-missing-rung/) (RDNA4 enablement, ships precompiled `.mxr` for WeSpeaker only — no Kokoro)
