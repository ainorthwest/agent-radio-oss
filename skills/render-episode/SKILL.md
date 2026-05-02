---
name: render-episode
description: >
  Render a script.json into an audio episode — TTS each segment with
  Kokoro, mix with music beds, normalize loudness, and write per-segment
  WAVs plus a manifest. Hits the per-segment cache so unchanged segments
  short-circuit Kokoro. Use after `gather-news` and the script writer
  produced a clean script.json. Pairs with `check-quality` and
  `edit-script` for the surgical re-render correction loop.
---

# render-episode

The audio production stage of the autonomous-station loop. This skill
invokes the Kokoro ONNX renderer through the `radio render episode`
CLI, which dispatches each segment to its engine, applies DSP and
loudness normalization, and writes per-segment WAVs into the episode
directory. Segment-level content addressing means re-running on a
mutated script re-renders only what changed.

## When to use

- After a script.json passes the script-quality gate
- After `edit-script` mutated the script and a re-render is required —
  cached segments short-circuit, mutated segments re-Kokoro
- Auditioning a new voice profile via `radio render audition` (one-off,
  not full episode — for that use `render-segment`)

## Decision shape

| Situation | Operation | Next |
|---|---|---|
| Fresh script, no cache yet | `radio render episode <script>` | Proceed to `check-quality` |
| Re-render after `edit-script` | `radio render episode <script>` (cache short-circuits unchanged segments) | Proceed to `check-quality` |
| Single-segment surgical re-render | `radio render segment <script> --index N` | Proceed to `check-quality` for that segment only |
| Voice audition (not full episode) | `radio render audition --voice <profile>` | Inspect output WAV; this is a tuning loop, not the station loop |
| Anomaly detector flagged a segment | Route to `edit-script` first; only re-render after the script mutation | The cache key invalidates automatically |
| Renderer crashes mid-episode | Re-invoke; cache hits restore the partial work, only the failed segment retries | If it crashes again on the same segment, escalate to `edit-script` (text issue) or hardware diagnostics (engine issue) |

## What stays upstream

The proprietary engines — MLX-driven Orpheus / Dia / CSM / Chatterbox
— stay in the upstream `agent-radio` repo. OSS ships **only the Kokoro
ONNX path** because Kokoro has Apache-2.0 weights and runs on CPU /
CUDA / ROCm / CoreML through ONNX Runtime. Voice cloning, AINW's
production voice cast, and reference-audio voice profiles all stay
upstream. The renderer's dispatch table preserves the proprietary
engine entries as `_engine_unavailable` stubs so the test suite still
covers dispatch logic without shipping the closed engines.

If you have your own engine, register it in `src/engines/` following
the Kokoro shape (`render(text, voice_profile_dict, register) -> mono
float32 numpy array at self.sample_rate`) and add it to
`SUPPORTED_ENGINES`.

## What this skill is NOT for

- Mutating the script — that's `edit-script`
- Quality scoring — that's `check-quality`
- Music generation (Stable Audio Open ships in v0.1.1, [GH#9](https://github.com/ainorthwest/agent-radio-oss/issues/9))
- Distribution or streaming — that's `broadcast`

## Decision: ship / review / reject

A render produces audio; the decision happens in `check-quality`. The
render itself either succeeds or fails:

- **Success** — proceed to `check-quality`
- **Failure on a single segment** — anomaly detector flags it; route to
  `edit-script` for the correction loop
- **Failure on the whole episode** — likely a config or hardware issue;
  inspect the renderer log, do not loop more than twice

## Scripts

- `scripts/render.py` — thin wrapper around `radio render episode`

## State contract

- **Reads:** `script.json` from the episode directory, `voices/*.yaml`
  voice profiles, `config/radio.yaml` for engine config and music asset
  paths, `data/segment-cache/` for content-addressed re-render
- **Writes:** per-segment WAVs in the episode directory,
  `manifest.json` (per-segment hashes, durations, cache hit/miss),
  `episode.mp3` after the mixer joins segments
- **Side effects:** none beyond filesystem writes
