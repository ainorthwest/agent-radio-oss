---
name: check-quality
description: >
  Score a rendered episode (or single segment) across the three-pillar
  quality stack — librosa spectral, torchmetrics perceived (DNSMOS /
  SRMR / PESQ / STOI), whisper.cpp round-trip WER. Read the named
  verdict (`ship` / `review` / `reject`) and the verdict_reason field;
  do not re-derive thresholds. Pillar 3 may be skipped when whisper.cpp
  is unavailable — check `wer_skipped` and decide accordingly.
---

# check-quality

Excellence #5 (audio engineering) lives here. Quality scoring is
deliberately verdict-shaped: an agent reads `quality.json` and acts on
the named field rather than interpreting raw scores. Three pillars, 23
metrics, one verdict, one verdict_reason.

## When to use

- After `render-episode` finishes and produced `episode.mp3` or
  per-segment WAVs
- Before `broadcast` — verdict gates whether the episode ships, goes to
  human review, or routes back to `edit-script`
- When debugging a degraded audio output — single-segment scoring
  isolates which voice / which segment is the offender

## Decision shape

| Verdict | Score range | Action |
|---|---|---|
| `ship` | overall ≥ 0.7 | Proceed to `broadcast` — confidence is high |
| `review` | 0.5 ≤ overall < 0.7 | Halt the auto-loop. Surface to a human or escalate to a richer agent. Do not auto-ship. |
| `reject` | overall < 0.5 | Route to `edit-script` for the correction loop. Re-render and re-check at most twice; then escalate. |
| `hold` (script-quality) | script gate failed | Route back to `gather-news` or `write-script`; do not render |

### Pillar 3 skip handling

When `wer_skipped: true` is set in `quality.json`, Pillar 3 (round-trip
WER intelligibility) did not run — almost always because `whisper.cpp`
is missing or its binary failed. The `wer_skip_reason` field names the
cause. Decision rule for the agent:

- `wer_skipped: true` is **not** an automatic block on shipping.
  Pillars 1 (spectral) and 2 (perceived) still produce a verdict.
- Surface the skip in agent-visible output (`warn.pillar3_skipped`) so
  the operator knows the verdict is two-pillar instead of three.
- If the operator runs without `whisper.cpp` long-term, note in their
  station's runbook that Pillar 3 is structurally absent — the verdict
  remains useful but is less stringent on garbled-text catches.

## What stays upstream

The quality stack itself ships in OSS — librosa, torchmetrics,
whisper.cpp are all permissively licensed. **The AINW-specific
weighting rubric** (which metrics matter most for which show, which
voice profile, which time of day) stays upstream. Operators tune
weights in `config/quality.yaml` — your taste, your show, your call.
The OSS thresholds (`SHIP_THRESHOLD = 0.7`, `REVIEW_THRESHOLD = 0.5`)
are honest defaults; tune them for your station.

Reference profiles (`voices/<engine>/reference.json`) used for
similarity scoring are also operator-supplied. AINW's production
reference fingerprints stay upstream. The OSS shipping default is no
reference — Pillar 1 still produces a standalone score, just without
a similarity comparison.

## What this skill is NOT for

- Re-rendering — that's `render-episode` / `edit-script`
- Mutating the script — that's `edit-script`
- Distribution decisions beyond the verdict — that's `broadcast`
- Tuning voice profiles based on quality scores (Excellence #7
  autoresearch — v0.1.1+, deliberately not in OSS yet)

## Decision: ship / review / reject

The verdict field IS the decision. An agent that re-derives the
verdict from raw scores is doing redundant work and will drift from
the quality module's source of truth (`SHIP_THRESHOLD`,
`REVIEW_THRESHOLD` live in `src/quality.py`). Read the verdict, read
verdict_reason, act.

## Scripts

- `scripts/check.py` — thin wrapper around the standalone quality
  module (`python -m src.quality <audio>`)

## State contract

- **Reads:** `episode.mp3` or per-segment WAVs, optional `manifest.json`
  for per-segment scoring, optional `script.json` for Pillar 3 WER,
  `config/quality.yaml` (or default thresholds), reference profile
  if configured
- **Writes:** `quality.json` (the verdict surface), per-segment
  `quality.json` entries when `--manifest` is supplied
- **Side effects:** invokes `whisper.cpp` subprocess for Pillar 3 when
  `script_text` is supplied; otherwise three-pillar minus WER
