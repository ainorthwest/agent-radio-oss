---
name: run-station
description: >
  The autonomous-station meta-skill. Orchestrates the full daily loop —
  gather-news → render-episode → check-quality → (edit-script if
  fixable) → publish-episode → broadcast — with deterministic state in
  the episode directory so a crash mid-loop resumes cleanly. This is
  the OSS reference autonomous-station playbook; operators write their
  own taste, voice, and editorial calendar on top.
---

# run-station

The autonomous-station thesis made concrete. A station without an
agent is just a CLI you run by hand. This skill is the agent's
operating manual for running the station autonomously, with optional
human-in-the-loop checkpoints at any step. Auto is the default.

## When to use

- Operating an autonomous station (cron, systemd timer, or agent
  harness loop)
- Producing the day's episode end-to-end without manual intervention
- Demoing the OSS station to a stranger (`radio demo` covers a single
  rendered slice; this skill covers the full loop including
  distribution)

## The loop

```
   ┌─────────────┐
   │ gather-news │  ← config/radio.yaml feeds
   └──────┬──────┘
          │ raw-items.json
          ▼
   ┌─────────────────────┐
   │ write-script        │  ← curator + script-quality gate
   │ (curator in v0.1.0) │
   └──────┬──────────────┘
          │ script.json (validated)
          ▼
   ┌────────────────┐
   │ render-episode │  ← Kokoro ONNX + segment cache
   └──────┬─────────┘
          │ episode.mp3 + manifest.json
          ▼
   ┌───────────────┐
   │ check-quality │  ← three-pillar verdict
   └──────┬────────┘
          │
   verdict=ship ────────────────┐
          │                     │
   verdict=review ──→ HALT (human review or escalate)
          │                     │
   verdict=reject ───┐          │
          │          │          │
          ▼          │          │
   ┌─────────────┐   │          │
   │ edit-script │←──┘          │  ← anomaly→action mapping
   └──────┬──────┘              │
          │                     │
          └→ render-episode (cache short-circuits unchanged segments)
                                ▼
                      ┌─────────────────┐
                      │ publish-episode │  ← Podcasting 2.0, llms.txt
                      └────────┬────────┘
                               ▼
                      ┌───────────┐
                      │ broadcast │  ← R2 + Discourse + RSS + AzuraCast
                      └───────────┘
```

## Decision shape

| Phase outcome | Next step | Recovery |
|---|---|---|
| `gather-news` produces `raw-items.json` with new items | Proceed to script writer | If empty: halt with `warn.no_new_items`; agent decides |
| Script written and passes script-quality gate | Proceed to `render-episode` | If gate fails: re-prompt the script writer or halt |
| Render succeeds, no anomalies | Proceed to `check-quality` | If single segment fails: route to `edit-script`, retry once |
| `check-quality` returns `ship` | Proceed to `publish-episode` then `broadcast` | None — happy path |
| `check-quality` returns `review` | Halt; surface to human or escalate to a higher-capability agent | Do not auto-ship |
| `check-quality` returns `reject` | Route to `edit-script`, re-render, re-check; max two retries | Escalate after two retries |
| `broadcast` partial-failure | Continue with surviving branches; surface failed branch | Operator follow-up |
| Pillar 3 skipped (`wer_skipped: true`) | Note in transcript, continue | Two-pillar verdict is still actionable |

## Resume contract

State lives in `output/episodes/<date>/` (legacy mode) or
`library/programs/<slug>/episodes/<date>/` (library mode). After any
crash, an agent re-invoking this skill should:

1. Inspect the episode directory for partial artifacts
   (`script.json`, `manifest.json`, `quality.json`, `episode.mp3`)
2. Resume from the latest complete artifact:
   - `script.json` present, no `manifest.json` → re-run `render-episode`
   - `manifest.json` present, no `quality.json` → re-run `check-quality`
   - `quality.json` present with `ship` verdict, no broadcast log →
     re-run `broadcast`
3. Cache hits in `data/segment-cache/` make re-renders cheap; cache is
   content-addressed (`sha256(text + speaker + register + voice +
   engine)`), so unchanged segments short-circuit Kokoro.

## What stays upstream

This skill is the **OSS reference autonomous-station playbook**. It is
*deliberately written fresh for OSS* — not ported from the proprietary
`agent-radio` newsroom modules (`bard.py`, `newsroom.py`,
`assignment_editor.py`, `wire_desk.py`, ~3,175 LoC of AINW-flavored
editorial code). Those stay upstream.

What that means for you:

- AINW's editorial calendar (which shows air which days, multi-day
  arcs, beat tracking) → operator territory
- AINW's Steward agent that decides what ships → write your own
  agent's prompt to express your taste
- AINW's Bard agent voice/voice-cast/script-style → operator's own
  prompt corpus and voice cast
- AINW's specific R2 bucket / Discourse / AzuraCast deployment →
  operator's own infrastructure

The OSS station-runner gives you the **shape** of an autonomous
station: which steps in which order with which verdict surfaces. Your
station-runner agent (Hermes profile, Claude Code skill, Gaia bundle,
or your own harness — see [`docs/AGENT_HARNESS.md`](../../docs/AGENT_HARNESS.md))
expresses the taste.

## What this skill is NOT for

- Producing a single slice quickly (use `radio demo`)
- Mid-render debugging (use the underlying skills directly)
- Scheduling — invoke this skill from cron, a systemd timer, or your
  agent harness's loop. v0.1.0 ships no scheduler.

## Decision: ship / review / reject

The meta-decision is whether the **whole loop** ran cleanly:

- All phases succeeded, broadcast happened → station is healthy
- Halted at `review` → human-in-the-loop expected; agent waits
- Halted at `gather` empty → agent waits or rotates source
- Crashed mid-loop → agent resumes per the contract above

## Scripts

- `scripts/loop.py` — sequential orchestrator that invokes each
  skill's wrapper. Exits non-zero on terminal failure; surfaces named
  warns/errors as JSON lines on stderr for harness consumption.

## State contract

- **Reads:** `config/radio.yaml`, the episode directory under
  `output/` or `library/`, env-var secrets for the broadcast branch
- **Writes:** every artifact the underlying skills write
- **Side effects:** the broadcast phase mutates remote state; all
  prior phases are filesystem-only
