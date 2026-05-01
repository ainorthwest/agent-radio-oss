---
name: publish-episode
description: >
  Generate derivative content from a rendered episode — markdown,
  Podcasting 2.0 chapters, agent-readable transcript, schema.org
  JSON-LD. Use after a render passes the quality gate, before
  distributing. Outputs are deterministic — re-running on the same
  inputs produces byte-identical files.
---

# publish-episode

Closes the "frictionless for agents who can't hear the stream" half
of the autonomous-station thesis. The script is the source of truth;
the publisher derives a half-dozen text artifacts so listeners,
agents, search indexes, and downstream blog publishing all see the
same content shape.

## When to use

- After `radio render episode` finishes
- Before `radio distribute` — Podcasting 2.0 RSS tags reference
  these files (`<podcast:transcript>`, `<podcast:chapters>`)
- Whenever an editor (`radio edit script`) changes the source script
  and you've re-rendered

## Outputs

| File | Purpose | Determinism |
|---|---|---|
| `episode.md` | Canonical markdown with YAML frontmatter, full transcript body | Pure function of script + manifest |
| `chapters.json` | Podcasting 2.0 cloud-chapters spec (one chapter per segment) | Pure function of manifest |
| `episode.txt` | Plain-text transcript with speaker attribution — agent payload | Pure function of script + manifest |
| `episode.jsonld` | schema.org PodcastEpisode JSON-LD | Pure function of script + manifest |
| `description.txt` | LLM-generated short summary (≤200 words) | Cached by `hash(prompt + script)` — landing in v0.1.1 |
| `social/{linkedin,bluesky}.txt` | LLM-generated short-form copy | Cached, landing in v0.1.1 |

The deterministic outputs ship in v0.1.0. The LLM outputs land in a
follow-up commit so the deterministic surface stays bisectable.

## Decision shape

| Situation | Action |
|---|---|
| Just finished render, anomalies clean | `radio publish episode <date_dir>` then `radio distribute` |
| Re-published existing episodes (tag bump, config change) | `radio publish episode` per episode, then `radio publish llms-index` |
| Adding a new show | `radio publish llms-index <program-dir> --show-name X --description Y` |

## What this skill is NOT

- Audio processing — that's mixer/renderer territory
- Re-rendering — that's `edit-script`
- Distribution — that's `radio distribute`

The publisher writes files into the episode directory. It does not
push to R2, post to Discourse, or hit AzuraCast. Distribution stages
read these files separately.

## Scripts

- `scripts/publish.py` — thin wrapper around `radio publish episode`
