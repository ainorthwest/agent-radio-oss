---
name: broadcast
description: >
  Distribute a finished episode — upload MP3 to Cloudflare R2, post
  show notes to Discourse, regenerate the podcast RSS feed, and roll
  the new episode into AzuraCast's live stream playlist. Each branch
  is feature-flagged; missing credentials produce named warnings, not
  failures. Use after `check-quality` returned `ship`.
---

# broadcast

The exit door of the autonomous-station loop. Excellence #6 (broadcast
management) lives here. Distribution is split into independent branches
so an operator running offline-first or local-only doesn't get blocked
on missing R2 / Discourse / AzuraCast configuration.

## When to use

- After `check-quality` returned `verdict: ship` (or operator approved
  a `review`)
- After `publish-episode` produced the derivative content (RSS reads
  `chapters.json`, `episode.txt`, `episode.md`)
- When manually re-publishing — pick the branch you want to re-run

## Branch matrix

| Branch | Required env / config | What happens | Failure mode |
|---|---|---|---|
| **R2 upload** | `AGENT_RADIO_R2_*` (endpoint, access key id, secret, public URL base) | Episode MP3 + transcripts uploaded to Cloudflare R2 (S3-compatible) | Missing credentials → branch skipped with `warn.r2_not_configured`; auth failure → `error.r2_auth` with the AWS error code |
| **Discourse post** | `AGENT_RADIO_DISCOURSE_API_KEY` + `discourse.base_url` in config | Show-notes topic posted to the configured category | Missing key → `warn.discourse_not_configured`; HTTP error → `error.discourse` with status |
| **Podcast RSS** | Library mode (`--program <slug>`) and a configured `feed_root` | RSS XML regenerated, optionally uploaded to R2 | Always runs in library mode; failures are local file-write issues |
| **AzuraCast** | `AGENT_RADIO_AZURACAST_API_KEY` + `azuracast.base_url` + station id | Episode uploaded, swapped into the rolling playlist (one-shot, no scheduler in v0.1.0) | Missing key → `warn.azuracast_not_configured`; API error → `error.azuracast` with response body |

## Decision shape

| Configuration | Operation | Outcome |
|---|---|---|
| All four configured | `radio distribute episode <date>` then `radio distribute feed` then `radio stream update` | Full broadcast — episode is live |
| Only R2 + Discourse | First two commands; skip stream update | Static distribution; live stream untouched |
| Local-only operator | `--no-distribute` upstream; this skill is a no-op | The pipeline already wrote files locally |
| AzuraCast scheduler needed | **v0.1.1 territory** ([roadmap](../../CHANGELOG.md)) | Schedule manually inside AzuraCast for v0.1.0 |

## What stays upstream

- **AzuraCast scheduler** — the production stack runs scheduled
  playlists with editorial calendars; OSS ships only the rolling
  one-shot update because that's the generic primitive
- **AINW Discourse category routing** — which category gets show notes,
  which tags, which pinned-vs-unpinned status — operator territory
- **AINW R2 buckets and CDN domains** — operators bring their own R2
  account; the OSS code doesn't presume a particular endpoint
- **Multi-feed publishing** (different RSS feeds for different audiences)
  — operator-specific; the OSS feed generator targets one feed per
  program

## What this skill is NOT for

- Scheduling future episodes — the v0.1.0 AzuraCast integration is
  a one-shot upload, not a calendar
- Generating show notes from scratch — `publish-episode` writes
  `episode.md` and `episode.txt`; this skill consumes them
- Quality decisions — `check-quality` decides ship/review/reject; this
  skill executes on `ship`

## Decision: ship / review / reject

Broadcast does not produce a verdict — it executes the decision
already taken. Failure modes are observable per branch (warnings or
errors) and surface in the agent's transcript. Common patterns:

- All four branches succeed → episode is fully distributed
- One branch fails (e.g., AzuraCast unreachable) → continue with the
  other branches, surface the failure for operator follow-up
- All branches misconfigured → skill is effectively a no-op; surface
  `warn.no_distribution_configured` so the operator knows the local
  files are the only artifact

## Scripts

- `scripts/broadcast.py` — thin orchestrator that invokes
  `radio distribute episode`, `radio distribute feed`, and
  `radio stream update` in sequence, treating each as best-effort

## State contract

- **Reads:** `episode.mp3`, `chapters.json`, `episode.md`,
  `episode.txt` from the episode directory; `config/radio.yaml` for
  endpoint configuration; env vars for secrets
- **Writes:** distribution manifest (success/failure per branch);
  podcast RSS XML (if library mode); AzuraCast media library state
  (remote)
- **Side effects:** **all four branches mutate remote state** (R2
  bucket, Discourse forum, AzuraCast station). This is the only skill
  in the OSS bundle with externally observable side effects. Treat its
  invocation accordingly — don't run it during testing of upstream
  skills.
