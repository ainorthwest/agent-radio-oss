---
name: gather-news
description: >
  Gather raw items for an episode — fetch RSS feeds, pull Discourse
  activity, deduplicate stories. Generic ingestion only. The agent
  decides which sources to consult and how to weight them; this skill
  performs the fetch, surfaces network failures with actionable
  remedies, and never silently drops stories. Use as the first step in
  the autonomous-station loop, before `write-script` or `render-episode`.
---

# gather-news

The autonomous station's wire desk — generic ingestion. The agent
configures sources in `config/radio.yaml` (or a per-show YAML), and this
skill executes the fetch, normalizes the output, and writes a stable
`raw-items.json` into the episode directory. It is deliberately simple:
no scoring, no editorial judgment, no taste rubric. Those decisions
belong to the agent's prompt, not to OSS code.

## When to use

- Starting a new episode — first call in the autonomous-station loop
- A previous run's `raw-items.json` is stale (cron rotation, time-window expired)
- Debugging a thin episode — call this skill alone to inspect what the
  fetch actually returned before blaming the script writer
- Adding a new source — drop a feed URL in `config/radio.yaml` and call
  this to verify the fetch shape

## Decision shape

| Signal | Operation | Outcome |
|---|---|---|
| Empty feed list in config | Halt with `error.no_feeds` and surface the config path | Operator action — no auto-recovery |
| All sources fail to fetch | Retry once with exponential backoff (2s/8s); halt with `warn.network` if both fail | Likely a real outage; do not pretend it's fine |
| Some sources succeed, some 4xx/5xx | Continue with successful sources; log dead feeds to `data/feed-health.json` | Partial degradation; episode still ships |
| All sources OK, zero new items since last run | Halt with `warn.no_new_items` | Agent decides: skip the episode, broaden the window, or republish |
| All sources OK, items present | Write `output/episodes/<date>/raw-items.json`, proceed to `write-script` | Happy path |

## What stays upstream

This skill is **commercial-generic**. AI Northwest's curated feed list,
the Wire Desk story-scoring rubric, multi-source coverage-worthiness
logic, and the AINW editorial weighting all stay in the proprietary
[`agent-radio`](https://github.com/ainorthwest/agent-radio) repo.
Operators bring their own feeds and let their agent's prompt express
their own editorial taste. That's the bifurcation rule.

If you need a story-scoring layer in your own deployment, write it as
your own skill that consumes `raw-items.json` and emits a
`scored-items.json`. Don't shoehorn taste into this skill — it would
fork the OSS contract and make every other operator's deployment
diverge.

## What this skill is NOT for

- Writing the script (that's `write-script` — not yet shipped in v0.1.0;
  the curator covers the basic case)
- Story scoring or editorial coverage decisions (operator territory)
- Storing long-term story state across episodes (Excellence #2 —
  editorial tracking — lands in v0.1.1)
- Web scraping arbitrary sites (use existing structured feeds; this
  skill does not include a generic crawler)

## Decision: ship / review / reject

There is no "ship" verdict on a gather pass — verdicts live downstream.
The agent inspects `raw-items.json` and either proceeds to scriptwriting
or halts. Common halt reasons:

- **Config gap** (`error.no_feeds`) — operator must configure sources
- **Network outage** (`warn.network`) — try again later; not an episode
  the agent can produce alone
- **No new content** (`warn.no_new_items`) — agent decides whether to
  widen the window, skip the day, or republish

## Scripts

- `scripts/gather.py` — thin wrapper around the curator (until a
  dedicated `radio gather` command lands in v0.1.1, gather is folded
  into `radio run --stop-after curator`-style invocations)

## State contract

- **Reads:** `config/radio.yaml` (or `--config <path>`), env-var secrets
  via `src.secrets`, network for feed URLs
- **Writes:** `output/episodes/<date>/raw-items.json` (or, in library
  mode, `library/programs/<slug>/episodes/<date>/raw-items.json`),
  optionally `data/feed-health.json` for dead-feed tracking
- **Side effects:** none beyond filesystem writes; no network mutations
