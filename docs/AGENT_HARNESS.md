# Agent Harness Integration

How to wire `agent-radio-oss` into an agent harness — Claude Code,
Hermes, Gaia, or your own. The OSS station is **harness-agnostic by
design**: every skill is a `SKILL.md` prose contract plus a thin
Python wrapper around the `radio` CLI. Whatever harness loads
`SKILL.md` files and can `subprocess` a wrapper script can run the
station.

## What "agent harness" means here

An agent harness is the runtime that:

1. Receives the agent's tool-use intent
2. Resolves it against a registered skill
3. Invokes the skill's executable surface (Python wrapper, shell
   script, MCP server, etc.)
4. Hands the output back to the agent

The OSS skills are deliberately small and prose-shaped so the harness
choice is reversible. Hermes is AINW's reference harness because the
production AI Northwest stack uses it; nothing in OSS depends on
Hermes. Gaia, Claude Code, and any harness that can subprocess Python
can run the station equally well.

The reference station-runner agent (the autonomous loop expressed as a
harness profile) is **written fresh** in OSS. It is *not* a port of
the upstream `agent-radio` newsroom modules
(`bard.py` / `newsroom.py` / `assignment_editor.py` / `wire_desk.py`),
which are AINW-specific and stay in the proprietary repo. See
[`oss-mvp-sprint.md`](../oss-mvp-sprint.md) §"Station Agent" for the
design constraints.

## Loading the skills bundle

Every harness has its own registration shape. The skills themselves
are identical across harnesses — what differs is how the harness
discovers them.

| Harness | Registration | Example |
|---|---|---|
| **Claude Code** | Drop `skills/*` paths into the project's `.claude/settings.json` under `"skills"`, or symlink `skills/` into `.claude/skills/` | `"skills": [{"path": "skills/run-station/SKILL.md"}]` |
| **Hermes** | Reference the skill paths from your agent's `agent.toml` under `[skills]` or include them in the agent's `SOUL.md` | `paths = ["./skills/run-station", "./skills/check-quality", ...]` |
| **Gaia** | `gaia config add-skills <path>` (or the equivalent in the harness's TUI) | n/a — harness-specific |
| **Custom / MCP** | Treat each `SKILL.md` as the contract and `scripts/*.py` as the entrypoint. Your harness adapter parses the front-matter and shells out to the wrapper. | See `skills/run-station/scripts/loop.py` for the shape |

In all cases, the wrapper scripts shell to `uv run radio …` (or `uv
run python -m src.quality …` for the standalone quality module) so
the harness does not need to import the OSS code directly. The
subprocess boundary is the contract.

## State contract

Where state lives, what's read-only, and which directories the agent
should treat as its working memory:

| Path | Role | Mutability |
|---|---|---|
| `output/episodes/<date>/` | Legacy-mode episode state. Script.json, manifest, audio, quality.json all live here. | Agent may read and write within an episode dir. |
| `library/programs/<slug>/episodes/<date>/` | Library-mode episode state (when the agent passes `--program`). | Same. |
| `data/segment-cache/` | Content-addressed WAV cache (sha256(text + speaker + register + voice + engine)). | Agent never writes directly. The renderer manages atomic writes. |
| `library/radio.db` | SQLite catalog of programs and episodes. | Agent never writes directly — the catalog API in `src/library.py` mediates writes. |
| `config/radio.yaml` | Operator-controlled runtime config: feeds, voices, engines, distribution endpoints. | Read-only at runtime. Operators edit out-of-band. |
| `config/quality.yaml` | Quality-stack thresholds and weights. | Read-only at runtime. Operators tune for their station. |
| `voices/*.yaml` | Voice profile definitions (Kokoro presets in OSS). | Read-only at runtime. Operators add new ones. |
| `.env` / env vars | Secrets only — R2, Discourse, AzuraCast, OpenRouter. Read via `src/secrets.py`. | Never logged, never committed, never re-emitted in tool output. |
| `data/feed-health.json` | Per-feed dead/alive tracking written by `gather-news`. | Agent may read; only `gather-news` writes. |
| `data/smoke-runs/*.log` | Smoke-test logs from `scripts/oss-smoke.sh`. | Test artifacts; not part of the runtime loop. |

**Rule of thumb:** if the file is in `config/`, `voices/`, or `.env`,
treat it as read-only. If it's in `output/`, `library/<programs>/.../episodes/`,
or `data/`, the agent and the OSS code may both write — coordinate
through the named artifacts (script.json, manifest.json, quality.json).

## Recovery from partial failures

The autonomous-station loop is designed to resume cleanly after any
crash. Each phase produces a named artifact; the next phase consumes
it. The agent's recovery procedure:

| Last complete artifact | Next step | Why |
|---|---|---|
| `raw-items.json` only | `write-script` (curator in v0.1.0) | Items are gathered; script not yet written |
| `script.json` (with `script-quality.json`) | `render-episode` | Script passed the gate; render hasn't started |
| `manifest.json` (no `quality.json`) | `check-quality` | Render finished; quality not scored |
| `manifest.json` partially complete (some segments missing WAVs) | `render-episode` again — segment cache short-circuits completed segments | Atomic writes mean partial state is safe |
| `quality.json` with `verdict: ship` (no broadcast log) | `broadcast` | Distribution didn't run yet |
| `quality.json` with `verdict: review` | Halt; surface to operator | Auto-loop should not ship a `review` |
| `quality.json` with `verdict: reject` | `edit-script` then `render-episode` then `check-quality` again | The correction loop |
| `broadcast` manifest with one branch failed | Re-run that branch only | Each branch is idempotent on the remote side |

### The segment cache makes re-render cheap

Critical for resume: the segment cache (`data/segment-cache/`) is
content-addressed on
`sha256(text + speaker + register + voice_profile + engine)`. When
`render-episode` re-runs, segments whose script content is unchanged
hit the cache and skip Kokoro entirely. This means:

- A crash mid-render costs one segment of work, not a whole episode.
- An `edit-script` mutation invalidates exactly the affected segments;
  unrelated segments stay cached.
- Voice-profile A/B testing is fast: only changed segments re-render.

The cache is safe under concurrent / interrupted writes because
`segment_cache.write()` uses `.tmp` + atomic rename. Don't manually
write into `data/segment-cache/`.

## Bifurcation note

OSS skills are deliberately **commercial-generic**. The bifurcation
rule:

> Generic primitives (RSS, Discourse API, render pipeline, quality
> stack, broadcast plumbing) ship here under Apache 2.0. AINW-specific
> content (curated feed lists, Bard's prompt corpus, Steward lore,
> AINW editorial taste, MLX engines, voice cloning, CC-BY-NC weights)
> stays in the proprietary upstream
> [`agent-radio`](https://github.com/ainorthwest/agent-radio) repo.

What this means for harness operators:

- **Your prompt corpus is yours.** The OSS station-runner agent
  references no AINW prompt material. When you wire the skills into
  your harness, you write your own SOUL / persona / editorial direction.
- **Your voice cast is yours.** OSS ships Kokoro voice presets; if you
  want voice cloning or a different engine family, register your own
  engine in `src/engines/` and add your own voice YAMLs.
- **Your taste is yours.** The skills decompose into capabilities, not
  decisions. The agent harness is where decisions live.

For a working *example* of a Hermes profile that runs an OSS station,
see the public `lightcone-crew` reference profiles (Aaron's
crew-architecture conventions). That is one shape; not the only shape.
The `run-station` skill's `loop.py` is the OSS reference orchestrator
expressed as a Python script for harnesses without a richer
agent-loop primitive.

## Skill catalog (v0.1.0)

| Skill | Primary CLI surface | When the agent reaches for it |
|---|---|---|
| [`gather-news`](../skills/gather-news/SKILL.md) | `radio run --no-distribute --no-music` (until v0.1.1 adds `radio gather`) | Start of the autonomous loop |
| [`render-episode`](../skills/render-episode/SKILL.md) | `radio render episode` | After script writer produced a clean script.json |
| [`check-quality`](../skills/check-quality/SKILL.md) | `python -m src.quality` | After render finishes, before broadcast |
| [`edit-script`](../skills/edit-script/SKILL.md) | `radio edit script` / `radio edit anomalies` | When anomaly detector or check-quality flagged a fixable issue |
| [`publish-episode`](../skills/publish-episode/SKILL.md) | `radio publish episode` | After verdict is ship; before broadcast |
| [`broadcast`](../skills/broadcast/SKILL.md) | `radio distribute episode` + `radio distribute feed` + `radio stream update` | Final step; mutates remote state |
| [`run-station`](../skills/run-station/SKILL.md) | `python skills/run-station/scripts/loop.py` | The meta-skill — wraps the whole loop |

## See also

- [`CLAUDE.md`](../CLAUDE.md) — full architecture and bifurcation rule
- [`README.md`](../README.md) — operator quick start + excellences scorecard
- [`CHANGELOG.md`](../CHANGELOG.md) — what landed in v0.1.0 and what's deferred
- [`oss-mvp-sprint.md`](../oss-mvp-sprint.md) — sprint plan and "what we are NOT doing" list
