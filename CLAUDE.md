# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`agent-radio-oss` is the **open distribution** of [Agent Radio](https://github.com/ainorthwest/agent-radio). It is a hardware-agnostic, commercially-deployable subset of the production stack — Apache 2.0 throughout. The proprietary `agent-radio` repo holds MLX engines, Steward/Bard agents, and CC-BY-NC music weights that cannot ship in OSS.

**Status:** active sprint toward `v0.1.0-mvp`. See `oss-mvp-sprint.md` for the day-by-day plan, the locked engine choices, the explicit "what we are NOT doing" list, and the autonomous-station thesis. Re-read that file before adding scope.

**The bifurcation rule:** the test is *commercial deployability* and *AINW-specificity*, not "anything that smells like an agent."

| Goes in OSS | Stays upstream in `agent-radio` |
|---|---|
| Generic news-gathering primitives (RSS, web fetch, Discourse API) | AINW's specific feed list, AINW editorial voice |
| Generic editorial scaffolding (script structure, timing, voice assignment) | Bard's full prompt corpus, Steward lore |
| **A** working autonomous-station agent profile (Hermes-driven reference implementation) | Lightcone-specific Steward IP, AINW editorial taste, proprietary newsroom workflows |
| Agent-harness abstraction (Hermes today, Gaia / Claude Code / others next) | Lightcone's deployment of those harnesses |
| Apache 2.0 / commercial-permissive code and weights | MLX, Chatterbox, Orpheus, Dia, CSM, Qwen3, MusicGen, voice cloning, reference-audio voice profiles |

**The OSS demo is the autonomous loop, not the CLI.** A station without an agent is just a CLI run by hand. The v0.1.0 demo target is: a stranger on AMD hardware runs one setup script and gets a real autonomous radio station — gather → edit → script → render → quality-check → broadcast — with optional human-in-the-loop checkpoints, auto by default.

### Two front doors, one engine: CLI + Skills

Both surfaces are **first-class** and call the same `src/*` primitives. Neither is a fallback for the other.

| Surface | Audience | When |
|---|---|---|
| **`radio` CLI (Typer)** | Operator, debugging Claude, scripted pipelines | Deep work, surgical fixes, single-step inspection, dev loops |
| **Skills bundle** (`skills/`) | Station-runner agent (Hermes / Claude Code / Gaia) | Autonomous station operation, full episode loops, decision-shaped work |

The skills bundle is **the agent's operating manual** — prose-shaped SKILL.md contracts plus thin Python scripts that wrap `src/*`. Hermes loads `skills/` natively. Claude Code does too. Other harnesses get a thin loader.

**Decomposition matters.** Skills must cover both major steps (`render-episode`, `check-quality`) and minor steps (`render-segment`, `score-segment`, `decide-ship-review-reject`) so the agent can recover from partial failures without re-running the whole episode. See `oss-mvp-sprint.md` "Station Agent" section for the catalog.

### Reference material for the station agent

The station-runner agent is **written fresh** in this repo, not ported from `agent-radio`. The production newsroom modules (`bard.py`, `newsroom.py`, `assignment_editor.py`, `wire_desk.py` — ~3,175 LoC) are AINW-flavored and stay upstream. They are *context*, not source-of-truth.

For Hermes harness shape — SOUL.md format, config templates, skills layout, llama-server routing, ntfy patterns — pull from:

- `~/WORK/LIGHTCONE_STUDIOS_LLC/lightcone-crew/CLAUDE.md` — crew architecture and conventions
- `~/WORK/LIGHTCONE_STUDIOS_LLC/lightcone-crew/agents/perry/SOUL.md` — reference SOUL.md voice and structure
- `~/WORK/LIGHTCONE_STUDIOS_LLC/lightcone-crew/agents/perry/config.hinoki.yaml.template` — config template shape
- `~/WORK/LIGHTCONE_STUDIOS_LLC/lightcone-crew/docs/model-routing.md` — local inference (qwen3.5:9b on llama-server, port 8088)
- `~/WORK/LIGHTCONE_STUDIOS_LLC/lightcone-crew/shared/skills/` — shared-skill conventions (ntfy, github, etc.)

The station-runner is not a member of the Lightcone Crew (that's Lightcone IP). It's a **reference Hermes profile shipped in OSS** that demonstrates one agent harness running an autonomous station. Operators can swap in Gaia / Claude Code / their own harness.

## Commands

The project uses `uv` for Python dependency management and `radio` (Typer) as the CLI.

```bash
# Sync deps (pick the extras you need)
uv sync --extra dev                              # tests + lint + types only
uv sync --extra tts --extra quality --extra dev  # full local dev
uv sync --extra tts --extra quality --extra distribute --extra dev  # what CI runs

# Lint, format, type-check (must pass before PR — CI runs all three)
uv run ruff check .
uv run ruff format --check .       # use `ruff format .` to fix
uv run mypy src/

# Tests
uv run pytest -v                                  # full suite
uv run pytest tests/test_renderer.py -v           # one file
uv run pytest tests/test_renderer.py::test_name   # one test
uv run pytest -k "kokoro" -v                      # by keyword

# CLI surface (entry point: `radio` -> src.cli:app)
uv run radio --help
uv run radio config show
uv run radio render audition --voice voices/kokoro-michael.yaml --text "Hello"
uv run radio run pipeline --program haystack-news --no-distribute
uv run radio library list
uv run radio stream status

# Module-level entry points (used by tests and ad-hoc runs)
uv run python -m src.pipeline --program haystack-news --dry-run
uv run python -m src.renderer output/episodes/{date}/script.json
uv run python -m src.quality output/episode.mp3
```

**Pre-commit hooks** (gitleaks + private-key detection + yaml/whitespace) run on every commit. Run `pre-commit install` once after cloning. Do not bypass with `--no-verify`.

## Architecture

### The pipeline

The core dataflow is a five-stage pipeline (`src/pipeline.py` orchestrates):

```
curate ── render ── mix ── quality ── distribute
   │        │       │        │           │
curator   renderer mixer  quality    distributor
   │        │       │        │           │
script   per-seg   episode quality   R2 + RSS +
 .json    WAVs     .mp3    .json     AzuraCast
```

- **`src/curator.py`** — Fetches forum activity (Discourse) and calls an OpenAI-compatible LLM (default: OpenRouter) to produce a structured multi-voice script JSON.
- **`src/renderer.py`** — Reads `script.json`, dispatches each segment to its engine, applies DSP and loudness normalization, writes per-segment WAVs + `manifest.json`.
- **`src/mixer.py`** — Assembles segments into a single episode with music beds, stings, ducking. Uses `src/dsp.py`.
- **`src/quality.py`** — Three-pillar evaluation: librosa spectral (Pillar 1), torchmetrics DNSMOS/SRMR/PESQ/STOI (Pillar 2), WER intelligibility (Pillar 3 — **stubbed**, returns `-1.0` until whisper.cpp lands on Day 3).
- **`src/distributor.py`** — Uploads MP3 to Cloudflare R2 (S3-compatible via boto3), posts to Discourse. R2 is feature-flagged — empty creds means local-only.
- **`src/podcast.py`** — RSS feed generation.
- **`src/stream.py`** — AzuraCast HTTP API client for live streaming. Apache 2.0; stays in OSS.

### The library

Two output modes coexist:

- **Legacy mode** (default): artifacts in `output/episodes/{date}/`. Used when `--program` is not set.
- **Library mode** (`--program <slug>`): artifacts in `library/programs/{slug}/episodes/{date}/`, indexed by `src/library.py` (SQLite catalog at `library/radio.db`). `src/paths.py` resolves canonical paths.

`library/programs/<slug>/program.yaml` is the show bible — cast, music assets, timing, distribution targets. Today the only show is `haystack-news`.

### The engine surface

`src/engines/__init__.py` defines an `Engine` Protocol (documentation only — not enforced at runtime) and a `SUPPORTED_ENGINES` list. **OSS supports only `kokoro`.** The renderer keeps non-OSS dispatch branches as `_engine_unavailable` stubs that raise on call — this preserves the test suite's coverage of dispatch logic without shipping the proprietary engines.

Adding an engine: write `src/engines/<name>.py` with a `get_<name>()` lazy loader, add `"<name>"` to `SUPPORTED_ENGINES`, wire dispatch in `src/renderer.py`. Match the Kokoro shape: `render(text, voice_profile_dict, register) -> mono float32 numpy array at self.sample_rate`.

### Hardware backend selection

This is the educational point of the OSS repo — three different cross-hardware abstractions in one codebase:

- **Kokoro (ONNX Runtime):** provider strings via env var. `KOKORO_PROVIDER=ROCMExecutionProvider | CUDAExecutionProvider | CoreMLExecutionProvider | CPUExecutionProvider`. Defaults to CPU; invalid values fall back to CPU with a stderr warning. See `src/engines/kokoro.py`.
- **whisper.cpp** (Day 3, not yet wired): compile-flag abstraction — `make GGML_HIPBLAS=1 | GGML_VULKAN=1 | GGML_METAL=1`.
- **Stable Audio Open** (Day 4, not yet wired): PyTorch `device = "cuda" | "mps" | "cpu"` (HIP aliases as cuda on ROCm).

Hardware quirks get documented in `docs/hardware/{amd-rocm,apple-silicon,cpu}.md` as they're discovered. The docs **are** the deliverable for the cross-hardware bring-up days.

### Voice profiles

`voices/kokoro-*.yaml` are pure named-preset profiles — no reference audio, no PII, no proprietary samples. Each profile must declare `engine: kokoro`. The `cast` block in `program.yaml` points at these files.

### Config and secrets

- **Non-secret config:** `config/radio.yaml` (gitignored — each operator writes their own). Loaded via `src/config.py` -> `load_config()`. The `RadioConfig` dataclass is the single source of truth for runtime config.
- **Secrets:** environment variables, optionally via `.env`. Read through `src/secrets.py:get_secret()`. Never log, echo, or commit.

### CLI structure

`src/cli/__init__.py` is the Typer root. Each command group is a separate module that exports `app`:

```
radio config     -> src/cli/config_cmd.py
radio distribute -> src/cli/distribute_cmd.py
radio library    -> src/cli/library_cmd.py
radio render     -> src/cli/render_cmd.py
radio run        -> src/cli/run_cmd.py
radio soundbooth -> src/cli/soundbooth_cmd.py
radio stream     -> src/cli/stream_cmd.py
```

Global state (config path, `--program`, `--dry-run`, `--json`, `--no-music`) is on `typer.Context.obj` as a `State` dataclass that lazy-loads `RadioConfig` on first access. New commands should pull from `ctx.obj` rather than re-parsing options.

The OSS CLI deliberately drops `edit`, `eval`, `music`, `viz`, `wire`, `write` from the production CLI — those depend on Steward / Bard / MLX / matplotlib paths that don't exist here.

### mypy posture

`pyproject.toml` runs `--strict` globally, then loosens specific modules that wrap numpy/scipy/torch (`src.dsp`, `src.mixer`, `src.library`, `src.distributor`, `src.stream`, `src.quality`, `src.renderer`, `src.pipeline`, `src.engines.kokoro`) and the Typer-decorated CLI. New non-numerical code should hold the strict line. If you add a new wrapper around a numerics library, prefer fixing types over adding the module to the override list.

## Working in this repo

- **Read `oss-mvp-sprint.md` first** for any non-trivial change. The sprint plan is the source of truth for scope and "not now" decisions. The "what we are NOT doing" list AND the autonomous-station thesis are both load-bearing.
- **Stay inside the bifurcation rule above** — the test is commercial deployability and AINW-specificity, not "anything agent-shaped."
- **Station-runner agent work is written fresh.** Don't port `bard.py` / `newsroom.py` / `assignment_editor.py` / `wire_desk.py` from `agent-radio` — they're AINW-flavored. Use `lightcone-crew` for Hermes profile shape; design the OSS station-runner as a generic reference implementation.
- **Hermes is the first harness, not the only harness.** Roadmap: Hermes → Gaia → Claude Code → others. Build the integration so the harness layer is swappable, not so Hermes is hardcoded everywhere.
- **Match existing test patterns.** Tests in `tests/` mirror `src/` one-to-one. `conftest.py` adds project root to `sys.path` so `from src.x import y` works. Use `pytest.mark.skipif` for backend-dependent tests (see `test_podcast.py` and the SRMR Linux-runner skip in recent commits for the pattern).
- **CI matrix:** Ubuntu + macOS-14, Python 3.11 + 3.12. ffmpeg is installed in CI. Don't add OS-specific behavior without a `skipif` for the other platforms.
- **Branch + PR for everything.** Never commit to `main`. Use `feature/`, `fix/`, `chore/` prefixes. The sprint cadence is one PR per day with the day's tag in the title (e.g., `feat/sprint-day-3-whisper-cpp-stt`).
- **Commit style:** Conventional Commits. Recent history shows `feat:`, `fix:`, `chore:` — match it.
