"""Publisher — derivative content fan-out from script.json + manifest.json.

The autonomous-station thesis says agents and humans who can't (or
won't) listen to the audio still get the full content. This module
turns each rendered episode into a set of agent-readable artifacts:

Pure-function outputs (deterministic, regenerated every call):
    episode.md      — canonical markdown with YAML frontmatter (browsable)
    chapters.json   — Podcasting 2.0 cloud chapters spec
    episode.txt     — script flattened to plain text (agent payload)
    episode.jsonld  — schema.org PodcastEpisode JSON-LD

LLM-derived outputs (cached by hash(prompt + script), regenerated only
when the script changes):
    description.txt — short ≤200-word summary for RSS / show notes
    social/linkedin.txt, social/bluesky.txt — short-form public copy

This MVP ships only the deterministic outputs. The LLM derivatives are
scoped behind ``llm_enabled``; they land in a follow-up commit so the
deterministic surface stays bisectable.

Design rules:
1. Pure functions take dicts, return strings/dicts. No I/O, no globals.
2. Determinism is a contract — re-running publish() on the same input
   produces byte-identical files. Test ``test_idempotent`` enforces it.
3. The publisher reads from the episode directory and writes back to it.
   It never reaches into ``library/`` for cross-episode state.
4. The publisher never modifies script.json or manifest.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _total_duration(manifest: dict[str, Any]) -> float:
    """Sum of segment durations (seconds)."""
    return float(sum(s.get("duration_seconds", 0.0) for s in manifest.get("segments", [])))


def _character_name(manifest: dict[str, Any], speaker: str) -> str:
    """Resolve a slot key (``host_a``) to its character name (``Michael``).

    Falls back to the slot key when the cast block is absent.
    """
    cast = manifest.get("cast", {})
    slot = cast.get(speaker, {})
    return str(slot.get("character_name", speaker))


def _iso_duration(seconds: float) -> str:
    """Format seconds as ISO 8601 duration (PT…S). Schema.org expects this."""
    if seconds < 60:
        return f"PT{seconds:g}S"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        if sec:
            return f"PT{int(minutes)}M{sec:g}S"
        return f"PT{int(minutes)}M"
    hours, minutes = divmod(int(minutes), 60)
    parts = [f"PT{hours}H"]
    if minutes:
        parts.append(f"{minutes}M")
    if sec:
        parts.append(f"{sec:g}S")
    return "".join(parts)


# ── chapters.json (Podcasting 2.0 cloud chapters) ────────────────────────────


def build_chapters(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build a Podcasting 2.0 cloud chapters dict.

    Spec: https://podcasting2.org/docs/podcast-namespace/tags/chapters
    Format: {"version": "1.2.0", "chapters": [{"startTime", "title"}, ...]}

    Chapter titles use the character name + topic when available; start
    times are cumulative durations from the manifest.
    """
    chapters: list[dict[str, Any]] = []
    cursor = 0.0
    for seg in manifest.get("segments", []):
        speaker = str(seg.get("speaker", ""))
        topic = str(seg.get("topic", "")).strip()
        name = _character_name(manifest, speaker)
        title = f"{name}: {topic}" if topic else name
        chapters.append({"startTime": round(cursor, 3), "title": title})
        cursor += float(seg.get("duration_seconds", 0.0))
    return {"version": "1.2.0", "chapters": chapters}


# ── episode.txt (agent-readable payload) ─────────────────────────────────────


def build_episode_text(script: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Flatten the script to a plain-text transcript with speaker attribution.

    Format::

        <title>
        ============

        Michael: First segment.

        Bella: Second segment.

        Adam: Third segment.

    Useful when a downstream agent can't process audio — the file is
    self-explanatory and grep-friendly.
    """
    title = str(script.get("title", "Untitled"))
    out_lines = [title, "=" * len(title), ""]
    for seg in script.get("segments", []):
        speaker = str(seg.get("speaker", ""))
        name = _character_name(manifest, speaker)
        text = str(seg.get("text", "")).strip()
        out_lines.append(f"{name}: {text}")
        out_lines.append("")
    return "\n".join(out_lines).rstrip() + "\n"


# ── episode.jsonld (schema.org PodcastEpisode) ───────────────────────────────


def build_jsonld(script: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Build a schema.org PodcastEpisode JSON-LD blob.

    Embedding this in an episode page (or shipping it as ``episode.jsonld``)
    makes the show indexable by Google / Bing / aggregators.
    """
    duration_s = _total_duration(manifest)
    return {
        "@context": "https://schema.org",
        "@type": "PodcastEpisode",
        "name": str(script.get("title", "Untitled")),
        "datePublished": str(script.get("date", "")),
        "duration": _iso_duration(duration_s),
        "description": str(script.get("summary", "")).strip(),
    }


# ── episode.md (canonical markdown) ──────────────────────────────────────────


def build_episode_markdown(script: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Build a full markdown document with YAML frontmatter + body.

    Frontmatter keys:
        title, date, program, duration_seconds, hosts (list), summary
    Body:
        Full transcript (same content as episode.txt) so the markdown
        page is self-contained for browsing/blog publishing.
    """
    duration_s = _total_duration(manifest)
    hosts = sorted(
        {_character_name(manifest, str(s.get("speaker", ""))) for s in script.get("segments", [])}
    )
    frontmatter = {
        "title": str(script.get("title", "Untitled")),
        "date": str(script.get("date", "")),
        "program": str(script.get("program", "")),
        "duration_seconds": round(duration_s, 3),
        "hosts": hosts,
        "summary": str(script.get("summary", "")).strip(),
    }
    fm_yaml = yaml.safe_dump(frontmatter, sort_keys=True, allow_unicode=True).rstrip()
    body = build_episode_text(script, manifest)
    return f"---\n{fm_yaml}\n---\n\n{body}"


# ── llms.txt (per-show index for AI agents) ─────────────────────────────────


def build_llms_txt(program_dir: Path, show_name: str, description: str) -> str:
    """Build an llms.txt index for a single show.

    Spec: https://llmstxt.org/ — H1 + blockquote + H2 link sections.
    Used by AI agents (and humans) to find an episode list in
    machine-readable form. Lives at
    ``library/programs/<slug>/llms.txt``; regenerated on every
    ``radio publish``.

    Episode entries point at ``episodes/<date>/episode.md`` and are
    listed newest-first by date string. Episodes without an
    ``episode.md`` (i.e. not yet published) are skipped.
    """
    program_dir = Path(program_dir)
    lines = [f"# {show_name}", ""]
    if description:
        lines.append(f"> {description}")
        lines.append("")
    lines.append("## Episodes")
    lines.append("")

    eps_dir = program_dir / "episodes"
    if eps_dir.exists():
        # Sort by directory name (date string), newest first
        ep_dirs = sorted(
            (d for d in eps_dir.iterdir() if d.is_dir() and (d / "episode.md").exists()),
            key=lambda d: d.name,
            reverse=True,
        )
        for d in ep_dirs:
            md_path = d / "episode.md"
            title = _frontmatter_title(md_path) or d.name
            rel = md_path.relative_to(program_dir).as_posix()
            lines.append(f"- [{title}]({rel}) — {d.name}")

    return "\n".join(lines).rstrip() + "\n"


def _frontmatter_title(md_path: Path) -> str | None:
    """Extract the ``title`` field from an episode.md's YAML frontmatter."""
    try:
        text = md_path.read_text()
    except OSError:
        return None
    if not text.startswith("---\n"):
        return None
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return None
    try:
        meta = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict):
        return None
    title = meta.get("title")
    return str(title) if title is not None else None


# ── publish() — orchestrator ─────────────────────────────────────────────────


def publish(episode_dir: Path, llm_enabled: bool = False) -> dict[str, Any]:
    """Generate all deterministic derivative artifacts in ``episode_dir``.

    Reads ``script.json`` and ``manifest.json`` from ``episode_dir``;
    writes ``episode.md``, ``chapters.json``, ``episode.txt``, and
    ``episode.jsonld`` back to the same directory.

    Returns a dict ``{"written": [<path>, ...], "skipped": [...]}``.

    ``llm_enabled`` is reserved for the LLM-derived outputs (description,
    social copy) that land in a follow-up commit.
    """
    episode_dir = Path(episode_dir)
    script_path = episode_dir / "script.json"
    manifest_path = episode_dir / "manifest.json"
    if not script_path.exists():
        raise FileNotFoundError(f"script not found at {script_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found at {manifest_path}")

    script = json.loads(script_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    written: list[str] = []
    skipped: list[str] = []

    # Deterministic outputs
    md = build_episode_markdown(script, manifest)
    (episode_dir / "episode.md").write_text(md)
    written.append("episode.md")

    chapters = build_chapters(manifest)
    (episode_dir / "chapters.json").write_text(json.dumps(chapters, indent=2))
    written.append("chapters.json")

    txt = build_episode_text(script, manifest)
    (episode_dir / "episode.txt").write_text(txt)
    written.append("episode.txt")

    ld = build_jsonld(script, manifest)
    (episode_dir / "episode.jsonld").write_text(json.dumps(ld, indent=2))
    written.append("episode.jsonld")

    if llm_enabled:
        # LLM-derived outputs land in a follow-up commit; placeholder.
        skipped.append("description.txt (llm_enabled requires curator API key)")
        skipped.append("social/{linkedin,bluesky}.txt (same)")

    return {"written": written, "skipped": skipped}
