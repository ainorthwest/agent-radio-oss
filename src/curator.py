"""Curator: Discourse API -> LLM API -> episode script JSON.

Fetches recent forum activity via the Discourse REST API, then calls an LLM
(via OpenRouter or any OpenAI-compatible endpoint) to synthesize a structured
multi-voice episode script. Saves the script to output/episodes/{date}/script.json.

Standalone:
    uv run python -m src.curator
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx
import openai
import yaml

from src.config import RadioConfig, load_config

# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------


class _StripHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._pieces: list[str] = []

    def handle_data(self, data: str) -> None:
        self._pieces.append(data)

    def get_text(self) -> str:
        return " ".join(self._pieces).strip()


def _strip_html(html_text: str) -> str:
    parser = _StripHTML()
    parser.feed(html_text)
    return re.sub(r"\s+", " ", parser.get_text()).strip()


# ---------------------------------------------------------------------------
# Discourse API helpers
# ---------------------------------------------------------------------------


def _discourse_headers(config: RadioConfig) -> dict[str, str]:
    return {
        "Api-Key": config.discourse.api_key,
        "Api-Username": config.discourse.api_username,
    }


def fetch_active_topics(config: RadioConfig) -> list[dict[str, Any]]:
    """Return topics with activity in the lookback window, ordered by recency."""
    cutoff = datetime.now(UTC) - timedelta(hours=config.discourse.lookback_hours)
    headers = _discourse_headers(config)

    with httpx.Client(base_url=config.discourse.base_url, headers=headers, timeout=30) as client:
        resp = client.get("/latest.json", params={"no_definitions": "true"})
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

    active = []
    for topic in data.get("topic_list", {}).get("topics", []):
        last_posted = str(topic.get("last_posted_at", ""))
        if not last_posted:
            continue
        # Normalize "Z" suffix for Python <3.11 compatibility
        last_dt = datetime.fromisoformat(last_posted.replace("Z", "+00:00"))
        if last_dt >= cutoff and int(topic.get("posts_count", 0)) > 1:
            active.append(topic)

    return active


def fetch_topic_posts(
    config: RadioConfig, topic_id: int, max_posts: int = 5
) -> list[dict[str, Any]]:
    """Fetch posts for a topic; return first post + most-liked posts, capped at max_posts."""
    headers = _discourse_headers(config)

    with httpx.Client(base_url=config.discourse.base_url, headers=headers, timeout=30) as client:
        resp = client.get(f"/t/{topic_id}.json")
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()

    posts: list[dict[str, Any]] = data.get("post_stream", {}).get("posts", [])

    if len(posts) <= max_posts:
        return posts

    first = posts[:1]
    rest_sorted = sorted(posts[1:], key=lambda p: int(p.get("like_count", 0)), reverse=True)
    return first + rest_sorted[: max_posts - 1]


# ---------------------------------------------------------------------------
# Forum summary builder
# ---------------------------------------------------------------------------


def build_forum_summary(
    config: RadioConfig, topics: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Build a text summary of forum activity and return metadata for show notes."""
    lines: list[str] = []
    threads_meta: list[dict[str, Any]] = []

    for topic in topics[:8]:  # cap to avoid context overflow
        topic_id = int(topic["id"])
        title = str(topic.get("title", ""))
        slug = str(topic.get("slug", ""))
        post_count = int(topic.get("posts_count", 1))
        url = f"{config.discourse.base_url}/t/{slug}/{topic_id}"

        threads_meta.append({"title": title, "url": url, "id": topic_id})

        lines.append(f"\n### {title}")
        lines.append(f"URL: {url}")
        lines.append(f"Replies: {post_count - 1}")

        try:
            posts = fetch_topic_posts(config, topic_id)
            for post in posts:
                author = str(post.get("username", "unknown"))
                cooked = str(post.get("cooked", ""))
                text = _strip_html(cooked)[:400]  # cap per post to keep prompt size sane
                lines.append(f"\n**{author}:** {text}")
        except httpx.HTTPStatusError as exc:
            lines.append(f"\n[Could not fetch posts: {exc.response.status_code}]")

    return "\n".join(lines), threads_meta


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _load_file(path: str) -> str:
    return Path(path).read_text()


def _build_episode_prompt(
    template: str,
    date: str,
    start_time: str,
    end_time: str,
    thread_count: int,
    forum_summary: str,
) -> str:
    return (
        template.replace("{{date}}", date)
        .replace("{{start_time}}", start_time)
        .replace("{{end_time}}", end_time)
        .replace("{{thread_count}}", str(thread_count))
        .replace("{{forum_summary}}", forum_summary)
    )


# ---------------------------------------------------------------------------
# Host identity loading
# ---------------------------------------------------------------------------


def _load_host_identities() -> dict[str, dict[str, Any]]:
    """Load host identity YAMLs referenced in cast.yaml.

    Returns a dict mapping slot names (host_a, host_b, ...) to their
    parsed identity data. Slots without an identity field are skipped.
    """
    cast_path = Path("cast.yaml")
    if not cast_path.exists():
        return {}

    cast = yaml.safe_load(cast_path.read_text())
    if not isinstance(cast, dict):
        return {}

    identities: dict[str, dict[str, Any]] = {}
    for slot, slot_config in cast.get("slots", {}).items():
        identity_path = slot_config.get("identity")
        if identity_path and Path(identity_path).exists():
            identity = yaml.safe_load(Path(identity_path).read_text())
            if isinstance(identity, dict):
                identities[slot] = identity

    return identities


def _build_host_profiles_text(identities: dict[str, dict[str, Any]]) -> str:
    """Build a text block describing host personalities for the LLM prompt."""
    if not identities:
        return ""

    lines: list[str] = []
    for slot, identity in sorted(identities.items()):
        name = identity.get("name", slot)
        role = identity.get("role", "host")
        tagline = identity.get("tagline", "")
        personality = identity.get("personality", {})

        lines.append(f"### {slot} — {name} ({role})")
        if tagline:
            lines.append(f"*{tagline}*")
        lines.append("")

        desc = personality.get("description", "").strip()
        if desc:
            lines.append(desc)
            lines.append("")

        vocab = personality.get("vocabulary_tendencies", [])
        if vocab:
            lines.append("**Vocabulary tendencies:**")
            for v in vocab:
                lines.append(f"- {v}")
            lines.append("")

        style = personality.get("sentence_style", "")
        if style:
            lines.append(f"**Sentence style:** {style}")
            lines.append("")

        tics = personality.get("verbal_tics", [])
        if tics:
            lines.append(f"**Verbal tics:** {', '.join(repr(t) for t in tics)}")
            lines.append("")

        avoids = personality.get("avoids", [])
        if avoids:
            lines.append(f"**Avoids:** {', '.join(repr(a) for a in avoids)}")
            lines.append("")

        topic_aff = identity.get("topic_affinity", {})
        leans = topic_aff.get("leans_into", [])
        defers = topic_aff.get("defers_on", [])
        if leans:
            lines.append(f"**Leans into:** {', '.join(leans)}")
        if defers:
            lines.append(f"**Defers on:** {', '.join(defers)}")
        if leans or defers:
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def _extract_script_json(raw_response: str) -> dict[str, Any]:
    """Extract a JSON object from an LLM response.

    Handles markdown code fences (with or without ``json`` tag) and raw JSON.
    Tries each fenced block in order, falling back to the raw response.
    """
    script: dict[str, Any] | None = None
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_response)
    for block in fenced_blocks:
        try:
            script = json.loads(block)
            break
        except json.JSONDecodeError:
            continue

    if script is None:
        try:
            script = json.loads(raw_response.strip())
        except json.JSONDecodeError:
            preview = raw_response[:500] if raw_response else "(empty response)"
            raise ValueError(
                f"Failed to parse episode script JSON from LLM response.\n"
                f"Response preview:\n{preview}"
            )

    return script


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def curate(
    config: RadioConfig,
    output_dir: Path = Path("output"),
    episode_dir: Path | None = None,
) -> Path:
    """Fetch forum activity, generate episode script via LLM, save JSON. Returns script path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    date_str = now.strftime("%Y-%m-%d")
    start_time = (now - timedelta(hours=config.discourse.lookback_hours)).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    end_time = now.strftime("%Y-%m-%d %H:%M UTC")

    # --- Stage 1: fetch ---
    print(f"Fetching Discourse activity for last {config.discourse.lookback_hours}h...")
    topics = fetch_active_topics(config)
    print(f"  {len(topics)} active topic(s) found")

    if not topics:
        print("  No active topics — generating a 'quiet day' placeholder script")

    forum_summary, _threads_meta = build_forum_summary(config, topics)

    # --- Stage 2: call LLM ---
    system_prompt = _load_file("prompts/curator-system.md")

    # Inject host personalities if available
    identities = _load_host_identities()
    if identities:
        host_profiles_text = _build_host_profiles_text(identities)
        system_prompt = system_prompt.replace("{{host_profiles}}", host_profiles_text)
        host_names = ", ".join(
            f"{slot}={ident.get('name', slot)}" for slot, ident in sorted(identities.items())
        )
        print(f"  Host identities loaded: {host_names}")
    else:
        # Remove the placeholder if no identities are available
        system_prompt = system_prompt.replace("{{host_profiles}}", "")

    episode_template = _load_file("prompts/curator-episode.md")
    user_prompt = _build_episode_prompt(
        episode_template,
        date=date_str,
        start_time=start_time,
        end_time=end_time,
        thread_count=len(topics),
        forum_summary=forum_summary if topics else "(No forum activity in this window.)",
    )

    print(f"Calling LLM ({config.curator.model}) to generate episode script...")
    api_key = config.curator.api_key  # resolved from env by load_config()
    client = openai.OpenAI(
        base_url=config.curator.base_url or "https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    response = client.chat.completions.create(
        model=config.curator.model,
        max_tokens=config.curator.max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_response = response.choices[0].message.content or ""
    script = _extract_script_json(raw_response)

    # --- Stage 3: save ---
    if episode_dir is None:
        episode_dir = output_dir / "episodes" / date_str
    episode_dir.mkdir(parents=True, exist_ok=True)
    output_path = episode_dir / "script.json"
    with output_path.open("w") as f:
        json.dump(script, f, indent=2)

    segment_count = len(script.get("segments", []))
    print(f"  Script saved: {output_path} ({segment_count} segments)")
    return output_path


if __name__ == "__main__":
    cfg = load_config()
    curate(cfg)
