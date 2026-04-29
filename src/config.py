"""Configuration loading for agent-radio.

Non-secret config comes from config/radio.yaml.
Secrets come from environment variables or .env file (see src/secrets.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.secrets import get_secret


@dataclass
class DiscourseConfig:
    base_url: str
    api_key: str
    api_username: str
    lookback_hours: int = 24
    categories: list[str] = field(default_factory=list)


@dataclass
class CuratorConfig:
    model: str = "anthropic/claude-sonnet-4"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    max_tokens: int = 4096
    target_duration_minutes: int = 5


@dataclass
class RendererConfig:
    engine: str = "kokoro"
    sample_rate: int = 24000
    output_format: str = "mp3"


@dataclass
class DistributorConfig:
    r2_bucket: str = ""
    r2_endpoint: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    public_url_base: str = ""
    post_to_discourse: bool = True
    discourse_category: str = ""  # Integer category ID as string (e.g. "6")


@dataclass
class StreamConfig:
    enabled: bool = False
    base_url: str = ""  # e.g. https://radio.example.com
    station_id: int = 1
    api_key: str = ""  # from AGENT_RADIO_AZURACAST_API_KEY env var
    playlist_name: str = ""  # rolling playlist name — created if absent


@dataclass
class LibraryConfig:
    root: str = "library"
    db_name: str = "radio.db"


@dataclass
class RadioConfig:
    discourse: DiscourseConfig
    curator: CuratorConfig
    renderer: RendererConfig
    distributor: DistributorConfig
    stream: StreamConfig
    voices: dict[str, str]
    library: LibraryConfig = field(default_factory=LibraryConfig)


def load_config(config_path: str | Path = "config/radio.yaml") -> RadioConfig:
    """Load config from YAML (non-secret settings) + env vars (secrets).

    Secrets are resolved from environment variables or .env file.
    YAML holds only non-secret config: URLs, usernames, model names, etc.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}\n"
            "Copy config/radio.example.yaml to config/radio.yaml and fill in values."
        )
    with path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    d: dict[str, Any] = raw.get("discourse", {})
    c: dict[str, Any] = raw.get("curator", {})
    r: dict[str, Any] = raw.get("renderer", {})
    dist: dict[str, Any] = raw.get("distributor", {})
    s: dict[str, Any] = raw.get("stream", {})
    v: dict[str, Any] = raw.get("voices", {})
    lib: dict[str, Any] = raw.get("library", {})

    return RadioConfig(
        discourse=DiscourseConfig(
            base_url=str(d.get("base_url", "https://community.ainorthwest.org")),
            api_key=get_secret("AGENT_RADIO_DISCOURSE_API_KEY"),
            api_username=str(d.get("api_username", "steward")),
            lookback_hours=int(d.get("lookback_hours", 24)),
            categories=[str(cat) for cat in d.get("categories", [])],
        ),
        curator=CuratorConfig(
            model=str(c.get("model", "anthropic/claude-sonnet-4")),
            base_url=str(c.get("base_url", "https://openrouter.ai/api/v1")),
            api_key=get_secret("OPENROUTER_API_KEY"),
            max_tokens=int(c.get("max_tokens", 4096)),
            target_duration_minutes=int(c.get("target_duration_minutes", 5)),
        ),
        renderer=RendererConfig(
            engine=str(r.get("engine", "kokoro")),
            sample_rate=int(r.get("sample_rate", 24000)),
            output_format=str(r.get("output_format", "mp3")),
        ),
        distributor=DistributorConfig(
            r2_bucket=get_secret("AGENT_RADIO_R2_BUCKET"),
            r2_endpoint=get_secret("AGENT_RADIO_R2_ENDPOINT"),
            r2_access_key_id=get_secret("AGENT_RADIO_R2_ACCESS_KEY_ID"),
            r2_secret_access_key=get_secret("AGENT_RADIO_R2_SECRET_ACCESS_KEY"),
            public_url_base=get_secret("AGENT_RADIO_R2_PUBLIC_URL_BASE"),
            post_to_discourse=bool(dist.get("post_to_discourse", True)),
            discourse_category=str(dist.get("discourse_category", "")),
        ),
        stream=StreamConfig(
            enabled=bool(s.get("enabled", False)),
            base_url=str(s.get("base_url", "")),
            station_id=int(s.get("station_id", 1)),
            api_key=get_secret("AGENT_RADIO_AZURACAST_API_KEY"),
            playlist_name=str(s.get("playlist_name", "")),
        ),
        voices={str(k): str(val) for k, val in v.items()},
        library=LibraryConfig(
            root=str(lib.get("root", "library")),
            db_name=str(lib.get("db_name", "radio.db")),
        ),
    )
