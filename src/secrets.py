"""Secret resolution for Agent Radio.

Secrets come from environment variables or a .env file — never from YAML.
The .env file is parsed with stdlib only (no python-dotenv dependency).

Priority: environment variable > .env file value.

Usage:
    from src.secrets import get_secret, require_secret

    api_key = get_secret("AGENT_RADIO_DISCOURSE_API_KEY")  # "" if unset
    api_key = require_secret("AGENT_RADIO_DISCOURSE_API_KEY")  # raises if unset

Env var naming convention:
    AGENT_RADIO_DISCOURSE_API_KEY
    AGENT_RADIO_R2_BUCKET
    AGENT_RADIO_R2_ENDPOINT
    AGENT_RADIO_R2_ACCESS_KEY_ID
    AGENT_RADIO_R2_SECRET_ACCESS_KEY
    AGENT_RADIO_R2_PUBLIC_URL_BASE
    AGENT_RADIO_AZURACAST_API_KEY
    OPENROUTER_API_KEY  (existing name kept for compatibility)
"""

from __future__ import annotations

import os
from pathlib import Path

# Lazy-loaded .env values — populated on first get_secret() call
_dotenv_loaded: bool = False
_dotenv_values: dict[str, str] = {}


def _load_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    """Parse a .env file into a dict. Stdlib only.

    Handles:
        KEY=value
        KEY="quoted value"
        KEY='single quoted'
        # comments
        blank lines
        export KEY=value (optional export prefix)
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip optional 'export ' prefix
            if line.startswith("export "):
                line = line[7:]
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Remove surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            values[key] = value

    return values


def _ensure_dotenv_loaded() -> None:
    """Load .env file on first access."""
    global _dotenv_loaded, _dotenv_values
    if not _dotenv_loaded:
        _dotenv_values = _load_dotenv()
        _dotenv_loaded = True


def get_secret(env_var: str) -> str:
    """Get a secret from environment variables or .env file.

    Returns empty string if the secret is not set.
    Caller decides whether an empty value is acceptable.
    """
    # Environment variable takes priority
    env_value = os.environ.get(env_var)
    if env_value is not None and env_value != "":
        return env_value

    # Fall back to .env file
    _ensure_dotenv_loaded()
    return _dotenv_values.get(env_var, "")


def require_secret(env_var: str) -> str:
    """Get a secret, raising ValueError if not set.

    Use this for credentials that are required for the operation to proceed.
    """
    value = get_secret(env_var)
    if not value:
        raise ValueError(
            f"Required secret {env_var} is not set. "
            f"Set it as an environment variable or in your .env file. "
            f"See .env.example for all required secrets."
        )
    return value
