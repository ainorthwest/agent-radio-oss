"""Tests for configuration loading and defaults."""

import os
import tempfile
from unittest.mock import patch

import yaml

from src.config import CuratorConfig, StreamConfig, load_config


def test_curator_config_defaults():
    """CuratorConfig should default to OpenRouter settings."""
    c = CuratorConfig()
    assert c.model == "anthropic/claude-sonnet-4"
    assert c.base_url == "https://openrouter.ai/api/v1"
    assert c.api_key == ""
    assert c.max_tokens == 4096
    assert c.target_duration_minutes == 5


def test_load_config_parses_curator_fields():
    """load_config should parse non-secret fields from YAML and secrets from env."""
    config_data = {
        "discourse": {
            "base_url": "https://example.com",
            "api_username": "bot",
        },
        "curator": {
            "model": "openai/gpt-4o",
            "base_url": "https://custom-endpoint.example.com/v1",
            "max_tokens": 2048,
            "target_duration_minutes": 3,
        },
        "renderer": {"engine": "kokoro"},
        "distributor": {},
        "voices": {"host_a": "voices/a.yaml"},
    }
    env = {
        "AGENT_RADIO_DISCOURSE_API_KEY": "test-discourse-key",
        "OPENROUTER_API_KEY": "sk-test-123",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        f.flush()
        with patch.dict(os.environ, env):
            cfg = load_config(f.name)

    assert cfg.curator.model == "openai/gpt-4o"
    assert cfg.curator.base_url == "https://custom-endpoint.example.com/v1"
    assert cfg.curator.api_key == "sk-test-123"
    assert cfg.curator.max_tokens == 2048
    assert cfg.discourse.api_key == "test-discourse-key"


def test_load_config_uses_defaults_for_missing_fields():
    """Missing fields should use defaults. Missing secrets return empty string."""
    config_data = {
        "discourse": {
            "base_url": "https://example.com",
            "api_username": "bot",
        },
        "curator": {},
        "renderer": {},
        "distributor": {},
        "voices": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.curator.base_url == "https://openrouter.ai/api/v1"
    # Secrets not set in env → empty string (not raised)
    assert cfg.curator.api_key == ""
    assert cfg.distributor.r2_bucket == ""


def test_load_config_file_not_found():
    """load_config should raise FileNotFoundError for missing config."""
    try:
        load_config("/nonexistent/path/radio.yaml")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "Config not found" in str(e)


def test_stream_config_defaults():
    """StreamConfig should default to disabled with empty fields."""
    s = StreamConfig()
    assert s.enabled is False
    assert s.base_url == ""
    assert s.station_id == 1
    assert s.api_key == ""
    assert s.playlist_name == ""


def test_load_config_parses_stream_section():
    """load_config should parse stream section from YAML + API key from env."""
    config_data = {
        "discourse": {"base_url": "https://example.com", "api_username": "bot"},
        "curator": {},
        "renderer": {},
        "distributor": {},
        "stream": {
            "enabled": True,
            "base_url": "https://radio.example.com",
            "station_id": 2,
            "playlist_name": "Test Show",
        },
        "voices": {},
    }
    env = {
        "AGENT_RADIO_DISCOURSE_API_KEY": "test-key",
        "AGENT_RADIO_AZURACAST_API_KEY": "az-test-key",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        f.flush()
        with patch.dict(os.environ, env):
            cfg = load_config(f.name)

    assert cfg.stream.enabled is True
    assert cfg.stream.base_url == "https://radio.example.com"
    assert cfg.stream.station_id == 2
    assert cfg.stream.api_key == "az-test-key"
    assert cfg.stream.playlist_name == "Test Show"


def test_load_config_stream_absent():
    """Missing stream section should use defaults (disabled)."""
    config_data = {
        "discourse": {"base_url": "https://example.com", "api_username": "bot"},
        "curator": {},
        "renderer": {},
        "distributor": {},
        "voices": {},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        f.flush()
        cfg = load_config(f.name)

    assert cfg.stream.enabled is False
    assert cfg.stream.base_url == ""
    assert cfg.stream.api_key == ""
