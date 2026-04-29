"""Tests for secret resolution."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.secrets import _load_dotenv, get_secret, require_secret


class TestLoadDotenv:
    """Tests for .env file parsing."""

    def test_simple_key_value(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\nBAZ=qux\n")
        result = _load_dotenv(env_file)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_double_quoted_value(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text('MY_KEY="hello world"\n')
        result = _load_dotenv(env_file)
        assert result["MY_KEY"] == "hello world"

    def test_single_quoted_value(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("MY_KEY='hello world'\n")
        result = _load_dotenv(env_file)
        assert result["MY_KEY"] == "hello world"

    def test_comments_and_blanks_skipped(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nFOO=bar\n# another\nBAZ=qux\n")
        result = _load_dotenv(env_file)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_export_prefix_stripped(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("export FOO=bar\nexport BAZ=qux\n")
        result = _load_dotenv(env_file)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = _load_dotenv(tmp_path / "nonexistent")
        assert result == {}

    def test_empty_value(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("EMPTY=\n")
        result = _load_dotenv(env_file)
        assert result["EMPTY"] == ""

    def test_value_with_equals_sign(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("URL=https://example.com?foo=bar&baz=qux\n")
        result = _load_dotenv(env_file)
        assert result["URL"] == "https://example.com?foo=bar&baz=qux"

    def test_line_without_equals_skipped(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("GOOD=value\nBAD_LINE\nALSO_GOOD=yes\n")
        result = _load_dotenv(env_file)
        assert result == {"GOOD": "value", "ALSO_GOOD": "yes"}


class TestGetSecret:
    """Tests for get_secret()."""

    def test_from_env_var(self) -> None:
        import src.secrets

        src.secrets._dotenv_loaded = False
        src.secrets._dotenv_values = {}

        with patch.dict(os.environ, {"TEST_SECRET": "from_env"}):
            assert get_secret("TEST_SECRET") == "from_env"

    def test_from_dotenv(self, tmp_path: Path) -> None:
        import src.secrets

        src.secrets._dotenv_loaded = False
        src.secrets._dotenv_values = {}

        env_file = tmp_path / ".env"
        env_file.write_text("TEST_DOTENV_SECRET=from_file\n")

        with patch.object(
            src.secrets, "_load_dotenv", return_value={"TEST_DOTENV_SECRET": "from_file"}
        ):
            src.secrets._dotenv_loaded = False
            assert get_secret("TEST_DOTENV_SECRET") == "from_file"

    def test_env_var_takes_priority(self) -> None:
        import src.secrets

        src.secrets._dotenv_loaded = True
        src.secrets._dotenv_values = {"PRIORITY_TEST": "from_file"}

        with patch.dict(os.environ, {"PRIORITY_TEST": "from_env"}):
            assert get_secret("PRIORITY_TEST") == "from_env"

    def test_missing_returns_empty(self) -> None:
        import src.secrets

        src.secrets._dotenv_loaded = True
        src.secrets._dotenv_values = {}

        with patch.dict(os.environ, {}, clear=False):
            result = get_secret("DEFINITELY_NOT_SET_12345")
            assert result == ""

    def test_empty_env_var_falls_through_to_dotenv(self) -> None:
        import src.secrets

        src.secrets._dotenv_loaded = True
        src.secrets._dotenv_values = {"EMPTY_TEST": "from_file"}

        with patch.dict(os.environ, {"EMPTY_TEST": ""}):
            assert get_secret("EMPTY_TEST") == "from_file"


class TestRequireSecret:
    """Tests for require_secret()."""

    def test_returns_value_when_set(self) -> None:
        import src.secrets

        src.secrets._dotenv_loaded = True
        src.secrets._dotenv_values = {}

        with patch.dict(os.environ, {"REQUIRED_TEST": "exists"}):
            assert require_secret("REQUIRED_TEST") == "exists"

    def test_raises_when_missing(self) -> None:
        import src.secrets

        src.secrets._dotenv_loaded = True
        src.secrets._dotenv_values = {}

        with pytest.raises(ValueError, match="Required secret"):
            require_secret("DEFINITELY_NOT_SET_99999")

    def test_error_message_includes_var_name(self) -> None:
        import src.secrets

        src.secrets._dotenv_loaded = True
        src.secrets._dotenv_values = {}

        with pytest.raises(ValueError, match="MY_MISSING_KEY"):
            require_secret("MY_MISSING_KEY")
