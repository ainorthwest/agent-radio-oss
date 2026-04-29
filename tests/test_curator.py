"""Tests for curator — JSON extraction and error handling."""

import json

import pytest

from src.curator import _extract_script_json


class TestJsonExtraction:
    """Tests for the LLM response JSON extraction logic."""

    def test_extracts_json_from_fenced_block(self):
        """Should extract JSON from markdown code fences."""
        response = '```json\n{"title": "Test", "segments": []}\n```'
        result = _extract_script_json(response)
        assert result["title"] == "Test"

    def test_extracts_json_without_json_tag(self):
        """Should extract JSON from bare code fences (no json tag)."""
        response = '```\n{"title": "Test", "segments": []}\n```'
        result = _extract_script_json(response)
        assert result["title"] == "Test"

    def test_extracts_json_from_raw_response(self):
        """Should parse raw JSON when no code fences present."""
        response = '{"title": "Test", "segments": []}'
        result = _extract_script_json(response)
        assert result["title"] == "Test"

    def test_skips_non_json_fenced_blocks(self):
        """Should skip non-JSON fenced blocks and find the JSON one."""
        response = (
            "Here's my analysis:\n"
            "```\nThis is just text, not JSON.\n```\n"
            "And here's the script:\n"
            '```json\n{"title": "Correct", "segments": []}\n```'
        )
        result = _extract_script_json(response)
        assert result["title"] == "Correct"

    def test_raises_on_malformed_response(self):
        """Should raise ValueError for completely unparseable responses."""
        response = "I couldn't generate a script because the forum was empty."
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _extract_script_json(response)

    def test_handles_whitespace_around_json(self):
        """Should handle extra whitespace inside code fences."""
        response = '```json\n\n  {"title": "Test", "segments": []}  \n\n```'
        result = _extract_script_json(response)
        assert result["title"] == "Test"
