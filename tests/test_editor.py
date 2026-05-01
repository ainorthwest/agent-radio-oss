"""Tests for src/editor.py — pure JSON operations on script.json.

The editor takes a script dict, applies an operation, and returns
``(new_script, ScriptDiff)``. No in-place mutation. The diff is what
the renderer uses to decide which segments to re-render and which to
load from the per-segment cache.
"""

from __future__ import annotations

import copy

import pytest

SAMPLE_SCRIPT = {
    "title": "Demo",
    "date": "sample",
    "program": "haystack-news",
    "segments": [
        {"speaker": "host_a", "text": "First segment.", "topic": "intro", "register": "baseline"},
        {"speaker": "host_b", "text": "Second segment.", "topic": "intro", "register": "baseline"},
        {"speaker": "host_c", "text": "Third segment.", "topic": "main", "register": "emphasis"},
        {"speaker": "host_a", "text": "Fourth segment.", "topic": "main", "register": "baseline"},
    ],
}


# ── delete_segment ───────────────────────────────────────────────────────────


class TestDeleteSegment:
    def test_removes_segment_at_index(self):
        from src.editor import delete_segment

        new_script, diff = delete_segment(SAMPLE_SCRIPT, 1)
        assert len(new_script["segments"]) == 3
        assert new_script["segments"][0]["text"] == "First segment."
        assert new_script["segments"][1]["text"] == "Third segment."
        assert diff.removed == [1]
        assert diff.added == []
        assert diff.modified == []
        assert diff.reordered is False

    def test_does_not_mutate_input(self):
        from src.editor import delete_segment

        original = copy.deepcopy(SAMPLE_SCRIPT)
        delete_segment(SAMPLE_SCRIPT, 0)
        assert SAMPLE_SCRIPT == original

    def test_preserves_top_level_metadata(self):
        from src.editor import delete_segment

        new_script, _ = delete_segment(SAMPLE_SCRIPT, 0)
        assert new_script["title"] == "Demo"
        assert new_script["date"] == "sample"
        assert new_script["program"] == "haystack-news"

    def test_out_of_range_raises(self):
        from src.editor import delete_segment

        with pytest.raises(IndexError):
            delete_segment(SAMPLE_SCRIPT, 99)

    def test_negative_index_raises(self):
        """Negative indices would be ambiguous — disallow."""
        from src.editor import delete_segment

        with pytest.raises(IndexError):
            delete_segment(SAMPLE_SCRIPT, -1)


# ── replace_text ─────────────────────────────────────────────────────────────


class TestReplaceText:
    def test_replaces_text_in_place(self):
        from src.editor import replace_text

        new_script, diff = replace_text(SAMPLE_SCRIPT, 2, "New third text.")
        assert new_script["segments"][2]["text"] == "New third text."
        assert new_script["segments"][2]["speaker"] == "host_c"  # other fields preserved
        assert new_script["segments"][2]["register"] == "emphasis"
        assert diff.modified == [2]
        assert diff.added == []
        assert diff.removed == []
        assert diff.reordered is False

    def test_does_not_mutate_input(self):
        from src.editor import replace_text

        original = copy.deepcopy(SAMPLE_SCRIPT)
        replace_text(SAMPLE_SCRIPT, 0, "Different.")
        assert SAMPLE_SCRIPT == original

    def test_empty_text_raises(self):
        from src.editor import replace_text

        with pytest.raises(ValueError, match="empty"):
            replace_text(SAMPLE_SCRIPT, 0, "")

    def test_whitespace_only_text_raises(self):
        from src.editor import replace_text

        with pytest.raises(ValueError, match="empty"):
            replace_text(SAMPLE_SCRIPT, 0, "   ")

    def test_out_of_range_raises(self):
        from src.editor import replace_text

        with pytest.raises(IndexError):
            replace_text(SAMPLE_SCRIPT, 99, "anything")


# ── reorder_segments ─────────────────────────────────────────────────────────


class TestReorderSegments:
    def test_reorders_by_index_list(self):
        from src.editor import reorder_segments

        new_script, diff = reorder_segments(SAMPLE_SCRIPT, [3, 0, 1, 2])
        assert new_script["segments"][0]["text"] == "Fourth segment."
        assert new_script["segments"][1]["text"] == "First segment."
        assert new_script["segments"][2]["text"] == "Second segment."
        assert new_script["segments"][3]["text"] == "Third segment."
        assert diff.reordered is True
        assert diff.added == []
        assert diff.removed == []
        assert diff.modified == []

    def test_identity_order_marks_not_reordered(self):
        """Reordering to the same order is a no-op — flag accordingly."""
        from src.editor import reorder_segments

        new_script, diff = reorder_segments(SAMPLE_SCRIPT, [0, 1, 2, 3])
        assert new_script["segments"] == SAMPLE_SCRIPT["segments"]
        assert diff.reordered is False

    def test_does_not_mutate_input(self):
        from src.editor import reorder_segments

        original = copy.deepcopy(SAMPLE_SCRIPT)
        reorder_segments(SAMPLE_SCRIPT, [3, 2, 1, 0])
        assert SAMPLE_SCRIPT == original

    def test_wrong_length_raises(self):
        from src.editor import reorder_segments

        with pytest.raises(ValueError, match="length"):
            reorder_segments(SAMPLE_SCRIPT, [0, 1, 2])

    def test_duplicate_indices_raises(self):
        from src.editor import reorder_segments

        with pytest.raises(ValueError, match="permutation"):
            reorder_segments(SAMPLE_SCRIPT, [0, 0, 2, 3])

    def test_out_of_range_index_raises(self):
        from src.editor import reorder_segments

        with pytest.raises(ValueError, match="permutation"):
            reorder_segments(SAMPLE_SCRIPT, [0, 1, 2, 99])


# ── insert_segment ───────────────────────────────────────────────────────────


class TestInsertSegment:
    def test_inserts_at_index(self):
        from src.editor import insert_segment

        new_seg = {
            "speaker": "host_b",
            "text": "Inserted!",
            "topic": "intro",
            "register": "baseline",
        }
        new_script, diff = insert_segment(SAMPLE_SCRIPT, 2, new_seg)
        assert len(new_script["segments"]) == 5
        assert new_script["segments"][2]["text"] == "Inserted!"
        assert new_script["segments"][3]["text"] == "Third segment."
        assert diff.added == [2]
        assert diff.removed == []
        assert diff.modified == []

    def test_insert_at_end(self):
        from src.editor import insert_segment

        new_seg = {"speaker": "host_a", "text": "Tail.", "topic": "outro", "register": "baseline"}
        new_script, diff = insert_segment(SAMPLE_SCRIPT, 4, new_seg)
        assert new_script["segments"][-1]["text"] == "Tail."
        assert diff.added == [4]

    def test_does_not_mutate_input(self):
        from src.editor import insert_segment

        original = copy.deepcopy(SAMPLE_SCRIPT)
        new_seg = {"speaker": "host_a", "text": "x", "topic": "x", "register": "baseline"}
        insert_segment(SAMPLE_SCRIPT, 0, new_seg)
        assert SAMPLE_SCRIPT == original

    def test_segment_missing_required_fields_raises(self):
        from src.editor import insert_segment

        with pytest.raises(ValueError, match="text"):
            insert_segment(SAMPLE_SCRIPT, 0, {"speaker": "host_a", "register": "baseline"})

    def test_out_of_range_raises(self):
        from src.editor import insert_segment

        new_seg = {"speaker": "host_a", "text": "x", "topic": "x", "register": "baseline"}
        with pytest.raises(IndexError):
            insert_segment(SAMPLE_SCRIPT, 99, new_seg)


# ── change_voice ─────────────────────────────────────────────────────────────


class TestChangeVoice:
    def test_changes_speaker(self):
        from src.editor import change_voice

        new_script, diff = change_voice(SAMPLE_SCRIPT, 0, "host_c")
        assert new_script["segments"][0]["speaker"] == "host_c"
        assert new_script["segments"][0]["text"] == "First segment."  # text unchanged
        assert diff.modified == [0]

    def test_no_op_change_marks_unmodified(self):
        """Changing to the same speaker is a no-op."""
        from src.editor import change_voice

        new_script, diff = change_voice(SAMPLE_SCRIPT, 0, "host_a")
        assert new_script["segments"] == SAMPLE_SCRIPT["segments"]
        assert diff.modified == []

    def test_does_not_mutate_input(self):
        from src.editor import change_voice

        original = copy.deepcopy(SAMPLE_SCRIPT)
        change_voice(SAMPLE_SCRIPT, 1, "host_c")
        assert SAMPLE_SCRIPT == original

    def test_out_of_range_raises(self):
        from src.editor import change_voice

        with pytest.raises(IndexError):
            change_voice(SAMPLE_SCRIPT, 99, "host_a")


# ── ScriptDiff serialization ─────────────────────────────────────────────────


class TestScriptDiff:
    def test_to_dict_serializable(self):
        from src.editor import ScriptDiff

        diff = ScriptDiff(added=[2], removed=[0], modified=[1, 3], reordered=True)
        d = diff.to_dict()
        assert d == {"added": [2], "removed": [0], "modified": [1, 3], "reordered": True}

    def test_default_diff_is_empty(self):
        from src.editor import ScriptDiff

        diff = ScriptDiff()
        assert diff.added == []
        assert diff.removed == []
        assert diff.modified == []
        assert diff.reordered is False
