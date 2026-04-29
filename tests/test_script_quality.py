"""Tests for script quality evaluation — Phase 2A structure + Phase 2B quality signals.

All tests use synthetic script dicts. No external dependencies.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.script_quality import (
    _compute_alternation,
    _compute_avg_word_length,
    _compute_jargon_density,
    _compute_ttr,
    _count_questions,
    _count_sentences,
    _count_tags,
    _count_words,
    _get_words_lower,
    _score_connector_frequency,
    _score_duration_estimate,
    _score_hook_density,
    _score_register_balance,
    _score_speaker_balance,
    _score_turn_length,
    _score_vocabulary_level,
    evaluate_script,
    main,
)

# ── Fixture helpers ──────────────────────────────────────────────────────────


def _seg(
    speaker: str = "host_a",
    text: str = "This is a test segment with about ten words in it.",
    register: str = "baseline",
    topic: str = "Test",
) -> dict[str, str]:
    return {"speaker": speaker, "text": text, "register": register, "topic": topic}


def _make_script(
    segments: list[dict[str, str]],
    title: str = "Test Episode",
    date: str = "2026-03-12",
) -> dict[str, Any]:
    return {"title": title, "date": date, "segments": segments}


# ── Synthetic scripts ────────────────────────────────────────────────────────

# 20 segments, ~40 words each, alternating speakers, mixed registers
_BALANCED_TEXT_A = "The field of artificial intelligence continues to evolve rapidly with new breakthroughs appearing every week. Researchers are finding novel approaches to problems that seemed intractable just a few years ago."
_BALANCED_TEXT_B = "What strikes me most is how these advances are changing the way we think about creativity and problem solving. The boundary between human and machine capability keeps shifting in surprising directions."


def _good_script() -> dict[str, Any]:
    """20 segments, balanced speakers, good register mix, ~750 words."""
    segs = []
    registers = ["baseline"] * 13 + ["emphasis"] * 3 + ["reflective"] * 2 + ["reactive"] * 2
    for i in range(20):
        speaker = "host_a" if i % 2 == 0 else "host_b"
        text = _BALANCED_TEXT_A if i % 2 == 0 else _BALANCED_TEXT_B
        segs.append(_seg(speaker=speaker, text=text, register=registers[i]))
    # Ensure last segment is host_b
    segs[-1]["speaker"] = "host_b"
    return _make_script(segs)


def _monologue_script() -> dict[str, Any]:
    """5 segments, one is 200 words — fails turn length."""
    long_text = " ".join(["word"] * 200)
    segs = [
        _seg(speaker="host_a", text=long_text),
        _seg(speaker="host_b", text="Short reply."),
        _seg(speaker="host_a", text="Another short one here."),
        _seg(speaker="host_b", text="And this too."),
        _seg(speaker="host_a", text="Final words here now."),
    ]
    return _make_script(segs)


def _empty_script() -> dict[str, Any]:
    """Empty segments list."""
    return _make_script([])


def _one_speaker_script() -> dict[str, Any]:
    """15 segments, all host_a."""
    segs = [_seg(speaker="host_a", text=_BALANCED_TEXT_A, register="baseline") for _ in range(15)]
    return _make_script(segs)


def _all_emphasis_script() -> dict[str, Any]:
    """16 segments, every register = emphasis."""
    segs = []
    for i in range(16):
        speaker = "host_a" if i % 2 == 0 else "host_b"
        segs.append(_seg(speaker=speaker, text=_BALANCED_TEXT_A, register="emphasis"))
    return _make_script(segs)


def _short_script() -> dict[str, Any]:
    """5 segments, ~50 words total — too short."""
    segs = [
        _seg(speaker="host_a" if i % 2 == 0 else "host_b", text="Just a few words.")
        for i in range(5)
    ]
    return _make_script(segs)


def _tag_heavy_script() -> dict[str, Any]:
    """15 segments with many paralinguistic tags."""
    tag_text = (
        "[laugh] So [chuckle] I was saying [cough] that [sigh] this [gasp] needs [groan] work."
    )
    segs = []
    for i in range(15):
        speaker = "host_a" if i % 2 == 0 else "host_b"
        segs.append(_seg(speaker=speaker, text=tag_text, register="baseline"))
    return _make_script(segs)


# ── Tests: SegmentStats ─────────────────────────────────────────────────────


class TestSegmentStats:
    def test_word_count_basic(self) -> None:
        assert _count_words("Hello world this is a test") == 6

    def test_word_count_strips_bracket_tags(self) -> None:
        assert _count_words("[laugh] Hello world [sigh]") == 2

    def test_word_count_strips_paren_tags(self) -> None:
        assert _count_words("(laughs) Hello (sighs) world") == 2

    def test_word_count_mixed_tags(self) -> None:
        assert _count_words("[laugh] Hello (sighs) world [cough]") == 2

    def test_tag_count_bracket(self) -> None:
        assert _count_tags("[laugh] text [cough]") == 2

    def test_tag_count_paren(self) -> None:
        assert _count_tags("(laughs) text (sighs)") == 2

    def test_tag_count_mixed(self) -> None:
        assert _count_tags("[laugh] (sighs) [gasp]") == 3

    def test_tag_count_no_tags(self) -> None:
        assert _count_tags("Just plain text here.") == 0

    def test_sentence_count(self) -> None:
        assert _count_sentences("One sentence. Two sentences. Three!") == 3

    def test_sentence_count_with_tags(self) -> None:
        assert _count_sentences("[laugh] One sentence. Two sentences.") == 2

    def test_sentence_count_ellipsis(self) -> None:
        # Ellipsis may split oddly but shouldn't crash
        result = _count_sentences("Wait... really? Yes.")
        assert result >= 2


# ── Tests: Turn Length ───────────────────────────────────────────────────────


class TestTurnLength:
    def test_ideal_scores_high(self) -> None:
        words = [35, 40, 30, 45, 38, 42, 33, 37, 41, 36]
        score, notes = _score_turn_length(words)
        assert score == 1.0
        assert not notes

    def test_monologue_scores_zero(self) -> None:
        words = [200, 5, 5, 5, 5]
        score, notes = _score_turn_length(words)
        assert score == 0.0
        assert any("FAIL" in n for n in notes)

    def test_empty_scores_zero(self) -> None:
        score, notes = _score_turn_length([])
        assert score == 0.0

    def test_slightly_long_scores_half(self) -> None:
        words = [70, 30, 40, 35, 25]  # max 70 > 60 but <= 80
        score, notes = _score_turn_length(words)
        assert score == 0.5
        assert any("WARN" in n for n in notes)


# ── Tests: Register Balance ──────────────────────────────────────────────────


class TestRegisterBalance:
    def test_ideal_mix(self) -> None:
        counts = {"baseline": 14, "emphasis": 3, "reflective": 2, "reactive": 1}
        score, notes = _score_register_balance(counts, 20)
        assert score == 1.0

    def test_all_baseline_fails(self) -> None:
        counts = {"baseline": 20}
        score, notes = _score_register_balance(counts, 20)
        assert score == 0.0
        assert any("FAIL" in n for n in notes)

    def test_no_baseline_fails(self) -> None:
        counts = {"emphasis": 10, "reflective": 5, "reactive": 5}
        score, notes = _score_register_balance(counts, 20)
        assert score == 0.0

    def test_slightly_off_warns(self) -> None:
        # 55% baseline — between 50% and 60% → 0.5
        counts = {"baseline": 11, "emphasis": 4, "reflective": 3, "reactive": 2}
        score, notes = _score_register_balance(counts, 20)
        assert score == 0.5

    def test_empty_script_fails(self) -> None:
        score, notes = _score_register_balance({}, 0)
        assert score == 0.0


# ── Tests: Speaker Balance ───────────────────────────────────────────────────


class TestSpeakerBalance:
    def test_equal_split(self) -> None:
        words = {"host_a": 400, "host_b": 380}
        segs = [
            {"speaker": "host_a"},
            {"speaker": "host_b"},
            {"speaker": "host_a"},
            {"speaker": "host_b"},
        ]
        score, notes = _score_speaker_balance(words, segs)
        assert score >= 0.5

    def test_one_speaker_fails(self) -> None:
        words = {"host_a": 500}
        segs = [{"speaker": "host_a"} for _ in range(10)]
        score, notes = _score_speaker_balance(words, segs)
        assert score == 0.0
        assert any("FAIL" in n for n in notes)

    def test_alternation_host_a_opens(self) -> None:
        segs = [
            {"speaker": "host_a"},
            {"speaker": "host_b"},
            {"speaker": "host_a"},
            {"speaker": "host_b"},
        ]
        score = _compute_alternation(segs)
        assert score == 1.0

    def test_alternation_wrong_open_close(self) -> None:
        segs = [
            {"speaker": "host_b"},
            {"speaker": "host_a"},
            {"speaker": "host_b"},
            {"speaker": "host_a"},
        ]
        score = _compute_alternation(segs)
        assert score < 1.0  # penalized for wrong open/close

    def test_alternation_long_run_penalized(self) -> None:
        segs = [
            {"speaker": "host_a"},
            {"speaker": "host_a"},
            {"speaker": "host_a"},  # run of 3
            {"speaker": "host_b"},
        ]
        score = _compute_alternation(segs)
        assert score < 1.0


# ── Tests: Connector Frequency ───────────────────────────────────────────────


class TestConnectorFrequency:
    def test_moderate_tags_ok(self) -> None:
        score, notes = _score_connector_frequency(2, 1, 15)
        assert score == 1.0

    def test_excessive_tags_fail(self) -> None:
        score, notes = _score_connector_frequency(8, 2, 15)
        assert score == 0.0

    def test_no_reactive_warns(self) -> None:
        score, notes = _score_connector_frequency(1, 0, 15)
        assert score == 0.5
        assert any("reactive" in n.lower() for n in notes)

    def test_four_tags_warns(self) -> None:
        score, notes = _score_connector_frequency(4, 1, 15)
        assert score == 0.5


# ── Tests: Duration Estimate ─────────────────────────────────────────────────


class TestDurationEstimate:
    def test_on_target(self) -> None:
        # 750 words / 150 wpm = 5 min = 300s → perfect
        score, notes = _score_duration_estimate(750)
        assert score == 1.0

    def test_too_short(self) -> None:
        # 100 words / 150 wpm = 40s → way too short
        score, notes = _score_duration_estimate(100)
        assert score == 0.0
        assert any("FAIL" in n for n in notes)

    def test_too_long(self) -> None:
        # 2000 words / 150 wpm = 800s → way too long
        score, notes = _score_duration_estimate(2000)
        assert score == 0.0
        assert any("FAIL" in n for n in notes)

    def test_slightly_short_warn_band(self) -> None:
        # 500 words / 150 wpm = 200s → inside 180–240s warn band
        score, notes = _score_duration_estimate(500)
        assert score == 0.5
        assert any("WARN" in n for n in notes)

    def test_too_short_fail(self) -> None:
        # 400 words / 150 wpm = 160s → below 180s, hard fail
        score, notes = _score_duration_estimate(400)
        assert score == 0.0

    def test_slightly_long(self) -> None:
        # 950 words / 150 wpm = 380s → slightly long, 0.5
        score, notes = _score_duration_estimate(950)
        assert score == 0.5


# ── Tests: Composite ─────────────────────────────────────────────────────────


class TestComposite:
    def test_good_script_passes(self) -> None:
        report = evaluate_script(_good_script())
        assert report.overall_score >= 0.5
        assert report.segment_count == 20
        assert report.total_words > 0

    def test_empty_scores_zero(self) -> None:
        report = evaluate_script(_empty_script())
        assert report.overall_score == 0.0
        assert any("FAIL" in n for n in report.notes)

    def test_one_speaker_penalized(self) -> None:
        report = evaluate_script(_one_speaker_script())
        assert report.overall_score < 0.8  # speaker balance drags it down

    def test_all_emphasis_penalized(self) -> None:
        report = evaluate_script(_all_emphasis_script())
        assert report.overall_score < 0.8  # register balance fails

    def test_short_script_penalized(self) -> None:
        report = evaluate_script(_short_script())
        assert report.overall_score < 0.5

    def test_notes_populated(self) -> None:
        report = evaluate_script(_monologue_script())
        assert len(report.notes) > 0

    def test_segment_stats_present(self) -> None:
        report = evaluate_script(_good_script())
        assert len(report.segment_stats) == 20
        assert "word_count" in report.segment_stats[0]

    def test_tag_heavy_flagged(self) -> None:
        report = evaluate_script(_tag_heavy_script())
        # Total tags across all segments should be flagged
        assert report.paralinguistic_tag_count > TARGET_MAX_TAGS
        assert any("tag" in n.lower() for n in report.notes)


TARGET_MAX_TAGS = 3  # mirrored for assertion


# ── Tests: Hook Density (Phase 2B) ──────────────────────────────────────────


def _hook_script() -> dict[str, Any]:
    """20 segments with questions in opening and closing."""
    segs = []
    registers = ["baseline"] * 13 + ["emphasis"] * 3 + ["reflective"] * 2 + ["reactive"] * 2
    for i in range(20):
        speaker = "host_a" if i % 2 == 0 else "host_b"
        if i == 0:
            text = "Have you ever wondered why some ideas stick and others don't? Today we explore."
        elif i == 1:
            text = "That's a great question. What if it comes down to framing?"
        elif i == 18:
            text = "So where does this leave us? Is there a formula for resonance?"
        elif i == 19:
            text = "Head to the forum and tell us what you think. Until next time?"
        else:
            text = _BALANCED_TEXT_A if i % 2 == 0 else _BALANCED_TEXT_B
        segs.append(_seg(speaker=speaker, text=text, register=registers[i]))
    segs[-1]["speaker"] = "host_b"
    return _make_script(segs)


def _no_hook_script() -> dict[str, Any]:
    """20 segments, no questions anywhere."""
    segs = []
    registers = ["baseline"] * 13 + ["emphasis"] * 3 + ["reflective"] * 2 + ["reactive"] * 2
    for i in range(20):
        speaker = "host_a" if i % 2 == 0 else "host_b"
        text = "This is a declarative statement with no engagement hooks at all."
        segs.append(_seg(speaker=speaker, text=text, register=registers[i]))
    segs[-1]["speaker"] = "host_b"
    return _make_script(segs)


class TestHookDensity:
    def test_questions_in_opening_and_closing(self) -> None:
        segs = [
            _seg(text="What is AI? Let's find out."),
            _seg(speaker="host_b", text="Good question. Here's what we know."),
            _seg(text="The details are interesting."),
            _seg(speaker="host_b", text="Indeed they are."),
            _seg(text="So what do you think?"),
        ]
        score, notes, stats = _score_hook_density(segs)
        assert stats["opening_questions"] >= 1
        assert stats["closing_questions"] >= 1
        assert score == 1.0

    def test_no_questions_fails(self) -> None:
        segs = [
            _seg(text="Welcome to the show."),
            _seg(speaker="host_b", text="Today we discuss things."),
            _seg(text="This is interesting."),
            _seg(speaker="host_b", text="Very interesting indeed."),
            _seg(text="That wraps it up."),
        ]
        score, notes, stats = _score_hook_density(segs)
        assert score == 0.0
        assert any("FAIL" in n for n in notes)

    def test_opening_only_half_score(self) -> None:
        segs = [
            _seg(text="What makes AI tick? Let's explore."),
            _seg(speaker="host_b", text="Great opening question."),
            _seg(text="Details follow."),
            _seg(speaker="host_b", text="More details."),
            _seg(text="That's all for today."),
        ]
        score, notes, stats = _score_hook_density(segs)
        assert stats["opening_questions"] >= 1
        assert stats["closing_questions"] == 0
        assert score == 0.5
        assert any("closing" in n.lower() for n in notes)

    def test_topic_labels_boost(self) -> None:
        segs = [
            _seg(text="Welcome to the show.", topic="Hook"),
            _seg(speaker="host_b", text="Today we discuss things."),
            _seg(text="This is interesting."),
            _seg(speaker="host_b", text="Very interesting indeed."),
            _seg(text="Head to the forum?", topic="Teaser"),
        ]
        score, notes, stats = _score_hook_density(segs)
        assert stats["hook_topics"] >= 1
        # Closing question only → 0.5, topic labels bump to 0.75
        assert score == 0.75

    def test_empty_segments(self) -> None:
        score, notes, stats = _score_hook_density([])
        assert score == 0.0

    def test_count_questions(self) -> None:
        assert _count_questions("What? Really? Yes.") == 2
        assert _count_questions("No questions here.") == 0
        assert _count_questions("[laugh] Is this working?") == 1


# ── Tests: Vocabulary Level (Phase 2B) ──────────────────────────────────────


class TestVocabularyHelpers:
    def test_get_words_lower(self) -> None:
        words = _get_words_lower("Hello World! [laugh] Test-case.")
        assert "hello" in words
        assert "world" in words
        assert "test-case" in words
        assert "[laugh]" not in words

    def test_ttr_unique_words(self) -> None:
        words = ["the", "cat", "sat", "on", "the", "mat"]
        ttr = _compute_ttr(words)
        assert ttr == 5 / 6  # 5 unique out of 6

    def test_ttr_empty(self) -> None:
        assert _compute_ttr([]) == 0.0

    def test_avg_word_length(self) -> None:
        words = ["hi", "hello", "hey"]  # 2 + 5 + 3 = 10 / 3
        awl = _compute_avg_word_length(words)
        assert abs(awl - 10 / 3) < 0.01

    def test_avg_word_length_empty(self) -> None:
        assert _compute_avg_word_length([]) == 0.0

    def test_jargon_density(self) -> None:
        words = ["the", "transformer", "model", "uses", "attention"]
        jd = _compute_jargon_density(words)
        # transformer + attention = 2 jargon out of 5
        assert abs(jd - 0.4) < 0.01

    def test_jargon_density_no_jargon(self) -> None:
        words = ["the", "cat", "sat", "on", "the", "mat"]
        assert _compute_jargon_density(words) == 0.0


class TestVocabularyLevel:
    def test_conversational_scores_high(self) -> None:
        # Normal conversational text
        words = _get_words_lower(
            "This is a normal conversational podcast about ideas and creativity. "
            "We explore different perspectives on how people think and work together. "
            "The boundary between human and machine capability keeps shifting."
        )
        score, notes, stats = _score_vocabulary_level(words)
        assert score >= 0.5

    def test_highly_technical_penalized(self) -> None:
        # Dense jargon text
        words = _get_words_lower(
            "The transformer architecture uses attention mechanisms and backpropagation "
            "with gradient descent. Quantization and distillation improve inference "
            "latency. The tokenizer handles embedding normalization. RLHF alignment "
            "training reduces hallucination through reinforcement regularization."
        )
        score, notes, stats = _score_vocabulary_level(words)
        assert stats["jargon_density"] > 0.05
        assert score < 1.0

    def test_empty_words_fails(self) -> None:
        score, notes, stats = _score_vocabulary_level([])
        assert score == 0.0

    def test_repetitive_vocabulary(self) -> None:
        # Very low TTR
        words = ["the"] * 50 + ["cat"] * 50
        score, notes, stats = _score_vocabulary_level(words)
        assert stats["ttr"] < 0.45
        assert any("repetitive" in n.lower() for n in notes)


# ── Tests: Gap Hint Coverage (Phase 2B — inert) ─────────────────────────────


class TestGapHintCoverage:
    def test_no_gap_hints_is_zero(self) -> None:
        report = evaluate_script(_good_script())
        assert report.gap_hint_coverage == 0.0

    def test_gap_hints_when_present(self) -> None:
        """If a future curator emits gap_hint, the metric should detect it."""
        segs = [
            {"speaker": "host_a", "text": "Hello.", "register": "baseline", "topic": "Test"},
            {
                "speaker": "host_b",
                "text": "World.",
                "register": "baseline",
                "topic": "Test",
                "gap_hint": "topic_change",
            },
        ]
        # Can't use evaluate_script directly due to low segment count,
        # but can verify the field is read
        script = _make_script(segs)
        report = evaluate_script(script)
        assert report.gap_hint_coverage == 0.5  # 1 of 2 segments


# ── Tests: Composite with 2B ────────────────────────────────────────────────


class TestCompositeWith2B:
    def test_hook_script_scores_higher(self) -> None:
        hook_report = evaluate_script(_hook_script())
        no_hook_report = evaluate_script(_no_hook_script())
        assert hook_report.hook_density_score > no_hook_report.hook_density_score

    def test_vocabulary_in_report(self) -> None:
        report = evaluate_script(_good_script())
        assert report.type_token_ratio > 0
        assert report.avg_word_length > 0
        assert isinstance(report.jargon_words_found, list)

    def test_json_includes_2b_fields(self) -> None:
        report = evaluate_script(_good_script())
        data = json.loads(report.to_json())
        assert "hook_density_score" in data
        assert "type_token_ratio" in data
        assert "avg_word_length" in data
        assert "jargon_density" in data
        assert "gap_hint_coverage" in data
        assert "opening_question_count" in data

    def test_display_includes_hooks_and_vocab(self, tmp_path: Any) -> None:
        script_path = tmp_path / "test.json"
        script_path.write_text(json.dumps(_hook_script()))

        import contextlib
        import io
        import sys as _sys

        f = io.StringIO()
        old_argv = _sys.argv
        _sys.argv = ["script_quality", str(script_path)]
        try:
            with contextlib.redirect_stdout(f):
                main()
        finally:
            _sys.argv = old_argv

        output = f.getvalue()
        assert "Hook density" in output
        assert "Vocabulary" in output
        assert "Type-token ratio" in output
        assert "Opening questions" in output


# ── Tests: CLI ───────────────────────────────────────────────────────────────


class TestCLI:
    def test_json_output_parseable(self, tmp_path: Any) -> None:
        script_path = tmp_path / "test.json"
        script_path.write_text(json.dumps(_good_script()))

        import contextlib
        import io

        f = io.StringIO()
        import sys as _sys

        old_argv = _sys.argv
        _sys.argv = ["script_quality", str(script_path), "--json"]
        try:
            with contextlib.redirect_stdout(f):
                main()
        finally:
            _sys.argv = old_argv

        output = f.getvalue()
        data = json.loads(output)
        assert "overall_score" in data
        assert isinstance(data["overall_score"], float)

    def test_strict_mode_exit_code(self, tmp_path: Any) -> None:
        script_path = tmp_path / "bad.json"
        script_path.write_text(json.dumps(_empty_script()))

        import sys as _sys

        old_argv = _sys.argv
        _sys.argv = ["script_quality", str(script_path), "--strict", "--threshold", "0.5"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        finally:
            _sys.argv = old_argv

    def test_human_readable_output(self, tmp_path: Any) -> None:
        script_path = tmp_path / "test.json"
        script_path.write_text(json.dumps(_good_script()))

        import contextlib
        import io

        f = io.StringIO()
        import sys as _sys

        old_argv = _sys.argv
        _sys.argv = ["script_quality", str(script_path)]
        try:
            with contextlib.redirect_stdout(f):
                main()
        finally:
            _sys.argv = old_argv

        output = f.getvalue()
        assert "Script Quality Report" in output
        assert "COMPOSITE SCORE" in output


# ── Tests: to_json ───────────────────────────────────────────────────────────


class TestReportSerialization:
    def test_to_json_roundtrip(self) -> None:
        report = evaluate_script(_good_script())
        data = json.loads(report.to_json())
        assert data["title"] == "Test Episode"
        assert isinstance(data["segment_stats"], list)
        assert isinstance(data["notes"], list)
