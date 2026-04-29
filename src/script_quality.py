"""Script quality evaluation: structural + editorial metrics for pre-render gating.

Pure-stdlib module — no external dependencies. Evaluates curator output
against target constraints before spending TTS compute on rendering.

Phase 2A — Structure metrics:
  1. Segment count — episode length vs target range
  2. Turn length — word distribution across segments
  3. Register balance — baseline ratio + non-baseline variety
  4. Speaker balance — word equity + alternation pattern
  5. Connector frequency — paralinguistic tags + reactive segments
  6. Duration estimate — projected runtime at conversational pace

Phase 2B — Quality signals:
  7. Hook density — questions and engagement devices in opening/closing
  8. Vocabulary level — type-token ratio, avg word length, jargon density
  (Gap hint coverage deferred — curator doesn't produce gap_hint field yet)

Standalone:
    uv run python -m src.script_quality script.json
    uv run python -m src.script_quality script.json --json
    uv run python -m src.script_quality script.json --strict
    uv run python -m src.script_quality script.json --threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Constants (from curator prompt targets) ──────────────────────────────────

WORDS_PER_MINUTE = 150  # conversational podcast rate
TARGET_DURATION_SECONDS = 300  # 5 minutes
TARGET_SEGMENTS = (15, 25)
TARGET_WORDS = (600, 900)
TARGET_BASELINE_RATIO = (0.60, 0.80)
TARGET_NON_BASELINE = (2, 4)  # distinct non-baseline register count
TARGET_MAX_TAGS = 3
TARGET_MAX_WORDS_PER_SEG = 60  # ~4 sentences max
VALID_REGISTERS = {"baseline", "emphasis", "reflective", "reactive"}

# Hook detection — opening = first N segments at 150 wpm ≈ first 30s
HOOK_OPENING_SEGMENTS = 4  # ~30s of content at conversational pace
HOOK_CLOSING_SEGMENTS = 2  # closing teaser zone

# Structural topic labels the curator sometimes emits (bonus signal)
_HOOK_TOPIC_LABELS = {
    "hook",
    "opening",
    "opening transition",
    "teaser",
    "intro",
    "closing",
    "closing reflection",
    "outro",
}

# Vocabulary — AI/ML domain jargon list for jargon density scoring
# Terms the curator prompt says to "translate into accessible conversation"
_JARGON_TERMS = frozenset(
    {
        "llm",
        "gpu",
        "tpu",
        "api",
        "sdk",
        "rlhf",
        "rag",
        "embeddings",
        "fine-tuning",
        "finetuning",
        "transformer",
        "transformers",
        "attention",
        "diffusion",
        "latent",
        "inference",
        "backpropagation",
        "gradient",
        "stochastic",
        "epoch",
        "hyperparameter",
        "hyperparameters",
        "tokenizer",
        "tokenization",
        "benchmark",
        "benchmarks",
        "perplexity",
        "logits",
        "softmax",
        "autoregressive",
        "multimodal",
        "chain-of-thought",
        "reinforcement",
        "alignment",
        "hallucination",
        "hallucinations",
        "grounding",
        "retrieval-augmented",
        "few-shot",
        "zero-shot",
        "in-context",
        "context-window",
        "weights",
        "parameters",
        "quantization",
        "distillation",
        "pruning",
        "architecture",
        "eval",
        "evals",
        "overfitting",
        "underfitting",
        "regularization",
        "convolution",
        "recurrent",
        "lstm",
        "gpt",
        "bert",
        "whisper",
        "mfcc",
        "spectrogram",
        "spectral",
        "prosody",
        "phoneme",
        "phonemes",
        "corpus",
        "corpora",
        "latency",
        "throughput",
        "scalability",
        "idempotent",
        "dependency",
        "dependencies",
        "containerization",
        "kubernetes",
        "docker",
        "microservice",
        "microservices",
        "monolith",
        "ontology",
        "taxonomy",
        "heuristic",
        "heuristics",
        "bayesian",
        "stationarity",
        "eigenvalue",
        "eigenvector",
        "sigmoid",
        "relu",
        "normalization",
        "vectorization",
        "embedding",
    }
)

# Target vocabulary ranges (calibrated from real Agent Radio scripts)
TARGET_TTR = (0.45, 0.75)  # type-token ratio — too low = repetitive, too high = scattershot
TARGET_AVG_WORD_LEN = (3.5, 6.0)  # conversational podcast sweet spot
TARGET_JARGON_DENSITY = (0.0, 0.05)  # max 5% jargon — accessible, not academic

# Default quality thresholds (match pipeline.py gates)
DEFAULT_THRESHOLD = 0.5

# ── Tag detection (duplicated from renderer.py to keep module dependency-free)

_BRACKET_TAG_RE = re.compile(
    r"\[(?:laugh|chuckle|cough|sigh|gasp|sniff|groan|clear throat|shush)\]"
)

_PAREN_TAG_RE = re.compile(
    r"\((?:laughs|chuckle|sighs|gasps|coughs|groans|sniffs|screams|inhales|exhales|"
    r"clears throat|singing|sings|mumbles|beep|claps|applause|burps|humming|sneezes|whistles)\)"
)


def _count_tags(text: str) -> int:
    """Count all paralinguistic tags (bracket + parenthetical) in text."""
    return len(_BRACKET_TAG_RE.findall(text)) + len(_PAREN_TAG_RE.findall(text))


def _strip_tags(text: str) -> str:
    """Remove all engine-specific non-speech tags."""
    text = _BRACKET_TAG_RE.sub("", text)
    text = _PAREN_TAG_RE.sub("", text)
    return text


def _count_words(text: str) -> int:
    """Count words in text after stripping paralinguistic tags."""
    cleaned = _strip_tags(text)
    return len(cleaned.split())


def _count_sentences(text: str) -> int:
    """Count sentences by terminal punctuation (. ! ?)."""
    cleaned = _strip_tags(text)
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r"[.!?]+(?:\s|$)", cleaned.strip())
    # Filter empty strings from trailing split
    return len([s for s in sentences if s.strip()])


def _count_questions(text: str) -> int:
    """Count question marks in text (after stripping tags)."""
    return _strip_tags(text).count("?")


def _get_words_lower(text: str) -> list[str]:
    """Return lowercase word list after stripping tags and punctuation."""
    cleaned = _strip_tags(text)
    # Remove punctuation but keep hyphens within words (e.g. "chain-of-thought")
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    return [w.lower() for w in cleaned.split() if w]


def _compute_ttr(words: list[str]) -> float:
    """Type-token ratio: unique words / total words."""
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _compute_avg_word_length(words: list[str]) -> float:
    """Average word length in characters."""
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _compute_jargon_density(words: list[str]) -> float:
    """Fraction of words that are domain jargon."""
    if not words:
        return 0.0
    jargon_count = sum(1 for w in words if w in _JARGON_TERMS)
    return jargon_count / len(words)


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class SegmentStats:
    """Per-segment structural data."""

    index: int
    speaker: str
    register: str
    word_count: int
    sentence_count: int
    has_paralinguistic_tag: bool
    tag_count: int


@dataclass
class ScriptReport:
    """Full script evaluation — mirrors QualityReport pattern."""

    # Identity
    title: str = ""
    date: str = ""
    segment_count: int = 0
    total_words: int = 0
    # Turn length
    words_per_segment: list[int] = field(default_factory=list)
    mean_words_per_segment: float = 0.0
    max_words_per_segment: int = 0
    word_count_variance: float = 0.0
    # Connectors
    paralinguistic_tag_count: int = 0
    reactive_segment_count: int = 0
    connector_ratio: float = 0.0
    # Register
    register_counts: dict[str, int] = field(default_factory=dict)
    register_ratios: dict[str, float] = field(default_factory=dict)
    baseline_ratio: float = 0.0
    non_baseline_count: int = 0
    # Speaker
    words_by_speaker: dict[str, int] = field(default_factory=dict)
    speaker_ratio: float = 0.0  # min/max word ratio
    speaker_alternation_score: float = 0.0
    # Duration
    estimated_duration_seconds: float = 0.0
    duration_vs_target: float = 0.0
    # Hook density (Phase 2B)
    opening_question_count: int = 0
    closing_question_count: int = 0
    hook_topic_count: int = 0  # segments with structural topic labels
    hook_density_score: float = 0.0
    # Vocabulary level (Phase 2B)
    type_token_ratio: float = 0.0
    avg_word_length: float = 0.0
    jargon_density: float = 0.0
    jargon_words_found: list[str] = field(default_factory=list)
    # Gap hint coverage (Phase 2B — inert, curator doesn't emit gap_hint yet)
    gap_hint_coverage: float = 0.0  # always 0.0 until curator wired
    # Composite
    overall_score: float = 0.0
    dimension_scores: dict[str, float] = field(default_factory=dict)
    segment_stats: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ── Scoring functions ────────────────────────────────────────────────────────
# Each returns (score: float, notes: list[str])


def _score_segment_count(n: int) -> tuple[float, list[str]]:
    """Score segment count against target range."""
    notes: list[str] = []
    lo, hi = TARGET_SEGMENTS
    if lo <= n <= hi:
        return 1.0, notes
    if 10 <= n < lo:
        notes.append(f"[WARN] Segment count {n} slightly below target {lo}–{hi}")
        return 0.5, notes
    if hi < n <= 30:
        notes.append(f"[WARN] Segment count {n} slightly above target {lo}–{hi}")
        return 0.5, notes
    if n < 10:
        notes.append(f"[FAIL] Only {n} segments (target {lo}–{hi})")
    else:
        notes.append(f"[FAIL] {n} segments exceeds max 30 (target {lo}–{hi})")
    return 0.0, notes


def _score_turn_length(words_list: list[int]) -> tuple[float, list[str]]:
    """Score word distribution across segments."""
    notes: list[str] = []
    if not words_list:
        notes.append("[FAIL] No segments to evaluate turn length")
        return 0.0, notes

    mean_w = sum(words_list) / len(words_list)
    max_w = max(words_list)

    score = 1.0

    if mean_w < 5:
        notes.append(f"[FAIL] Mean words/segment {mean_w:.0f} too low (min 5)")
        return 0.0, notes

    if max_w > 80:
        notes.append(f"[FAIL] Longest segment {max_w} words exceeds 80")
        return 0.0, notes

    if max_w > TARGET_MAX_WORDS_PER_SEG:
        notes.append(
            f"[WARN] Longest segment {max_w} words exceeds target {TARGET_MAX_WORDS_PER_SEG}"
        )
        score = min(score, 0.5)

    if mean_w > TARGET_MAX_WORDS_PER_SEG:
        notes.append(
            f"[WARN] Mean words/segment {mean_w:.0f} exceeds target {TARGET_MAX_WORDS_PER_SEG}"
        )
        score = min(score, 0.5)

    return score, notes


def _score_register_balance(counts: dict[str, int], total: int) -> tuple[float, list[str]]:
    """Score register distribution."""
    notes: list[str] = []
    if total == 0:
        notes.append("[FAIL] No segments to evaluate register balance")
        return 0.0, notes

    baseline_n = counts.get("baseline", 0)
    ratio = baseline_n / total
    non_baseline = sum(1 for r, c in counts.items() if r != "baseline" and c > 0)

    lo, hi = TARGET_BASELINE_RATIO
    nb_lo, nb_hi = TARGET_NON_BASELINE

    if lo <= ratio <= hi and nb_lo <= non_baseline <= nb_hi:
        return 1.0, notes

    score = 1.0

    if ratio < 0.50:
        notes.append(f"[FAIL] Baseline ratio {ratio:.0%} below 50% — too much variety")
        return 0.0, notes
    if ratio > 0.90:
        notes.append(f"[FAIL] Baseline ratio {ratio:.0%} above 90% — too monotone")
        return 0.0, notes

    if ratio < lo:
        notes.append(f"[WARN] Baseline ratio {ratio:.0%} below target {lo:.0%}–{hi:.0%}")
        score = min(score, 0.5)
    elif ratio > hi:
        notes.append(f"[WARN] Baseline ratio {ratio:.0%} above target {lo:.0%}–{hi:.0%}")
        score = min(score, 0.5)

    if non_baseline < nb_lo:
        notes.append(
            f"[WARN] Only {non_baseline} non-baseline registers used (target {nb_lo}–{nb_hi})"
        )
        score = min(score, 0.5)

    return score, notes


def _score_speaker_balance(
    words_by_speaker: dict[str, int],
    segments: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    """Score speaker balance: word equity + alternation."""
    notes: list[str] = []
    if not words_by_speaker:
        notes.append("[FAIL] No speakers found")
        return 0.0, notes

    if len(words_by_speaker) < 2:
        notes.append("[FAIL] Single speaker — no dialogue")
        return 0.0, notes

    word_values = list(words_by_speaker.values())
    min_w = min(word_values)
    max_w = max(word_values)
    ratio = min_w / max_w if max_w > 0 else 0.0

    # Alternation scoring
    alt_score = _compute_alternation(segments)

    # Combine: 60% word ratio, 40% alternation
    combined = 0.6 * ratio + 0.4 * alt_score

    if ratio < 0.50:
        notes.append(f"[WARN] Speaker word ratio {ratio:.2f} — significant imbalance")

    if combined >= 0.70:
        score = 1.0
    elif combined >= 0.50:
        score = 0.5
        if ratio >= 0.50:
            notes.append(f"[WARN] Speaker balance {combined:.2f} — adequate but uneven")
    else:
        score = 0.0
        notes.append(f"[FAIL] Speaker balance {combined:.2f} — poor equity or alternation")

    return score, notes


def _compute_alternation(segments: list[dict[str, Any]]) -> float:
    """Score speaker alternation pattern (0–1).

    Checks: first seg = host_a, last seg = host_b, penalizes runs of 3+
    consecutive same-speaker segments.
    """
    if len(segments) < 2:
        return 0.0

    speakers = [s.get("speaker", "") for s in segments]
    score = 1.0

    # Opening/closing convention
    if speakers[0] != "host_a":
        score -= 0.15
    if speakers[-1] != "host_b":
        score -= 0.15

    # Penalize long runs (3+ same speaker in a row)
    run_length = 1
    run_penalties = 0
    for i in range(1, len(speakers)):
        if speakers[i] == speakers[i - 1]:
            run_length += 1
            if run_length >= 3:
                run_penalties += 1
        else:
            run_length = 1

    # Each 3+ run costs 0.1, capped at total deduction of 0.7
    score -= min(run_penalties * 0.1, 0.7)

    return max(0.0, score)


def _score_connector_frequency(
    tag_count: int, reactive_count: int, total_segments: int
) -> tuple[float, list[str]]:
    """Score paralinguistic tag frequency and reactive segment usage."""
    notes: list[str] = []
    if total_segments == 0:
        return 0.0, [("[FAIL] No segments")]

    score = 1.0

    if tag_count > 5:
        notes.append(f"[FAIL] {tag_count} paralinguistic tags — too many (max 5)")
        return 0.0, notes

    if tag_count > TARGET_MAX_TAGS:
        notes.append(f"[WARN] {tag_count} paralinguistic tags (target ≤{TARGET_MAX_TAGS})")
        score = min(score, 0.5)

    if reactive_count == 0:
        notes.append("[WARN] No reactive register segments — may feel flat")
        score = min(score, 0.5)

    return score, notes


def _score_duration_estimate(total_words: int) -> tuple[float, list[str]]:
    """Score estimated episode duration against target."""
    notes: list[str] = []
    if total_words == 0:
        notes.append("[FAIL] No words — zero duration")
        return 0.0, notes

    est_seconds = (total_words / WORDS_PER_MINUTE) * 60

    if 240 <= est_seconds <= 360:
        return 1.0, notes
    if 180 <= est_seconds < 240:
        notes.append(f"[WARN] Estimated {est_seconds:.0f}s — slightly short (target 240–360s)")
        return 0.5, notes
    if 360 < est_seconds <= 420:
        notes.append(f"[WARN] Estimated {est_seconds:.0f}s — slightly long (target 240–360s)")
        return 0.5, notes

    if est_seconds < 180:
        notes.append(f"[FAIL] Estimated {est_seconds:.0f}s — too short (target 240–360s)")
    else:
        notes.append(f"[FAIL] Estimated {est_seconds:.0f}s — too long (target 240–360s)")
    return 0.0, notes


# ── Phase 2B: Quality signal scoring ─────────────────────────────────────────


def _score_hook_density(
    segments: list[dict[str, Any]],
) -> tuple[float, list[str], dict[str, int]]:
    """Score engagement hooks in opening and closing segments.

    Returns (score, notes, stats_dict) where stats_dict has
    opening_questions, closing_questions, hook_topics.
    """
    notes: list[str] = []
    stats = {"opening_questions": 0, "closing_questions": 0, "hook_topics": 0}
    if not segments:
        notes.append("[FAIL] No segments to evaluate hooks")
        return 0.0, notes, stats

    n = len(segments)
    opening = segments[: min(HOOK_OPENING_SEGMENTS, n)]
    closing = segments[max(0, n - HOOK_CLOSING_SEGMENTS) :]

    # Count questions in opening and closing zones
    for seg in opening:
        stats["opening_questions"] += _count_questions(seg.get("text", ""))
    for seg in closing:
        stats["closing_questions"] += _count_questions(seg.get("text", ""))

    # Count structural topic labels anywhere in script
    for seg in segments:
        topic = seg.get("topic", "").lower().strip()
        if topic in _HOOK_TOPIC_LABELS:
            stats["hook_topics"] += 1

    # Scoring: opening needs at least 1 question, closing needs at least 1
    score = 0.0
    has_opening = stats["opening_questions"] >= 1
    has_closing = stats["closing_questions"] >= 1
    has_topic_labels = stats["hook_topics"] >= 1

    if has_opening and has_closing:
        score = 1.0
    elif has_opening or has_closing:
        score = 0.5
        missing = "closing" if not has_closing else "opening"
        notes.append(f"[WARN] No questions in {missing} segments")
    else:
        score = 0.0
        notes.append("[FAIL] No questions in opening or closing segments")

    # Bonus: topic labels bump a 0.5 to 0.75 (editorial intent signal)
    if has_topic_labels and score == 0.5:
        score = 0.75

    return score, notes, stats


def _score_vocabulary_level(
    all_words: list[str],
) -> tuple[float, list[str], dict[str, float | list[str]]]:
    """Score vocabulary complexity: TTR, avg word length, jargon density.

    Returns (score, notes, vocab_stats).
    """
    notes: list[str] = []
    vocab_stats: dict[str, float | list[str]] = {
        "ttr": 0.0,
        "avg_word_length": 0.0,
        "jargon_density": 0.0,
        "jargon_words_found": [],
    }
    if not all_words:
        notes.append("[FAIL] No words to evaluate vocabulary")
        return 0.0, notes, vocab_stats

    ttr = _compute_ttr(all_words)
    awl = _compute_avg_word_length(all_words)
    jd = _compute_jargon_density(all_words)

    # Collect actual jargon words found (deduplicated, sorted)
    jargon_found = sorted(set(w for w in all_words if w in _JARGON_TERMS))

    vocab_stats["ttr"] = ttr
    vocab_stats["avg_word_length"] = awl
    vocab_stats["jargon_density"] = jd
    vocab_stats["jargon_words_found"] = jargon_found

    # Score each sub-dimension, then average
    sub_scores: list[float] = []

    # TTR scoring
    ttr_lo, ttr_hi = TARGET_TTR
    if ttr_lo <= ttr <= ttr_hi:
        sub_scores.append(1.0)
    elif ttr < ttr_lo:
        notes.append(f"[WARN] Type-token ratio {ttr:.2f} — repetitive vocabulary")
        sub_scores.append(0.5)
    else:
        notes.append(f"[WARN] Type-token ratio {ttr:.2f} — unusually diverse")
        sub_scores.append(0.5)

    # Average word length scoring
    awl_lo, awl_hi = TARGET_AVG_WORD_LEN
    if awl_lo <= awl <= awl_hi:
        sub_scores.append(1.0)
    elif awl < awl_lo:
        notes.append(f"[WARN] Avg word length {awl:.1f} — too simple")
        sub_scores.append(0.5)
    else:
        notes.append(f"[WARN] Avg word length {awl:.1f} — too academic")
        sub_scores.append(0.5)

    # Jargon density scoring
    _, jd_hi = TARGET_JARGON_DENSITY
    if jd <= jd_hi:
        sub_scores.append(1.0)
    elif jd <= jd_hi * 2:  # up to 10% — warning zone
        notes.append(f"[WARN] Jargon density {jd:.1%} — moderately technical")
        sub_scores.append(0.5)
    else:
        notes.append(f"[FAIL] Jargon density {jd:.1%} — too technical for podcast")
        sub_scores.append(0.0)

    return sum(sub_scores) / len(sub_scores), notes, vocab_stats


# ── Composite scoring ────────────────────────────────────────────────────────

SCORE_WEIGHTS = {
    # Phase 2A — Structure (70%)
    "segment_count": 0.10,
    "turn_length": 0.15,
    "register_balance": 0.15,
    "speaker_balance": 0.15,
    "duration_estimate": 0.10,
    "connector_frequency": 0.05,
    # Phase 2B — Quality signals (30%)
    "hook_density": 0.15,
    "vocabulary_level": 0.15,
}


def evaluate_script(script: dict[str, Any]) -> ScriptReport:
    """Evaluate a script dict and return a ScriptReport."""
    report = ScriptReport()
    report.title = script.get("title", "")
    report.date = script.get("date", "")

    segments = script.get("segments", [])
    report.segment_count = len(segments)

    if not segments:
        report.notes.append("[FAIL] Empty script — no segments")
        report.overall_score = 0.0
        return report

    # ── Per-segment analysis ─────────────────────────────────────────────
    seg_stats: list[SegmentStats] = []
    words_list: list[int] = []
    register_counter: Counter[str] = Counter()
    speaker_words: Counter[str] = Counter()
    total_tags = 0
    reactive_count = 0

    for i, seg in enumerate(segments):
        text = seg.get("text", "")
        speaker = seg.get("speaker", "unknown")
        register = seg.get("register", "baseline")

        wc = _count_words(text)
        sc = _count_sentences(text)
        tc = _count_tags(text)

        seg_stats.append(
            SegmentStats(
                index=i,
                speaker=speaker,
                register=register,
                word_count=wc,
                sentence_count=sc,
                has_paralinguistic_tag=tc > 0,
                tag_count=tc,
            )
        )

        words_list.append(wc)
        register_counter[register] += 1
        speaker_words[speaker] += wc
        total_tags += tc
        if register == "reactive":
            reactive_count += 1

    report.words_per_segment = words_list
    report.total_words = sum(words_list)
    report.mean_words_per_segment = report.total_words / len(words_list)
    report.max_words_per_segment = max(words_list) if words_list else 0
    report.word_count_variance = _variance(words_list)
    report.paralinguistic_tag_count = total_tags
    report.reactive_segment_count = reactive_count
    report.connector_ratio = (total_tags + reactive_count) / len(segments) if segments else 0.0
    report.register_counts = dict(register_counter)
    report.register_ratios = {r: c / len(segments) for r, c in register_counter.items()}
    report.baseline_ratio = register_counter.get("baseline", 0) / len(segments)
    report.non_baseline_count = sum(
        1 for r, c in register_counter.items() if r != "baseline" and c > 0
    )
    report.words_by_speaker = dict(speaker_words)
    word_values = list(speaker_words.values())
    report.speaker_ratio = (
        min(word_values) / max(word_values)
        if len(word_values) >= 2 and max(word_values) > 0
        else 0.0
    )
    report.speaker_alternation_score = _compute_alternation(segments)
    report.estimated_duration_seconds = (report.total_words / WORDS_PER_MINUTE) * 60
    report.duration_vs_target = report.estimated_duration_seconds / TARGET_DURATION_SECONDS
    report.segment_stats = [asdict(s) for s in seg_stats]

    # ── Phase 2B: Vocabulary analysis ─────────────────────────────────────
    all_words: list[str] = []
    for seg in segments:
        all_words.extend(_get_words_lower(seg.get("text", "")))
    report.type_token_ratio = _compute_ttr(all_words)
    report.avg_word_length = _compute_avg_word_length(all_words)
    report.jargon_density = _compute_jargon_density(all_words)
    report.jargon_words_found = sorted(set(w for w in all_words if w in _JARGON_TERMS))

    # Gap hint coverage (inert — curator doesn't emit gap_hint yet)
    gap_hints = sum(1 for seg in segments if seg.get("gap_hint"))
    report.gap_hint_coverage = gap_hints / len(segments) if segments else 0.0

    # ── Score each dimension ─────────────────────────────────────────────
    scores: dict[str, float] = {}

    # Phase 2A — Structure
    s, n = _score_segment_count(report.segment_count)
    scores["segment_count"] = s
    report.notes.extend(n)

    s, n = _score_turn_length(words_list)
    scores["turn_length"] = s
    report.notes.extend(n)

    s, n = _score_register_balance(dict(register_counter), len(segments))
    scores["register_balance"] = s
    report.notes.extend(n)

    s, n = _score_speaker_balance(dict(speaker_words), segments)
    scores["speaker_balance"] = s
    report.notes.extend(n)

    s, n = _score_connector_frequency(total_tags, reactive_count, len(segments))
    scores["connector_frequency"] = s
    report.notes.extend(n)

    s, n = _score_duration_estimate(report.total_words)
    scores["duration_estimate"] = s
    report.notes.extend(n)

    # Phase 2B — Quality signals
    s, n, hook_stats = _score_hook_density(segments)
    scores["hook_density"] = s
    report.notes.extend(n)
    report.opening_question_count = hook_stats["opening_questions"]
    report.closing_question_count = hook_stats["closing_questions"]
    report.hook_topic_count = hook_stats["hook_topics"]
    report.hook_density_score = s

    s, n, vocab_stats = _score_vocabulary_level(all_words)
    scores["vocabulary_level"] = s
    report.notes.extend(n)

    # ── Weighted composite ───────────────────────────────────────────────
    report.overall_score = sum(scores[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
    report.dimension_scores = dict(scores)

    return report


def _variance(values: list[int]) -> float:
    """Compute population variance."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _format_report(report: ScriptReport) -> str:
    """Format a human-readable report with status annotations."""
    lines: list[str] = []
    lines.append(f"\n{'=' * 55}")
    lines.append(f"  Script Quality Report: {report.title}")
    lines.append(f"  Date: {report.date}")
    lines.append(f"{'=' * 55}\n")

    # Overview
    est_min = report.estimated_duration_seconds / 60
    lines.append(
        f"  Segments: {report.segment_count}    Words: {report.total_words}    Est. duration: {est_min:.1f}min"
    )
    lines.append("")

    # Dimension scores — re-score for display
    scores: dict[str, float] = {}
    s, _ = _score_segment_count(report.segment_count)
    scores["Segment count"] = s
    s, _ = _score_turn_length(report.words_per_segment)
    scores["Turn length"] = s
    s, _ = _score_register_balance(report.register_counts, report.segment_count)
    scores["Register balance"] = s
    s, _ = _score_speaker_balance(report.words_by_speaker, report.segment_stats)
    scores["Speaker balance"] = s
    s, _ = _score_connector_frequency(
        report.paralinguistic_tag_count, report.reactive_segment_count, report.segment_count
    )
    scores["Connector freq"] = s
    s, _ = _score_duration_estimate(report.total_words)
    scores["Duration est."] = s
    scores["Hook density"] = report.hook_density_score
    scores["Vocabulary"] = 0.0
    if report.total_words > 0:
        # Re-derive from stored metrics
        sub = []
        ttr_lo, ttr_hi = TARGET_TTR
        sub.append(1.0 if ttr_lo <= report.type_token_ratio <= ttr_hi else 0.5)
        awl_lo, awl_hi = TARGET_AVG_WORD_LEN
        sub.append(1.0 if awl_lo <= report.avg_word_length <= awl_hi else 0.5)
        _, jd_hi = TARGET_JARGON_DENSITY
        if report.jargon_density <= jd_hi:
            sub.append(1.0)
        elif report.jargon_density <= jd_hi * 2:
            sub.append(0.5)
        else:
            sub.append(0.0)
        scores["Vocabulary"] = sum(sub) / len(sub)

    lines.append("  Dimension          Score  Status")
    lines.append("  ─────────────────  ─────  ──────")
    for name, val in scores.items():
        status = "[OK]" if val >= 0.8 else "[WARN]" if val >= 0.5 else "[FAIL]"
        lines.append(f"  {name:<19} {val:5.2f}  {status}")

    lines.append("")
    lines.append(f"  {'─' * 35}")
    lines.append(f"  COMPOSITE SCORE:   {report.overall_score:.2f}")
    lines.append("")

    # Register breakdown
    lines.append("  Registers:")
    for reg in ["baseline", "emphasis", "reflective", "reactive"]:
        count = report.register_counts.get(reg, 0)
        ratio = report.register_ratios.get(reg, 0.0)
        lines.append(f"    {reg:<12} {count:3d}  ({ratio:.0%})")
    lines.append("")

    # Speaker breakdown
    lines.append("  Speakers:")
    for spk, wc in sorted(report.words_by_speaker.items()):
        pct = wc / report.total_words if report.total_words > 0 else 0.0
        lines.append(f"    {spk:<12} {wc:4d} words ({pct:.0%})")
    lines.append(f"    Alternation: {report.speaker_alternation_score:.2f}")
    lines.append("")

    # Hook density breakdown
    lines.append("  Hooks:")
    lines.append(f"    Opening questions: {report.opening_question_count}")
    lines.append(f"    Closing questions: {report.closing_question_count}")
    lines.append(f"    Topic labels:      {report.hook_topic_count}")
    lines.append("")

    # Vocabulary breakdown
    lines.append("  Vocabulary:")
    lines.append(f"    Type-token ratio:  {report.type_token_ratio:.2f}")
    lines.append(f"    Avg word length:   {report.avg_word_length:.1f}")
    lines.append(f"    Jargon density:    {report.jargon_density:.1%}")
    if report.jargon_words_found:
        lines.append(f"    Jargon terms:      {', '.join(report.jargon_words_found[:10])}")
    lines.append("")

    # Notes
    if report.notes:
        lines.append("  Notes:")
        for note in report.notes:
            lines.append(f"    {note}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Radio — script quality evaluation")
    parser.add_argument("script", type=Path, help="Path to episode script JSON")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if score below threshold",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Score threshold for --strict mode (default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--viz",
        type=Path,
        default=None,
        metavar="DIR",
        help="Generate script anatomy PNG in the given directory",
    )
    args = parser.parse_args()

    if not args.script.exists():
        print(f"ERROR: {args.script} not found", file=sys.stderr)
        sys.exit(1)

    script_data = json.loads(args.script.read_text())
    report = evaluate_script(script_data)

    if args.json:
        print(report.to_json())
    else:
        print(_format_report(report))

    if args.viz:
        from src.visualize import render_script_anatomy

        viz_path = render_script_anatomy(script_data, args.viz, score=report.overall_score)
        print(f"  Anatomy: {viz_path}", file=sys.stderr if args.json else sys.stdout)

    if args.strict and report.overall_score < args.threshold:
        sys.exit(1)


if __name__ == "__main__":
    main()
