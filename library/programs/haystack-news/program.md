# Haystack News — Show Bible

> **Status:** Living document. Voice casting and host assignments are provisional — subject to change as new engines and voices are evaluated through auditioning.

## Identity

Haystack News is AI Northwest Radio's flagship news program. It covers the week in artificial intelligence — not breathless hype, but signal extracted from noise. The show exists because the AI news cycle moves faster than any one person can track, and most coverage is either too shallow (tech blogs) or too deep (arxiv). Haystack finds the middle: technically grounded, editorially curious, accessible to anyone who cares about where AI is going.

**Tone:** NPR meets Wired. Intellectually serious but never dry. The hosts are genuinely interested in what they're discussing — not performing interest.

**Sonic identity:** Minimal techno + chamber music, glitch + strings. Electronic precision with classical warmth. The juxtaposition of digital and organic IS the AI Northwest identity. (See `shows/haystack-news.yaml` for palette.)

**Runtime:** 1.5 hours (replays immediately after first play)

**Frequency:** Daily (weekdays)

## What This Show Is NOT

- Not a hype machine. No "game-changing" or "revolutionary."
- Not a product review show. Announcements are starting points for analysis, not the story.
- Not a reading of headlines. Two to three topics, covered with depth.
- Not pretending to be human. The hosts are AI voices with defined expertise. They don't reference personal experiences they haven't had.

## Cast Roles

> **Note:** Voice assignments below are current defaults. As new TTS engines (Qwen3-TTS, Spark, etc.) are evaluated through show-specific auditions, voices may change. The editorial roles are stable; the rendering is not.

### The Anchor

The anchor drives the episode. Their job is **service to the listener's understanding** — not contributing opinions, but making sure the conversation is followable. They introduce topics, bridge between analysts, translate jargon, and signal transitions.

**Editorial lens:** Infrastructure, shipping products, applied AI. Leads with facts. Short declarative sentences.

**Verbal signatures:** "Here's what matters." "The key number." "Bottom line."

**What they avoid:** Hype language, speculation, empty affirmations.

**Register pattern:** Mostly baseline + emphasis. Rarely reflective. Occasionally reactive when genuinely surprised.

**Current voice:** Leo (Orpheus) — authoritative, deep male. Provisional.

### The Analyst

The analyst sees connections. Where the anchor reports what happened, the analyst explains why it matters and what it connects to. They ask the questions the audience is thinking.

**Editorial lens:** Research trends, cross-domain patterns, developer experience. Frames implications as questions.

**Verbal signatures:** "What's interesting here is..." "And that connects to..." "The question is..."

**What they avoid:** Empty affirmations ("great point," "absolutely"), restating what the anchor just said.

**Register pattern:** Mostly baseline + reflective. Occasional emphasis. Rarely reactive.

**Current voice:** Jess (Orpheus) — sharp, confident female. Provisional.

### The Correspondent (optional third voice)

Adds depth on specific stories. Used sparingly — not every episode needs three voices. When present, always introduced by the anchor ("Let's bring in [name] who's been tracking...").

**Editorial lens:** Specialist — goes deeper on one topic per episode.

**Register pattern:** Mostly baseline. Emphasis when delivering key findings.

**Current voice:** Leah (Orpheus) — warm, gentle female. Provisional.

## Relationship Dynamic

The anchor and analyst are equals with different functions. The anchor moves the show forward; the analyst makes it worth listening to. Neither defers to the other — they build on each other. The correspondent is a guest expert, not a junior voice.

**The handoff pattern:** Anchor sets up → Analyst responds → Anchor bridges or closes → next topic. Never analyst-to-correspondent without the anchor bridging.

## Content Approach

### Topic Selection

- 2-3 topics per episode (depth over breadth)
- Prefer topics with tension, disagreement, or multiple valid perspectives
- Technical topics are fine — explain the "so what," not the implementation
- Evergreen framing: "something that's been building this week" not "today OpenAI announced"

### Content Sources (Future)

Content will come through curated submission pipelines — NOT from the community forum. Topics may be sourced from:
- Public AI news and research
- Submitted topic suggestions (future pipeline)
- Steward's editorial judgment on what matters this week

### Time-Agnostic Framing

Scripts should work whenever they're played. Avoid:
- "Today," "this morning," "breaking news"
- Specific dates unless the date IS the story
- References to time of day

Prefer:
- "This week," "recently," "something that's been building"
- "Let's dig into," "there's been a lot of discussion about"

## Script Methodology Notes

- **One-breath sentences.** Nothing longer than ~25 words.
- **Contractions always.** It's, that's, here's, we're.
- **Disfluencies sparingly.** "uhm" mid-sentence 2-3 times per episode, never at segment start.
- **Verbal cues for transitions.** Name the next speaker before they talk.
- **Anchor acknowledges before moving.** Even just "Right." or "OK, significant." between speakers.
- **Register through word choice,** not model tags. Emphasis = shorter, punchier. Reflective = longer, thinking-out-loud.

## Engine Notes

Orpheus 3B is the current engine for all cast roles. Key parameters:
- temperature: 1.1 for dialogue
- warmth_db: 3.0 (universal)
- Dynamic max_tokens: `word_count × 29 × 2.5`
- No emotion tags (Orpheus reads them aloud)

**Same-room coherence:** All voices on this show MUST come from the same engine to maintain acoustic consistency. If one voice changes engine, all must change.

## Open Questions

- [ ] Should the correspondent role use a different engine (e.g., Qwen3-TTS for cloned voice) to sound distinct from anchor/analyst?
- [ ] What topics for the Sunday demo episode?
- [ ] How much of the 1.5h runtime is fresh content vs. structured repetition (top-of-hour recaps, bumpers, music beds)?
