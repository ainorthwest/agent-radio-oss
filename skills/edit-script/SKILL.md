---
name: edit-script
description: >
  Edit a rendered episode's script.json to fix bad segments, then
  re-render only the changed segments using the per-segment cache.
  Use when the anomaly detector flags a segment, when round-trip WER
  reports an outlier, or when an editorial review notices a misread.
---

# edit-script

The autonomous-station agent's correction loop. The anomaly detector
or the WER round-trip score points at a segment that didn't land; this
skill mutates the script and triggers a targeted re-render.

## When to use

- `radio edit anomalies <manifest>` flagged one or more segments
- The round-trip WER for a segment is above the episode's gate
- A human reviewer marked a segment for revision

## Decision shape

For each flagged segment, choose one operation:

| Anomaly | Operation | When |
|---|---|---|
| Silence / dropout | `radio edit script ... --delete N` then re-render | The segment came out empty; better to drop than retry forever |
| Mispronunciation | `radio edit script ... --replace N --text "..."` | Reword to spell-out tricky names; cache invalidates automatically |
| Wrong voice for content | `radio edit script ... --change-voice N --speaker host_b` | The host-of-record was wrong |
| Order is off | `radio edit script ... --reorder "0,2,1,3"` | Reordering doesn't invalidate the cache |

After mutation, `radio render episode <script>` re-renders. Unchanged
segments hit the per-segment cache (free); changed segments re-render
through Kokoro.

## What this skill is NOT for

- Wholesale script rewrite (use the curator instead)
- Mixing changes (handled by the mixer)
- Voice cast changes at the show level (edit `program.yaml`)

## Decision: ship / review / reject

After running this skill once, re-check anomalies:
- All clear → ship
- Still flagged → escalate to human review (don't loop more than 2x)
- Anomalies got worse → reject the episode and queue a new render

## Scripts

- `scripts/edit-segment.py` — thin wrapper around `radio edit script`
- `scripts/check-anomalies.py` — thin wrapper around `radio edit anomalies`
