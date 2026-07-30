# What "adjusted accuracy" actually is — and why it can't be recomputed from a formula

**Short version: Day 17's 57.2% figure came from a person looking at rows and
deciding if they were right. It was not the output of
`sum(tier_count × (1 - wrong_rate)) / total_rows`. Applying that formula to a
new run's tier tally will not reproduce it, and no number should be labeled
"adjusted" going forward unless it is backed by an actual review of that run's
own rows.**

## What went wrong when this was attempted on 28-07-2026

While updating CLAUDE.md's Paddle baseline, I tried to verify a supplied
"57.2% adjusted" figure by applying the documented per-tier wrong-rates
(Tier1-exact 0%, Tier1b-recovery 13.6%, Tier2-exact/merge ~8.7%,
Tier2-recovery ~20%, Tier4-confusion ~4.8%) to the 28-07 Paddle run's own tier
tally, using the formula CLAUDE.md itself describes:

> Compute it yourself: `sum(tier_count × (1 - wrong_rate)) / total_rows`.

That produced a **range, not a number** — 47.6% to 56.9%, depending on how the
~83-row `Tier3-fallback-novalid`/`other` slice (no documented wrong-rate) was
handled. None of the variants landed on 57.2%. That mismatch is the reason
this note exists: the formula and the historical figure are not the same
thing, and treating them as interchangeable was the actual error, not an
arithmetic mistake in either attempt.

## Why the formula can't reproduce a manual-review figure

The per-tier wrong-rates in CLAUDE.md (13.6%, ~8.7%, ~20%, ~4.8%) are
themselves each the output of a **manual ground-truth review of a specific
set of rows on a specific day** — e.g. the Tier1b figure came from reviewing
that tier's actual passes and checking each one against the true container
ID. They are measured constants for *that reviewed population*, not universal
per-tier constants that hold for any future run.

Two things break the moment you carry them into a new run's tally:

1. **Tier composition shifts run to run even at matched headline accuracy.**
   The whole point of today's finding that 515/841 reproduced exactly across
   the 17-07 and 28-07 Paddle runs is that headline accuracy is stable — it
   says nothing about whether the *same 515 rows* landed in the *same tiers*
   both times, or whether a different run with a different checkpoint/model
   would split its passes across tiers the same way. A wrong-rate measured
   against one run's Tier1b population does not transfer to another run's
   Tier1b population if that population is drawn from different images,
   different failure modes, or a different mix of easy/hard sources.

2. **Some tiers have no measured wrong-rate at all**, and this is not an
   oversight — it's the current state of the project. `Tier3-fallback-novalid`
   is explicitly flagged elsewhere as unmeasured for TrOCR (the reasoning:
   TrOCR's Tier 3 risk mechanism differs from Paddle's, since there's no
   cross-box splicing when the recognizer returns only one line). Paddle's
   Tier 3 wrong-rate isn't independently established for the current,
   post-Pass1-removal code either. Any formula-based "adjusted" number is
   silently assuming a value for this slice — excluding it, treating it as
   perfect, or treating it as worthless are all different assumptions, and
   the three choices produce three different answers (this is exactly why my
   attempt returned a range instead of a number).

## What a current adjusted figure actually requires

A defensible adjusted accuracy for either of today's two runs
(`benchmark_results_trocr_cascade_FULL_thin_20260728.csv`,
`benchmark_results_paddle_cascade_FULL_20260728.csv`) requires **a manual
ground-truth review of that run's own valid rows** — the same process that
produced the 13.6%/8.7%/20%/4.8% figures in the first place, applied to
today's rows, not a reuse of those older numbers.

**This has not been done for either run.** Task 2 of this session (the
52-row TrOCR Tier 3 review) is a step toward exactly that, for TrOCR's Tier 3
slice specifically. It is not itself a full adjusted-accuracy computation —
it only covers Tier 3, and Tier1b/Tier2/Tier4 in the current runs would each
need their own review before a complete adjusted figure could be stated for
either recognizer.

## Standing rule going forward

- **Headline accuracy** (rows/rows, all valid rows counted) can always be
  quoted directly from a CSV — it needs no review.
- **"Adjusted accuracy"** is not a formula output. Do not compute one by
  reapplying old per-tier wrong-rates to a new run's tier tally, and do not
  quote a specific adjusted percentage for a run that hasn't had its own
  rows reviewed.
- If an adjusted figure is needed for a specific run, say so explicitly and
  scope the review (which tiers, how many rows) before attempting a number.

## Distinct finding, 28-07-2026: tier breakdowns are unreliable under
## multi-rotation crops — both live and offline capture

While building the Tier 3 ground-truth review gallery for
`benchmark_results_trocr_cascade_FULL_thin_20260728.csv`, re-deriving the
true originating tier for the 52 rows tagged `Tier3-fallback-novalid` +
`is_valid=True` found that **only 3 of the 52 are genuinely Tier 3.** 32 are
actually Tier 1b (single-line check-digit recovery) and 17 are actually
Tier 1 (single-line exact). All 52 reproduce exactly (0/52 mismatches)
against the original CSV once reproduced correctly, so this is not a
reproducibility question — the rows are real, the *label* is wrong.

**Root cause:** `app/inference/ocr.py`'s `_best_portrait_rotation` tries both
rotations of a portrait crop and keeps the winning result, but the "Tier "
log line is captured at module scope without being scoped per-attempt — so
whichever rotation attempt logs *last* overwrites the tier label, even when
an *earlier* rotation attempt is the one whose result is actually returned.
This is the same bug CLAUDE.md already documents for the benchmark's own
capture mechanism ("Tier labels in benchmark output are approximate ...
records the last Tier log line across all rotation attempts, not necessarily
the one that produced the returned result").

**What's new here:** that bug isn't confined to the benchmark harness. Any
*offline* re-derivation of tier that re-invokes the same
`_best_portrait_rotation` code path — as this session's own
`tier3_review_generate.py` initially did before being corrected to capture
per-attempt and identify the winning call — inherits the identical failure
mode unless it explicitly disaggregates by attempt (crop, result) pair, the
way the corrected version of that script does. There is no capture path,
live or offline, that gets tier-attribution right "for free" — every one of
them needs the same per-attempt-then-match-the-winner correction applied in
this session.

**What is and isn't affected:**
- **`is_valid` and headline accuracy are unaffected.** The bug is purely in
  which tier label gets attached to an already-correct pass/fail result — it
  never changes whether a row is valid.
- **Any tier-level breakdown is unreliable** for both recognizers' 28-07 runs
  (`benchmark_results_trocr_cascade_FULL_thin_20260728.csv` and
  `benchmark_results_paddle_cascade_FULL_20260728.csv`) — the TrOCR case is
  now measured (94% of the "Tier 3" bucket was mislabeled in this one
  52-row sample), and the same mechanism applies structurally to the Paddle
  run's tier column too, though Paddle's has not been independently
  re-derived. Don't treat either CSV's `tier` column as ground truth without
  the same per-attempt correction.
- Rows that only ever take **one** rotation attempt (`n_rotation_calls == 1`,
  i.e. landscape crops, no portrait retry) are not exposed to this bug —
  there's only one attempt to mislabel from. The risk is specific to
  portrait crops that trigger `_best_portrait_rotation`'s two-attempt path.

**Paddle's 28-07 tier table was deliberately not re-derived.** The recognizer
decision (TrOCR ships) doesn't depend on tier-level figures for either
recognizer — headline accuracy, which is unaffected by this bug, was the
deciding number. Re-deriving Paddle's tier breakdown the same way would cost
real time for a correction that wouldn't change any decision already made,
so it's being skipped intentionally, not overlooked.
