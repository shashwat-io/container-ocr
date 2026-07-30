# Note: today's targeted 131-row spot check (separate from tomorrow's full 786-run)

**This is NOT the official benchmark result.** See `STATUS_tier3fix_pending_benchmark.md`
for the actual pending task (full 786-image run, still needs to happen with a full
time block). This note is scratch findings from a smaller, time-boxed, deliberately
biased sample (all 71 former Tier3-Pass2 rows + 60 random control rows) run this
evening. Do not treat any number below as the real accuracy figure.

## What ran
`scratch_targeted_benchmark.py` — real OCR (not CSV replay) on 131 rows:
71 former Tier3-Pass2 rows (100% of that population, not a sample) + 60 random
control rows (seed 42) from the rest of the dataset. Output in
`scratch_targeted_benchmark_results.csv`.

## Control group: clean
60/60 rows exactly matched the baseline (`benchmark_results.csv`) — same code,
same valid/invalid status. Confirms the Tier3-Pass2 fix's scope is correctly
isolated; nothing outside Tier3-Pass2 changed behavior.

## Tier3-Pass2 group: diverged sharply from the Step 3 dry-run projection
Step 3's CSV-replay sanity check (using frozen raw OCR text) projected 68 invalid
+ 3 rescued-with-same-code. The live re-run instead found:
- **30 invalid**
- **41 valid** — but only **12 have the same code as before** (matching the
  Step 3 rescue pattern); **29 have a completely different, unverified code**.

## Why the divergence: PaddleOCR is highly non-deterministic on THIS specific population
Checked raw OCR text byte-for-byte against baseline: only 17/71 (24%) matched
exactly. This is far above the ~0.8% general dataset non-determinism rate noted
in AGENTS.md. Breaking down by prior docpp origin:
- Rows where docpp-ON primary alone was previously sufficient (n=55): 71% had
  different raw text on re-run.
- Rows where docpp-OFF retry was already being used before (n=16): 94% had
  different raw text on re-run — even higher, meaning this isn't just the
  two-pass wrapper exercising a new fallback path for the first time, it's
  that PaddleOCR itself is highly unstable on this specific hardest-crop
  population regardless of docpp setting.

**This is not a bug in the Tier3-Pass2 fix** (diff was scope-clean, control
group is 100% stable) — it's a pre-existing property of OCR behavior on the
hardest crops in the dataset, only visible here because this sample
specifically re-ran ALL 71 of those crops live rather than replaying frozen
text.

## Practical implication for tomorrow's real run
Because these 71 crops are this volatile, tomorrow's full-benchmark outcome for
them is not reliably predictable from either the Step 3 projection (~542 total)
or today's live snapshot (~539 non-Tier3-Pass2 + 41 today's-tier3pass2-valid ≈
580 total, extrapolating the control group's 0% change rate to the other 770
rows). Expect the real number to land somewhere in a wide band around there —
possibly outside it — because re-running these same 71 crops again tomorrow
could reshuffle the invalid/valid split yet again. The 29 different-code
rescues found today are unverified; treat any of them with the same caution as
the original Tier3-Pass2 concern until manually checked.
