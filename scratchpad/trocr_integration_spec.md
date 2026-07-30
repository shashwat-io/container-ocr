# TrOCR production-integration spec

**Status:** scoping only. Nothing in `app/` has been modified. Written 28-07-2026
while the full 786-image TrOCR benchmark (PID 7528) was mid-run.

**Scope:** what it would take for `app/inference/ocr.py` to reproduce the exact
path that `benchmark_trocr_cascade_quicksample.py` measures, so the benchmark's
accuracy numbers transfer to production rather than merely resembling it.

Everything below was read off the source. Points I could not verify from code
alone are marked **[UNCERTAIN]** rather than smoothed over.

---

## Correction to the framing of this task

The task brief says the harness "monkeypatches out PaddleOCR's doc-preprocessing
stack (UVDoc, doc_ori, textline_ori, PP-OCRv6 det+rec)" and that this is a
divergence from production. That is true as a *consequence*, but the mechanism is
narrower and more favourable than it sounds, and the distinction drives the whole
rest of this spec:

**The harness calls production's own `extract_container_code()` unmodified.** It
replaces exactly one module-level name — `ocr.run_ocr_on_crop` — and nothing
else. Every other production function (`crop_detection`, `_post_process`,
`preprocess_crop`, `_best_portrait_rotation`, `_try_extract_best_valid`,
`_try_extract`, all tier logic, `validate_iso6346`, the `is_*` helpers) executes
byte-identically to production.

So the PaddleX doc-preprocessing stack disappears **only because it lives
entirely inside `run_ocr_on_crop`'s `ocr.predict()` call**, which is the thing
being replaced. There is no second, separate divergence to reconcile. The
integration problem is therefore much smaller than "reimplement the harness in
production" — it is "give `run_ocr_on_crop` a second backend."

---

## (a) Exactly what the harness monkeypatches

One assignment, at `benchmark_trocr_cascade_quicksample.py:99`:

```python
def _run_trocr_as_ocr_line(crop_image, docpp_off=False):
    text = run_trocr_on_crop(crop_image)
    if not text.strip():
        return []
    return [(text, 1.0)]      # placeholder confidence

ocr.run_ocr_on_crop = _run_trocr_as_ocr_line
```

Because `_try_extract` (`ocr.py:226`) resolves `run_ocr_on_crop` through the
module global at call time, patching the module attribute redirects every call
site in the graph:

```
extract_container_code                       (ocr.py:504)  UNMODIFIED
  └─ _best_portrait_rotation                 (ocr.py:476)  UNMODIFIED
       └─ _try_extract_best_valid            (ocr.py:444)  UNMODIFIED
            └─ _try_extract                  (ocr.py:210)  UNMODIFIED
                 └─ run_ocr_on_crop          (ocr.py:165)  ←── REPLACED
                      └─ get_ocr() / paddlex pipeline      ←── never reached
```

**What is bypassed by that single replacement:**

- the entire PaddleX `OCR` pipeline object (`get_ocr()`, `ocr.py:45`) — never
  instantiated, so `PP-LCNet_x1_0_doc_ori`, `UVDoc`, `PP-LCNet_x1_0_textline_ori`,
  `PP-OCRv6_medium_det`, `PP-OCRv6_medium_rec` are never loaded.
  *Verified empirically:* zero `Creating model:` lines in any TrOCR run log,
  versus five in every PaddleOCR run log.
- `_ocr_lock` (`ocr.py:42`) — the Paddle-predictor serialisation lock is never
  taken. See (d)/§thread-safety: TrOCR needs its own, and `ocr_trocr.py` has one.
- the `docpp_off=True` code path (`ocr.py:176`) — the replacement accepts the
  kwarg and **ignores it**.
- Paddle's per-line text detection, i.e. multiple `(text, score)` tuples per
  crop. TrOCR returns **exactly 0 or 1** tuples. This single fact drives all of
  section (c).

**What is NOT bypassed:** our own preprocessing. `_post_process` (upscale to
min-dimension 300 via LANCZOS, contrast ×1.5, SHARPEN, grayscale→RGB) runs on
every crop in both paths, because it is applied in `_best_portrait_rotation` /
`extract_container_code` *before* the recogniser is called. Same for
`crop_detection`'s 10px padding.

---

## (b) Harness per-image path vs production per-image path

| Stage | Harness | Production (`predictor.py` → `ocr.py`) | Diverges? |
|---|---|---|---|
| YOLO load | `YOLO(models/best_resplit.pt)` | `YOLO(models/best_resplit.pt)` default (`predictor.py:32`) | **No** — same default weights |
| Detection | `model(img_np, verbose=False)`, iterate `r.boxes.xyxy` | same call shape, plus reads `box.conf` for display | **No** (functionally) |
| Per-box entry | `extract_container_code(image_bytes, x1,y1,x2,y2)` | `extract_container_code(image_bytes, x1,y1,x2,y2)` | **No** — identical entry point |
| Crop | `crop_detection`, padding=10 | same | **No** |
| Orientation branch | `h > w` → portrait | same | **No** |
| Preprocess | `_post_process` (resize/contrast/sharpen/gray) | same | **No** |
| **Recognise** | TrOCR: 1 string, conf hardcoded `1.0` | PaddleX pipeline: N lines, real `rec_scores`, plus doc-ori/unwarp/textline-ori | **YES — the only real divergence** |
| docpp-off retry | runs, but recogniser ignores flag → byte-identical rerun | genuinely re-runs Paddle with doc-preproc disabled | **YES (degenerate, see below)** |
| Rotation retries | `_best_portrait_rotation`, ±90°, real rank+conf tiebreak | same code | **No code diff; different behaviour** (see below) |
| Tier cascade | `_try_extract` Tiers 1/1b/2/3/4 unmodified | same | **No code diff; tiers 2 dead** (see (c)) |
| Validation | `validate_iso6346` | same | **No** |
| Result dict | same keys consumed | `predictor.py:88-98` reads the same keys | **No** |

### Two behavioural divergences that are not code divergences

**1. The docpp-off retry becomes a pure no-op that doubles TrOCR calls.**

`_try_extract_best_valid` (`ocr.py:454-457`) calls `_try_extract` a second time
with `docpp_off=True` whenever the first pass is invalid. Under TrOCR the
replacement ignores `docpp_off`, so the retry reproduces pass 1 exactly and can
never change the outcome. *Verified empirically:* the `docpp-off-retry` tier
label appears in every PaddleOCR run's tier breakdown and in **none** of the
TrOCR runs.

Cost, per detection box, when the code does not validate:
- landscape: 2 TrOCR generate calls instead of 1
- portrait: **4** instead of 2 (2 rotations × 2 passes)

This is dead compute already baked into the 7.2 s/img benchmark figure. Removing
it is the one change that makes production *faster* than the benchmark while
producing identical output.

**2. `_best_portrait_rotation`'s confidence tiebreak degenerates.**

`avg_conf` (`ocr.py:231`) becomes exactly `1.0` (non-empty) or `0.0` (empty,
early-returned at `ocr.py:228`) — nothing else is reachable. So the tiebreak at
`ocr.py:498` (`conf_90 >= conf_m90`) is `1.0 >= 1.0` → **always true** whenever
both rotations produced any text. The rank ladder above it (`ocr.py:491-492`)
still does real work, but once ranks tie, rotation selection collapses to
"**always prefer +90**" rather than anything confidence-driven.

This is not a bug and it is faithfully reflected in the benchmark numbers — but
it means `ocr_confidence` under TrOCR is **not a confidence** and must not be
displayed or thresholded as one. *Checked:* no code in `app/` compares
`ocr_confidence` against any threshold; `predictor.py:95` only passes it through
for display. So this is a UI-honesty issue, not a control-flow risk, **except**
inside that one tiebreak.

---

## (c) Tier-by-tier assessment under TrOCR

The governing fact: `_run_trocr_as_ocr_line` returns a list of length **0 or 1**,
never more.

| Tier | Fate under TrOCR | Why |
|---|---|---|
| **Tier 1** (single-line exact 11-char) | **Fires, dominant** | One string, cleaned; if exactly 11 chars and format-valid it validates. Carries the whole result set — 92–97 of ~166 rows on quick sample. Wrong-rate 0%. |
| **Tier 1b** (10-char check-digit recovery) | **Fires** | Same single line at length 10. Real, but inherits the measured **13.6% wrong rate** — and under TrOCR it is a *larger share* of passes than under Paddle, so the blended error rate shifts. |
| **Tier 2** (adjacent line pairs — exact, one-char-removal, check-digit) | **STRUCTURALLY DEAD — cannot ever fire** | `for i in range(len(texts) - 1)` (`ocr.py:280`) with `len(texts) ≤ 1` gives `range(0)` or `range(-1)`, both empty. *Verified empirically:* `Tier2-exact` / `Tier2-recovery` appear in every PaddleOCR run and in **zero** TrOCR runs. |
| ↳ ambiguity machinery | **Inert** | `ambiguous_merge_found` stays `False` and `ambiguous_candidates` stays empty, so `_tag_ambiguous` is an identity function, the Tier-3 ambiguity exclusion at `ocr.py:397` never excludes anything, and the defensive re-check at `ocr.py:406-412` is unreachable. `is_ambiguous_merge()` will always return `False` → the "ambiguous merge" UI state becomes permanently dead. |
| ↳ `is_merge_based()` | **Always False** | Only Tier 2 emits "multi-line merge". So `predictor.py:97`'s `is_recovery_based` flag narrows to *only* Tier1b — a **semantic change to what that flag means** downstream, even though no code changed. Worth flagging to whoever owns the review UI. |
| **Tier 3** (merged fallback) | **Fires, and can still return `is_valid=True`** | With Pass 1 and Pass 2 removed, the surviving scan at `ocr.py:393-405` still picks the first structurally-valid 11-char window and still runs `validate_iso6346` on it — so a checksum-passing window is still accepted, just labelled `Tier3-fallback-novalid`. This is CLAUDE.md's "removal is near-cosmetic" note, and it holds under TrOCR too. Appears in all TrOCR runs. |
| **Tier 4** (single-char confusion correction) | **Structurally live, empirically never succeeds** | Recogniser-agnostic: operates on `best_code`, not on line structure. But `Tier4-confusion` appears in **zero** TrOCR runs' pass counts. **[UNCERTAIN]** — `tier_pass_counts` only counts *valid* results, so this shows Tier 4 never successfully corrected, not that it never executed. Plausible reading: TrOCR's errors are whole-token, not single-confusable-character, so the "exactly one format violation" precondition rarely holds. Not verified directly. |

### The one tier whose risk profile genuinely changes

**Tier 3 is safer under TrOCR than under PaddleOCR, and the existing wrong-rate
evidence does not transfer.**

The documented case against Tier 3's passes (Pass 2 at 81% wrong, Pass 1 at 2
confirmed-wrong / 2 suspicious / 0 confirmed-correct) rests on a specific
mechanism: the sliding window ran over text **concatenated across multiple
detection boxes**, so it could splice fragments of unrelated lines into a
coincidentally checksum-valid string, with no per-line confidence to gate on.

Under TrOCR there is only ever one string from one crop, so **cross-box splicing
cannot occur**. The window slides within a single recogniser output. That is a
materially different — and lower — risk.

**[UNCERTAIN, and this is the most important uncertainty in this document]** I
have no measured wrong-rate for Tier 3 under TrOCR. The Paddle-era figures are
not applicable in either direction: they are not a valid upper bound (different
mechanism) *or* a valid reassurance. `Tier3-fallback-novalid` is running at
roughly 8–14 rows per 166 on quick sample, so it is a small but non-trivial slice
of passes. **Any adjusted-accuracy figure quoted for TrOCR must either exclude
Tier 3 or state that its wrong-rate is unmeasured.** A manual ground-truth review
of those rows, on the same pattern as the 27-07 Pass 1 review, is the missing
evidence — and it is cheap, since there are only a few dozen rows in the full run.

---

## (d) Minimal edits to make production match the benchmarked path

Listed, not applied. Ordered by necessity. Items 1–4 are required for parity;
5–6 are required for safety; 7 is optional.

**In `app/inference/ocr.py`:**

1. **Add a recogniser selector and dispatch inside `run_ocr_on_crop`.** This is
   the single edit that carries the whole integration.
   ```python
   OCR_RECOGNIZER = os.environ.get('OCR_RECOGNIZER', 'paddle')   # 'paddle' | 'trocr'
   ```
   At the top of `run_ocr_on_crop` (`ocr.py:165`), when `OCR_RECOGNIZER == 'trocr'`,
   delegate to a TrOCR adapter and return; otherwise fall through to the existing
   Paddle body completely unchanged. Default `'paddle'` keeps current production
   behaviour byte-identical when the var is unset — same safety pattern as the
   existing `FINETUNED_REC_MODEL_DIR`.

2. **The adapter must reproduce `_run_trocr_as_ocr_line` exactly**, including the
   two details that look incidental but are load-bearing:
   - `return []` when `text.strip()` is empty (drives `'No text detected'` at
     `ocr.py:228`, and the `0.0` conf that makes the rotation tiebreak work);
   - `return [(text, 1.0)]` otherwise — the literal `1.0`.
   Any other confidence value changes `_best_portrait_rotation`'s tiebreak and
   silently de-calibrates the benchmark. Import `run_trocr_on_crop` from
   `ocr_trocr` **lazily inside the function**, not at module top — `ocr_trocr`
   imports `clean_ocr_text`/`validate_iso6346` from `ocr`, so a top-level import
   creates a circular import.

3. **Skip the docpp-off retry when the recogniser is TrOCR.** In
   `_try_extract_best_valid` (`ocr.py:457`), guard the second `_try_extract`
   call. Provably outcome-neutral (the flag is ignored), and it removes 50% of
   TrOCR generate calls on non-validating crops. **This is the only edit that
   changes measured behaviour vs. the benchmark — and it changes only latency,
   never output.**

4. **Do not touch the tier cascade.** Tier 2's dead loop, the inert ambiguity
   machinery, and Tier 3's surviving accept-path must all be left exactly as they
   are. They are what the benchmark measured. Deleting "dead" Tier 2 code would
   also break the Paddle path, which still needs it.

**In `app/inference/ocr_trocr.py`** (outside the letter of "edits to ocr.py", but
required for the above to work — flagging rather than omitting):

5. **Make `FINETUNED_TROCR_MODEL_DIR` read lazily inside `get_trocr()`** instead
   of at module import time (`ocr_trocr.py:44`). See (f) — as written, the module
   silently loads the **zero-shot base model** if it is imported before the env
   var is set. In a Flask app the import order is not under the operator's
   control the way it is in a script. This is the highest-severity footgun here:
   the failure is silent and produces plausible-looking wrong output.

6. **Add a fail-fast startup guard** mirroring the harness's own
   (`benchmark_trocr_cascade_quicksample.py:76-82`): if TrOCR is selected and the
   checkpoint dir does not exist, raise. Production currently has no equivalent
   and would serve zero-shot TrOCR indefinitely without a single error log line.

**Optional (7):** surface `ocr_confidence` as `None`/"n/a" rather than `1.0` when
the recogniser is TrOCR, so the UI does not present a placeholder as a real
score. No code depends on the value, so this is presentation-only.

### Not required, and deliberately excluded

- Thread-safety needs no new work: `ocr_trocr.get_trocr()` already has its own
  `_trocr_lock` for lazy init. **[UNCERTAIN]** — that lock guards *initialisation*
  only; `run_trocr_on_crop` calls `model.generate()` **unlocked**. Under Flask's
  threaded server, concurrent requests would run concurrent generates on one
  shared `VisionEncoderDecoderModel`. PyTorch eval-mode forward passes are
  generally re-entrant, so this is *probably* fine — unlike Paddle's static
  predictor, which genuinely is not. I have not verified it under load, and the
  benchmark is strictly single-threaded so it provides **no evidence either way**.
  Worth a concurrency smoke test before any deployment.

---

## (e) Benchmark-only scaffolding that must NOT reach production

1. **The monkeypatch itself.** `ocr.run_ocr_on_crop = _run_trocr_as_ocr_line`
   mutates a module global at import time. Fine for a single-purpose script;
   in an app it is action-at-a-distance that any later import can undo or
   double-apply. Replace with the explicit dispatch of (d)(1).
2. **`ocr_logger.propagate = False`** (harness line ~120). This detaches
   `app.inference.ocr` from the root logger so tier lines only reach the capture
   handler. Carried into production it would **silently delete all OCR logging**.
   Highest-risk line in the file to copy by accident.
3. **`TierCaptureHandler`, `_last_tier_msg`, `TIER_LABELS`, `classify_tier`.**
   Benchmark attribution only. Note the known defect: it records the *last*
   `Tier ` log line across all rotation attempts, not necessarily the one that
   produced the returned result — so tier labels are approximate by construction.
   Do not build production logic on this mechanism.
4. **Hardcoded `sys.path.insert('/home/shashwatkumar/Testing/container_ocr_test')`
   and `os.chdir(...)`** (harness lines 73-74).
5. **Placeholder confidence `1.0`** — must be carried into the *adapter* for
   parity (d)(2), but must not be presented to users as a confidence (d)(7).
   These two requirements are in tension; resolve at the display layer, not the
   recogniser.
6. **CSV checkpointing, `stats` accumulation, `CHECKPOINT_EVERY`,
   `BENCH_*` env vars.**
7. **The "QUICK SAMPLE" results banner**, which prints on full runs too —
   cosmetic mislabel, already documented in CLAUDE.md.

---

## (f) Checkpoint-layout and env-var dependencies production must handle

**1. Two env var names for one thing.** The harness reads `TROCR_CHECKPOINT_DIR`
and forwards it into `FINETUNED_TROCR_MODEL_DIR`, which is what `ocr_trocr`
actually reads (harness lines 63-64). Note also that neither is
`FINETUNED_REC_MODEL_DIR` — that is the *PaddleOCR* override and pointing it at a
TrOCR checkpoint would fail, since it expects PaddleOCR inference format
(`.pdmodel`/`.pdiparams`). Production should settle on one name and document it.

**2. Import-order coupling — the silent-fallback footgun.** `ocr_trocr.py:44`
reads the env var into a module constant **at import time**. The harness handles
this by setting `os.environ[...]` before any import (lines 63-64, above even
`import csv`). A Flask app cannot rely on that ordering. If the module is
imported first, `FINETUNED_TROCR_MODEL_DIR` is `''`, `model_source` falls back to
`TROCR_MODEL_ID`, and production serves the **zero-shot base model** with no
error and no log line distinguishing it. Fix per (d)(5).

**3. The checkpoint dir is only half the model — and the other half comes from
the network.** `get_trocr()` (`ocr_trocr.py:69-70`) builds the tokenizer and image
processor from the **hub ID** `microsoft/trocr-base-printed`, deliberately (see
the fast-tokenizer comment at `ocr_trocr.py:56-66`), and takes *only the weights*
from the checkpoint dir. Consequences:

   - The checkpoint dir needs `config.json` + `model.safetensors`. The
     `preprocessor_config.json` / `tokenizer_config.json` / `tokenizer.json` files
     present in `~/Downloads/trocr_checkpoint_thin` are **ignored by design**.
   - **Cold start requires either network reachability or a warm HF cache.**
     This is not theoretical: it fired this morning. The first launch of the
     786-image run failed outright with
     `OSError: Can't load image processor for 'microsoft/trocr-base-printed'`
     because the machine had booted ~5 minutes earlier and DNS was not yet up.
     Worse, transformers' normal offline-cache fallback did **not** engage — httpx
     raised `Cannot send a request, as the client has been closed` rather than a
     clean connection error.
   - **Production implication:** a deployment box that reboots, or has no
     outbound internet, will fail to start the recogniser — or, if the failure
     mode differs slightly, could fall back to something unintended. Mitigation is
     either `HF_HUB_OFFLINE=1` with a pre-warmed cache, or vendoring the processor
     config into the checkpoint dir and loading it from there. **[UNCERTAIN]** —
     I have not tested whether `HF_HUB_OFFLINE=1` alone is sufficient with the
     current cache state; the cache contains a `.no_exist` negative marker for
     `processor_config.json`, which offline mode *should* honour, but that is
     inference from the cache layout, not a verified run. Verifying it needs a
     process launch, which is out of scope while the benchmark holds the CPU.

**4. Model size / memory.** ~1.3 GB of weights per loaded checkpoint, held
resident in the `_trocr_instance` singleton, versus PaddleOCR's much smaller CNN
stack. On a 15 GB box this is fine for one worker; it is a real constraint if the
app is ever run multi-process (e.g. gunicorn with N workers each loading its own
copy). **[UNCERTAIN]** — I did not check the app's deployment/worker
configuration.

---

## Summary of what actually blocks integration

Ranked by how much they can hurt:

1. **Tier 3's wrong-rate under TrOCR is unmeasured**, and the Paddle-era evidence
   does not transfer in either direction. This is an accuracy-reporting blocker,
   not a code blocker. Cheap to fix with a manual review of a few dozen rows.
2. **The silent zero-shot fallback** ((f)(2)) — wrong output, no error.
3. **The HF-hub dependency at cold start** ((f)(3)) — already caused one real
   failure today.
4. **Tier 2 dead ⇒ `is_recovery_based` silently narrows in meaning** for the
   review UI, even though no code changed.
5. Everything else is mechanical.

Latency is **not** on this list. The 27-07 measurements put the TrOCR cascade at
~7.2 s/img end-to-end versus ~10.8 s/img for the PaddleOCR cascade on the same
CPU box, and edit (d)(3) would improve the TrOCR figure further. The long-standing
assumption in CLAUDE.md that CPU latency is the blocking risk for TrOCR is not
supported by the data.
