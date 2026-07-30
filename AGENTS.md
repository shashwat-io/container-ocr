# AGENTS.md — Container OCR Project Context

## Project overview
Flask web app (multi-user login, image upload, container code detection + OCR) 
built as a 6-week internship project. Stack: Flask + SQLAlchemy + Flask-Login + 
flask-bcrypt + Flask-Migrate + Gunicorn + plain HTML/CSS/vanilla JS + SQLite 
(dev) / PostgreSQL (prod). Four DB tables: users, uploads, detections, audit_log.

Detection: YOLOv8s (best.pt in models/). OCR: PaddleX pipeline in 
app/inference/ocr.py. ISO 6346 validation (4 letters + 6 digits + 1 check 
digit) with check-digit math.

## Active directory
/home/shashwatkumar/Testing/container_ocr_test/

## Key files
- app/inference/ocr.py — the entire OCR pipeline (crop, preprocess, PaddleX, 
  tiered extraction, Tier 4 confusion correction, ISO 6346 validation)
- benchmark.py — runs full 786-image test, writes benchmark_results.csv
- benchmark_results.csv — current live results
- incomplete_samples/ — 8 sample full+crop image pairs of incomplete-read 
  failures for visual diagnosis
- scratch_docpp_eval.py — docpp net-impact evaluation script
- scripts/replay_tier_attribution.py — offline, read-only replay of tier-matching 
  logic against already-captured ocr_raw CSV text (no live OCR). Use this for any 
  future need to get accurate tier-by-tier breakdowns; benchmark_full_live.py's 
  live tier-logging is known unreliable (captures whichever of up to 4 per-image 
  _try_extract calls logged last, not necessarily the winner — does not affect 
  is_valid pass/fail totals, only the tier label)
- quick_sample_set.txt — fixed, reusable, stratified 157-image sample (seed=42, 
  ~20% of the 786 test set, proportional across index blocks). Use for fast 
  directional checks (~30-40 min) instead of the full 786-image benchmark 
  (~2.5-2.65 hrs) when validating small changes. Full 786 remains required 
  for any number actually reported/committed to.

## Current OCR accuracy (authoritative — see Changelog for how we got here)
571 / 786 images (72.6%) with ≥1 valid detection, or 580/841 rows (69.0%), 
from a full, uninterrupted 786-image benchmark validated on 02-07-2026.
Do not quote "73.8%" — that figure mixed row-count and image-count 
denominators from an earlier, incorrect report. Earlier numbers (62.6%, 
610/786, 542/786 projection) are stale/superseded — see Changelog.

## OCR pipeline — tier logic
_try_extract() runs four tiers:
- Tier 1: single-line exact 11-char match + validate
- Tier 1b: single-line exact 10-char match + compute+append check digit
- Tier 2: adjacent line pair, same two checks (STRICT len==10 exact match 
  only, no sliding window — fixed to prevent false positives)
- Tier 3: original merged-string sliding window fallback (Pass 1: 11-char; 
  Pass 2: 10-char check-digit recovery — DISABLED, see Known Tier Logic 
  Defects below)
- Tier 4: single-position character-confusion correction on best_code 
  (only if exactly 1 type-violation, lookup in _CHAR_CONFUSION_MAP, 
  full validate_iso6346 required to pass)

_CHAR_CONFUSION_MAP: O↔0, I↔1, S↔5, B↔8, Z↔2, G↔6, C→6 (asymmetric, 
no reverse 6→C).

Recovered codes tagged "Valid (check-digit recovered)" or 
"Valid (character confusion corrected: X→Y at position N)" in reason string.

is_recovery_based(reason) — pure helper, single source of truth for whether 
a result came from check-digit recovery (Tier1b or Tier2-recovery paths). 
Threaded through extract_container_code()'s result dict, both 
_real_predict/_stub_predict in app/inference/predictor.py (defaults False 
in stub mode), surfaced as an amber "Recovery-based match — please verify" 
badge in static/js/inference.js (_renderImageCard). Transient (JSON response 
only) — not persisted to the Detection model or DB.

PaddleX pipeline init:
paddlex.create_pipeline('OCR', device='cpu', 
  engine_config={'paddle_static': {'enable_new_ir': False, 
  'run_mode': 'paddle'}})
Threading lock (_ocr_lock) wraps ALL predict() calls — PaddleX is NOT 
thread-safe.

## Known tier logic defects — NEW
**Tier 2/3 merge (pairing) logic discards spatial position entirely.** 
Adjacent-line pairing currently uses OCR return order as an adjacency 
proxy instead of actual bounding-box coordinates. Confidence scores ARE 
computed at this stage but are NOT used in merge decisions. This produced 
the SEKU6384890 failure — three different fabricated codes assembled from 
fragments of a single container's crop, because return-order adjacency 
incorrectly paired unrelated line fragments. Absolute confidence filtering 
was tested and ruled out as a fix: the wrong merge used maximum-confidence 
(1.00) fragments on both sides, so a confidence floor would not have caught 
it. **Spatial position (actual bounding-box coordinates of each line 
fragment) is the next lever to try, not confidence-based filtering.**

**Check-digit recovery ≠ correctness verification.** Any tier that computes 
an ISO 6346 check digit from a partial/uncertain OCR reading and accepts it 
purely because the math is internally consistent is NOT verifying the 
reading is correct — only that it's self-consistent. Manual image-by-image 
verification found Tier3-Pass2 specifically wrong ~81% of the time (17/21 
samples checked), because it searches a sliding 10-char window across 
CONCATENATED multi-line OCR text with no per-character confidence signal 
to gate on. This tier's recovery is now disabled (backup at 
ocr.py.pre_tier3fix.bak). Tier1b's recovery is lower-risk (~13.6% wrong, 
45-sample manual review) and left active — OCR confidence does NOT predict 
correctness for this tier (wrong-answer mean confidence 0.877 vs. 
correct-answer mean 0.885 — statistically indistinguishable); do not 
attempt confidence-based filtering here without new evidence.

**First out-of-test-split accuracy check (08-07-2026): confirms known risks,
surfaces one update.** Ran extract_container_code() directly against 569
manually-labeled (human-verified) rows from train/valid via a Kaggle CPU-only
test session (no YOLO involved — used stored box coords from
manual_labels.csv). This is the first time pipeline output has been checked
against ground truth outside the 786-image test benchmark.

Results: 343/569 exact match (60.3%), 379/569 pipeline-valid (66.6%),
190/569 honest failures (33.4%, mostly "too short" and "first 4 characters
must be letters" -- 87% of honest failures, consistent with a box-tightness
problem rather than an OCR-quality problem specifically).

Confidently wrong (is_valid=True but exact_match != ground truth): 36 rows,
22 distinct crops after accounting for a since-fixed labeling-tool
duplication bug (see Changelog). Breakdown by resolving tier:
- check-digit recovered: 14/22 (64%)
- multi-line merge: 4/22 (18%)
- plain Tier1-exact ("Valid", no recovery/merge flag): 2/22 (9%)
- character confusion corrected: 2/22 (9%)

This confirms the existing risk model rather than contradicting it --
check-digit recovery dominates the wrong-answer bucket exactly as the
13.6%/81% wrong-rate findings above would predict. Two things are updates
to existing claims, not just confirmation:

1. Plain Tier1-exact produced 2 wrong answers here, against the standing
   claim (0/44 wrong, test-split manual review) that it's "the only
   trustworthy auto-tier." Still a low rate (2 wrong / ~158 plain-Valid
   results in this run), but the "0 wrong" claim should now be read as
   "0 wrong in the test-split sample checked so far," not as an
   established universal property of Tier1-exact.

2. The SEKU6384890 fabricated-code merge failure (documented above under
   spatial-position-discarded merge bug) reproduced identically on a
   different data split -- two rows in this test
   (4654256381207060402_0320 and _0260, both predicting CKMS6848397
   against ground truth SEKU6384890). This is the same failure recurring,
   not a new one, but confirms it's a repeatable pattern worth the
   spatial-position fix, not a one-off artifact.

60.3% vs. the 69.0%/72.6% test-split benchmark is a real, converged gap
(stable from n=400 onward, not sampling noise) -- current leading
hypotheses (not yet confirmed): train/valid images may be harder on
average than the curated test split, and/or this is the first check
against genuinely-verified ground truth rather than self-consistent
(checksum-passing) ground truth. Not yet root-caused.

## Hard rules — always follow these
1. NEVER disable docpp globally (confirmed net loss — see Model/Training notes)
2. ALWAYS clear app/inference/__pycache__ before running benchmark.py 
   after any ocr.py change
3. NEVER start benchmark.py while another benchmark is running — PaddleX 
   CPU contention + CSV collision risk
4. ALWAYS restart server after ocr.py changes and confirm clean startup 
   (no import errors, 302 on /)
5. NEVER run anything that invokes PaddleX while benchmark.py is running 
   in another shell — diagnostic scripts that stub run_ocr_on_crop are safe, 
   real OCR calls are not
6. benchmark.py writes its CSV in ONE SHOT at the very end — do NOT read 
   benchmark_results.csv mid-run to check progress, it shows the PREVIOUS 
   run's data. Only the printed [N/786] checkpoint lines are live
7. Use python3 not python. Use flush=True + PYTHONUNBUFFERED=1 on scripts
8. Do NOT change UI/styling during debugging sessions
9. Do NOT produce progress reports or handoff notes mid-session — only at 
   end of day or when explicitly asked
10. No git repo — all verification via .bak files + manual diff or 
    reconstruction. Back up BEFORE any edit you'll want to verify later, 
    rather than reconstructing from memory after the fact.

## Known PaddleX/PaddleOCR traps
- PaddleX NOT thread-safe: always acquire _ocr_lock before predict()
- PaddleOCR v3 API: use paddlex.create_pipeline('OCR',...) NOT PaddleOCR()
- show_log parameter removed in v3, use .predict() not .ocr()
- PP-OCRv6_medium is the top tier — no higher pretrained model exists in 
  PaddleX. Don't suggest upgrading the model tier. PP-OCRv5 was tested in 
  isolation (50-crop sample, ground-truth boxes) and ruled out — 6% vs. 30% 
  checksum-valid, systematic type-code bleed-through failure mode. Do not 
  re-attempt without a materially different successor model.
- PaddleOCR is ~99.2% deterministic on CPU overall, not 100% — but 
  significantly less deterministic on hard-to-read crops specifically: 
  on the population of crops that fail the primary OCR pass and need a 
  docpp-off retry, run-to-run raw text agreement was only ~24% in a live 
  same-day test (vs. ~0.8% general non-determinism). Don't trust a single 
  benchmark run's exact number for this subset without caveat, and don't 
  assume a "fix" that changes this subset's results by some delta is fully 
  attributable to the fix — some of any delta may just be OCR noise. Run 
  twice and check for consistency before concluding, if precision matters.
- limit_type='min' means limit_side_len sets MINIMUM side — 64 does NOT 
  downscale our 200-400px crops. Raising it upscales. Don't treat lsl=64 
  as a downscaling problem (already ruled out)
- text_det_box_thresh=0.6 EXONERATED — lowering it changed nothing on 
  failing crops. Not the cause of blank reads.

## ISO 6346 VALUE_MAP — use this exactly, do not regenerate
A=10,B=12,C=13,D=14,E=15,F=16,G=17,H=18,I=19,J=20,
K=21,L=23,M=24,N=25,O=26,P=27,Q=28,R=29,S=30,T=31,
U=32,V=34,W=35,X=36,Y=37,Z=38, digits 0-9 map to themselves.
Letters skip multiples of 11. Hardcoded in _ISO6346_VALUE_MAP.

## Diagnostic / benchmark patterns
**Dry-run / diagnostic pattern (use this before any full benchmark):**
Parse ocr_raw column from CSV back into (text, score) tuples.
Monkeypatch run_ocr_on_crop to return them.
Call _try_extract(None) — tests all tier logic in seconds, no PaddleX.
Always check for regressions on currently-valid rows, not just gains on 
invalid ones. A net improvement requires: gains > regressions.

**Self-verifying benchmark pattern (always use for full runs):**
Before running benchmark.py after any ocr.py change:
1. rm -rf app/inference/__pycache__
2. Have the benchmark script print ocr.py mtime, PID, start timestamp, 
   and grep its own loaded source for a known string unique to the new 
   code — print "FRESH CODE CONFIRMED" or abort if stale
3. Check ps aux | grep benchmark.py — confirm exactly one process

## Model / training (separate from Flask app)
YOLOv8s best: mAP50=0.995, mAP50-95=0.855 (100-epoch run). Saved as 
yolov8s_100ep_CONFIRMED_GOOD on Drive. YOLOv8m 100-epoch run also 
completed: mAP50=0.995, mAP50-95=0.860, saved as 
container-ocr/models/yolov8m_100ep_CONFIRMED_GOOD/. Training runs in 
Google Colab / Kaggle with Drive at /content/drive/MyDrive/container-ocr/.

Do NOT resume training with optimizer='auto' on converged weights — use 
SGD + conservative lr0.

**Kaggle / checkpoint gotchas — NEW**
- Kaggle P100 GPUs have shown compatibility issues with this training 
  setup — switch to T4x2 if a P100 session fails to start or errors early.
- /kaggle/working/ is session-scoped scratch — if the session dies 
  (timeout, disconnect, crash), anything only saved there is unrecoverable. 
  Save weights to Drive promptly during long runs, not just at the end.
- resume=True only reconstructs training while the SAME kernel/session is 
  still alive. A keyboard interrupt that kills the kernel (not just pauses 
  it) requires a fresh restart — the checkpoint cannot be resumed even 
  though weight files exist on disk. BUT: not every interruption kills the 
  kernel. Before assuming a checkpoint is unresumable, check whether the 
  kernel actually died (session still shows connected? does nvidia-smi 
  still respond?) rather than defaulting to a fresh restart.
- If args.yaml for a saved run shows the run as already complete (epoch 
  count matches the configured total), resume=True will NOT continue — 
  Ultralytics treats it as finished and may silently fall back to training 
  on its own bundled default dataset instead of erroring. Always check 
  args.yaml's epoch count against what you actually intended before 
  trusting resume=True.

## Data & labeling files — NEW
**manual_labels.csv**
- Location: container-ocr/training_labels/manual_labels.csv on Drive.
- Columns: image_path, box_index, x_center, y_center, width, height, 
  label_text, checksum_valid, split, status, timestamp.
- status ∈ {labeled, skip, notacode, clipped}.
- clipped = code genuinely cut off by the box edge (not just hard to 
  read) — evidence for the box-tightness problem specifically. Only 
  visibly-confirmed characters should be recorded; anything only readable 
  by inference/guessing should NOT be typed as a confident character.
- Populated by the ipywidgets-based labeling tool 
  (container-ocr/notebooks/Labelling.ipynb) — resumable across runtime 
  restarts, skips already-labeled (image_path, box_index) pairs on every 
  fresh run.

**review_sheet.csv — PENDING, not yet confirmed**
Columns (from the Tier1-exact/Tier2-exact manual review sheets): 
sample_number, filename, claimed_code, ocr_raw, ocr_confidence, 
validation_reason, verified_correct, saved_crop_image(s), box_ambiguous, 
box_count_mismatch, recorded_num_boxes, recomputed_num_boxes, box_idx, 
image_index, block. For rows where box_ambiguous=True, saved_crop_image(s) 
holds TWO comma-separated filenames in one cell (e.g. 
"sample17_a.jpg, sample17_b.jpg") rather than one — a naive single-value 
parse of that column will break on those rows. Flagging this as the likely 
"parsing gotcha" based on what's recorded elsewhere, but this wasn't 
pinned down with certainty — confirm before treating it as fact.

## Changelog

**08-07-2026:** Found and fixed a labeling-tool duplication bug: Labelling.ipynb's
candidate-sampling cell (Cell 4) called random.seed(RANDOM_SEED) with a fixed
seed (42) before shuffling the remaining-boxes pool on every run. Since
`remaining` shrinks between re-runs as more boxes get labeled, the same fixed
seed applied to a different-length list produced batches that heavily
overlapped previous batches, causing already-labeled crops to be silently
re-offered and re-labeled as if new. Root-caused via Claude Code timestamp
analysis: duplicate groups clustered in ~10-11-row bursts at regular ~6-8 min
intervals within single sessions (227 pure-duplicate groups where repeated
labels agreed, 4 genuine disagreement groups where they didn't), not a
cross-session Drive-sync issue as first suspected. Fix: removed the fixed
seed (plain random.shuffle() with no seed argument). Net effect on
manual_labels.csv: 1290 raw rows -> 377 real distinct labels after dedup
(891 pure-duplicate rows dropped, 4 disagreement groups manually resolved
by re-checking the actual crop image). Old 1290-row file preserved as
manual_labels_OLD_1290_20260708.csv for audit purposes; corrected 377-row
file is now the live Drive copy.

**01-07-2026 EOD:** Benchmark was stale/incomplete — last officially 
completed number was 610/786 (77.6%), but that was BEFORE that day's 
Tier3-Pass2 fix; the fix was live in code but the confirming 786-image 
benchmark was killed mid-run (~100/786) due to forced shutdown. 610/786 
and the dry-run projection of 542/786 were both stale/unconfirmed at that 
point. ocr.py had a live, unbenchmarked fix: Tier3-Pass2 check-digit 
recovery disabled. Backup at ocr.py.pre_tier3fix.bak. Diff scope manually 
confirmed clean (only the Pass-2 block + related comments changed).

**02-07-2026:** Full 786-image benchmark validated clean (786/786, 
uninterrupted) — this is the current authoritative number (see top of 
file). _try_extract_best_valid()'s docpp-off-retry success path used to 
overwrite the tier-specific reason string with a generic "Valid 
(docpp-off retry)" tag, silently losing recovery-tier information for 
~23% of results — fixed to append ", check-digit recovered" instead of 
discarding it; does not change retry firing conditions or which result 
wins. Confirmed current OCR pipeline is PP-OCRv6_medium specifically (not 
a generic "PaddleX pipeline" label).

**End of day 9 (30-06-2026), for historical reference only — SUPERSEDED 
by 02-07-2026 above:** 492/786 (62.6%), baseline 450/786 (57.3%), +42 from 
Tier 4 character-confusion correction. Remaining failure breakdown at that 
point (349 invalid rows): incomplete/missing data 153 (43.8%, dominant 
bucket), genuine garbled/multi-char misreads 99 (28.4%), separated 
contamination 61 (17.5%), merged contamination 36 (10.3%). The docpp 
double-rotation bug was found this day: PaddleX's internal doc 
preprocessor was auto-rotating already-upright crops, fighting 
_best_portrait_rotation. Disabling docpp globally recovers ~30 of 153 
incomplete-read failures BUT causes ~21% regression on currently-valid 
crops (~95 projected regressions) — net loss, do not do this (see Hard 
Rule #1). Fix implemented as two-pass "best valid wins": run with docpp ON 
first, only retry with docpp OFF if invalid, accept whichever is valid.

## ADDITION 1 — append to the Changelog section

**07-07-2026:** Ran a fresh (non-original) 8-image hard-case comparison
between YOLOv8s (deployed) and YOLOv8m — original Phase 1 six-image
hard-case set/script unrecoverable, so this is a directionally-informative
but not identical replication, explicitly labeled as such. Result: m-model
tighter on 6/8 images, mean IoU 0.961 vs 0.920, mean area ratio
(predicted/GT) 1.014 vs 1.030 (both oversized on average, m less so), zero
detection regressions either direction. Comparison crops saved to
hard_case_comparison_07-07/. Real signal that the m-model produces tighter
boxes, not just a better aggregate mAP — but n=8 and not the original set,
so treat as supporting evidence for a switch decision, not conclusive proof.


## ADDITION 2 — new entry, place near "Known tier logic defects"

**Known trap: Tier1-exact is NOT a safe source of "free" fine-tuning labels as-is.**
benchmark_full_live.py (and therefore every Tier1-exact/Tier1b/etc. row in
benchmark_results.csv) only ever reads from container.v2i.yolov11/test/images
— never train/valid. This means ALL Tier1-exact rows are test-split images.
Using them as fine-tuning training labels would contaminate the held-out
786-image benchmark that every reported accuracy number depends on — the
model would be partially trained on the same images used to score it.
Confirmed 07-07-2026: attempted to bootstrap free labels from the 280
Tier1-exact rows in the benchmark, found 100% were test-split, stopped
before generating any output file. If free/bootstrapped labels are wanted
again, the correct approach is to run the OCR pipeline directly against
train/valid images (not extract from the existing benchmark), so the
resulting labels never touch the test set. This has NOT been done yet —
a full train/valid run was estimated at ~18-21 hrs, extrapolated from the
known ~2.5-2.65 hr / 786-image benchmark rate (train+valid ≈ 7-8x the size
of test). A smaller sample is the intended approach instead; sample size
not yet decided.



# Known Findings
Model size does not affect box tightness (09-07-2026): Quantitative comparison (IoU + area ratio vs ground truth, 30 valid images) across YOLOv8s/m/l showed all three statistically indistinguishable on box tightness (IoU 0.924–0.925, area ratio 1.007–1.022). The Day-11 mentor observation about loose boxes is not resolved by model capacity — likely driven by per-image clutter rather than architecture. Recommendation: carry m forward, not l, since l's only advantage is a marginal mAP50-95 gain (0.868 vs 0.860) at real training/inference cost. Scripts: box_tightness_compare_smld.py, zoom_bsiu_anomaly.py (not yet committed to repo).

