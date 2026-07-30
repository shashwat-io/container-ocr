# CLAUDE.md — Current State (read this first)

This file is the **current-state source of truth**. `AGENTS.md` is a valuable
historical changelog (root-cause chains, bug archaeology, rationale) — keep
reading it for *why* something is the way it is, but if it and this file ever
disagree on what's true **right now**, this file wins. Update this file
whenever current state changes; leave AGENTS.md's history alone.

Last updated: 28-07-2026 (Day 24).

---

## Environment facts (confirmed, stop re-checking these)

### Local Ubuntu machine (`vehant`)

- Project root: `~/Testing/container_ocr_test/`
- **No git repo** (confirmed 17-07-2026, re-confirmed 27-07-2026). All
  verification is via `.bak` files, manual diff, or fresh re-derivation.
  Don't attempt `git log`/`git blame`/`git status` — they will fail.
- **No rclone, gdown, or Drive mount.** The only Drive access is the Drive
  MCP connector, which uploads/downloads files individually via API — not a
  directory-sync tool, and impractical for files above a few MB (base64
  round-trip through a text channel). For anything larger (checkpoint
  exports, zipped datasets), the person downloads via browser and transfers
  manually — ask for the local path once placed, don't try to pull it
  yourself.
- The Drive MCP connector **cannot delete files**. Any stale/duplicate file
  found on Drive needs the person to delete it manually in the Drive UI.
- 8 CPU cores, ~15 GB RAM. All cascade benchmarks run here, on CPU.
- Two work environments: **office laptop** (has repo + Claude Code access)
  and **personal laptop** (remote days — GitLab and Claude Code access
  restricted). Confirm which one is active if it matters for a task.

### GPU server (`dobby` / `10.77.14.47`) — added 27-07-2026

- Path: `/mnt/Data/container-ocrpoc/test/`
- GPU: **NVIDIA RTX 2000 Ada, 16 GB VRAM.** Runs 24×7. A single TrOCR
  fine-tune uses ~8.1 GB VRAM at 100% GPU utilisation and hits the 70 W
  power cap — so a second concurrent training run would time-slice, not
  parallelise. Run one at a time.
- **Claude Code has NO SSH access to this server and must never be given or
  asked for credentials.** This is a deliberate access-control decision.
  Anything that must run server-side gets handed to the person as
  copy-paste-ready shell commands for their own SSH session. Claude Code is
  local-machine only (file prep, transfers, local benchmarking, repo edits).
- `eval-venv` contains **torch 2.13.0 and transformers 5.14.1 only** — no
  ultralytics, no PaddleOCR, no opencv. The cascade benchmark **cannot run
  on the server** without installing those first (PaddleX installs are
  historically finicky; weigh the risk before attempting).
- transformers 5.14.1 dated 21-07-2026, unchanged across the 27-07 reboot —
  library drift is *not* a live confound, stop re-testing this.

### Server gotchas (all cost real time on 27-07-2026)

- NFS home `/nfs/interns/shashwatkumar` **is not mounted**. Every login
  prints `Could not chdir to home directory ...`. Harmless in itself, but it
  pollutes the stream `scp -O` uses for its protocol handshake and makes
  transfers fail with a misleading `Is a directory` error, once per file.
  **Use `rsync -avz` for all transfers to/from this server, not `scp`.**
- Consequence of the same missing mount: `~/.bash_history` doesn't persist,
  and the HF cache resolves into a nonexistent path, so base model weights
  re-download on every training run. Wasteful, not incorrect.
- **Image zips carry an `images/` prefix internally.** Unzip with `-d .`
  into the data directory, **never `-d images/`** — the latter produces
  `images/images/...` and the training script fails on the first batch with
  `FileNotFoundError`. This bit two runs on 27-07-2026.
- Colab remains the platform for PaddleOCR fine-tuning (moved off Kaggle
  17-07-2026 after three Kaggle-specific data-loss incidents in two days).

---

## Current best models

### YOLO detector — `models/best_resplit.pt` (confirmed 17-07-2026)

Saved to Drive as `container-ocr/models/yolov8s_resplit_full_CONFIRMED_GOOD/`.
Trained on the leakage-clean, video-level resplit
(`container.v2i.yolov11.resplit/`, 0 group overlap across train/valid/test,
1565 distinct groups via `group_id()` in
`scripts/paddle_finetune/video_safety.py`). mAP50-95 = 0.813.

Detection is stable and not currently a bottleneck: every one of the seven
cascade benchmarks run on 27-07-2026 detected **155/157 images (98.7%)**,
regardless of recognizer checkpoint.

**The older `models/best.pt` (in `yolov8s_100ep_CONFIRMED_GOOD/`) was trained
on a contaminated split (97.4% train/test near-duplicate overlap) — its
historical accuracy numbers are inflated by leakage. Keep it only for
explicit before/after comparisons, never as a "good" reference on its own.**

YOLOv8s, YOLOv8m and YOLOv8l are statistically indistinguishable on box
tightness (IoU 0.924–0.925). YOLOv8s is the carry-forward model.

### PaddleOCR recognizer

**The production app and all cascade benchmarks to date have used stock
zero-shot PaddleOCR**, never a fine-tuned checkpoint. `app/inference/ocr.py`'s
`get_ocr()` calls stock `pdx.create_pipeline('OCR', ...)`.

⚠️ **69.0% headline / 64.7% adjusted is RETIRED — do not use for comparison.**
That figure was measured on the OLD detector (`best.pt`) against the
ORIGINAL CONTAMINATED test split (97.4% train/test near-duplicate overlap).
It is not a valid baseline for anything measured against the leakage-clean
resplit split.

**Current baseline, clean resplit test set, `models/best_resplit.pt`, current
code (post-Tier3-Pass1-removal): 61.2% headline (rows/rows).**
Confirmed by TWO independent runs, 11 days apart, matching to the row —
**515/841 valid** both times:
- `benchmark_results_full_resplit_CONFIRMED_GOOD.csv`, 17-07-2026
- `benchmark_results_paddle_cascade_FULL_20260728.csv`, 28-07-2026 (run after
  the 27-07-2026 Tier3-Pass1 removal)

This is strong evidence the Tier3-Pass1 removal is **cosmetic at the headline
level, confirmed** — headline accuracy is unchanged across the removal, exactly
as the fallback-scan argument elsewhere in this file predicted.

An **adjusted** figure (discounting Tier1b/Tier2/Tier4 passes by their
documented per-tier wrong-rates) was attempted against the 28-07 run and did
**not** cleanly reproduce a single number — it lands in a **47.6%–56.9%**
range depending on how the ~83-row `Tier3-fallback-novalid`/`other` slice
(no documented wrong-rate — see the TrOCR section below, same open problem)
is treated: excluded from the numerator only, or from both numerator and
denominator. **Do not cite a single "adjusted" percentage for this baseline
without first fixing that methodology choice explicitly.**

- A `FINETUNED_REC_MODEL_DIR` env-var override was added to `get_ocr()` on
  17-07-2026 (empty by default = stock behaviour, verified byte-identical
  when unset). This expects **PaddleOCR inference format**
  (`.pdmodel`/`.pdiparams`) — do **not** point it at a HuggingFace/TrOCR
  checkpoint directory, they are unrelated formats.
- Clean 150-epoch Colab retrain (17-07-2026) produced epoch 128, isolated
  val exact-match 24.33%, norm_edit_dis 0.5078. Isolated-crop eval only —
  not comparable to cascade numbers.
- Day 17's fine-tuned-cascade benchmark found the fine-tuned checkpoint
  **underperformed stock** in the full pipeline.

### TrOCR recognizer — never integrated into production

TrOCR has only ever been evaluated through the standalone benchmark script.
It has **never been wired into `app/inference/ocr.py`** and its **CPU
inference latency has never been measured**. It is a ~334M-parameter
transformer where PaddleOCR is a small CNN; latency on the deployment box is
an unproven, potentially blocking risk regardless of accuracy.

---

## TrOCR results as of 27-07-2026 EOD

**No full 786-image benchmark exists for any TrOCR checkpoint on current
code.** Everything below is the 157-image quick sample (rows/rows).

| Checkpoint | Crops | Groups | Mean/group | Valid | Tier1-exact |
|---|---|---|---|---|---|
| 3896crop (Friday, replayed today) | 3,896 | 1,190 | 3.27 | **71.1%** | **97** |
| thin cap=3 | 3,301 | 1,225 | 2.69 | 69.9% | 92 |
| thin cap=2 | 2,265 | 1,225 | 1.85 | 67.5% | 91 |
| 4700_midpoint | 4,700 | 961 | 4.89 | 61.4% | 48 |
| 5695crop run 1 | 5,695 | 1,225 | 4.65 | 60.2% | 72 |
| 5695crop run 2 | 5,695 | 1,225 | 4.65 | 55.4% | 74 |
| 3936_v2 | 3,936 | 754 | 5.22 | 49.4% | 47 |

Isolated val-exact, **fixed metric only** (see the `-100` bug below — earlier
numbers are not comparable): thin cap=3 70.78% · thin cap=2 70.15% ·
4700 55.39% · 5695 53.75% · 3936_v2 42.62%.

### Reliability caveats on the above

- **Training variance is ~±5 points.** The two 5695 runs used *identical
  data and seed* and landed 4.8 points apart on headline (60.2% vs 55.4%).
  Tier1-exact is the stabler signal — those same two runs differed by only
  2 rows on it.
- At n=166 rows the standard error is ~±3.6 points. **The quick sample
  cannot resolve differences smaller than ~5 points.** Don't run experiments
  whose expected effect is 1–2 points and expect to read the result.
- Friday's 3896crop replay reproduced **bit-for-bit** (118/166 both times,
  zero differing predictions), confirming the benchmark pipeline is
  deterministic on CPU and that checkpoint transfers are intact.

### Local checkpoint locations

`~/Downloads/trocr_final_checkpoint_3896crop` (Friday's, current best) ·
`trocr_final_checkpoint_5695crop` · `trocr_final_checkpoint_5695crop_run2` ·
`trocr_checkpoint_3936_v2` · `trocr_checkpoint_4700_midpoint` ·
`trocr_checkpoint_thin` · `trocr_checkpoint_thin_cap2`

---

## Key findings from 27-07-2026 (Day 23)

### The `-100` metric bug — corrupted checkpoint *selection*, not just scores

`compute_metrics` in `trocr_finetune.py` replaced the `-100` padding sentinel
in `labels_ids` before decoding but **not** in `pred_ids`. When `-100`
reached the tokenizer it threw `OverflowError` on newer tokenizer backends;
when it didn't crash, it silently produced garbage decodes and deflated the
epoch's score.

The consequential part: the script sets
`load_best_model_at_end=True` with `metric_for_best_model="exact_match"`.
**The corrupted metric was therefore driving which epoch's weights got
saved.** Any checkpoint trained before 27-07-2026 is a differently-selected
artifact and is not strictly comparable to later ones.

Fixed by inserting, before the prediction decode:

```python
pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
```

Whether the bug fires at all is data-dependent — Friday's run hit the same
latent bug and never crashed. Absence of a crash is not evidence the metric
was correct.

### Diversity finding — group coverage AND crops-per-group both matter

Recognizer quality tracks **source-video diversity**, not crop count. Both
dimensions do independent work:

- **3936_v2 fails on coverage** — 3,936 crops but only 754 groups → 49.4%.
- **5695 fails on density** — full 1,225 groups but 4.65 crops each → 55–60%.
- **Winners sit at ~1,200 groups and ~2.7–3.3 crops per group** → 69–71%.

The cleanest single comparison: thin cap=3 and 5695 have **identical group
coverage (1,225)** and come from the same pool. The 2,394 extra redundant
crops in 5695 cost ~18–20 clean Tier1-exact reads.

Practical consequence: the final labeling push added 1,799 crops but only
**35 new groups** — almost all of it deepened already-covered videos, which
is why it didn't help.

Density has been bracketed: 1.85 → 67.5%, 2.69 → 69.9%, 3.27 → 71.1%,
4.65 → 55–60%. The optimum is somewhere around 3–4 crops/group. Differences
within that range are inside noise at n=166 and **cannot be resolved by more
quick-sample runs**.

Untested lever if this line of work resumes: selecting *visually dissimilar*
crops within each group (different frame indices/rotations) rather than
random ones — attacks redundancy directly instead of by proxy.

### Epoch count is settled at 15

Both thin runs plateau by epoch ~10 and stay flat through 15. Train loss
reaches ~1e-04 (training set fully extracted) and the LR schedule decays to
~8e-08 by the end, so additional epochs on the current schedule would barely
update weights. Not a lever.

---

## `manual_labels.csv` / labeling pool status (confirmed 27-07-2026)

- **Labeling is COMPLETE.** All safe crops in the remaining pool are labeled.
  No open labeling work remains in the project.
- **Current authoritative row count: 6,282 rows**
  (`training_labels/manual_labels.csv`). Status breakdown:
  **6,197 labeled / 47 clipped / 27 skip / 11 confirmed_asis.**
  `checksum_valid`: 6,199 True / 73 empty / 10 False.
  `split` column: 4,535 train / 1,011 valid / 736 test.
- **7 known pre-existing duplicate rows remain** — agree-with-self, harmless,
  left untouched intentionally, don't "fix" these again. They are keyed on
  **`(image_path basename, box_index)`**; `file_name`/`crop_filename` is
  unique. Any script joining through `manifest.csv` on the composite key must
  consume matches in order rather than using a plain dict lookup, or it will
  silently collapse those pairs.
- ⚠️ **The CSV's own `split` column is not authoritative for leakage
  safety.** `extract_crops.py` applies a stricter, video-level filter
  (`EXCLUDE_WHOLE_SOURCE_VIDEO`, using `group_id()` against the resplit test
  set) on top of it — a row marked `split=train` is still dropped if its
  source video has any frames in the resplit test split. Trust `group_id()`,
  not the column.
- **Standing reminder: never re-run `Labelling.ipynb` Cell 5 alone** — always
  Cell 3 → Cell 4b → Cell 5 together if resetting mid-session. The
  documented reset risk has fired for real once (22-07-2026, 120 excess rows).

### The 587-row gap (reconciled, not a bug)

`manual_labels.csv` has 6,282 rows; `train.jsonl` + `val.jsonl` for the 5695
set have 5,695. The 587-row difference is fully accounted for:
**512 excluded by the video-level leakage filter** + 47 clipped + 27 skip +
1 confirmed_asis. Verified programmatically (512/512 matched). Working as
designed — don't re-investigate.

Open question, **not acted on**: `EXCLUDE_WHOLE_SOURCE_VIDEO` was chosen when
the pool was ~377 rows. At 6,282 rows it now costs 512 genuinely-valid
labeled rows (8.9%). Worth revisiting only if TrOCR becomes the long-term
production path.

---

## Data pipeline (3 steps, not one script)

1. `scripts/paddle_finetune/extract_crops.py` — reads `manual_labels.csv`,
   applies `is_label_usable()` + the `group_id()` leakage filter, writes
   `data/paddle_finetune/manifest.csv`
2. `scripts/paddle_finetune/generate_labels.py` — reads `manifest.csv`,
   writes `data/paddle_finetune/dataset/{train,val}.txt` (85/15 by **group**)
3. `scripts/trocr_finetune/convert_to_jsonl.py` — reads those, writes
   `data/trocr_finetune/{train,val}.jsonl` verbatim, no further filtering

`scripts/trocr_finetune/package_images_zip.py` builds the transfer zip. Its
hardcoded `EXPECTED_COUNT = 2902` was **fixed on 27-07-2026** to derive the
count dynamically, with an optional `--expected-count` sanity flag; output is
now named `images_<count>.zip`.

### ⚠️ Misnamed artifacts (will confuse someone — check contents, not names)

- `data/trocr_finetune_3896_v2/` actually holds **3,936** crops
- the 5,695-crop set's zip is still named **`images_1968.zip`** (1968 is a
  stale count from the older PaddleOCR dataset)
- `images_3896crop_bak/` on the server is debris from a 24-07 transfer mixup
  (3,895 files, one filename collision) — unused, safe to ignore or delete
- `crop_NNNN_` indices are **re-numbered fresh on every extraction run** and
  the crops dir is never cleaned between runs, so index N from one run can
  collide with index N from another. Never use crop index alone to identify
  which dataset build a file came from.

---

## Benchmark facts

### There is only one script

`benchmark_trocr_cascade_quicksample.py`. The "FULL" 786-image run is the
**same script** invoked with `BENCH_SAMPLE_FILE=''` (falsy → no sample
filter) and `BENCH_OUTPUT_CSV` overridden. Its results banner still prints
"QUICK SAMPLE" on full runs — cosmetic mislabel, don't let it confuse a file
later.

- Checkpoint via `TROCR_CHECKPOINT_DIR` env var (forwarded internally to
  `FINETUNED_TROCR_MODEL_DIR`) — **not** `FINETUNED_REC_MODEL_DIR`, which is
  the PaddleOCR override.
- Sample file is **`quick_sample_set_resplit.txt`** (169 lines: 12
  header/comment + 157 filenames), *not* `quick_sample_set.txt`.
- Test dir: `container.v2i.yolov11.resplit/test/images` (786 files, 29 MB).
  YOLO weights: `models/best_resplit.pt` (22.5 MB).
- Hardcoded `sys.path.insert` / `os.chdir` to `~/Testing/container_ocr_test`
  — it cannot be relocated to another machine without editing those paths.
- Runtime on local CPU: ~19–20 min for 157 images → **~90–120 min for 786**.
  (The older "~2.3–2.8 hrs" figure is the *PaddleOCR* cascade, not TrOCR.)

### Denominator convention — always use rows/rows

Full-cascade accuracy has two possible denominators: valid-rows ÷ **images**
(786) or valid-rows ÷ **rows** (some images have >1 detection, so rows >
images). **Always use rows/rows** — it's what the historical 69.0%/64.7%
figures use, and mixing conventions produces incomparable numbers (this exact
mixup produced a bad "73.8%" figure once already). State which convention any
number uses; don't quote a bare percentage.

### Per-tier wrong-rates (from manual review, carry forward unless re-reviewed)

Tier1-exact 0% · Tier1b-recovery 13.6% · Tier2-exact/merge ~8.7% ·
Tier2-recovery ~20% (worst) · Tier4-confusion ~4.8%.

**Tier3-Pass1 and Pass2 are both now removed from `ocr.py`.** Pass 2 went
first (81% wrong, 17/21 verified). Pass 1 was removed 27-07-2026 after manual
ground-truth review of its 5 rows found **2 confirmed wrong, 2 suspicious,
1 unverifiable, 0 confirmed correct** — supersedes the earlier "negligible"
estimate in this file, which was a bad proxy.

Both were blind sliding-window matches over concatenated multi-line OCR text,
accepting the first window that happened to pass checksum. The check digit
catches roughly 10 in 11 random strings, so scanning dozens of window
positions will eventually hit a coincidental pass. There is no per-line
confidence to gate on, because the window spans text merged across detection
boxes.

Note the removal is **near-cosmetic in practice**: the fallback scan below it
picks the first structurally valid window and still validates it, so a
checksum-passing window is still accepted, just labelled differently. The
only behaviour that changed is the case where an earlier window fails
checksum and a later one passes. Verified against real CSV rows before and
after.

⚠️ **Tier labels in benchmark output are approximate** — the capture handler
records the *last* Tier log line across all rotation attempts, not
necessarily the one that produced the returned result. Caveat every tier
breakdown accordingly.

### Replay script

`scripts/replay_tier_attribution.py` (or the `_resplit.py` variant — check
which matches the CSV schema first) computes per-tier counts and a
reconciliation check, but does **not** output a single adjusted-accuracy
percentage. Compute it yourself:
`sum(tier_count × (1 - wrong_rate)) / total_rows`.

**Known gap:** its Tier2 logic predates the 03-07-2026 "multi-line merge" fix
in `ocr.py` — it will mark some genuinely-valid rows UNRECOVERABLE. Check
`validation_reason` for the "multi-line merge" tag on any UNRECOVERABLE row
before treating it as a real gap; fold tagged rows into Tier2-exact and
reconcile the row count exactly.

---

## Drive layout (`container-ocr/` on Drive is single source of truth)

- `models/yolov8s_resplit_full_CONFIRMED_GOOD/` — current best YOLO detector
- `models/yolov8s_100ep_CONFIRMED_GOOD/` — old, leakage-contaminated,
  historical only
- `models/paddle_finetune_run_colab_day17/` — Colab retrain output
  (`best_accuracy/` = epoch 128 winner, `latest/` = epoch 150 final state;
  intermediate `iter_epoch_N/` snapshots deleted after run completion)
- `data/paddle_finetune/` — corrected PaddleOCR fine-tune dataset
  (`dict.txt`=35 lines, `train.txt`=1668, `val.txt`=300, `images/`=1968
  referenced files — the local `images/` folder may contain stale
  unreferenced extras from before the Day-15 `group_id()` fix; **always read
  image lists from train.txt/val.txt, never glob the folder**)
- `scripts/paddle_finetune/` — synced from local machine 17-07-2026

---

## Standing process rules (in addition to AGENTS.md's Hard Rules)

- **One benchmark/training run at a time.** Confirm via `ps aux` before
  launching another, even on a different platform. Measured 27-07-2026: a
  single CPU cascade benchmark consumes ~650% of 8 cores with load average
  9.12 — there is **no headroom** for a concurrent run, and two would
  oversubscribe rather than parallelise. On the GPU server, a single
  fine-tune saturates the card at 100% utilisation and the power cap.
- Long runs always launch via `nohup ... & disown`, never a foreground
  blocking call, so they survive disconnects. Exception: Colab cells are
  foreground by design — a long training cell blocking the notebook's queue
  is expected, not a hang.
- **Verify large binary transfers by size/count, not a "success" message.**
  For files small enough to diff (scripts, configs), verify by hash/byte-diff.
  Content passing through a text-API round-trip (Drive MCP read-then-rewrite)
  is more failure-prone than a raw browser download — a real 12-dropped-byte
  transcription bug was caught this way on 17-07-2026.
- **Clear `.pyc` cache before benchmarking any `ocr.py` edit.** A stale
  `app/inference/__pycache__/ocr.cpython-310.pyc` newer than the source has
  caused a benchmark to silently execute pre-edit code before.
- **Before pursuing any new accuracy lever, explicitly weigh it against
  remaining timeline and diminishing returns** rather than defaulting to
  "let's also try this." Once a clean baseline exists, further gains are
  optional and time-boxed. Specifically: don't run experiments whose expected
  effect is smaller than the measurement instrument can resolve.
- **Evidence before action** — no new investigation opens until current
  results are in hand and analysed. Multiple hypotheses on 27-07-2026 were
  proposed and refuted at real cost; verify one before proposing the next.

---

## Open items / not yet started

- **Full 786-image benchmark on current code** — none exists for any TrOCR
  checkpoint. Staged command ready for thin (cap=3). PaddleOCR also needs a
  re-run for comparability post-Pass1-removal.
- **TrOCR production integration** into `app/inference/ocr.py` — never
  started. **CPU inference latency never measured.** This is the genuine
  blocking risk, not accuracy.
- **Migration drift** — `uploads`, `detections`, `audit_log` tables lack
  proper Alembic migration files. Flagged repeatedly, still untouched.
- **Tier 3 "never returns valid"** — a larger change than the Pass 1 removal
  (forcing `best_valid = False` on the Tier 3 path). Deferred as a separate
  decision.
- Known limitation: `_best_portrait_rotation`'s tie-break (`conf_90 >=
  conf_m90`) is meaningless under TrOCR, since TrOCR's confidence is a
  hardcoded 1.0 placeholder for both rotations. This only activates when
  BOTH rotations already fail to validate (`is_valid` is unaffected either
  way) — it only decides which garbage OCR text is shown as a hint for
  manual correction on rows already flagged invalid. Biased toward +90
  unconditionally. Not fixed due to remaining timeline; a real fix would
  need a TrOCR-appropriate tie-break signal since confidence isn't
  meaningful here.
- Bootstrap-vs-Tailwind colour reconciliation across the app (only
  login/signup pages fixed).
- Stale comment at `ocr.py:389` reads "Same exclusion as Pass 1 above" but
  Pass 1's exclusion code no longer exists. Cosmetic.
- RTX 2000 Ada GPU server proposal — awaiting reply as of Day 17. Note the
  current server already identifies as an RTX 2000 Ada; worth confirming
  whether this is the same machine.
