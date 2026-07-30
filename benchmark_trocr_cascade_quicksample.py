"""
Full-cascade (YOLO detection -> crop -> TrOCR fine-tuned recognizer)
benchmark, directional-only, on the 157-image quick_sample_set_resplit.txt
(seed=42, paired with the resplit test set).

First-ever full-cascade eval of the fine-tuned TrOCR checkpoint -- it has
only been evaluated so far on isolated ground-truth crops (46.33%
exact-match, no detector, no cascade). Adapted from
benchmark_full_live_resplit.py (used for the Day 17 PaddleOCR fine-tuned
cascade run): same YOLO detector call, same tier-attribution log-capture
instrumentation, same CSV schema/checkpointing structure. The only thing
swapped is the text recognizer: app.inference.ocr.run_ocr_on_crop is
monkeypatched (module-level, same pattern already used by other
diagnostic/comparison scripts in this repo, e.g. scripts/compare_ppocrv5.py)
to call TrOCR instead of PaddleOCR. ocr.py itself is never edited -- its
REAL confidence-based rotation logic (_best_portrait_rotation) and REAL
tiered validation cascade (_try_extract) run completely unmodified, just
fed TrOCR output instead of PaddleOCR output. This deliberately does NOT
reuse ocr_trocr.py's own extract_container_code_trocr_rotated /
simplified landscape-rotate-90 heuristic -- that heuristic was a
TRAINING-TIME shortcut; the point of this run is to measure the fine-tuned
checkpoint under the pipeline that's actually deployed.

Known, expected artifacts of swapping a whole-crop single-line recognizer
into a pipeline built around a multi-line detector+recognizer:
- TrOCR returns exactly one predicted string per crop (no per-line boxes),
  so Tier 2's adjacent-line-pair merge logic structurally can never fire --
  every resolution lands in Tier1/Tier1b/Tier3/Tier4 territory, or goes
  unresolved. Expected, not a bug.
- TrOCR has no per-result confidence score comparable to PaddleOCR's
  rec_scores. A fixed placeholder confidence of 1.0 is reported for any
  non-empty prediction (0.0 for empty), purely so _best_portrait_rotation's
  confidence tiebreak and the CSV's ocr_confidence column don't error out.
  Do NOT compare these numbers to PaddleOCR confidence figures from other
  benchmark runs.
- ocr.py's docpp_off retry pass (a PaddleX-specific doc-orientation-classify
  toggle) has no TrOCR equivalent; the monkeypatched recognizer ignores the
  docpp_off flag entirely, so the retry pass always reproduces the primary
  pass's result byte-for-byte. This roughly doubles compute for
  already-invalid crops but never changes the outcome. Expected, not a bug.

Reads:  container.v2i.yolov11.resplit/test/images/ (leakage-clean resplit
        test set) filtered to quick_sample_set_resplit.txt (157 images,
        seed=42 -- NOT interchangeable with quick_sample_set.txt, which is
        drawn from the old contaminated split; see that file's own header).
Model:  models/best_resplit.pt (YOLO detector, CONFIRMED_GOOD)
        TrOCR checkpoint dir from TROCR_CHECKPOINT_DIR env var (default:
        trocr_finetune_output/final) -- must be a HF-loadable
        VisionEncoderDecoderModel checkpoint dir (checkpoint-1672,
        best@epoch 8). Script refuses to run if this dir doesn't exist --
        it will NOT silently fall back to the zero-shot base model.
Writes: benchmark_results_trocr_cascade_quicksample.csv (per-row raw OCR
        output + validation outcome). Written every CHECKPOINT_EVERY images
        AND at completion -- NOT atomic/completion-only (matches
        benchmark_full_live_resplit.py's own behavior), so partial output
        mid-run is expected, not a sign of a stall.
"""
import os

# Must be set before importing app.inference.ocr_trocr -- that module reads
# this env var into a module-level constant (FINETUNED_TROCR_MODEL_DIR) at
# import time, mirroring ocr.py's own FINETUNED_REC_MODEL_DIR pattern.
TROCR_CHECKPOINT_DIR = os.environ.get('TROCR_CHECKPOINT_DIR', 'trocr_finetune_output/final')
os.environ['FINETUNED_TROCR_MODEL_DIR'] = TROCR_CHECKPOINT_DIR

import csv
import logging
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

if not os.path.isdir(TROCR_CHECKPOINT_DIR):
    print(f"FATAL: TROCR_CHECKPOINT_DIR does not exist: {TROCR_CHECKPOINT_DIR}", flush=True)
    print("Set TROCR_CHECKPOINT_DIR to the transferred checkpoint's directory "
          "(the one containing config.json + model weights) before running -- "
          "refusing to silently fall back to the zero-shot base model.", flush=True)
    sys.exit(1)

from ultralytics import YOLO
import app.inference.ocr as ocr
from app.inference.ocr import extract_container_code
from app.inference.ocr_trocr import get_trocr, run_trocr_on_crop

# ── Monkeypatch: swap PaddleOCR out for TrOCR inside ocr.py's own call graph ──
# extract_container_code -> _best_portrait_rotation / _try_extract_best_valid
# -> _try_extract -> run_ocr_on_crop. Patching the module-level name makes
# every one of those (unmodified) functions call TrOCR instead, so the REAL
# confidence-based rotation logic and REAL tier cascade run over TrOCR output.
def _run_trocr_as_ocr_line(crop_image, docpp_off=False):
    text = run_trocr_on_crop(crop_image)
    if not text.strip():
        return []
    return [(text, 1.0)]  # placeholder confidence -- see module docstring

ocr.run_ocr_on_crop = _run_trocr_as_ocr_line

TEST_DIR = '/home/shashwatkumar/Testing/container_ocr_test/container.v2i.yolov11.resplit/test/images'
MODEL_PATH = os.environ.get('BENCH_MODEL_PATH', 'models/best_resplit.pt')
OUTPUT_CSV = os.environ.get('BENCH_OUTPUT_CSV',
                             '/home/shashwatkumar/Testing/container_ocr_test/benchmark_results_trocr_cascade_quicksample.csv')
SAMPLE_FILE = os.environ.get('BENCH_SAMPLE_FILE', 'quick_sample_set_resplit.txt')
CHECKPOINT_EVERY = 25  # smaller than the 786-image scripts' 100 -- this is a 157-image run

# ── Tier attribution via log capture (identical to benchmark_full_live_resplit.py) ──
_last_tier_msg = {'msg': None}

class TierCaptureHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith('Tier ') or msg.startswith('Portrait:') or msg.startswith('Landscape:'):
            if msg.startswith('Tier '):
                _last_tier_msg['msg'] = msg

ocr_logger = logging.getLogger('app.inference.ocr')
ocr_logger.setLevel(logging.INFO)
ocr_logger.addHandler(TierCaptureHandler())
ocr_logger.propagate = False

TIER_LABELS = [
    ('Tier 1 (single-line exact)', 'Tier1-exact'),
    ('Tier 1b (single-line check-digit recovery)', 'Tier1b-recovery'),
    ('Tier 2 (adjacent-pair 11-char)', 'Tier2-exact'),
    ('Tier 2 (adjacent-pair check-digit)', 'Tier2-recovery'),
    ('Tier 3 (merged Pass 1)', 'Tier3-Pass1'),
    ('Tier 3 (merged fallback, no valid)', 'Tier3-fallback-novalid'),
    ('Tier 4 (confusion correction)', 'Tier4-confusion'),
]

def classify_tier(reason, tier_msg):
    if tier_msg:
        for prefix, label in TIER_LABELS:
            if tier_msg.startswith(prefix):
                return label
    if not reason:
        return 'none'
    if 'docpp-off retry' in reason:
        return 'docpp-off-retry'
    if 'character confusion corrected' in reason:
        return 'Tier4-confusion'
    if 'check-digit recovered' in reason:
        return 'check-digit-recovery'
    if reason == 'Valid':
        return 'literal-match'
    return 'unresolved'


print(f"Loading TrOCR from checkpoint: {TROCR_CHECKPOINT_DIR}", flush=True)
_, _, trocr_device = get_trocr()
print(f"TrOCR loaded on device={trocr_device}", flush=True)

model = YOLO(MODEL_PATH)

images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

if SAMPLE_FILE:
    with open(SAMPLE_FILE) as f:
        sample_names = {l.strip() for l in f if l.strip() and not l.startswith('#')}
    images = [f for f in images if f in sample_names]
    missing = sample_names - set(images)
    if missing:
        print(f"WARNING: {len(missing)} sample filenames not found in {TEST_DIR}: {sorted(missing)[:5]}...", flush=True)

total = len(images)
print(f"Found {total} images (model={MODEL_PATH}, test_dir={TEST_DIR}, sample_file={SAMPLE_FILE})", flush=True)

stats = {'detected': 0, 'ocr_read': 0, 'valid': 0, 'no_detection': 0, 'invalid': 0}
tier_pass_counts = {}
start = time.time()
rows = []

fieldnames = ['filename', 'detected', 'num_boxes', 'container_code', 'is_valid',
              'validation_reason', 'tier', 'ocr_confidence', 'ocr_raw']

def write_csv():
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

for i, fname in enumerate(images, 1):
    path = os.path.join(TEST_DIR, fname)
    try:
        img_pil = Image.open(path).convert('RGB')
        img_np = np.array(img_pil)
        with open(path, 'rb') as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"  Load error {fname}: {e}", flush=True)
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                     'container_code': '', 'is_valid': False, 'validation_reason': f'load error: {e}',
                     'tier': 'none', 'ocr_confidence': 0.0, 'ocr_raw': ''})
        continue

    try:
        results = model(img_np, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                boxes.append((x1, y1, x2, y2))
    except Exception as e:
        print(f"  YOLO error {fname}: {e}", flush=True)
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                     'container_code': '', 'is_valid': False, 'validation_reason': f'YOLO error: {e}',
                     'tier': 'none', 'ocr_confidence': 0.0, 'ocr_raw': ''})
        continue

    if not boxes:
        stats['no_detection'] += 1
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                     'container_code': '', 'is_valid': False, 'validation_reason': 'No detection',
                     'tier': 'none', 'ocr_confidence': 0.0, 'ocr_raw': ''})
    else:
        stats['detected'] += 1
        for x1, y1, x2, y2 in boxes:
            _last_tier_msg['msg'] = None
            try:
                result = extract_container_code(image_bytes, x1, y1, x2, y2)
                tier = classify_tier(result['validation_reason'], _last_tier_msg['msg'])
                if result['container_code']:
                    stats['ocr_read'] += 1
                if result['is_valid']:
                    stats['valid'] += 1
                    tier_pass_counts[tier] = tier_pass_counts.get(tier, 0) + 1
                else:
                    stats['invalid'] += 1
                rows.append({
                    'filename': fname,
                    'detected': True,
                    'num_boxes': len(boxes),
                    'container_code': result['container_code'],
                    'is_valid': result['is_valid'],
                    'validation_reason': result['validation_reason'],
                    'tier': tier,
                    'ocr_confidence': result['ocr_confidence'],
                    'ocr_raw': result['ocr_raw_output'],
                })
            except Exception as e:
                stats['invalid'] += 1
                rows.append({'filename': fname, 'detected': True, 'num_boxes': len(boxes),
                             'container_code': '', 'is_valid': False, 'validation_reason': f'OCR error: {e}',
                             'tier': 'none', 'ocr_confidence': 0.0, 'ocr_raw': ''})

    if i % CHECKPOINT_EVERY == 0 or i == total:
        elapsed = time.time() - start
        write_csv()
        tier_summary = ', '.join(f"{k}:{v}" for k, v in sorted(tier_pass_counts.items()))
        print(
            f"[CHECKPOINT {i}/{total}] {elapsed:.0f}s elapsed | avg {elapsed/i:.1f}s/img | "
            f"pass so far: {stats['valid']} | fail so far: {stats['invalid'] + stats['no_detection']} | "
            f"tiers: {tier_summary} | checkpoint CSV written ({len(rows)} rows)",
            flush=True
        )

elapsed = time.time() - start
write_csv()

d = stats['detected'] or 1
total_rows = len(rows)
print(f"\n=== BENCHMARK RESULTS (TrOCR FINE-TUNED CASCADE, QUICK SAMPLE) ===", flush=True)
print(f"YOLO model:       {MODEL_PATH}", flush=True)
print(f"TrOCR checkpoint: {TROCR_CHECKPOINT_DIR}", flush=True)
print(f"Sample file:      {SAMPLE_FILE}", flush=True)
print(f"Total images:     {total}", flush=True)
print(f"Total rows:       {total_rows} (rows > images when any image has >1 YOLO box)", flush=True)
print(f"YOLO detected:    {stats['detected']} ({stats['detected']/total*100:.1f}% of images)", flush=True)
print(f"OCR read code:    {stats['ocr_read']} ({stats['ocr_read']/d*100:.1f}% of detected)", flush=True)
print(f"ISO 6346 valid (rows/rows, per CLAUDE.md convention): {stats['valid']}/{total_rows} "
      f"= {stats['valid']/total_rows*100:.1f}%", flush=True)
print(f"ISO 6346 valid (valid/images, NOT the convention -- for reference only): "
      f"{stats['valid']}/{total} = {stats['valid']/total*100:.1f}%", flush=True)
print(f"No detection:     {stats['no_detection']} ({stats['no_detection']/total*100:.1f}%)", flush=True)
print(f"\n--- Tier breakdown (passing rows only) ---", flush=True)
for k, v in sorted(tier_pass_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v}", flush=True)
print(f"\nTotal time:       {elapsed:.0f}s | Avg per image: {elapsed/total:.1f}s", flush=True)
print(f"Results saved to: {OUTPUT_CSV}", flush=True)
print(f"RUN COMPLETED UNINTERRUPTED", flush=True)
