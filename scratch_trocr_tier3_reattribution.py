"""
One-off analysis pass: re-derive the CORRECT tier for the 15 passing rows
currently labeled 'Tier3-fallback-novalid' in
benchmark_results_trocr_cascade_quicksample.csv.

Does NOT re-run the benchmark, does NOT touch pass/fail numbers -- purely
re-labels tier attribution for these 15 rows using isolated per-rotation
log capture (instead of the shared _last_tier_msg dict the original
benchmark script used, which only keeps the LAST Tier-message logged
across BOTH rotation attempts, not necessarily the one
_best_portrait_rotation() actually selected).

For each of the 15 rows: re-run YOLO on the image (deterministic, same
weights) to recover the box, re-crop, and call ocr._best_portrait_rotation
directly but with two SEPARATE, isolated tier captures (one per rotation)
instead of one shared capture -- so we know exactly which rotation's tier
resolved the FINAL, selected result, independent of call order.

All 15 target rows have num_boxes=1, so there's no box-index ambiguity.
"""
import csv
import os
import sys

TROCR_CHECKPOINT_DIR = 'trocr_finetune_output'
os.environ['FINETUNED_TROCR_MODEL_DIR'] = TROCR_CHECKPOINT_DIR

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

import numpy as np
from PIL import Image
from ultralytics import YOLO
import app.inference.ocr as ocr
from app.inference.ocr_trocr import get_trocr, run_trocr_on_crop

def _run_trocr_as_ocr_line(crop_image, docpp_off=False):
    text = run_trocr_on_crop(crop_image)
    if not text.strip():
        return []
    return [(text, 1.0)]

ocr.run_ocr_on_crop = _run_trocr_as_ocr_line

TEST_DIR = 'container.v2i.yolov11.resplit/test/images'
MODEL_PATH = 'models/best_resplit.pt'
CSV_PATH = 'benchmark_results_trocr_cascade_quicksample.csv'

with open(CSV_PATH, newline='') as f:
    all_rows = list(csv.DictReader(f))

target_rows = [r for r in all_rows if r['tier'] == 'Tier3-fallback-novalid' and r['is_valid'] == 'True']
print(f"Target rows to re-derive: {len(target_rows)}")
assert len(target_rows) == 15, f"expected 15, got {len(target_rows)} -- CSV may have changed"

_, _, device = get_trocr()
print(f"TrOCR loaded on device={device}")

model = YOLO(MODEL_PATH)

# Per-rotation isolated tier capture: instead of one shared dict, capture
# into whichever dict is "active" at call time.
_active_capture = {'dict': None}

import logging

class IsolatedTierCaptureHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith('Tier '):
            d = _active_capture['dict']
            if d is not None:
                d['msg'] = msg

ocr_logger = logging.getLogger('app.inference.ocr')
ocr_logger.setLevel(logging.INFO)
ocr_logger.addHandler(IsolatedTierCaptureHandler())
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

def classify_from_msg(tier_msg):
    if not tier_msg:
        return 'unresolved'
    for prefix, label in TIER_LABELS:
        if tier_msg.startswith(prefix):
            return label
    return 'unresolved'


def isolated_try_extract_best_valid(crop):
    """Same as ocr._try_extract_best_valid, but with its OWN fresh capture dict."""
    capture = {'msg': None}
    _active_capture['dict'] = capture
    result = ocr._try_extract_best_valid(crop)
    _active_capture['dict'] = None
    return result, capture['msg']


def rederive(img_path, x1, y1, x2, y2):
    with open(img_path, 'rb') as f:
        image_bytes = f.read()
    crop = ocr.crop_detection(image_bytes, x1, y1, x2, y2)
    w, h = crop.size
    is_portrait = h > w

    if not is_portrait:
        processed = ocr._post_process(crop)
        (code, valid, reason, raw, conf), tier_msg = isolated_try_extract_best_valid(processed)
        return {
            'orientation': 'landscape',
            'winning_rotation': 'n/a',
            'winning_tier': classify_from_msg(tier_msg),
            'code': code, 'valid': valid, 'reason': reason, 'conf': conf,
            'plus90_tier': None, 'minus90_tier': None,
        }

    crop_90 = ocr._post_process(crop.copy().rotate(90, expand=True))
    crop_m90 = ocr._post_process(crop.copy().rotate(-90, expand=True))

    (code_90, valid_90, reason_90, raw_90, conf_90), tier_msg_90 = isolated_try_extract_best_valid(crop_90)
    (code_m90, valid_m90, reason_m90, raw_m90, conf_m90), tier_msg_m90 = isolated_try_extract_best_valid(crop_m90)

    # Reproduce _best_portrait_rotation's own selection logic exactly.
    is_retry_90 = valid_90 and 'docpp-off retry' in reason_90
    is_retry_m90 = valid_m90 and 'docpp-off retry' in reason_m90
    rank_90 = 2 if (valid_90 and not is_retry_90) else (1 if valid_90 else 0)
    rank_m90 = 2 if (valid_m90 and not is_retry_m90) else (1 if valid_m90 else 0)

    if rank_90 > rank_m90:
        winner = '+90'
    elif rank_m90 > rank_90:
        winner = '-90'
    elif conf_90 >= conf_m90:
        winner = '+90'
    else:
        winner = '-90'

    winning = (code_90, valid_90, reason_90, conf_90) if winner == '+90' else (code_m90, valid_m90, reason_m90, conf_m90)
    winning_tier_msg = tier_msg_90 if winner == '+90' else tier_msg_m90

    return {
        'orientation': 'portrait',
        'winning_rotation': winner,
        'winning_tier': classify_from_msg(winning_tier_msg),
        'code': winning[0], 'valid': winning[1], 'reason': winning[2], 'conf': winning[3],
        'plus90_tier': classify_from_msg(tier_msg_90),
        'minus90_tier': classify_from_msg(tier_msg_m90),
    }


results = []
for i, row in enumerate(target_rows, 1):
    fname = row['filename']
    path = os.path.join(TEST_DIR, fname)
    img_pil = Image.open(path).convert('RGB')
    img_np = np.array(img_pil)

    yolo_results = model(img_np, verbose=False)
    boxes = []
    for r in yolo_results:
        for box in r.boxes:
            bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0]]
            boxes.append((bx1, by1, bx2, by2))

    assert len(boxes) == 1, f"{fname}: expected 1 box (num_boxes={row['num_boxes']}), got {len(boxes)}"
    x1, y1, x2, y2 = boxes[0]

    info = rederive(path, x1, y1, x2, y2)

    # Sanity check: does the re-derived result match the CSV's recorded outcome?
    matches = (info['code'] == row['container_code'] and str(info['valid']) == row['is_valid'])

    results.append({
        'filename': fname,
        'recorded_code': row['container_code'],
        'recorded_reason': row['validation_reason'],
        'original_tier_label': row['tier'],
        **info,
        'reproduction_matches': matches,
    })

    print(f"[{i}/15] {fname}")
    print(f"    recorded: code={row['container_code']!r} reason={row['validation_reason']!r} (original tier label: Tier3-fallback-novalid)")
    print(f"    re-derived: orientation={info['orientation']} winner={info['winning_rotation']} "
          f"code={info['code']!r} valid={info['valid']} reason={info['reason']!r}")
    if info['orientation'] == 'portrait':
        print(f"    per-rotation tiers: +90={info['plus90_tier']}  -90={info['minus90_tier']}  => CORRECT tier = {info['winning_tier']}")
    else:
        print(f"    single-call tier (no rotation ambiguity possible): {info['winning_tier']}")
    print(f"    reproduction matches CSV row: {matches}")
    print()

print("\n=== SUMMARY ===")
mismatches = [r for r in results if not r['reproduction_matches']]
print(f"Rows where re-derivation did NOT reproduce the recorded result: {len(mismatches)}")
for r in mismatches:
    print(f"  {r['filename']}: recorded={r['recorded_code']!r} vs re-derived={r['code']!r}")

print("\nCorrected tier assignment for the 15 rows:")
from collections import Counter
corrected_counts = Counter(r['winning_tier'] for r in results)
for tier, count in corrected_counts.most_common():
    print(f"  {tier}: {count}")

print("\nPer-row correction detail:")
for r in results:
    changed = r['winning_tier'] != r['original_tier_label']
    print(f"  {r['filename']:65s} Tier3-fallback-novalid -> {r['winning_tier']:22s} {'(CHANGED)' if changed else '(unchanged)'}")

import csv as csvmod
with open('scratch_trocr_tier3_reattribution_results.csv', 'w', newline='') as f:
    fieldnames = ['filename', 'recorded_code', 'recorded_reason', 'original_tier_label',
                  'orientation', 'winning_rotation', 'plus90_tier', 'minus90_tier',
                  'winning_tier', 'reproduction_matches']
    writer = csvmod.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(results)
print("\nDetail written to scratch_trocr_tier3_reattribution_results.csv")
