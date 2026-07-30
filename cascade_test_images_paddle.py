"""
One-off cascade run of test1.jpg-test9.jpg through the real detection+OCR
pipeline (YOLO -> app.inference.ocr.extract_container_code), unmodified.

Adapted from benchmark_full_live_resplit.py: same model load, same
extract_container_code call, same tier-capture log handler. Only the image
source changed (9 personal sanity-check images instead of the resplit test
set) and per-box results are printed directly instead of just aggregated,
since this is 9 images, not 786.

Local-only. Does not touch the production server. Run with:
    OCR_RECOGNIZER=paddle python3 cascade_test_images_paddle.py
"""
import os
import sys
import logging
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from ultralytics import YOLO
from app.inference.ocr import extract_container_code, get_recognizer

PROJECT_DIR = 'test_images'
FALLBACK_DIR = os.path.expanduser('~/Downloads/test images')
MODEL_PATH = 'models/best_resplit.pt'

if all(os.path.isfile(os.path.join(PROJECT_DIR, f'test{i}.jpg')) for i in range(1, 10)):
    TEST_DIR = PROJECT_DIR
elif all(os.path.isfile(os.path.join(FALLBACK_DIR, f'test{i}.jpg')) for i in range(1, 10)):
    TEST_DIR = FALLBACK_DIR
else:
    print("FATAL: could not find all of test1.jpg..test9.jpg in either "
          f"{PROJECT_DIR} or {FALLBACK_DIR}", flush=True)
    sys.exit(1)

# ── Tier attribution via log capture (identical to benchmark_full_live_resplit.py) ──
_last_tier_msg = {'msg': None}

class TierCaptureHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
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


print(f"OCR_RECOGNIZER resolved to: {get_recognizer()!r}", flush=True)
if get_recognizer() != 'paddle':
    print("FATAL: OCR_RECOGNIZER must be 'paddle' for this run -- "
          "set it in the environment before launching.", flush=True)
    sys.exit(1)

print(f"Loading YOLO model: {MODEL_PATH}", flush=True)
model = YOLO(MODEL_PATH)

images = [f'test{i}.jpg' for i in range(1, 10)]
print(f"Test dir: {os.path.abspath(TEST_DIR)}\n", flush=True)

for fname in images:
    path = os.path.join(TEST_DIR, fname)
    img_pil = Image.open(path).convert('RGB')
    img_np = np.array(img_pil)
    with open(path, 'rb') as f:
        image_bytes = f.read()

    results = model(img_np, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            boxes.append((x1, y1, x2, y2))

    print(f"=== {fname} ===")
    if not boxes:
        print("  No detection")
        continue

    for bi, (x1, y1, x2, y2) in enumerate(boxes, 1):
        _last_tier_msg['msg'] = None
        result = extract_container_code(image_bytes, x1, y1, x2, y2)
        tier = classify_tier(result['validation_reason'], _last_tier_msg['msg'])
        print(f"  box {bi}/{len(boxes)}: code={result['container_code']!r}  "
              f"is_valid={result['is_valid']}  tier={tier}  "
              f"reason={result['validation_reason']!r}")
    print()

print("DONE", flush=True)
