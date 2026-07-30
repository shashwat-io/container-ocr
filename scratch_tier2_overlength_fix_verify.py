"""
READ-ONLY verification. Live extract_container_code() calls (real OCR)
against all 23 Tier2-exact + 35 Tier2-recovery rows from today's review
sheets, to confirm the one-char-over-length removal fix:
  - fixes sample 17 (CXNU1955119)
  - leaves sample 13 unaffected (still the pre-existing wrong IRKU8476800)
  - introduces zero regressions on any row previously graded TRUE/correct
No ocr.py/predictor.py edits happen here -- this only exercises the
already-edited code.
"""
import os
import sys
import csv
import logging
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

# Surface the new AMBIGUOUS warning (and Tier2 info logs) to stdout so any
# multi-valid-removal hit during this run is visible, not silently swallowed.
_ambiguous_hits = []
class _AmbiguousCaptureHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if 'AMBIGUOUS' in msg:
            _ambiguous_hits.append(msg)
            print(f"  *** {msg}")
ocr_logger = logging.getLogger('app.inference.ocr')
ocr_logger.setLevel(logging.INFO)
ocr_logger.addHandler(_AmbiguousCaptureHandler())

from ultralytics import YOLO
from app.inference.ocr import extract_container_code

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'

model = YOLO('models/best.pt')
_box_cache = {}
def boxes_for(fname):
    if fname in _box_cache:
        return _box_cache[fname]
    img = np.array(Image.open(os.path.join(TEST_DIR, fname)).convert('RGB'))
    res = model(img, verbose=False)
    bxs = [[float(v) for v in b.xyxy[0]] for r in res for b in r.boxes]
    _box_cache[fname] = bxs
    return bxs

def run_row(fname, box_idx):
    bxs = boxes_for(fname)
    x1, y1, x2, y2 = bxs[box_idx]
    with open(os.path.join(TEST_DIR, fname), 'rb') as fh:
        image_bytes = fh.read()
    return extract_container_code(image_bytes, x1, y1, x2, y2)

def process_sheet(path, tier_name):
    rows = list(csv.DictReader(open(path)))
    print("=" * 100)
    print(f"{tier_name}: {len(rows)} rows")
    print("=" * 100)
    regressions = []
    changes = []
    ambiguous_hits = []
    for r in rows:
        fname = r['filename']
        box_idx = int(r['box_idx'])
        old_code = r['claimed_code']
        verified = r['verified_correct'].strip()
        was_correct = verified.upper().startswith('TRUE') or 'true' in verified.lower()

        try:
            result = run_row(fname, box_idx)
        except Exception as e:
            print(f"  ERROR on {fname}: {e}")
            continue

        new_code = result['container_code']
        new_valid = result['is_valid']
        new_reason = result['validation_reason']
        new_ambig = result['is_ambiguous_merge']

        changed = (new_code != old_code)
        tag = ""
        if changed:
            changes.append((r['sample_number'], fname, old_code, new_code, was_correct, verified))
            if was_correct:
                regressions.append((r['sample_number'], fname, old_code, new_code, verified))
                tag = "  <<< REGRESSION (was correct, now changed)"
            else:
                tag = "  <<< changed (was marked wrong/known-bad)"
        if new_ambig:
            tag += "  <<< is_ambiguous_merge=True"
            ambiguous_hits.append((r['sample_number'], fname))

        print(f"sample{r['sample_number']:>3} | {fname[:45]:45s} | old={old_code:12s} new={new_code:12s} "
              f"valid={new_valid} ambig={new_ambig} reason={new_reason!r}{tag}")

    print()
    print(f"{tier_name} summary: {len(changes)} changed codes out of {len(rows)}; {len(regressions)} regressions (were TRUE/correct)")
    print(f"{tier_name} is_ambiguous_merge=True count: {len(ambiguous_hits)} -- {ambiguous_hits}")
    if regressions:
        print("REGRESSIONS:")
        for reg in regressions:
            print("  ", reg)
    return regressions, changes, ambiguous_hits

reg1, ch1, amb1 = process_sheet('tier2_exact_review_samples/review_sheet.csv', 'Tier2-exact')
print()
reg2, ch2, amb2 = process_sheet('tier2_recovery_review_samples/review_sheet.csv', 'Tier2-recovery')

print()
print("=" * 100)
print(f"TOTAL REGRESSIONS: {len(reg1) + len(reg2)}")
print(f"TOTAL is_ambiguous_merge=True rows: {len(amb1) + len(amb2)} -- {amb1 + amb2}")
print(f"AMBIGUOUS log warnings during this run: {len(_ambiguous_hits)}")
for h in _ambiguous_hits:
    print("  ", h)
print("=" * 100)
