"""
Targeted check for the new Tier 3 TrOCR gate in app/inference/ocr.py: run the
exact 3 rows confirmed as genuine Tier 3 passes (see
scratchpad/tier3_review_data.json) through the real extract_container_code()
production entry point, OCR_RECOGNIZER=trocr, and confirm is_valid is now
False for all 3.

Uses the same correct image-loading path as tier3_review_generate.py's fixed
version (PIL -> RGB -> np.array -> YOLO), not a raw path string.
"""
import os, sys

os.environ['OCR_RECOGNIZER'] = 'trocr'
os.environ['FINETUNED_TROCR_MODEL_DIR'] = os.path.expanduser('~/Downloads/trocr_checkpoint_thin')

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from PIL import Image
import numpy as np
from ultralytics import YOLO
from app.inference.ocr import extract_container_code
import app.inference.ocr as ocr
assert ocr.OCR_RECOGNIZER == 'trocr'

TEST_DIR = 'container.v2i.yolov11.resplit/test/images'

TARGETS = [
    {'filename': 'Event20231020100456005-Avi_3965_jpg.rf.632625046b01e7ea5311c4298ee5cab3.jpg',
     'box_index': 0, 'expected_code': 'BMOU8359431'},
    {'filename': 'IMG_0960_region_JPG_jpg.rf.db49285e93ebf8a15860ab4c8b339110.jpg',
     'box_index': 0, 'expected_code': 'ZCSU3272002'},
    {'filename': 'z4944210696040_9b9a08ff91d574e1a136ed57f30341d7_jpg.rf.0b1a868a89dd112422b86a0045d9257e.jpg',
     'box_index': 0, 'expected_code': 'TCLU8662927'},
]

print("Loading TrOCR + YOLO...", flush=True)
model = YOLO('models/best_resplit.pt')
print("Ready.", flush=True)

all_pass = True
for t in TARGETS:
    path = os.path.join(TEST_DIR, t['filename'])
    with open(path, 'rb') as f:
        image_bytes = f.read()
    img_pil = Image.open(path).convert('RGB')
    img_np = np.array(img_pil)
    yres = model(img_np, verbose=False)
    boxes = []
    for r in yres:
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            boxes.append((x1, y1, x2, y2))
    x1, y1, x2, y2 = boxes[t['box_index']]
    res = extract_container_code(image_bytes, x1, y1, x2, y2)

    same_code = res['container_code'] == t['expected_code']
    now_invalid = res['is_valid'] == False
    ok = same_code and now_invalid
    all_pass = all_pass and ok
    print(f"{t['filename']}", flush=True)
    print(f"  expected code (unchanged): {t['expected_code']}  got: {res['container_code']}  match={same_code}", flush=True)
    print(f"  is_valid: {res['is_valid']}  (expected False)  reason: {res['validation_reason']}", flush=True)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}", flush=True)
    print(flush=True)

print(f"=== OVERALL: {'ALL 3 PASS' if all_pass else 'FAILURE -- see above'} ===", flush=True)
