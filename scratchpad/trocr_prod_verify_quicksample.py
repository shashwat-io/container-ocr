"""
Production-path TrOCR verification: same harness as noop_verify_quicksample.py
(direct extract_container_code() calls, not the benchmark script's monkeypatch),
but with OCR_RECOGNIZER=trocr and FINETUNED_TROCR_MODEL_DIR set, so this
exercises the real dispatch branch the integration patch added to
run_ocr_on_crop -- including real rotation retries, tier handling, and the
docpp-off retry skip -- rather than benchmark_trocr_cascade_quicksample.py's
patched run_ocr_on_crop.
"""
import os, sys, csv

os.environ['OCR_RECOGNIZER'] = 'trocr'
os.environ['FINETUNED_TROCR_MODEL_DIR'] = os.path.expanduser('~/Downloads/trocr_checkpoint_thin')

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from PIL import Image
import numpy as np
from ultralytics import YOLO
from app.inference.ocr import extract_container_code
import app.inference.ocr as ocr
assert ocr.OCR_RECOGNIZER == 'trocr', f"expected trocr, got {ocr.OCR_RECOGNIZER!r}"

TEST_DIR = 'container.v2i.yolov11.resplit/test/images'
SAMPLE_FILE = 'quick_sample_set_resplit.txt'

with open(SAMPLE_FILE) as f:
    sample_names = {l.strip() for l in f if l.strip() and not l.startswith('#')}

images = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
images = [f for f in images if f in sample_names]
print(f"images: {len(images)}", flush=True)

model = YOLO('models/best_resplit.pt')

rows = []
for i, fname in enumerate(images, 1):
    path = os.path.join(TEST_DIR, fname)
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
    if not boxes:
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                      'container_code': '', 'is_valid': False, 'validation_reason': 'No detection',
                      'ocr_confidence': 0.0, 'ocr_raw': ''})
    else:
        for x1, y1, x2, y2 in boxes:
            res = extract_container_code(image_bytes, x1, y1, x2, y2)
            rows.append({'filename': fname, 'detected': True, 'num_boxes': len(boxes),
                          'container_code': res['container_code'], 'is_valid': res['is_valid'],
                          'validation_reason': res['validation_reason'],
                          'ocr_confidence': res['ocr_confidence'], 'ocr_raw': res['ocr_raw_output']})
    if i % 25 == 0 or i == len(images):
        print(f"  {i}/{len(images)}", flush=True)

out = 'scratchpad/trocr_prod_verify_quicksample.csv'
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['filename', 'detected', 'num_boxes', 'container_code',
                                       'is_valid', 'validation_reason', 'ocr_confidence', 'ocr_raw'])
    w.writeheader(); w.writerows(rows)

valid = sum(1 for r in rows if r['is_valid'])
print(f"Wrote {out} ({len(rows)} rows) | valid: {valid}/{len(rows)} = {100*valid/len(rows):.1f}%")
