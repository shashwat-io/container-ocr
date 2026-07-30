"""
Standalone, offline side-by-side measurement: current PaddleX pipeline
(PP-OCRv6_medium det+rec, as used in production ocr.py) vs. a PP-OCRv5_server
det+rec pipeline, on the SAME 50 crops.

Scope, deliberately kept narrow (measurement, not integration):
- Bounding boxes come from the dataset's own ground-truth YOLO labels
  (test/labels/*.txt), NOT from re-running our YOLO detector or any OCR.
  This isolates the variable under test to "OCR engine quality" rather than
  mixing in our detector's own error rate.
- Crops are extracted with crop_detection() and preprocessed with
  preprocess_crop(), both imported directly from app/inference/ocr.py --
  not reimplemented, so there's no drift from what production actually feeds
  into OCR for a landscape-oriented crop. The +-90 degree portrait-rotation
  search (_best_portrait_rotation) is intentionally NOT applied here to keep
  this a simple, controlled, identical-treatment comparison for both
  pipelines -- results are lower than the live pipeline's numbers for
  portrait crops, but that's fine since it's applied equally to both sides.
- "Checksum-valid" here means: exact 11-char structural match (single line,
  or the merged-string sliding window), checksum verified via
  validate_iso6346 imported from ocr.py. Recovery/correction tiers
  (check-digit recovery, character-confusion correction) are intentionally
  NOT applied, because those are pipeline-level cleverness that would mask
  genuine differences in raw OCR read quality -- which is exactly what this
  script is trying to measure.

Does NOT modify ocr.py, does NOT touch the live Flask app, does NOT restart
anything. Builds a second, throwaway PP-OCRv5 pipeline instance purely for
this offline comparison.

Usage: python3 scripts/compare_ppocrv5.py
Writes:
  ppocrv5_compare_samples/  (the 50 sampled crop images, for eyeballing)
  ppocrv5_compare_results.csv
"""
import copy
import csv
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from app.inference.ocr import (
    crop_detection,
    preprocess_crop,
    clean_ocr_text,
    validate_iso6346,
    run_ocr_on_crop,
    _ocr_lock,
)

import paddlex
from paddlex.inference.pipelines import load_pipeline_config

TEST_IMAGES_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
TEST_LABELS_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/labels'
SAMPLE_DIR = '/home/shashwatkumar/Testing/container_ocr_test/ppocrv5_compare_samples'
OUTPUT_CSV = '/home/shashwatkumar/Testing/container_ocr_test/ppocrv5_compare_results.csv'
SAMPLE_SIZE = 50
SEED = 42
PPOCRV5_DET = 'PP-OCRv5_server_det'
PPOCRV5_REC = 'PP-OCRv5_server_rec'


def build_ppocrv5_pipeline():
    """Same OCR pipeline config as ocr.py::get_ocr(), with det/rec models
    swapped to PP-OCRv5_server. Everything else (doc preprocessor,
    textline orientation, engine config, device) matches production exactly,
    so the only variable under test is the det/rec model choice."""
    cfg = load_pipeline_config('OCR')
    cfg = copy.deepcopy(cfg)
    cfg['SubModules']['TextDetection']['model_name'] = PPOCRV5_DET
    cfg['SubModules']['TextRecognition']['model_name'] = PPOCRV5_REC
    return paddlex.create_pipeline(
        config=cfg,
        device='cpu',
        engine_config={'paddle_static': {'enable_new_ir': False, 'run_mode': 'paddle'}},
    )


def run_pipeline(pipeline, crop_np, lock=None):
    """Mirrors run_ocr_on_crop's predict-and-parse shape, for a given
    pipeline instance. lock is only needed for the shared production
    singleton; the throwaway PP-OCRv5 instance isn't shared, so no lock
    needed for it."""
    if lock is not None:
        with lock:
            result = list(pipeline.predict(crop_np))
    else:
        result = list(pipeline.predict(crop_np))
    texts = []
    for r in result:
        rec_texts = r.get('rec_texts', [])
        rec_scores = r.get('rec_scores', [])
        for text, score in zip(rec_texts, rec_scores):
            if text.strip():
                texts.append((text, float(score)))
    return texts


def best_exact_checksum_candidate(texts):
    """Minimal candidate selection: Tier1-equivalent (single-line exact
    11-char match) then Tier3-Pass1-equivalent (merged-string sliding
    11-char window), both checksum-verified via validate_iso6346. No
    check-digit recovery, no confusion correction -- deliberately, so
    recovery-tier cleverness doesn't mask raw OCR read quality.
    Returns (code, valid, best_raw_text_for_reporting).
    """
    if not texts:
        return '', False, ''

    for text, _ in texts:
        line = clean_ocr_text(text)
        if len(line) == 11 and line[:4].isalpha() and line[4:11].isdigit():
            valid, _ = validate_iso6346(line)
            if valid:
                return line, True, text

    full_text = clean_ocr_text(''.join(t for t, _ in texts))
    for start in range(len(full_text) - 10):
        candidate = full_text[start:start + 11]
        if len(candidate) < 11:
            break
        if not (candidate[:4].isalpha() and candidate[4:11].isdigit()):
            continue
        valid, _ = validate_iso6346(candidate)
        if valid:
            return candidate, True, full_text

    # No exact checksum-valid candidate found; report the best-format guess
    # (unvalidated) so there's still something to eyeball.
    best_code = ''
    for start in range(len(full_text) - 10):
        candidate = full_text[start:start + 11]
        if len(candidate) < 11:
            break
        if candidate[:4].isalpha() and candidate[4:11].isdigit():
            best_code = candidate
            break
    if not best_code:
        best_code = full_text[:11] if len(full_text) >= 11 else full_text
    return best_code, False, full_text


def yolo_label_to_box(label_path, img_w, img_h):
    line = open(label_path).read().strip().split('\n')[0]
    _, xc, yc, w, h = (float(v) for v in line.split())
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2


def main():
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    all_images = sorted(f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    # Only keep images that have a matching ground-truth label file.
    candidates = [f for f in all_images if os.path.exists(
        os.path.join(TEST_LABELS_DIR, os.path.splitext(f)[0] + '.txt'))]
    random.seed(SEED)
    sample = random.sample(candidates, SAMPLE_SIZE)

    print(f"Sampled {len(sample)} images (seed={SEED}) from {len(candidates)} labeled test images", flush=True)

    print("Building PP-OCRv5_server pipeline (may download weights on first use)...", flush=True)
    ppocrv5 = build_ppocrv5_pipeline()
    print("PP-OCRv5 pipeline ready.", flush=True)

    rows = []
    current_valid = 0
    v5_valid = 0
    disagreements = []

    for i, fname in enumerate(sample, 1):
        img_path = os.path.join(TEST_IMAGES_DIR, fname)
        label_path = os.path.join(TEST_LABELS_DIR, os.path.splitext(fname)[0] + '.txt')

        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size
        with open(img_path, 'rb') as f:
            image_bytes = f.read()

        x1, y1, x2, y2 = yolo_label_to_box(label_path, img_w, img_h)
        crop = crop_detection(image_bytes, x1, y1, x2, y2)
        crop_saved_path = os.path.join(SAMPLE_DIR, f"{i:02d}_{fname}")
        crop.save(crop_saved_path)

        processed = preprocess_crop(crop)
        processed_np = np.array(processed)

        # Current production pipeline (reused via run_ocr_on_crop + get_ocr(), unmodified).
        current_texts = run_ocr_on_crop(processed)
        current_code, current_ok, current_raw = best_exact_checksum_candidate(current_texts)
        current_conf = sum(c for _, c in current_texts) / len(current_texts) if current_texts else 0.0

        # PP-OCRv5_server pipeline (throwaway instance built above).
        v5_texts = run_pipeline(ppocrv5, processed_np)
        v5_code, v5_ok, v5_raw = best_exact_checksum_candidate(v5_texts)
        v5_conf = sum(c for _, c in v5_texts) / len(v5_texts) if v5_texts else 0.0

        if current_ok:
            current_valid += 1
        if v5_ok:
            v5_valid += 1

        disagreement = current_ok != v5_ok
        if disagreement:
            disagreements.append(fname)

        rows.append({
            'filename': fname,
            'crop_saved_as': os.path.basename(crop_saved_path),
            'current_ocr_raw': ' | '.join(f"{t}({c:.2f})" for t, c in current_texts),
            'current_code': current_code,
            'current_checksum_valid': current_ok,
            'current_avg_conf': round(current_conf, 4),
            'ppocrv5_ocr_raw': ' | '.join(f"{t}({c:.2f})" for t, c in v5_texts),
            'ppocrv5_code': v5_code,
            'ppocrv5_checksum_valid': v5_ok,
            'ppocrv5_avg_conf': round(v5_conf, 4),
            'disagreement': disagreement,
        })

        print(f"[{i}/{SAMPLE_SIZE}] {fname}: current={current_ok}({current_code!r}) "
              f"v5={v5_ok}({v5_code!r}){'  <-- DISAGREE' if disagreement else ''}", flush=True)

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=== SIDE-BY-SIDE RESULT (50-image sample) ===")
    print(f"Current pipeline (PP-OCRv6_medium): {current_valid}/{SAMPLE_SIZE} checksum-valid")
    print(f"PP-OCRv5_server pipeline:            {v5_valid}/{SAMPLE_SIZE} checksum-valid")
    print(f"Disagreements (one valid, other not): {len(disagreements)}")
    for fname in disagreements:
        r = next(r for r in rows if r['filename'] == fname)
        winner = 'current' if r['current_checksum_valid'] else 'ppocrv5'
        print(f"  {fname}: winner={winner} | current={r['current_code']!r}({r['current_checksum_valid']}) "
              f"v5={r['ppocrv5_code']!r}({r['ppocrv5_checksum_valid']})")
    print()
    print(f"Crops saved to: {SAMPLE_DIR}/")
    print(f"Full results:   {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
