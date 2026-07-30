"""
Scoped, zero-shot, CPU-only, diagnostic-only experiment: does unsharp masking
help the STOCK (non-fine-tuned) PP-OCRv6_medium_rec recognizer read genuinely
blurry crops? Does NOT touch the fine-tuned checkpoint, the running Colab
training, or the running benchmark_full_live_resplit.py process -- this is a
fully independent, small-sample, CPU-fine measurement to decide whether the
idea is worth pursuing further, not a pipeline change.

Sample selection (from data/paddle_finetune/blur_measurements.csv, produced
by measure_motion_blur.py):
  1. Filter to laplacian_var < 150 (the "below 150" band -- 176 images).
  2. Recover each crop's original source-image basename (crop filenames are
     "crop_{i:04d}_{original_stem}_box{box_index}.jpg" per extract_crops.py)
     and group with video_safety.group_id() -- the same repeat-frame-safe
     grouping extract_crops.py itself uses -- to avoid the same repeat-frame
     inflation the worst-15 list in measure_motion_blur.py showed (several of
     those were the same source frame at multiple crop indices).
  3. Pick exactly 1 crop per group: the lowest laplacian_var in that group.
  4. Take up to 20 of the deduped, sorted-by-blurriest crops.

Preprocessing (identical params to the manual test done earlier today; NOT
Wiener/blind deconvolution -- that showed clear ringing artifacts and isn't
being tested further right now):
    gray = grayscale of the crop
    blurred = cv2.GaussianBlur(gray, (0,0), 3)
    sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
addWeighted requires matching channel counts, so both sides of the formula
are single-channel gray -- "original" in the spec means the grayscale
original, not the raw BGR crop. The sharpened single-channel result is
replicated to 3 channels only when handing it to the recognizer (which
expects an RGB-shaped array), exactly analogous to how the untouched
original crop is loaded via PIL .convert('RGB') for its own prediction.

Model loading mirrors zero_shot_rec_only_eval.py exactly: stock
PP-OCRv6_medium_rec, recognizer only (no detector), device='cpu', with the
same resolved-model-identity assertions, so this is the same zero-shot
baseline methodology as the earlier gap-detection investigation applied to a
different, blur-targeted sample.

Usage: python3 scripts/paddle_finetune/unsharp_zero_shot_eval.py
Writes: data/paddle_finetune/unsharp_zero_shot/results.csv
"""
import csv
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = '/home/shashwatkumar/Testing/container_ocr_test'
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import paddlex  # noqa: E402
from scripts.paddle_finetune.video_safety import group_id  # noqa: E402

DATASET_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/dataset')
TRAIN_TXT = os.path.join(DATASET_DIR, 'train.txt')
VAL_TXT = os.path.join(DATASET_DIR, 'val.txt')
BLUR_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/blur_measurements.csv')
OUT_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/unsharp_zero_shot')
OUT_CSV = os.path.join(OUT_DIR, 'results.csv')
MODEL_NAME = 'PP-OCRv6_medium_rec'

LAPLACIAN_THRESHOLD = 150
MAX_SAMPLE_SIZE = 20

CROP_FILENAME_RE = re.compile(r'^crop_\d+_(.+)_box\d+\.jpg$')


def recover_source_basename(crop_filename):
    """'crop_1056_IMG_..._jpg.rf.<hash>_box0.jpg' -> 'IMG_..._jpg.rf.<hash>.jpg',
    i.e. undo extract_crops.py's f"crop_{i:04d}_{original_stem}_box{box_index}.jpg"
    naming so group_id() sees exactly what it was designed to group on (the
    original source-frame basename), not the crop index/box-index wrapper."""
    m = CROP_FILENAME_RE.match(crop_filename)
    if not m:
        raise ValueError(f"crop filename doesn't match expected pattern: {crop_filename!r}")
    return m.group(1) + '.jpg'


def load_label_lookup():
    """image_path (as it appears in train.txt/val.txt, matching blur_measurements.csv) -> ground truth label."""
    lookup = {}
    for split_file in (TRAIN_TXT, VAL_TXT):
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                img_path, label = line.split('\t', 1)
                lookup[img_path] = label
    return lookup


def select_sample():
    """Filter blur_measurements.csv to laplacian_var < threshold, dedup by
    group_id (lowest laplacian_var wins per group), return up to
    MAX_SAMPLE_SIZE rows sorted blurriest-first."""
    with open(BLUR_CSV, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        candidates = [
            {
                'image_path': row['image_path'],
                'laplacian_var': float(row['laplacian_var']),
            }
            for row in reader
            if float(row['laplacian_var']) < LAPLACIAN_THRESHOLD
        ]

    print(f"Candidates with laplacian_var < {LAPLACIAN_THRESHOLD}: {len(candidates)}")

    best_per_group = {}
    for c in candidates:
        basename = os.path.basename(c['image_path'])
        source_basename = recover_source_basename(basename)
        gid = group_id(source_basename)
        c['group_id'] = gid
        if gid not in best_per_group or c['laplacian_var'] < best_per_group[gid]['laplacian_var']:
            best_per_group[gid] = c

    deduped = sorted(best_per_group.values(), key=lambda c: c['laplacian_var'])
    print(f"Distinct groups (1 crop/group, lowest laplacian_var kept): {len(deduped)}")

    sample = deduped[:MAX_SAMPLE_SIZE]
    print(f"Sample size for this experiment: {len(sample)} (cap={MAX_SAMPLE_SIZE})")
    return sample


def unsharp_mask(gray):
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
    return sharpened


def predict(model, arr):
    result = list(model.predict(arr))
    r = result[0]
    predicted = r.get('rec_text', '') or ''
    score = float(r.get('rec_score', 0.0) or 0.0)
    return predicted, score


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    label_lookup = load_label_lookup()
    sample = select_sample()

    print(f"\nInitializing STOCK {MODEL_NAME} recognizer (no detector, no fine-tuned "
          f"checkpoint, device=cpu)...", flush=True)
    model = paddlex.create_model(
        MODEL_NAME,
        device='cpu',
        engine_config={'paddle_static': {'enable_new_ir': False, 'run_mode': 'paddle'}},
    )
    resolved_name = getattr(model._predictor, 'model_name', '<unknown>')
    resolved_dir = getattr(model._predictor, 'model_dir', '<unknown>')
    print(f"Resolved predictor model_name={resolved_name!r} model_dir={resolved_dir!r}", flush=True)
    assert resolved_name == MODEL_NAME, (
        f"FATAL: resolved model_name {resolved_name!r} != requested {MODEL_NAME!r}."
    )
    assert 'official_models' in str(resolved_dir), (
        f"FATAL: model_dir {resolved_dir!r} does not look like PaddleX's official "
        f"pretrained cache -- may be accidentally loading a fine-tuned checkpoint."
    )
    print("Confirmed: stock official pretrained weights (zero-shot, not the fine-tune).\n", flush=True)

    rows = []
    orig_matches = 0
    unsharp_matches = 0

    for i, c in enumerate(sample, 1):
        rel_path = c['image_path']
        abs_path = os.path.join(DATASET_DIR, rel_path)
        ground_truth = label_lookup.get(rel_path)
        if ground_truth is None:
            print(f"WARNING: no label found for {rel_path}, skipping", file=sys.stderr)
            continue

        orig_rgb = np.array(Image.open(abs_path).convert('RGB'))
        gray = cv2.cvtColor(cv2.imread(abs_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        sharpened_gray = unsharp_mask(gray)
        sharpened_rgb = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2RGB)

        orig_pred, orig_score = predict(model, orig_rgb)
        unsharp_pred, unsharp_score = predict(model, sharpened_rgb)

        orig_match = orig_pred == ground_truth
        unsharp_match = unsharp_pred == ground_truth
        orig_matches += orig_match
        unsharp_matches += unsharp_match

        if orig_match and unsharp_match:
            matched_which = 'both'
        elif orig_match:
            matched_which = 'original'
        elif unsharp_match:
            matched_which = 'unsharp'
        else:
            matched_which = 'neither'

        rows.append({
            'image_path': rel_path,
            'group_id': c['group_id'],
            'laplacian_var': round(c['laplacian_var'], 2),
            'ground_truth': ground_truth,
            'original_pred': orig_pred,
            'original_score': round(orig_score, 4),
            'original_match': orig_match,
            'unsharp_pred': unsharp_pred,
            'unsharp_score': round(unsharp_score, 4),
            'unsharp_match': unsharp_match,
            'matched_which': matched_which,
        })

        print(f"[{i}/{len(sample)}] {rel_path} (lap={c['laplacian_var']:.1f}) "
              f"gt={ground_truth!r} | orig={orig_pred!r} ({'MATCH' if orig_match else 'miss'}) "
              f"| unsharp={unsharp_pred!r} ({'MATCH' if unsharp_match else 'miss'})", flush=True)

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print()
    print("=== UNSHARP MASKING ZERO-SHOT RESULT (diagnostic only, no fix applied) ===")
    print(f"Model: {resolved_name} (stock, from {resolved_dir})")
    print(f"Sample size: {n} crops (laplacian_var < {LAPLACIAN_THRESHOLD}, 1 per source-video group)")
    print(f"Original exact match:  {orig_matches}/{n} ({100 * orig_matches / n:.1f}%)")
    print(f"Unsharp exact match:   {unsharp_matches}/{n} ({100 * unsharp_matches / n:.1f}%)")
    print()
    print("Per-crop breakdown:")
    print(f"{'image_path':60s} {'gt':14s} {'orig_pred':14s} {'unsharp_pred':14s} matched_which")
    for r in rows:
        print(f"{r['image_path']:60s} {r['ground_truth']:14s} {r['original_pred']:14s} "
              f"{r['unsharp_pred']:14s} {r['matched_which']}")
    print()
    print(f"Full CSV: {OUT_CSV}")


if __name__ == '__main__':
    main()
