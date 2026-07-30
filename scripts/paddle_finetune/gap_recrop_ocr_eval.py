"""
Zero-shot OCR impact test for the gap-detection re-crop preprocessing built
by gap_recrop_build.py. Isolates ONE variable -- "does cropping off the
bleed column before recognition help the STOCK (non-fine-tuned)
PP-OCRv6_medium_rec recognizer" -- the same way zero_shot_rec_only_eval.py
isolated "recognition-only vs full detect+recognize pipeline" for the
original zero-shot-vs-fine-tuned comparison. Does NOT touch the fine-tuned
checkpoint or the fine-tuning pipeline.

Model init, resolved-weights assertions, and the norm_edit_dis metric are
copied verbatim from zero_shot_rec_only_eval.py so these numbers are
directly comparable to the 23.0% / 0.4923 figures already in the handoff
notes -- this is deliberately NOT a from-scratch reimplementation.

For every non-skipped crop in gap_recrop_manifest.csv (built by
gap_recrop_build.py, not recomputed here), runs the stock recognizer on
BOTH the original crop and its re-cropped version, against the same
ground truth. Ground truth lookup mirrors the notebook's Cell 5 fallback
(val.txt primary, since a chunk of tall-bucket paths were dropped when
val.txt was regenerated after zero_shot_rotation_test/results.csv --
see that CSV's own ground_truth column as fallback).

Reports two numbers, NOT one:
  - PRIMARY: the ~74 group-deduplicated representative crops (one per
    distinct source-video/container group) -- this is the trustworthy
    number, since no single heavily-duplicated video/container can
    dominate it.
  - SECONDARY/FULL: all 218 crops, kept only as a reference point to show
    how much (if at all) the per-crop framing differs from the
    per-container framing.

Does NOT modify tall_crop_bleed_labels.csv, gap_detection_results.csv,
gap_recrop_manifest.csv, or any existing contact sheet/notebook. Does NOT
touch the fine-tuned model or fine-tuning config.

Usage: python3 scripts/paddle_finetune/gap_recrop_ocr_eval.py
Writes: data/paddle_finetune/gap_recrop_ocr_results.csv
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = '/home/shashwatkumar/Testing/container_ocr_test'
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import paddlex  # noqa: E402

DATASET_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/dataset')
VAL_TXT = os.path.join(DATASET_DIR, 'val.txt')
ROTATION_RESULTS_CSV = os.path.join(
    REPO_ROOT, 'data/paddle_finetune/zero_shot_rotation_test/results.csv')
MANIFEST_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_manifest.csv')
RECROP_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_images')
OUT_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_ocr_results.csv')
MODEL_NAME = 'PP-OCRv6_medium_rec'


def load_val_entries(path):
    entries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            img_path, label = line.split('\t', 1) if '\t' in line else line.split(None, 1)
            entries[img_path.strip()] = label.strip()
    return entries


def build_ground_truth_lookup():
    val_entries = load_val_entries(VAL_TXT)
    rotation_gt = {}
    if os.path.exists(ROTATION_RESULTS_CSV):
        for r in csv.DictReader(open(ROTATION_RESULTS_CSV)):
            rotation_gt[r['image_path']] = r['ground_truth']

    def lookup(image_path):
        if image_path in val_entries:
            return val_entries[image_path], 'val.txt'
        if image_path in rotation_gt:
            return rotation_gt[image_path], 'rotation_results(STALE)'
        return None, None

    return lookup


def norm_edit_dis(pred, gt):
    """Copied from zero_shot_rec_only_eval.py so this metric matches the
    23.0% / 0.4923 fine-tuned comparison exactly."""
    try:
        import Levenshtein
        d = Levenshtein.distance(pred, gt)
    except ImportError:
        m, n = len(pred), len(gt)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                tmp = dp[j]
                dp[j] = prev if pred[i - 1] == gt[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
                prev = tmp
        d = dp[n]
    return 1 - d / max(len(pred), len(gt), 1)


def main():
    manifest = list(csv.DictReader(open(MANIFEST_CSV)))
    processed = [r for r in manifest if r['decision'] == 'ok']
    print(f"manifest rows: {len(manifest)}  processed (non-skipped): {len(processed)}")

    lookup_gt = build_ground_truth_lookup()
    missing_gt = [r['image_path'] for r in processed if lookup_gt(r['image_path'])[0] is None]
    if missing_gt:
        raise RuntimeError(f"{len(missing_gt)} processed crops have no ground truth "
                            f"in either val.txt or rotation results, e.g. {missing_gt[:3]}")

    print(f"Initializing STOCK {MODEL_NAME} recognizer (no detector, no fine-tuned "
          f"checkpoint)...", flush=True)
    model = paddlex.create_model(
        MODEL_NAME,
        device='cpu',
        engine_config={'paddle_static': {'enable_new_ir': False, 'run_mode': 'paddle'}},
    )
    resolved_name = getattr(model._predictor, 'model_name', '<unknown>')
    resolved_dir = getattr(model._predictor, 'model_dir', '<unknown>')
    print(f"Resolved predictor model_name={resolved_name!r} model_dir={resolved_dir!r}", flush=True)
    assert resolved_name == MODEL_NAME
    assert 'official_models' in str(resolved_dir)
    print("Confirmed: stock official pretrained weights.", flush=True)

    def predict_one(img_path):
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        r = list(model.predict(arr))[0]
        text = r.get('rec_text', '') or ''
        score = float(r.get('rec_score', 0.0) or 0.0)
        return text, score

    out_rows = []
    for i, r in enumerate(processed, 1):
        rel_path = r['image_path']
        gt, gt_source = lookup_gt(rel_path)

        orig_path = os.path.join(DATASET_DIR, rel_path)
        recrop_path = os.path.join(REPO_ROOT, 'data/paddle_finetune', r['recrop_path'])

        orig_pred, orig_score = predict_one(orig_path)
        recrop_pred, recrop_score = predict_one(recrop_path)

        out_rows.append({
            'image_path': rel_path,
            'group_id': r['group_id'],
            'is_group_representative': r['is_group_representative'],
            'ground_truth': gt,
            'ground_truth_source': gt_source,
            'orig_predicted': orig_pred,
            'orig_exact_match': orig_pred == gt,
            'orig_norm_edit_dis': round(norm_edit_dis(orig_pred, gt), 4),
            'orig_rec_score': round(orig_score, 4),
            'recrop_predicted': recrop_pred,
            'recrop_exact_match': recrop_pred == gt,
            'recrop_norm_edit_dis': round(norm_edit_dis(recrop_pred, gt), 4),
            'recrop_rec_score': round(recrop_score, 4),
        })

        if i % 25 == 0 or i == len(processed):
            print(f"  [{i}/{len(processed)}] {rel_path}: gt={gt!r} "
                  f"orig={orig_pred!r} recrop={recrop_pred!r}", flush=True)

    fieldnames = list(out_rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    def report(name, rows):
        n = len(rows)
        if n == 0:
            print(f"{name}: no rows")
            return
        orig_exact = sum(1 for r in rows if r['orig_exact_match'])
        recrop_exact = sum(1 for r in rows if r['recrop_exact_match'])
        orig_ned = sum(r['orig_norm_edit_dis'] for r in rows) / n
        recrop_ned = sum(r['recrop_norm_edit_dis'] for r in rows) / n
        print(f"\n=== {name} (n={n}) ===")
        print(f"  ORIGINAL:  exact_match {orig_exact}/{n} ({100*orig_exact/n:.1f}%)  "
              f"norm_edit_dis {orig_ned:.4f}")
        print(f"  RECROPPED: exact_match {recrop_exact}/{n} ({100*recrop_exact/n:.1f}%)  "
              f"norm_edit_dis {recrop_ned:.4f}")
        print(f"  DELTA:     exact_match {recrop_exact-orig_exact:+d} "
              f"({100*(recrop_exact-orig_exact)/n:+.1f}pp)  "
              f"norm_edit_dis {recrop_ned-orig_ned:+.4f}")

    primary_rows = [r for r in out_rows if r['is_group_representative'] == 'True']
    report("PRIMARY (group-deduplicated)", primary_rows)
    report("SECONDARY/FULL (all processed crops)", out_rows)

    print(f"\nFull per-row CSV: {OUT_CSV}")


if __name__ == '__main__':
    main()
