"""
Diagnostic-only test (no retraining, no production changes): does rotating
vertically-stacked ("tall") crops 90 degrees before recognition improve the
STOCK PP-OCRv6_medium_rec's raw exact-match / norm_edit_dis, relative to the
zero_shot_rec_only baseline (tall bucket: 0/249 exact, norm_edit_dis 0.0456)?

Orientation is NOT globally consistent -- confirmed by manual inspection
before writing this script: rotate(90) (CCW) restores correct left-to-right
reading order for crop_1036 and crop_1546, but produces mirrored/backwards
text for crop_0224 (which needs the opposite direction instead). So this
script tries BOTH rotations per crop and keeps whichever the recognizer
itself is more confident about (higher rec_score) -- cheap, since it's just
one extra CPU inference per crop, and correct per-crop rather than a global
guess that would silently fail on however many crops use the other winding
direction.

Does NOT touch extract_crops.py, generate_labels.py, make_config.py, or any
dataset/images/ file. Rotated images only ever exist in memory (np arrays
fed straight to predict()) or, for the handful of side-by-side examples, as
throwaway files under the scratchpad dir.

Usage: python3 scripts/paddle_finetune/zero_shot_rotation_test.py
Writes: data/paddle_finetune/zero_shot_rotation_test/results.csv
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
BASELINE_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/zero_shot_rec_only/results.csv')
OUT_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/zero_shot_rotation_test')
OUT_CSV = os.path.join(OUT_DIR, 'results.csv')
MODEL_NAME = 'PP-OCRv6_medium_rec'


def load_val_entries(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            img_path, label = line.split('\t', 1) if '\t' in line else line.split(None, 1)
            entries.append((img_path.strip(), label.strip()))
    return entries


def norm_edit_dis(pred, gt):
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
    os.makedirs(OUT_DIR, exist_ok=True)

    entries = {rel: gt for rel, gt in load_val_entries(VAL_TXT)}
    baseline = {r['image_path']: r for r in csv.DictReader(open(BASELINE_CSV))}
    assert set(entries) == set(baseline), "val.txt and baseline CSV image sets don't match"

    def bucket(rel_path):
        with Image.open(os.path.join(DATASET_DIR, rel_path)) as im:
            w, h = im.size
        ar = w / h
        if ar < 1.0:
            return 'tall'
        elif ar < 2.0:
            return 'near_square'
        return 'wide'

    tall = [p for p in entries if bucket(p) == 'tall']
    near_square = [p for p in entries if bucket(p) == 'near_square']
    print(f"tall bucket: {len(tall)}  near_square bucket: {len(near_square)}", flush=True)

    print(f"Initializing STOCK {MODEL_NAME} recognizer...", flush=True)
    model = paddlex.create_model(
        MODEL_NAME,
        device='cpu',
        engine_config={'paddle_static': {'enable_new_ir': False, 'run_mode': 'paddle'}},
    )
    assert model._predictor.model_name == MODEL_NAME
    assert 'official_models' in str(model._predictor.model_dir)
    print("Confirmed stock official weights, same as zero_shot_rec_only run.", flush=True)

    def predict_one(arr):
        r = list(model.predict(arr))[0]
        return r.get('rec_text', '') or '', float(r.get('rec_score', 0.0) or 0.0)

    def best_of_rotations(rel_path):
        img = Image.open(os.path.join(DATASET_DIR, rel_path)).convert('RGB')
        ccw = np.array(img.rotate(90, expand=True))
        cw = np.array(img.rotate(-90, expand=True))
        text_ccw, score_ccw = predict_one(ccw)
        text_cw, score_cw = predict_one(cw)
        if score_ccw >= score_cw:
            return text_ccw, score_ccw, 'ccw', text_cw, score_cw
        return text_cw, score_cw, 'cw', text_ccw, score_ccw

    def run_bucket(name, paths):
        rows = []
        exact = 0
        sum_ned = 0.0
        for i, rel_path in enumerate(paths, 1):
            gt = entries[rel_path]
            base_pred = baseline[rel_path]['predicted']
            base_ned = float(baseline[rel_path]['norm_edit_dis'])
            base_exact = baseline[rel_path]['exact_match'] == 'True'

            pred, score, direction, other_text, other_score = best_of_rotations(rel_path)
            is_exact = pred == gt
            ned = norm_edit_dis(pred, gt)
            exact += is_exact
            sum_ned += ned

            rows.append({
                'image_path': rel_path,
                'bucket': name,
                'ground_truth': gt,
                'predicted_before': base_pred,
                'norm_edit_dis_before': round(base_ned, 4),
                'exact_match_before': base_exact,
                'predicted_after_rotation': pred,
                'rotation_direction_used': direction,
                'rec_score_after': round(score, 4),
                'norm_edit_dis_after': round(ned, 4),
                'exact_match_after': is_exact,
                'other_rotation_text': other_text,
                'other_rotation_score': round(other_score, 4),
            })
            print(f"[{name} {i}/{len(paths)}] {rel_path}: gt={gt!r} "
                  f"before={base_pred!r}(ned={base_ned:.3f}) "
                  f"after[{direction}]={pred!r}(ned={ned:.3f}, score={score:.3f})", flush=True)
        n = len(paths)
        return rows, exact, sum_ned, n

    all_rows = []
    print("\n=== TALL BUCKET (main event) ===", flush=True)
    tall_rows, tall_exact, tall_sum_ned, tall_n = run_bucket('tall', tall)
    all_rows += tall_rows

    print("\n=== NEAR-SQUARE BUCKET (side-check) ===", flush=True)
    ns_rows, ns_exact, ns_sum_ned, ns_n = run_bucket('near_square', near_square)
    all_rows += ns_rows

    fieldnames = list(all_rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    def bucket_before_stats(name, paths):
        exact = sum(1 for p in paths if baseline[p]['exact_match'] == 'True')
        ned = sum(float(baseline[p]['norm_edit_dis']) for p in paths) / len(paths)
        return exact, ned

    tall_before_exact, tall_before_ned = bucket_before_stats('tall', tall)
    ns_before_exact, ns_before_ned = bucket_before_stats('near_square', near_square)

    print()
    print("=== BEFORE / AFTER ROTATION RESULT ===")
    print(f"TALL bucket (n={tall_n}):")
    print(f"  BEFORE: {tall_before_exact}/{tall_n} exact ({100*tall_before_exact/tall_n:.1f}%), "
          f"norm_edit_dis {tall_before_ned:.4f}")
    print(f"  AFTER:  {tall_exact}/{tall_n} exact ({100*tall_exact/tall_n:.1f}%), "
          f"norm_edit_dis {tall_sum_ned/tall_n:.4f}")
    print()
    print(f"NEAR-SQUARE bucket (n={ns_n}):")
    print(f"  BEFORE: {ns_before_exact}/{ns_n} exact ({100*ns_before_exact/ns_n:.1f}%), "
          f"norm_edit_dis {ns_before_ned:.4f}")
    print(f"  AFTER:  {ns_exact}/{ns_n} exact ({100*ns_exact/ns_n:.1f}%), "
          f"norm_edit_dis {ns_sum_ned/ns_n:.4f}")
    print()
    print(f"Full per-row CSV: {OUT_CSV}")


if __name__ == '__main__':
    main()
