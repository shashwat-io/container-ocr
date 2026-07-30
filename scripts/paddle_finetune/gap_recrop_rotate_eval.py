"""
Three-way zero-shot comparison on the SAME 65-crop group-deduplicated
primary set used by gap_recrop_ocr_eval.py: original vs. rotate-alone vs.
recrop-then-rotate. Tests whether removing the bleed column BEFORE rotating
avoids the "re-orienting a multi-region crop" failure mode that
zero_shot_rotation_test.py's own mixed result (58 improved / 80 worse / 111
unchanged, on the full 249-crop tall bucket) was root-caused to.

Does NOT re-derive the group-deduplicated set or the cut points -- both are
read straight from gap_recrop_manifest.csv (built by gap_recrop_build.py),
filtered to is_group_representative == 'True' and decision == 'ok' (65 rows,
confirmed against gap_recrop_ocr_results.csv's own primary-row count).

Does NOT re-run inference for the "original" variant -- those predictions
already exist, unchanged, in gap_recrop_ocr_results.csv's orig_* columns for
these same 65 rows; reusing them avoids redundant CPU inference and any risk
of drift between this script's number and the already-reported recrop-alone
number.

The rotation logic (try both +90/-90, keep whichever the recognizer itself
scores higher) is copied verbatim from zero_shot_rotation_test.py's
best_of_rotations() closure -- it can't be imported directly since it's a
closure over that script's local `model`, so this is a straight copy of the
same six lines, not a reimplementation of new rotation logic.

Model init and norm_edit_dis (a similarity score, 1.0 = perfect match, same
convention already used throughout this pipeline) are copied verbatim from
zero_shot_rec_only_eval.py / gap_recrop_ocr_eval.py for the same reason: so
all three numbers are directly comparable.

Does NOT touch the fine-tuning pipeline, tall_crop_bleed_labels.csv,
gap_detection_results.csv, gap_recrop_manifest.csv, gap_recrop_ocr_results.csv,
or the existing gap_recrop_images/ outputs.

Usage: python3 scripts/paddle_finetune/gap_recrop_rotate_eval.py
Writes:
  data/paddle_finetune/gap_recrop_rotate_images/rotate_alone/<crop>.jpg
  data/paddle_finetune/gap_recrop_rotate_images/recrop_then_rotate/<crop>.jpg
  data/paddle_finetune/gap_recrop_rotate_results.csv
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
MANIFEST_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_manifest.csv')
PRIOR_OCR_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_ocr_results.csv')
RECROP_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_images')
OUT_ROTATE_ALONE_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_rotate_images/rotate_alone')
OUT_RECROP_ROTATE_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_rotate_images/recrop_then_rotate')
OUT_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_rotate_results.csv')
MODEL_NAME = 'PP-OCRv6_medium_rec'


def norm_edit_dis(pred, gt):
    """Copied verbatim from zero_shot_rec_only_eval.py / gap_recrop_ocr_eval.py."""
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
    manifest = {r['image_path']: r for r in csv.DictReader(open(MANIFEST_CSV))}
    prior = list(csv.DictReader(open(PRIOR_OCR_CSV)))
    primary_prior = [r for r in prior if r['is_group_representative'] == 'True']
    print(f"primary (group-deduplicated) rows reused from {PRIOR_OCR_CSV}: {len(primary_prior)}")

    for r in primary_prior:
        mrow = manifest[r['image_path']]
        assert mrow['decision'] == 'ok', f"{r['image_path']} not processed in manifest"

    os.makedirs(OUT_ROTATE_ALONE_DIR, exist_ok=True)
    os.makedirs(OUT_RECROP_ROTATE_DIR, exist_ok=True)

    print(f"Initializing STOCK {MODEL_NAME} recognizer...", flush=True)
    model = paddlex.create_model(
        MODEL_NAME,
        device='cpu',
        engine_config={'paddle_static': {'enable_new_ir': False, 'run_mode': 'paddle'}},
    )
    resolved_name = getattr(model._predictor, 'model_name', '<unknown>')
    resolved_dir = getattr(model._predictor, 'model_dir', '<unknown>')
    assert resolved_name == MODEL_NAME
    assert 'official_models' in str(resolved_dir)
    print(f"Confirmed stock official weights ({resolved_dir}).", flush=True)

    def predict_one(arr):
        r = list(model.predict(arr))[0]
        return r.get('rec_text', '') or '', float(r.get('rec_score', 0.0) or 0.0)

    def best_of_rotations(img):
        """Copied verbatim (rotation math + tie-break rule) from
        zero_shot_rotation_test.py's best_of_rotations()."""
        ccw = np.array(img.rotate(90, expand=True))
        cw = np.array(img.rotate(-90, expand=True))
        text_ccw, score_ccw = predict_one(ccw)
        text_cw, score_cw = predict_one(cw)
        if score_ccw >= score_cw:
            return text_ccw, score_ccw, 'ccw', ccw
        return text_cw, score_cw, 'cw', cw

    out_rows = []
    for i, r in enumerate(primary_prior, 1):
        rel_path = r['image_path']
        gt = r['ground_truth']
        crop_basename = os.path.basename(rel_path)

        # (a) original -- reused, not recomputed
        orig_pred = r['orig_predicted']
        orig_exact = r['orig_exact_match'] == 'True'
        orig_ned = float(r['orig_norm_edit_dis'])

        # (b) rotate-alone
        orig_img = Image.open(os.path.join(DATASET_DIR, rel_path)).convert('RGB')
        rot_text, rot_score, rot_dir, rot_arr = best_of_rotations(orig_img)
        rot_exact = rot_text == gt
        rot_ned = norm_edit_dis(rot_text, gt)
        Image.fromarray(rot_arr).save(os.path.join(OUT_ROTATE_ALONE_DIR, crop_basename))

        # (c) recrop-then-rotate
        recrop_img = Image.open(os.path.join(RECROP_DIR, crop_basename)).convert('RGB')
        rr_text, rr_score, rr_dir, rr_arr = best_of_rotations(recrop_img)
        rr_exact = rr_text == gt
        rr_ned = norm_edit_dis(rr_text, gt)
        Image.fromarray(rr_arr).save(os.path.join(OUT_RECROP_ROTATE_DIR, crop_basename))

        out_rows.append({
            'image_path': rel_path,
            'group_id': r['group_id'],
            'ground_truth': gt,
            'orig_predicted': orig_pred,
            'orig_exact_match': orig_exact,
            'orig_norm_edit_dis': round(orig_ned, 4),
            'rotate_alone_predicted': rot_text,
            'rotate_alone_direction': rot_dir,
            'rotate_alone_exact_match': rot_exact,
            'rotate_alone_norm_edit_dis': round(rot_ned, 4),
            'recrop_then_rotate_predicted': rr_text,
            'recrop_then_rotate_direction': rr_dir,
            'recrop_then_rotate_exact_match': rr_exact,
            'recrop_then_rotate_norm_edit_dis': round(rr_ned, 4),
        })

        if i % 15 == 0 or i == len(primary_prior):
            print(f"  [{i}/{len(primary_prior)}] {rel_path}: gt={gt!r} "
                  f"orig={orig_pred!r} rotate_alone[{rot_dir}]={rot_text!r} "
                  f"recrop_then_rotate[{rr_dir}]={rr_text!r}", flush=True)

    fieldnames = list(out_rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n = len(out_rows)

    def summarize(name, exact_key, ned_key):
        exact = sum(1 for r in out_rows if r[exact_key])
        ned = sum(r[ned_key] for r in out_rows) / n
        print(f"  {name:<20} exact_match {exact}/{n} ({100*exact/n:.1f}%)  norm_edit_dis {ned:.4f}")

    print(f"\n=== THREE-WAY COMPARISON (n={n}, group-deduplicated primary set) ===")
    summarize('original', 'orig_exact_match', 'orig_norm_edit_dis')
    summarize('rotate_alone', 'rotate_alone_exact_match', 'rotate_alone_norm_edit_dis')
    summarize('recrop_then_rotate', 'recrop_then_rotate_exact_match', 'recrop_then_rotate_norm_edit_dis')

    improved = worse = unchanged = 0
    for r in out_rows:
        a, b = r['recrop_then_rotate_norm_edit_dis'], r['rotate_alone_norm_edit_dis']
        if a > b:
            improved += 1
        elif a < b:
            worse += 1
        else:
            unchanged += 1

    print(f"\n=== recrop_then_rotate vs. rotate_alone, per-crop norm_edit_dis (n={n}) ===")
    print(f"  improved: {improved}/{n}  ({100*improved/n:.1f}%)")
    print(f"  worse:    {worse}/{n}  ({100*worse/n:.1f}%)")
    print(f"  unchanged:{unchanged}/{n}  ({100*unchanged/n:.1f}%)")

    print(f"\nFull per-row CSV: {OUT_CSV}")
    print(f"Rotate-alone images: {OUT_ROTATE_ALONE_DIR}")
    print(f"Recrop-then-rotate images: {OUT_RECROP_ROTATE_DIR}")


if __name__ == '__main__':
    main()
