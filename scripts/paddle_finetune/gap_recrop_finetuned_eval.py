"""
Same recrop-then-rotate vs. rotate-alone vs. original three-way comparison as
gap_recrop_rotate_eval.py, but against the ACTUAL FINE-TUNED PP-OCRv6_medium_rec
checkpoint instead of the stock zero-shot model -- this is what actually matters
for production.

Checkpoint used (confirmed, not assumed): the checkpoint at
/home/shashwatkumar/paddle_backup_inspect/checkpoints_backup/best_accuracy/inference
-- NOT anything under this repo's own data/paddle_finetune/training_output(_local)/,
neither of which is the epoch-150 run (training_output has no saved weights at all;
training_output_local is a separate short 6-epoch run whose own
best_accuracy.info.json records acc=0.0, best_epoch=6 -- clearly not it). The backup
checkpoint's own best_accuracy.info.json records
{"acc": 0.2299..., "norm_edit_dis": 0.4923..., "best_epoch": 150}, an exact match to
the 23.0% / 0.4923 / epoch 150 figures already cited throughout this pipeline's
scripts, confirming this is the right one. Its inference.yml embeds the character
dict directly (37-char CTC dict, no external dict.txt dependency), so
paddlex.create_model(..., model_dir=<that inference dir>) is self-contained.

Reuses, does NOT regenerate:
  - The same 65-crop group-deduplicated primary set and ground_truth values from
    gap_recrop_rotate_results.csv (already resolved there against val.txt / rotation
    results fallback).
  - The existing recrop-then-rotate images in
    data/paddle_finetune/gap_recrop_rotate_images/recrop_then_rotate/ AS-IS -- these
    already had their rotation direction picked (by the zero-shot recognizer's own
    confidence, per zero_shot_rotation_test.py's tie-break rule) and cropped; this
    script only runs the fine-tuned recognizer on top, it does not re-crop or re-pick
    a rotation direction for this arm.
  - The rotation math (try both +90/-90, keep whichever the recognizer scores
    higher) and norm_edit_dis metric, copied verbatim from
    zero_shot_rotation_test.py / gap_recrop_rotate_eval.py, now driven by the
    fine-tuned model's own predict() and rec_score for the rotate-alone arm (its
    direction choice is independently made per-model, unlike the recrop-then-rotate
    arm which reuses the zero-shot-era choice).

Also re-confirms the checkpoint against the FULL 300-crop val.txt as a sanity check
against the cited 23.0%/0.4923 baseline (item 4 of the task) -- some drift is
expected since val.txt was regenerated after that baseline was originally scored
(documented staleness issue elsewhere in this pipeline), so a modest gap is a
finding to report, not a bug, as long as it isn't wildly off.

Does NOT touch the fine-tuning pipeline (no training/resuming). Does NOT modify any
existing CSV, script, or image from prior steps -- outputs go to new files only.

Usage: python3 scripts/paddle_finetune/gap_recrop_finetuned_eval.py
Writes: data/paddle_finetune/gap_recrop_finetuned_results.csv
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
ROTATE_RESULTS_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_rotate_results.csv')
RECROP_THEN_ROTATE_DIR = os.path.join(
    REPO_ROOT, 'data/paddle_finetune/gap_recrop_rotate_images/recrop_then_rotate')
OUT_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_finetuned_results.csv')

FINETUNED_MODEL_DIR = '/home/shashwatkumar/paddle_backup_inspect/checkpoints_backup/best_accuracy/inference'
MODEL_NAME = 'PP-OCRv6_medium_rec'
CITED_ACC = 0.23
CITED_NED = 0.4923
CITED_EPOCH = 150


def norm_edit_dis(pred, gt):
    """Copied verbatim from zero_shot_rec_only_eval.py / gap_recrop_rotate_eval.py."""
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


def main():
    primary = list(csv.DictReader(open(ROTATE_RESULTS_CSV)))
    print(f"group-deduplicated primary crops reused from {ROTATE_RESULTS_CSV}: {len(primary)}")

    print(f"Initializing FINE-TUNED {MODEL_NAME} from {FINETUNED_MODEL_DIR} ...", flush=True)
    model = paddlex.create_model(
        MODEL_NAME,
        model_dir=FINETUNED_MODEL_DIR,
        device='cpu',
        engine_config={'paddle_static': {'enable_new_ir': False, 'run_mode': 'paddle'}},
    )
    resolved_name = getattr(model._predictor, 'model_name', '<unknown>')
    resolved_dir = str(getattr(model._predictor, 'model_dir', '<unknown>'))
    print(f"Resolved predictor model_name={resolved_name!r} model_dir={resolved_dir!r}", flush=True)
    assert resolved_name == MODEL_NAME
    assert resolved_dir == FINETUNED_MODEL_DIR, (
        f"FATAL: resolved model_dir {resolved_dir!r} != requested fine-tuned "
        f"checkpoint dir -- would silently be scoring the wrong model.")
    print("Confirmed: fine-tuned checkpoint (epoch 150 best_accuracy), matching the "
          "cited 23.0%/0.4923 checkpoint's own recorded metrics.", flush=True)

    def predict_one(img_path):
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        r = list(model.predict(arr))[0]
        return r.get('rec_text', '') or '', float(r.get('rec_score', 0.0) or 0.0)

    def best_of_rotations(img):
        ccw = np.array(img.rotate(90, expand=True))
        cw = np.array(img.rotate(-90, expand=True))
        r_ccw = list(model.predict(ccw))[0]
        r_cw = list(model.predict(cw))[0]
        text_ccw, score_ccw = r_ccw.get('rec_text', '') or '', float(r_ccw.get('rec_score', 0.0) or 0.0)
        text_cw, score_cw = r_cw.get('rec_text', '') or '', float(r_cw.get('rec_score', 0.0) or 0.0)
        if score_ccw >= score_cw:
            return text_ccw, score_ccw, 'ccw'
        return text_cw, score_cw, 'cw'

    # --- Sanity check: full 300-crop val.txt vs. the cited 23.0%/0.4923 figure ---
    print("\nSanity check: scoring fine-tuned checkpoint on full 300-crop val.txt...", flush=True)
    val_entries = load_val_entries(VAL_TXT)
    exact = 0
    sum_ned = 0.0
    for rel_path, gt in val_entries.items():
        pred, _ = predict_one(os.path.join(DATASET_DIR, rel_path))
        exact += (pred == gt)
        sum_ned += norm_edit_dis(pred, gt)
    n_val = len(val_entries)
    val_acc = exact / n_val
    val_ned = sum_ned / n_val
    print(f"  current val.txt (n={n_val}): exact_match {exact}/{n_val} ({100*val_acc:.1f}%), "
          f"norm_edit_dis {val_ned:.4f}")
    print(f"  cited figure (checkpoint's own training-time eval): "
          f"{100*CITED_ACC:.1f}% exact, norm_edit_dis {CITED_NED:.4f} (epoch {CITED_EPOCH})")
    acc_gap_pp = 100 * (val_acc - CITED_ACC)
    print(f"  gap: {acc_gap_pp:+.1f}pp exact_match, {val_ned - CITED_NED:+.4f} norm_edit_dis")

    # --- Three-way comparison on the 65-crop group-deduplicated primary set ---
    out_rows = []
    for i, r in enumerate(primary, 1):
        rel_path = r['image_path']
        gt = r['ground_truth']
        crop_basename = os.path.basename(rel_path)

        orig_pred, orig_score = predict_one(os.path.join(DATASET_DIR, rel_path))
        orig_exact = orig_pred == gt
        orig_ned = norm_edit_dis(orig_pred, gt)

        orig_img = Image.open(os.path.join(DATASET_DIR, rel_path)).convert('RGB')
        rot_text, rot_score, rot_dir = best_of_rotations(orig_img)
        rot_exact = rot_text == gt
        rot_ned = norm_edit_dis(rot_text, gt)

        rr_pred, rr_score = predict_one(os.path.join(RECROP_THEN_ROTATE_DIR, crop_basename))
        rr_exact = rr_pred == gt
        rr_ned = norm_edit_dis(rr_pred, gt)

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
            'recrop_then_rotate_predicted': rr_pred,
            'recrop_then_rotate_exact_match': rr_exact,
            'recrop_then_rotate_norm_edit_dis': round(rr_ned, 4),
        })

        if i % 15 == 0 or i == len(primary):
            print(f"  [{i}/{len(primary)}] {rel_path}: gt={gt!r} orig={orig_pred!r} "
                  f"rotate_alone[{rot_dir}]={rot_text!r} recrop_then_rotate={rr_pred!r}", flush=True)

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

    print(f"\n=== FINE-TUNED MODEL: THREE-WAY COMPARISON (n={n}, group-deduplicated primary set) ===")
    summarize('original', 'orig_exact_match', 'orig_norm_edit_dis')
    summarize('rotate_alone', 'rotate_alone_exact_match', 'rotate_alone_norm_edit_dis')
    summarize('recrop_then_rotate', 'recrop_then_rotate_exact_match', 'recrop_then_rotate_norm_edit_dis')

    def breakdown(name, other_key):
        improved = worse = unchanged = 0
        for r in out_rows:
            a, b = r['recrop_then_rotate_norm_edit_dis'], r[other_key]
            if a > b:
                improved += 1
            elif a < b:
                worse += 1
            else:
                unchanged += 1
        print(f"\n=== recrop_then_rotate vs. {name}, per-crop norm_edit_dis (n={n}) ===")
        print(f"  improved: {improved}/{n}  ({100*improved/n:.1f}%)")
        print(f"  worse:    {worse}/{n}  ({100*worse/n:.1f}%)")
        print(f"  unchanged:{unchanged}/{n}  ({100*unchanged/n:.1f}%)")

    breakdown('rotate_alone', 'rotate_alone_norm_edit_dis')
    breakdown('original', 'orig_norm_edit_dis')

    print(f"\nFull per-row CSV: {OUT_CSV}")


if __name__ == '__main__':
    main()
