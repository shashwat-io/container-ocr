"""
Recrop-alone (bleed removal, NO rotation) vs. original, against the fine-tuned
PP-OCRv6_medium_rec checkpoint, on the same 65-crop group-deduplicated primary
set used throughout this line of testing. Closes out the last untested variant:
rotate-alone and recrop-then-rotate both collapsed against this checkpoint
(0/65 exact each, vs. 31/65 for untouched original) because the fine-tuned
model was trained directly on bleed-contaminated, vertically-stacked crops and
rotation pushes it out of that learned orientation. Recrop-alone keeps the
native orientation and only removes the bleed column, so it isn't subject to
that same failure mode -- this is a fair, still-open question.

Reuses, does NOT regenerate or rerun:
  - The existing recrop images in data/paddle_finetune/gap_recrop_images/,
    built by gap_recrop_build.py using the column_variance +
    column_edge_density agreement rule (verified here to cover all 65 of these
    crops already -- decision == 'ok' and is_group_representative == 'True' for
    all 65, confirmed before running).
  - The "original" fine-tuned predictions/metrics from
    gap_recrop_finetuned_results.csv (31/65 exact, norm_edit_dis 0.6668) --
    not rerun, since nothing about the original-crop input changes between
    that test and this one.

Only new inference performed here: fine-tuned checkpoint on the 65 existing
recrop-alone images.

Checkpoint used (same one confirmed in gap_recrop_finetuned_eval.py):
/home/shashwatkumar/paddle_backup_inspect/checkpoints_backup/best_accuracy/inference
(best_accuracy.info.json: acc=0.23, norm_edit_dis=0.4923, best_epoch=150 --
matches the cited 23.0%/0.4923 figures exactly).

Does NOT touch the fine-tuning pipeline, existing CSVs, or prior scripts/images.

Usage: python3 scripts/paddle_finetune/gap_recrop_finetuned_norotate_eval.py
Writes: data/paddle_finetune/gap_recrop_finetuned_norotate_results.csv
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

MANIFEST_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_manifest.csv')
FINETUNED_RESULTS_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_finetuned_results.csv')
OUT_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_finetuned_norotate_results.csv')

FINETUNED_MODEL_DIR = '/home/shashwatkumar/paddle_backup_inspect/checkpoints_backup/best_accuracy/inference'
MODEL_NAME = 'PP-OCRv6_medium_rec'


def norm_edit_dis(pred, gt):
    """Copied verbatim from the prior eval scripts in this pipeline."""
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
    finetuned_rows = list(csv.DictReader(open(FINETUNED_RESULTS_CSV)))
    manifest = {r['image_path']: r for r in csv.DictReader(open(MANIFEST_CSV))}
    print(f"primary crops reused from {FINETUNED_RESULTS_CSV}: {len(finetuned_rows)}")

    for r in finetuned_rows:
        m = manifest[r['image_path']]
        assert m['decision'] == 'ok' and m['is_group_representative'] == 'True', (
            f"{r['image_path']} not a processed group-representative recrop")
        recrop_full = os.path.join(REPO_ROOT, 'data/paddle_finetune', m['recrop_path'])
        assert os.path.exists(recrop_full), f"missing recrop image: {recrop_full}"

    print(f"Initializing FINE-TUNED {MODEL_NAME} from {FINETUNED_MODEL_DIR} ...", flush=True)
    model = paddlex.create_model(
        MODEL_NAME,
        model_dir=FINETUNED_MODEL_DIR,
        device='cpu',
        engine_config={'paddle_static': {'enable_new_ir': False, 'run_mode': 'paddle'}},
    )
    resolved_name = getattr(model._predictor, 'model_name', '<unknown>')
    resolved_dir = str(getattr(model._predictor, 'model_dir', '<unknown>'))
    assert resolved_name == MODEL_NAME
    assert resolved_dir == FINETUNED_MODEL_DIR, (
        f"FATAL: resolved model_dir {resolved_dir!r} != requested fine-tuned checkpoint")
    print(f"Confirmed: fine-tuned checkpoint at {resolved_dir}", flush=True)

    def predict_one(img_path):
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        r = list(model.predict(arr))[0]
        return r.get('rec_text', '') or '', float(r.get('rec_score', 0.0) or 0.0)

    out_rows = []
    for i, r in enumerate(finetuned_rows, 1):
        rel_path = r['image_path']
        gt = r['ground_truth']
        m = manifest[rel_path]
        recrop_full = os.path.join(REPO_ROOT, 'data/paddle_finetune', m['recrop_path'])

        rc_pred, rc_score = predict_one(recrop_full)
        rc_exact = rc_pred == gt
        rc_ned = norm_edit_dis(rc_pred, gt)

        out_rows.append({
            'image_path': rel_path,
            'group_id': r['group_id'],
            'ground_truth': gt,
            'orig_predicted': r['orig_predicted'],
            'orig_exact_match': r['orig_exact_match'],
            'orig_norm_edit_dis': r['orig_norm_edit_dis'],
            'recrop_alone_predicted': rc_pred,
            'recrop_alone_exact_match': rc_exact,
            'recrop_alone_norm_edit_dis': round(rc_ned, 4),
        })

        if i % 15 == 0 or i == len(finetuned_rows):
            print(f"  [{i}/{len(finetuned_rows)}] {rel_path}: gt={gt!r} "
                  f"orig={r['orig_predicted']!r} recrop_alone={rc_pred!r}", flush=True)

    fieldnames = list(out_rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    n = len(out_rows)
    orig_exact = sum(1 for r in out_rows if r['orig_exact_match'] == 'True')
    orig_ned = sum(float(r['orig_norm_edit_dis']) for r in out_rows) / n
    rc_exact = sum(1 for r in out_rows if r['recrop_alone_exact_match'])
    rc_ned = sum(r['recrop_alone_norm_edit_dis'] for r in out_rows) / n

    print(f"\n=== FINE-TUNED MODEL: recrop-alone vs. original (n={n}) ===")
    print(f"  original:      exact_match {orig_exact}/{n} ({100*orig_exact/n:.1f}%)  norm_edit_dis {orig_ned:.4f}")
    print(f"  recrop_alone:  exact_match {rc_exact}/{n} ({100*rc_exact/n:.1f}%)  norm_edit_dis {rc_ned:.4f}")
    print(f"  DELTA:         exact_match {rc_exact-orig_exact:+d} ({100*(rc_exact-orig_exact)/n:+.1f}pp)  "
          f"norm_edit_dis {rc_ned-orig_ned:+.4f}")

    improved = worse = unchanged = 0
    for r in out_rows:
        a, b = r['recrop_alone_norm_edit_dis'], float(r['orig_norm_edit_dis'])
        if a > b:
            improved += 1
        elif a < b:
            worse += 1
        else:
            unchanged += 1

    print(f"\n=== recrop_alone vs. original, per-crop norm_edit_dis (n={n}) ===")
    print(f"  improved: {improved}/{n}  ({100*improved/n:.1f}%)")
    print(f"  worse:    {worse}/{n}  ({100*worse/n:.1f}%)")
    print(f"  unchanged:{unchanged}/{n}  ({100*unchanged/n:.1f}%)")

    print(f"\nFull per-row CSV: {OUT_CSV}")


if __name__ == '__main__':
    main()
