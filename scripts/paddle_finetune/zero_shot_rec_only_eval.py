"""
Clean, task-shape-matched zero-shot baseline: the STOCK (non-fine-tuned)
PP-OCRv6_medium_rec RECOGNIZER ONLY -- no text detector, no multi-region
concatenation -- on the same 300-crop val split (data/paddle_finetune/dataset/
val.txt) the fine-tune was scored against (23.0% exact-match, norm_edit_dis
0.4923, epoch 150).

Why this run exists: zero_shot_baseline_eval.py (the previous run) used the
full 'OCR' pipeline (detection+recognition), which fragments a crop into
however many text regions the detector finds and forced a "concatenate
everything" comparison -- 0.3% exact match, but 18.7% of predictions
contained the correct code as a substring, i.e. that number mostly measured
detector fragmentation, not recognition quality. PaddleX's rec fine-tuning
(and its eval) never runs a detector at all: train.txt/val.txt feed whole
crop images directly into the recognizer, which assumes the entire image IS
one text line/field and produces exactly one string. This script reproduces
that same assumption for the zero-shot model, so the two numbers are
finally comparable on identical footing.

Base model confirmation (matters -- an accidental size-variant mismatch
would make this comparison wrong again):
- scripts/paddle_finetune/make_config.py's fine-tuning config uses
  `Global.model: PP-OCRv6_medium_rec` and
  `pretrain_weight_path: https://.../PP-OCRv6_medium_rec_pretrained.pdparams`
  -- i.e. fine-tuning started from the stock PP-OCRv6_medium_rec checkpoint.
- The backup checkpoint zips' own inference.yml (best_accuracy/inference/
  inference.yml) records `model_name: PP-OCRv6_medium_rec`, confirming the
  same tier end-to-end.
- production app/inference/ocr.py's pdx.create_pipeline('OCR', ...) also
  resolves to PP-OCRv6_medium under the hood (per AGENTS.md).
So `paddlex.create_model('PP-OCRv6_medium_rec', device='cpu')` below --
with NO model_dir override -- pulls PaddleX's official cached weights from
~/.paddlex/official_models/PP-OCRv6_medium_rec, i.e. exactly the same
starting checkpoint the fine-tune began from, confirmed by inspecting
predictor.model_name / predictor.model_dir at runtime (both printed below).

Caveat worth carrying into any read of the result: many of these crops are
NOT single text lines (container codes are often physically split across
two stacked lines, e.g. owner+serial on one row and check-digit/ISO type on
another). A recognition-only model with no detector has to resize the whole
multi-line crop into its expected single-line input shape, which can badly
garble multi-line content. That distortion applies identically to what the
fine-tune's own training/eval saw, though, so it's a fair, not a bugged,
comparison -- it is exactly what "fine-tuning learned to cope with this
crop shape and the stock model didn't" would look like if that's what
happened.

Usage: python3 scripts/paddle_finetune/zero_shot_rec_only_eval.py
Writes: data/paddle_finetune/zero_shot_rec_only/results.csv
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
OUT_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/zero_shot_rec_only')
OUT_CSV = os.path.join(OUT_DIR, 'results.csv')
MODEL_NAME = 'PP-OCRv6_medium_rec'


def load_val_entries(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if not line.strip():
                continue
            if '\t' in line:
                img_path, label = line.split('\t', 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise ValueError(f"val.txt line {line_no} unparsable: {line!r}")
                img_path, label = parts
            entries.append((img_path.strip(), label.strip()))
    return entries


def norm_edit_dis(pred, gt):
    """PaddleOCR-style similarity: 1 - Levenshtein(pred, gt) / max(len(pred), len(gt), 1).
    Matches the metric baked into best_accuracy.info.json (0.4923) so the
    two numbers are directly comparable."""
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

    entries = load_val_entries(VAL_TXT)
    print(f"Loaded {len(entries)} val entries from {VAL_TXT}", flush=True)
    if len(entries) != 300:
        print(f"WARNING: expected exactly 300 lines, got {len(entries)}", flush=True)

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
    assert resolved_name == MODEL_NAME, (
        f"FATAL: resolved model_name {resolved_name!r} != requested {MODEL_NAME!r} -- "
        f"comparison would be against the wrong base model variant."
    )
    assert 'official_models' in str(resolved_dir), (
        f"FATAL: model_dir {resolved_dir!r} does not look like PaddleX's official "
        f"pretrained cache -- may be accidentally loading a fine-tuned checkpoint."
    )
    print("Confirmed: stock official pretrained weights, matching the fine-tune's "
          "starting checkpoint tier exactly.", flush=True)

    rows = []
    exact_matches = 0
    empty_predictions = 0
    non_ascii_predictions = 0
    sum_norm_edit_dis = 0.0

    for i, (rel_path, ground_truth) in enumerate(entries, 1):
        img_path = os.path.join(DATASET_DIR, rel_path)
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)

        result = list(model.predict(arr))
        # Rec-only predictor: one input image -> one result dict -> one rec_text/rec_score.
        r = result[0]
        predicted = r.get('rec_text', '') or ''
        score = float(r.get('rec_score', 0.0) or 0.0)

        is_exact = predicted == ground_truth
        ned = norm_edit_dis(predicted, ground_truth)
        is_empty = len(predicted) == 0
        is_non_ascii = any(ord(c) > 127 for c in predicted)

        exact_matches += is_exact
        empty_predictions += is_empty
        non_ascii_predictions += is_non_ascii
        sum_norm_edit_dis += ned

        rows.append({
            'image_path': rel_path,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'exact_match': is_exact,
            'norm_edit_dis': round(ned, 4),
            'rec_score': round(score, 4),
        })

        status = 'MATCH' if is_exact else 'miss'
        print(f"[{i}/{len(entries)}] {rel_path}: gt={ground_truth!r} pred={predicted!r} "
              f"score={score:.3f} ned={ned:.3f} [{status}]", flush=True)

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(entries)
    print()
    print("=== ZERO-SHOT REC-ONLY BASELINE RESULT ===")
    print(f"Model: {resolved_name} (stock, from {resolved_dir})")
    print(f"Exact match: {exact_matches}/{n} ({100 * exact_matches / n:.1f}%)")
    print(f"Average norm_edit_dis: {sum_norm_edit_dis / n:.4f}")
    print(f"Empty predictions: {empty_predictions}/{n}")
    print(f"Non-ASCII predictions: {non_ascii_predictions}/{n}")
    print()
    print("For reference -- fine-tuned PP-OCRv6_medium_rec (epoch 150): "
          "23.0% exact match, norm_edit_dis 0.4923")
    print()
    print(f"Full per-row CSV: {OUT_CSV}")


if __name__ == '__main__':
    main()
