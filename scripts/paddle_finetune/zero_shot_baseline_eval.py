"""
Zero-shot (non-fine-tuned) PaddleX OCR raw exact-match baseline, measured on
the SAME 300-crop val split (data/paddle_finetune/dataset/val.txt) that the
PP-OCRv6_medium fine-tune was scored against (23.0% exact-match, epoch 150).

Deliberately mirrors production's zero-shot pipeline exactly:
- get_ocr() / run_ocr_on_crop() are imported unmodified from app/inference/ocr.py,
  so this uses the identical pdx.create_pipeline('OCR', device='cpu',
  engine_config={'paddle_static': {'enable_new_ir': False, 'run_mode': 'paddle'}})
  init and the identical predict()-under-lock call shape. No fine-tuned
  checkpoint is loaded anywhere in this script.
- NO cascade: none of _try_extract's Tier1/1b/2/3, no checksum recovery
  (validate_iso6346 is not called), no _CHAR_CONFUSION_MAP. Whatever
  run_ocr_on_crop returns is the answer.
- NO custom preprocessing (crop_detection padding, preprocess_crop contrast/
  sharpen/grayscale): the fine-tune's own val-set eval scored PaddleX's
  internal eval harness directly against the raw crop files in val.txt, with
  no such preprocessing applied. Running the zero-shot model on the same
  untouched crop bytes is what makes this apples-to-apples with the 23.0%
  figure; running it on our own custom-preprocessed version would not be.

"Raw predicted text" for a crop = every rec_texts entry PaddleX's detector
finds in that crop, concatenated with NO separator, in the order returned.
This is a real methodological wrinkle worth flagging up front: these crops
often contain more than just the container code (ISO size/type code, door/
weight placard digits, etc. each become their own detected text region), so
naive concatenation will often glue extra characters onto the code and fail
an exact-match even when the code itself was read correctly. That's not a
bug -- it's what "no merge logic" actually looks like -- but it means this
number is a floor, not a ceiling, on zero-shot quality. Per-row rec_texts
are preserved in the CSV so this can be inspected/re-sliced later without
re-running inference.

Usage: python3 scripts/paddle_finetune/zero_shot_baseline_eval.py
Writes: data/paddle_finetune/zero_shot_baseline/results.csv
"""
import csv
import os
import sys

from PIL import Image

REPO_ROOT = '/home/shashwatkumar/Testing/container_ocr_test'
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from app.inference.ocr import get_ocr, run_ocr_on_crop  # noqa: E402

DATASET_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/dataset')
VAL_TXT = os.path.join(DATASET_DIR, 'val.txt')
OUT_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/zero_shot_baseline')
OUT_CSV = os.path.join(OUT_DIR, 'results.csv')


def load_val_entries(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip('\n')
            if not line.strip():
                continue
            # PaddleOCR label format: image_path<TAB or space(s)>label
            if '\t' in line:
                img_path, label = line.split('\t', 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    raise ValueError(f"val.txt line {line_no} unparsable: {line!r}")
                img_path, label = parts
            entries.append((img_path.strip(), label.strip()))
    return entries


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    entries = load_val_entries(VAL_TXT)
    print(f"Loaded {len(entries)} val entries from {VAL_TXT}", flush=True)
    if len(entries) != 300:
        print(f"WARNING: expected exactly 300 lines, got {len(entries)}", flush=True)

    print("Initializing zero-shot PaddleX OCR pipeline (get_ocr(), no fine-tuned "
          "weights)...", flush=True)
    get_ocr()  # force init now so progress logging below is all inference time
    print("Pipeline ready.", flush=True)

    rows = []
    exact_matches = 0
    upper_matches = 0
    empty_predictions = 0
    non_ascii_predictions = 0

    for i, (rel_path, ground_truth) in enumerate(entries, 1):
        img_path = os.path.join(DATASET_DIR, rel_path)
        img = Image.open(img_path).convert('RGB')

        texts = run_ocr_on_crop(img)  # [(text, score), ...] in detection order
        predicted = ''.join(t for t, _ in texts)

        is_exact = predicted == ground_truth
        is_upper = predicted.upper() == ground_truth.upper()
        is_empty = len(texts) == 0
        is_non_ascii = any(ord(c) > 127 for c in predicted)

        exact_matches += is_exact
        upper_matches += is_upper
        empty_predictions += is_empty
        non_ascii_predictions += is_non_ascii

        rows.append({
            'image_path': rel_path,
            'ground_truth': ground_truth,
            'predicted': predicted,
            'exact_match': is_exact,
            'exact_match_case_insensitive': is_upper,
            'num_detected_regions': len(texts),
            'rec_texts_detail': ' | '.join(f"{t}({c:.2f})" for t, c in texts),
        })

        status = 'MATCH' if is_exact else ('match(ci)' if is_upper else 'miss')
        print(f"[{i}/{len(entries)}] {rel_path}: gt={ground_truth!r} pred={predicted!r} [{status}]",
              flush=True)

    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n = len(entries)
    print()
    print("=== ZERO-SHOT RAW BASELINE RESULT ===")
    print(f"Exact match (case-sensitive, as specified): {exact_matches}/{n} "
          f"({100 * exact_matches / n:.1f}%)")
    print(f"Exact match (case-insensitive, secondary):  {upper_matches}/{n} "
          f"({100 * upper_matches / n:.1f}%)")
    print(f"Empty predictions (no text detected):       {empty_predictions}/{n}")
    print(f"Non-ASCII predictions:                      {non_ascii_predictions}/{n}")
    print()
    print(f"Full per-row CSV: {OUT_CSV}")


if __name__ == '__main__':
    main()
