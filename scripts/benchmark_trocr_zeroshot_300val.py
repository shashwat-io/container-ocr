"""
Zero-shot TrOCR (microsoft/trocr-base-printed) baseline over the same 300
ground-truth val crops used for the PaddleOCR zero-shot/fine-tuned isolated-
crop evals (data/paddle_finetune/dataset/val.txt + dataset/images/).

Isolated-crop eval: ground-truth crop -> text, no detector, no cascade --
same methodology as the PaddleOCR zero-shot baseline
(data/paddle_finetune/zero_shot_baseline/results.csv) and the fine-tuned
checkpoint's val-set eval (24.33% exact-match, see CLAUDE.md).

Reads:  data/paddle_finetune/dataset/val.txt (300 lines, filename\\tground_truth)
        data/paddle_finetune/dataset/images/<filename>
Writes: benchmark_results_trocr_zeroshot_300val.csv
"""
import csv
import os
import sys
import time

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from PIL import Image
from app.inference.ocr_trocr import run_trocr_on_crop, get_trocr

VAL_TXT = 'data/paddle_finetune/dataset/val.txt'
IMAGES_DIR = 'data/paddle_finetune/dataset'
OUTPUT_CSV = 'benchmark_results_trocr_zeroshot_300val.csv'


def main():
    with open(VAL_TXT) as f:
        entries = []
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            path, gt = line.split('\t')
            entries.append((path, gt))

    print(f"Loaded {len(entries)} val entries from {VAL_TXT}")

    # Force model load up front so the per-row timing loop only measures
    # inference, and so any load failure surfaces before burning time on
    # rows that would all fail identically.
    _, _, device = get_trocr()
    print(f"TrOCR loaded on device={device}")

    rows = []
    t0 = time.time()
    for i, (rel_path, gt) in enumerate(entries, 1):
        img_path = os.path.join(IMAGES_DIR, rel_path)
        filename = os.path.basename(rel_path)
        try:
            crop = Image.open(img_path)
            pred = run_trocr_on_crop(crop)
        except Exception as e:
            pred = ''
            print(f"  [{i}/{len(entries)}] ERROR on {filename}: {e}")
        exact_match = (pred == gt)
        rows.append({
            'filename': filename,
            'ground_truth': gt,
            'trocr_raw': pred,
            'exact_match': exact_match,
        })
        if i % 50 == 0 or i == len(entries):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(entries)}] elapsed={elapsed:.1f}s "
                  f"({elapsed / i:.2f}s/crop)")

    total_time = time.time() - t0

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'ground_truth', 'trocr_raw', 'exact_match'])
        writer.writeheader()
        writer.writerows(rows)

    correct = sum(1 for r in rows if r['exact_match'])
    print()
    print(f"=== Zero-shot TrOCR baseline complete ===")
    print(f"Device: {device}")
    print(f"Total runtime: {total_time:.1f}s for {len(rows)} crops "
          f"({total_time / len(rows):.2f}s/crop avg)")
    print(f"Exact match: {correct}/{len(rows)} = {correct / len(rows) * 100:.2f}%")
    print(f"Written to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
