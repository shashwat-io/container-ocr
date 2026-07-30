"""
Rotation-retry zero-shot TrOCR baseline over the same 300 val crops as
scripts/benchmark_trocr_zeroshot_300val.py, testing whether the portrait-
aspect-ratio squish diagnosed in that run (crops resized to a fixed square
384x384 by ViTImageProcessor) was the actual cause of its 0% exact-match
score. Uses extract_container_code_trocr_rotated() from ocr_trocr.py (0deg,
+90deg, -90deg, best-of-three).

Reads:  data/paddle_finetune/dataset/val.txt (300 lines, filename\\tground_truth)
        data/paddle_finetune/dataset/images/<filename>
Writes: benchmark_results_trocr_zeroshot_rotated_300val.csv
"""
import csv
import os
import sys
import time

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from PIL import Image
from app.inference.ocr_trocr import extract_container_code_trocr_rotated, get_trocr

VAL_TXT = 'data/paddle_finetune/dataset/val.txt'
IMAGES_DIR = 'data/paddle_finetune/dataset'
OUTPUT_CSV = 'benchmark_results_trocr_zeroshot_rotated_300val.csv'


def main():
    with open(VAL_TXT) as f:
        entries = []
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            path, gt = line.split('\t')
            entries.append((path, gt))

    print(f"Loaded {len(entries)} val entries from {VAL_TXT}", flush=True)

    _, _, device = get_trocr()
    print(f"TrOCR loaded on device={device}", flush=True)

    rows = []
    t0 = time.time()
    for i, (rel_path, gt) in enumerate(entries, 1):
        img_path = os.path.join(IMAGES_DIR, rel_path)
        filename = os.path.basename(rel_path)
        try:
            crop = Image.open(img_path)
            result = extract_container_code_trocr_rotated(crop)
            pred = result['trocr_raw_best']
            rotation = result['rotation_used']
        except Exception as e:
            pred = ''
            rotation = 'error'
            print(f"  [{i}/{len(entries)}] ERROR on {filename}: {e}", flush=True)
        exact_match = (pred == gt)
        rows.append({
            'filename': filename,
            'ground_truth': gt,
            'trocr_raw_best': pred,
            'rotation_used': rotation,
            'exact_match': exact_match,
        })
        if i % 25 == 0 or i == len(entries):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(entries)}] elapsed={elapsed:.1f}s "
                  f"({elapsed / i:.2f}s/crop)", flush=True)

    total_time = time.time() - t0

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['filename', 'ground_truth', 'trocr_raw_best', 'rotation_used', 'exact_match'])
        writer.writeheader()
        writer.writerows(rows)

    correct = sum(1 for r in rows if r['exact_match'])
    rotation_counts = {}
    for r in rows:
        rotation_counts[r['rotation_used']] = rotation_counts.get(r['rotation_used'], 0) + 1

    print(flush=True)
    print(f"=== Rotated zero-shot TrOCR baseline complete ===", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Total runtime: {total_time:.1f}s for {len(rows)} crops "
          f"({total_time / len(rows):.2f}s/crop avg)", flush=True)
    print(f"Exact match: {correct}/{len(rows)} = {correct / len(rows) * 100:.2f}%", flush=True)
    print(f"Rotation-used breakdown: {rotation_counts}", flush=True)
    print(f"Written to: {OUTPUT_CSV}", flush=True)


if __name__ == '__main__':
    main()
