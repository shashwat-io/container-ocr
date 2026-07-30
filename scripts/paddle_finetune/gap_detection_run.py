"""
Run all gap_detection.py candidate methods across every right_column_bleed
crop in tall_crop_bleed_labels.csv and save per-crop, per-method results.

Detection + reporting only -- does NOT crop, save cropped images, or touch
OCR/inference code. Does NOT modify tall_crop_bleed_labels.csv or any other
existing file; only reads it and writes a new CSV.

Usage: python3 scripts/paddle_finetune/gap_detection_run.py
Writes: data/paddle_finetune/gap_detection_results.csv
"""
import csv
import os
import sys

REPO_ROOT = '/home/shashwatkumar/Testing/container_ocr_test'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts/paddle_finetune'))
os.chdir(REPO_ROOT)

from gap_detection import detect_all_methods, METHODS  # noqa: E402

LABELS_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/tall_crop_bleed_labels.csv')
DATASET_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/dataset')
OUT_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_detection_results.csv')


def main():
    rows = list(csv.DictReader(open(LABELS_CSV)))
    right_bleed = [r for r in rows if r['bleed_label'] == 'right_column_bleed']
    print(f"right_column_bleed crops to process: {len(right_bleed)}")

    method_names = list(METHODS.keys())
    fieldnames = ['image_path', 'width', 'height']
    for m in method_names:
        fieldnames += [f'{m}_gap_x', f'{m}_gap_frac', f'{m}_run_width', f'{m}_ambiguous']

    out_rows = []
    for i, r in enumerate(right_bleed, 1):
        rel_path = r['image_path']
        full_path = os.path.join(DATASET_DIR, rel_path)
        results = detect_all_methods(full_path)

        from PIL import Image
        with Image.open(full_path) as im:
            W, H = im.size

        out_row = {'image_path': rel_path, 'width': W, 'height': H}
        for m in method_names:
            res = results[m]
            gap_x = res['gap_x']
            out_row[f'{m}_gap_x'] = '' if gap_x is None else round(gap_x, 2)
            out_row[f'{m}_gap_frac'] = '' if gap_x is None else round(gap_x / W, 4)
            out_row[f'{m}_run_width'] = res['run_width']
            out_row[f'{m}_ambiguous'] = res['ambiguous']
        out_rows.append(out_row)

        if i % 50 == 0 or i == len(right_bleed):
            print(f"  processed {i}/{len(right_bleed)}", flush=True)

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} rows to {OUT_CSV}")


if __name__ == '__main__':
    main()
