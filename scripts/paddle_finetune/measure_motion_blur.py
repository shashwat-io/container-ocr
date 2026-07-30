"""
Standalone, CPU-only, read-only measurement of blur severity across the
PaddleOCR crop pool (data/paddle_finetune/dataset/). Does not touch any
running benchmark/GPU process. Measurement only -- applies no fix.

For every image referenced in train.txt + val.txt (not a blind glob of
images/, which is known to contain stale unreferenced files), computes:
  - Laplacian variance (blur severity; lower = blurrier)
  - Sobel gy/gx squared-mean ratio (directional/motion blur indicator;
    far from 1.0 = directional bias, near 1.0 = no strong directional bias)

Writes one row per image to a CSV and prints summary stats.
"""
import csv
import os
import sys

import cv2
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "paddle_finetune", "dataset")
DATASET_DIR = os.path.normpath(DATASET_DIR)
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data", "paddle_finetune", "blur_measurements.csv")
OUTPUT_CSV = os.path.normpath(OUTPUT_CSV)

GRADIENT_RATIO_LOW = 0.7
GRADIENT_RATIO_HIGH = 1.4
LAPLACIAN_BANDS = (100, 150, 200)
TOP_N_BLURRIEST = 15


def load_split_paths(dataset_dir, split_filename):
    """Read image paths (first tab-separated column) from a PaddleOCR label file."""
    split_path = os.path.join(dataset_dir, split_filename)
    paths = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            image_rel_path = line.split("\t", 1)[0]
            paths.append(image_rel_path)
    return paths


def compute_blur_metrics(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    gx = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0)
    gy = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1)
    mean_gx2 = np.mean(gx ** 2)
    mean_gy2 = np.mean(gy ** 2)
    gradient_ratio = mean_gy2 / mean_gx2 if mean_gx2 != 0 else float("inf")

    return laplacian_var, gradient_ratio


def main():
    entries = []  # (image_rel_path, split)
    for split_name, split_file in (("train", "train.txt"), ("val", "val.txt")):
        for rel_path in load_split_paths(DATASET_DIR, split_file):
            entries.append((rel_path, split_name))

    print(f"Loaded {len(entries)} image references from train.txt + val.txt")

    rows = []
    missing = []
    unreadable = []

    for rel_path, split in entries:
        abs_path = os.path.join(DATASET_DIR, rel_path)
        if not os.path.exists(abs_path):
            missing.append(rel_path)
            continue

        metrics = compute_blur_metrics(abs_path)
        if metrics is None:
            unreadable.append(rel_path)
            continue

        laplacian_var, gradient_ratio = metrics
        rows.append({
            "image_path": rel_path,
            "split": split,
            "laplacian_var": laplacian_var,
            "gradient_ratio": gradient_ratio,
        })

    if missing:
        print(f"WARNING: {len(missing)} referenced image(s) not found on disk (first 5: {missing[:5]})", file=sys.stderr)
    if unreadable:
        print(f"WARNING: {len(unreadable)} image(s) failed to decode (first 5: {unreadable[:5]})", file=sys.stderr)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "split", "laplacian_var", "gradient_ratio"])
        for r in rows:
            writer.writerow([r["image_path"], r["split"], f"{r['laplacian_var']:.4f}", f"{r['gradient_ratio']:.6f}"])

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

    if not rows:
        print("No rows computed; skipping summary stats.")
        return

    lap_vars = np.array([r["laplacian_var"] for r in rows])
    grad_ratios = np.array([r["gradient_ratio"] for r in rows])

    print("\n=== Laplacian Variance (blur severity; lower = blurrier) ===")
    print(f"  min:    {lap_vars.min():.2f}")
    print(f"  max:    {lap_vars.max():.2f}")
    print(f"  mean:   {lap_vars.mean():.2f}")
    print(f"  median: {np.median(lap_vars):.2f}")
    print(f"  p10:    {np.percentile(lap_vars, 10):.2f}")
    print(f"  p25:    {np.percentile(lap_vars, 25):.2f}")

    print("\n=== Gradient Ratio (mean(gy^2)/mean(gx^2); directional blur indicator) ===")
    print(f"  min:    {grad_ratios.min():.4f}")
    print(f"  max:    {grad_ratios.max():.4f}")
    print(f"  mean:   {grad_ratios.mean():.4f}")
    print(f"  median: {np.median(grad_ratios):.4f}")
    outside_count = int(np.sum((grad_ratios < GRADIENT_RATIO_LOW) | (grad_ratios > GRADIENT_RATIO_HIGH)))
    print(f"  outside [{GRADIENT_RATIO_LOW}, {GRADIENT_RATIO_HIGH}] (rough directional-blur-present flag, unvalidated threshold):"
          f" {outside_count} / {len(rows)} ({100 * outside_count / len(rows):.1f}%)")

    print("\n=== Laplacian Variance severity bands (descriptive only, unvalidated cutoffs) ===")
    for band in LAPLACIAN_BANDS:
        count = int(np.sum(lap_vars < band))
        print(f"  below {band}: {count} / {len(rows)} ({100 * count / len(rows):.1f}%)")

    print(f"\n=== {TOP_N_BLURRIEST} lowest-Laplacian-variance images (most likely genuinely blurry) ===")
    sorted_rows = sorted(rows, key=lambda r: r["laplacian_var"])
    for r in sorted_rows[:TOP_N_BLURRIEST]:
        print(f"  {r['laplacian_var']:10.2f}  grad_ratio={r['gradient_ratio']:.3f}  [{r['split']}]  {r['image_path']}")


if __name__ == "__main__":
    main()
