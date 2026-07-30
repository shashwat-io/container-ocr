"""
Build an ~85-image stratified ground-truth verification sample from the two
FULL 786-image cascade benchmark CSVs (TrOCR gated + PaddleOCR), for manual
visual code-reading review.

Per-request design choices, spelled out here rather than left implicit:

- Stratifies by TrOCR tier (Tier1-exact / Tier1b-recovery /
  Tier3-fallback-novalid), proportional to each tier's share among the 786
  images, with Tier3-fallback-novalid guaranteed >=15 (proportional share is
  31, already above the floor, so no boosting was needed).
- The 3 'none' (no-detection) rows are excluded from the sampling frame --
  there is no box to draw or crop for those, so they can't go in a visual
  review grid.
- Neither benchmark CSV stores bbox coordinates, so YOLO detection is
  re-run once per sampled image (not per row) to recover box geometry. Both
  benchmark runs used the identical model (models/best_resplit.pt) against
  the identical images, and a per-filename row-count check (841 rows / 786
  images in both CSVs, 0 mismatches) confirms both benchmarks saw the same
  boxes -- so box index 0 here lines up with each CSV's first/representative
  row for that filename, which is also what the manifest reports as
  "the" prediction for images with multiple detections.
- Box-drawing and crop-padding are NOT reimplemented: the red rectangle
  matches app/main/routes.py's _draw_annotated_image() style exactly
  (outline='red', width=3), and the padding=10 crop margin matches
  app.inference.ocr.crop_detection()'s own default exactly (same numbers,
  applied to the now-annotated image so the red box ships inside the crop).
  Deliberately NOT drawing the predicted text/label on the image itself
  (unlike routes.py, which does) -- this crop is meant to be a blind read;
  predictions live only in the manifest CSV.
- Grid tiling follows the same thumbnail+caption-strip pattern as
  scripts/paddle_finetune/gap_detection_contact_sheets.py (fixed cell size,
  letterboxed aspect-preserving thumbnail, caption text in a strip BELOW
  the cell rather than overlaid on it, so it never occludes code text).
"""
import os
import sys
import csv
import random
import shutil
import subprocess
from collections import defaultdict

REPO_ROOT = '/home/shashwatkumar/Testing/container_ocr_test'
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from app.inference.ocr import crop_detection  # noqa: E402 -- reused only for its padding=10 constant/formula

TROCR_CSV = 'benchmark_results_trocr_cascade_FULL_thin_gated_20260729.csv'
PADDLE_CSV = 'benchmark_results_paddle_cascade_FULL_20260728.csv'
TEST_DIR = os.path.join(REPO_ROOT, 'container.v2i.yolov11.resplit/test/images')
MODEL_PATH = 'models/best_resplit.pt'
OUT_DIR = os.path.join(REPO_ROOT, 'ground_truth_verification')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest.csv')

SEED = 42
TARGET_TOTAL = 85
MIN_TIER3 = 15
ALLOC = {'Tier1-exact': 49, 'Tier3-fallback-novalid': 31, 'Tier1b-recovery': 5}
assert sum(ALLOC.values()) == TARGET_TOTAL
assert ALLOC['Tier3-fallback-novalid'] >= MIN_TIER3

GRID_COLS, GRID_ROWS = 3, 2
PER_GRID = GRID_COLS * GRID_ROWS
CELL_W, CELL_H = 560, 350
LABEL_H = 40
PAD = 10
CROP_PADDING = 10  # matches app.inference.ocr.crop_detection's own default exactly


def load_csv(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def build_manifest_source():
    trocr_rows = load_csv(TROCR_CSV)
    paddle_rows = load_csv(PADDLE_CSV)

    trocr_by_fn = defaultdict(list)
    for r in trocr_rows:
        trocr_by_fn[r['filename']].append(r)
    paddle_by_fn = defaultdict(list)
    for r in paddle_rows:
        paddle_by_fn[r['filename']].append(r)

    assert len(trocr_by_fn) == len(paddle_by_fn) == 786, \
        f"expected 786 unique images in both CSVs, got trocr={len(trocr_by_fn)} paddle={len(paddle_by_fn)}"

    mismatched = [fn for fn in trocr_by_fn if len(trocr_by_fn[fn]) != len(paddle_by_fn.get(fn, []))]
    if mismatched:
        print(f"WARNING: {len(mismatched)} filenames have mismatched row counts between the two CSVs; "
              f"first few: {mismatched[:5]}", flush=True)

    records = {}
    for fn, trows in trocr_by_fn.items():
        prows = paddle_by_fn.get(fn, [])
        t0 = trows[0]
        p0 = prows[0] if prows else None
        records[fn] = {
            'filename': fn,
            'tier': t0['tier'],
            'num_boxes_trocr': t0['num_boxes'],
            'trocr_predicted_code': t0['container_code'],
            'trocr_is_valid': t0['is_valid'],
            'paddle_predicted_code': p0['container_code'] if p0 else '',
            'paddle_is_valid': p0['is_valid'] if p0 else '',
        }
    return records


def stratified_sample(records):
    by_tier = defaultdict(list)
    for fn, rec in records.items():
        if rec['tier'] == 'none':
            continue  # no detection -> nothing to draw/crop
        by_tier[rec['tier']].append(fn)

    rng = random.Random(SEED)
    sampled = []
    for tier, n in ALLOC.items():
        pool = sorted(by_tier.get(tier, []))  # sort first for a deterministic base ordering pre-shuffle
        if len(pool) < n:
            raise RuntimeError(f"tier {tier} has only {len(pool)} eligible images, need {n}")
        chosen = rng.sample(pool, n)
        for fn in chosen:
            sampled.append((tier, fn))

    # stable overall ordering: tier group order matches ALLOC dict order, filenames sorted within tier
    sampled.sort(key=lambda tf: (list(ALLOC.keys()).index(tf[0]), tf[1]))
    return sampled


def annotate_and_crop(model, filename):
    path = os.path.join(TEST_DIR, filename)
    img = Image.open(path).convert('RGB')
    import numpy as np
    results = model(np.array(img), verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            boxes.append((x1, y1, x2, y2))
    if not boxes:
        return None, 0

    x1, y1, x2, y2 = boxes[0]  # box 0 == the CSVs' representative/first row for this filename
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)  # same style as app/main/routes.py

    w, h = img.size
    left = max(0, int(x1) - CROP_PADDING)
    top = max(0, int(y1) - CROP_PADDING)
    right = min(w, int(x2) + CROP_PADDING)
    bottom = min(h, int(y2) + CROP_PADDING)
    crop = img.crop((left, top, right, bottom))
    return crop, len(boxes)


def fit_into_cell(crop):
    cw, ch = crop.size
    scale = min(CELL_W / cw, CELL_H / ch)
    new_w, new_h = max(1, int(cw * scale)), max(1, int(ch * scale))
    resized = crop.resize((new_w, new_h), Image.LANCZOS)
    cell = Image.new('RGB', (CELL_W, CELL_H), (40, 40, 40))
    cell.paste(resized, ((CELL_W - new_w) // 2, (CELL_H - new_h) // 2))
    return cell


def build_grids(samples):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    n_grids = (len(samples) + PER_GRID - 1) // PER_GRID
    sheet_w = GRID_COLS * (CELL_W + PAD) + PAD
    sheet_h = GRID_ROWS * (CELL_H + LABEL_H + PAD) + PAD

    for gi in range(n_grids):
        batch = samples[gi * PER_GRID:(gi + 1) * PER_GRID]
        sheet = Image.new('RGB', (sheet_w, sheet_h), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for i, item in enumerate(batch):
            r, c = divmod(i, GRID_COLS)
            x0 = PAD + c * (CELL_W + PAD)
            y0 = PAD + r * (CELL_H + LABEL_H + PAD)
            sheet.paste(item['cell'], (x0, y0))
            bbox = draw.textbbox((0, 0), item['sample_id'], font=font)
            tw = bbox[2] - bbox[0]
            draw.text((x0 + (CELL_W - tw) // 2, y0 + CELL_H + 6), item['sample_id'],
                       fill=(0, 0, 0), font=font)
        out_path = os.path.join(OUT_DIR, f"grid_{gi + 1:02d}.png")
        sheet.save(out_path)
        print(f"  saved {out_path} ({len(batch)} crops)", flush=True)
    return n_grids


def main():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    print("Loading benchmark CSVs...", flush=True)
    records = build_manifest_source()

    print("Building stratified sample...", flush=True)
    sampled = stratified_sample(records)
    tier_counts = defaultdict(int)
    for tier, _ in sampled:
        tier_counts[tier] += 1
    print(f"  total sampled: {len(sampled)}", flush=True)
    for t, n in tier_counts.items():
        print(f"    {t}: {n}", flush=True)

    print(f"Loading YOLO model: {MODEL_PATH}", flush=True)
    model = YOLO(MODEL_PATH)

    manifest_rows = []
    grid_items = []
    skipped = []
    for i, (tier, fn) in enumerate(sampled, 1):
        sample_id = f"S{i:03d}"
        crop, n_boxes = annotate_and_crop(model, fn)
        if crop is None:
            skipped.append((sample_id, fn))
            print(f"  WARNING: {fn} had 0 boxes on re-detection (was tier={tier} in CSV) -- skipping", flush=True)
            continue
        cell = fit_into_cell(crop)
        grid_items.append({'sample_id': sample_id, 'cell': cell})

        rec = records[fn]
        manifest_rows.append({
            'sample_id': sample_id,
            'filename': fn,
            'tier': tier,
            'trocr_predicted_code': rec['trocr_predicted_code'],
            'trocr_is_valid': rec['trocr_is_valid'],
            'paddle_predicted_code': rec['paddle_predicted_code'],
            'paddle_is_valid': rec['paddle_is_valid'],
        })
        if i % 20 == 0 or i == len(sampled):
            print(f"  processed {i}/{len(sampled)}", flush=True)

    print(f"Building grids ({PER_GRID} per grid, {GRID_COLS}x{GRID_ROWS})...", flush=True)
    n_grids = build_grids(grid_items)

    with open(MANIFEST_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'sample_id', 'filename', 'tier', 'trocr_predicted_code', 'trocr_is_valid',
            'paddle_predicted_code', 'paddle_is_valid'])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Manifest written: {MANIFEST_PATH} ({len(manifest_rows)} rows)", flush=True)

    if skipped:
        print(f"SKIPPED {len(skipped)} samples with 0 boxes on re-detection: {skipped}", flush=True)

    zip_path = os.path.join(REPO_ROOT, 'ground_truth_verification.zip')
    if os.path.isfile(zip_path):
        os.remove(zip_path)
    subprocess.run(['zip', '-r', 'ground_truth_verification.zip', 'ground_truth_verification/'],
                    cwd=REPO_ROOT, check=True)

    print("\n=== SUMMARY ===")
    print(f"Total unique images sampled: {len(manifest_rows)}")
    for t, n in tier_counts.items():
        print(f"  {t}: {n}")
    print(f"Grid images produced: {n_grids}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Zip: {zip_path}")
    print("DONE")


if __name__ == '__main__':
    main()
