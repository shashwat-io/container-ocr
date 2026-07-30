"""
Save contact-sheet grid images visualizing the gap_detection.py candidate
methods across all 218 right_column_bleed crops -- a fast, non-interactive
way to eyeball whether the detected line consistently lands between the
code column and the bleed column.

Detection + visualization only -- does NOT crop, save cropped images, or
touch OCR/inference code. Does NOT modify tall_crop_bleed_labels.csv,
gap_detection_results.csv, or any other existing file; only reads them and
writes new PNGs.

Each crop gets ONE overlay (not 3 side-by-side panels) with all three
methods' detected gap drawn as a distinct-colored vertical line -- chosen
over separate panels because these crops are narrow/tall already, so three
colored lines on one image are easy to compare directly (do they land on
top of each other or not) without tripling the already-cramped width.

Usage: python3 scripts/paddle_finetune/gap_detection_contact_sheets.py
Writes: data/paddle_finetune/gap_detection_viz/contact_sheets/sheet_XX.png
"""
import csv
import os
import sys

REPO_ROOT = '/home/shashwatkumar/Testing/container_ocr_test'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts/paddle_finetune'))
os.chdir(REPO_ROOT)

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from gap_detection import detect_all_methods  # noqa: E402

LABELS_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/tall_crop_bleed_labels.csv')
DATASET_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/dataset')
OUT_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_detection_viz/contact_sheets')

COLORS = {
    'column_variance': (255, 60, 60),
    'column_bg_deviation': (60, 200, 60),
    'column_edge_density': (60, 140, 255),
}

CELL_H = 260          # thumbnail height, aspect-preserved
CELL_W = 160           # thumbnail cell width (crop centered/padded into this)
LABEL_H = 34           # per-cell caption strip height
COLS = 5
ROWS = 4
PER_SHEET = COLS * ROWS
PAD = 6


def make_thumbnail_with_overlay(rel_path):
    full_path = os.path.join(DATASET_DIR, rel_path)
    results = detect_all_methods(full_path)
    img = Image.open(full_path).convert('RGB')
    W, H = img.size
    draw = ImageDraw.Draw(img)
    for method, color in COLORS.items():
        res = results[method]
        if res['gap_x'] is not None:
            x = int(round(res['gap_x']))
            draw.line([(x, 0), (x, H)], fill=color, width=max(1, W // 150))

    scale = CELL_H / H
    new_w = max(1, int(round(W * scale)))
    img = img.resize((new_w, CELL_H), Image.LANCZOS)

    cell = Image.new('RGB', (CELL_W, CELL_H), (30, 30, 30))
    offset_x = max(0, (CELL_W - new_w) // 2)
    if new_w > CELL_W:
        img = img.resize((CELL_W, CELL_H), Image.LANCZOS)
        offset_x = 0
    cell.paste(img, (offset_x, 0))
    return cell, results


def caption_lines(rel_path, results):
    short = os.path.basename(rel_path)
    short = short[:22] + '...' if len(short) > 25 else short
    amb_flags = ''.join('A' if results[m]['ambiguous'] else '.' for m in COLORS)
    return short, amb_flags


def build_sheets():
    rows = list(csv.DictReader(open(LABELS_CSV)))
    right_bleed = [r['image_path'] for r in rows if r['bleed_label'] == 'right_column_bleed']
    print(f"crops to render: {len(right_bleed)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    n_sheets = (len(right_bleed) + PER_SHEET - 1) // PER_SHEET
    sheet_w = COLS * (CELL_W + PAD) + PAD
    sheet_h = ROWS * (CELL_H + LABEL_H + PAD) + PAD + 50  # +50 for 2-line legend header

    for sheet_idx in range(n_sheets):
        batch = right_bleed[sheet_idx * PER_SHEET:(sheet_idx + 1) * PER_SHEET]
        sheet = Image.new('RGB', (sheet_w, sheet_h), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        draw.text((PAD, 6), 'column_variance=red   column_bg_deviation=green   column_edge_density=blue',
                   fill=(0, 0, 0), font=font)
        draw.text((PAD, 22), '(caption line 2: A/. per method in variance/bg_dev/edge_density order)',
                   fill=(0, 0, 0), font=font)

        for i, rel_path in enumerate(batch):
            r, c = divmod(i, COLS)
            x0 = PAD + c * (CELL_W + PAD)
            y0 = 50 + PAD + r * (CELL_H + LABEL_H + PAD)
            cell, results = make_thumbnail_with_overlay(rel_path)
            sheet.paste(cell, (x0, y0))
            short, amb_flags = caption_lines(rel_path, results)
            draw.text((x0, y0 + CELL_H + 2), short, fill=(0, 0, 0), font=font)
            draw.text((x0, y0 + CELL_H + 16), amb_flags, fill=(120, 0, 0), font=font)

        out_path = os.path.join(OUT_DIR, f"sheet_{sheet_idx + 1:02d}.png")
        sheet.save(out_path)
        print(f"  saved {out_path} ({len(batch)} crops)")

    print(f"\nDone. {n_sheets} sheets in {OUT_DIR}")


if __name__ == '__main__':
    build_sheets()
