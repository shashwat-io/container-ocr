"""
Build the actual re-cropped (code-column-only) images for right_column_bleed
crops, using the gap_detection.py results already computed in
gap_detection_results.csv (this script does NOT re-run detection -- it only
reads that CSV and acts on it).

Two crop sets are both fully decided and recropped here so the later OCR
eval script can report both numbers from the same underlying data:

  - PRIMARY (group-deduplicated, ~74 crops): one representative crop per
    distinct source-video/container group (group_id() from video_safety.py,
    reused unmodified -- same grouping already used for YOLO's
    EXCLUDE_WHOLE_SOURCE_VIDEO test-split safety). Representative = the
    crop with the lexicographically-first crop filename in its group (crop
    filenames carry extract_crops.py's sequential crop index as a zero-
    padded prefix, so "first by filename" is a deterministic, detection-
    outcome-blind pick -- it does NOT cherry-pick whichever crop in the
    group happened to get the most confident gap detection).
  - SECONDARY/FULL (all 218 crops): every right_column_bleed crop, for
    comparison, so we can see how much the primary number shifts once
    duplicate frames from heavy videos are removed.

Cut-point rule: require AGREEMENT between column_variance and
column_edge_density (both must be non-ambiguous, and their detected gap_x
must fall within 10% of the crop width of each other). column_edge_density's
gap_x is used as the actual cut point when they agree, since gap_detection.py's
own docstring documents it as the most robust of the three to lighting
gradients. column_bg_deviation is deliberately NOT part of this rule --
Turn 9's report found it had a much higher ambiguous rate than the other two,
so requiring its agreement too would just inflate the skip count without
adding real confidence. Crops where either method is ambiguous, or the two
disagree by more than the 10% tolerance, are skipped rather than forced --
this is a real-file-producing step, so a wrong guess here isn't just a
mislabeled diagnostic row, it actively corrupts the OCR eval input.

Does NOT modify tall_crop_bleed_labels.csv, gap_detection_results.csv, or
any existing contact sheet/notebook. Does NOT touch the original crop
images in data/paddle_finetune/dataset/images/ -- recropped images are
written to a new directory only.

Usage: python3 scripts/paddle_finetune/gap_recrop_build.py
Writes:
  data/paddle_finetune/gap_recrop_images/<crop_filename>.jpg  (recropped)
  data/paddle_finetune/gap_recrop_manifest.csv
"""
import csv
import os
import re
import sys

REPO_ROOT = '/home/shashwatkumar/Testing/container_ocr_test'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts/paddle_finetune'))
os.chdir(REPO_ROOT)

from PIL import Image  # noqa: E402
from video_safety import group_id  # noqa: E402

LABELS_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/tall_crop_bleed_labels.csv')
GAP_RESULTS_CSV = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_detection_results.csv')
DATASET_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/dataset')
RECROP_DIR = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_images')
OUT_MANIFEST = os.path.join(REPO_ROOT, 'data/paddle_finetune/gap_recrop_manifest.csv')

AGREEMENT_TOLERANCE_FRAC = 0.10

# Matches extract_crops.py line 185:
#   f"crop_{i:04d}_{os.path.splitext(os.path.basename(row['image_path']))[0]}_box{row['box_index']}.jpg"
CROP_RE = re.compile(r'^crop_\d+_(.+)_box\d+\.jpg$')


def recover_source_stem(crop_filename):
    m = CROP_RE.match(os.path.basename(crop_filename))
    if not m:
        raise ValueError(f"unrecognized crop filename format: {crop_filename}")
    return m.group(1)


def decide_cut(gap_row, width):
    """Returns (cut_x:int|None, reason:str)."""
    var_gap_raw = gap_row['column_variance_gap_x']
    edge_gap_raw = gap_row['column_edge_density_gap_x']
    if var_gap_raw == '' or edge_gap_raw == '':
        return None, 'no_gap_detected'

    var_amb = gap_row['column_variance_ambiguous'] == 'True'
    edge_amb = gap_row['column_edge_density_ambiguous'] == 'True'
    if var_amb or edge_amb:
        return None, 'ambiguous'

    var_gap = float(var_gap_raw)
    edge_gap = float(edge_gap_raw)
    if abs(var_gap - edge_gap) > AGREEMENT_TOLERANCE_FRAC * width:
        return None, 'disagreement'

    return int(round(edge_gap)), 'ok'


def main():
    labels = list(csv.DictReader(open(LABELS_CSV)))
    right_bleed_paths = [r['image_path'] for r in labels if r['bleed_label'] == 'right_column_bleed']
    print(f"right_column_bleed crops: {len(right_bleed_paths)}")

    gap_results = {r['image_path']: r for r in csv.DictReader(open(GAP_RESULTS_CSV))}
    missing = [p for p in right_bleed_paths if p not in gap_results]
    if missing:
        raise RuntimeError(f"{len(missing)} right_column_bleed crops missing from "
                            f"{GAP_RESULTS_CSV}, e.g. {missing[:3]}")

    # Group assignment (reusing group_id() unmodified).
    groups = {}  # group -> list of image_path
    for p in right_bleed_paths:
        stem = recover_source_stem(p)
        g = group_id(stem)
        groups.setdefault(g, []).append(p)
    print(f"distinct source-video/container groups: {len(groups)}")

    representative_of = {}  # image_path -> True if it's its group's representative
    for g, members in groups.items():
        rep = sorted(members)[0]
        representative_of[rep] = True

    os.makedirs(RECROP_DIR, exist_ok=True)

    manifest_rows = []
    skip_counts = {'no_gap_detected': 0, 'ambiguous': 0, 'disagreement': 0}
    n_ok = 0

    path_to_group = {p: g for g, members in groups.items() for p in members}

    for p in right_bleed_paths:
        gap_row = gap_results[p]
        width = int(gap_row['width'])
        height = int(gap_row['height'])
        cut_x, reason = decide_cut(gap_row, width)

        is_rep = p in representative_of
        recrop_rel = None

        if cut_x is None:
            skip_counts[reason] += 1
        else:
            cut_x = max(2, min(width - 2, cut_x))
            full_path = os.path.join(DATASET_DIR, p)
            img = Image.open(full_path).convert('RGB')
            recropped = img.crop((0, 0, cut_x, height))
            crop_basename = os.path.basename(p)
            recrop_rel = crop_basename
            recropped.save(os.path.join(RECROP_DIR, crop_basename))
            n_ok += 1

        manifest_rows.append({
            'image_path': p,
            'group_id': path_to_group[p],
            'is_group_representative': is_rep,
            'width': width,
            'height': height,
            'cut_x': cut_x if cut_x is not None else '',
            'decision': reason,
            'recrop_path': f'gap_recrop_images/{recrop_rel}' if recrop_rel else '',
        })

    fieldnames = ['image_path', 'group_id', 'is_group_representative', 'width', 'height',
                  'cut_x', 'decision', 'recrop_path']
    with open(OUT_MANIFEST, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_total = len(right_bleed_paths)
    n_skipped = sum(skip_counts.values())
    rep_rows = [r for r in manifest_rows if r['is_group_representative']]
    rep_skipped = sum(1 for r in rep_rows if r['decision'] != 'ok')

    print()
    print("=== FULL SET (all 218 right_column_bleed crops) ===")
    print(f"  processed (cut applied): {n_ok}/{n_total}")
    print(f"  skipped: {n_skipped}/{n_total}  -> {skip_counts}")
    print()
    print(f"=== PRIMARY SET ({len(rep_rows)} group representatives) ===")
    print(f"  processed (cut applied): {len(rep_rows) - rep_skipped}/{len(rep_rows)}")
    print(f"  skipped: {rep_skipped}/{len(rep_rows)}")
    print()
    print(f"Recropped images: {RECROP_DIR}")
    print(f"Manifest: {OUT_MANIFEST}")


if __name__ == '__main__':
    main()
