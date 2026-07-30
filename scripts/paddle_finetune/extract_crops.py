"""
Step 1 -- Crop extraction script.

Reads manual_labels.csv (Drive-authoritative, see check_drive_freshness),
applies the label-quality and test-split-leakage filters, and writes
(crop_image, ground_truth_text) pairs to data/paddle_finetune/crops/ plus an
index manifest CSV for Step 2/3 to consume.

Hard constraints enforced here (see ocr_pipeline_rebuild_spec.md):
  1. No test-split contamination -- excludes by SOURCE VIDEO, not just exact
     filename, because this dataset's images are frames sampled from videos
     (Roboflow re-hashes filenames per export, so a naive path/filename
     compare misses exact-duplicate frames across splits -- confirmed: 119
     of 377 labeled rows are frame-identical to a test-split image under a
     different hash). User explicitly chose the conservative "exclude whole
     source video" policy over narrower alternatives after this was surfaced
     -- see EXCLUDE_WHOLE_SOURCE_VIDEO below. This drops usable rows from
     ~373 (post label-quality filter) to ~35: expected and intentional, not
     a bug.
  2. manual_labels.csv on Drive is authoritative -- see check_drive_freshness.
  3. Single source of crop truth -- see crop_utils.py's module docstring for
     the caveat on Labelling.ipynb not being available to diff against.
"""
import argparse
import csv
import hashlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from scripts.paddle_finetune.crop_utils import extract_crop
from scripts.paddle_finetune.video_safety import group_id, build_test_video_ids

DEFAULT_LABELS_CSV = os.path.join(REPO_ROOT, 'training_labels', 'manual_labels.csv')
# Local mirror of the Roboflow export that manual_labels.csv's image_path
# entries (Kaggle-session paths like /content/local_dataset/train/images/...)
# actually resolve against on this machine. Confirmed: every one of the 377
# manual_labels.csv rows' basenames exists under here.
LOCAL_IMAGE_ROOT = '/home/shashwatkumar/Downloads/container.v2i.yolov11'
# Leakage-safety reference now points at the corrected, leakage-free resplit
# (container.v2i.yolov11 was confirmed contaminated -- see leakage_check.py).
TEST_IMAGES_DIR = os.path.join(REPO_ROOT, 'container.v2i.yolov11.resplit', 'test', 'images')

OUTPUT_CROPS_DIR = os.path.join(REPO_ROOT, 'data', 'paddle_finetune', 'crops')
OUTPUT_MANIFEST = os.path.join(REPO_ROOT, 'data', 'paddle_finetune', 'manifest.csv')

USABLE_STATUSES = {'labeled', 'confirmed_asis'}

# User's explicit choice (asked 2026-07-10, given the 119/222/36 breakdown of
# exact-duplicate / same-video-different-offset / no-overlap rows): exclude
# every row whose source video appears ANYWHERE in the test split, not just
# exact-duplicate frames. Kept as a module constant (not a CLI flag) because
# loosening it is a decision that needs the same explicit confirmation, not
# a casual flag flip.
EXCLUDE_WHOLE_SOURCE_VIDEO = True


def check_drive_freshness(labels_csv, drive_labels_csv):
    """
    manual_labels.csv on Drive is the authoritative copy (Hard Constraint #2).
    If a freshly-synced Drive path is provided, compare row count + sha256
    against the local copy and abort loudly on any mismatch. If no Drive
    path is provided, this refuses to silently assume the local copy is
    current -- it prints a loud warning rather than pretending to have
    verified something it didn't.
    """
    def sha256_of(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            h.update(f.read())
        return h.hexdigest()

    def row_count(path):
        with open(path, newline='') as f:
            return sum(1 for _ in f) - 1  # minus header

    if not drive_labels_csv:
        print(
            "WARNING: no --drive-labels-csv given -- cannot verify that "
            f"{labels_csv} matches the authoritative Drive copy. Proceeding "
            "on the LOCAL copy only. Pass --drive-labels-csv pointing at a "
            "freshly-synced/mounted Drive path to enable the real freshness "
            "check before trusting this for anything beyond local dev.",
            file=sys.stderr,
        )
        return

    if not os.path.exists(drive_labels_csv):
        print(f"FATAL: --drive-labels-csv path does not exist: {drive_labels_csv}", file=sys.stderr)
        sys.exit(1)

    local_rows, drive_rows = row_count(labels_csv), row_count(drive_labels_csv)
    if local_rows != drive_rows:
        print(
            f"FATAL: local manual_labels.csv has {local_rows} rows but Drive "
            f"copy has {drive_rows} rows -- local copy is stale. Refusing to "
            "proceed. Re-copy from Drive before re-running.",
            file=sys.stderr,
        )
        sys.exit(1)

    local_hash, drive_hash = sha256_of(labels_csv), sha256_of(drive_labels_csv)
    if local_hash != drive_hash:
        print(
            "FATAL: local manual_labels.csv and Drive copy have the same row "
            "count but differ byte-for-byte (sha256 mismatch) -- local copy "
            "is stale or was hand-edited. Refusing to proceed.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Drive freshness check OK: {local_rows} rows, sha256 matches.")


def load_test_video_ids():
    """Thin wrapper around the shared video_safety.build_test_video_ids,
    preserving this script's original FATAL-and-exit behavior instead of
    letting a missing dir raise."""
    if not os.path.isdir(TEST_IMAGES_DIR):
        print(f"FATAL: test images dir not found: {TEST_IMAGES_DIR}", file=sys.stderr)
        sys.exit(1)
    return build_test_video_ids(TEST_IMAGES_DIR)


def is_label_usable(row):
    if row['status'] not in USABLE_STATUSES:
        return False, f"status={row['status']!r} not in {USABLE_STATUSES}"
    if row['checksum_valid'] == 'True':
        return True, None
    if row['status'] == 'confirmed_asis':
        return True, None
    return False, f"checksum_valid={row['checksum_valid']!r} and status != confirmed_asis"


def resolve_local_image_path(row):
    basename = os.path.basename(row['image_path'])
    return os.path.join(LOCAL_IMAGE_ROOT, row['split'], 'images', basename)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels-csv', default=DEFAULT_LABELS_CSV)
    ap.add_argument('--drive-labels-csv', default=None,
                     help='Path to a freshly-synced Drive copy, for the freshness check.')
    ap.add_argument('--output-crops-dir', default=OUTPUT_CROPS_DIR)
    ap.add_argument('--output-manifest', default=OUTPUT_MANIFEST)
    args = ap.parse_args()

    check_drive_freshness(args.labels_csv, args.drive_labels_csv)
    test_video_ids = load_test_video_ids()
    print(f"Loaded {len(test_video_ids)} distinct source-video IDs from the test split.")

    with open(args.labels_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    os.makedirs(args.output_crops_dir, exist_ok=True)

    kept, excluded_status, excluded_leakage = [], 0, 0
    for row in rows:
        usable, reason = is_label_usable(row)
        if not usable:
            excluded_status += 1
            continue
        basename = os.path.basename(row['image_path'])
        if EXCLUDE_WHOLE_SOURCE_VIDEO and group_id(basename) in test_video_ids:
            excluded_leakage += 1
            continue
        kept.append(row)

    print(f"Total rows in manual_labels.csv: {len(rows)}")
    print(f"Excluded by status/checksum_valid filter: {excluded_status}")
    print(f"Excluded by test-split source-video leakage check: {excluded_leakage}")
    print(f"Usable rows: {len(kept)}")

    manifest_rows = []
    for i, row in enumerate(kept):
        local_image_path = resolve_local_image_path(row)
        if not os.path.exists(local_image_path):
            print(f"FATAL: source image not found on disk: {local_image_path}", file=sys.stderr)
            sys.exit(1)

        crop = extract_crop(local_image_path, row['x_center'], row['y_center'], row['width'], row['height'])
        crop_filename = f"crop_{i:04d}_{os.path.splitext(os.path.basename(row['image_path']))[0]}_box{row['box_index']}.jpg"
        crop_path = os.path.join(args.output_crops_dir, crop_filename)
        crop.convert('RGB').save(crop_path, quality=95)

        manifest_rows.append({
            'crop_filename': crop_filename,
            'label_text': row['label_text'],
            'source_image_path': row['image_path'],
            'source_image_basename': os.path.basename(row['image_path']),
            'box_index': row['box_index'],
            'original_split': row['split'],
            'checksum_valid': row['checksum_valid'],
            'status': row['status'],
        })

    with open(args.output_manifest, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} crops to {args.output_crops_dir}")
    print(f"Wrote manifest to {args.output_manifest}")


if __name__ == '__main__':
    main()
