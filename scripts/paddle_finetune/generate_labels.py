"""
Step 2 -- Label file generation, and Step 3 -- train/valid split.

Reads data/paddle_finetune/manifest.csv (Step 1's output) and produces a
PaddleX text-recognition dataset directory:

    data/paddle_finetune/dataset/
        images/*.jpg
        train.txt   (image_path<TAB>label, one per line)
        val.txt
        dict.txt    (one character per line)

train.txt/val.txt format and dict.txt requirement verified directly against
the installed paddlex package's dataset checker (paddlex/modules/
text_recognition/dataset_checker/dataset_src/check_dataset.py) rather than
assumed -- it requires exactly 2 tab-separated fields per line, image paths
resolved relative to dataset_dir, and a dict.txt in dataset_dir.

Split is by SOURCE VIDEO ID (not by crop, and not just by exact image path),
consistent with the project's leakage-avoidance principle. Splitting by exact
image path is NOT enough: this dataset contains at least one case
(z4944210769393_e588d7aaa69420580cf57da233ec59b2, box_index=1) where the
identical source photo (same box coordinates, same label) was exported
twice under two different Roboflow hash filenames, both landing in the
original "train" split. A naive per-file split would happily put one copy
in train.txt and its duplicate in val.txt, making the validation set
trivially "see" a training example. Grouping by the same video-id key used
for the Step 1 test-split leakage check (basename up to the first
underscore) avoids that. This is a *different* split from the original
train/valid/test benchmark split -- it only decides how the
already-leakage-filtered fine-tuning set gets divided into a training
subset and a validation subset for the fine-tuning run itself.
"""
import argparse
import csv
import os
import random
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from scripts.paddle_finetune.video_safety import group_id

DEFAULT_MANIFEST = os.path.join(REPO_ROOT, 'data', 'paddle_finetune', 'manifest.csv')
DEFAULT_CROPS_DIR = os.path.join(REPO_ROOT, 'data', 'paddle_finetune', 'crops')
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, 'data', 'paddle_finetune', 'dataset')


def build_dict(label_texts, dataset_dir):
    """
    Build dict.txt from the characters actually present in the label set,
    rather than hardcoding "uppercase letters + digits" -- per the spec,
    manual labels can contain edge cases a hardcoded assumption would miss.
    """
    chars = sorted(set(''.join(label_texts)))
    dict_path = os.path.join(dataset_dir, 'dict.txt')
    with open(dict_path, 'w', encoding='utf-8') as f:
        for ch in chars:
            f.write(ch + '\n')
    return chars, dict_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--crops-dir', default=DEFAULT_CROPS_DIR)
    ap.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    ap.add_argument('--train-fraction', type=float, default=0.85,
                     help='Fraction of SOURCE IMAGES assigned to train (default 0.85, i.e. 85/15). '
                          'This is a judgment call, not a hard requirement -- a fine-tuning valid set '
                          'can reasonably be smaller than a benchmark valid set.')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    with open(args.manifest, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("FATAL: manifest is empty -- nothing to split.", file=sys.stderr)
        sys.exit(1)

    images_out_dir = os.path.join(args.dataset_dir, 'images')
    os.makedirs(images_out_dir, exist_ok=True)

    # Split by source VIDEO ID, not by crop and not by exact image path --
    # see module docstring for the exact-duplicate-file case this guards
    # against (same photo, two different Roboflow hash filenames).
    video_id_of = lambda row: group_id(row['source_image_basename'])
    video_ids = sorted({video_id_of(row) for row in rows})
    rng = random.Random(args.seed)
    rng.shuffle(video_ids)
    n_train = round(len(video_ids) * args.train_fraction)
    # Guarantee both sides are non-empty when there's more than one video.
    if len(video_ids) > 1:
        n_train = min(max(n_train, 1), len(video_ids) - 1)
    train_video_ids = set(video_ids[:n_train])
    val_video_ids = set(video_ids[n_train:])

    print(f"Source videos: {len(video_ids)} total -> {len(train_video_ids)} train / {len(val_video_ids)} val")

    chars, dict_path = build_dict([row['label_text'] for row in rows], args.dataset_dir)
    print(f"dict.txt: {len(chars)} distinct characters -> {dict_path}")

    train_lines, val_lines = [], []
    for row in rows:
        src_crop = os.path.join(args.crops_dir, row['crop_filename'])
        dst_crop = os.path.join(images_out_dir, row['crop_filename'])
        shutil.copyfile(src_crop, dst_crop)

        rel_path = os.path.join('images', row['crop_filename'])
        line = f"{rel_path}\t{row['label_text']}"
        vid = video_id_of(row)
        if vid in train_video_ids:
            train_lines.append(line)
        elif vid in val_video_ids:
            val_lines.append(line)
        else:
            print(f"FATAL: video id {vid!r} landed in neither split -- logic bug.", file=sys.stderr)
            sys.exit(1)

    for name, lines in (('train.txt', train_lines), ('val.txt', val_lines)):
        path = os.path.join(args.dataset_dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"Wrote {len(lines)} lines to {path}")

    # Sanity-check: every character in every label appears in dict.txt.
    dict_chars = set(chars)
    for row in rows:
        missing = set(row['label_text']) - dict_chars
        if missing:
            print(f"FATAL: label {row['label_text']!r} has characters not in dict.txt: {missing}", file=sys.stderr)
            sys.exit(1)
    print("Sanity check OK: every label character is present in dict.txt.")


if __name__ == '__main__':
    main()
