"""
build_resplit.py

Materializes the new leakage-free train/valid/test folder structure from
split_assignment_v2.csv. Copies (does not move) files — original dataset
is left fully intact.

Usage:
    python build_resplit.py --dataset-root container.v2i.yolov11 \
                             --assignment split_assignment_v2.csv \
                             --out container.v2i.yolov11.resplit
"""
import argparse
import csv
import re
import shutil
from pathlib import Path

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def pre_hash_stem(filename: str) -> str:
    return re.sub(r"\.rf\.[0-9a-f]+$", "", Path(filename).stem)


def group_id(filename: str) -> str:
    stem = Path(filename).stem
    first_token = stem.split("_")[0]
    if first_token.isdigit():
        return first_token
    return pre_hash_stem(filename)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.dataset_root)
    out_root = Path(args.out)

    if out_root.exists():
        raise SystemExit(f"ERROR: {out_root} already exists — refusing to overwrite. "
                          f"Remove it manually first if you want to rebuild.")

    # Load assignment
    assignment = {}
    with open(args.assignment) as f:
        reader = csv.DictReader(f)
        for row in reader:
            assignment[row["video_id"]] = row["new_split"]

    # Create output structure
    for split in SPLITS:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    copied = {"train": 0, "valid": 0, "test": 0}
    missing_labels = []
    unassigned = []

    for split in SPLITS:  # source splits — irrelevant now, we're re-sorting everything
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            gid = group_id(img_path.name)
            new_split = assignment.get(gid)
            if new_split is None:
                unassigned.append(str(img_path))
                continue

            # Copy image
            dst_img = out_root / new_split / "images" / img_path.name
            shutil.copy2(img_path, dst_img)

            # Copy matching label file, if present
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                dst_lbl = out_root / new_split / "labels" / lbl_path.name
                shutil.copy2(lbl_path, dst_lbl)
            else:
                missing_labels.append(str(img_path))

            copied[new_split] += 1

    # Copy data.yaml as-is — relative paths (../train/images etc.) are still
    # correct since the resplit output preserves the same train/valid/test
    # + images/labels folder layout as the original dataset root.
    src_yaml = root / "data.yaml"
    if src_yaml.exists():
        shutil.copy2(src_yaml, out_root / "data.yaml")

    total_copied = sum(copied.values())

    print("Per-split copy counts:")
    for split in SPLITS:
        print(f"  {split:6s}: {copied[split]} images")
    print(f"\nTotal images copied: {total_copied}")
    print(f"Unassigned (group_id not found in assignment): {len(unassigned)}")
    if unassigned:
        print("First few unassigned files (investigate before proceeding):")
        for u in unassigned[:10]:
            print(f"  {u}")
    print(f"Missing labels (image with no matching .txt): {len(missing_labels)}")
    if missing_labels:
        print("First few images with missing labels:")
        for m in missing_labels[:10]:
            print(f"  {m}")

    print(f"\nResplit dataset written to {out_root}")


if __name__ == "__main__":
    main()
