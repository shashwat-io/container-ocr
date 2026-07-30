"""
leakage_check.py

Re-validates a dataset split for cross-split video/photo leakage using
group_id() (the corrected grouping logic). Run against the new resplit
folder to confirm zero contamination; can also be re-run against the
original folder to reproduce Friday's numbers as a sanity check.

Usage:
    python leakage_check.py --dataset-root container.v2i.yolov11.resplit
"""
import argparse
import re
from pathlib import Path
from collections import defaultdict

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
    args = ap.parse_args()

    root = Path(args.dataset_root)
    group_splits = defaultdict(set)      # group_id -> set of splits it appears in
    group_split_images = defaultdict(lambda: defaultdict(int))  # group_id -> split -> image count

    for split in SPLITS:
        img_dir = root / split / "images"
        if not img_dir.exists():
            print(f"WARNING: {img_dir} not found, skipping")
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            gid = group_id(img_path.name)
            group_splits[gid].add(split)
            group_split_images[gid][split] += 1

    # Pairwise overlap
    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
    print(f"Leakage check for: {root}\n")

    total_leaked_groups = 0
    for a, b in pairs:
        overlap_groups = [g for g, splits in group_splits.items() if a in splits and b in splits]
        a_images_affected = sum(group_split_images[g][a] for g in overlap_groups)
        b_images_affected = sum(group_split_images[g][b] for g in overlap_groups)
        total_a = sum(v[a] for v in group_split_images.values())
        total_b = sum(v[b] for v in group_split_images.values())
        pct_a = (a_images_affected / total_a * 100) if total_a else 0.0
        pct_b = (b_images_affected / total_b * 100) if total_b else 0.0

        print(f"{a} ∩ {b}: {len(overlap_groups)} group_ids overlap")
        print(f"  {a}: {a_images_affected}/{total_a} images affected ({pct_a:.1f}%)")
        print(f"  {b}: {b_images_affected}/{total_b} images affected ({pct_b:.1f}%)")
        print()

    all_three = [g for g, splits in group_splits.items() if len(splits) == 3]
    print(f"Group_ids appearing in all three splits: {len(all_three)}")

    multi_split_groups = [g for g, splits in group_splits.items() if len(splits) > 1]
    print(f"\nTotal group_ids spanning >1 split: {len(multi_split_groups)}")
    print(f"Total distinct group_ids: {len(group_splits)}")

    if multi_split_groups:
        print(f"\nRESULT: FAIL — {len(multi_split_groups)} group_ids leak across splits.")
    else:
        print(f"\nRESULT: PASS — zero group_ids span more than one split.")


if __name__ == "__main__":
    main()
