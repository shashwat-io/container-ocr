"""
diagnose_img_bucket.py

Checks whether the "IMG" video_id bucket is one genuinely duplicate-heavy
cluster or a false merge of many distinct source photos sharing a generic
camera-default filename prefix. Also checks whether other non-numeric
video_id values show the same collapse risk.

Usage:
    python diagnose_img_bucket.py --dataset-root container.v2i.yolov11 \
                                   --inventory video_inventory.csv
"""
import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict

SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def pre_hash_stem(filename: str) -> str:
    """Strip Roboflow's '.rf.<hash>' suffix to get the original filename stem."""
    return re.sub(r"\.rf\.[0-9a-f]+$", "", Path(filename).stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--inventory", required=True)
    args = ap.parse_args()

    root = Path(args.dataset_root)

    # Identify all non-numeric video_ids from the inventory (potential collapse risks)
    non_numeric_ids = []
    with open(args.inventory) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row["video_id"]
            if not vid.isdigit():
                non_numeric_ids.append((vid, int(row["n_images"])))

    print("Non-numeric video_ids found (potential prefix-collapse risk):")
    for vid, n in sorted(non_numeric_ids, key=lambda x: -x[1]):
        print(f"  {vid}: {n} images")
    print()

    # For each non-numeric id, check how many DISTINCT pre-hash stems exist
    for target_vid, _ in sorted(non_numeric_ids, key=lambda x: -x[1]):
        stem_counts = defaultdict(int)
        stem_splits = defaultdict(set)

        for split in SPLITS:
            img_dir = root / split / "images"
            if not img_dir.exists():
                continue
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                first_token = img_path.stem.split("_")[0]
                if first_token != target_vid:
                    continue
                stem = pre_hash_stem(img_path.name)
                stem_counts[stem] += 1
                stem_splits[stem].add(split)

        n_distinct_stems = len(stem_counts)
        n_total_images = sum(stem_counts.values())
        avg_copies = n_total_images / n_distinct_stems if n_distinct_stems else 0
        stems_spanning_multiple_splits = sum(1 for s in stem_splits.values() if len(s) > 1)

        print(f"--- video_id='{target_vid}' ---")
        print(f"  Total images: {n_total_images}")
        print(f"  Distinct pre-hash stems (likely distinct source photos): {n_distinct_stems}")
        print(f"  Avg copies per distinct photo: {avg_copies:.1f}")
        print(f"  Distinct photos with copies spanning >1 split: {stems_spanning_multiple_splits}")
        if n_distinct_stems > 1:
            print(f"  -> Verdict: likely a FALSE MERGE of {n_distinct_stems} distinct photos. "
                  f"Grouping by '{target_vid}' alone is too coarse.")
        else:
            print(f"  -> Verdict: single source photo, heavily duplicated. "
                  f"Current grouping is directionally correct.")
        print()


if __name__ == "__main__":
    main()
