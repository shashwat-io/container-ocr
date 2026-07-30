"""
inventory_videos.py

Builds a full video-level inventory across the current (contaminated)
train/valid/test split, using the same source_video_id() logic that
found the leakage. Output feeds directly into the resplit script.

Usage:
    python inventory_videos.py --dataset-root /path/to/container.v2i.yolov11 \
                                --out video_inventory.csv
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict

# Reuse the exact heuristic that already found the contamination —
# do not reimplement, import it.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "paddle_finetune"))
from video_safety import group_id


SPLITS = ["train", "valid", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def count_boxes(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    with open(label_path) as f:
        return sum(1 for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True,
                     help="Path to the canonical dataset folder "
                          "(the one confirmed as canonical, not the duplicate)")
    ap.add_argument("--out", default="video_inventory.csv")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    # video_id -> {"splits": set(), "images": int, "boxes": int, "files": []}
    inventory = defaultdict(lambda: {"splits": set(), "images": 0, "boxes": 0, "files": []})

    total_images_seen = 0
    unmatched = []

    for split in SPLITS:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists():
            print(f"WARNING: {img_dir} does not exist, skipping split '{split}'")
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            total_images_seen += 1

            vid = group_id(img_path.name)
            if vid is None:
                unmatched.append(str(img_path))
                continue

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            boxes = count_boxes(lbl_path)

            entry = inventory[vid]
            entry["splits"].add(split)
            entry["images"] += 1
            entry["boxes"] += boxes
            entry["files"].append(str(img_path))

    # Write inventory CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "current_splits", "n_images", "n_boxes"])
        for vid, data in sorted(inventory.items()):
            writer.writerow([
                vid,
                "|".join(sorted(data["splits"])),
                data["images"],
                data["boxes"],
            ])

    # Summary
    print(f"Total images scanned: {total_images_seen}")
    print(f"Unmatched (no video_id extracted): {len(unmatched)}")
    if unmatched:
        print("First few unmatched files (investigate before proceeding):")
        for u in unmatched[:10]:
            print(f"  {u}")
    print(f"Unique video_ids found: {len(inventory)}")

    multi_split = [v for v, d in inventory.items() if len(d["splits"]) > 1]
    print(f"Video_ids currently spanning >1 split (the contamination): {len(multi_split)}")

    total_images = sum(d["images"] for d in inventory.values())
    total_boxes = sum(d["boxes"] for d in inventory.values())
    print(f"Total images across all video_ids: {total_images}")
    print(f"Total boxes across all video_ids: {total_boxes}")
    print(f"\nInventory written to {args.out}")


if __name__ == "__main__":
    main()
