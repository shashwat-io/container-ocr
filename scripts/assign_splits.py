"""
assign_splits.py

Assigns each video_id to exactly one of train/valid/test using greedy
largest-first bin balancing against target image-count ratios. Reads the
output of inventory_videos.py, writes a new video_id -> split mapping.

Usage:
    python assign_splits.py --inventory video_inventory.csv \
                             --out split_assignment.csv \
                             --seed 42
"""
import argparse
import csv
import random

TARGET_RATIOS = {"train": 4501 / 6289, "valid": 1002 / 6289, "test": 786 / 6289}
DOMINANCE_THRESHOLD = 0.05  # flag any single video accounting for >5% of total images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", default="split_assignment.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    videos = []  # (video_id, n_images, n_boxes)
    with open(args.inventory) as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append((row["video_id"], int(row["n_images"]), int(row["n_boxes"])))

    total_images = sum(v[1] for v in videos)

    # --- Dominance check: flag any video that alone could distort a split target ---
    dominant = [(vid, n_img, n_img / total_images) for vid, n_img, _ in videos
                if n_img / total_images > DOMINANCE_THRESHOLD]
    if dominant:
        print(f"WARNING: {len(dominant)} video_id(s) each account for more than "
              f"{DOMINANCE_THRESHOLD*100:.0f}% of total images — this can make target "
              f"ratios hard to hit cleanly and is worth a manual look:")
        for vid, n_img, pct in sorted(dominant, key=lambda x: -x[2]):
            print(f"  video_id={vid}: {n_img} images ({pct*100:.1f}% of total)")
        print()
    else:
        print(f"No single video exceeds {DOMINANCE_THRESHOLD*100:.0f}% of total images "
              f"— pool looks reasonably distributed.\n")

    # Also report the overall size distribution shape, since a handful of
    # medium-large videos collectively can distort things even if no single
    # one crosses the threshold above.
    sizes = sorted((v[1] for v in videos), reverse=True)
    top5_share = sum(sizes[:5]) / total_images
    print(f"Top 5 largest videos combined: {top5_share*100:.1f}% of total images "
          f"({sizes[:5]})\n")

    targets = {s: r * total_images for s, r in TARGET_RATIOS.items()}

    # Deterministic tie-breaking among equal-size videos
    rng = random.Random(args.seed)
    rng.shuffle(videos)
    videos.sort(key=lambda v: v[1], reverse=True)  # largest first, ties in shuffled order

    current = {"train": 0, "valid": 0, "test": 0}
    box_current = {"train": 0, "valid": 0, "test": 0}
    assignment = {}

    for vid, n_img, n_box in videos:
        # Assign to whichever split is furthest below its target (as a fraction of target)
        deficits = {s: targets[s] - current[s] for s in current}
        chosen = max(deficits, key=deficits.get)
        assignment[vid] = chosen
        current[chosen] += n_img
        box_current[chosen] += n_box

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "new_split"])
        for vid, split in sorted(assignment.items()):
            writer.writerow([vid, split])

    print("Target vs achieved (images):")
    for s in ["train", "valid", "test"]:
        pct_target = TARGET_RATIOS[s] * 100
        pct_actual = current[s] / total_images * 100
        diff = pct_actual - pct_target
        flag = "  <-- check this" if abs(diff) > 3 else ""
        print(f"  {s:6s}: target {pct_target:5.1f}%  actual {pct_actual:5.1f}%  "
              f"(diff {diff:+.1f}pt)  ({current[s]} images, {box_current[s]} boxes){flag}")

    print(f"\nAssignment written to {args.out}")


if __name__ == "__main__":
    main()
