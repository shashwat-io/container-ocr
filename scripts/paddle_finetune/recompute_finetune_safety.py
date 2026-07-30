"""
recompute_finetune_safety.py

Recomputes fine-tuning crop safety using the corrected group_id() grouping
against the NEW resplit test set, instead of the old source_video_id()
logic against the OLD contaminated test set. Pure analysis — does not
modify manual_labels.csv, the crop manifest, or any extraction output.

Usage:
    python recompute_finetune_safety.py \
        --manual-labels training_labels/manual_labels.csv \
        --resplit-test-dir container.v2i.yolov11.resplit/test/images \
        --manifest data/paddle_finetune/manifest.csv \
        --out finetune_safety_recompute.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, "scripts/paddle_finetune")
from video_safety import group_id  # corrected grouping logic


def build_corrected_test_groups(test_images_dir):
    """Distinct group_ids present anywhere in the NEW resplit test split."""
    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(f"test images dir not found: {test_images_dir}")
    return {group_id(f) for f in os.listdir(test_images_dir)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual-labels", required=True)
    ap.add_argument("--resplit-test-dir", required=True)
    ap.add_argument("--manifest", required=False,
                     help="Path to the current 463-crop manifest.csv, if available, "
                          "to cross-reference which specific crops are affected")
    ap.add_argument("--out", default="finetune_safety_recompute.csv")
    args = ap.parse_args()

    corrected_test_groups = build_corrected_test_groups(args.resplit_test_dir)
    print(f"Corrected test-split group_ids: {len(corrected_test_groups)}")

    rows = []
    with open(args.manual_labels) as f:
        reader = csv.DictReader(f)
        for row in reader:
            basename = os.path.basename(row["image_path"])
            gid = group_id(basename)
            corrected_safe = gid not in corrected_test_groups
            rows.append({**row, "group_id": gid, "corrected_safe": corrected_safe})

    total = len(rows)
    corrected_safe_count = sum(1 for r in rows if r["corrected_safe"])
    corrected_unsafe_count = total - corrected_safe_count

    print(f"\nTotal manual_labels.csv rows: {total}")
    print(f"Corrected SAFE: {corrected_safe_count}")
    print(f"Corrected UNSAFE: {corrected_unsafe_count}")

    with open(args.out, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull recompute written to {args.out}")

    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            print(f"\nManifest columns found: {fieldnames}")
            candidate_cols = [c for c in fieldnames
                               if "image" in c.lower() or "path" in c.lower() or "source" in c.lower()]
            if not candidate_cols:
                print("WARNING: could not auto-detect an image-path-like column in manifest.csv — "
                      "skipping cross-reference. Inspect manifest.csv columns manually.")
            else:
                col = candidate_cols[0]
                print(f"Using manifest column '{col}' as the image reference")
                manifest_basenames = {os.path.basename(row[col]) for row in reader}

                unsafe_by_basename = {os.path.basename(r["image_path"]) for r in rows if not r["corrected_safe"]}
                flipped_to_unsafe = manifest_basenames & unsafe_by_basename
                print(f"\nCurrent manifest crops: {len(manifest_basenames)}")
                print(f"Of those, now flip to UNSAFE under corrected logic: {len(flipped_to_unsafe)}")
                if flipped_to_unsafe:
                    print("These need to be pulled from the fine-tuning set:")
                    for b in sorted(flipped_to_unsafe)[:20]:
                        print(f"  {b}")

                growth_candidates = [
                    r for r in rows
                    if r["corrected_safe"]
                    and os.path.basename(r["image_path"]) not in manifest_basenames
                    and r.get("status") == "labeled"
                    and r.get("checksum_valid") == "True"
                ]
                print(f"\nNewly-safe rows not currently in manifest, meeting quality bar "
                      f"(status=labeled, checksum_valid=True): {len(growth_candidates)}")
                print("(Upper-bound estimate only — still needs the same box-level quality "
                      "filtering extract_crops.py applies, e.g. clipped/skip exclusions, before "
                      "treating this as a final pool-growth number.)")


if __name__ == "__main__":
    main()
