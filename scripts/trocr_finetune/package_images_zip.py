"""
Builds a clean zip of only the crop images referenced by
data/trocr_finetune/{train,val}.jsonl, for manual upload to Drive.

Parses file_name out of every JSONL row, verifies every referenced file
exists under data/paddle_finetune/dataset/, then zips them preserving the
"images/<file>" relative path so the zip extracts directly to an images/
folder. The distinct-file count is derived from the JSONL files themselves
at run time rather than hardcoded, since it changes every time the dataset
is resized (it has already drifted from an earlier hardcoded value once).
Pass --expected-count for an explicit sanity check against a known number.

Reads:  data/trocr_finetune/train.jsonl
        data/trocr_finetune/val.jsonl
        data/paddle_finetune/dataset/images/*
Writes: data/trocr_finetune/images_<count>.zip
"""
import argparse
import json
import os
import zipfile

DATASET_DIR = 'data/paddle_finetune/dataset'
JSONL_DIR = 'data/trocr_finetune'


def collect_file_names(*jsonl_paths):
    names = []
    for path in jsonl_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                names.append(json.loads(line)['file_name'])
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jsonl-dir', default=JSONL_DIR,
                     help='Directory containing train.jsonl/val.jsonl; the output '
                          'zip is written here too. Defaults to data/trocr_finetune.')
    ap.add_argument('--dataset-dir', default=DATASET_DIR,
                     help='Directory the file_name paths in the JSONLs resolve '
                          'against (where the actual crop images live).')
    ap.add_argument('--expected-count', type=int, default=None,
                     help='Optional sanity check: abort if the distinct file count '
                          'read from the JSONLs does not match this. Omit to skip '
                          'the check and just use whatever count is found.')
    args = ap.parse_args()

    all_names = collect_file_names(
        os.path.join(args.jsonl_dir, 'train.jsonl'),
        os.path.join(args.jsonl_dir, 'val.jsonl'),
    )
    distinct_names = sorted(set(all_names))

    print(f"Total file_name rows read: {len(all_names)}")
    print(f"Distinct file_name values: {len(distinct_names)}")
    if len(all_names) != len(distinct_names):
        print(f"WARNING: {len(all_names) - len(distinct_names)} duplicate file_name value(s) across train/val")

    if args.expected_count is not None and len(distinct_names) != args.expected_count:
        print(f"STOP: expected {args.expected_count} distinct files, got {len(distinct_names)}. Not proceeding.")
        return

    out_zip = os.path.join(args.jsonl_dir, f'images_{len(distinct_names)}.zip')

    missing = []
    for rel in distinct_names:
        if not os.path.exists(os.path.join(args.dataset_dir, rel)):
            missing.append(rel)

    if missing:
        print(f"MISSING FILES ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
        print("STOP: not zipping while files are missing.")
        return

    print("All referenced files verified present. Zipping...")
    os.makedirs(args.jsonl_dir, exist_ok=True)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel in distinct_names:
            src = os.path.join(args.dataset_dir, rel)
            zf.write(src, arcname=rel)

    zipped_count = len(zipfile.ZipFile(out_zip).namelist())
    size_bytes = os.path.getsize(out_zip)
    print(f"Wrote {out_zip}")
    print(f"Files in zip: {zipped_count}")
    print(f"Zip size: {size_bytes} bytes ({size_bytes / 1024 / 1024:.2f} MB)")


if __name__ == '__main__':
    main()
