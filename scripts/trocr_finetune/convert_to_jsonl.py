"""
Converts the existing PaddleOCR fine-tune dataset (data/paddle_finetune/
dataset/{train,val}.txt, tab-delimited "images/<crop>.jpg\\t<label>") into
HF-compatible JSONL for the TrOCR fine-tuning Seq2SeqTrainer dataset class:
one JSON object per line, {"file_name": <path relative to dataset root>,
"text": <label>}.

Reuses the exact same 1668/300 crops and split -- no reshuffling, no new
labeling. file_name is kept relative to the dataset root (e.g.
"images/crop_...jpg") rather than absolute, so the JSONL stays valid once
copied to Drive alongside its images/ folder, regardless of local path.

Reads:  data/paddle_finetune/dataset/train.txt (1668 rows)
        data/paddle_finetune/dataset/val.txt (300 rows)
Writes: data/trocr_finetune/train.jsonl
        data/trocr_finetune/val.jsonl
"""
import json
import os

PADDLE_DATASET_DIR = 'data/paddle_finetune/dataset'
OUT_DIR = 'data/trocr_finetune'


def convert(src_txt, dst_jsonl):
    n = 0
    missing = []
    with open(src_txt) as f_in, open(dst_jsonl, 'w') as f_out:
        for line in f_in:
            line = line.rstrip('\n')
            if not line:
                continue
            rel_path, text = line.split('\t')
            abs_check_path = os.path.join(PADDLE_DATASET_DIR, rel_path)
            if not os.path.exists(abs_check_path):
                missing.append(rel_path)
            f_out.write(json.dumps({'file_name': rel_path, 'text': text}) + '\n')
            n += 1
    return n, missing


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_n, train_missing = convert(
        os.path.join(PADDLE_DATASET_DIR, 'train.txt'),
        os.path.join(OUT_DIR, 'train.jsonl'),
    )
    val_n, val_missing = convert(
        os.path.join(PADDLE_DATASET_DIR, 'val.txt'),
        os.path.join(OUT_DIR, 'val.jsonl'),
    )

    print(f"train.jsonl: {train_n} rows written, {len(train_missing)} missing image files")
    print(f"val.jsonl:   {val_n} rows written, {len(val_missing)} missing image files")
    if train_missing:
        print(f"MISSING FILES referenced by train.txt ({len(train_missing)}):")
        for p in train_missing:
            print(f"  {p}")
    if val_missing:
        print(f"MISSING FILES referenced by val.txt ({len(val_missing)}):")
        for p in val_missing:
            print(f"  {p}")


if __name__ == '__main__':
    main()
