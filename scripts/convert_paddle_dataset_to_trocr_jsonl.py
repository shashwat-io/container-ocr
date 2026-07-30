"""
Converts the existing PaddleOCR fine-tune dataset (data/paddle_finetune/
dataset/{train,val}.txt, tab-delimited "image_path\\tlabel") into JSONL
usable by a Hugging Face Seq2SeqTrainer TrOCR fine-tuning script: one JSON
object per line, keys `image_path` and `text`.

Reuses the exact same crops and train/val split -- no reshuffling, no new
labeling, no copying of image files (image_path points back at the
original data/paddle_finetune/dataset/images/ location). Purely additive:
does not touch data/paddle_finetune/ at all.

Reads:  data/paddle_finetune/dataset/train.txt (1668 rows)
        data/paddle_finetune/dataset/val.txt (300 rows)
Writes: data/trocr_finetune/train.jsonl
        data/trocr_finetune/val.jsonl
"""
import json
import os
import sys

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

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
            image_path = os.path.join(PADDLE_DATASET_DIR, rel_path)
            if not os.path.exists(image_path):
                missing.append(image_path)
            f_out.write(json.dumps({'image_path': image_path, 'text': text}) + '\n')
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
    if train_missing or val_missing:
        print("MISSING FILES (first 10):", (train_missing + val_missing)[:10])


if __name__ == '__main__':
    main()
