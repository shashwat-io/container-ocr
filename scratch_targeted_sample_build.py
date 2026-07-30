#!/usr/bin/env python3
"""
READ-ONLY (no OCR). Builds the targeted row list for the time-boxed
tier3-pass2-fix spot check: all ~71 former Tier3-Pass2 rows + a random
control sample from the rest of the dataset (seed=42).

Source: benchmark_results.csv, which still holds the OLD (pre-tier3fix,
610/786 two-pass) data — confirmed byte-identical to
benchmark_results_pre_tier3fix.csv via md5sum earlier.
"""
import csv, random
from collections import Counter
from scratch_tier_classify import parse_texts, classify

SRC_CSV = 'benchmark_results.csv'
rows = list(csv.DictReader(open(SRC_CSV)))
seen = Counter()
keyed_rows = []
for r in rows:
    f = r['filename']; idx = seen[f]; seen[f] += 1
    keyed_rows.append((f, idx, r))

tier3pass2 = []
rest = []
for f, idx, r in keyed_rows:
    if r['is_valid'].strip().lower() == 'true':
        texts = parse_texts(r['ocr_raw'])
        code, bucket, tier, conf = classify(texts)
        if tier == 'Tier3-Pass2':
            tier3pass2.append((f, idx, r))
            continue
    rest.append((f, idx, r))

print(f"Tier3-Pass2 rows: {len(tier3pass2)}")
print(f"Rest of dataset: {len(rest)}")

random.seed(42)
control_sample = random.sample(rest, min(60, len(rest)))
print(f"Control sample: {len(control_sample)}")

targeted = [(f, idx, 'tier3pass2') for f, idx, _ in tier3pass2] + \
           [(f, idx, 'control') for f, idx, _ in control_sample]
print(f"Total targeted rows: {len(targeted)}")

with open('scratch_targeted_sample_rows.csv', 'w', newline='') as out:
    w = csv.writer(out)
    w.writerow(['filename', 'box_idx', 'group'])
    for f, idx, g in targeted:
        w.writerow([f, idx, g])
print("Written to scratch_targeted_sample_rows.csv")
