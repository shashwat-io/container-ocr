"""
Detection-only proxy scan for label fill-down errors (per the confirmed
crop_1173 case: ground truth IDCU1004472 copy-pasted onto a crop that
actually shows CNCU152365[2]).

Does NOT modify manual_labels.csv, val.txt, or train.txt -- read-only.

First pass used video_safety.py's group_id() directly and produced a huge,
mostly-false-positive suspect list (22 val / 260 train). Investigating a
sample of those showed two systematic false-positive sources that group_id()
isn't designed to handle (it exists for train/test leakage safety, not
same-container identity):

  1. "Event<date><time><chunk>-Avi_<offset>_jpg" filenames (CCTV/dashcam
     frame exports): group_id()'s non-numeric-prefix fallback treats every
     distinct frame offset as its own group, so a single truck sitting in
     frame for several minutes while a fixed camera's NVR rolls over
     segment files looks like "many unrelated sources" when it's actually
     one continuous observation. Confirmed by inspecting the suspect list:
     labels repeat across near-identical timestamps and adjacent 5-minute
     segment chunks (e.g. 001456004 and 001456005 differ only in the
     chunk suffix) -- classic NVR segment rollover, not a labeling error.
  2. "IMG_<number>_ce_png_jpg" vs "IMG_<number>_region_JPG_jpg": these are
     two derivative exports of the literal same source photo (contrast-
     enhanced vs region-cropped versions), sharing the same IMG number --
     group_id()'s pre-hash-stem fallback incorrectly splits them because
     the suffix differs.

This script corrects for both: IMG_-prefixed sources are regrouped by IMG
number alone (ignoring export suffix), and Event-Avi/DSCF-prefixed sources
are tagged as a coarse "camera-roll" category so a same-category mismatch
(Event vs Event, DSCF vs DSCF) is reported at LOW confidence (explainable by
session/segment adjacency) rather than being conflated with a genuine
cross-device mismatch (e.g. a 40-frame numeric video ID vs a single IMG_
photo -- the crop_1173 fingerprint) which is reported at HIGH confidence.
"""
import csv
import re
import sys
from collections import defaultdict

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')

EVENT_RE = re.compile(r'^(Event\d+-Avi)_')
IMG_RE = re.compile(r'^(IMG_\d+)')
DSCF_RE = re.compile(r'^(DSCF\d+)')


def pre_hash_stem(filename):
    return re.sub(r'\.rf\.[0-9a-f]+$', '', filename.rsplit('.', 1)[0] if filename.count('.') > 1 else filename)


def classify(basename):
    """Returns (source_type, group_key). group_key is what we group same-
    source occurrences by; source_type drives the confidence heuristic."""
    stem = pre_hash_stem(basename)
    first_token = stem.split('_')[0]

    if first_token.isdigit():
        return 'video', first_token

    m = EVENT_RE.match(stem)
    if m:
        return 'cctv_event', m.group(1)  # session prefix, ignoring frame offset

    m = IMG_RE.match(stem)
    if m:
        return 'img_photo', m.group(1)  # IMG number only, ignoring export suffix

    m = DSCF_RE.match(stem)
    if m:
        return 'dscf_photo', m.group(1)

    if stem.startswith('WhatsApp'):
        return 'whatsapp_photo', stem

    if first_token.startswith('z') and first_token[1:].isdigit():
        return 'z_photo', stem

    return 'other_photo', stem


SINGLE_PHOTO_TYPES = {'img_photo', 'dscf_photo', 'whatsapp_photo', 'z_photo', 'other_photo'}


def confidence(dominant_type, dominant_n, outlier_type, outlier_n):
    if dominant_type == 'video' and outlier_type == 'video':
        return 'HIGH'  # two distinct, unrelated numeric video IDs sharing a label
    if {dominant_type, outlier_type} <= {'video', 'cctv_event'} and dominant_type != outlier_type:
        return 'HIGH'  # a real multi-frame video cluster vs a different one -- shouldn't happen if same container
    if dominant_type in ('video', 'cctv_event') and outlier_type in SINGLE_PHOTO_TYPES:
        return 'HIGH'  # multi-frame video/CCTV cluster vs one unrelated single photo -- the crop_1173 fingerprint
    if outlier_type in ('video', 'cctv_event') and dominant_type in SINGLE_PHOTO_TYPES:
        return 'HIGH'
    if dominant_type == outlier_type == 'cctv_event':
        return 'LOW'  # same camera, segment rollover -- explainable, not a suspect
    if dominant_type == outlier_type == 'dscf_photo':
        return 'LOW'  # same camera roll, sequential numbering
    return 'MEDIUM'


def load_split(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            img_path, label = line.split('\t', 1)
            entries.append((img_path.split('/')[-1], label))
    return entries


def analyze(name, entries, manifest):
    by_label = defaultdict(list)
    for crop_filename, label in entries:
        src_basename = manifest[crop_filename]
        stype, gkey = classify(src_basename)
        by_label[label].append((crop_filename, stype, gkey))

    total_rows = len(entries)
    unique_labels = len(by_label)
    repeated_labels = {lbl: occ for lbl, occ in by_label.items() if len(occ) > 1}
    repeated_rows = sum(len(occ) for occ in repeated_labels.values())

    high, medium, low = [], [], []
    single_group_count = 0

    for lbl, occ in repeated_labels.items():
        group_counts = defaultdict(list)
        for crop_filename, stype, gkey in occ:
            group_counts[(stype, gkey)].append(crop_filename)

        if len(group_counts) == 1:
            single_group_count += 1
            continue

        group_sizes = sorted(group_counts.items(), key=lambda kv: -len(kv[1]))
        (dom_type, dom_key), dom_crops = group_sizes[0]
        for (out_type, out_key), out_crops in group_sizes[1:]:
            conf = confidence(dom_type, len(dom_crops), out_type, len(out_crops))
            entry = {
                'label': lbl, 'confidence': conf,
                'dominant': f"{dom_type}:{dom_key}", 'dominant_n': len(dom_crops),
                'outlier': f"{out_type}:{out_key}", 'outlier_crops': out_crops,
            }
            {'HIGH': high, 'MEDIUM': medium, 'LOW': low}[conf].append(entry)

    print(f"\n=== {name} ===")
    print(f"Total rows: {total_rows} | Unique labels: {unique_labels}")
    print(f"Repeated labels (>1 occurrence): {len(repeated_labels)} ({repeated_rows} rows, "
          f"{100*repeated_rows/total_rows:.1f}% of rows)")
    print(f"  -> single-group repeats (all occurrences same source cluster): {single_group_count} labels")
    print(f"  -> HIGH-confidence suspects: {len(high)} outlier occurrences")
    print(f"  -> MEDIUM-confidence (worth a look): {len(medium)} outlier occurrences")
    print(f"  -> LOW-confidence (likely same-session artifact, not suspects): {len(low)} outlier occurrences")

    return high, medium, low


def main():
    manifest = {}
    with open('data/paddle_finetune/manifest.csv') as f:
        for row in csv.DictReader(f):
            manifest[row['crop_filename']] = row['source_image_basename']

    val = load_split('data/paddle_finetune/dataset/val.txt')
    train = load_split('data/paddle_finetune/dataset/train.txt')

    val_high, val_med, val_low = analyze('VAL.TXT (300 rows)', val, manifest)
    train_high, train_med, train_low = analyze('TRAIN.TXT (1670 rows)', train, manifest)

    def dump(name, items):
        print(f"\n--- {name} ({len(items)}) ---")
        for e in items:
            print(f"  label={e['label']!r}  dominant={e['dominant']}(n={e['dominant_n']})  "
                  f"outlier={e['outlier']}  crops={e['outlier_crops']}")

    dump('VAL HIGH-confidence suspects', val_high)
    dump('VAL MEDIUM-confidence', val_med)
    dump('TRAIN HIGH-confidence suspects', train_high)
    dump('TRAIN MEDIUM-confidence', train_med)

    print(f"\n(LOW-confidence, not printed: {len(val_low)} val + {len(train_low)} train "
          f"outlier occurrences -- same-camera-session artifacts, e.g. Event-Avi segment "
          f"rollover or DSCF camera-roll adjacency)")


if __name__ == '__main__':
    main()
