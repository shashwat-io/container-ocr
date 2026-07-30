"""
Read-only reproduction of the 52 candidate rows to determine, per row, which
rotation/attempt inside ocr.py's _best_portrait_rotation actually produced the
RETURNED result -- the benchmark's tier column is known-approximate (last
'Tier ' log line across ALL rotation attempts, not necessarily the winner).

Monkeypatches only in this process's memory (same technique
benchmark_trocr_cascade_quicksample.py already uses for run_ocr_on_crop).
Does not write to app/ or anywhere outside scratchpad/.
"""
import os, sys, csv, json, logging
from collections import Counter, defaultdict

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')
os.environ['FINETUNED_TROCR_MODEL_DIR'] = os.path.expanduser('~/Downloads/trocr_checkpoint_thin')

from ultralytics import YOLO
from PIL import Image
import numpy as np
import app.inference.ocr as ocr
from app.inference.ocr_trocr import get_trocr, run_trocr_on_crop

def _run_trocr_as_ocr_line(crop_image, docpp_off=False):
    text = run_trocr_on_crop(crop_image)
    if not text.strip():
        return []
    return [(text, 1.0)]
ocr.run_ocr_on_crop = _run_trocr_as_ocr_line

# ── Per-call capture: wrap _try_extract_best_valid to record (crop, result) ──
_capture = []
_real_tebv = ocr._try_extract_best_valid
def _wrapped_tebv(crop):
    result = _real_tebv(crop)
    _capture.append((crop, result))
    return result
ocr._try_extract_best_valid = _wrapped_tebv
ocr._best_portrait_rotation.__globals__['_try_extract_best_valid'] = _wrapped_tebv

# ── Per-call tier-line capture, segmented by _try_extract_best_valid call ───
_tier_log = defaultdict(list)  # call_index -> [tier messages]
class SegmentedTierHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith('Tier '):
            _tier_log[len(_capture)].append(msg.split(':')[0].strip())
ocr_logger = logging.getLogger('app.inference.ocr')
ocr_logger.setLevel(logging.INFO)
ocr_logger.addHandler(SegmentedTierHandler())

print("Loading TrOCR + YOLO...", flush=True)
_, _, device = get_trocr()
model = YOLO('models/best_resplit.pt')
print(f"Ready (device={device}).", flush=True)

# ── Load the 52 target rows ──────────────────────────────────────────────────
rows = list(csv.DictReader(open('benchmark_results_trocr_cascade_FULL_thin_20260728.csv')))
seen = Counter()
targets = []
for r in rows:
    fn = r['filename']
    bi = seen[fn]; seen[fn] += 1
    v = str(r.get('is_valid','')).strip().lower() == 'true'
    if r.get('tier') == 'Tier3-fallback-novalid' and v:
        targets.append({'filename': fn, 'box_index': bi, 'csv_code': r['container_code'],
                         'csv_reason': r['validation_reason'], 'csv_raw': r['ocr_raw']})
print(f"Target rows: {len(targets)}", flush=True)

TEST_DIR = 'container.v2i.yolov11.resplit/test/images'
by_file = defaultdict(list)
for t in targets:
    by_file[t['filename']].append(t)

results = []
mismatches = []
for fname, ts in sorted(by_file.items()):
    print(f'  processing {fname}...', flush=True)
    path = os.path.join(TEST_DIR, fname)
    with open(path, 'rb') as f:
        image_bytes = f.read()
    img_pil = Image.open(path).convert('RGB')
    img_np = np.array(img_pil)
    yres = model(img_np, verbose=False)
    boxes = []
    for r in yres:
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            boxes.append((x1, y1, x2, y2))

    for t in ts:
        bi = t['box_index']
        if bi >= len(boxes):
            print(f"  WARNING: box_index {bi} out of range for {fname} ({len(boxes)} boxes)")
            continue
        x1, y1, x2, y2 = boxes[bi]

        _capture.clear()
        _tier_log.clear()
        res = ocr.extract_container_code(image_bytes, x1, y1, x2, y2)

        match = (res['container_code'] == t['csv_code'] and
                 res['is_valid'] == True and
                 res['validation_reason'] == t['csv_reason'])
        if not match:
            mismatches.append((fname, bi, t, res))

        # Identify which captured (crop, result) call matches the final return
        winner_idx = None
        for i, (crop, cresult) in enumerate(_capture):
            code, valid, reason, raw, conf = cresult
            if (code == res['container_code'] and valid == res['is_valid']
                    and reason == res['validation_reason']):
                winner_idx = i
                break

        n_calls = len(_capture)
        rotation = None
        if n_calls == 2:
            rotation = '+90' if winner_idx == 0 else ('-90' if winner_idx == 1 else '?')
        elif n_calls == 1:
            rotation = 'landscape (single attempt)'
        elif n_calls == 0:
            rotation = 'NO CAPTURE (unexpected)'
        else:
            rotation = f'{n_calls} calls (unexpected)'

        true_tiers = _tier_log.get(winner_idx, ['(none captured)']) if winner_idx is not None else ['(winner not identified)']

        crop_img = _capture[winner_idx][0] if winner_idx is not None else None
        crop_path = None
        if crop_img is not None:
            safe_name = fname.replace('.jpg','').replace('.rf.','_rf_')[:60]
            crop_path = f'scratchpad/tier3_crops/{safe_name}_box{bi}.png'
            crop_img.save(crop_path)

        results.append({
            'filename': fname, 'box_index': bi,
            'container_code': res['container_code'],
            'ocr_raw': res['ocr_raw_output'],
            'benchmark_tier_label': 'Tier3-fallback-novalid',  # what the CSV said
            'true_tier_chain': true_tiers,                     # what actually fired on the WINNING attempt
            'rotation_used': rotation,
            'n_rotation_calls': n_calls,
            'crop_path': crop_path,
            'reproduced_exactly': match,
        })

print(f"\nProcessed {len(results)} rows. Mismatches vs original CSV: {len(mismatches)}")
for m in mismatches[:10]:
    print("  MISMATCH:", m)

c = Counter(tuple(r['true_tier_chain']) for r in results)
print("\nTRUE originating tier (winning attempt only), corrected:")
for k, n in c.most_common():
    print(f"  {n:>3}  {k}")

json.dump(results, open('scratchpad/tier3_review_data.json','w'), indent=2)
print("\nSaved scratchpad/tier3_review_data.json")
