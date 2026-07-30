import os
import csv
import time
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

from ultralytics import YOLO
from app.inference.ocr import extract_container_code

TEST_DIR = '/home/shashwatkumar/Downloads/container.v2i.yolov11/test/images'
OUTPUT_CSV = '/home/shashwatkumar/Testing/container_ocr_test/benchmark_results.csv'

model = YOLO('models/best.pt')

images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
total = len(images)
print(f"Found {total} images", flush=True)

stats = {'detected': 0, 'ocr_read': 0, 'valid': 0, 'no_detection': 0}
start = time.time()
rows = []

for i, fname in enumerate(images, 1):
    path = os.path.join(TEST_DIR, fname)
    try:
        img_pil = Image.open(path).convert('RGB')
        img_np = np.array(img_pil)
        with open(path, 'rb') as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"  Load error {fname}: {e}", flush=True)
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                     'container_code': '', 'is_valid': False, 'ocr_raw': f'load error: {e}'})
        continue

    try:
        results = model(img_np, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                boxes.append((x1, y1, x2, y2))
    except Exception as e:
        print(f"  YOLO error {fname}: {e}", flush=True)
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                     'container_code': '', 'is_valid': False, 'ocr_raw': f'YOLO error: {e}'})
        continue

    if not boxes:
        stats['no_detection'] += 1
        rows.append({'filename': fname, 'detected': False, 'num_boxes': 0,
                     'container_code': '', 'is_valid': False, 'ocr_raw': ''})
    else:
        stats['detected'] += 1
        for x1, y1, x2, y2 in boxes:
            try:
                result = extract_container_code(image_bytes, x1, y1, x2, y2)
                if result['container_code']:
                    stats['ocr_read'] += 1
                if result['is_valid']:
                    stats['valid'] += 1
                rows.append({
                    'filename': fname,
                    'detected': True,
                    'num_boxes': len(boxes),
                    'container_code': result['container_code'],
                    'is_valid': result['is_valid'],
                    'ocr_raw': result['ocr_raw_output'],
                })
            except Exception as e:
                rows.append({'filename': fname, 'detected': True, 'num_boxes': len(boxes),
                             'container_code': '', 'is_valid': False, 'ocr_raw': f'OCR error: {e}'})

    if i % 50 == 0:
        elapsed = time.time() - start
        print(f"[{i}/{total}] {elapsed:.0f}s elapsed | avg {elapsed/i:.1f}s/img | valid so far: {stats['valid']}", flush=True)

elapsed = time.time() - start

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['filename','detected','num_boxes','container_code','is_valid','ocr_raw'])
    writer.writeheader()
    writer.writerows(rows)

d = stats['detected'] or 1
print(f"\n=== BENCHMARK RESULTS ===", flush=True)
print(f"Total images:     {total}", flush=True)
print(f"YOLO detected:    {stats['detected']} ({stats['detected']/total*100:.1f}%)", flush=True)
print(f"OCR read code:    {stats['ocr_read']} ({stats['ocr_read']/d*100:.1f}% of detected)", flush=True)
print(f"ISO 6346 valid:   {stats['valid']} ({stats['valid']/d*100:.1f}% of detected)", flush=True)
print(f"No detection:     {stats['no_detection']} ({stats['no_detection']/total*100:.1f}%)", flush=True)
print(f"Total time:       {elapsed:.0f}s | Avg per image: {elapsed/total:.1f}s", flush=True)
print(f"Results saved to: {OUTPUT_CSV}", flush=True)
