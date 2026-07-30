"""
One-off feasibility smoke test: can ultralytics YOLO (torch), PaddleOCR
(paddlepaddle, via the existing get_ocr() in app/inference/ocr.py), and
TrOCR (transformers, torch) all be imported, initialized, and run inference
in the SAME process without one crashing/hanging the others?

This is deliberately NOT wired into predictor.py or any new ocr_trocr.py
module -- pure standalone feasibility check per the Day-20 request. Reuses
crop_detection() from app/inference/ocr.py (a pure helper, read-only import)
so the TrOCR crop is taken from the same box as the PaddleOCR crop.

Usage: python3 scripts/smoke_test_trocr_coexistence.py
"""
import os
import sys
import time
import traceback

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

TEST_IMAGE_DIR = '/home/shashwatkumar/Testing/container_ocr_test/container.v2i.yolov11.resplit/test/images'
YOLO_MODEL_PATH = 'models/best_resplit.pt'
TROCR_MODEL_ID = 'microsoft/trocr-base-printed'


def rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    return -1.0


def checkpoint(label, t0, mem_log):
    mem = rss_mb()
    elapsed = time.time() - t0
    mem_log.append((label, mem))
    print(f"[{elapsed:7.1f}s] {label:40s} RSS={mem:8.1f} MB")


def main():
    t0 = time.time()
    mem_log = []
    results = {}

    checkpoint("start", t0, mem_log)

    # ---- 1. Import all three frameworks (import-time cost/conflicts) ----
    print("\n=== Stage 1: imports ===")
    try:
        from ultralytics import YOLO
        checkpoint("import ultralytics", t0, mem_log)
    except Exception:
        print("IMPORT FAILED: ultralytics")
        traceback.print_exc()
        results['import_ultralytics'] = 'FAIL'
        return report(results, mem_log)
    results['import_ultralytics'] = 'OK'

    try:
        import paddlex  # noqa: F401  -- same import get_ocr() does lazily
        checkpoint("import paddlex", t0, mem_log)
    except Exception:
        print("IMPORT FAILED: paddlex")
        traceback.print_exc()
        results['import_paddlex'] = 'FAIL'
        return report(results, mem_log)
    results['import_paddlex'] = 'OK'

    try:
        from transformers import (
            TrOCRProcessor, VisionEncoderDecoderModel,
            RobertaTokenizer, ViTImageProcessor,
        )
        checkpoint("import transformers", t0, mem_log)
    except Exception:
        print("IMPORT FAILED: transformers")
        traceback.print_exc()
        results['import_transformers'] = 'FAIL'
        return report(results, mem_log)
    results['import_transformers'] = 'OK'

    from app.inference.ocr import get_ocr, extract_container_code, crop_detection
    checkpoint("import app.inference.ocr", t0, mem_log)

    # ---- 2. Initialize all three in the same process ----
    print("\n=== Stage 2: initialization ===")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        checkpoint("YOLO() loaded", t0, mem_log)
        results['init_yolo'] = 'OK'
    except Exception:
        print("INIT FAILED: YOLO")
        traceback.print_exc()
        results['init_yolo'] = 'FAIL'
        yolo_model = None

    try:
        ocr_pipeline = get_ocr()
        checkpoint("get_ocr() (stock PaddleOCR) loaded", t0, mem_log)
        results['init_paddleocr'] = 'OK'
    except Exception:
        print("INIT FAILED: get_ocr()")
        traceback.print_exc()
        results['init_paddleocr'] = 'FAIL'
        ocr_pipeline = None

    try:
        # NOTE: TrOCRProcessor.from_pretrained(..., use_fast=False) does NOT
        # work as of transformers 5.14.1 -- it still routes tokenizer
        # resolution through AutoTokenizer's fast-path, which fails to parse
        # trocr-base-printed's (old-format) vocab.json with "ValueError: not
        # enough values to unpack (expected 2, got 1)". Constructing the
        # slow RobertaTokenizer + ViTImageProcessor explicitly and wrapping
        # them in TrOCRProcessor bypasses that broken auto-resolution path.
        # This is a transformers/model-repo compatibility issue, unrelated
        # to torch/paddle coexistence -- see smoke test report.
        tokenizer = RobertaTokenizer.from_pretrained(TROCR_MODEL_ID)
        image_processor = ViTImageProcessor.from_pretrained(TROCR_MODEL_ID)
        trocr_processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
        trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
        trocr_model.eval()
        checkpoint("TrOCR (processor+model) loaded", t0, mem_log)
        results['init_trocr'] = 'OK'
    except Exception:
        print("INIT FAILED: TrOCR")
        traceback.print_exc()
        results['init_trocr'] = 'FAIL'
        trocr_processor = None
        trocr_model = None

    # ---- 3. One real inference call through each, in sequence ----
    print("\n=== Stage 3: inference (sequential, same process) ===")

    images = sorted(
        f for f in os.listdir(TEST_IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not images:
        print(f"No sample images found in {TEST_IMAGE_DIR}")
        results['sample_image'] = 'FAIL: none found'
        return report(results, mem_log)

    sample_path = os.path.join(TEST_IMAGE_DIR, images[0])
    with open(sample_path, 'rb') as f:
        image_bytes = f.read()
    print(f"Using sample image: {images[0]}")

    # 3a. YOLO detection -> box
    box = None
    if yolo_model is not None:
        try:
            import numpy as np
            from PIL import Image
            img_np = np.array(Image.open(sample_path).convert('RGB'))
            yolo_results = yolo_model(img_np, verbose=False)
            for r in yolo_results:
                for b in r.boxes:
                    box = tuple(float(v) for v in b.xyxy[0])
                    break
                if box:
                    break
            checkpoint("YOLO inference done", t0, mem_log)
            if box:
                print(f"  YOLO detected box: {box}")
                results['infer_yolo'] = 'OK'
            else:
                print("  YOLO ran but found no box in this image -- trying next images")
                for fname in images[1:20]:
                    p = os.path.join(TEST_IMAGE_DIR, fname)
                    img_np = np.array(Image.open(p).convert('RGB'))
                    yolo_results = yolo_model(img_np, verbose=False)
                    found = False
                    for r in yolo_results:
                        for b in r.boxes:
                            box = tuple(float(v) for v in b.xyxy[0])
                            found = True
                            break
                        if found:
                            break
                    if found:
                        sample_path = p
                        with open(sample_path, 'rb') as f:
                            image_bytes = f.read()
                        print(f"  Found detection in {fname}: {box}")
                        break
                results['infer_yolo'] = 'OK' if box else 'OK (no box found in first 20 images)'
        except Exception:
            print("INFERENCE FAILED: YOLO")
            traceback.print_exc()
            results['infer_yolo'] = 'FAIL'
    else:
        results['infer_yolo'] = 'SKIPPED (init failed)'

    if box is None:
        # Fall back to a large central-ish box so PaddleOCR/TrOCR stages
        # still get exercised even if detection produced nothing.
        from PIL import Image
        w, h = Image.open(sample_path).size
        box = (w * 0.1, h * 0.1, w * 0.9, h * 0.9)
        print(f"  Using fallback full-frame-ish box for OCR stages: {box}")

    # 3b. PaddleOCR inference
    if ocr_pipeline is not None:
        try:
            ocr_result = extract_container_code(image_bytes, *box)
            checkpoint("PaddleOCR inference done", t0, mem_log)
            print(f"  PaddleOCR result: code={ocr_result['container_code']!r} "
                  f"valid={ocr_result['is_valid']} raw={ocr_result.get('ocr_raw_output', '')!r}")
            results['infer_paddleocr'] = 'OK'
        except Exception:
            print("INFERENCE FAILED: PaddleOCR")
            traceback.print_exc()
            results['infer_paddleocr'] = 'FAIL'
    else:
        results['infer_paddleocr'] = 'SKIPPED (init failed)'

    # 3c. TrOCR inference
    if trocr_model is not None:
        try:
            import torch
            crop = crop_detection(image_bytes, *box)
            pixel_values = trocr_processor(images=crop, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = trocr_model.generate(pixel_values)
            trocr_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            checkpoint("TrOCR inference done", t0, mem_log)
            print(f"  TrOCR result: text={trocr_text!r}")
            results['infer_trocr'] = 'OK'
        except Exception:
            print("INFERENCE FAILED: TrOCR")
            traceback.print_exc()
            results['infer_trocr'] = 'FAIL'
    else:
        results['infer_trocr'] = 'SKIPPED (init failed)'

    checkpoint("all stages complete", t0, mem_log)
    return report(results, mem_log)


def report(results, mem_log):
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:20s} {v}")
    print()
    print("Memory checkpoints (RSS, MB):")
    for label, mem in mem_log:
        print(f"  {mem:8.1f}  {label}")
    peak = max((m for _, m in mem_log), default=-1)
    print(f"\nPeak RSS observed: {peak:.1f} MB")
    return results


if __name__ == '__main__':
    main()
