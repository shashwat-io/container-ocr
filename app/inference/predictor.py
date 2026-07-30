import logging
import os
import time
from PIL import Image
import io
from app.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Override the YOLO detector checkpoint path without a code edit (e.g. to
# swap in a newer confirmed-good model). Leave unset to use the current
# default, models/best_resplit.pt.
YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', '')


class ContainerPredictor:
    def __init__(self):
        self.config = get_config()
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        if self.config.USE_STUB_MODEL:
            self._initialize_stub_model()
        else:
            self._initialize_real_model()

    def _initialize_real_model(self):
        try:
            logger.info("Loading real ML model...")
            from ultralytics import YOLO
            default_path = os.path.join(os.path.dirname(__file__), '../../models/best_resplit.pt')
            model_path = os.path.abspath(YOLO_MODEL_PATH or default_path)
            logger.info(f"Loading model from: {model_path}")
            self.model = YOLO(model_path)
            logger.info("Real ML model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load real model: {e}")
            raise RuntimeError(f"Failed to load real ML model: {e}")

    def _initialize_stub_model(self):
        logger.info("Initializing stub model for development/testing")

    def predict(self, image_bytes, filename=None):
        if self.config.USE_STUB_MODEL:
            return self._stub_predict(image_bytes, filename)
        return self._real_predict(image_bytes, filename)

    def _real_predict(self, image_bytes, filename=None):
        try:
            import numpy as np
            from app.inference.ocr import extract_container_code, get_recognizer

            logger.info(f"Processing image with real model: {filename}")
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)

            start = time.perf_counter()

            results = self.model(image_np)

            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])

                    det = {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(confidence),
                        "class": class_id,
                        "label": self._get_label(class_id),
                        # OCR fields default to empty; overwritten below
                        "container_code": "",
                        "is_valid": False,
                        "ocr_raw_output": "",
                        "ocr_confidence": 0.0,
                        "validation_reason": "",
                        "is_recovery_based": False,
                        "is_ambiguous_merge": False,
                    }

                    # Run OCR on this detection crop
                    try:
                        ocr_result = extract_container_code(
                            image_bytes,
                            float(x1), float(y1), float(x2), float(y2)
                        )
                        det["container_code"]     = ocr_result["container_code"]
                        det["is_valid"]           = ocr_result["is_valid"]
                        det["ocr_raw_output"]     = ocr_result["ocr_raw_output"]
                        det["ocr_confidence"]     = ocr_result["ocr_confidence"]
                        det["validation_reason"]  = ocr_result["validation_reason"]
                        det["is_recovery_based"]  = ocr_result["is_recovery_based"]
                        det["is_ambiguous_merge"] = ocr_result["is_ambiguous_merge"]
                    except Exception as ocr_err:
                        # Log the traceback, and surface the failure in the
                        # detection itself. Previously validation_reason was
                        # left as "" here, so a crashed OCR call was
                        # indistinguishable in the UI from a box that was
                        # simply never read -- the operator saw a blank status
                        # with no indication anything had gone wrong.
                        logger.error(f"OCR failed for detection: {ocr_err}", exc_info=True)
                        det["validation_reason"] = f"OCR error: {ocr_err}"

                    detections.append(det)

            elapsed = time.perf_counter() - start

            recognizer = get_recognizer()
            model_type = f"YOLOv8 + {'TrOCR' if recognizer == 'trocr' else 'PaddleOCR'}"

            logger.info(f"Real model detected {len(detections)} objects")
            return {
                "status": "success",
                "model_type": model_type,
                "is_stub": False,
                "detections": detections,
                "processing_time": elapsed
            }
        except Exception as e:
            logger.error(f"Real model prediction failed: {e}")
            try:
                from app.inference.ocr import get_recognizer
                recognizer = get_recognizer()
                model_type = f"YOLOv8 + {'TrOCR' if recognizer == 'trocr' else 'PaddleOCR'}"
            except Exception:
                model_type = "YOLOv8 + OCR"
            return {
                "status": "error",
                "model_type": model_type,
                "is_stub": False,
                "error": str(e),
                "detections": []
            }

    def _stub_predict(self, image_bytes, filename=None):
        logger.info(f"Processing image with stub model: {filename}")

        start = time.perf_counter()

        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        center_x, center_y = width / 2, height / 2
        box_size = min(width, height) * 0.6

        x1 = center_x - box_size / 2
        y1 = center_y - box_size / 2
        x2 = center_x + box_size / 2
        y2 = center_y + box_size / 2

        mock_detections = [{
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": 0.95,
            "class": 0,
            "label": "container",
            "container_code": "",
            "is_valid": False,
            "ocr_raw_output": "",
            "ocr_confidence": 0.0,
            "validation_reason": "Stub mode — OCR not run",
            "is_recovery_based": False,
            "is_ambiguous_merge": False,
        }]

        elapsed = time.perf_counter() - start

        return {
            "status": "success",
            "model_type": "Demo Mode (Mock Data)",
            "is_stub": True,
            "detections": mock_detections,
            "processing_time": elapsed
        }

    def _get_label(self, class_id):
        labels = {0: "container_number"}
        return labels.get(class_id, f"class_{class_id}")
