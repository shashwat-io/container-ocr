"""
Single source of crop truth for the fine-tuning pipeline.

manual_labels.csv stores YOLO-normalized boxes (x_center, y_center, width,
height, all 0-1 relative to full image), while app/inference/ocr.py's
crop_detection() takes absolute pixel x1,y1,x2,y2 and a padding value. This
module does the normalized->pixel conversion and then calls crop_detection()
directly, so a fine-tuning crop is produced by the exact same padding/crop
logic that feeds the deployed OCR pipeline at inference time.

Known open assumption (flagged, not silently resolved): Labelling.ipynb --
the notebook a human used to actually view crops and type label_text -- only
exists in Kaggle/Colab sessions, not in this repo, so its crop logic could
not be read or diffed against this one. Reusing crop_detection() was chosen
as the more defensible default over guessing at the notebook's own preview
logic, since it keeps fine-tuning crops consistent with what the deployed
model actually sees at inference -- but if Labelling.ipynb used a different
padding value, a fine-tuned crop could show slightly more/less border than
what the labeler actually verified against. Confirm against the real
notebook before trusting this at scale.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.inference.ocr import crop_detection

CROP_PADDING = 10  # matches app/inference/ocr.py's crop_detection default


def yolo_box_to_pixels(x_center, y_center, width, height, img_w, img_h):
    """Convert normalized YOLO-format box (0-1 range) to absolute pixel x1,y1,x2,y2."""
    x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
    x1 = (x_center - width / 2) * img_w
    y1 = (y_center - height / 2) * img_h
    x2 = (x_center + width / 2) * img_w
    y2 = (y_center + height / 2) * img_h
    return x1, y1, x2, y2


def extract_crop(image_path, x_center, y_center, width, height, padding=CROP_PADDING):
    """Read image_path from disk, convert its YOLO box to pixels, and return the PIL crop."""
    from PIL import Image

    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    img_w, img_h = Image.open(image_path).size
    x1, y1, x2, y2 = yolo_box_to_pixels(x_center, y_center, width, height, img_w, img_h)
    return crop_detection(image_bytes, x1, y1, x2, y2, padding=padding)
