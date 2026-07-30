"""
Single source of truth for the video_id-based test-split safety definition.

Both extract_crops.py (leakage exclusion at crop-extraction time) and
notebooks/Labelling.ipynb (safe-video steering at labeling time) import
from here, so "safe" means exactly the same thing in both places -- no
drift between what gets labeled and what later survives extract_crops.py's
filters. See extract_crops.py's module docstring for the full rationale
(EXCLUDE_WHOLE_SOURCE_VIDEO): a row is unsafe if its source video appears
*anywhere* in the test split, not just as an exact-duplicate frame.
"""
import os
import re
from pathlib import Path


def pre_hash_stem(filename: str) -> str:
    """Strip Roboflow's '.rf.<hash>' suffix to get the original filename stem."""
    return re.sub(r"\.rf\.[0-9a-f]+$", "", Path(filename).stem)


def group_id(filename: str) -> str:
    """
    Corrected grouping key for leakage-safety purposes.

    - If the first '_'-delimited token is numeric: it's a genuine video-frame
      ID, group by that (matches original source_video_id() behavior).
    - Otherwise: the first token is a generic camera/export prefix (IMG,
      Event2023..., DSCF, WhatsApp, etc.) that does NOT reliably identify a
      shared source — fall back to the pre-hash stem, so each distinct
      source photo becomes its own group, while genuine duplicate copies
      (same stem, different Roboflow hash) still group together correctly.
    """
    stem = Path(filename).stem
    first_token = stem.split("_")[0]
    if first_token.isdigit():
        return first_token
    return pre_hash_stem(filename)


def source_video_id(basename):
    """'1114039705751937665_0370_jpg.rf.<hash>.jpg' -> '1114039705751937665'."""
    return basename.split('_')[0]


def build_test_video_ids(test_images_dir):
    """Distinct group_ids present anywhere in the test split."""
    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(f"test images dir not found: {test_images_dir}")
    return {group_id(f) for f in os.listdir(test_images_dir)}


def is_video_safe(image_path, test_video_ids):
    """True if image_path's group_id never appears in the test split
    (neither an exact-duplicate frame nor a same-video-different-offset
    frame, nor a same-source-photo-different-hash duplicate for the
    non-numeric-prefix buckets), i.e. it's safe to label/use for
    fine-tuning under the EXCLUDE_WHOLE_SOURCE_VIDEO policy."""
    return group_id(os.path.basename(image_path)) not in test_video_ids
