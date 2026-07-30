"""
Isolated TrOCR (microsoft/trocr-base-printed) inference module.

Independent of app/inference/ocr.py (PaddleOCR) -- no imports from it, no
shared checkpoint path, no shared env var. Not wired into predictor.py.
This exists purely as the zero-shot TrOCR baseline path; see
scripts/smoke_test_trocr_coexistence.py for the coexistence feasibility
check this was built on top of.
"""
import io
import logging
import os
import threading

import torch
from PIL import Image
from transformers import (
    RobertaTokenizer,
    ViTImageProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

# Reused, not reimplemented, per the rotation-retry design: clean_ocr_text
# normalizes raw OCR text the same way ocr.py's own tier logic does, and
# validate_iso6346 is the single source of truth for checksum validity.
# Import-only -- ocr.py itself is never modified.
from app.inference.ocr import clean_ocr_text, validate_iso6346

logger = logging.getLogger(__name__)

TROCR_MODEL_ID = 'microsoft/trocr-base-printed'

# Override the recognition model's weights with a fine-tuned checkpoint dir
# (e.g. trocr_finetune_output/final, checkpoint-1672). Mirrors ocr.py's
# FINETUNED_REC_MODEL_DIR pattern: unset/empty is always safe and reproduces
# the exact zero-shot behavior every prior TrOCR script here has used.
# Tokenizer/image-processor are still built from TROCR_MODEL_ID regardless
# (fine-tuning this checkpoint never touched the vocab), only the model
# weights are swapped -- avoids re-triggering the fast-tokenizer bug
# documented in get_trocr() below if the checkpoint dir carries its own
# copied-over (identically broken) tokenizer files.
#
# READ LAZILY, NOT AT IMPORT TIME. This used to be a module-level constant:
#
#     FINETUNED_TROCR_MODEL_DIR = os.environ.get('FINETUNED_TROCR_MODEL_DIR', '')
#
# which silently coupled correctness to import ORDER. The benchmark scripts get
# away with it by setting os.environ before their first import (see
# benchmark_trocr_cascade_quicksample.py, which sets it above even `import
# csv`), but a Flask app cannot guarantee that ordering. If this module were
# imported before the env var was set, model_source fell back to TROCR_MODEL_ID
# and production served the ZERO-SHOT BASE MODEL -- with no exception, no
# warning, and output plausible enough to pass casual inspection. Resolving the
# value inside get_trocr() removes the ordering dependency entirely.
def _resolve_checkpoint_dir():
    """Checkpoint dir from the environment, resolved at call time."""
    return os.environ.get('FINETUNED_TROCR_MODEL_DIR', '').strip()

_trocr_instance = None  # (processor, model, device) tuple
# Mirrors app/inference/ocr.py's _ocr_lock pattern: serialise access to the
# lazily-initialised singleton so concurrent callers don't double-init or
# race on the underlying torch model's forward-pass buffers.
#
# This lock now guards BOTH lazy init (get_trocr) and inference
# (run_trocr_on_crop's model.generate call). Previously it covered only init,
# leaving generate() to run concurrently across gunicorn's worker threads on a
# single shared VisionEncoderDecoderModel.
#
# Serialising inference is a deliberate choice, not caution about correctness:
# the deployment runs gunicorn with --threads 4, and a single TrOCR generate
# already saturates the CPU on this box (~650% of 8 cores was measured for one
# cascade process). Four concurrent generates cannot go faster than four serial
# ones on a saturated CPU -- they would only multiply peak resident memory
# across a ~1.3 GB model's activations and risk memory pressure for no
# throughput gain. Queueing is the better failure mode than thrashing.
#
# ⚠ HAZARD: threading.Lock is NOT reentrant. run_trocr_on_crop must call
# get_trocr() BEFORE acquiring this lock, never inside it -- nesting the two
# acquisitions self-deadlocks the worker thread on first use. If you ever need
# them nested, switch this to threading.RLock() in the same commit.
_trocr_lock = threading.Lock()


def get_trocr():
    global _trocr_instance
    if _trocr_instance is None:
        with _trocr_lock:
            if _trocr_instance is None:
                # NOTE: TrOCRProcessor.from_pretrained(...) (with or without
                # use_fast=False) fails on transformers 5.14.1 -- the fast
                # RoBERTa tokenizer path can't parse trocr-base-printed's
                # (old-format) vocab.json ("ValueError: not enough values to
                # unpack (expected 2, got 1)"), and use_fast=False does not
                # avoid this because TrOCRProcessor still routes through the
                # same broken AutoTokenizer resolution internally. Building
                # the tokenizer and image processor explicitly and handing
                # them to TrOCRProcessor directly bypasses that path
                # entirely. Confirmed working in
                # scripts/smoke_test_trocr_coexistence.py -- don't
                # rediscover this, keep the explicit construction.
                tokenizer = RobertaTokenizer.from_pretrained(TROCR_MODEL_ID)
                image_processor = ViTImageProcessor.from_pretrained(TROCR_MODEL_ID)
                processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

                checkpoint_dir = _resolve_checkpoint_dir()

                # Fail fast rather than silently degrading. A configured-but-
                # missing checkpoint dir previously fell through to the hub ID
                # and served the zero-shot base model; the benchmark harness
                # guards against exactly this with its own sys.exit (see
                # benchmark_trocr_cascade_quicksample.py), and production had
                # no equivalent. An explicit crash at startup is far cheaper
                # than discovering weeks of degraded reads after the fact.
                if checkpoint_dir and not os.path.isdir(checkpoint_dir):
                    raise FileNotFoundError(
                        f"FINETUNED_TROCR_MODEL_DIR is set to {checkpoint_dir!r} "
                        "but that directory does not exist. Refusing to fall "
                        f"back to the zero-shot base model ({TROCR_MODEL_ID}) "
                        "silently -- fix the path or unset the variable to "
                        "request the base model deliberately."
                    )

                model_source = checkpoint_dir or TROCR_MODEL_ID
                model = VisionEncoderDecoderModel.from_pretrained(model_source)
                model.eval()

                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model.to(device)

                # Always name the resolved weights source and say plainly which
                # of the two modes we are in. The silent-base-model failure this
                # guards against is invisible in output -- the model still
                # returns confident-looking text -- so it needs a log line that
                # states the mode outright rather than one that has to be
                # inferred from the presence or absence of a path.
                if checkpoint_dir:
                    logger.info(
                        "get_trocr: recognizer=FINE-TUNED checkpoint | "
                        "weights=%s | tokenizer/image_processor=%s (hub) | device=%s",
                        checkpoint_dir, TROCR_MODEL_ID, device,
                    )
                else:
                    logger.warning(
                        "get_trocr: recognizer=HUB BASE MODEL (ZERO-SHOT, NOT "
                        "fine-tuned) | weights=%s | device=%s -- "
                        "FINETUNED_TROCR_MODEL_DIR is unset. If this is a "
                        "production deployment, this is almost certainly wrong: "
                        "accuracy will be far below every benchmarked figure.",
                        TROCR_MODEL_ID, device,
                    )
                _trocr_instance = (processor, model, device)
    return _trocr_instance


def crop_detection_trocr(image_bytes, x1, y1, x2, y2, padding=10):
    """
    Independent re-implementation of ocr.py's crop_detection() -- not
    imported from ocr.py, to keep this module fully decoupled.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    width, height = image.size
    left = max(0, x1 - padding)
    top = max(0, y1 - padding)
    right = min(width, x2 + padding)
    bottom = min(height, y2 + padding)
    return image.crop((left, top, right, bottom))


def run_trocr_on_crop(crop_image):
    """
    Runs TrOCR generation on a single PIL crop image. Returns the raw
    decoded text (str), stripped. No ISO 6346 validation, no tiered
    recovery logic -- this is a zero-shot text-recognition baseline only.
    """
    # get_trocr() acquires _trocr_lock for lazy init, so it MUST be called
    # before we take the lock ourselves -- threading.Lock is not reentrant and
    # nesting these would self-deadlock on the very first request. See the
    # hazard note at the _trocr_lock definition.
    processor, model, device = get_trocr()

    # Serialise inference across gunicorn's worker threads: one generate
    # already saturates this CPU, so concurrency buys no throughput and only
    # multiplies peak memory. Held across preprocessing, generate and decode so
    # a single request's tensors are not interleaved with another's.
    with _trocr_lock:
        pixel_values = processor(images=crop_image.convert('RGB'), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()


def extract_container_code_trocr(image_bytes, x1, y1, x2, y2):
    """
    Full-image-bytes + bbox entry point, shaped like ocr.py's
    extract_container_code(image_bytes, x1, y1, x2, y2) for interface
    parity (e.g. if predictor.py were ever wired to call this instead).
    Independent implementation -- does not call into ocr.py.

    Returns dict with: container_code, is_valid, ocr_raw_output,
    ocr_confidence, validation_reason. is_valid/container_code are left at
    their default (unvalidated) since this module does no ISO 6346 checking
    -- that's out of scope for a zero-shot raw-text baseline.
    """
    result = {
        'container_code': '',
        'is_valid': False,
        'ocr_raw_output': '',
        'ocr_confidence': 0.0,
        'validation_reason': 'Not validated (zero-shot TrOCR raw text only)',
    }
    try:
        crop = crop_detection_trocr(image_bytes, x1, y1, x2, y2)
        text = run_trocr_on_crop(crop)
        result['ocr_raw_output'] = text
    except Exception as e:
        logger.error(f"extract_container_code_trocr failed: {e}")
        result['validation_reason'] = f'TrOCR error: {e}'
    return result


def _is_degenerate_text(text):
    """
    A prediction is 'degenerate' if it collapses to <=1 alphanumeric
    character once cleaned -- matches the failure mode seen in the
    unrotated zero-shot run (160/300 outputs were a single punctuation
    mark or lone letter/digit, e.g. ':', '=', '1').
    """
    return len(clean_ocr_text(text)) <= 1


def _find_checksum_valid_candidate(text):
    """
    Slide an 11-char window over the cleaned text and return the first
    substring that passes ISO 6346 checksum validation (validate_iso6346,
    imported from ocr.py -- not reimplemented here), or None if none do.

    This windowing is our own, independent scan (not copied from ocr.py's
    Tier logic) -- it exists because TrOCR's raw output is essentially
    never a clean, unspaced 11-char string on its own (it comes out like
    'CAILU 849453 41'), so a direct length-11 equality check would almost
    never fire.
    """
    cleaned = clean_ocr_text(text)
    for start in range(max(0, len(cleaned) - 10)):
        candidate = cleaned[start:start + 11]
        if len(candidate) < 11:
            break
        if not (candidate[:4].isalpha() and candidate[4:11].isdigit()):
            continue
        valid, _ = validate_iso6346(candidate)
        if valid:
            return candidate
    return None


def extract_container_code_trocr_rotated(crop_image):
    """
    Rotation-retry variant of run_trocr_on_crop: tries the crop at 0deg,
    +90deg, and -90deg (same rotate() angle convention as ocr.py's
    _best_portrait_rotation -- rotate(90, expand=True) is labeled '+90',
    rotate(-90, expand=True) is labeled '-90'), and picks the best of the
    three. TrOCR has no per-result confidence score the way PaddleOCR does,
    so "best" is defined as:

      1. Any rotation whose output contains an ISO 6346 checksum-valid
         11-char substring wins outright. Ties broken by rotation order
         (0, then +90, then -90) -- mirrors _best_portrait_rotation's
         "a clean primary result is never displaced by a retry" bias
         toward the earlier/primary attempt.
      2. Otherwise, the longest non-degenerate output wins (some signal
         beats no signal) -- ties broken the same way.
      3. If all three are degenerate, falls back to the 0deg result.

    Returns dict: {'trocr_raw_best': str, 'rotation_used': '0'|'+90'|'-90',
                   'raw_by_rotation': {'0': str, '+90': str, '-90': str}}
    """
    rotations = [
        ('0', crop_image),
        ('+90', crop_image.rotate(90, expand=True)),
        ('-90', crop_image.rotate(-90, expand=True)),
    ]

    raw_by_rotation = {}
    for label, img in rotations:
        raw_by_rotation[label] = run_trocr_on_crop(img)

    for label, _ in rotations:
        raw_text = raw_by_rotation[label]
        if _find_checksum_valid_candidate(raw_text):
            return {
                'trocr_raw_best': raw_text,
                'rotation_used': label,
                'raw_by_rotation': raw_by_rotation,
            }

    non_degenerate = [
        (label, raw_by_rotation[label]) for label, _ in rotations
        if not _is_degenerate_text(raw_by_rotation[label])
    ]
    if non_degenerate:
        best_label, best_text = max(non_degenerate, key=lambda item: len(clean_ocr_text(item[1])))
        return {
            'trocr_raw_best': best_text,
            'rotation_used': best_label,
            'raw_by_rotation': raw_by_rotation,
        }

    return {
        'trocr_raw_best': raw_by_rotation['0'],
        'rotation_used': '0',
        'raw_by_rotation': raw_by_rotation,
    }
