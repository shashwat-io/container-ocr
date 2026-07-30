"""
Gap-detection heuristic candidates for right_column_bleed tall crops.

Detection-only, no cropping. Each crop is a tall/narrow container-code
region (vertically-stacked characters) with an adjacent ISO-type/
dimension-plate column bleeding in from the right. The goal is to find the
vertical whitespace/uniform-paint strip that separates the two columns, so
a later step can crop past it.

Three candidate per-column signals, all fed through the same
find_widest_gap() run-finder so they're compared apples-to-apples:

1. column_variance   -- np.var of grayscale intensity down each column.
   Cheap, no tuning. A uniform (background/paint) column has low variance;
   a column crossing character strokes has high variance. Weakness: a
   column that's uniformly a DIFFERENT color (e.g. a paint discoloration
   stripe, a shadow band) also reads as "low variance" even though it
   isn't the real gap.

2. column_bg_deviation -- mean absolute deviation from the crop's own
   median grayscale value (used as a background-paint estimate, since
   background pixels typically outnumber character-stroke pixels even
   with bleed content included). More specific than variance: a
   differently-colored-but-uniform stripe won't fool this one as easily,
   since it directly checks "does this column actually look like
   background", not just "is this column internally uniform". Weakness:
   the median-based background estimate can drift if bleed content
   occupies a large fraction of the crop, or under a strong lighting
   gradient across the crop.

3. column_edge_density -- mean Sobel gradient magnitude per column
   (Gaussian-blurred first to suppress JPEG-noise edges). Most robust to
   smooth lighting gradients (a slow brightness ramp across a true gap
   produces near-zero local gradient even though it would raise
   variance/deviation), since it measures local contrast, not absolute
   level. Weakness: more compute, and can be fooled by genuine texture
   (rust, scratches, dirt) inside what's actually a gap.

All three operate on the full crop height (rows), producing one scalar per
column (width), so "gap" here means a vertical strip, matching the
physical layout (code column left, bleed column right).
"""
import numpy as np
import cv2
from PIL import Image
from scipy.ndimage import binary_closing


def load_grayscale(path):
    with Image.open(path) as im:
        gray = np.array(im.convert("L"), dtype=np.float64)
    return gray  # shape (H, W)


def column_variance(gray):
    return gray.var(axis=0)


def column_bg_deviation(gray):
    bg = np.median(gray)
    return np.mean(np.abs(gray - bg), axis=0)


def column_edge_density(gray):
    gray_u8 = gray.astype(np.uint8)
    blurred = cv2.GaussianBlur(gray_u8, (3, 3), 0)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag.mean(axis=0)


METHODS = {
    "column_variance": column_variance,
    "column_bg_deviation": column_bg_deviation,
    "column_edge_density": column_edge_density,
}


def otsu_threshold(values, bins=256):
    """1D Otsu threshold (inter-class variance maximization) over an
    arbitrary-scale array -- used instead of a fixed percentile because a
    fixed percentile assumes a roughly-known content/background ratio, and
    that ratio varies a lot crop to crop (confirmed on real data: a 30th-
    percentile split badly misclassified both the left margin AND the real
    gap as "content" on a 225px-wide sample where content only covered
    ~40% of the width). Otsu finds the split the data itself supports."""
    hist, bin_edges = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    sum_total = (hist * bin_centers).sum()
    sum_b, w_b = 0.0, 0.0
    max_var, threshold = -1.0, bin_centers[0]
    for i in range(len(hist)):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += hist[i] * bin_centers[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var, threshold = var_between, bin_centers[i]
    return threshold


def find_widest_gap(signal, margin_frac=0.02,
                     min_run_frac=0.02, contrast_ratio=0.6, close_frac=0.05):
    """Given a per-column activity signal, find the real code/bleed gap:
    the first sufficiently-wide low-activity run that comes AFTER the
    first sufficiently-wide high-activity run (the code column itself).

    This is NOT simply "the widest low-activity run in the crop" -- an
    earlier version was, and turned out to be wrong on real crops: the
    left edge-padding before the code text even starts is itself a
    low-activity run, and on several sampled crops it was wider/more
    uniform than the actual code-to-bleed gap, so "widest run anywhere"
    pointed at the left margin instead of the real boundary (confirmed by
    eye on 2 of 6 visually-checked samples before this fix). Scanning
    left-to-right for "first real content block, then first real gap
    after it" structurally skips any leading margin, since a leading low
    run -- however wide -- is never mistaken for the content block itself.

    Before run-finding, a morphological closing (close_frac) merges any
    low-activity dip narrower than that fraction of the width back into
    "content". This turned out to matter in practice too: characters
    stacked one-per-row taper in width (a narrow "1" vs. a wide "U"), so
    the code block's own column signal can dip briefly near its left/right
    edges where only the wider characters reach -- without closing, that
    dip got misread as the real gap on at least one visually-checked
    sample, cutting the line straight through the code's edge.

    Returns a dict with:
      gap_x        -- midpoint column index of the detected gap run
                       (None if no run found at all)
      run_start, run_end -- column bounds of that run (end exclusive)
      run_width    -- width in columns of that run
      ambiguous    -- True if no clean content-block-then-gap pattern was
                       found (e.g. code and bleed content run together
                       with no sufficiently wide low-activity run between
                       them) -- i.e. no confidently-detectable gap, not
                       just "found a gap at position X".
    """
    W = len(signal)
    margin = max(1, int(round(W * margin_frac)))
    lo, hi = margin, W - margin
    if hi <= lo:
        return {"gap_x": None, "run_start": None, "run_end": None,
                "run_width": 0, "ambiguous": True}

    zone = signal[lo:hi]
    thresh = otsu_threshold(zone)
    is_high = zone > thresh  # True = content-like, False = background-like

    close_px = max(1, int(round(len(zone) * close_frac)))
    is_high = binary_closing(is_high, structure=np.ones(close_px, dtype=bool))

    runs = []  # list of (is_high, start, end) end exclusive
    start = 0
    cur = bool(is_high[0])
    for i in range(1, len(is_high)):
        if bool(is_high[i]) != cur:
            runs.append((cur, start, i))
            start = i
            cur = bool(is_high[i])
    runs.append((cur, start, len(is_high)))

    min_run_px = max(2, int(round(W * min_run_frac)))

    first_content_idx = None
    for idx, (is_h, s, e) in enumerate(runs):
        if is_h and (e - s) >= min_run_px:
            first_content_idx = idx
            break

    if first_content_idx is None:
        return {"gap_x": None, "run_start": None, "run_end": None,
                "run_width": 0, "ambiguous": True}

    gap_run = None
    for idx in range(first_content_idx + 1, len(runs)):
        is_h, s, e = runs[idx]
        if (e - s) < min_run_px:
            continue  # too narrow to be either a real gap or a real content block -- skip past noise
        if not is_h:
            gap_run = (s, e)
        break  # first sufficiently-wide run after the content block, whichever kind it is

    if gap_run is None:
        return {"gap_x": None, "run_start": None, "run_end": None,
                "run_width": 0, "ambiguous": True}

    run_start, run_end = gap_run
    run_width = run_end - run_start

    run_vals = zone[run_start:run_end]
    rest_vals = np.concatenate([zone[:run_start], zone[run_end:]])
    run_mean = run_vals.mean()
    rest_mean = rest_vals.mean() if len(rest_vals) else run_mean
    contrast_ok = (rest_mean <= 1e-9) or (run_mean <= contrast_ratio * rest_mean)

    ambiguous = not contrast_ok

    gap_x = lo + (run_start + run_end) / 2.0
    return {
        "gap_x": gap_x,
        "run_start": lo + run_start,
        "run_end": lo + run_end,
        "run_width": run_width,
        "ambiguous": bool(ambiguous),
    }


def detect_all_methods(image_path):
    """Returns {method_name: find_widest_gap(...) result} for one crop."""
    gray = load_grayscale(image_path)
    results = {}
    for name, fn in METHODS.items():
        signal = fn(gray)
        results[name] = find_widest_gap(signal)
    return results
