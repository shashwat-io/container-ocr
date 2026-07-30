"""
READ-ONLY data-analysis script. Does NOT modify ocr.py, review sheets, or any
other existing file -- only reads CSVs and prints statistics.

Goal: for each row in tier1_exact, tier1b, and tier2_{exact,recovery} review
sheets, identify which individual OCR fragment(s) actually contributed to the
row's claimed_code, and record those fragments' own confidence scores (not
the crop-level average). Then split into "contributed to a TRUE-graded
(correct) result" vs "contributed to a FALSE-graded (wrong) result" and
report the two distributions.

Attribution method: run the REAL _try_extract() (dry-run, monkeypatched
run_ocr_on_crop, no PaddleX -- same pattern as scratch_tier3_ambiguous_exclude_verify.py)
against each row's own captured OCR fragments, capture its logger.info/warning
output, and use the logged pair/candidate strings to identify exactly which
list-index fragment(s) were combined -- rather than re-implementing the tier
logic by hand and risking divergence from the actual code.
"""
import os
import re
import sys
import csv
import logging
import statistics as stats
from unittest import mock

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

import app.inference.ocr as ocr_mod

_FRAG_RE = re.compile(r'^(.*)\(([\d.]+)\)$')


def classify_verified(verified):
    """
    Classify a review sheet's verified_correct free-text field into
    'TRUE', 'FALSE', or 'exclude' (ambiguous/ungradeable -- box mismatch,
    "Ungradable", or anything not recognizable as either grade).

    Observed formats across the four sheets (checked directly against the
    data, not assumed): plain 'TRUE' / 'FALSE'; 'FALSE(correct code - X...'
    (still a FALSE grade, with a note); 'box = a, TRUE' / 'box =a, TRUE' /
    'box = b, TRUE' (a box-ambiguous detection where the reviewer confirmed
    the SPECIFIC box that produced this row's own claimed_code was correct
    -- still a TRUE grade for this row, "box=a" is just provenance, not a
    caveat); 'Ungradable'; 'box mismatch — neither matches'.

    A naive `.startswith('TRUE')` misses every 'box = a, TRUE' style value
    (starts with 'BOX') and would silently misclassify 10 actually-correct
    rows across the sheets as wrong -- checked via Counter over all four
    sheets before writing this.
    """
    v = verified.strip()
    vl = v.lower()
    if vl.startswith('ungrad') or 'mismatch' in vl:
        return 'exclude'
    vu = v.upper()
    if 'FALSE' in vu:
        return 'FALSE'
    if 'TRUE' in vu:
        return 'TRUE'
    return 'exclude'


def parse_ocr_raw(raw):
    texts = []
    for chunk in raw.split(' | '):
        m = _FRAG_RE.match(chunk.strip())
        if not m:
            raise ValueError(f"Could not parse fragment: {chunk!r} in {raw!r}")
        text, score = m.group(1), float(m.group(2))
        texts.append((text, score))
    return texts


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def run_row_dry(texts):
    cap = _Capture()
    ocr_logger = logging.getLogger('app.inference.ocr')
    prev_level = ocr_logger.level
    ocr_logger.setLevel(logging.INFO)
    ocr_logger.addHandler(cap)
    try:
        with mock.patch.object(ocr_mod, 'run_ocr_on_crop', return_value=texts):
            result = ocr_mod._try_extract(None)
    finally:
        ocr_logger.removeHandler(cap)
        ocr_logger.setLevel(prev_level)
    return result, cap.records


def attribute_fragments(texts, log_lines):
    """
    Return list of (fragment_text, fragment_confidence) that contributed to
    whichever candidate the real code just returned, based on which tier/pair
    fired (per the captured log lines) -- or None if attribution couldn't be
    determined from the log + fragment list alone.
    """
    clean = ocr_mod.clean_ocr_text

    for msg in log_lines:
        # Tier 1: single-line exact match. Logs the final code only, which
        # *is* the cleaned fragment text itself for this tier.
        m = re.match(r"Tier 1 \(single-line exact\): code='(.+)'$", msg)
        if m:
            code = m.group(1)
            for t, s in texts:
                if clean(t) == code:
                    return [(t, s)]
            return None

        # Tier 1b: logs 'line' (10-char source) -> 'candidate' (11-char w/ check digit).
        m = re.match(r"Tier 1b \(single-line check-digit recovery\): '(.+)' → '(.+)'$", msg)
        if m:
            line = m.group(1)
            for t, s in texts:
                if clean(t) == line:
                    return [(t, s)]
            return None

        # Tier 2 removal-search (logs the FULL pair concatenation, verbatim).
        m = re.match(r"Tier 2 \(adjacent-pair, one-char-over-length removal\): '(.+)' → '(.+)'$", msg)
        if m:
            pair_str = m.group(1)
            for i in range(len(texts) - 1):
                if clean(texts[i][0] + texts[i + 1][0]) == pair_str:
                    return [texts[i], texts[i + 1]]
            return None

        # Tier 2 check-digit recovery (logs the exact 10-char pair).
        m = re.match(r"Tier 2 \(adjacent-pair check-digit\): '(.+)' → '(.+)'$", msg)
        if m:
            pair_str = m.group(1)
            for i in range(len(texts) - 1):
                if clean(texts[i][0] + texts[i + 1][0]) == pair_str:
                    return [texts[i], texts[i + 1]]
            return None

        # Tier 2 plain 11-char contiguous window (only logs the final candidate,
        # not the pair) -- find which adjacent pair's concatenation contains it.
        m = re.match(r"Tier 2 \(adjacent-pair 11-char\): code='(.+)'$", msg)
        if m:
            candidate = m.group(1)
            for i in range(len(texts) - 1):
                if candidate in clean(texts[i][0] + texts[i + 1][0]):
                    return [texts[i], texts[i + 1]]
            return None

        # Tier 3 -- spans the full multi-fragment concatenation, not a clean
        # pair. Not expected in the tier2_* sheets (those are pre-classified
        # as Tier2 hits) but flagged if encountered.
        if msg.startswith("Tier 3"):
            return 'TIER3_UNSUPPORTED'

    return None


def load_master_ocr_raw():
    rows = list(csv.DictReader(open('benchmark_results_full_live_corrected_tiers.csv')))
    by_name = {}
    for r in rows:
        by_name.setdefault(r['filename'], []).append(r)
    return by_name


def process_tier1_exact():
    path = 'tier1_exact_review_samples/review_sheet.csv'
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        grade = classify_verified(r['verified_correct'])
        if grade == 'exclude':
            out.append((r, None, 'excluded-ungradeable'))
            continue
        texts = parse_ocr_raw(r['ocr_raw'])
        result, log_lines = run_row_dry(texts)
        code, valid, reason, raw_out, conf = result
        frags = attribute_fragments(texts, log_lines)
        out.append((r, frags, grade))
    return out


def process_tier2_sheet(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        grade = classify_verified(r['verified_correct'])
        if grade == 'exclude':
            out.append((r, None, 'excluded-ungradeable'))
            continue
        texts = parse_ocr_raw(r['ocr_raw'])
        result, log_lines = run_row_dry(texts)
        code, valid, reason, raw_out, conf = result
        frags = attribute_fragments(texts, log_lines)
        out.append((r, frags, grade))
    return out


def process_tier1b():
    path = 'tier1b_review_samples/review_sheet.csv'
    rows = list(csv.DictReader(open(path)))
    master = load_master_ocr_raw()
    out = []
    for r in rows:
        grade = classify_verified(r['verified_correct'])
        if grade == 'exclude':
            out.append((r, None, 'excluded-ungradeable'))
            continue
        fname = r['filename']
        candidates = master.get(fname, [])
        target_conf = float(r['ocr_confidence'])
        match = None
        for c in candidates:
            try:
                if abs(float(c['ocr_confidence']) - target_conf) < 1e-6:
                    match = c
                    break
            except (ValueError, KeyError):
                continue
        if match is None and len(candidates) == 1:
            match = candidates[0]
        if match is None:
            out.append((r, None, 'no-master-match'))
            continue
        texts = parse_ocr_raw(match['ocr_raw'])
        result, log_lines = run_row_dry(texts)
        frags = attribute_fragments(texts, log_lines)
        out.append((r, frags, grade))
    return out


def summarize(label, entries):
    print("=" * 100)
    print(label)
    print("=" * 100)
    true_confs = []
    false_confs = []
    unresolved = []
    for r, frags, grade in entries:
        sn = r['sample_number']
        if grade in ('excluded-ungradeable', 'no-master-match'):
            print(f"  sample{sn:>3}: EXCLUDED ({grade}) verified_correct={r['verified_correct']!r}")
            continue
        if frags is None:
            unresolved.append(sn)
            print(f"  sample{sn:>3}: grade={grade} -- attribution FAILED (no matching log line), skipping")
            continue
        if frags == 'TIER3_UNSUPPORTED':
            unresolved.append(sn)
            print(f"  sample{sn:>3}: grade={grade} -- resolved via Tier 3, not Tier 2 as expected, skipping")
            continue
        confs = [c for _, c in frags]
        frag_str = ", ".join(f"{t!r}={c:.2f}" for t, c in frags)
        print(f"  sample{sn:>3}: grade={grade} contributing_fragments=[{frag_str}]")
        if grade == 'TRUE':
            true_confs.extend(confs)
        else:
            false_confs.extend(confs)
    print()
    print(f"  Contributing-fragment confidences from TRUE (correct) rows: n={len(true_confs)}")
    if true_confs:
        print(f"    min={min(true_confs):.3f} max={max(true_confs):.3f} "
              f"mean={stats.mean(true_confs):.3f} median={stats.median(true_confs):.3f}")
        print(f"    sorted: {sorted(true_confs)}")
    print(f"  Contributing-fragment confidences from FALSE (wrong) rows: n={len(false_confs)}")
    if false_confs:
        print(f"    min={min(false_confs):.3f} max={max(false_confs):.3f} "
              f"mean={stats.mean(false_confs):.3f} median={stats.median(false_confs):.3f}")
        print(f"    sorted: {sorted(false_confs)}")
    if unresolved:
        print(f"  UNRESOLVED attribution: samples {unresolved}")
    print()
    return true_confs, false_confs


t1e = process_tier1_exact()
t1e_true, t1e_false = summarize("TIER 1-EXACT", t1e)

t2e = process_tier2_sheet('tier2_exact_review_samples/review_sheet.csv')
t2e_true, t2e_false = summarize("TIER 2-EXACT", t2e)

t2r = process_tier2_sheet('tier2_recovery_review_samples/review_sheet.csv')
t2r_true, t2r_false = summarize("TIER 2-RECOVERY", t2r)

t1b = process_tier1b()
t1b_true, t1b_false = summarize("TIER 1B", t1b)

print("=" * 100)
print("COMBINED TIER2 (exact + recovery) -- this is the main dataset for the merge-confidence question")
print("=" * 100)
combined_true = t2e_true + t2r_true
combined_false = t2e_false + t2r_false
print(f"TRUE (correct) merges: n={len(combined_true)} min={min(combined_true):.3f} max={max(combined_true):.3f} "
      f"mean={stats.mean(combined_true):.3f} median={stats.median(combined_true):.3f}")
print(f"FALSE (wrong) merges:  n={len(combined_false)} min={min(combined_false):.3f} max={max(combined_false):.3f} "
      f"mean={stats.mean(combined_false):.3f} median={stats.median(combined_false):.3f}")
print()
print("Sorted TRUE:", sorted(combined_true))
print("Sorted FALSE:", sorted(combined_false))
