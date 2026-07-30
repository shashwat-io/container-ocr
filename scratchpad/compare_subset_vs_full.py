#!/usr/bin/env python3
"""
Read-only comparison: does the 157-image quick sample reproduce inside the
full 786-image run, and how far apart are their accuracies?

The quick sample is a seed-42 stratified subset of the SAME resplit test set
the full run uses, and the cascade is confirmed deterministic on CPU (Friday's
3896crop replay reproduced bit-for-bit). So every quick-sample row should
appear verbatim inside the full run's output. Any mismatch means either
non-determinism or a checkpoint/code drift between the two runs, and is a
finding in its own right.

Reads only. Writes nothing. No inference, no model loading -- pure CSV
arithmetic, safe to run alongside a benchmark.

  python3 scratchpad/compare_subset_vs_full.py

=== Join key ===
NEITHER CSV HAS A box_index COLUMN. Both are written one row per YOLO
detection box, in image order, with the filename repeated for multi-box
images. box_index is therefore DERIVED here as the ordinal position of a row
within its filename group, in file order.

That makes in-order consumption mandatory, not stylistic: in the thin cap=3
quick-sample CSV, 157 distinct filenames produce 166 rows -- 7 filenames carry
more than one row (9 extra rows, so at least two carry three). Keying a plain
dict on filename alone would silently collapse those into one another and
quietly drop 9 rows from the comparison.

(Unrelated to CLAUDE.md's "7 known duplicate rows" in manual_labels.csv, which
are keyed on (image_path basename, box_index) through manifest.csv. The count
matching at 7 is a coincidence. The in-order discipline is the same either way.)

=== Accuracy convention ===
rows/rows throughout -- valid rows divided by TOTAL rows, never by image
count. This is the convention the historical 69.0%/64.7% figures use. Mixing
denominators has already produced one bad "73.8%" figure in this project.
Every percentage printed below is rows/rows and says so.
"""
import csv
import os
import sys
from collections import defaultdict

REPO = '/home/shashwatkumar/Testing/container_ocr_test'

SAMPLE_FILE = os.path.join(REPO, 'quick_sample_set_resplit.txt')
FULL_CSV    = os.path.join(REPO, 'benchmark_results_trocr_cascade_FULL_thin_20260728.csv')
QUICK_CSV   = os.path.join(REPO, 'quick_sample_thin_20260727.csv')

EXPECTED_SAMPLE_IMAGES = 157
EXPECTED_FULL_IMAGES   = 786

# Fields whose disagreement counts as a genuine prediction difference.
# tier is deliberately EXCLUDED: CLAUDE.md documents that the benchmark's tier
# capture records the LAST "Tier " log line across all rotation attempts, not
# necessarily the one that produced the returned result, so it is approximate
# by construction and a tier-only difference is not evidence of a prediction
# difference. It is still reported separately below for visibility.
COMPARE_FIELDS = ['container_code', 'is_valid', 'validation_reason']


def die(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def load_sample_names(path):
    """Filenames from the sample list, skipping '#' comment/header lines."""
    if not os.path.exists(path):
        die(f"sample file not found: {path}")
    names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                names.append(os.path.basename(line))
    return names


def load_rows(path, label):
    """
    Rows in file order, each tagged with its derived box_index.

    Returns (rows, key_to_rows) where key_to_rows maps
    (basename, box_index) -> list of rows in the order they appeared.
    """
    if not os.path.exists(path):
        die(f"{label} CSV not found: {path}")
    rows = []
    seen_count = defaultdict(int)
    key_to_rows = defaultdict(list)
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            die(f"{label} CSV is empty: {path}")
        for r in reader:
            base = os.path.basename(r['filename'])
            box_index = seen_count[base]
            seen_count[base] += 1
            r['_basename'] = base
            r['_box_index'] = box_index
            rows.append(r)
            key_to_rows[(base, box_index)].append(r)
    return rows, key_to_rows


def is_valid_true(row):
    """The CSV stores Python bools stringified; be tolerant about casing."""
    return str(row.get('is_valid', '')).strip().lower() == 'true'


def pct(numer, denom):
    return (100.0 * numer / denom) if denom else 0.0


def fmt_acc(label, valid, total):
    return f"{label}: {valid}/{total} rows = {pct(valid, total):.1f}% (rows/rows)"


def main():
    print("=" * 74)
    print("Quick-sample subset vs full-run comparison -- TrOCR thin (cap=3)")
    print("=" * 74)

    # ── Inputs ───────────────────────────────────────────────────────────────
    sample_names = load_sample_names(SAMPLE_FILE)
    sample_set = set(sample_names)
    print(f"\nSample file : {SAMPLE_FILE}")
    print(f"              {len(sample_names)} filenames "
          f"({len(sample_set)} distinct)")
    if len(sample_names) != EXPECTED_SAMPLE_IMAGES:
        print(f"  WARNING: expected {EXPECTED_SAMPLE_IMAGES} sample filenames, "
              f"got {len(sample_names)}")

    full_rows, full_keyed = load_rows(FULL_CSV, 'full-run')
    quick_rows, quick_keyed = load_rows(QUICK_CSV, 'quick-sample')

    full_images = len({r['_basename'] for r in full_rows})
    quick_images = len({r['_basename'] for r in quick_rows})

    print(f"\nFull CSV    : {FULL_CSV}")
    print(f"              {len(full_rows)} rows / {full_images} distinct images")
    print(f"Quick CSV   : {QUICK_CSV}")
    print(f"              {len(quick_rows)} rows / {quick_images} distinct images")

    # ── Refuse to analyse a partial full run ─────────────────────────────────
    # The full CSV is rewritten at every checkpoint, so a mid-run file looks
    # perfectly well-formed while containing a fraction of the data. Comparing
    # against it would silently produce a meaningless answer.
    if full_images < EXPECTED_FULL_IMAGES:
        die(f"full-run CSV holds only {full_images} of {EXPECTED_FULL_IMAGES} "
            f"images -- the run is still in progress (the CSV is rewritten at "
            f"every checkpoint). Re-run this script after it completes.")

    # ── Extract the sample subset from the full run ──────────────────────────
    subset_rows = [r for r in full_rows if r['_basename'] in sample_set]
    subset_images = len({r['_basename'] for r in subset_rows})
    print(f"\nSubset extracted from full run: {len(subset_rows)} rows / "
          f"{subset_images} distinct images")
    missing = sample_set - {r['_basename'] for r in full_rows}
    if missing:
        print(f"  WARNING: {len(missing)} sample images absent from the full run:")
        for name in sorted(missing)[:5]:
            print(f"    {name}")

    # ── (a) Row-for-row reproduction ─────────────────────────────────────────
    # Walk the QUICK csv in its own order and consume the matching full-run row
    # at the same (basename, box_index). Both files are written in the same
    # sorted-filename order by the same script, so ordinals line up positionally.
    print("\n" + "-" * 74)
    print("(a) Row-for-row reproduction")
    print("-" * 74)

    consumed = defaultdict(int)
    differing = []
    tier_only_diffs = []
    unmatched = []
    compared = 0

    for q in quick_rows:
        key = (q['_basename'], q['_box_index'])
        candidates = full_keyed.get(key, [])
        idx = consumed[key]
        if idx >= len(candidates):
            unmatched.append(q)
            continue
        f = candidates[idx]
        consumed[key] += 1
        compared += 1

        diffs = {fld: (q.get(fld, ''), f.get(fld, ''))
                 for fld in COMPARE_FIELDS
                 if q.get(fld, '') != f.get(fld, '')}
        if diffs:
            differing.append((q, f, diffs))
        elif q.get('tier', '') != f.get('tier', ''):
            tier_only_diffs.append((q, f))

    print(f"  rows compared        : {compared}")
    print(f"  identical predictions: {compared - len(differing)}")
    print(f"  DIFFERING rows       : {len(differing)}")
    if unmatched:
        print(f"  UNMATCHED quick rows : {len(unmatched)} "
              f"(no counterpart at the same (basename, box_index))")
        for q in unmatched[:5]:
            print(f"    {q['_basename']} [box {q['_box_index']}]")
    if tier_only_diffs:
        print(f"  tier-label-only diffs: {len(tier_only_diffs)} "
              f"(prediction identical; tier capture is approximate by design)")

    if differing:
        print(f"\n  First {min(10, len(differing))} differing rows:")
        for q, f, diffs in differing[:10]:
            print(f"\n    {q['_basename']} [box {q['_box_index']}]")
            for fld, (qv, fv) in diffs.items():
                print(f"      {fld}:")
                print(f"        quick: {qv!r}")
                print(f"        full : {fv!r}")
    else:
        print("\n  EXACT REPRODUCTION -- every compared row matches on "
              f"{', '.join(COMPARE_FIELDS)}.")

    # ── (b)(c)(d) Accuracy ───────────────────────────────────────────────────
    print("\n" + "-" * 74)
    print("(b)(c)(d) Accuracy -- rows/rows convention throughout")
    print("-" * 74)

    quick_valid  = sum(1 for r in quick_rows  if is_valid_true(r))
    subset_valid = sum(1 for r in subset_rows if is_valid_true(r))
    full_valid   = sum(1 for r in full_rows   if is_valid_true(r))

    subset_acc = pct(subset_valid, len(subset_rows))
    full_acc   = pct(full_valid, len(full_rows))
    quick_acc  = pct(quick_valid, len(quick_rows))

    print(f"\n  (reference) standalone quick-sample run")
    print(f"    {fmt_acc('  157-image run', quick_valid, len(quick_rows))}")
    print(f"\n  (b) 157-image subset, measured INSIDE the full run")
    print(f"    {fmt_acc('  subset', subset_valid, len(subset_rows))}")
    print(f"\n  (c) full 786-image run")
    print(f"    {fmt_acc('  full', full_valid, len(full_rows))}")
    print(f"\n  (d) GAP")
    print(f"      subset - full = {subset_acc - full_acc:+.1f} points")
    print(f"      (subset {subset_acc:.1f}% vs full {full_acc:.1f}%)")

    # The remaining 629 images are the part the quick sample never saw. If the
    # subset reproduces but the full number is lower, this is where it went.
    rest_rows = [r for r in full_rows if r['_basename'] not in sample_set]
    rest_valid = sum(1 for r in rest_rows if is_valid_true(r))
    print(f"\n  For context -- the {len(rest_rows)} rows the quick sample never covered:")
    print(f"    {fmt_acc('  non-sample remainder', rest_valid, len(rest_rows))}")

    # ── Interpretation guard ─────────────────────────────────────────────────
    # n=166 rows carries a standard error of roughly +/-3.6 points, so the
    # quick sample cannot resolve differences smaller than about 5 points.
    print("\n" + "-" * 74)
    print("Reading this result")
    print("-" * 74)
    gap = abs(subset_acc - full_acc)
    print(f"  At n={len(subset_rows)} rows the subset's standard error is ~+/-3.6 points,")
    print(f"  so a gap under ~5 points is inside sampling noise.")
    if gap < 5:
        print(f"  Observed gap {gap:.1f} points -> WITHIN noise; the quick sample")
        print("  is not demonstrably biased as an estimator of the full set.")
    else:
        print(f"  Observed gap {gap:.1f} points -> EXCEEDS noise; the quick sample")
        print("  is a biased estimator of full-set performance and any decision")
        print("  made on quick-sample deltas should be revisited.")
    print()


if __name__ == '__main__':
    main()
