"""
Standalone, read-only replay of the _try_extract tier-matching logic
(app/inference/ocr.py lines 160-273) against already-captured ocr_raw text
from benchmark_results_full_live.csv.

Why this exists: the benchmark's tier-capture instrumentation grabs whichever
internal _try_extract call logged a "Tier N (...)" message LAST across up to
4 attempts per image (docpp-on/off retry x +-90 deg rotation), which is not
necessarily the attempt whose result was actually returned. That corrupts the
per-tier breakdown for ~36% of passing rows. ocr_raw and validation_reason in
the CSV are NOT affected by that bug (they come straight from the winning
call's own return value), so re-running the pure tier-matching logic against
the stored ocr_raw recovers the true tier for most rows.

This script does NOT call run_ocr_on_crop, _try_extract, or anything that
touches PaddleX/YOLO -- it only imports pure helper functions from ocr.py
(clean_ocr_text, _compute_check_digit, validate_iso6346,
_CHAR_CONFUSION_MAP) and reimplements the tier control flow around them, so
there's zero drift risk against the tier-matching rules themselves while
staying fully offline.

Usage: INPUT_CSV=... OUTPUT_CSV=... python3 scripts/replay_tier_attribution_resplit.py
Reads:  $INPUT_CSV (default benchmark_results_full_live.csv)
Writes: $OUTPUT_CSV (default benchmark_results_full_live_corrected_tiers.csv)

Copy of scripts/replay_tier_attribution.py with INPUT_CSV/OUTPUT_CSV made
env-var overridable so it can be reused, unmodified in logic, against the
quick-sample resplit benchmark CSVs. Original script untouched.
"""
import csv
import os
import re
import sys

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')

from app.inference.ocr import (
    clean_ocr_text,
    _compute_check_digit,
    validate_iso6346,
    _CHAR_CONFUSION_MAP,
)

INPUT_CSV = os.environ.get(
    'INPUT_CSV',
    '/home/shashwatkumar/Testing/container_ocr_test/benchmark_results_full_live.csv')
OUTPUT_CSV = os.environ.get(
    'OUTPUT_CSV',
    '/home/shashwatkumar/Testing/container_ocr_test/benchmark_results_full_live_corrected_tiers.csv')

_RAW_PIECE_RE = re.compile(r'^(.*)\((\d+\.\d+)\)$')


def parse_ocr_raw(ocr_raw):
    """Inverse of ' | '.join(f"{t}({c:.2f})" for t, c in texts) in ocr.py."""
    texts = []
    if not ocr_raw:
        return texts
    for piece in ocr_raw.split(' | '):
        m = _RAW_PIECE_RE.match(piece)
        if not m:
            continue
        text, conf = m.group(1), m.group(2)
        texts.append((text, float(conf)))
    return texts


def replay_try_extract(texts):
    """
    Reimplementation of the tier-matching control flow from
    app/inference/ocr.py::_try_extract, lines 160-273 (Tier 1 through Tier 4).
    Returns (tier_label, code, valid, reason).

    Deliberately mirrors the source structure and order 1:1 so there is no
    behavioral drift; only the pure helpers (clean_ocr_text,
    _compute_check_digit, validate_iso6346, _CHAR_CONFUSION_MAP) are imported
    directly from ocr.py rather than reimplemented.
    """
    if not texts:
        return 'none', '', False, 'No text detected'

    # -- Tier 1: per-line exact 11-char match -------------------------------
    for text, _ in texts:
        line = clean_ocr_text(text)
        if len(line) == 11 and line[:4].isalpha() and line[4:11].isdigit():
            valid, reason = validate_iso6346(line)
            if valid:
                return 'Tier1-exact', line, True, reason

    # -- Tier 1b: per-line 10-char check-digit recovery ----------------------
    for text, _ in texts:
        line = clean_ocr_text(text)
        if len(line) == 10 and line[:4].isalpha() and line[4:10].isdigit():
            try:
                check = _compute_check_digit(line)
                candidate = line + str(check)
                valid, reason = validate_iso6346(candidate)
                if valid:
                    return 'Tier1b-recovery', candidate, True, "Valid (check-digit recovered)"
            except KeyError:
                pass

    # -- Tier 2: adjacent line pairs -----------------------------------------
    for i in range(len(texts) - 1):
        pair = clean_ocr_text(texts[i][0] + texts[i + 1][0])
        for start in range(len(pair) - 10):
            candidate = pair[start:start + 11]
            if not (candidate[:4].isalpha() and candidate[4:11].isdigit()):
                continue
            valid, reason = validate_iso6346(candidate)
            if valid:
                return 'Tier2-exact', candidate, True, reason
        if len(pair) == 10 and pair[:4].isalpha() and pair[4:10].isdigit():
            try:
                check = _compute_check_digit(pair)
                candidate = pair + str(check)
                valid, reason = validate_iso6346(candidate)
                if valid:
                    return 'Tier2-recovery', candidate, True, "Valid (check-digit recovered)"
            except KeyError:
                pass

    # -- Tier 3: full merged-string fallback ---------------------------------
    full_text = clean_ocr_text(''.join(t for t, _ in texts))

    # Pass 1: sliding 11-char window - full ISO 6346 structural match + checksum.
    for start in range(len(full_text) - 10):
        candidate = full_text[start:start + 11]
        if len(candidate) < 11:
            break
        if not (candidate[:4].isalpha() and candidate[4:11].isdigit()):
            continue
        valid, reason = validate_iso6346(candidate)
        if valid:
            return 'Tier3-Pass1', candidate, True, reason

    # Pass 2 is permanently disabled in ocr.py (81% error rate) - no replay needed.

    # No valid found - best-format candidate for manual correction.
    best_code = ''
    for start in range(len(full_text) - 10):
        candidate = full_text[start:start + 11]
        if len(candidate) < 11:
            break
        if candidate[:4].isalpha() and candidate[4:11].isdigit():
            best_code = candidate
            break

    if not best_code:
        best_code = full_text[:11] if len(full_text) >= 11 else full_text

    best_valid, best_reason = validate_iso6346(best_code) if len(best_code) == 11 else (False, 'Text too short')

    # -- Tier 4: single-position character-confusion correction --------------
    if not best_valid and len(best_code) == 11:
        violations = []
        for pos, ch in enumerate(best_code):
            if pos <= 3 and not ch.isalpha():
                violations.append(pos)
            elif pos >= 4 and not ch.isdigit():
                violations.append(pos)
        if len(violations) == 1:
            vpos = violations[0]
            old_char = best_code[vpos]
            new_char = _CHAR_CONFUSION_MAP.get(old_char)
            if new_char is not None:
                t4_candidate = best_code[:vpos] + new_char + best_code[vpos + 1:]
                t4_valid, t4_reason = validate_iso6346(t4_candidate)
                if t4_valid:
                    reason = (
                        f"Valid (character confusion corrected: "
                        f"{old_char}→{new_char} at position {vpos + 1})"
                    )
                    return 'Tier4-confusion', t4_candidate, True, reason

    return 'unresolved', best_code, best_valid, best_reason


def main():
    with open(INPUT_CSV, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    tier_counts = {}
    unrecoverable = []
    out_rows = []

    for row in rows:
        is_valid_csv = row['is_valid'] == 'True'
        out_row = dict(row)
        out_row['corrected_tier'] = ''
        out_row['replay_code'] = ''
        out_row['replay_agrees'] = ''

        if not is_valid_csv:
            # Out of scope: only passing rows are subject to the tier
            # log-capture bug (see module docstring). Leave as-is.
            out_rows.append(out_row)
            continue

        texts = parse_ocr_raw(row['ocr_raw'])
        tier, code, valid, reason = replay_try_extract(texts)

        agrees = valid and (code == row['container_code'])
        out_row['replay_code'] = code
        out_row['replay_agrees'] = str(agrees)

        if agrees:
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            out_row['corrected_tier'] = tier
        else:
            out_row['corrected_tier'] = 'UNRECOVERABLE'
            unrecoverable.append({
                'filename': row['filename'],
                'csv_code': row['container_code'],
                'csv_reason': row['validation_reason'],
                'replay_code': code,
                'replay_valid': valid,
                'replay_reason': reason,
            })

        out_rows.append(out_row)

    out_fieldnames = fieldnames + ['corrected_tier', 'replay_code', 'replay_agrees']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    total_valid = sum(1 for r in rows if r['is_valid'] == 'True')
    total_tier = sum(tier_counts.values())

    print("=== Replayed tier attribution (offline, from stored ocr_raw) ===")
    for tier in ['Tier1-exact', 'Tier1b-recovery', 'Tier2-exact', 'Tier2-recovery',
                 'Tier3-Pass1', 'Tier4-confusion']:
        print(f"  {tier:20s} {tier_counts.get(tier, 0)}")
    other_tiers = set(tier_counts) - {
        'Tier1-exact', 'Tier1b-recovery', 'Tier2-exact', 'Tier2-recovery',
        'Tier3-Pass1', 'Tier4-confusion',
    }
    for tier in sorted(other_tiers):
        print(f"  {tier:20s} {tier_counts[tier]}  (unexpected label - investigate)")

    print()
    print(f"Unrecoverable (replay disagrees with CSV): {len(unrecoverable)}")
    print(f"Reconciliation: {total_tier} (tier-attributed) + {len(unrecoverable)} (unrecoverable) "
          f"= {total_tier + len(unrecoverable)}  (expected: {total_valid} passing rows)")
    if total_tier + len(unrecoverable) != total_valid:
        print("  *** MISMATCH - counts do not reconcile, investigate before trusting output ***")

    if unrecoverable:
        print()
        print("--- Sample of unrecoverable rows (up to 10) ---")
        for u in unrecoverable[:10]:
            print(f"  {u['filename']}: csv_code={u['csv_code']!r} csv_reason={u['csv_reason']!r} "
                  f"| replay_code={u['replay_code']!r} replay_valid={u['replay_valid']} "
                  f"replay_reason={u['replay_reason']!r}")

    print()
    print(f"Corrected per-row CSV written to: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
