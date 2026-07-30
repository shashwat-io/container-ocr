#!/usr/bin/env python3
"""
READ-ONLY. No OCR/PaddleX calls, no ocr.py edits, no server restart.

Replays the EXACT tier precedence from app.inference.ocr._try_extract (Tier 1,
1b, 2, 3-Pass1, 3-Pass2, Tier 4) against the raw OCR texts already stored in
benchmark_results.csv's ocr_raw column, using the real validate_iso6346 /
_compute_check_digit / clean_ocr_text functions (pure math, no live OCR).

Classifies each valid row into exactly one of:
  a) literal        - Tier 1, Tier 2 (literal 11-char), or Tier 3 Pass 1:
                       an 11-char string was read directly and its checksum
                       validated as-is, no digit was computed.
  b) check_digit_recovered - Tier 1b, Tier 2 (10-char recovery), or Tier 3
                       Pass 2: a 10-char shape match with a COMPUTED check
                       digit appended.
  c) confusion_corrected - Tier 4: single-position character substitution.

Sanity check: the code this classifier arrives at must match the CSV's
recorded container_code for every row, or the replay is wrong.
"""
import csv, re, sys
from collections import Counter

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
from app.inference.ocr import clean_ocr_text, validate_iso6346, _compute_check_digit, _CHAR_CONFUSION_MAP

NEW_CSV = 'benchmark_results.csv'
BASE_CSV = 'benchmark_results_pre_twopass.csv'

def parse_texts(raw):
    out = []
    for p in raw.split(' | '):
        p = p.strip()
        if not p:
            continue
        m = re.match(r'^(.*)\(([0-9.]+)\)$', p)
        if m:
            out.append((m.group(1), float(m.group(2))))
        else:
            out.append((p, None))
    return out

def classify(texts):
    """Replays _try_extract's tier cascade. Returns (code, bucket, tier_name, owner_line_conf)."""
    if not texts:
        return '', None, None, None

    # Tier 1: per-line exact 11-char match
    for text, conf in texts:
        line = clean_ocr_text(text)
        if len(line) == 11 and line[:4].isalpha() and line[4:11].isdigit():
            valid, _ = validate_iso6346(line)
            if valid:
                return line, 'literal', 'Tier1', conf

    # Tier 1b: per-line 10-char + computed check digit
    for text, conf in texts:
        line = clean_ocr_text(text)
        if len(line) == 10 and line[:4].isalpha() and line[4:10].isdigit():
            try:
                check = _compute_check_digit(line)
                candidate = line + str(check)
                valid, _ = validate_iso6346(candidate)
                if valid:
                    return candidate, 'check_digit_recovered', 'Tier1b', conf
            except KeyError:
                pass

    # Tier 2: adjacent line pairs
    for i in range(len(texts) - 1):
        text_i, conf_i = texts[i]
        text_ip1, conf_ip1 = texts[i + 1]
        pair_conf = (conf_i, conf_ip1)
        pair = clean_ocr_text(text_i + text_ip1)
        for start in range(len(pair) - 10):
            candidate = pair[start:start + 11]
            if not (candidate[:4].isalpha() and candidate[4:11].isdigit()):
                continue
            valid, _ = validate_iso6346(candidate)
            if valid:
                return candidate, 'literal', 'Tier2-literal', pair_conf
        if len(pair) == 10 and pair[:4].isalpha() and pair[4:10].isdigit():
            try:
                check = _compute_check_digit(pair)
                candidate = pair + str(check)
                valid, _ = validate_iso6346(candidate)
                if valid:
                    return candidate, 'check_digit_recovered', 'Tier2-recovery', pair_conf
            except KeyError:
                pass

    # Tier 3: full merged-string fallback
    full_text = clean_ocr_text(''.join(t for t, _ in texts))

    for start in range(len(full_text) - 10):
        candidate = full_text[start:start + 11]
        if len(candidate) < 11:
            break
        if not (candidate[:4].isalpha() and candidate[4:11].isdigit()):
            continue
        valid, _ = validate_iso6346(candidate)
        if valid:
            return candidate, 'literal', 'Tier3-Pass1', None

    for start in range(len(full_text) - 9):
        ten = full_text[start:start + 10]
        if len(ten) < 10:
            break
        if not (ten[:4].isalpha() and ten[4:10].isdigit()):
            continue
        try:
            check = _compute_check_digit(ten)
            candidate = ten + str(check)
            valid, _ = validate_iso6346(candidate)
            if valid:
                return candidate, 'check_digit_recovered', 'Tier3-Pass2', None
        except KeyError:
            continue

    # No valid found via 1-Pass2 — best-format fallback, then Tier 4
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

    best_valid, _ = validate_iso6346(best_code) if len(best_code) == 11 else (False, '')
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
                t4_valid, _ = validate_iso6346(t4_candidate)
                if t4_valid:
                    return t4_candidate, 'confusion_corrected', 'Tier4', None

    return best_code, None, None, None  # invalid, shouldn't happen for rows we filter as is_valid


def load_rows(path):
    return list(csv.DictReader(open(path)))

if __name__ == '__main__':
    new_rows = load_rows(NEW_CSV)
    mismatches = 0
    total_valid = 0
    for r in new_rows:
        if r['is_valid'].strip().lower() != 'true':
            continue
        total_valid += 1
        texts = parse_texts(r['ocr_raw'])
        code, bucket, tier, conf = classify(texts)
        if code != r['container_code']:
            mismatches += 1
            if mismatches <= 10:
                print(f"MISMATCH: {r['filename'][:40]} csv_code={r['container_code']!r} replay_code={code!r} bucket={bucket} tier={tier} raw={r['ocr_raw']!r}")
    print(f"\nTotal valid rows: {total_valid}")
    print(f"Replay mismatches: {mismatches} ({mismatches/total_valid*100:.1f}%)")
