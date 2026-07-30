#!/usr/bin/env python3
"""
READ-ONLY. No OCR/PaddleX calls, no server restart. ocr.py has already been
edited (Tier3-Pass2 check-digit recovery disabled) and backed up to
app/inference/ocr.py.pre_tier3fix.bak.

Sanity-check dry run: replay the raw OCR texts already captured in
benchmark_results.csv (generated under the OLD pre-fix logic) through the
NEW tier cascade (mirroring the just-edited _try_extract, with Pass 2
removed) to see what the full benchmark run would produce, before spending
60-90 min on a live re-run.

Confirms:
  1. All 71 former Tier3-Pass2 rows: new status (invalid, or picked up by
     the existing Tier4 fallback that already ran for other rows).
  2. Tier1b (157 rows) and Tier2-recovery (30 rows): unchanged code + status.
  3. No valid row outside bucket (b) is touched.
"""
import csv, re, sys
from collections import Counter

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
from app.inference.ocr import clean_ocr_text, validate_iso6346, _compute_check_digit, _CHAR_CONFUSION_MAP
from scratch_tier_classify import parse_texts, classify as classify_old

NEW_CSV = 'benchmark_results.csv'  # pre-tier3fix, still the 610/786 result

def classify_new(texts):
    """Mirrors the freshly-edited _try_extract: Tier1, Tier1b, Tier2, Tier3-Pass1 only (Pass2 removed), then Tier4."""
    if not texts:
        return '', None, None

    for text, conf in texts:
        line = clean_ocr_text(text)
        if len(line) == 11 and line[:4].isalpha() and line[4:11].isdigit():
            valid, _ = validate_iso6346(line)
            if valid:
                return line, 'literal', 'Tier1'

    for text, conf in texts:
        line = clean_ocr_text(text)
        if len(line) == 10 and line[:4].isalpha() and line[4:10].isdigit():
            try:
                check = _compute_check_digit(line)
                candidate = line + str(check)
                valid, _ = validate_iso6346(candidate)
                if valid:
                    return candidate, 'check_digit_recovered', 'Tier1b'
            except KeyError:
                pass

    for i in range(len(texts) - 1):
        text_i, _ = texts[i]
        text_ip1, _ = texts[i + 1]
        pair = clean_ocr_text(text_i + text_ip1)
        for start in range(len(pair) - 10):
            candidate = pair[start:start + 11]
            if not (candidate[:4].isalpha() and candidate[4:11].isdigit()):
                continue
            valid, _ = validate_iso6346(candidate)
            if valid:
                return candidate, 'literal', 'Tier2-literal'
        if len(pair) == 10 and pair[:4].isalpha() and pair[4:10].isdigit():
            try:
                check = _compute_check_digit(pair)
                candidate = pair + str(check)
                valid, _ = validate_iso6346(candidate)
                if valid:
                    return candidate, 'check_digit_recovered', 'Tier2-recovery'
            except KeyError:
                pass

    full_text = clean_ocr_text(''.join(t for t, _ in texts))

    for start in range(len(full_text) - 10):
        candidate = full_text[start:start + 11]
        if len(candidate) < 11:
            break
        if not (candidate[:4].isalpha() and candidate[4:11].isdigit()):
            continue
        valid, _ = validate_iso6346(candidate)
        if valid:
            return candidate, 'literal', 'Tier3-Pass1'

    # Pass 2 REMOVED — falls straight through to best-format fallback + Tier4

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
                    return t4_candidate, 'confusion_corrected', 'Tier4'

    return best_code, None, None  # invalid


if __name__ == '__main__':
    rows = list(csv.DictReader(open(NEW_CSV)))
    seen = Counter()
    total_valid_old = 0
    now_invalid = []
    now_tier4_rescued = []
    tier1b_changed = []
    tier2r_changed = []
    other_bucket_touched = []
    unaffected_count = Counter()

    for r in rows:
        f = r['filename']; idx = seen[f]; seen[f] += 1
        if r['is_valid'].strip().lower() != 'true':
            continue
        total_valid_old += 1
        texts = parse_texts(r['ocr_raw'])
        old_code, old_bucket, old_tier, _ = classify_old(texts)
        new_code, new_bucket, new_tier = classify_new(texts)

        assert old_code == r['container_code'], f"OLD replay mismatch on {f} idx={idx}"

        if old_tier == 'Tier3-Pass2':
            if new_bucket is None:
                now_invalid.append((f, idx, old_code))
            elif new_bucket == 'confusion_corrected':
                now_tier4_rescued.append((f, idx, old_code, new_code))
            else:
                now_tier4_rescued.append((f, idx, old_code, new_code))  # unexpected path, still log it
        elif old_tier == 'Tier1b':
            if new_code != old_code or new_bucket != old_bucket:
                tier1b_changed.append((f, idx, old_code, new_code, old_bucket, new_bucket))
            else:
                unaffected_count['Tier1b'] += 1
        elif old_tier == 'Tier2-recovery':
            if new_code != old_code or new_bucket != old_bucket:
                tier2r_changed.append((f, idx, old_code, new_code, old_bucket, new_bucket))
            else:
                unaffected_count['Tier2-recovery'] += 1
        else:
            # literal or confusion_corrected rows — must be 100% unaffected
            if new_code != old_code or new_bucket != old_bucket:
                other_bucket_touched.append((f, idx, old_code, new_code, old_bucket, new_bucket))
            else:
                unaffected_count[old_bucket] += 1

    print(f"Total valid rows in current (pre-fix) CSV: {total_valid_old}")
    print(f"\nFormer Tier3-Pass2 rows (71 expected): now_invalid={len(now_invalid)}  rescued_by_later_tier={len(now_tier4_rescued)}")
    if now_tier4_rescued:
        print("  -- rescued by Tier4 or other later tier --")
        for f, idx, oldc, newc in now_tier4_rescued:
            print(f"    {f[:46]} idx={idx}: {oldc!r} -> {newc!r}")

    print(f"\nTier1b rows unaffected: {unaffected_count.get('Tier1b',0)} / (changed: {len(tier1b_changed)})")
    if tier1b_changed:
        print("  !! Tier1b rows CHANGED (should be zero) !!")
        for row in tier1b_changed[:10]:
            print(f"    {row}")

    print(f"\nTier2-recovery rows unaffected: {unaffected_count.get('Tier2-recovery',0)} / (changed: {len(tier2r_changed)})")
    if tier2r_changed:
        print("  !! Tier2-recovery rows CHANGED (should be zero) !!")
        for row in tier2r_changed[:10]:
            print(f"    {row}")

    print(f"\nOther bucket rows (literal / confusion_corrected) touched (should be zero): {len(other_bucket_touched)}")
    if other_bucket_touched:
        for row in other_bucket_touched[:10]:
            print(f"    {row}")

    new_total_valid = total_valid_old - len(now_invalid)
    print(f"\nProjected new valid count: {total_valid_old} - {len(now_invalid)} (now-invalid) = {new_total_valid}")
    print(f"(vs current benchmark_results.csv total valid: {total_valid_old})")
