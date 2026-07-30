"""
Dry-test _try_extract() (Tiers 1-4) against benchmark_results.csv rows
where is_valid=False, without invoking PaddleX at all.

Run: python3 dry_run_tier_check.py
"""
import csv
import re
import sys
import os
import logging

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

# Silence all logging from the OCR module so output is clean.
logging.getLogger('app.inference.ocr').setLevel(logging.CRITICAL)

import app.inference.ocr as ocr_mod

CSV_PATH = 'benchmark_results.csv'

# ── ocr_raw parser ────────────────────────────────────────────────────────────
_ENTRY_RE = re.compile(r'(.+?)\(([0-9.]+)\)')

def parse_ocr_raw(ocr_raw: str) -> list:
    """'FOO(0.99) | BAR(0.94)' → [('FOO', 0.99), ('BAR', 0.94)]"""
    if not ocr_raw.strip():
        return []
    results = []
    for part in ocr_raw.split(' | '):
        part = part.strip()
        m = _ENTRY_RE.fullmatch(part)
        if m:
            results.append((m.group(1), float(m.group(2))))
    return results

# ── load ALL rows from CSV ────────────────────────────────────────────────────
all_rows = []
with open(CSV_PATH, newline='') as f:
    for row in csv.DictReader(f):
        all_rows.append(row)

total_rows   = len(all_rows)
orig_valid   = sum(1 for r in all_rows if r['is_valid'].strip().lower() == 'true')
orig_invalid = total_rows - orig_valid

print(f"benchmark_results.csv — 842 lines on disk (1 header + {total_rows} data rows)")
print(f"  Originally valid  : {orig_valid}")
print(f"  Originally invalid: {orig_invalid}")
print(f"  Total             : {total_rows}")
print()

# ── replay every row through _try_extract() with stubbed OCR ─────────────────
# Categories:
#   valid   → valid,   same code      (expected, no change)
#   valid   → valid,   different code (worth noting)
#   valid   → invalid                 (REGRESSION)
#   invalid → valid                   (improvement)
#   invalid → invalid, same code      (unchanged failure)
#   invalid → invalid, different code (changed but not fixed — design concern)

vv_same      = []  # (row)
vv_different = []  # (row, new_code, new_reason)
regressions  = []  # (row, new_code, new_reason)  ← valid→invalid
flipped      = []  # (row, new_code, new_reason)  ← invalid→valid
inv_changed  = []  # (row, new_code, new_reason)  ← invalid→invalid, code changed
inv_same     = []  # (row)

_stub_texts = []

_orig_run_ocr = ocr_mod.run_ocr_on_crop
def _stub_run_ocr(_crop):
    return _stub_texts
ocr_mod.run_ocr_on_crop = _stub_run_ocr

for row in all_rows:
    old_code  = row['container_code']
    old_valid = row['is_valid'].strip().lower() == 'true'
    ocr_raw   = row['ocr_raw']

    _stub_texts = parse_ocr_raw(ocr_raw)
    new_code, new_valid, new_reason, _, _ = ocr_mod._try_extract(None)

    if old_valid and new_valid:
        if new_code == old_code:
            vv_same.append(row)
        else:
            vv_different.append((row, new_code, new_reason))
    elif old_valid and not new_valid:
        regressions.append((row, new_code, new_reason))
    elif not old_valid and new_valid:
        flipped.append((row, new_code, new_reason))
    else:  # both invalid
        if new_code != old_code:
            inv_changed.append((row, new_code, new_reason))
        else:
            inv_same.append(row)

ocr_mod.run_ocr_on_crop = _orig_run_ocr  # restore

# ── print results ─────────────────────────────────────────────────────────────
print(f"{'='*72}")
print(f"valid   → valid  (same code)     : {len(vv_same):4d}  ✓ expected")
print(f"valid   → valid  (diff code)     : {len(vv_different):4d}  (review below)")
print(f"valid   → invalid (REGRESSION)   : {len(regressions):4d}  ← must be zero")
print(f"invalid → valid                  : {len(flipped):4d}  ✓ improvement")
print(f"invalid → invalid (code changed) : {len(inv_changed):4d}  (review below)")
print(f"invalid → invalid (unchanged)    : {len(inv_same):4d}")
print(f"{'='*72}")
print(f"Total rows checked: {total_rows}")

# ── REGRESSIONS (must be zero) ────────────────────────────────────────────────
print(f"\n{'─'*72}")
if regressions:
    print(f"REGRESSIONS — {len(regressions)} row(s) where previously-valid code became invalid:")
    print(f"{'─'*72}")
    print(f"{'filename':<55} {'old_code':<14} {'new_code':<14} reason")
    for row, new_code, reason in regressions:
        print(f"{row['filename'][:54]:<55} {row['container_code']:<14} {new_code:<14} {reason}")
else:
    print("0 regressions found — no previously-valid row became invalid.")

# ── valid→valid different code ────────────────────────────────────────────────
print(f"\n{'─'*72}")
if vv_different:
    print(f"{len(vv_different)} valid→valid rows where code CHANGED (not necessarily wrong):")
    print(f"{'─'*72}")
    print(f"{'filename':<55} {'old_code':<14} {'new_code':<14} reason")
    for row, new_code, reason in vv_different:
        print(f"{row['filename'][:54]:<55} {row['container_code']:<14} {new_code:<14} {reason}")
else:
    print("No valid→valid code changes.")

# ── invalid→valid flips breakdown ─────────────────────────────────────────────
print(f"\n{'─'*72}")
reason_counts = {}
for _, _, reason in flipped:
    tag = reason if ('character confusion' in reason or 'check-digit' in reason) else 'Valid (other/checksum)'
    reason_counts[tag] = reason_counts.get(tag, 0) + 1
print(f"{len(flipped)} invalid→valid improvements, by reason:")
for tag, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:4d}  {tag}")

# ── invalid→invalid code-changed rows ────────────────────────────────────────
print(f"\n{'─'*72}")
if inv_changed:
    print(f"WARNING — {len(inv_changed)} invalid→invalid row(s) where best_code changed:")
    print(f"{'─'*72}")
    print(f"{'filename':<55} {'old_code':<14} {'new_code':<14} reason")
    for row, new_code, reason in inv_changed:
        print(f"{row['filename'][:54]:<55} {row['container_code']:<14} {new_code:<14} {reason}")
else:
    print("No invalid→invalid code changes — fallback output is stable.")
