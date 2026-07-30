"""
READ-ONLY, no-PaddleX verification of the Tier3-must-not-resurface-an-
ambiguous-candidate fix.

Follows the repo's documented dry-run pattern (see AGENTS.md): parse the
`ocr_raw` column already captured in today's review sheets back into
(text, score) tuples, monkeypatch run_ocr_on_crop to return them verbatim,
and call _try_extract(None) directly. This exercises the pure tier logic
deterministically against the exact fragments already on record -- no live
OCR, no YOLO, no server.

Checks:
  - sample17 (tier2_exact row 17): must now return container_code that is
    NEITHER "XINU1955119" NOR "CXNU1955119", is_valid=False, and
    is_ambiguous_merge=True still present in the result.
  - All 23 tier2_exact + 35 tier2_recovery rows: zero regressions --every
    row previously graded TRUE/correct must still come out with the exact
    same container_code.
"""
import os
import re
import sys
import csv
from unittest import mock

sys.path.insert(0, '/home/shashwatkumar/Testing/container_ocr_test')
os.chdir('/home/shashwatkumar/Testing/container_ocr_test')

import app.inference.ocr as ocr_mod

_FRAG_RE = re.compile(r'^(.*)\(([\d.]+)\)$')


def parse_ocr_raw(raw):
    texts = []
    for chunk in raw.split(' | '):
        m = _FRAG_RE.match(chunk.strip())
        if not m:
            raise ValueError(f"Could not parse fragment: {chunk!r} in {raw!r}")
        text, score = m.group(1), float(m.group(2))
        texts.append((text, score))
    return texts


def run_row_dry(raw):
    texts = parse_ocr_raw(raw)
    with mock.patch.object(ocr_mod, 'run_ocr_on_crop', return_value=texts):
        return ocr_mod._try_extract(None)


def process_sheet(path, tier_name):
    rows = list(csv.DictReader(open(path)))
    print("=" * 100)
    print(f"{tier_name}: {len(rows)} rows")
    print("=" * 100)
    regressions = []
    changes = []
    ambiguous_hits = []
    for r in rows:
        old_code = r['claimed_code']
        verified = r['verified_correct'].strip()
        was_correct = verified.upper().startswith('TRUE')

        code, valid, reason, raw_out, conf = run_row_dry(r['ocr_raw'])
        new_ambig = ocr_mod.is_ambiguous_merge(reason)

        changed = (code != old_code)
        tag = ""
        if changed:
            changes.append((r['sample_number'], old_code, code, was_correct, verified))
            if was_correct:
                regressions.append((r['sample_number'], r['filename'], old_code, code, verified))
                tag = "  <<< REGRESSION (was correct, now changed)"
            else:
                tag = "  <<< changed (was marked wrong/known-bad)"
        if new_ambig:
            tag += "  <<< is_ambiguous_merge=True"
            ambiguous_hits.append((r['sample_number'], r['filename']))

        print(f"sample{r['sample_number']:>3} | old={old_code:12s} new={code:12s} "
              f"valid={valid} ambig={new_ambig} reason={reason!r}{tag}")

    print()
    print(f"{tier_name} summary: {len(changes)} changed codes out of {len(rows)}; "
          f"{len(regressions)} regressions (were TRUE/correct)")
    print(f"{tier_name} is_ambiguous_merge=True count: {len(ambiguous_hits)} -- {ambiguous_hits}")
    if regressions:
        print("REGRESSIONS:")
        for reg in regressions:
            print("  ", reg)
    return regressions, changes, ambiguous_hits


reg1, ch1, amb1 = process_sheet('tier2_exact_review_samples/review_sheet.csv', 'Tier2-exact')
print()
reg2, ch2, amb2 = process_sheet('tier2_recovery_review_samples/review_sheet.csv', 'Tier2-recovery')

print()
print("=" * 100)
print(f"TOTAL REGRESSIONS: {len(reg1) + len(reg2)}")
print(f"TOTAL is_ambiguous_merge=True rows: {len(amb1) + len(amb2)} -- {amb1 + amb2}")
print(f"TOTAL changed rows (regressions + expected/known-bad changes): {len(ch1) + len(ch2)}")
for c in ch1 + ch2:
    print("  changed:", c)
print("=" * 100)
