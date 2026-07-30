import logging
import os
import re
import threading
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io

logger = logging.getLogger(__name__)

# Override the OCR pipeline's text-recognition submodel with a fine-tuned
# checkpoint's exported inference/ dir (must contain inference.json,
# inference.pdiparams, inference.yml). Leave unset to use the stock
# PP-OCRv6_medium_rec pipeline -- the only configuration this app or its
# benchmarks have ever run with to date. Unset/empty is always safe to fall
# back to; this must be explicitly set to change production behavior.
FINETUNED_REC_MODEL_DIR = os.environ.get('FINETUNED_REC_MODEL_DIR', '')

# Which text recognizer the cascade runs on. 'paddle' (the default) leaves
# every code path in this module byte-identical to what production has always
# run -- same safety contract as FINETUNED_REC_MODEL_DIR above. 'trocr' swaps
# ONLY the recognizer call inside run_ocr_on_crop(); cropping, _post_process,
# rotation selection, the entire tier cascade and validation are deliberately
# left untouched, because that is exactly the configuration
# benchmark_trocr_cascade_quicksample.py measured. Changing anything else
# breaks the correspondence between the benchmark numbers and production.
OCR_RECOGNIZER = os.environ.get('OCR_RECOGNIZER', 'paddle').strip().lower()
if OCR_RECOGNIZER not in ('paddle', 'trocr'):
    raise ValueError(
        f"OCR_RECOGNIZER must be 'paddle' or 'trocr', got {OCR_RECOGNIZER!r}. "
        "Refusing to guess: an unrecognised value quietly falling back to "
        "'paddle' is precisely how a TrOCR deployment ends up serving "
        "PaddleOCR without anyone noticing."
    )

# ISO 6346 character value table (letters skip multiples of 11).
# Placed at module level so both validate_iso6346 and _compute_check_digit share it.
_ISO6346_VALUE_MAP: dict = {
    'A':10,'B':12,'C':13,'D':14,'E':15,'F':16,'G':17,'H':18,'I':19,'J':20,
    'K':21,'L':23,'M':24,'N':25,'O':26,'P':27,'Q':28,'R':29,'S':30,'T':31,
    'U':32,'V':34,'W':35,'X':36,'Y':37,'Z':38,
    **{str(d): d for d in range(10)},
}

_CHAR_CONFUSION_MAP = {
    'O': '0', '0': 'O',
    'I': '1', '1': 'I',
    'S': '5', '5': 'S',
    'B': '8', '8': 'B',
    'Z': '2', '2': 'Z',
    'G': '6', '6': 'G',
    'C': '6',
}

_ocr_instance = None
# Paddle's static-graph inference predictor is NOT thread-safe: its input/output
# buffers are shared across calls. Serialise every predict() call with this lock.
# Also guards lazy initialisation to avoid double-init under concurrent requests.
_ocr_lock = threading.Lock()


def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:           # fast path — no lock needed once initialised
        with _ocr_lock:
            if _ocr_instance is None:   # double-check: another thread may have init'd
                import paddlex as pdx
                if FINETUNED_REC_MODEL_DIR:
                    from paddlex.inference.pipelines import load_pipeline_config
                    config = load_pipeline_config('OCR')
                    config['SubModules']['TextRecognition']['model_dir'] = FINETUNED_REC_MODEL_DIR
                    logger.info(f"get_ocr: FINETUNED_REC_MODEL_DIR set -- overriding "
                                f"TextRecognition.model_dir -> {FINETUNED_REC_MODEL_DIR}")
                    _ocr_instance = pdx.create_pipeline(
                        config=config,
                        device='cpu',
                        engine_config={'paddle_static': {'enable_new_ir': False, 'run_mode': 'paddle'}}
                    )
                else:
                    _ocr_instance = pdx.create_pipeline(
                        'OCR',
                        device='cpu',
                        engine_config={'paddle_static': {'enable_new_ir': False, 'run_mode': 'paddle'}}
                    )
    return _ocr_instance


def crop_detection(image_bytes, x1, y1, x2, y2, padding=10):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    w, h = image.size
    left   = max(0, int(x1) - padding)
    top    = max(0, int(y1) - padding)
    right  = min(w, int(x2) + padding)
    bottom = min(h, int(y2) + padding)
    return image.crop((left, top, right, bottom))


def clean_ocr_text(text):
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text


def _compute_check_digit(ten: str) -> int:
    """Return the ISO 6346 check digit for a 10-character owner+serial string."""
    total = sum(_ISO6346_VALUE_MAP[ten[i]] * (2 ** i) for i in range(10))
    remainder = total % 11
    return 0 if remainder == 10 else remainder


def validate_iso6346(code):
    if len(code) != 11:
        return False, f"Wrong length: {len(code)} (expected 11)"
    if not code[:4].isalpha():
        return False, "First 4 characters must be letters"
    if not code[4:10].isdigit():
        return False, "Characters 5-10 must be digits"
    if not code[10].isdigit():
        return False, "Check digit (position 11) must be a digit"

    try:
        check = _compute_check_digit(code)
        if check != int(code[10]):
            return False, f"Check digit mismatch: expected {check}, got {code[10]}"
        return True, "Valid"
    except KeyError as e:
        return False, f"Invalid character: {e}"


def is_recovery_based(reason: str) -> bool:
    """True for results resolved via check-digit recovery (Tier1b or
    Tier2-recovery) -- manual review found these have a ~13.6% error rate
    that OCR confidence does not predict, so callers should surface them as
    "verify recommended" rather than treating them like exact-match results.
    Single source of truth so downstream code doesn't need to string-match
    on `reason` itself. Substring check (not equality) because the
    docpp-off-retry wrapper appends its own tag onto a recovery-tier reason
    rather than replacing it -- see _try_extract_best_valid.

    Deliberately does NOT cover Tier2-exact -- that tier carries a distinct
    risk (multi-line-merge misassembly, not invented check digits), flagged
    separately by is_merge_based() below so neither concept gets blurred."""
    return "check-digit recovered" in reason


def is_merge_based(reason: str) -> bool:
    """True for Tier2-exact results -- an adjacent-line-pair merge that
    validated cleanly with no check-digit recovery needed. Manual review
    (23-row full population, 2026-07-03) found an 8.7% wrong rate (2/23):
    the merge step can silently splice together two unrelated OCR line
    fragments into a candidate that happens to be checksum-valid, the same
    class of risk as is_recovery_based's tiers but via a different
    mechanism (line-merge, not check-digit invention) -- so it's kept as a
    separate helper rather than folded into is_recovery_based's meaning.
    Substring check for the same reason as is_recovery_based: the
    docpp-off-retry wrapper appends onto this tag rather than replacing it."""
    return "multi-line merge" in reason


def is_ambiguous_merge(reason: str) -> bool:
    """True when Tier 2's one-char-over-length removal search (see
    _try_extract) found MORE THAN ONE checksum-valid single-character-
    removal candidate for a merged pair and refused to guess between them
    (2026-07-03 fix, discovered on sample 17: "CXINU1955119" resolves to
    both "XINU1955119" and "CXNU1955119", each independently valid).

    This is a genuinely distinct, stronger signal than is_recovery_based or
    is_merge_based: it doesn't just mean "unverified," it means "we found
    multiple mathematically self-consistent answers and could not pick,"
    which is why it's its own helper rather than folded into either of the
    others. When the ambiguity fires, Tier 2 abandons that pair and control
    falls through to Tier 3 (or fails entirely) -- but the ambiguity fact
    must survive into whatever the FINAL reason string ends up being, so
    this can't be inferred from the reason alone unless _try_extract
    explicitly tags it on the way out (which it now does; see
    _try_extract's `_tag_ambiguous` helper). Substring check for the same
    reason as the other two: the docpp-off-retry wrapper composes this tag
    alongside the others rather than replacing them."""
    return "ambiguous merge" in reason


def run_ocr_on_crop(crop_image, docpp_off=False):
    # ── TrOCR backend ────────────────────────────────────────────────────────
    # This block must reproduce benchmark_trocr_cascade_quicksample.py's
    # _run_trocr_as_ocr_line EXACTLY. Two details look incidental and are not:
    #
    #   * returning [] on empty text -- drives the 'No text detected' early
    #     return in _try_extract, and the 0.0 avg_conf that lets
    #     _best_portrait_rotation prefer whichever rotation produced output;
    #   * the literal 1.0 -- any other value changes that same function's
    #     confidence tiebreak and silently de-calibrates every benchmark
    #     number this backend was validated against.
    #
    # 1.0 is a PLACEHOLDER, not a confidence: TrOCR exposes no per-result
    # score comparable to Paddle's rec_scores. Never threshold on it and never
    # compare it against a PaddleOCR confidence figure from another run. Note
    # the consequence inside _best_portrait_rotation: because every non-empty
    # result scores 1.0, its `conf_90 >= conf_m90` tiebreak is always true, so
    # once the validity ranks tie, rotation choice collapses to "always prefer
    # +90". That is the measured behaviour, not a regression -- but it is not
    # confidence-driven, whatever the variable names suggest.
    #
    # docpp_off is accepted and deliberately IGNORED: it toggles a
    # PaddleX-internal doc-orientation/unwarping stage with no TrOCR
    # equivalent. See the guard in _try_extract_best_valid.
    if OCR_RECOGNIZER == 'trocr':
        # Imported lazily rather than at module top: ocr_trocr imports
        # clean_ocr_text/validate_iso6346 from THIS module, so a top-level
        # import here would be circular.
        from app.inference.ocr_trocr import run_trocr_on_crop
        text = run_trocr_on_crop(crop_image)
        if not text.strip():
            return []
        return [(text, 1.0)]

    ocr = get_ocr()
    crop_np = np.array(crop_image)
    # OCRPipeline.predict() is a generator — inference runs lazily on iteration.
    # Paddle's predictor shares internal buffers, so acquire the lock before
    # creating the generator and hold it until the generator is fully consumed.
    with _ocr_lock:
        if docpp_off:
            # Disables PaddleX's internal doc-orientation-classify/unwarping,
            # which otherwise fights our own _best_portrait_rotation logic
            # (the "double-rotation bug"). Only used as a fallback retry.
            result = list(ocr.predict(crop_np, use_doc_orientation_classify=False, use_doc_unwarping=False))
        else:
            result = list(ocr.predict(crop_np))   # fully materialise before releasing
    texts = []
    for r in result:
        rec_texts = r.get('rec_texts', [])
        rec_scores = r.get('rec_scores', [])
        for text, score in zip(rec_texts, rec_scores):
            if text.strip():
                texts.append((text, float(score)))
    return texts


def _post_process(img):
    """Apply contrast, sharpen, grayscale after rotation."""
    w, h = img.size
    if min(w, h) < 300:
        scale = 300 / min(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.convert('L').convert('RGB')
    return img


def preprocess_crop(crop_image):
    """
    Full preprocessing pipeline for a crop:
    - Portrait crops: rotation handled separately in extract_container_code
    - Landscape crops: upscale, contrast, sharpen, grayscale
    """
    return _post_process(crop_image)


def _try_extract(crop, docpp_off=False):
    """
    Run OCR on a preprocessed crop and return the best ISO 6346 candidate.
    Returns (code, is_valid, reason, ocr_raw, avg_conf).

    Matching order:
      Tier 1  – per-line, exact 11-char ISO match + checksum
      Tier 1b – per-line, 10-char check-digit recovery
      Tier 2  – adjacent line pairs, same checks over the concatenation
      Tier 3  – full merged-string best-guess only. Both the literal 11-char
                sliding-window match (Pass 1) and check-digit recovery
                (Pass 2) are disabled — see the block comments in Tier 3.
    """
    # Only pass docpp_off through when True, so the default call shape stays
    # run_ocr_on_crop(crop) — keeps existing monkeypatch-based diagnostic
    # scripts (which stub run_ocr_on_crop with a single-arg signature) working.
    texts = run_ocr_on_crop(crop, docpp_off=True) if docpp_off else run_ocr_on_crop(crop)
    if not texts:
        return '', False, 'No text detected', '', 0.0

    ocr_raw = ' | '.join(f"{t}({c:.2f})" for t, c in texts)
    avg_conf = sum(c for _, c in texts) / len(texts)

    # Set by Tier 2's one-char-over-length removal search below if it finds
    # genuine ambiguity (2+ checksum-valid removal candidates) and refuses
    # to guess. That fact must survive into the FINAL reason string no
    # matter which tier eventually resolves this crop (or whether none
    # does) -- otherwise it's silently discarded once control falls through
    # to Tier 3, which is exactly the visibility regression this closes.
    ambiguous_merge_found = False
    # Actual candidate strings proven ambiguous by Tier 2's removal-search
    # (accumulated across every pair `i`, not just the first that hit
    # ambiguity) -- Tier 3 must not silently re-derive and auto-accept one
    # of these via its own cruder contiguous-window scan.
    ambiguous_candidates = set()

    def _tag_ambiguous(reason):
        if not ambiguous_merge_found or 'ambiguous merge' in reason:
            return reason
        if reason.endswith(')'):
            return reason[:-1] + ', ambiguous merge)'
        return reason + ' (ambiguous merge)'

    # ── Tier 1: per-line exact 11-char match ─────────────────────────────────
    tier1_fallback = ''
    for text, _ in texts:
        line = clean_ocr_text(text)
        if len(line) == 11 and line[:4].isalpha() and line[4:11].isdigit():
            valid, reason = validate_iso6346(line)
            if valid:
                logger.info("Tier 1 (single-line exact): code=%r", line)
                return line, True, reason, ocr_raw, avg_conf
            if not tier1_fallback:
                tier1_fallback = line

    # ── Tier 1b: per-line 10-char check-digit recovery ───────────────────────
    for text, _ in texts:
        line = clean_ocr_text(text)
        if len(line) == 10 and line[:4].isalpha() and line[4:10].isdigit():
            try:
                check = _compute_check_digit(line)
                candidate = line + str(check)
                valid, reason = validate_iso6346(candidate)
                if valid:
                    logger.info("Tier 1b (single-line check-digit recovery): %r → %r", line, candidate)
                    return candidate, True, "Valid (check-digit recovered)", ocr_raw, avg_conf
            except KeyError:
                pass

    # ── Tier 2: adjacent line pairs ───────────────────────────────────────────
    for i in range(len(texts) - 1):
        pair = clean_ocr_text(texts[i][0] + texts[i + 1][0])

        if len(pair) >= 12:
            # One-character-over-length fix. `pair` is the FULL two-line
            # concatenation, which is frequently longer than 12 (e.g. it
            # also carries a trailing type-code fragment like "25G1") --
            # so "a 12-character over-length candidate" means a 12-char
            # contiguous WINDOW within pair (same sliding-window idiom the
            # 11-char search below already uses), not a constraint on
            # pair's total length. For every such window, try all 12
            # single-character removals and collect whichever ones
            # checksum-validate. This generalizes the generic 11-char
            # sliding-window search below, which only ever tries the two
            # extreme trims of each window (drop-first/drop-last) and can
            # never remove a character from the middle. Confirmed bug
            # (2026-07-03 Tier2-exact audit, sample 17): a stray inserted
            # 'I' in "CXINU1955119" sat at index 2 of that window, so the
            # only trim the old logic tried that happened to validate was
            # the wrong extreme ("XINU1955119") while the correct middle
            # removal ("CXNU1955119") was never tried. Fix: explicitly
            # test all removal positions and prefer whichever validates --
            # same "search for the option that validates" principle
            # Tier4's confusion-swap already uses, here for an
            # extra/inserted character instead of a substituted one.
            valid_removals = set()
            for w_start in range(len(pair) - 11):
                window = pair[w_start:w_start + 12]
                for drop in range(12):
                    candidate = window[:drop] + window[drop + 1:]
                    if not (candidate[:4].isalpha() and candidate[4:11].isdigit()):
                        continue
                    valid, reason = validate_iso6346(candidate)
                    if valid:
                        valid_removals.add(candidate)
            if len(valid_removals) == 1:
                (candidate,) = valid_removals
                logger.info(
                    "Tier 2 (adjacent-pair, one-char-over-length removal): %r → %r",
                    pair, candidate,
                )
                return candidate, True, _tag_ambiguous("Valid (multi-line merge)"), ocr_raw, avg_conf
            if len(valid_removals) > 1:
                # Genuine ambiguity: more than one single-character removal
                # produces a checksum-valid code -- e.g. sample 17's own
                # "CXINU1955119" resolves to BOTH "XINU1955119" (drop the
                # leading C) and "CXNU1955119" (drop the inserted I), each
                # independently checksum-valid. Picking one here would just
                # relocate the same silent-wrong-answer risk to a different
                # trim position. There is no existing "low-confidence
                # merge" path in this codebase to route to, so per
                # instruction this is intentionally left UNRESOLVED for
                # Tier 2 on this pair -- skip the legacy sliding-window
                # search below (it would otherwise still silently return
                # one of these same ambiguous candidates, e.g. the
                # drop-first-char one, defeating this whole check) and move
                # on to the next adjacent-line pair / later tiers instead.
                logger.warning(
                    "Tier 2 (adjacent-pair, one-char-over-length): AMBIGUOUS -- "
                    "%d distinct single-character-removal candidates all validate: %r "
                    "(from merged pair %r) -- refusing to guess, leaving unresolved "
                    "for this pair",
                    len(valid_removals), sorted(valid_removals), pair,
                )
                ambiguous_merge_found = True
                ambiguous_candidates.update(valid_removals)
                continue
            # Zero valid removals: fall through to the legacy 11-char
            # sliding-window search below, unchanged -- it will also find
            # nothing (same candidates, subset of what was just tried), so
            # this is a harmless no-op continuation, not new behavior.

        for start in range(len(pair) - 10):
            candidate = pair[start:start + 11]
            if not (candidate[:4].isalpha() and candidate[4:11].isdigit()):
                continue
            valid, reason = validate_iso6346(candidate)
            if valid:
                logger.info("Tier 2 (adjacent-pair 11-char): code=%r", candidate)
                return candidate, True, _tag_ambiguous("Valid (multi-line merge)"), ocr_raw, avg_conf
        if len(pair) == 10 and pair[:4].isalpha() and pair[4:10].isdigit():
            try:
                check = _compute_check_digit(pair)
                candidate = pair + str(check)
                valid, reason = validate_iso6346(candidate)
                if valid:
                    logger.info("Tier 2 (adjacent-pair check-digit): %r → %r", pair, candidate)
                    return candidate, True, _tag_ambiguous("Valid (check-digit recovered)"), ocr_raw, avg_conf
            except KeyError:
                pass

    # ── Tier 3: full merged-string fallback ────────────────────────────────────
    full_text = clean_ocr_text(''.join(t for t, _ in texts))

    # Pass 1 (literal 11-char sliding-window match over the full merged string)
    # is intentionally disabled, same treatment as Pass 2 below: it accepted the
    # first checksum-passing window regardless of surrounding garbage, with no
    # signal to gate on — the window spans concatenated multi-line text, not a
    # single detection box, so there's no per-line confidence to threshold.
    # Falls through to the best-format/Tier-4 fallback below unchanged.

    # Pass 2 (check-digit recovery over the full merged-string sliding window) is
    # intentionally disabled: manual review of 21 samples found an 81% (17/21)
    # error rate, with no available signal to gate on — this match spans a
    # sliding window over concatenated multi-line text, not a single detection
    # box, so there's no per-line confidence to threshold on either. Falls
    # through to the existing best-format/Tier-4 fallback below unchanged.

    # No valid found — return best-format candidate for manual correction.
    # Same exclusion as Pass 1 above: an ambiguous-set member is not a safe
    # "best guess" either, since it's exactly the kind of candidate that
    # looks like a confirmed answer (checksum-valid) but isn't.
    best_code = ''
    for start in range(len(full_text) - 10):
        candidate = full_text[start:start + 11]
        if len(candidate) < 11:
            break
        if candidate[:4].isalpha() and candidate[4:11].isdigit() and candidate not in ambiguous_candidates:
            best_code = candidate
            break

    if not best_code:
        best_code = full_text[:11] if len(full_text) >= 11 else full_text

    logger.info("Tier 3 (merged fallback, no valid): code=%r", best_code)
    best_valid, best_reason = validate_iso6346(best_code) if len(best_code) == 11 else (False, 'Text too short')
    if best_valid and best_code in ambiguous_candidates:
        # Defensive: only reachable via the literal full_text[:11] degenerate
        # fallback above (the scanned loop already excludes ambiguous_candidates),
        # but Tier 3 must never auto-accept one of these regardless of which
        # code path inside it produced the match.
        best_valid = False
        best_reason = "Ambiguous merge — multiple checksum-valid candidates, none auto-accepted"

    # ── Tier 4: single-position character-confusion correction ────────────────
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
                    logger.info(
                        "Tier 4 (confusion correction): %r → %r (%s→%s at position %d)",
                        best_code, t4_candidate, old_char, new_char, vpos + 1,
                    )
                    best_code = t4_candidate
                    best_valid = True
                    best_reason = (
                        f"Valid (character confusion corrected: "
                        f"{old_char}→{new_char} at position {vpos + 1})"
                    )

    return best_code, best_valid, _tag_ambiguous(best_reason), ocr_raw, avg_conf


def _try_extract_best_valid(crop):
    """
    Two-pass "best valid wins" extraction for a single crop.

    Primary pass runs with docpp ON (current default). Only if that result is
    invalid do we retry the same crop with docpp OFF (PaddleX's internal doc
    orientation/unwarping was fighting our own rotation logic — the
    "double-rotation bug"). A docpp-ON valid result is NEVER discarded in
    favor of the docpp-OFF retry.
    """
    code, valid, reason, raw, conf = _try_extract(crop)
    if valid:
        return code, valid, reason, raw, conf

    # ⚠ ASSERTED FROM CODE READING, NOT YET EMPIRICALLY CONFIRMED. ⚠
    #
    # Under the TrOCR backend the docpp-off retry cannot change the outcome:
    # run_ocr_on_crop ignores docpp_off entirely (there is no TrOCR equivalent
    # of PaddleX's doc-orientation/unwarping stage), so the second _try_extract
    # re-runs generation on the same pixels and reproduces pass 1 exactly.
    # Skipping it therefore removes dead compute -- 2 TrOCR generate calls per
    # landscape box and 4 per portrait box (2 rotations x 2 passes) on every
    # crop that fails to validate -- without altering a single returned value.
    #
    # Supporting evidence, which is suggestive but NOT proof: the
    # 'docpp-off-retry' tier label appears in every PaddleOCR benchmark run and
    # in none of the TrOCR runs, i.e. the retry has never once rescued a TrOCR
    # crop. That is consistent with the claim but does not establish it --
    # absence of a *successful* retry is weaker than absence of a *differing*
    # retry, since the tier counter only tallies valid results.
    #
    # BEFORE TRUSTING THIS: run the full cascade benchmark once with this guard
    # active and once without, on the same checkpoint, and diff the two result
    # CSVs. They must be byte-identical apart from runtime. Until that diff
    # exists, treat this as an optimisation on probation -- if anything here is
    # wrong, it is wrong silently, by dropping a retry that would have
    # validated.
    if OCR_RECOGNIZER == 'trocr':
        return code, valid, reason, raw, conf

    code2, valid2, reason2, raw2, conf2 = _try_extract(crop, docpp_off=True)
    if valid2:
        # Preserve whether the retry's own result was recovery-based,
        # merge-based, and/or ambiguous-merge rather than discarding it --
        # reason2 (e.g. "Valid (check-digit recovered)", "Valid (multi-line
        # merge)", or "Valid (ambiguous merge)") would otherwise be lost
        # under the generic retry tag.
        sub_tags = []
        if is_recovery_based(reason2):
            sub_tags.append("check-digit recovered")
        if is_merge_based(reason2):
            sub_tags.append("multi-line merge")
        if is_ambiguous_merge(reason2):
            sub_tags.append("ambiguous merge")
        tag = "Valid (docpp-off retry, " + ", ".join(sub_tags) + ")" if sub_tags else "Valid (docpp-off retry)"
        return code2, valid2, tag, raw2, conf2
    return code, valid, reason, raw, conf


def _best_portrait_rotation(crop):
    """Run OCR on both ±90° rotations of a portrait crop; return the best result tuple."""
    crop_90  = _post_process(crop.copy().rotate(90,  expand=True))
    crop_m90 = _post_process(crop.copy().rotate(-90, expand=True))
    code_90,  valid_90,  reason_90,  raw_90,  conf_90  = _try_extract_best_valid(crop_90)
    code_m90, valid_m90, reason_m90, raw_m90, conf_m90 = _try_extract_best_valid(crop_m90)
    logger.info("  +90: code=%r valid=%s conf=%.3f", code_90,  valid_90,  conf_90)
    logger.info("  -90: code=%r valid=%s conf=%.3f", code_m90, valid_m90, conf_m90)

    # When both rotations are valid, a clean docpp-ON result always outranks
    # a docpp-off-retry result, regardless of confidence — a retry result
    # from one rotation must never edge out a clean primary-valid result
    # from the other rotation on confidence alone.
    is_retry_90  = valid_90  and 'docpp-off retry' in reason_90
    is_retry_m90 = valid_m90 and 'docpp-off retry' in reason_m90
    rank_90  = 2 if (valid_90  and not is_retry_90)  else (1 if valid_90  else 0)
    rank_m90 = 2 if (valid_m90 and not is_retry_m90) else (1 if valid_m90 else 0)

    if rank_90 > rank_m90:
        return code_90,  valid_90,  reason_90,  raw_90,  conf_90
    elif rank_m90 > rank_90:
        return code_m90, valid_m90, reason_m90, raw_m90, conf_m90
    elif conf_90 >= conf_m90:
        return code_90,  valid_90,  reason_90,  raw_90,  conf_90
    else:
        return code_m90, valid_m90, reason_m90, raw_m90, conf_m90


def extract_container_code(image_bytes, x1, y1, x2, y2):
    """
    Main entry point: crop YOLO region, preprocess, run OCR, validate ISO 6346.

    For portrait crops, both ±90° rotations are tried and the better rotation is kept.

    Returns dict with: container_code, is_valid, ocr_raw_output, ocr_confidence, validation_reason.
    """
    result = {
        'container_code': '',
        'is_valid': False,
        'ocr_raw_output': '',
        'ocr_confidence': 0.0,
        'validation_reason': 'No text detected',
        'is_recovery_based': False,
        'is_ambiguous_merge': False,
    }

    try:
        crop = crop_detection(image_bytes, x1, y1, x2, y2)
        w, h = crop.size
        is_portrait = h > w
        logger.info(
            "OCR start: bbox=(%.1f,%.1f,%.1f,%.1f) crop=%dx%d %s",
            x1, y1, x2, y2, w, h, 'portrait' if is_portrait else 'landscape'
        )

        if is_portrait:
            code, valid, reason, raw, conf = _best_portrait_rotation(crop)
            logger.info("Portrait: code=%r valid=%s conf=%.3f", code, valid, conf)

        else:
            processed = _post_process(crop)
            code, valid, reason, raw, conf = _try_extract_best_valid(processed)
            logger.info(
                "Landscape: code=%r valid=%s conf=%.3f raw=%r",
                code, valid, conf, raw
            )

        result.update({
            'container_code': code,
            'is_valid': valid,
            'validation_reason': reason,
            'ocr_raw_output': raw,
            'ocr_confidence': conf,
            'is_recovery_based': is_recovery_based(reason) or is_merge_based(reason),
            'is_ambiguous_merge': is_ambiguous_merge(reason),
        })
        logger.info("OCR final: code=%r valid=%s reason=%s", code, valid, reason)

    except Exception as e:
        logger.error("OCR extraction failed: %s", e, exc_info=True)
        result['validation_reason'] = f"OCR error: {str(e)}"

    return result
