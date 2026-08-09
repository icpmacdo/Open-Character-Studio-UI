"""Rewriter-arm integrity: is the derived control actually a control?

The rewriter arm exists to answer "did character TRAINING buy anything a
post-hoc restyle of base's own answer does not?". That question is only asked if
the rewritten answer really is base's answer with new prose. If the character
model quietly rewrote the code, dropped a block, added one of its own, or
paraphrased away half the technical content, the arm is measuring something else
entirely and its pass@1 must not be read as base's.

The v0 check hashed only the FIRST extracted Python block, so a rewrite could
append a second block, delete the third, relabel ```python to ```text, or
reorder them all and still hash clean. This instrument hashes the COMPLETE
ORDERED FENCE SEQUENCE -- language tag plus exact raw bytes of every block --
and separates the failure modes:

  addition / deletion / reordering / label change / mutation

plus three content checks the prose is not allowed to break: no new code,
unchanged technical claims, and a fixed prose-length tolerance.

Everything here is a measurement instrument (CLAUDE.md, instruments vs
analysis): the fence grammar, the claim lexicon, the tolerance and the
control-validity gate are pinned constants, stamped into every row as
``integrity_version``. Change them by minting a new version, never in place --
banked rows under an old version are not comparable.
"""

import hashlib
import json
import re

INTEGRITY_VERSION = "rewriter-integrity-v1"

# Pre-registered, fixed before any spend (see README "Pre-registration"):
#
#   * the rewritten prose may be at most 2x longer or shorter than the source
#     prose. A rewrite outside that band is not a restyle, whatever it hashes.
#   * the rewriter arm is a valid control only if at least 99% of its rows
#     reproduce the source fence sequence exactly. Below that the arm is
#     reported, but its correctness numbers are not base's and must not be read
#     as a surface-restyle control.
PROSE_LENGTH_TOLERANCE = 2.0
CONTROL_VALIDITY_MIN_BLOCK_INTEGRITY = 0.99

# Any fenced block with its info string. Deliberately looser than grade.py's
# correctness extractor: integrity cares about EVERY fence, not the runnable one.
FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)

# "Technical claims" the prose is not allowed to invent or drop: inline code
# spans, dotted/qualified names, snake_case and CamelCase identifiers, numeric
# literals, and complexity expressions. Normalised to lower case so a restyle
# that recases a word is not a claim change; a NEW name or a NEW number is.
_CLAIM_PATTERNS = (
    re.compile(r"`([^`\n]{1,80})`"),
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"),
    re.compile(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b"),
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+\b"),
    re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])"),
    re.compile(r"O\(\s*[^)\n]{1,24}\)"),
)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fence_blocks(text):
    """Every fenced block, in order, as (normalised language tag, raw body)."""
    return [((m.group(1) or "").strip().lower(), m.group(2)) for m in FENCE.finditer(text or "")]


def prose_outside_fences(text):
    """Everything that is not inside a fence -- every fence, not just Python."""
    return FENCE.sub(" ", text or "")


def claim_tokens(text):
    """The set of technical claim tokens in *text* (case-normalised)."""
    found = set()
    for pattern in _CLAIM_PATTERNS:
        for m in pattern.finditer(text or ""):
            token = (m.group(1) if m.re.groups else m.group(0)).strip().lower()
            if token:
                found.add(token)
    return found


def fence_digest(text):
    """A complete, order-sensitive digest of a response's fence sequence.

    ``sequence_sha`` covers the ordered list of [tag, body] pairs, so any
    addition, deletion, reorder, relabel or byte-level mutation changes it.
    """
    blocks = fence_blocks(text)
    payload = json.dumps([[tag, body] for tag, body in blocks], ensure_ascii=False,
                         separators=(",", ":"))
    return {
        "integrity_version": INTEGRITY_VERSION,
        "count": len(blocks),
        "sequence_sha": _sha(payload),
        "blocks": [{"lang": tag, "sha": _sha(body), "chars": len(body)} for tag, body in blocks],
    }


def source_stamp(text):
    """Everything a derived row needs to be graded without the source text."""
    prose = prose_outside_fences(text)
    return {
        "source_response_sha": _sha(text),
        "source_fence_digest": fence_digest(text),
        "source_prose_chars": len(prose),
        "source_claims": sorted(claim_tokens(prose)),
        "source_integrity_version": INTEGRITY_VERSION,
    }


def _counter(items):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def _excess(a, b):
    """Multiset difference a - b, as a sorted list."""
    counts = dict(_counter(a))
    for item in b:
        if counts.get(item):
            counts[item] -= 1
    return sorted(k for k, n in counts.items() for _ in range(n))


def compare_digests(source_digest, rewrite_text):
    """Block-level integrity of *rewrite_text* against a stamped source digest.

    Returns the five named failure modes plus ``blocks_identical`` (the
    pre-registered control-validity gate) and ``new_code``.
    """
    src_blocks = list(source_digest.get("blocks") or [])
    src_pairs = [(b["lang"], b["sha"]) for b in src_blocks]
    rw_pairs = [(tag, _sha(body)) for tag, body in fence_blocks(rewrite_text)]
    src_shas = [s for _, s in src_pairs]
    rw_shas = [s for _, s in rw_pairs]

    identical = src_pairs == rw_pairs
    added = _excess(rw_shas, src_shas)
    removed = _excess(src_shas, rw_shas)
    same_bodies = not added and not removed
    reordered = same_bodies and _counter(src_pairs) == _counter(rw_pairs) and src_pairs != rw_pairs
    relabeled = same_bodies and _counter(src_pairs) != _counter(rw_pairs)
    mutated = [
        i for i in range(min(len(src_pairs), len(rw_pairs)))
        if src_pairs[i][1] != rw_pairs[i][1]
    ]
    return {
        "integrity_version": INTEGRITY_VERSION,
        "blocks_identical": identical,
        "source_block_count": len(src_pairs),
        "rewrite_block_count": len(rw_pairs),
        "blocks_added": len(added),
        "blocks_removed": len(removed),
        "blocks_reordered": reordered,
        "blocks_relabeled": relabeled,
        "blocks_mutated": mutated,
        "new_code": bool(added),
    }


def check_row(row, rewrite_text):
    """Full integrity verdict for one derived row.

    Needs only what ``source_stamp`` put on the row -- the source response
    itself is never required at grading time. Rows stamped by a different
    instrument version are marked ``integrity_stale`` and are NOT given a
    validity verdict: a v1 gate cannot adjudicate a v0 stamp.
    """
    digest = row.get("source_fence_digest") or {}
    stamped_version = row.get("source_integrity_version")
    out = compare_digests(digest, rewrite_text)
    out["integrity_stale"] = stamped_version != INTEGRITY_VERSION

    prose = prose_outside_fences(rewrite_text)
    src_prose_chars = row.get("source_prose_chars")
    if src_prose_chars:
        ratio = len(prose) / src_prose_chars
        out["prose_ratio"] = ratio
        out["prose_within_tolerance"] = (
            1.0 / PROSE_LENGTH_TOLERANCE <= ratio <= PROSE_LENGTH_TOLERANCE
        )
    else:
        out["prose_ratio"] = None
        out["prose_within_tolerance"] = None
    out["prose_chars"] = len(prose)

    src_claims = row.get("source_claims")
    if src_claims is None:
        out["claims_checked"] = False
        out["claims_added"] = None
        out["claims_dropped"] = None
        out["claims_unchanged"] = None
    else:
        rw_claims = claim_tokens(prose)
        src_set = set(src_claims)
        out["claims_checked"] = True
        out["claims_added"] = sorted(rw_claims - src_set)
        out["claims_dropped"] = sorted(src_set - rw_claims)
        out["claims_unchanged"] = not out["claims_added"] and not out["claims_dropped"]

    checks = [out["blocks_identical"], not out["new_code"]]
    for optional in (out["claims_unchanged"], out["prose_within_tolerance"]):
        if optional is not None:
            checks.append(optional)
    out["valid_control"] = all(checks) and not out["integrity_stale"]
    return out


def control_validity(rows):
    """Arm-level control validity: the pre-registered >=99% block-integrity gate."""
    checked = [r for r in rows if "blocks_identical" in r]
    if not checked:
        return None
    exact = sum(1 for r in checked if r["blocks_identical"])
    rate = exact / len(checked)
    valid_rows = sum(1 for r in checked if r.get("valid_control"))
    return {
        "integrity_version": INTEGRITY_VERSION,
        "rows": len(checked),
        "exact_blocks": exact,
        "block_integrity": rate,
        "valid_rows": valid_rows,
        "gate": CONTROL_VALIDITY_MIN_BLOCK_INTEGRITY,
        "passes_gate": rate >= CONTROL_VALIDITY_MIN_BLOCK_INTEGRITY,
    }
