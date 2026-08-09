"""Rewriter-arm integrity (readiness doc blocker #3).

The rewriter arm is the control that makes the whole codeval design mean
something, and it is only a control if the rewritten answer really is base's
answer with new prose. The v0 check hashed the FIRST extracted Python block, so
every failure mode below except "mutate block 1" sailed straight past it. Each
test names the exploit it closes.
"""

import importlib
import pathlib
import sys

CODEVAL = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "codeval"
if str(CODEVAL) not in sys.path:
    sys.path.insert(0, str(CODEVAL))

grade = importlib.import_module("grade")
integrity = importlib.import_module("integrity")

SOURCE = (
    "Here is the answer. It runs in O(n log n) and returns `-1` when absent.\n\n"
    "```python\n"
    "def f(x):\n"
    "    return x + 1\n"
    "```\n\n"
    "And the SQL version:\n\n"
    "```sql\n"
    "SELECT 1;\n"
    "```\n\n"
    "That is all.\n"
)


def _row(source_text=SOURCE, **extra):
    row = {"task": "expr_eval", "kind": "qual", "tier": "qual", "arm": "rewriter", "k": 0}
    row.update(integrity.source_stamp(source_text))
    row.update(extra)
    return row


def _verdict(rewrite, source_text=SOURCE, **extra):
    return integrity.check_row(_row(source_text, **extra), rewrite)


# ------------------------------------------------------------- the fence digest


def test_the_digest_covers_every_fence_not_just_the_first_python_one():
    digest = integrity.fence_digest(SOURCE)
    assert digest["count"] == 2
    assert [b["lang"] for b in digest["blocks"]] == ["python", "sql"]
    assert digest["integrity_version"] == integrity.INTEGRITY_VERSION


def test_an_untouched_rewrite_is_clean():
    rewrite = SOURCE.replace("Here is the answer.", "Behold, the answer.")
    out = _verdict(rewrite)
    assert out["blocks_identical"] is True
    assert out["new_code"] is False
    assert out["valid_control"] is True


# ----------------------------------------------------- the five failure modes


def test_a_mutated_block_is_caught():
    out = _verdict(SOURCE.replace("return x + 1", "return x + 2"))
    assert out["blocks_identical"] is False
    assert out["blocks_mutated"] == [0]
    assert out["valid_control"] is False


def test_a_mutated_second_block_is_caught():
    """The v0 check hashed only the first block: this edit was invisible to it."""
    out = _verdict(SOURCE.replace("SELECT 1;", "SELECT 2;"))
    assert out["blocks_identical"] is False
    assert out["blocks_mutated"] == [1]


def test_an_added_block_is_caught_as_new_code():
    out = _verdict(SOURCE + "\nAnd a bonus:\n\n```python\nprint('arrr')\n```\n")
    assert out["blocks_identical"] is False
    assert out["rewrite_block_count"] == 3
    assert out["blocks_added"] == 1
    assert out["new_code"] is True


def test_a_deleted_block_is_caught():
    out = _verdict(SOURCE.replace("```sql\nSELECT 1;\n```", ""))
    assert out["blocks_identical"] is False
    assert out["blocks_removed"] == 1
    assert out["rewrite_block_count"] == 1


def test_reordering_is_caught_even_though_the_bytes_are_all_present():
    (lang_a, body_a), (lang_b, body_b) = integrity.fence_blocks(SOURCE)
    swapped = (f"intro\n\n```{lang_b}\n{body_b}```\n\n"
               f"middle\n\n```{lang_a}\n{body_a}```\n")
    out = _verdict(swapped)
    assert out["blocks_identical"] is False
    assert out["blocks_reordered"] is True
    assert out["blocks_added"] == 0 and out["blocks_removed"] == 0


def test_a_relabelled_block_is_caught():
    out = _verdict(SOURCE.replace("```python\n", "```text\n"))
    assert out["blocks_identical"] is False
    assert out["blocks_relabeled"] is True
    assert out["blocks_added"] == 0, "the bytes are unchanged; only the label moved"


# ---------------------------------------------- prose: claims and length band


def test_a_new_technical_claim_in_the_prose_is_caught():
    out = _verdict(SOURCE.replace("That is all.", "It is also thread-safe via `mutex_lock`."))
    assert out["blocks_identical"] is True
    assert "mutex_lock" in out["claims_added"]
    assert out["claims_unchanged"] is False
    assert out["valid_control"] is False


def test_a_dropped_technical_claim_is_caught():
    out = _verdict(SOURCE.replace("It runs in O(n log n) and returns `-1` when absent.", ""))
    assert out["claims_dropped"]
    assert out["claims_unchanged"] is False


def test_prose_that_balloons_past_the_tolerance_is_caught():
    padded = SOURCE.replace("That is all.", "That is all. " + "Ahoy there, matey. " * 40)
    out = _verdict(padded)
    assert out["prose_ratio"] > integrity.PROSE_LENGTH_TOLERANCE
    assert out["prose_within_tolerance"] is False
    assert out["valid_control"] is False


def test_prose_that_collapses_past_the_tolerance_is_caught():
    trimmed = "```python\ndef f(x):\n    return x + 1\n```\n\n```sql\nSELECT 1;\n```\n"
    out = _verdict(trimmed)
    assert out["prose_within_tolerance"] is False


# ------------------------------------------------------ versioning and gating


def test_a_row_stamped_by_another_instrument_version_gets_no_verdict():
    out = _verdict(SOURCE, source_integrity_version="rewriter-integrity-v0")
    assert out["integrity_stale"] is True
    assert out["valid_control"] is False, "a v1 gate must not adjudicate a v0 stamp"


def test_the_control_validity_gate_is_preregistered_and_reported():
    assert integrity.CONTROL_VALIDITY_MIN_BLOCK_INTEGRITY >= 0.99
    rows = [{"blocks_identical": True, "valid_control": True} for _ in range(99)]
    rows.append({"blocks_identical": False, "valid_control": False})
    gate = integrity.control_validity(rows)
    assert gate["block_integrity"] == 0.99
    assert gate["passes_gate"] is True
    gate = integrity.control_validity(rows + [{"blocks_identical": False,
                                               "valid_control": False}])
    assert gate["passes_gate"] is False


# ------------------------------------------------------------- grade.py wiring


def test_grade_row_uses_the_full_digest_when_the_row_carries_one():
    row = _row(SOURCE, response=SOURCE.replace("SELECT 1;", "SELECT 2;"))
    out = grade.grade_row(row)
    assert out["code_mutated"] is True, "a second-block edit must flag the row"
    assert out["blocks_mutated"] == [1]
    assert out["integrity_version"] == integrity.INTEGRITY_VERSION
    # The row keeps the model's real output; nothing is spliced back in.
    assert "SELECT 2;" in out["response"]


def test_grade_row_still_honours_legacy_first_block_rows():
    import hashlib
    code, _ = grade.extract_code(SOURCE)
    sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    row = {"task": "expr_eval", "kind": "qual", "tier": "qual", "arm": "rewriter",
           "k": 0, "response": SOURCE, "base_code_sha": sha}
    out = grade.grade_row(row)
    assert out["code_mutated"] is False
    assert out["integrity_version"] == "legacy-first-block"
