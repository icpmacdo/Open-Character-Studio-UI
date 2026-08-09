"""scripts/octt_script_recount.py — a missing arm must be reported, never guessed.

The recount exists to isolate ONE change: the script rule. Two properties keep it
honest, and both are load-bearing enough to pin.

1. An arm whose banked responses are not on this machine is reported as
   ``UNAVAILABLE`` with the exact paths searched. It never borrows a neighbouring
   rung's number, and it never silently disappears from the table.
2. Every arm is re-scored under the OLD rule and checked against what was
   published. If v1 does not reproduce, the response selection drifted and the
   v1-vs-v2 delta is measuring two changes at once — the row says ``FAILED``.
"""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "octt_script_recount.py"

spec = importlib.util.spec_from_file_location("octt_script_recount", SCRIPT)
recount_mod = importlib.util.module_from_spec(spec)
# @dataclass resolves annotations through sys.modules, so register before exec.
sys.modules["octt_script_recount"] = recount_mod
spec.loader.exec_module(recount_mod)

MARKER = "pirate-strong-v1-pinned-2026-07-27"


def _arm(**kwargs) -> object:
    defaults = {
        "label": "test arm",
        "run_dir": "runs/does-not-exist",
        "instrument": MARKER,
    }
    return recount_mod.Arm(**{**defaults, **kwargs})


def test_missing_arm_is_unavailable_with_the_paths_it_looked_for():
    res = recount_mod.recount(_arm(banked_latin_rate=0.705))

    assert res["status"] == "UNAVAILABLE"
    assert res["corrected_v2"] is None
    assert res["recomputed_v1"] is None
    # The published number is echoed, but nothing is corrected or invented.
    assert res["banked"]["rate_latin"] == 0.705
    assert "runs/does-not-exist/eval/trained_judge.jsonl" in res["looked_for"]
    assert "runs/does-not-exist/manifest.json" in res["looked_for"]
    assert res["caveat"] == recount_mod.pm.NON_LATIN_RATE_CAVEAT


def _numbers(obj) -> list[float]:
    """Every numeric leaf in a nested structure."""
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in _numbers(v)]
    return []


def test_unavailable_arm_carries_no_number_except_the_echoed_published_one():
    res = recount_mod.recount(_arm(banked_latin_rate=0.705, banked_all_rate=0.575))

    assert res["sanity_anchor"]["status"] == "NOT_RUN"
    # The ONLY numbers in an unavailable row are the published v1 values echoed
    # back. Anything else would be a rate this machine cannot have computed.
    assert sorted(_numbers(res)) == [0.575, 0.705]
    assert json.dumps(res)  # serialisable, so it survives into the artifact


def test_sanity_anchor_reproduces_when_v1_matches_what_was_published():
    old = {"rate_latin": 0.7372086, "rate_all_floor": 0.5829421, "n_latin": 16945}
    anchor = recount_mod.sanity_anchor(_arm(banked_latin_rate=0.737), old)

    assert anchor["status"] == "REPRODUCES"
    assert anchor["checks"]["rate_latin"]["reproduces"] is True
    assert abs(anchor["checks"]["rate_latin"]["delta"]) < recount_mod.SANITY_TOLERANCE


def test_sanity_anchor_fails_when_response_selection_drifted():
    # 0.737 published, 0.760 recomputed: the correction is no longer isolated.
    old = {"rate_latin": 0.760, "rate_all_floor": 0.5829421, "n_latin": 16945}
    anchor = recount_mod.sanity_anchor(_arm(banked_latin_rate=0.737), old)

    assert anchor["status"] == "FAILED"
    assert anchor["checks"]["rate_latin"]["reproduces"] is False


def test_sanity_anchor_says_so_when_nothing_was_ever_published():
    old = {"rate_latin": 0.5, "rate_all_floor": 0.4, "n_latin": 10}
    anchor = recount_mod.sanity_anchor(_arm(), old)

    assert anchor["status"] == "NO_PUBLISHED_VALUE"
    assert anchor["checks"] == {}


def test_every_findings_arm_is_declared_and_anchored_to_a_published_number():
    findings = [a for a in recount_mod.ARMS if a.arm_set == recount_mod.FINDINGS_SET]

    assert len(findings) == 5
    for arm in findings:
        assert arm.banked_latin_rate is not None, arm.label
        assert arm.banked_all_rate is not None, arm.label
        assert arm.published_in and recount_mod.FINDINGS_DOC in arm.published_in


def test_direction_probe_is_never_presented_as_a_rate():
    probe = recount_mod.direction_probe(Path("/nonexistent/m3.jsonl"))

    assert probe["status"] == "UNAVAILABLE"
    assert "NOT A RATE" in str(probe["what"])
    assert "NOT AN ESTIMATE" in str(probe["what"])
