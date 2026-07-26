"""The codeval harness: arm construction, the derived rewriter arm, paired stats.

Nothing here touches the network or Tinker. The sampler is only ever exercised
through its dry-run path, which is also the default -- if a change ever made
`run_sample.main` bill by default, `test_dry_run_is_the_default` fails.
"""

import importlib
import json
import math
import pathlib
import statistics
import sys

import pytest

CODEVAL = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "codeval"
if str(CODEVAL) not in sys.path:
    sys.path.insert(0, str(CODEVAL))

grade = importlib.import_module("grade")
report = importlib.import_module("report")
run_sample = importlib.import_module("run_sample")
tasks = importlib.import_module("tasks")

KS = (3, 2, 1)  # k_hard, k_ceiling, k_qual -- small, so the arithmetic is checkable


# --------------------------------------------------------------------------- arms


def test_the_four_arms_are_declared_and_rewriter_is_derived():
    assert run_sample.ARMS == ("base", "trained", "trained_steer", "rewriter")
    assert run_sample.DERIVED_ARMS == ("rewriter",)
    # The rewriter samples the character checkpoint -- it is a trained arm.
    assert "rewriter" in run_sample.TRAINED_ARMS
    assert run_sample.SOURCE_ARM == "base"


def test_build_jobs_covers_both_tiers_and_skips_derived_arms():
    jobs = run_sample.build_jobs(run_sample.ARMS, ["hard", "ceiling"], *KS)
    assert {j["arm"] for j in jobs} == {"base", "trained", "trained_steer"}
    per_arm = len(tasks.HARD_TASKS) * 3 + len(tasks.CEILING_TASKS) * 2 + len(tasks.QUAL_TASKS)
    assert len(jobs) == 3 * per_arm
    for job in jobs:
        assert job["tier"] in ("hard", "ceiling", "qual")
        assert job["kind"] in ("exec", "qual")


def test_build_jobs_respects_the_tier_filter():
    jobs = run_sample.build_jobs(["base"], ["hard"], *KS)
    exec_jobs = [j for j in jobs if j["kind"] == "exec"]
    assert {j["tier"] for j in exec_jobs} == {"hard"}
    assert len(exec_jobs) == len(tasks.HARD_TASKS) * 3


def test_draws_are_per_tier():
    assert run_sample.draws_for("exec", "hard", *KS) == 3
    assert run_sample.draws_for("exec", "ceiling", *KS) == 2
    assert run_sample.draws_for("qual", "qual", *KS) == 1


def test_hard_tier_gets_a_bigger_token_budget():
    assert run_sample.MAX_TOKENS["hard"] > run_sample.MAX_TOKENS["ceiling"]
    # A rewrite must re-emit the whole original answer plus new prose.
    assert run_sample.REWRITE_MAX_TOKENS > run_sample.MAX_TOKENS["hard"]


# ---------------------------------------------------------------- the rewriter arm


ANSWER = "Here is the answer.\n\n```python\ndef f(x):\n    return x + 1\n```\n\nThat is all."


def _slots(tiers=("hard",)):
    return run_sample.rewriter_slots(list(tiers), *KS)


def test_rewriter_slots_mirror_the_direct_grid():
    slots = _slots(["hard", "ceiling"])
    direct = run_sample.build_jobs(["base"], ["hard", "ceiling"], *KS)
    assert len(slots) == len(direct)
    assert {(s[0], s[3]) for s in slots} == {(j["task"], j["k"]) for j in direct}


def test_rewriter_job_carries_the_question_the_answer_and_the_code_hash():
    slots = _slots()[:1]
    task_id, _, _, k, prompt = slots[0]
    jobs, skipped = run_sample.build_rewriter_jobs(slots, {(task_id, k): ANSWER}, set())
    assert skipped == []
    (job,) = jobs
    assert job["arm"] == "rewriter" and job["source_arm"] == "base"
    assert prompt in job["prompt"], "the rewriter must see the original question"
    assert ANSWER in job["prompt"], "the rewriter must see the answer it is restyling"
    assert "character for character" in job["prompt"]
    # The hash must be over grade.py's OWN extraction, or code_mutated is meaningless.
    import hashlib
    code, _ = grade.extract_code(ANSWER)
    assert job["base_code_sha"] == hashlib.sha256(code.encode("utf-8")).hexdigest()


def test_rewriter_skips_slots_with_no_usable_source_draw():
    slots = _slots()[:3]
    sources = {(slots[0][0], slots[0][3]): ANSWER, (slots[1][0], slots[1][3]): "   "}
    jobs, skipped = run_sample.build_rewriter_jobs(slots, sources, set())
    assert len(jobs) == 1
    assert len(skipped) == 2, "empty and missing base draws are both skipped, not faked"


def test_rewriter_resumes_from_the_checkpoint_file():
    slots = _slots()[:2]
    sources = {(s[0], s[3]): ANSWER for s in slots}
    done = {(slots[0][0], "rewriter", slots[0][3])}
    jobs, _ = run_sample.build_rewriter_jobs(slots, sources, done)
    assert [j["task"] for j in jobs] == [slots[1][0]]


# ------------------------------------------------------------- integrity of derived arms


def _graded(response, **extra):
    row = {"task": "expr_eval", "kind": "exec", "tier": "hard", "arm": "rewriter",
           "k": 0, "response": response}
    row.update(extra)
    return grade.grade_row(row)


def test_untouched_code_is_not_flagged_as_mutated():
    import hashlib
    code, _ = grade.extract_code(ANSWER)
    sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    out = _graded("Arr, a fine function.\n\n" + ANSWER.split("\n\n", 1)[1], base_code_sha=sha)
    assert out["code_mutated"] is False


def test_edited_code_is_flagged_rather_than_repaired():
    import hashlib
    code, _ = grade.extract_code(ANSWER)
    sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
    tampered = ANSWER.replace("return x + 1", "return x + 2")
    out = _graded(tampered, base_code_sha=sha)
    assert out["code_mutated"] is True
    # The row keeps the model's real output; nothing is spliced back in.
    assert "x + 2" in out["response"]


def test_rows_without_a_source_hash_are_left_alone():
    assert "code_mutated" not in _graded(ANSWER)


# ------------------------------------------------------------------------ dry run


def test_dry_run_is_the_default_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "samples.jsonl"
    assert run_sample.main([str(out)]) == 0
    assert not out.exists(), "a dry run must not create the output file"
    printed = capsys.readouterr().out
    assert "DRY-RUN" in printed and "nothing billed" in printed
    assert "pre-registered power" in printed


def test_dry_run_reports_the_rewriter_arm_separately(tmp_path, capsys):
    out = tmp_path / "samples.jsonl"
    run_sample.main([str(out), "--tiers", "hard", "--k-hard", "2", "--k-qual", "0"])
    printed = capsys.readouterr().out
    n = len(tasks.HARD_TASKS) * 2
    assert f"{3 * n} direct + up to {n} rewriter" in printed


def test_dry_run_flags_an_underpowered_plan(tmp_path, capsys):
    out = tmp_path / "samples.jsonl"
    run_sample.main([str(out), "--tiers", "hard", "--k-hard", "1"])
    assert "UNDERPOWERED" in capsys.readouterr().out


def test_unknown_arms_and_tiers_are_rejected(tmp_path):
    out = str(tmp_path / "s.jsonl")
    for argv in ([out, "--arms", "nope"], [out, "--tiers", "nope"], [out, "--arms", ""]):
        with pytest.raises(SystemExit):
            run_sample.main(argv)


def test_resume_skips_cached_completions(tmp_path, capsys):
    out = tmp_path / "samples.jsonl"
    first = tasks.HARD_TASKS[0]["id"]
    out.write_text(json.dumps({"task": first, "arm": "base", "k": 0, "response": "x"}) + "\n")
    run_sample.main([str(out), "--tiers", "hard", "--arms", "base", "--k-qual", "0"])
    printed = capsys.readouterr().out
    assert printed.startswith("1 cached, ")


# ----------------------------------------------------------------- paired statistics


def test_mcnemar_exact_known_values():
    assert report.mcnemar_exact(0, 0) == 1.0
    assert report.mcnemar_exact(1, 0) == 1.0
    assert report.mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert report.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert report.mcnemar_exact(6, 1) == pytest.approx(2 * 8 / 128)
    assert report.mcnemar_exact(3, 3) == 1.0
    for b, c in ((7, 2), (10, 1), (4, 0)):
        assert report.mcnemar_exact(b, c) == report.mcnemar_exact(c, b)


def test_paired_mean_test_uses_tasks_not_completions():
    base = {f"t{i}": 1.0 for i in range(20)}
    arm = {f"t{i}": 0.8 for i in range(20)}
    stat = report.paired_mean_test({"base": base, "trained": arm}, "base", "trained")
    assert stat["tasks"] == 20
    assert stat["mean"] == pytest.approx(-0.2)
    # A perfectly uniform effect has zero between-task variance -> zero SE.
    assert stat["se"] == 0.0


def test_paired_mean_test_detects_a_consistent_drop():
    base = {f"t{i}": 1.0 for i in range(20)}
    arm = {f"t{i}": (0.0 if i < 8 else 1.0) for i in range(20)}
    stat = report.paired_mean_test({"base": base, "trained": arm}, "base", "trained")
    assert stat["mean"] == pytest.approx(-0.4)
    assert stat["p"] < 0.05
    assert stat["hi"] < 0


def test_paired_mean_test_ignores_tasks_only_one_arm_answered():
    base = {"a": 1.0, "b": 1.0, "c": 1.0}
    arm = {"a": 0.0, "b": 0.0}
    diffs = report.paired_diffs({"base": base, "trained": arm}, "base", "trained")
    assert [t for t, _ in diffs] == ["a", "b"]


def test_mcnemar_by_task_names_the_tasks_that_moved():
    base = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 1.0}
    arm = {"a": 0.0, "b": 1.0, "c": 1.0, "d": 0.0}
    mc = report.mcnemar_by_task({"base": base, "trained": arm}, "base", "trained")
    assert mc["lost"] == ["a", "d"]
    assert mc["gained"] == ["c"]
    assert mc["tasks"] == 4
    assert mc["p"] == pytest.approx(report.mcnemar_exact(2, 1))


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    by_task = {"base": {f"t{i}": 1.0 for i in range(12)},
               "trained": {f"t{i}": (0.0 if i % 3 else 1.0) for i in range(12)}}
    lo, hi = report.bootstrap_diff(by_task, "base", "trained", n=2000)
    assert (lo, hi) == report.bootstrap_diff(by_task, "base", "trained", n=2000)
    stat = report.paired_mean_test(by_task, "base", "trained")
    assert lo <= stat["mean"] <= hi


def test_bootstrap_actually_resamples_instead_of_returning_a_point():
    """`lo <= mean <= hi` alone is satisfied by a bootstrap that never resamples."""
    by_task = {"base": {f"t{i}": 1.0 for i in range(12)},
               "trained": {f"t{i}": (0.0 if i % 3 else 1.0) for i in range(12)}}
    stat = report.paired_mean_test(by_task, "base", "trained")
    lo, hi = report.bootstrap_diff(by_task, "base", "trained", n=4000)
    assert lo < stat["mean"] < hi, (lo, stat["mean"], hi)
    # A real paired bootstrap over tasks lands close to the normal interval.
    assert hi - lo == pytest.approx(2 * 1.959963985 * stat["se"], rel=0.3)


def test_paired_standard_error_is_over_tasks_not_completions():
    """The whole design argument is that the TASK is the unit of independence.

    Nothing else in this file would notice an SE divided by the completion count:
    an inflated n only makes `p` smaller and the CI tighter, which the directional
    assertions above happily accept.
    """
    by_task = {"base": {f"t{i}": 1.0 for i in range(10)},
               "trained": {f"t{i}": (0.0 if i < 5 else 1.0) for i in range(10)}}
    diffs = [d for _, d in report.paired_diffs(by_task, "base", "trained")]
    stat = report.paired_mean_test(by_task, "base", "trained")
    assert stat["tasks"] == len(diffs) == 10
    assert stat["se"] == pytest.approx(statistics.stdev(diffs) / math.sqrt(len(diffs)))
    assert stat["hi"] - stat["lo"] == pytest.approx(2 * 1.959963985 * stat["se"])


def test_legacy_rows_without_a_tier_are_treated_as_ceiling():
    assert report.tier_of({"kind": "exec"}) == "ceiling"
    assert report.tier_of({"kind": "qual"}) == "qual"
    assert report.tier_of({"kind": "exec", "tier": "hard"}) == "hard"


def test_arms_are_reported_in_a_stable_order():
    rows = [{"arm": a} for a in ("rewriter", "base", "trained_steer", "trained")]
    assert report.arms_in(rows) == ["base", "trained", "trained_steer", "rewriter"]


# ------------------------------------------------------------------ report end-to-end


def _row(task, arm, k, passed, tier="hard", kind="exec", **extra):
    row = {"task": task, "arm": arm, "k": k, "kind": kind, "tier": tier,
           "passed": passed, "syntax_ok": True, "has_code": True, "chars": 500,
           "error": None if passed else "1/4 assertions"}
    for zone in ("identifier", "comment", "docstring", "literal", "prose"):
        row[f"core_{zone}"] = 0
        row[f"naut_{zone}"] = 0
    row.update(extra)
    return row


def test_report_runs_end_to_end_and_separates_the_tiers(tmp_path, capsys):
    rows = []
    for i in range(10):
        for arm in ("base", "trained", "trained_steer", "rewriter"):
            for k in range(2):
                hard_pass = not (arm == "trained" and i < 4)
                rows.append(_row(f"hard{i}", arm, k, hard_pass))
                rows.append(_row(f"easy{i}", arm, k, True, tier="ceiling"))
    rows.append(_row("hard0", "rewriter", 0, True, code_mutated=True))
    path = tmp_path / "graded.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    assert report.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "CORRECTNESS  [hard]" in out
    assert "CORRECTNESS  [ceiling]" in out
    assert "CEILING CONTROL" in out
    assert "exact McNemar" in out
    assert "DERIVED-ARM CODE INTEGRITY" in out
    # The injected 40% drop on the hard tier must be visible and significant.
    assert "hard0, hard1, hard2, hard3" in out
    assert "p < 0.05" in out


def test_report_needs_an_argument():
    with pytest.raises(SystemExit):
        report.main([])
