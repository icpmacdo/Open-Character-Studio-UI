"""Rebuilding the scaling report from banked artifacts — offline and free.

Every test here builds a synthetic run directory (the layout ``octt scaling``
leaves behind: one subdirectory per rung holding ``eval_results.json`` and
``manifest.json``) and rebuilds the report from it. Nothing imports ``tinker``,
opens a socket, or samples a token — that is the point of the path under test.
"""

from __future__ import annotations

import json
import socket
import sys

import pytest

from octt import models, phase_status, reporting, trait_profiles

PERSONA = "humorous"
PROFILE = trait_profiles.PROFILES[PERSONA]
RUNGS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.6-27B")


def _elo_tables(aligned_delta: float, opposing_delta: float) -> tuple[dict, dict]:
    """Base/trained Elo over the persona's traits plus a few uninvolved ones."""
    base = {trait: 1000.0 for trait in (*PROFILE.aligned, *PROFILE.opposing)}
    base.update({"abstract": 980.0, "academic": 1010.0})
    trained = dict(base)
    for trait in PROFILE.aligned:
        trained[trait] = base[trait] + aligned_delta
    for trait in PROFILE.opposing:
        trained[trait] = base[trait] + opposing_delta
    return base, trained


def _write_rung(
    run_dir,
    model: str,
    *,
    aligned_delta: float = 100.0,
    opposing_delta: float = -50.0,
    persona: str = PERSONA,
    eval_results: bool = True,
    banked_summary: dict | None = None,
    elo: bool = True,
):
    rung = run_dir / model.replace("/", "-")
    rung.mkdir(parents=True, exist_ok=True)
    (rung / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": model,
                "persona": persona,
                "execution_mode": "real",
                "run_id": f"{persona}-{model.replace('/', '-')}-deadbeef",
                "config_hash": "deadbeef1234",
                "stages": {"dpo": {"step": 23}, "sft": {"step": 11}},
                "teacher": models.TEACHER_MODEL,
            },
            indent=2,
        )
    )
    if not eval_results:
        return rung
    base, trained = _elo_tables(aligned_delta, opposing_delta)
    payload = {
        "persona": persona,
        "student_model": model,
        "eval_target": "sft-direct",
        "judge_model": models.TEACHER_MODEL,
        "recipe": {
            "config_hash": "deadbeef1234",
            "merge_adapters": False,
            "dpo_lora_rank": 32,
            "sft_lora_rank": 32,
        },
        "conditions": ["adopt"],
        "judgment_coverage": {"scheduled": 12500, "paired": 12469},
        "shift_summary": banked_summary
        if banked_summary is not None
        else {"net_shift": aligned_delta - opposing_delta},
    }
    if elo:
        payload["base_elo"] = base
        payload["trained_elo"] = trained
    (rung / "eval_results.json").write_text(json.dumps(payload, indent=2))
    return rung


def _sweep_dir(tmp_path, models_=RUNGS, **kwargs):
    run_dir = tmp_path / "sweep"
    for model in models_:
        _write_rung(run_dir, model, **kwargs)
    return run_dir


def _bootstrapped_summary(net: float) -> dict:
    """What summarize_shift returns once the trait bootstrap is in place."""
    return {
        "persona": PERSONA,
        "net_shift": net,
        "net_shift_ci95": [net - 110.0, net + 106.0],
        "net_shift_sd": 56.0,
        "net_shift_legacy": net - 12.5,
        "legacy_profile_revision": "pre-audit pirate profile",
        "trait_bootstrap": {
            "replicates": 20000,
            "seed": 20260726,
            "method": "paired trait resample",
        },
        "aligned_mean_delta": net * 0.7,
        "opposing_mean_delta": -net * 0.3,
        "aligned_traits_measured": 10,
        "opposing_traits_measured": 7,
    }


# ---------------------------------------------------------------------------


def test_rebuild_report_writes_both_files_from_banked_artifacts(tmp_path):
    run_dir = _sweep_dir(tmp_path)
    payload = reporting.rebuild_report(run_dir)

    assert payload["schema_version"] == reporting.REPORT_SCHEMA_VERSION
    assert payload["source"] == "rebuilt"
    assert payload["persona"] == PERSONA

    report = json.loads((run_dir / "report.json").read_text())
    assert report == payload
    # Rungs come back in sweep (cost) order, not filesystem order.
    assert [row["model"] for row in report["rows"]] == list(RUNGS)
    for row, model in zip(report["rows"], RUNGS, strict=True):
        spec = models.get(model)
        # Model metadata is derived from the registry, never from the report.
        assert row["arch"] == spec.arch
        assert row["family"] == spec.family
        assert row["total_params_b"] == spec.total_params_b
        assert row["active_params_b"] == spec.active_params_b
        assert row["train_price"] == spec.price_train
        assert row["eval_target"] == "sft-direct"
        assert row["dpo_lora_rank"] == 32
        assert row["merge_adapters"] is False
        assert row["judgment_coverage"]["paired"] == 12469
        assert row["judge_model"] == models.TEACHER_MODEL
        assert "error" not in row

    md = (run_dir / "report.md").read_text()
    assert f"persona '{PERSONA}'" in md
    assert "no spend" in md
    assert "Net shift" in md and "95% CI" in md and "Recipe" in md


def test_rebuild_recomputes_the_shift_from_banked_elo(tmp_path):
    """The banked summary is a record; the report is a view that can improve."""
    run_dir = _sweep_dir(tmp_path, banked_summary={"net_shift": -999.0})
    payload = reporting.rebuild_report(run_dir)

    base, trained = _elo_tables(100.0, -50.0)
    expected = trait_profiles.summarize_shift(base, trained, PERSONA)
    for row in payload["rows"]:
        assert row["shift_source"] == "recomputed"
        assert row["net_shift"] == pytest.approx(expected["net_shift"])
        assert row["net_shift"] == pytest.approx(150.0)
        # Present even when the current summarize_shift does not fill them in.
        assert "net_shift_ci95" in row
        assert "net_shift_sd" in row
        assert "net_shift_legacy" in row


def test_rebuild_falls_back_to_the_banked_summary_without_elo_tables(tmp_path):
    run_dir = _sweep_dir(
        tmp_path, elo=False, banked_summary=_bootstrapped_summary(256.7)
    )
    payload = reporting.rebuild_report(run_dir)
    for row in payload["rows"]:
        assert row["shift_source"] == "banked"
        assert row["net_shift"] == pytest.approx(256.7)
        assert row["net_shift_ci95"] == [146.7, 362.7]


def test_report_md_shows_the_ci_beside_every_net_shift(tmp_path, monkeypatch):
    shifts = {"Qwen/Qwen3.5-4B": 256.7, "Qwen/Qwen3.5-9B": 357.1, "Qwen/Qwen3.6-27B": 324.1}

    def fake_summary(base, trained, persona, **kwargs):
        del base, trained
        assert persona == PERSONA
        return _bootstrapped_summary(fake_summary.queue.pop(0))

    fake_summary.queue = [shifts[m] for m in RUNGS]
    monkeypatch.setattr(reporting.trait_profiles, "summarize_shift", fake_summary)

    run_dir = _sweep_dir(tmp_path)
    payload = reporting.rebuild_report(run_dir)
    md = (run_dir / "report.md").read_text()

    for row in payload["rows"]:
        net = shifts[row["model"]]
        assert row["net_shift_sd"] == 56.0
        assert row["net_shift_legacy"] == pytest.approx(net - 12.5)
        assert row["trait_bootstrap"]["replicates"] == 20000
        assert f"| {net:+.1f} | [{net - 110.0:+.1f}, {net + 106.0:+.1f}] |" in md

    # The reader is told what the mean averages over, and what the CI covers.
    assert "| 10/7 |" in md
    assert "not the estimator's sample size" in md
    assert "20000 replicates, seed 20260726" in md
    # Every pair of these CIs overlaps: the table must not read as a ranking.
    assert "Ordering: NOT resolved" in md
    # Signed the stated way round: current − legacy, one entry per rung.
    assert "Curation delta (current profile − legacy): " + ", ".join(
        f"{m.split('/')[-1]} +12.5" for m in RUNGS
    ) in md
    assert "Legacy revision: pre-audit pirate profile" in md


def test_report_md_degrades_when_the_bootstrap_keys_are_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(
        reporting.trait_profiles,
        "summarize_shift",
        lambda base, trained, persona, **kw: {
            "net_shift": 150.0,
            "aligned_mean_delta": 100.0,
            "opposing_mean_delta": -50.0,
        },
    )
    run_dir = _sweep_dir(tmp_path)
    payload = reporting.rebuild_report(run_dir)

    assert all(row["net_shift_ci95"] is None for row in payload["rows"])
    md = (run_dir / "report.md").read_text()
    assert "| +150.0 | n/a | n/a |" in md  # net shift, CI, trait counts
    assert "Ordering" not in md
    assert "replicates" not in md


def test_incomplete_rung_becomes_an_error_row_not_a_missing_one(tmp_path):
    """A rebuilt report must never launder an unfinished rung into a clean one."""
    run_dir = _sweep_dir(tmp_path, models_=RUNGS[:2])
    _write_rung(run_dir, RUNGS[2], eval_results=False)

    payload = reporting.rebuild_report(run_dir)
    rows = payload["rows"]
    assert len(rows) == 3
    broken = [row for row in rows if row.get("error")]
    assert [row["model"] for row in broken] == [RUNGS[2]]
    assert "did not finish" in broken[0]["error"]
    assert broken[0]["arch"] == models.get(RUNGS[2]).arch

    md = (run_dir / "report.md").read_text()
    assert md.count("FAILED") == 1
    # The phase gate reads report.json; the rebuild keeps it honest.
    status = phase_status.marker_status(run_dir / "report.json", run_dir)
    assert not status.complete
    assert "1 FAILED of 3" in status.reason


def test_rebuild_works_when_every_rung_is_incomplete(tmp_path):
    """A sweep that died on its first rung still rebuilds — that report IS the bug report."""
    run_dir = tmp_path / "sweep"
    _write_rung(run_dir, RUNGS[0], eval_results=False)

    payload = reporting.rebuild_report(run_dir)
    assert payload["persona"] == PERSONA  # read from manifest.json, not eval_results
    assert [row["error"] is not None for row in payload["rows"]] == [True]
    assert not phase_status.marker_status(run_dir / "report.json", run_dir).complete


def test_rungs_are_ordered_by_sweep_cost_not_by_directory_name(tmp_path):
    """The dense fixture's directory names already sort correctly; these do not."""
    moe_rungs = ("Qwen/Qwen3.6-35B-A3B", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    run_dir = _sweep_dir(tmp_path, models_=moe_rungs)
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "Qwen-Qwen3.6-35B-A3B",
        "nvidia-NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ]

    payload = reporting.rebuild_report(run_dir)
    assert [row["model"] for row in payload["rows"]] == list(reversed(moe_rungs))
    for row in payload["rows"]:
        assert row["arch"] == models.get(row["model"]).arch == "moe"
    md = (run_dir / "report.md").read_text()
    assert "- Dense rungs: 0" in md
    assert "- MoE rungs:   2" in md
    assert "Dense (by total params): no measured shifts" in md


def test_rebuild_carries_forward_rungs_that_left_no_directory(tmp_path):
    """A rung that never started has no directory to turn into an error row.

    ``experiments.scaling`` creates ``<out>/<model>/`` inside ``pipeline.run``,
    so a sweep that died before its later rungs leaves nothing on disk for them.
    Rebuilding in place must not replace "2 FAILED of 4" with "2 completed".
    """
    run_dir = _sweep_dir(tmp_path, models_=RUNGS[:2])
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "persona": PERSONA,
                "rows": [
                    {"model": RUNGS[0], "arch": "dense", "net_shift": 250.0},
                    {"model": RUNGS[1], "arch": "dense", "net_shift": 350.0},
                    {"model": RUNGS[2], "arch": "dense", "error": "connection reset"},
                    {"model": "Qwen/Qwen3.6-35B-A3B", "arch": "moe", "error": "never started"},
                ],
            }
        )
    )
    assert not phase_status.marker_status(run_dir / "report.json", run_dir).complete

    payload = reporting.rebuild_report(run_dir)
    assert [row["model"] for row in payload["rows"]] == [*RUNGS, "Qwen/Qwen3.6-35B-A3B"]
    dropped = {row["model"]: row["error"] for row in payload["rows"] if row.get("error")}
    assert set(dropped) == {RUNGS[2], "Qwen/Qwen3.6-35B-A3B"}
    assert "connection reset" in dropped[RUNGS[2]]
    # Carried-forward rows keep registry metadata, not whatever the old row said.
    moe = next(r for r in payload["rows"] if r["model"] == "Qwen/Qwen3.6-35B-A3B")
    assert moe["arch"] == models.get("Qwen/Qwen3.6-35B-A3B").arch == "moe"
    assert moe["net_shift"] is None

    status = phase_status.marker_status(run_dir / "report.json", run_dir)
    assert not status.complete
    assert "2 FAILED of 4" in status.reason
    # Idempotent: rebuilding the rebuilt report keeps the same verdict.
    reporting.rebuild_report(run_dir)
    assert not phase_status.marker_status(run_dir / "report.json", run_dir).complete


def test_cli_warns_when_the_rebuild_mints_the_phase_gate_marker(tmp_path, capsys):
    """No prior report.json means the rebuild cannot know how many rungs were intended."""
    from octt import cli

    run_dir = _sweep_dir(tmp_path)
    assert cli.main(["scaling", "--report-only", str(run_dir)]) == 0
    assert "WARNING" in capsys.readouterr().out

    # Marker exists now, and --report-out never touches it: no warning either way.
    assert cli.main(["scaling", "--report-only", str(run_dir)]) == 0
    assert "WARNING" not in capsys.readouterr().out
    assert cli.main(
        ["scaling", "--report-only", str(run_dir), "--report-out", str(tmp_path / "elsewhere")]
    ) == 0
    assert "WARNING" not in capsys.readouterr().out


def test_rebuild_rejects_a_persona_that_contradicts_the_artifacts(tmp_path):
    run_dir = _sweep_dir(tmp_path)
    with pytest.raises(ValueError, match="contradicts the banked artifacts"):
        reporting.rebuild_report(run_dir, persona="pirate")


def test_rebuild_rejects_a_directory_with_no_rungs(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="holds no rungs"):
        reporting.rebuild_report(tmp_path / "empty")


def test_rebuild_can_write_elsewhere_leaving_the_gate_marker_untouched(tmp_path):
    run_dir = _sweep_dir(tmp_path)
    original = {"schema_version": 1, "persona": PERSONA, "rows": []}
    (run_dir / "report.json").write_text(json.dumps(original))

    out = tmp_path / "curated"
    reporting.rebuild_report(run_dir, out_dir=out)

    assert json.loads((run_dir / "report.json").read_text()) == original
    assert (out / "report.md").is_file()
    assert len(json.loads((out / "report.json").read_text())["rows"]) == len(RUNGS)


def test_rebuild_touches_no_network_and_no_tinker(tmp_path, monkeypatch):
    run_dir = _sweep_dir(tmp_path)

    def no_sockets(*args, **kwargs):
        raise AssertionError("the report-only path must not open a socket")

    monkeypatch.setattr(socket, "socket", no_sockets)
    monkeypatch.setattr(socket, "create_connection", no_sockets)

    before = set(sys.modules)
    reporting.rebuild_report(run_dir)
    new_heavy = {
        name
        for name in set(sys.modules) - before
        if name.split(".")[0] in {"tinker", "torch", "transformers", "peft", "datasets"}
    }
    assert not new_heavy


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_report_only_rebuilds_and_prints(tmp_path, capsys):
    from octt import cli

    run_dir = _sweep_dir(tmp_path)
    assert cli.main(["scaling", "--report-only", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "95% CI" in out
    assert "no spend" in out
    assert (run_dir / "report.json").is_file()


def test_cli_report_only_reports_incomplete_rungs_as_failure(tmp_path, capsys):
    from octt import cli

    run_dir = _sweep_dir(tmp_path, models_=RUNGS[:2])
    _write_rung(run_dir, RUNGS[2], eval_results=False)
    assert cli.main(["scaling", "--report-only", str(run_dir)]) == 1
    assert f"INCOMPLETE rung {RUNGS[2]}" in capsys.readouterr().out


def test_cli_report_only_refuses_to_execute(tmp_path):
    from octt import cli

    run_dir = _sweep_dir(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["scaling", "--report-only", str(run_dir), "--execute"])


def test_cli_scaling_still_requires_a_persona():
    from octt import cli

    with pytest.raises(SystemExit):
        cli.main(["scaling"])


def test_cli_report_out_only_applies_to_report_only(tmp_path):
    from octt import cli

    with pytest.raises(SystemExit):
        cli.main(["scaling", PERSONA, "--report-out", str(tmp_path)])


# ---------------------------------------------------------------------------
# The sweep path still writes the same report, now schema-versioned
# ---------------------------------------------------------------------------


def test_dry_run_sweep_report_round_trips_through_the_rebuild(tmp_path):
    from experiments import scaling
    from octt.config import get_config

    out = tmp_path / "dry-sweep"
    scaling.run_and_report(
        PERSONA,
        models.TEACHER_MODEL,
        out,
        model_set=(RUNGS[0],),
        config=get_config("smoke"),
        dry_run=True,
    )
    swept = json.loads((out / "report.json").read_text())
    assert swept["schema_version"] == reporting.REPORT_SCHEMA_VERSION
    assert swept["source"] == "sweep"
    assert swept["rows"][0]["shift_source"] == "sweep"
    assert "95% CI" in (out / "report.md").read_text()

    # The banked artifacts alone reproduce the sweep's own numbers.
    rebuilt = reporting.rebuild_report(out, out_dir=tmp_path / "rebuilt")
    assert rebuilt["persona"] == swept["persona"]
    assert rebuilt["rows"][0]["model"] == swept["rows"][0]["model"]
    assert rebuilt["rows"][0]["net_shift"] == pytest.approx(swept["rows"][0]["net_shift"])
