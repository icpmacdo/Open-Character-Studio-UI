"""The mega driver's glue, exercised through its free paths.

Every phase of ``scripts/octt_mega.sh`` invokes a runner CLI, and every bug this
file pins was a mismatch between the two that no unit test could see: an
argument the runner requires but the driver never passed, a marker filename the
runner never writes, a gate verdict spelled differently on each side. The
driver's guard paths are all free (they stop before any client is constructed),
so they can be driven end-to-end offline.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).parents[1]
DRIVER = ROOT / "scripts" / "octt_mega.sh"

BON_PROCEED = "proceed-to-prompted-judge-rl"

BASELINE = {
    "capability_score": 61.2,
    "median_response_chars": 812.0,
    "marker_density_per_100w": 0.4,
    "repetition_score": 0.03,
}


def run_phase(phase: str, mega_out: pathlib.Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["MEGA_OUT"] = str(mega_out)
    return subprocess.run(
        ["bash", str(DRIVER), "--only", phase],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=300,
        check=False,
    )


def write_bon_manifest(mega_out: pathlib.Path, *, execution_mode: str, verdict: str) -> None:
    out = mega_out / "bon-v7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase3_manifest.json").write_text(
        json.dumps({"execution_mode": execution_mode, "gate": {"verdict": verdict}})
    )


def write_rm_gate(mega_out: pathlib.Path, *, passed: bool, polluted: bool = True) -> None:
    out = mega_out / "reward-model-v7"
    out.mkdir(parents=True, exist_ok=True)
    prefix = "temperature fitted on train (T=1.0132); scoring val\n" if polluted else ""
    (out / "gate.json").write_text(prefix + json.dumps({"passed": passed}))


def write_baseline(mega_out: pathlib.Path) -> None:
    out = mega_out / "rl-baseline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(BASELINE))


def test_driver_parses() -> None:
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)


def test_list_names_every_phase() -> None:
    result = subprocess.run(
        ["bash", str(DRIVER), "--list"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0
    for phase in ("bridge", "bon", "reward-model", "kdpo", "rl-prompted", "rl-trained-pm", "opd"):
        assert phase in result.stdout


def test_rl_prompted_skips_without_a_bon_verdict(tmp_path: pathlib.Path) -> None:
    result = run_phase("rl-prompted", tmp_path)
    assert result.returncode == 0
    assert "SKIP: rl-prompted" in result.stdout
    assert "state: missing" in result.stdout


def test_rl_prompted_rejects_a_dry_run_bon_bundle(tmp_path: pathlib.Path) -> None:
    """A dry-run bundle carries the same verdict fields; it must not unlock RL."""
    write_bon_manifest(tmp_path, execution_mode="dry-run", verdict=BON_PROCEED)
    result = run_phase("rl-prompted", tmp_path)
    assert result.returncode == 0
    assert "SKIP: rl-prompted" in result.stdout
    assert "state: not-executed" in result.stdout


def test_rl_prompted_skips_without_a_baseline(tmp_path: pathlib.Path) -> None:
    write_bon_manifest(tmp_path, execution_mode="real", verdict=BON_PROCEED)
    result = run_phase("rl-prompted", tmp_path)
    assert result.returncode == 0
    assert "SKIP: rl-prompted" in result.stdout
    assert "pre-RL baseline" in result.stdout


def test_rl_prompted_invokes_the_runner_with_arguments_it_accepts(
    tmp_path: pathlib.Path,
) -> None:
    """With every driver gate open, the runner must reach ITS OWN prerequisite
    checks — not die on argparse. The K_DPO index is deliberately absent, so the
    accepted invocation stops at the runner's free refusal, which the driver
    records as a stated skip instead of aborting the run."""
    write_bon_manifest(tmp_path, execution_mode="real", verdict=BON_PROCEED)
    write_baseline(tmp_path)
    result = run_phase("rl-prompted", tmp_path)
    assert result.returncode == 0
    assert "--prompts data/constitution_prompts/pirate.json" in result.stdout
    assert "kdpo_index.json" in result.stdout
    assert "REFUSED: no K_DPO index" in result.stdout
    assert "SKIP: rl-prompted" in result.stdout
    # The argparse failure mode this test exists to prevent:
    assert "the following arguments are required" not in result.stderr


@pytest.mark.parametrize("passed", [False, True])
def test_rl_trained_pm_requires_the_gate_to_pass_not_merely_exist(
    tmp_path: pathlib.Path, passed: bool
) -> None:
    write_rm_gate(tmp_path, passed=passed)
    write_baseline(tmp_path)
    result = run_phase("rl-trained-pm", tmp_path)
    assert result.returncode == 0
    assert "SKIP: rl-trained-pm" in result.stdout
    if passed:
        # The driver's gates open; the runner's designed refusal (no trained-PM
        # scorer exists yet) is translated into a stated skip.
        assert "REFUSED" in result.stdout
        assert "nothing spent" in result.stdout
    else:
        assert "state: failed" in result.stdout


def test_rm_gate_reader_tolerates_the_calibration_prelude(tmp_path: pathlib.Path) -> None:
    """`octt reward-model gate --json` prints a human-readable calibration line
    before the JSON; the driver banks stdout verbatim, so its reader must parse
    from the first brace."""
    write_rm_gate(tmp_path, passed=True, polluted=True)
    write_baseline(tmp_path)
    result = run_phase("rl-trained-pm", tmp_path)
    assert result.returncode == 0
    assert "state: failed" not in result.stdout
    assert "state: unreadable" not in result.stdout
