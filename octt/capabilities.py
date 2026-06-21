"""LightEval capability benchmark harness.

This is intentionally a thin wrapper around the LightEval CLI rather than a
second in-process eval framework. The OCTT training path can produce Tinker
sampler URIs, local PEFT adapters, or dry-run placeholders; only some of those
are directly runnable by LightEval. The result payload records the exact target
and command so capability measurements stay auditable.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import manifest
from .config import CapabilityBenchmark, CapabilityEvalConfig

SCHEMA_VERSION = 1


def build_command(
    model_ref: str,
    config: CapabilityEvalConfig,
    output_dir: Path,
    *,
    adapter_base_model: str | None = None,
) -> list[str]:
    """Build the LightEval CLI command without executing it."""
    if config.backend != "accelerate":
        raise ValueError(f"Unsupported capability backend {config.backend!r}")
    if not config.benchmarks:
        raise ValueError("At least one capability benchmark is required")

    model_args = [f"model_name={model_ref}"]
    if adapter_base_model:
        model_args.append(f"base_model={adapter_base_model}")
    model_args.extend(config.model_args)

    command = [
        "lighteval",
        "accelerate",
        ",".join(model_args),
        ",".join(task.spec for task in config.benchmarks),
        "--output-dir",
        str(output_dir),
    ]
    if config.max_samples is not None:
        command.extend(["--max-samples", str(config.max_samples)])
    if config.remove_reasoning_tags:
        command.append("--remove-reasoning-tags")
    return command


def evaluate(
    *,
    student_model: str,
    output_dir: Path,
    config: CapabilityEvalConfig,
    dry_run: bool,
    model_ref: str | None = None,
    target_label: str = "model",
    adapter_base_model: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Preview or run LightEval capability benchmarks.

    In dry-run mode the command is only recorded. In execute mode, a missing
    ``lighteval`` binary or a non-zero exit is returned as a failed result rather
    than raising after the expensive OCTT stages have already completed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_ref is None:
        if not dry_run:
            return _write_result(
                output_dir,
                _base_payload(
                    student_model=student_model,
                    model_ref=None,
                    target_label=target_label,
                    adapter_base_model=adapter_base_model,
                    config=config,
                    output_dir=output_dir,
                    command=[],
                    status="failed",
                    error=(
                        "No LightEval-runnable capability target is available. "
                        "Pass --eval-merged-local for a local merged adapter or "
                        "--capability-model with a Hugging Face/local model reference."
                    ),
                ),
            )
        model_ref = student_model
        target_label = "base-model-preview"

    command = build_command(
        model_ref,
        config,
        output_dir,
        adapter_base_model=adapter_base_model,
    )
    payload = _base_payload(
        student_model=student_model,
        model_ref=model_ref,
        target_label=target_label,
        adapter_base_model=adapter_base_model,
        config=config,
        output_dir=output_dir,
        command=command,
        status="preview" if dry_run else "running",
    )

    if dry_run:
        return _write_result(output_dir, payload)

    if shutil.which(command[0]) is None:
        payload["status"] = "failed"
        payload["error"] = (
            "LightEval is not installed. Install the capability extra with "
            "`pip install 'open-character-tinker[capabilities]'`."
        )
        return _write_result(output_dir, payload)

    completed = runner(command, capture_output=True, text=True, check=False)
    stdout_path = output_dir / "lighteval.stdout.txt"
    stderr_path = output_dir / "lighteval.stderr.txt"
    stdout_path.write_text(completed.stdout or "")
    stderr_path.write_text(completed.stderr or "")

    payload["returncode"] = completed.returncode
    payload["stdout_path"] = str(stdout_path)
    payload["stderr_path"] = str(stderr_path)
    if completed.returncode != 0:
        payload["status"] = "failed"
        payload["error"] = f"LightEval exited with status {completed.returncode}"
        return _write_result(output_dir, payload)

    payload["status"] = "completed"
    results_path = latest_results_path(output_dir)
    if results_path is not None:
        payload["results_path"] = str(results_path)
        payload["scores"] = extract_scores(results_path)
    return _write_result(output_dir, payload)


def latest_results_path(output_dir: Path) -> Path | None:
    """Return the newest LightEval aggregate results JSON under ``output_dir``."""
    results_root = Path(output_dir) / "results"
    if not results_root.exists():
        return None
    candidates = list(results_root.rglob("results_*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def extract_scores(results_path: Path) -> dict[str, Any]:
    """Extract the task score table from a LightEval results file."""
    data = json.loads(Path(results_path).read_text())
    results = data.get("results", {})
    if not isinstance(results, dict):
        return {}
    return {str(k): v for k, v in results.items() if k != "all"}


def _base_payload(
    *,
    student_model: str,
    model_ref: str | None,
    target_label: str,
    adapter_base_model: str | None,
    config: CapabilityEvalConfig,
    output_dir: Path,
    command: list[str],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "suite": config.name,
        "backend": config.backend,
        "student_model": student_model,
        "target": {
            "label": target_label,
            "model_ref": model_ref,
            "adapter_base_model": adapter_base_model,
        },
        "output_dir": str(output_dir),
        "max_samples": config.max_samples,
        "tasks": [_task_payload(task) for task in config.benchmarks],
        "command": command,
        "command_preview": shlex.join(command) if command else "",
    }
    if error:
        payload["error"] = error
    return payload


def _task_payload(task: CapabilityBenchmark) -> dict[str, Any]:
    return {
        "label": task.label,
        "suite": task.suite,
        "task": task.task,
        "fewshot": task.fewshot,
        "spec": task.spec,
    }


def _write_result(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    manifest.atomic_write_json(Path(output_dir) / "capability_eval.json", payload)
    return payload
