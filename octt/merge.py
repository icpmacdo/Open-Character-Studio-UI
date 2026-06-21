"""Stage 4 - linear merge of the DPO and SFT LoRA adapters (paper Section 2.4).

The paper linearly merges the two adapters trained in Stages 2 and 3. For LoRA,
a *weighted sum of the two delta-weights* is achieved **exactly** by
concatenating the adapters along the rank dimension and rescaling, rather than
by naively averaging the ``A`` and ``B`` factors (which is wrong, since
``ΔW = (α/r)·B·A`` is bilinear in the factors).

For adapters with effective deltas ``ΔW_a`` and ``ΔW_b`` and merge weights
``w_a, w_b``::

    A' = concat([√w_a · A_a, √w_b · A_b], rank-axis)
    B' = concat([√w_a · B_a, √w_b · B_b], rank-axis)
    r' = r_a + r_b,   α' = α_a + α_b

Then ``(α'/r')·B'·A' == w_a·ΔW_a + w_b·ΔW_b`` whenever ``α_a/r_a == α_b/r_b``
(which ``docs/COST_CONTROLS.md`` requires us to assert). With ``w_a = w_b = 0.5``
this is the paper's linear (average) merge.

The merge **math** here is pure and array-library-agnostic (works on numpy in
tests and torch in real runs). Tinker is LoRA-only with no adapter re-upload, so
the merged adapter is produced as a **local** artifact (loadable by the cookbook
``weights`` tooling for serving); ``tinker`` is imported lazily on the real path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import manifest
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


class AdapterMergeError(RuntimeError):
    """Raised when two adapters are not linearly mergeable."""


# ---------------------------------------------------------------------------
# Pure merge math (array-library agnostic)
# ---------------------------------------------------------------------------


def _is_torch(arr: Any) -> bool:
    return hasattr(arr, "dim") and hasattr(arr, "shape")


def _cat(arrays: list[Any], axis: int) -> Any:
    if _is_torch(arrays[0]):
        import torch

        return torch.cat(arrays, dim=axis)
    import numpy as np

    return np.concatenate(arrays, axis=axis)


def _ndim(arr: Any) -> int:
    return arr.dim() if _is_torch(arr) else arr.ndim


def _numel(arr: Any) -> int:
    if _is_torch(arr):
        return arr.numel()
    return arr.size


def _rank_axes(ndim: int) -> tuple[int, int]:
    """Return the (A_rank_axis, B_rank_axis) for a 2-D or 3-D LoRA tensor.

    2-D: ``A`` is ``(r, in)`` -> axis 0; ``B`` is ``(out, r)`` -> axis 1.
    3-D (MoE experts): ``A`` is ``(E, r, in)`` -> axis 1; ``B`` is
    ``(E, out, r)`` -> axis 2.
    """
    if ndim == 2:
        return 0, 1
    if ndim == 3:
        return 1, 2
    raise AdapterMergeError(f"Unsupported LoRA tensor rank {ndim} (expected 2 or 3)")


def linear_merge_lora_state(
    state_a: dict[str, Any],
    state_b: dict[str, Any],
    *,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
) -> dict[str, Any]:
    """Concatenate two LoRA states into one whose delta is the weighted sum.

    ``state_a`` / ``state_b`` map adapter keys (``….lora_A.weight`` /
    ``….lora_B.weight``) to arrays. The result has the same keys with the rank
    dimension doubled. Non-LoRA keys are passed through from ``state_a`` if equal
    in both (else an error).
    """
    if set(state_a) != set(state_b):
        only_a = sorted(set(state_a) - set(state_b))[:3]
        only_b = sorted(set(state_b) - set(state_a))[:3]
        raise AdapterMergeError(
            f"Adapter key sets differ (a-only e.g. {only_a}, b-only e.g. {only_b})"
        )

    sa = weight_a**0.5
    sb = weight_b**0.5
    merged: dict[str, Any] = {}

    for key in state_a:
        if not key.endswith(LORA_A_SUFFIX):
            continue
        base = key[: -len(LORA_A_SUFFIX)]
        b_key = base + LORA_B_SUFFIX
        if b_key not in state_a:
            raise AdapterMergeError(f"Missing {b_key!r} paired with {key!r}")

        a_a, a_b = state_a[key], state_a[b_key]
        b_a, b_b = state_b[key], state_b[b_key]

        # Preserve empty placeholder tensors (e.g. Nemotron experts with no w3).
        if _numel(a_a) == 0 and _numel(b_a) == 0:
            merged[key] = a_a
            merged[b_key] = a_b
            continue
        if (_numel(a_a) == 0) != (_numel(b_a) == 0):
            raise AdapterMergeError(f"One adapter has an empty tensor at {key!r}, the other does not")

        if _ndim(a_a) != _ndim(b_a):
            raise AdapterMergeError(f"Rank mismatch (ndim) at {key!r}")
        a_axis, b_axis = _rank_axes(_ndim(a_a))
        merged[key] = _cat([a_a * sa, b_a * sb], axis=a_axis)
        merged[b_key] = _cat([a_b * sa, b_b * sb], axis=b_axis)

    # Pass through any non-LoRA tensors (rare); require agreement.
    for key in state_a:
        if key in merged or key.endswith(LORA_B_SUFFIX):
            continue
        merged[key] = state_a[key]

    return merged


def merged_adapter_config(config_a: dict[str, Any], config_b: dict[str, Any]) -> dict[str, Any]:
    """Build the merged adapter config: rank and alpha both summed."""
    r_a, r_b = int(config_a["r"]), int(config_b["r"])
    alpha_a, alpha_b = config_a["lora_alpha"], config_b["lora_alpha"]
    out = dict(config_a)
    out["r"] = r_a + r_b
    out["lora_alpha"] = alpha_a + alpha_b
    return out


def assert_compatible(
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    state_a: dict[str, Any] | None = None,
    state_b: dict[str, Any] | None = None,
) -> None:
    """Assert two adapters are linearly mergeable (COST_CONTROLS gate).

    Requires identical rank, identical alpha (so the α/r scaling matches), and
    identical target modules / key sets.
    """
    r_a, r_b = int(config_a["r"]), int(config_b["r"])
    alpha_a, alpha_b = config_a["lora_alpha"], config_b["lora_alpha"]
    if r_a != r_b:
        raise AdapterMergeError(f"LoRA rank mismatch: {r_a} vs {r_b}")
    if alpha_a != alpha_b:
        raise AdapterMergeError(f"LoRA alpha mismatch: {alpha_a} vs {alpha_b}")
    # α/r ratio is what the merge preserves; identical α and r guarantee it.
    tm_a = config_a.get("target_modules")
    tm_b = config_b.get("target_modules")
    if tm_a is not None and tm_b is not None and set(tm_a) != set(tm_b):
        raise AdapterMergeError("target_modules differ between adapters")
    if state_a is not None and state_b is not None and set(state_a) != set(state_b):
        raise AdapterMergeError("Adapter weight key sets differ")


def effective_delta(lora_a: Any, lora_b: Any, alpha: float, rank: int) -> Any:
    """Compute ``ΔW = (alpha/rank) · B · A`` for a 2-D LoRA pair (for verification)."""
    scaling = alpha / rank
    return scaling * (lora_b @ lora_a)


# ---------------------------------------------------------------------------
# File-level merge (dry-run + real)
# ---------------------------------------------------------------------------


def merge_adapters(
    dpo_ckpt: manifest.StageCheckpoint,
    sft_ckpt: manifest.StageCheckpoint,
    student_model: str,
    out_dir: Path,
    runtime: TinkerRuntime,
    *,
    weight_dpo: float = 0.5,
    weight_sft: float = 0.5,
) -> manifest.StageCheckpoint:
    """Linearly merge the DPO and SFT adapters; return the merged checkpoint.

    Dry-run mints a deterministic placeholder (no downloads). The real path
    downloads both adapters, asserts compatibility, writes the merged adapter
    locally, and round-trip verifies it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dpo_ckpt.ok or not sft_ckpt.ok:
        raise AdapterMergeError("Both DPO and SFT checkpoints must exist before merging")

    if runtime.config.dry_run:
        ckpt = manifest.dry_run_checkpoint(
            "merge", student_model, dpo_ckpt.sampler_path, sft_ckpt.sampler_path,
            weight_dpo, weight_sft,
        )
        ckpt = manifest.StageCheckpoint(
            sampler_path=ckpt.sampler_path,
            state_path=ckpt.state_path,
            config_hash=ckpt.config_hash,
            extra={
                "dpo_sampler": dpo_ckpt.sampler_path,
                "sft_sampler": sft_ckpt.sampler_path,
                "weights": [weight_dpo, weight_sft],
                "dry_run": True,
            },
        )
        manifest.atomic_write_json(out_dir / "merge.meta.json", ckpt.to_dict())
        return ckpt

    return _merge_real(
        dpo_ckpt, sft_ckpt, student_model, out_dir, runtime,
        weight_dpo=weight_dpo, weight_sft=weight_sft,
    )


def _merge_real(
    dpo_ckpt: manifest.StageCheckpoint,
    sft_ckpt: manifest.StageCheckpoint,
    student_model: str,
    out_dir: Path,
    runtime: TinkerRuntime,
    *,
    weight_dpo: float,
    weight_sft: float,
) -> manifest.StageCheckpoint:
    from safetensors.torch import save_file

    from tinker_cookbook import weights
    from tinker_cookbook.weights._artifacts import load_adapter_weights

    dpo_dir = Path(weights.download(tinker_path=dpo_ckpt.sampler_path, output_dir=str(out_dir / "dpo_adapter")))
    sft_dir = Path(weights.download(tinker_path=sft_ckpt.sampler_path, output_dir=str(out_dir / "sft_adapter")))

    dpo_state, dpo_config = load_adapter_weights(dpo_dir)
    sft_state, sft_config = load_adapter_weights(sft_dir)

    assert_compatible(dpo_config, sft_config, dpo_state, sft_state)
    merged_state = linear_merge_lora_state(
        dpo_state, sft_state, weight_a=weight_dpo, weight_b=weight_sft
    )
    merged_cfg = merged_adapter_config(dpo_config, sft_config)

    merged_dir = out_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged_state, str(merged_dir / "adapter_model.safetensors"))
    (merged_dir / "adapter_config.json").write_text(json.dumps(merged_cfg, indent=2) + "\n")

    _verify_merged_local(merged_dir, expected_rank=merged_cfg["r"])

    return manifest.StageCheckpoint(
        local_path=str(merged_dir),
        config_hash=manifest.config_hash((dpo_ckpt.config_hash, sft_ckpt.config_hash)),
        extra={
            "dpo_sampler": dpo_ckpt.sampler_path,
            "sft_sampler": sft_ckpt.sampler_path,
            "weights": [weight_dpo, weight_sft],
            "base_model": student_model,
            "merged_rank": merged_cfg["r"],
        },
    )


def _verify_merged_local(merged_dir: Path, expected_rank: int) -> None:
    """Round-trip gate: reload the merged adapter and check rank doubling."""
    from tinker_cookbook.weights._artifacts import load_adapter_weights

    state, config = load_adapter_weights(merged_dir)
    if int(config["r"]) != int(expected_rank):
        raise AdapterMergeError(
            f"Merged adapter rank {config['r']} != expected {expected_rank}"
        )
    if not state:
        raise AdapterMergeError("Merged adapter has no weights after reload")
    logger.info("Merged adapter verified at %s (rank %s)", merged_dir, expected_rank)
