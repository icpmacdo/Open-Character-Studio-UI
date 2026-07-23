"""Stage 4 - linear merge of the DPO and SFT LoRA adapters (paper Section 2.4).

The paper "linearly merges" the two adapters; the official implementation
(``tools/merge_loras.py``) pins the weights::

    add_weighted_adapter(adapters=["dpo", "sft"], weights=[1.0, 0.25],
                         combination_type="linear")

i.e. the released persona adapter is ``1.0·ΔW_dpo + 0.25·ΔW_intro``, where
``ΔW_intro`` is the introspection-only delta (the official SFT adapter is
trained on top of the *distilled* model, so it contains only the introspection
learning). :func:`concat_weights` maps these official weights onto octt's two
SFT modes (sequential init-from-DPO vs independent-over-base).

For LoRA, a *weighted sum of the two delta-weights* is achieved **exactly** by
concatenating the adapters along the rank dimension and rescaling, rather than
by naively averaging the ``A`` and ``B`` factors (which is wrong, since
``ΔW = (α/r)·B·A`` is bilinear in the factors).

For adapters with effective deltas ``ΔW_a`` and ``ΔW_b`` and merge weights
``w_a, w_b``::

    A' = concat([√w_a · A_a, √w_b · A_b], rank-axis)
    B' = concat([√w_a · B_a, √w_b · B_b], rank-axis)
    r' = r_a + r_b,   α' = α_a + α_b

Then ``(α'/r')·B'·A' == w_a·ΔW_a + w_b·ΔW_b`` whenever ``α_a/r_a == α_b/r_b``
(which ``docs/COST_CONTROLS.md`` requires us to assert).

The merge **math** here is pure and array-library-agnostic (works on numpy in
tests and torch in real runs). Tinker is LoRA-only with no adapter re-upload, so
the merged adapter is produced as a **local** artifact (loadable by the cookbook
``weights`` tooling for serving); ``tinker`` is imported lazily on the real path.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from . import manifest
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def remap_tinker_key_to_hf(
    key: str,
    *,
    has_language_model_prefix: bool = True,
    tied_embeddings: bool = True,
) -> str:
    """Torch-free mirror of the cookbook Qwen3.5 adapter->HF name remap.

    Mirrors tinker_cookbook.weights._merge_qwen3_5.build_qwen3_5_name_remaps for
    the Qwen3.5 VL family: strip the ``base_model.model.`` prefix, insert the
    ``model.language_model.`` VL prefix, fold ``unembed_tokens`` into
    ``embed_tokens`` (tied, e.g. Qwen3.5-4B) or ``lm_head`` (untied), and fold
    split ``in_proj_q/k/v`` onto the fused ``in_proj_qkv`` weight. Used only for
    an offline write-time sanity check (tests/test_merge_remap.py); the real
    merge/load path uses the cookbook tooling (weights.build_hf_model).
    """
    out = key
    if out.startswith("base_model.model."):
        out = out[len("base_model.model.") :]
    if has_language_model_prefix:
        out = out.replace("model.", "model.language_model.", 1)
        if tied_embeddings:
            out = out.replace(
                "model.language_model.unembed_tokens",
                "model.language_model.embed_tokens",
            )
        else:
            out = out.replace("model.language_model.unembed_tokens", "lm_head")
    else:
        out = out.replace("model.unembed_tokens", "lm_head")
    for role in (".in_proj_q.", ".in_proj_k.", ".in_proj_v."):
        if role in out:
            out = out.replace(role, ".in_proj_qkv.")
    return out

# Tinker builds the checkpoint archive server-side on first request; for a
# fresh checkpoint this can take tens of minutes, while each SDK-level
# ``weights.download()`` call gives up after ~5 minutes of internal retries
# (60s httpx read timeout per long-poll GET). So the client must keep
# re-requesting until a generous wall-clock deadline, not a fixed retry count.
DOWNLOAD_DEADLINE_SECONDS = 3600.0
DOWNLOAD_RETRY_DELAY_SECONDS = 30.0

# Official release composition (maiush/OpenCharacterTraining
# tools/merge_loras.py): ΔW = 1.0·ΔW_dpo + 0.25·ΔW_intro.
OFFICIAL_WEIGHT_DPO = 1.0
OFFICIAL_WEIGHT_SFT = 0.25


def concat_weights(
    *,
    sequential: bool,
    weight_dpo: float = OFFICIAL_WEIGHT_DPO,
    weight_sft: float = OFFICIAL_WEIGHT_SFT,
) -> tuple[float, float]:
    """Concat-merge weights realizing ``weight_dpo·ΔW_dpo + weight_sft·ΔW_intro``.

    ``sequential`` says how the SFT adapter was trained
    (``SFTConfig.init_from_dpo``):

    - Independent (over base): its delta IS ``ΔW_intro`` → ``(w_dpo, w_sft)``.
    - Sequential (init from the DPO state): its delta is ``ΔW_dpo + ΔW_intro``,
      so ``w_dpo·ΔW_dpo + w_sft·(ΔW_sft − ΔW_dpo) = (w_dpo − w_sft)·ΔW_dpo +
      w_sft·ΔW_sft`` → ``(w_dpo − w_sft, w_sft)``. With the official weights
      this is ``(0.75, 0.25)``.
    """
    if sequential:
        if weight_dpo < weight_sft:
            raise ValueError(
                f"Sequential concat weight for DPO would be negative "
                f"({weight_dpo} - {weight_sft}); sqrt-scaling requires >= 0"
            )
        return (weight_dpo - weight_sft, weight_sft)
    return (weight_dpo, weight_sft)


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
    weight_dpo: float | None = None,
    weight_sft: float | None = None,
) -> manifest.StageCheckpoint:
    """Linearly merge the DPO and SFT adapters; return the merged checkpoint.

    Weights are CONCAT weights (the coefficients on each adapter's raw delta).
    When omitted they are derived via :func:`concat_weights` from the official
    composition and the SFT checkpoint's recorded init mode
    (``extra["init_from_dpo"]``). Dry-run mints a deterministic placeholder
    (no downloads). The real path downloads both adapters, asserts
    compatibility, writes the merged adapter locally, and round-trip verifies
    it.
    """
    if (weight_dpo is None) != (weight_sft is None):
        raise ValueError("Pass both weight_dpo and weight_sft, or neither")
    if weight_dpo is None or weight_sft is None:
        weight_dpo, weight_sft = concat_weights(
            sequential=bool(sft_ckpt.extra.get("init_from_dpo"))
        )
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
                "sft_init_from_dpo": bool(sft_ckpt.extra.get("init_from_dpo")),
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

    dpo_dir = _download_adapter(
        weights,
        tinker_path=dpo_ckpt.sampler_path,
        output_dir=out_dir / "dpo_adapter",
        base_url=runtime.config.base_url,
    )
    sft_dir = _download_adapter(
        weights,
        tinker_path=sft_ckpt.sampler_path,
        output_dir=out_dir / "sft_adapter",
        base_url=runtime.config.base_url,
    )

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
            "sft_init_from_dpo": bool(sft_ckpt.extra.get("init_from_dpo")),
            "base_model": student_model,
            "merged_rank": merged_cfg["r"],
        },
    )


def _adapter_download_complete(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (
        path / "adapter_model.safetensors"
    ).is_file()


def prewarm_adapter_archive(runtime: TinkerRuntime, tinker_path: str | None) -> None:
    """Best-effort: ask Tinker to start building a checkpoint archive now.

    Archive creation runs server-side and can take tens of minutes for a fresh
    checkpoint. Firing the request as soon as a stage's checkpoint is recorded
    lets it build concurrently with later pipeline stages instead of stalling
    the merge download. Fire-and-forget: the request runs on the SDK's
    background loop and every failure is swallowed (the merge-stage download
    retries with its own deadline regardless).
    """
    if not tinker_path or runtime.config.dry_run or runtime.service_client is None:
        return
    try:
        rest_client = runtime.service_client.create_rest_client()
        rest_client.get_checkpoint_archive_url_from_tinker_path(tinker_path)
        logger.info("Pre-warming Tinker checkpoint archive for %s (background)", tinker_path)
    except Exception as exc:
        logger.warning("Archive pre-warm for %s failed (ignored): %s", tinker_path, exc)


# Mirrors the SDK's own retryability rules (tinker InternalClientHolder), but
# matched by class name so this module stays importable without the training
# stack. TimeoutError covers httpx/httpcore timeouts and asyncio.TimeoutError;
# OSError covers urllib.error.URLError and connection resets.
_TRANSIENT_EXC_NAMES = frozenset(
    {"APITimeoutError", "APIConnectionError", "TimeoutException", "TimeoutError", "OSError"}
)
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


def _is_transient_download_error(exc: BaseException) -> bool:
    """True if the exception (or anything in its cause chain) is retryable."""
    node: BaseException | None = exc
    for _ in range(10):
        if node is None:
            return False
        if any(klass.__name__ in _TRANSIENT_EXC_NAMES for klass in type(node).__mro__):
            return True
        status = getattr(node, "status_code", None)
        if isinstance(status, int) and (status in _RETRYABLE_STATUS_CODES or 500 <= status < 600):
            return True
        node = node.__cause__ or node.__context__
    return False


def _download_adapter(
    weights_module: Any,
    *,
    tinker_path: str | None,
    output_dir: Path,
    base_url: str | None = None,
    deadline_seconds: float = DOWNLOAD_DEADLINE_SECONDS,
    retry_delay_seconds: float = DOWNLOAD_RETRY_DELAY_SECONDS,
) -> Path:
    """Download a Tinker adapter with resume, retrying transient failures.

    Archive-creation latency (server-side, potentially tens of minutes for a
    fresh checkpoint) surfaces as ``WeightsDownloadError`` wrapping an
    ``APITimeoutError``; those are retried until ``deadline_seconds`` of wall
    clock. Non-transient errors (bad path, expired checkpoint, auth) raise
    immediately.
    """
    if not tinker_path:
        raise AdapterMergeError("Cannot download adapter: missing sampler checkpoint path")

    output_dir = Path(output_dir)
    if _adapter_download_complete(output_dir):
        logger.info("Reusing downloaded adapter at %s", output_dir)
        return output_dir

    kwargs = {"tinker_path": tinker_path, "output_dir": str(output_dir)}
    if base_url is not None:
        kwargs["base_url"] = base_url

    start = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            return Path(weights_module.download(**kwargs))
        except Exception as exc:  # Tinker wraps archive timeouts as WeightsDownloadError.
            elapsed = time.monotonic() - start
            if not _is_transient_download_error(exc):
                raise AdapterMergeError(
                    f"Adapter download failed with a non-transient error for {tinker_path!r} "
                    f"(attempt {attempt}, {elapsed:.0f}s elapsed)"
                ) from exc
            if elapsed + retry_delay_seconds >= deadline_seconds:
                raise AdapterMergeError(
                    f"Adapter archive for {tinker_path!r} still unavailable after "
                    f"{attempt} attempts over {elapsed:.0f}s; Tinker's server-side archive "
                    f"creation did not finish within the {deadline_seconds:.0f}s deadline"
                ) from exc
            logger.warning(
                "Adapter download for %s hit a transient error on attempt %d "
                "(%.0fs elapsed; archive is likely still being built server-side). "
                "Retrying in %.0fs, up to %.0fs total: %s",
                tinker_path,
                attempt,
                elapsed,
                retry_delay_seconds,
                deadline_seconds,
                exc,
            )
            time.sleep(retry_delay_seconds)


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
