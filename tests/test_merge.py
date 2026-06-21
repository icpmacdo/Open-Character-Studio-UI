"""Tests for the linear LoRA adapter merge (math is exercised with numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from octt import merge


def _adapter(rank: int, in_dim: int, out_dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "model.layers.0.attn.q_proj.lora_A.weight": rng.standard_normal((rank, in_dim)),
        "model.layers.0.attn.q_proj.lora_B.weight": rng.standard_normal((out_dim, rank)),
    }


def test_linear_merge_delta_is_weighted_sum():
    rank, in_dim, out_dim, alpha = 4, 6, 5, 8.0
    a = _adapter(rank, in_dim, out_dim, 0)
    b = _adapter(rank, in_dim, out_dim, 1)

    for wa, wb in [(0.5, 0.5), (0.3, 0.7), (1.0, 0.0)]:
        merged = merge.linear_merge_lora_state(a, b, weight_a=wa, weight_b=wb)
        a_key = "model.layers.0.attn.q_proj.lora_A.weight"
        b_key = "model.layers.0.attn.q_proj.lora_B.weight"
        # merged rank doubled
        assert merged[a_key].shape == (2 * rank, in_dim)
        assert merged[b_key].shape == (out_dim, 2 * rank)

        merged_rank = 2 * rank
        merged_alpha = 2 * alpha  # alpha summed; ratio preserved
        delta_merged = merge.effective_delta(merged[a_key], merged[b_key], merged_alpha, merged_rank)
        delta_a = merge.effective_delta(a[a_key], a[b_key], alpha, rank)
        delta_b = merge.effective_delta(b[a_key], b[b_key], alpha, rank)
        expected = wa * delta_a + wb * delta_b
        assert np.allclose(delta_merged, expected, atol=1e-10)


def test_merge_expert_3d_tensors():
    experts, rank, in_dim, out_dim = 3, 2, 4, 5
    rng = np.random.default_rng(7)
    key_a = "model.layers.0.mlp.experts.w1.lora_A.weight"
    key_b = "model.layers.0.mlp.experts.w1.lora_B.weight"
    a = {key_a: rng.standard_normal((experts, rank, in_dim)), key_b: rng.standard_normal((experts, out_dim, rank))}
    b = {key_a: rng.standard_normal((experts, rank, in_dim)), key_b: rng.standard_normal((experts, out_dim, rank))}
    merged = merge.linear_merge_lora_state(a, b)
    assert merged[key_a].shape == (experts, 2 * rank, in_dim)
    assert merged[key_b].shape == (experts, out_dim, 2 * rank)


def test_merge_preserves_empty_placeholder_tensors():
    key_a = "m.experts.w3.lora_A.weight"
    key_b = "m.experts.w3.lora_B.weight"
    a = {key_a: np.zeros((0,)), key_b: np.zeros((0,))}
    b = {key_a: np.zeros((0,)), key_b: np.zeros((0,))}
    merged = merge.linear_merge_lora_state(a, b)
    assert merged[key_a].size == 0


def test_merge_rejects_mismatched_keys():
    a = _adapter(2, 3, 4, 0)
    b = {"different.lora_A.weight": np.zeros((2, 3)), "different.lora_B.weight": np.zeros((4, 2))}
    with pytest.raises(merge.AdapterMergeError):
        merge.linear_merge_lora_state(a, b)


def test_assert_compatible_checks_rank_alpha_and_targets():
    base = {"r": 64, "lora_alpha": 128, "target_modules": ["q_proj", "v_proj"]}
    merge.assert_compatible(base, dict(base))  # ok
    with pytest.raises(merge.AdapterMergeError):
        merge.assert_compatible(base, {**base, "r": 32})
    with pytest.raises(merge.AdapterMergeError):
        merge.assert_compatible(base, {**base, "lora_alpha": 256})
    with pytest.raises(merge.AdapterMergeError):
        merge.assert_compatible(base, {**base, "target_modules": ["q_proj"]})


def test_merged_adapter_config_sums_rank_and_alpha():
    cfg = merge.merged_adapter_config(
        {"r": 64, "lora_alpha": 128, "target_modules": ["q"]},
        {"r": 64, "lora_alpha": 128, "target_modules": ["q"]},
    )
    assert cfg["r"] == 128
    assert cfg["lora_alpha"] == 256
    # ratio preserved
    assert cfg["lora_alpha"] / cfg["r"] == 128 / 64
