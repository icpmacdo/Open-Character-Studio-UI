"""CLI overrides for the training-strength axis.

A fixed recipe holds rank, lr and epochs constant across model sizes, which makes
the training intervention weaker the larger the model gets: LoRA trainable params
scale as r*d while total params scale as d^2, so the trainable fraction falls as
r/d. These flags are how one rung gets retrained at matched effective strength.
"""

from __future__ import annotations

import argparse

import pytest

from octt.cli import _recipe_config_from_args


def _effective_scale(cfg):
    """Tinker pins lora_alpha=32 server-side, so lr*(alpha/rank) is comparable."""
    return cfg.sft.learning_rate * (32 / cfg.sft.lora_rank)


def test_sft_epochs_flag_scales_only_the_sft_stage():
    """DPO is pinned to one epoch in distillation.py, so the flag must not claim it."""
    cfg = _recipe_config_from_args(
        argparse.Namespace(scale="paper-half-uncapped", sft_epochs=2)
    )
    assert cfg.sft.epochs == 2

    base = _recipe_config_from_args(argparse.Namespace(scale="paper-half-uncapped"))
    assert base.sft.epochs == 1
    # Everything else must be untouched: this is the training-strength axis only.
    assert cfg.sft.lora_rank == base.sft.lora_rank
    assert cfg.sft.learning_rate == base.sft.learning_rate
    assert cfg.sft.token_budget == base.sft.token_budget
    assert cfg.dpo == base.dpo


def test_rank_and_lr_overrides_compose_with_epochs():
    """Arm C of the 27B training-strength probe: rank 64, lr 2e-4, 2 epochs."""
    cfg = _recipe_config_from_args(
        argparse.Namespace(
            scale="paper-half-uncapped", lora_rank=64, learning_rate=2e-4, sft_epochs=2
        )
    )
    assert (cfg.dpo.lora_rank, cfg.sft.lora_rank) == (64, 64)
    assert (cfg.dpo.learning_rate, cfg.sft.learning_rate) == (2e-4, 2e-4)
    assert cfg.sft.epochs == 2
    assert _effective_scale(cfg) == pytest.approx(1e-4)


def test_the_three_arms_differ_in_capacity_not_in_effective_scale():
    """Arms A/B/C vary trainable capacity and optimizer steps, holding lr*(alpha/r)."""
    arm_a = _recipe_config_from_args(
        argparse.Namespace(scale="paper-half-uncapped", lora_rank=32, learning_rate=1e-4)
    )
    arm_b = _recipe_config_from_args(
        argparse.Namespace(scale="paper-half-uncapped", lora_rank=64, learning_rate=2e-4)
    )
    arm_c = _recipe_config_from_args(
        argparse.Namespace(
            scale="paper-half-uncapped", lora_rank=64, learning_rate=2e-4, sft_epochs=2
        )
    )
    assert _effective_scale(arm_a) == pytest.approx(_effective_scale(arm_b))
    assert _effective_scale(arm_b) == pytest.approx(_effective_scale(arm_c))
    assert arm_b.sft.lora_rank == 2 * arm_a.sft.lora_rank
    assert (arm_a.sft.epochs, arm_b.sft.epochs, arm_c.sft.epochs) == (1, 1, 2)
    # The data is identical across all three -- only training strength varies.
    assert arm_a.dpo.num_prompts == arm_b.dpo.num_prompts == arm_c.dpo.num_prompts
    assert arm_a.sft.token_budget == arm_c.sft.token_budget
