"""Smoke tests for the scaffold: config, constitution loading, model registry."""

from pathlib import Path

from octt import constitution, models
from octt.config import PAPER, SMOKE, for_scaling_study, get_config


def test_paper_config_matches_published_recipe():
    assert PAPER.dpo.lora_rank == 64
    assert PAPER.dpo.beta == 0.1
    assert PAPER.dpo.nll_coeff == 0.1
    assert PAPER.sft.epochs == 1
    assert PAPER.sft.self_reflection_count == 10_000
    assert PAPER.sft.self_interaction_count == 2_000


def test_scaling_study_policy_is_uniform_rank32_with_lr_compensation():
    # Tinker pins alpha=32, so the study runs every rung at rank 32 with lr
    # doubled: lr * (alpha/r) = 1e-4 * 1.0 matches the paper's 5e-5 * 2.0.
    study = for_scaling_study(PAPER)
    assert study.dpo.lora_rank == 32 and study.sft.lora_rank == 32
    assert study.dpo.learning_rate == 1e-4 and study.sft.learning_rate == 1e-4
    # Everything else — data sizing, eval protocol, merge policy — untouched.
    assert study.dpo.token_budget == PAPER.dpo.token_budget
    assert study.eval == PAPER.eval
    assert study.merge_adapters == PAPER.merge_adapters
    # The PAPER preset itself still records the paper-stated rank.
    assert PAPER.dpo.lora_rank == 64


def test_quick_config_is_downscaled():
    quick = get_config("quick")
    assert quick.dpo.num_prompts < PAPER.dpo.num_prompts
    assert quick.sft.self_reflection_count < PAPER.sft.self_reflection_count


def test_smoke_config_is_smallest_scale():
    quick = get_config("quick")
    assert SMOKE.dpo.num_prompts < quick.dpo.num_prompts
    assert SMOKE.sft.self_reflection_count < quick.sft.self_reflection_count
    assert SMOKE.eval.num_judgments < quick.eval.num_judgments


def test_constitution_loads():
    c = constitution.load("humorous")
    assert c.persona == "humorous"
    assert len(c.assertions) == 10  # full App F humorous constitution
    assert all(not a.startswith("-") for a in c.assertions)


def test_constitutions_discoverable():
    assert "humorous" in constitution.available()


def test_dense_ladder_is_all_dense():
    assert all(models.get(m).arch == "dense" for m in models.DENSE_LADDER)


def test_moe_ladder_is_all_moe():
    assert all(models.get(m).arch == "moe" for m in models.MOE_LADDER)


def test_scaling_set_covers_both_ladders():
    assert set(models.SCALING_SET) == set(models.DENSE_LADDER) | set(models.MOE_LADDER)


def test_arch_control_pair_is_dense_vs_moe_same_family():
    dense, moe = (models.get(m) for m in models.ARCH_CONTROL_PAIR)
    assert dense.arch == "dense" and moe.arch == "moe"
    assert dense.family == moe.family == "Qwen"


def test_teacher_is_registered_or_known():
    assert models.TEACHER_MODEL  # a teacher is chosen (not None)


def test_constitutions_dir_exists():
    assert Path("constitutions").is_dir()
