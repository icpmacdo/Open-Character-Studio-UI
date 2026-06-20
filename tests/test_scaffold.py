"""Smoke tests for the scaffold: config, constitution loading, model registry."""

from pathlib import Path

from octt import constitution, models
from octt.config import PAPER, get_config


def test_paper_config_matches_published_recipe():
    assert PAPER.dpo.lora_rank == 64
    assert PAPER.dpo.beta == 0.1
    assert PAPER.dpo.nll_coeff == 0.1
    assert PAPER.sft.epochs == 1
    assert PAPER.sft.self_reflection_count == 10_000
    assert PAPER.sft.self_interaction_count == 2_000


def test_quick_config_is_downscaled():
    quick = get_config("quick")
    assert quick.dpo.num_prompts < PAPER.dpo.num_prompts
    assert quick.sft.self_reflection_count < PAPER.sft.self_reflection_count


def test_constitution_loads():
    c = constitution.load("humorous")
    assert c.persona == "humorous"
    assert len(c.assertions) == 3
    assert all(not a.startswith("-") for a in c.assertions)


def test_constitutions_discoverable():
    assert "humorous" in constitution.available()


def test_dense_ladder_is_all_dense():
    assert all(models.get(m).arch == "dense" for m in models.DENSE_LADDER)


def test_moe_ladder_is_all_moe():
    assert all(models.get(m).arch == "moe" for m in models.MOE_LADDER)


def test_scaling_set_covers_both_ladders():
    assert set(models.SCALING_SET) == set(models.DENSE_LADDER) | set(models.MOE_LADDER)


def test_constitutions_dir_exists():
    assert Path("constitutions").is_dir()
