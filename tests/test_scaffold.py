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


def test_scaling_triangle_has_both_architectures():
    specs = [models.get(m) for m in models.SCALING_TRIANGLE]
    arches = {s.arch for s in specs}
    assert arches == {"dense", "moe"}


def test_constitutions_dir_exists():
    assert Path("constitutions").is_dir()
