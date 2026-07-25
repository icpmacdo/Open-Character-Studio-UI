"""Token-budget data sizing (paper App A ~6M DPO / App B.3 ~8M SFT)."""

from __future__ import annotations

from octt import distillation, introspection
from octt.config import PAPER, get_config


def test_paper_preset_sets_official_token_budgets():
    assert PAPER.dpo.token_budget == 6_000_000
    assert PAPER.sft.token_budget == 8_000_000
    assert get_config("smoke").dpo.token_budget is None
    assert get_config("quick").sft.token_budget is None


def test_paper_half_preset_halves_budgets_and_keeps_recipe_constants():
    paper = get_config("paper")
    half = get_config("paper-half")
    assert half.dpo.token_budget == paper.dpo.token_budget // 2
    assert half.sft.token_budget == paper.sft.token_budget // 2
    assert half.dpo.num_prompts == paper.dpo.num_prompts // 2
    assert half.sft.self_reflection_count == paper.sft.self_reflection_count // 2
    assert half.sft.self_interaction_count == paper.sft.self_interaction_count // 2
    assert half.eval.num_judgments == paper.eval.num_judgments // 2
    # Everything that is not a data/judgment budget is the paper recipe.
    for field_name in ("lora_rank", "batch_size", "learning_rate", "beta", "nll_coeff", "kl_coeff"):
        assert getattr(half.dpo, field_name) == getattr(paper.dpo, field_name)
    for field_name in ("lora_rank", "batch_size", "learning_rate", "epochs", "init_from_dpo",
                       "self_interaction_turns"):
        assert getattr(half.sft, field_name) == getattr(paper.sft, field_name)
    for field_name in ("num_traits", "judge_max_tokens", "responder_max_tokens",
                       "judge_temperature", "responder_temperature"):
        assert getattr(half.eval, field_name) == getattr(paper.eval, field_name)


def test_paper_preset_uses_official_loss_coefficients():
    assert PAPER.dpo.kl_coeff == 0.001  # official fork's --kl_loss_coef
    assert PAPER.dpo.nll_coeff == 0.1
    assert PAPER.dpo.lora_alpha == 32  # Tinker-fixed; see PAPER_GAP_AUDIT F1


def test_transcript_token_budget_caps_and_is_deterministic():
    transcripts = [
        [{"role": "user", "content": "q" * 400}, {"role": "assistant", "content": "a" * 400}]
        for _ in range(20)
    ]
    kept, total, dropped = introspection._apply_token_budget(
        transcripts, 1000, "Qwen/Qwen3.5-4B", offline=True
    )
    assert dropped > 0
    assert total <= 1000
    again, total2, dropped2 = introspection._apply_token_budget(
        transcripts, 1000, "Qwen/Qwen3.5-4B", offline=True
    )
    assert (total, dropped) == (total2, dropped2)
    assert [id(t) for t in kept] == [id(t) for t in again]


def test_transcript_token_budget_noop_when_unset_or_roomy():
    transcripts = [[{"role": "user", "content": "hi"}]]
    kept, total, dropped = introspection._apply_token_budget(
        transcripts, None, "Qwen/Qwen3.5-4B", offline=True
    )
    assert kept == transcripts and dropped == 0
    kept, total, dropped = introspection._apply_token_budget(
        transcripts, 10_000, "Qwen/Qwen3.5-4B", offline=True
    )
    assert kept == transcripts and dropped == 0


def test_pair_token_budget_caps_and_preserves_order():
    n = 12
    prompts = [f"prompt {i} " + "x" * 200 for i in range(n)]
    chosen = ["c" * 400 for _ in range(n)]
    rejected = ["r" * 400 for _ in range(n)]
    usable = list(range(n))
    kept, total, dropped = distillation._apply_pair_token_budget(
        usable, prompts, chosen, rejected, 800, "Qwen/Qwen3.5-4B", offline=True
    )
    assert dropped > 0
    assert total <= 800
    assert kept == sorted(kept)  # original order preserved
