"""Canonical recipe hyperparameters.

Values are the paper-faithful defaults from Open Character Training
(arXiv 2511.01689, Sections 2.3-2.4). The ``quick`` presets are downscaled for
smoke-testing the pipeline; ``paper`` reproduces the published recipe.

Keeping these in one side-effect-free module means the scaling experiments can
hold the recipe constant while only the model changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Scale = Literal["smoke", "quick", "paper"]
CapabilitySuite = Literal["smoke", "full"]


@dataclass(frozen=True)
class DPOConfig:
    """Stage 2 - distillation via DPO (paper Section 2.3)."""

    lora_rank: int = 64
    # Tinker fixes LoRA alpha at 32 server-side; the API exposes no alpha
    # parameter (verified against a downloaded adapter_config.json — see
    # docs/PAPER_GAP_AUDIT_2026-07-01.md, F1). The paper's alpha=128 (alpha/r=2)
    # is unachievable on Tinker: effective alpha/r is 0.5 at r=64 and 1.0 at
    # Ultra's r=32 cap. This field records the real value for metadata/merge.
    lora_alpha: int = 32
    batch_size: int = 32
    learning_rate: float = 5e-5
    beta: float = 0.1
    nll_coeff: float = 0.1  # NLL term on chosen generations, per Grattafiori/Pang
    # Additional per-token KL penalty on the DPO loss (paper App A). Official
    # impl (maiush/OpenRLHF dpo_trainer.py + distillation launch scripts):
    # per-sequence mean over response tokens of (logp_pi - logp_ref)^2,
    # averaged over BOTH chosen and rejected sequences, coefficient 0.001.
    kl_coeff: float = 0.001
    num_prompts: int = 1500
    # Max total training-data tokens (prompt + chosen + rejected per pair,
    # tokenizer-counted on the real path). Paper App A: ~6M tokens/persona.
    # None = no cap (data sized by num_prompts alone).
    token_budget: int | None = None


@dataclass(frozen=True)
class SFTConfig:
    """Stage 3 - introspection via SFT (paper Section 2.4)."""

    lora_rank: int = 64
    lora_alpha: int = 32  # Tinker-fixed; see DPOConfig.lora_alpha
    batch_size: int = 32
    learning_rate: float = 5e-5
    epochs: int = 1
    # Initialize SFT from the post-DPO weights (fresh optimizer), as the
    # official implementation does (finetuning/introspection/*.sh trains with
    # --pretrain models/distilled/<model>-<persona>). False trains an
    # independent adapter over the base student (the pre-2026-07-06 octt
    # behavior, kept for the A3 ablation). The merge weights depend on this —
    # see octt.merge.concat_weights.
    init_from_dpo: bool = True
    self_reflection_count: int = 10_000  # 10 prompts x 1000 responses
    self_interaction_count: int = 2_000  # greeting-seeded self-chats
    # Generated messages per self-chat (the official K): the two sides
    # alternate, assistant first, so a chat carries ceil(turns/2) assistant turns.
    self_interaction_turns: int = 10
    # Max total transcript tokens. Paper App B.3: ~8M tokens/persona.
    # None = no cap (data sized by the counts alone).
    token_budget: int | None = None


@dataclass(frozen=True)
class EvalConfig:
    """Revealed-preferences evaluation (paper Section 3.1)."""

    num_judgments: int = 25_000
    judge_temperature: float = 0.1
    judge_top_p: float = 0.95
    num_traits: int = 144  # the exact single-word trait list (Appendix G)
    # Responder (model-under-test) sampling params, from the official impl
    # (character/preferences/preferences.py: temp 0.7, top_p 0.95, min_p 0,
    # max_tokens 1024). Pinned here so they are explicit, hashed, and constant
    # across the scaling study. (The official code also sets
    # repetition_penalty=1.1, which Tinker's SamplingParams does not expose.)
    responder_temperature: float = 0.7
    responder_top_p: float = 0.95
    responder_max_tokens: int = 1024
    # Judge emits the winning trait between <answer></answer> tags (official
    # judgements.py protocol); 64 tokens is ample with thinking disabled.
    judge_max_tokens: int = 64


@dataclass(frozen=True)
class CapabilityBenchmark:
    """One LightEval task specification for the capability harness."""

    label: str
    task: str
    fewshot: int
    suite: str = "leaderboard"

    @property
    def spec(self) -> str:
        return f"{self.task}|{self.fewshot}"


@dataclass(frozen=True)
class CapabilityEvalConfig:
    """LightEval capability benchmark plan.

    Kept separate from :class:`RecipeConfig` so opting into the capability phase
    does not perturb training/eval manifest hashes for the OCTT recipe itself.
    """

    name: CapabilitySuite = "smoke"
    benchmarks: tuple[CapabilityBenchmark, ...] = ()
    backend: str = "accelerate"
    max_samples: int | None = None
    remove_reasoning_tags: bool = True
    model_args: tuple[str, ...] = ()


MMLU_SUBJECTS: tuple[str, ...] = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


CAPABILITY_CORE_BENCHMARKS: tuple[CapabilityBenchmark, ...] = (
    CapabilityBenchmark("truthfulqa", "truthfulqa:mc", 0),
    CapabilityBenchmark("winogrande", "winogrande", 5),
    CapabilityBenchmark("hellaswag", "hellaswag", 10),
    CapabilityBenchmark("arc-c", "arc:challenge", 25),
)

CAPABILITY_MMLU_BENCHMARKS: tuple[CapabilityBenchmark, ...] = tuple(
    CapabilityBenchmark(f"mmlu:{subject}", f"mmlu:{subject}", 5)
    for subject in MMLU_SUBJECTS
)

CAPABILITY_SMOKE = CapabilityEvalConfig(
    name="smoke",
    benchmarks=(CAPABILITY_CORE_BENCHMARKS[0],),
    max_samples=8,
)

CAPABILITY_FULL = CapabilityEvalConfig(
    name="full",
    benchmarks=CAPABILITY_CORE_BENCHMARKS + CAPABILITY_MMLU_BENCHMARKS,
)


@dataclass(frozen=True)
class RecipeConfig:
    """The full recipe, held constant across the scaling study."""

    dpo: DPOConfig = field(default_factory=DPOConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    merge_adapters: bool = True  # linearly merge DPO + SFT adapters (paper)


# Tiny preset for validating plumbing before any paid run. The eval uses a small
# trait pool so the persona trait is actually sampled (and a preference shift is
# observable) even at this scale.
SMOKE = RecipeConfig(
    dpo=DPOConfig(batch_size=2, num_prompts=2),
    sft=SFTConfig(
        batch_size=2,
        self_reflection_count=4,
        self_interaction_count=2,
        self_interaction_turns=2,
    ),
    eval=EvalConfig(num_judgments=40, num_traits=8),
)

# Downscaled preset for fast end-to-end smoke tests (~1% of paper scale).
QUICK = RecipeConfig(
    dpo=DPOConfig(num_prompts=32),
    sft=SFTConfig(self_reflection_count=100, self_interaction_count=20),
    eval=EvalConfig(num_judgments=200, num_traits=30),
)

# Paper-faithful preset. Token budgets per App A (~6M) / App B.3 (~8M).
PAPER = RecipeConfig(
    dpo=DPOConfig(token_budget=6_000_000),
    sft=SFTConfig(token_budget=8_000_000),
)


# Scaling-study rank policy (docs/PAPER_GAP_AUDIT_2026-07-01.md, F1 resolution).
# Tinker pins LoRA alpha at 32 server-side, so alpha/r varies with rank while
# the paper trained at alpha/r = 2.0 (rank 64, alpha 128, lr 5e-5): effective
# update scale lr*(alpha/r) = 1e-4. Running every rung at rank 32 (Ultra's
# hard cap) with lr doubled to 1e-4 reproduces that effective scale exactly
# (1e-4 * 1.0) AND holds the adapter recipe constant across all rungs — the
# only option that removes the cross-rung alpha/r confound without dropping
# Ultra. `octt run` / PAPER keep the paper-stated rank 64 for single-model
# replication; an explicit --lora-rank on `octt scaling` overrides this policy
# verbatim (no lr compensation).
SCALING_STUDY_LORA_RANK = 32
SCALING_STUDY_LR = 1e-4


def for_scaling_study(cfg: RecipeConfig) -> RecipeConfig:
    """Apply the uniform-rank scaling-study policy (rank 32, lr 1e-4) to *cfg*."""
    from dataclasses import replace

    return replace(
        cfg,
        dpo=replace(
            cfg.dpo, lora_rank=SCALING_STUDY_LORA_RANK, learning_rate=SCALING_STUDY_LR
        ),
        sft=replace(
            cfg.sft, lora_rank=SCALING_STUDY_LORA_RANK, learning_rate=SCALING_STUDY_LR
        ),
    )


def get_config(scale: Scale = "quick") -> RecipeConfig:
    if scale == "paper":
        return PAPER
    if scale == "smoke":
        return SMOKE
    return QUICK


def get_capability_config(suite: CapabilitySuite = "smoke") -> CapabilityEvalConfig:
    if suite == "full":
        return CAPABILITY_FULL
    return CAPABILITY_SMOKE
