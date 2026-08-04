"""Tinker client setup, renderer selection, and spend preflight helpers.

This module is intentionally import-safe without ``tinker`` installed. Anything
that needs the paid runtime or tokenizer stack is imported lazily so tests and
planning commands can still run from a fresh checkout with the vendored
``tinker-cookbook`` only.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from . import models
from .config import RecipeConfig, get_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COOKBOOK_PATH = PROJECT_ROOT / "tinker-cookbook"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs"

# Pessimistic fallback for teachers outside the registry (the study teacher and
# Inkling are now priced in models.CANDIDATES, so this rarely applies).
TEACHER_SAMPLE_PRICE_USD_PER_MTOK = 7.5

# Inkling's tml_v0 renderer has no disable-thinking variant; thinking is a float
# effort in [0.0, 1.0) rendered as a system-level directive. OCTT pins it to the
# minimum on BOTH the generation and supervised-rendering paths via a registered
# subclass (see _register_pinned_effort_renderer): persisted training text
# carries visible answers only, so both sampling and training must condition on
# the effort level whose native behavior that text approximates.
TML_PINNED_EFFORT = 0.0
TML_PINNED_RENDERER_NAME = "octt/tml_v0_pinned_effort"

DIRECT_ANSWER_RENDERER_OVERRIDES = {
    # The cookbook's default Qwen3.5 renderer enables thinking. OCTT persists
    # sampled completions as supervised/preference training text, so prefer the
    # direct-answer variant to keep reasoning traces out of the dataset.
    "qwen3_5": "qwen3_5_disable_thinking",
    # Nemotron-3's recommended renderers are full-reasoning; without these the
    # MoE ladder would run thinking-ON against the thinking-OFF Qwen ladder and
    # the dense-vs-MoE Elo would measure renderer mode, not character.
    "nemotron3": "nemotron3_disable_thinking",
    "nemotron3_ultra": "nemotron3_ultra_disable_thinking",
    # Inkling: pin thinking effort to its minimum (same policy, different knob).
    "tml_v0": TML_PINNED_RENDERER_NAME,
}


# The vendored cookbook predates Inkling-Small: model_info and tokenizer_utils
# both exact-match the full "thinkingmachines/Inkling" id, so every other model
# in the org falls through to "unknown model" / an HF-hub tokenizer load. All
# TML models render via tml_v0 on the o200k chat tokenizer, so octt bridges the
# gap here instead of patching the read-only vendor. The renderer fallback only
# fires when the vendored lookup raises, so a future re-vendor that learns the
# org (or ships per-model renderers) takes precedence automatically.
TML_ORG = "thinkingmachines"


def _is_tml_org(model_id: str) -> bool:
    return model_id.split(":", 1)[0].split("/", 1)[0] == TML_ORG


def _recommended_renderer_name(model_id: str, vendored_lookup: Any) -> str:
    try:
        return vendored_lookup(model_id)
    except Exception:
        if _is_tml_org(model_id):
            return "tml_v0"
        raise


def _tml_aware_get_tokenizer(model_name: str) -> Any:
    tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
    if _is_tml_org(model_name) and not tokenizer_utils.is_tokenizer_registered(model_name):
        # Same facade the vendored get_tokenizer returns for the full Inkling id.
        return tokenizer_utils.TmlRenderersTokenizerAdapter(model_name)
    return tokenizer_utils.get_tokenizer(model_name)


def _register_tml_tokenizers(tokenizer_utils: ModuleType) -> None:
    """Register TML-org registry models in the vendored custom-tokenizer registry.

    Cookbook dataset builders (DPO/SFT) resolve renderers by NAME and call the
    vendored ``get_tokenizer(model_name)`` themselves, so the stack-level
    wrapper above never reaches them — for Inkling-Small that path fell through
    to an HF AutoTokenizer load the tml renderer rejects (2026-07-30 smoke
    crash). The registry is checked first on every by-name lookup, which makes
    it the one hook that covers all call sites without patching the vendor.
    """
    for model_id in models.CANDIDATES:
        if model_id.split("/", 1)[0] != TML_ORG:
            continue
        if tokenizer_utils.is_tokenizer_registered(model_id):
            continue
        tokenizer_utils.register_tokenizer(
            model_id,
            lambda model_id=model_id: tokenizer_utils.TmlRenderersTokenizerAdapter(model_id),
        )


def renderer_supports_think_prefill(renderer_name: str) -> bool:
    """Whether App A's ``<think>`` prefill trick can ride this renderer.

    tml_v0 (Inkling) rejects partial assistant messages outright, and the
    Inkling self-distillation design wants teacher and student sampled at the
    same pinned effort anyway — the teacher samples exactly like the student
    and the prefill is skipped (INKLING_PLAN.md, Phase 0.3).
    """
    return renderer_name not in ("tml_v0", TML_PINNED_RENDERER_NAME)


class TinkerSetupError(RuntimeError):
    """Raised when Tinker or the cookbook cannot be initialized."""


@dataclass(frozen=True)
class TinkerClientConfig:
    dry_run: bool = False
    base_url: str | None = None
    cookbook_path: Path = DEFAULT_COOKBOOK_PATH
    api_key_env: str = "TINKER_API_KEY"


@dataclass(frozen=True)
class RendererPlan:
    model_id: str
    renderer_name: str


@dataclass(frozen=True)
class RendererBinding:
    model_id: str
    renderer_name: str
    tokenizer: Any
    renderer: Any


@dataclass(frozen=True)
class TinkerStack:
    tinker: ModuleType
    renderers: ModuleType
    get_tokenizer: Any
    get_recommended_renderer_name: Any


@dataclass(frozen=True)
class DryRunSampleSequence:
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class DryRunSample:
    sequences: list[DryRunSampleSequence] = field(
        default_factory=lambda: [DryRunSampleSequence()]
    )


class DryRunSamplingClient:
    """No-op sampling client with the async shape used by Tinker recipes."""

    async def sample_async(self, *args: Any, **kwargs: Any) -> DryRunSample:
        return DryRunSample()


@dataclass(frozen=True)
class TinkerRuntime:
    config: TinkerClientConfig
    service_client: Any | None
    renderer_bindings: dict[str, RendererBinding]
    renderer_plans: dict[str, RendererPlan]
    # Thinking-enabled bindings (no direct-answer override), used ONLY for the
    # teacher's chosen-generation with the App A `<think>` prefill. Populated
    # just for models whose recommended renderer differs from the override.
    thinking_renderer_bindings: dict[str, RendererBinding] = field(default_factory=dict)

    def create_sampling_client(
        self, base_model: str, model_path: str | None = None
    ) -> Any:
        if self.config.dry_run:
            return DryRunSamplingClient()
        if self.service_client is None:
            raise TinkerSetupError("Tinker service client is not initialized")
        return self.service_client.create_sampling_client(
            base_model=base_model,
            model_path=model_path,
        )

    def require_service_client(self) -> Any:
        if self.config.dry_run:
            raise TinkerSetupError("Dry-run runtime has no Tinker service client")
        if self.service_client is None:
            raise TinkerSetupError("Tinker service client is not initialized")
        return self.service_client

    def renderer_plan(self, model_id: str) -> RendererPlan:
        return self.renderer_plans[model_id]

    def renderer_binding(self, model_id: str) -> RendererBinding:
        if self.config.dry_run:
            raise TinkerSetupError("Dry-run runtime has renderer plans but no tokenizer bindings")
        return self.renderer_bindings[model_id]

    def thinking_renderer_binding(self, model_id: str) -> RendererBinding:
        """The recommended (thinking-enabled) renderer for *model_id*.

        Falls back to the default binding when the model has no separate
        thinking variant (i.e. no direct-answer override applied).
        """
        if self.config.dry_run:
            raise TinkerSetupError("Dry-run runtime has renderer plans but no tokenizer bindings")
        return self.thinking_renderer_bindings.get(model_id) or self.renderer_bindings[model_id]


@dataclass(frozen=True)
class CostEstimateLine:
    stage: str
    model_id: str
    token_millions: float
    unit_price_usd: float
    subtotal_usd: float


@dataclass(frozen=True)
class CostEstimate:
    lines: tuple[CostEstimateLine, ...]

    @property
    def total_usd(self) -> float:
        return sum(line.subtotal_usd for line in self.lines)


@dataclass(frozen=True)
class PreflightReport:
    dry_run: bool
    api_key_set: bool
    cookbook_path: Path
    output_dir: Path
    renderer_plans: tuple[RendererPlan, ...]
    cost_estimate: CostEstimate
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.blockers


def local_cookbook_path(root: Path = PROJECT_ROOT) -> Path:
    return root / "tinker-cookbook"


def ensure_cookbook_on_path(cookbook_path: Path = DEFAULT_COOKBOOK_PATH) -> Path:
    resolved = cookbook_path.resolve()
    package_dir = resolved / "tinker_cookbook"
    if not package_dir.is_dir():
        raise TinkerSetupError(f"tinker_cookbook package not found at {resolved}")
    path_str = str(resolved)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return resolved


def require_api_key(
    env: Mapping[str, str] = os.environ,
    api_key_env: str = "TINKER_API_KEY",
) -> str:
    api_key = env.get(api_key_env, "").strip()
    if not api_key:
        raise TinkerSetupError(f"{api_key_env} is required for non-dry-run Tinker calls")
    return api_key


def import_model_info(cookbook_path: Path = DEFAULT_COOKBOOK_PATH) -> ModuleType:
    ensure_cookbook_on_path(cookbook_path)
    try:
        return importlib.import_module("tinker_cookbook.model_info")
    except ImportError as exc:
        raise TinkerSetupError(
            "Could not import tinker_cookbook.model_info from the vendored cookbook"
        ) from exc


def resolve_renderer_name(model_id: str, cookbook_path: Path = DEFAULT_COOKBOOK_PATH) -> str:
    model_info = import_model_info(cookbook_path)
    try:
        recommended = _recommended_renderer_name(
            model_id, model_info.get_recommended_renderer_name
        )
        return DIRECT_ANSWER_RENDERER_OVERRIDES.get(recommended, recommended)
    except Exception as exc:
        raise TinkerSetupError(f"Could not resolve renderer for {model_id!r}: {exc}") from exc


def plan_renderers(
    model_ids: Sequence[str],
    cookbook_path: Path = DEFAULT_COOKBOOK_PATH,
) -> tuple[RendererPlan, ...]:
    seen: set[str] = set()
    plans: list[RendererPlan] = []
    for model_id in model_ids:
        if model_id in seen:
            continue
        seen.add(model_id)
        plans.append(
            RendererPlan(
                model_id=model_id,
                renderer_name=resolve_renderer_name(model_id, cookbook_path),
            )
        )
    return tuple(plans)


def _register_pinned_effort_renderer(renderers: ModuleType) -> None:
    """Idempotently register the pinned-effort tml_v0 subclass.

    Registering it under :data:`TML_PINNED_RENDERER_NAME` lets the name flow
    through cookbook dataset builders (DPO/SFT training render conversations by
    renderer *name*), so training-time rendering carries the same effort
    directive as generation. The caller-facing ``effort`` argument is
    deliberately ignored: OCTT's renderer policy is a recipe constant, not a
    per-call knob.
    """
    if renderers.is_renderer_registered(TML_PINNED_RENDERER_NAME):
        return
    tml_v0 = importlib.import_module("tinker_cookbook.renderers.tml_v0")
    base = importlib.import_module("tinker_cookbook.renderers.base")
    all_assistant = base.TrainOnWhat.ALL_ASSISTANT_MESSAGES

    class PinnedEffortTmlV0Renderer(tml_v0.TmlV0Renderer):
        """tml_v0 with thinking effort pinned to TML_PINNED_EFFORT everywhere."""

        def build_generation_prompt(  # type: ignore[override]
            self, messages, role="assistant", prefill=None, effort=None
        ):
            return super().build_generation_prompt(
                messages, role, prefill, effort=TML_PINNED_EFFORT
            )

        def build_supervised_examples(  # type: ignore[override]
            self, messages, train_on_what=all_assistant, effort=None
        ):
            return super().build_supervised_examples(
                messages, train_on_what, effort=TML_PINNED_EFFORT
            )

        def build_supervised_example(  # type: ignore[override]
            self, messages, train_on_what=all_assistant, effort=None
        ):
            return super().build_supervised_example(
                messages, train_on_what, effort=TML_PINNED_EFFORT
            )

    def _factory(tokenizer: Any, image_processor: Any = None) -> Any:
        return PinnedEffortTmlV0Renderer(tokenizer)

    renderers.register_renderer(TML_PINNED_RENDERER_NAME, _factory)


def import_tinker_stack(cookbook_path: Path = DEFAULT_COOKBOOK_PATH) -> TinkerStack:
    ensure_cookbook_on_path(cookbook_path)
    try:
        tinker = importlib.import_module("tinker")
        renderers = importlib.import_module("tinker_cookbook.renderers")
        tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
        model_info = importlib.import_module("tinker_cookbook.model_info")
    except ImportError as exc:
        raise TinkerSetupError(
            "Could not import the Tinker runtime stack. Install it with "
            '`pip install -e ".[train]"` and keep the vendored tinker-cookbook '
            "available for renderer metadata."
        ) from exc

    _register_pinned_effort_renderer(renderers)
    _register_tml_tokenizers(tokenizer_utils)
    return TinkerStack(
        tinker=tinker,
        renderers=renderers,
        get_tokenizer=_tml_aware_get_tokenizer,
        get_recommended_renderer_name=lambda model_id: _recommended_renderer_name(
            model_id, model_info.get_recommended_renderer_name
        ),
    )


def resolve_renderer_binding(
    model_id: str, stack: TinkerStack, *, thinking: bool = False
) -> RendererBinding:
    recommended = stack.get_recommended_renderer_name(model_id)
    renderer_name = (
        recommended if thinking else DIRECT_ANSWER_RENDERER_OVERRIDES.get(recommended, recommended)
    )
    tokenizer = stack.get_tokenizer(model_id)
    renderer = stack.renderers.get_renderer(renderer_name, tokenizer, model_name=model_id)
    return RendererBinding(
        model_id=model_id,
        renderer_name=renderer_name,
        tokenizer=tokenizer,
        renderer=renderer,
    )


def create_runtime(
    model_ids: Sequence[str],
    config: TinkerClientConfig | None = None,
) -> TinkerRuntime:
    cfg = config or TinkerClientConfig()
    plans = {
        plan.model_id: plan
        for plan in plan_renderers(model_ids, cookbook_path=cfg.cookbook_path)
    }

    if cfg.dry_run:
        return TinkerRuntime(
            config=cfg,
            service_client=None,
            renderer_bindings={},
            renderer_plans=plans,
        )

    require_api_key(api_key_env=cfg.api_key_env)
    stack = import_tinker_stack(cfg.cookbook_path)
    service_client = stack.tinker.ServiceClient(base_url=cfg.base_url)
    bindings = {
        model_id: resolve_renderer_binding(model_id, stack)
        for model_id in plans
    }
    thinking_bindings = {}
    for model_id, binding in bindings.items():
        thinking = resolve_renderer_binding(model_id, stack, thinking=True)
        if thinking.renderer_name != binding.renderer_name:
            thinking_bindings[model_id] = thinking
    return TinkerRuntime(
        config=cfg,
        service_client=service_client,
        renderer_bindings=bindings,
        renderer_plans=plans,
        thinking_renderer_bindings=thinking_bindings,
    )


def _price_for(model_id: str, price_kind: str) -> float | None:
    spec = models.CANDIDATES.get(model_id)
    if spec is None:
        return None
    return getattr(spec, price_kind)


def _append_cost_line(
    lines: list[CostEstimateLine],
    stage: str,
    model_id: str,
    token_count: int,
    unit_price_usd: float | None,
) -> None:
    token_millions = token_count / 1_000_000
    price = unit_price_usd or 0.0
    lines.append(
        CostEstimateLine(
            stage=stage,
            model_id=model_id,
            token_millions=token_millions,
            unit_price_usd=price,
            subtotal_usd=token_millions * price,
        )
    )


def estimate_tinker_cost(
    config: RecipeConfig,
    student_models: Sequence[str],
    teacher_model: str = models.TEACHER_MODEL,
    *,
    dpo_sample_tokens: int = 1024,
    dpo_teacher_sample_tokens: int = 2048,  # thinking trace + visible answer
    dpo_prompt_tokens: int = 1024,  # constitution system prompt + user prompt
    reflection_sample_tokens: int = 2048,  # official self-reflection envelope
    interaction_sample_tokens: int = 1024,  # official per-message envelope
    eval_sample_tokens: int | None = None,
    eval_prompt_tokens: int = 1024,  # embody system prompt + WildChat prompt
    judge_sample_tokens: int | None = None,
    eval_conditions: int = 1,
    judge_model: str | None = None,
) -> CostEstimate:
    """Estimate billed Tinker spend using max-token stage envelopes.

    Deliberately PESSIMISTIC for budget gates (docs/COST_CONTROLS.md): every
    sampled call is costed at its max-token envelope, prompt/prefill tokens are
    included at ``price_prefill``, a self-interaction chat bills ``turns``
    generations (one per generated message, official K semantics), and SFT
    training tokens account for the last-assistant prefix explosion (a chat
    with A assistant turns trains once per assistant turn, re-billing its
    prefixes ~(A+1)/2 times). Token budgets, when set, cap the corresponding
    training-data lines. ``eval_conditions`` accounts for the full repeated
    judgment budget when more than one embodiment condition is requested.
    """

    if eval_conditions < 1:
        raise ValueError("eval_conditions must be at least 1")
    lines: list[CostEstimateLine] = []
    eval_sample_tokens = (
        config.eval.responder_max_tokens
        if eval_sample_tokens is None
        else eval_sample_tokens
    )
    judge_sample_tokens = (
        config.eval.judge_max_tokens
        if judge_sample_tokens is None
        else judge_sample_tokens
    )
    turns = config.sft.self_interaction_turns
    reflections = config.sft.self_reflection_count
    interactions = config.sft.self_interaction_count

    dpo_sampled_teacher = config.dpo.num_prompts * dpo_teacher_sample_tokens
    dpo_sampled_student = config.dpo.num_prompts * dpo_sample_tokens
    dpo_prefill = config.dpo.num_prompts * dpo_prompt_tokens
    dpo_train_tokens = config.dpo.num_prompts * (dpo_prompt_tokens + 2 * dpo_sample_tokens)
    if config.dpo.token_budget is not None:
        dpo_train_tokens = min(dpo_train_tokens, config.dpo.token_budget)

    # Each self-chat samples exactly `turns` messages, the two sides
    # alternating (assistant first, official K semantics).
    chat_generations = max(1, turns)
    introspection_sampled = (
        reflections * reflection_sample_tokens
        + interactions * chat_generations * interaction_sample_tokens
    )
    # Prefill grows with the conversation; envelope: system (~1k) plus the
    # rolling history, averaged at half the final transcript length.
    chat_history_tokens = chat_generations * interaction_sample_tokens
    introspection_prefill = (
        reflections * 1024
        + interactions * chat_generations * (1024 + chat_history_tokens // 2)
    )
    transcript_tokens = (
        reflections * reflection_sample_tokens + interactions * chat_history_tokens
    )
    if config.sft.token_budget is not None:
        transcript_tokens = min(transcript_tokens, config.sft.token_budget)
    # Last-assistant prefix explosion: a chat with A = ceil(turns/2) assistant
    # turns becomes A examples over growing prefixes, ~(A+1)/2 x the transcript
    # tokens; reflections are 1:1 (the factor is applied to the whole pool for
    # pessimism).
    assistant_turns = max(1, (turns + 1) // 2)
    sft_train_tokens = int(transcript_tokens * max(1, (assistant_turns + 1) / 2))

    # The revealed-preferences eval runs once on the base model and once on the
    # trained target for each student model (per condition).
    eval_responses = 2 * config.eval.num_judgments * eval_conditions
    eval_sampled = eval_responses * eval_sample_tokens
    eval_prefill = eval_responses * eval_prompt_tokens
    judge_sampled = eval_responses * judge_sample_tokens
    judge_prefill = eval_responses * (eval_sample_tokens + 256)  # response + template

    teacher_sample_price = (
        _price_for(teacher_model, "price_sample") or TEACHER_SAMPLE_PRICE_USD_PER_MTOK
    )
    teacher_prefill_price = _price_for(teacher_model, "price_prefill") or teacher_sample_price

    # The judge defaults to the teacher (paper convention) but can be a cheaper
    # model — the eval.judge lines dominate paper-scale spend.
    judge = judge_model or teacher_model
    judge_sample_price = (
        _price_for(judge, "price_sample") or TEACHER_SAMPLE_PRICE_USD_PER_MTOK
    )
    judge_prefill_price = _price_for(judge, "price_prefill") or judge_sample_price

    for model_id in student_models:
        student_sample_price = _price_for(model_id, "price_sample")
        student_prefill_price = _price_for(model_id, "price_prefill") or student_sample_price
        _append_cost_line(
            lines, "dpo.teacher_sample", teacher_model, dpo_sampled_teacher, teacher_sample_price
        )
        _append_cost_line(
            lines, "dpo.teacher_prefill", teacher_model, dpo_prefill, teacher_prefill_price
        )
        _append_cost_line(
            lines, "dpo.student_rejected_sample", model_id, dpo_sampled_student,
            student_sample_price,
        )
        _append_cost_line(
            lines, "dpo.student_prefill", model_id, dpo_prefill, student_prefill_price
        )
        _append_cost_line(
            lines, "dpo.train", model_id, dpo_train_tokens, _price_for(model_id, "price_train")
        )
        _append_cost_line(
            lines, "introspection.sample", model_id, introspection_sampled, student_sample_price
        )
        _append_cost_line(
            lines, "introspection.prefill", model_id, introspection_prefill,
            student_prefill_price,
        )
        _append_cost_line(
            lines, "introspection.sft_train", model_id, sft_train_tokens,
            _price_for(model_id, "price_train"),
        )
        _append_cost_line(
            lines, "eval.model_sample", model_id, eval_sampled, student_sample_price
        )
        _append_cost_line(
            lines, "eval.model_prefill", model_id, eval_prefill, student_prefill_price
        )
        _append_cost_line(
            lines, "eval.judge", judge, judge_sampled, judge_sample_price
        )
        _append_cost_line(
            lines, "eval.judge_prefill", judge, judge_prefill, judge_prefill_price
        )
    return CostEstimate(lines=tuple(lines))


def _output_dir_warnings(output_dir: Path) -> list[str]:
    parent = output_dir if output_dir.exists() else output_dir.parent
    if output_dir.exists() and not output_dir.is_dir():
        return [f"Output path is not a directory: {output_dir}"]
    if not parent.exists():
        return [f"Output parent directory does not exist: {parent}"]
    if not os.access(parent, os.W_OK):
        return [f"Output directory parent is not writable: {parent}"]
    return []


def _missing_price_warnings(
    student_models: Sequence[str],
    teacher_model: str,
) -> list[str]:
    warnings: list[str] = []
    for model_id in student_models:
        spec = models.CANDIDATES.get(model_id)
        if spec is None:
            warnings.append(f"{model_id} is not in octt.models; cost/context metadata unavailable")
            continue
        if spec.price_sample is None:
            warnings.append(f"{model_id} is missing sample price metadata")
        if spec.price_train is None:
            warnings.append(f"{model_id} is missing train price metadata")
    if teacher_model not in models.CANDIDATES and teacher_model != models.TEACHER_MODEL:
        warnings.append(
            f"{teacher_model} is not in octt.models; using "
            f"${TEACHER_SAMPLE_PRICE_USD_PER_MTOK:.2f}/M sampled tokens for estimates"
        )
    return warnings


def validate_lora_rank_limits(
    student_models: Sequence[str],
    config: RecipeConfig,
) -> list[str]:
    """Return blockers for known Tinker LoRA-rank caps."""
    blockers: list[str] = []
    for model_id in student_models:
        spec = models.CANDIDATES.get(model_id)
        if spec is None or spec.max_lora_rank is None:
            continue
        limit = spec.max_lora_rank
        too_high = [
            f"{stage} rank {rank}"
            for stage, rank in (("DPO", config.dpo.lora_rank), ("SFT", config.sft.lora_rank))
            if rank > limit
        ]
        if too_high:
            used = " and ".join(too_high)
            blockers.append(
                f"{model_id} max LoRA rank is {limit}, but config uses {used}"
            )
    return blockers


def validate_merge_feasibility(
    student_models: Sequence[str],
    config: RecipeConfig,
) -> list[str]:
    """Return blockers for models whose base weights cannot merge locally.

    Distinct from :func:`_merge_disk_warnings` (a disk-space *warning* for
    large-but-feasible rungs): a model flagged ``local_merge_feasible=False``
    (e.g. Inkling, 975B total) can never complete the local merge stage, so a
    merge-enabled recipe is a hard blocker, not a warning.
    """
    if not config.merge_adapters:
        return []
    blockers: list[str] = []
    for model_id in student_models:
        spec = models.CANDIDATES.get(model_id)
        if spec is not None and not spec.local_merge_feasible:
            blockers.append(
                f"{model_id} base weights ({spec.total_params_b:.0f}B total) cannot be "
                "merged locally; run with --no-merge"
            )
    return blockers


def _existing_parent(path: Path) -> Path:
    parent = path if path.exists() else path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent


def _merge_disk_warnings(
    student_models: Sequence[str],
    config: RecipeConfig,
    output_dir: Path,
) -> list[str]:
    if not config.merge_adapters:
        return []
    large_models = [
        spec
        for model_id in student_models
        if (spec := models.CANDIDATES.get(model_id)) is not None
        and (spec.tier.lower() == "large" or spec.total_params_b >= 100)
    ]
    if not large_models:
        return []

    names = ", ".join(spec.tinker_id.split("/")[-1] for spec in large_models)
    warning = (
        "Local merge downloads both DPO and SFT adapters for large rungs "
        f"({names}) and can exhaust disk; use --no-merge to evaluate the SFT "
        "sampler directly while keeping the paid training/eval path live."
    )
    try:
        parent = _existing_parent(output_dir)
        free_gib = shutil.disk_usage(parent).free / (1024 ** 3)
        warning += f" Free space near {parent}: {free_gib:.1f} GiB."
    except OSError:
        pass
    return [warning]


def build_preflight_report(
    *,
    student_models: Sequence[str] = models.SCALING_SET,
    teacher_model: str = models.TEACHER_MODEL,
    config: RecipeConfig | None = None,
    dry_run: bool = False,
    budget_usd: float | None = None,
    eval_conditions: int = 1,
    judge_model: str | None = None,
    cookbook_path: Path = DEFAULT_COOKBOOK_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    env: Mapping[str, str] = os.environ,
    api_key_env: str = "TINKER_API_KEY",
) -> PreflightReport:
    cfg = config or get_config("smoke")
    blockers: list[str] = []
    warnings: list[str] = []
    renderer_plans: tuple[RendererPlan, ...] = ()

    api_key_set = bool(env.get(api_key_env, "").strip())
    if not dry_run and not api_key_set:
        blockers.append(f"{api_key_env} is required for non-dry-run Tinker calls")

    try:
        renderer_plans = plan_renderers(
            (teacher_model, *student_models),
            cookbook_path=cookbook_path,
        )
    except TinkerSetupError as exc:
        blockers.append(str(exc))

    warnings.extend(_missing_price_warnings(student_models, teacher_model))
    warnings.extend(_output_dir_warnings(output_dir))
    warnings.extend(_merge_disk_warnings(student_models, cfg, output_dir))
    blockers.extend(validate_lora_rank_limits(student_models, cfg))
    blockers.extend(validate_merge_feasibility(student_models, cfg))

    estimate = estimate_tinker_cost(
        cfg,
        student_models,
        teacher_model,
        eval_conditions=eval_conditions,
        judge_model=judge_model,
    )
    if budget_usd is not None and estimate.total_usd > budget_usd:
        blockers.append(
            f"Estimated spend ${estimate.total_usd:.2f} exceeds budget ${budget_usd:.2f}"
        )

    return PreflightReport(
        dry_run=dry_run,
        api_key_set=api_key_set,
        cookbook_path=cookbook_path,
        output_dir=output_dir,
        renderer_plans=renderer_plans,
        cost_estimate=estimate,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )
