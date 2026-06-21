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
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from . import models
from .config import RecipeConfig, get_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COOKBOOK_PATH = PROJECT_ROOT / "tinker-cookbook"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs"

# The teacher is outside the local student registry. Keep the estimate explicit
# until the full Tinker catalog/price sheet is wired in.
TEACHER_SAMPLE_PRICE_USD_PER_MTOK = 5.0

DIRECT_ANSWER_RENDERER_OVERRIDES = {
    # The cookbook's default Qwen3.5 renderer enables thinking. OCTT persists
    # sampled completions as supervised/preference training text, so prefer the
    # direct-answer variant to keep reasoning traces out of the dataset.
    "qwen3_5": "qwen3_5_disable_thinking",
}


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
        recommended = model_info.get_recommended_renderer_name(model_id)
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

    return TinkerStack(
        tinker=tinker,
        renderers=renderers,
        get_tokenizer=tokenizer_utils.get_tokenizer,
        get_recommended_renderer_name=model_info.get_recommended_renderer_name,
    )


def resolve_renderer_binding(model_id: str, stack: TinkerStack) -> RendererBinding:
    recommended = stack.get_recommended_renderer_name(model_id)
    renderer_name = DIRECT_ANSWER_RENDERER_OVERRIDES.get(recommended, recommended)
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
    return TinkerRuntime(
        config=cfg,
        service_client=service_client,
        renderer_bindings=bindings,
        renderer_plans=plans,
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
    introspection_sample_tokens: int = 512,
    eval_sample_tokens: int = 512,
) -> CostEstimate:
    """Estimate billed Tinker spend using max-token stage envelopes.

    The estimate is deliberately simple and pessimistic enough for budget gates:
    sampled-token stages use sampling prices, and fine-tuning stages use the
    training prices registered in ``octt.models``.
    """

    lines: list[CostEstimateLine] = []
    dpo_prompt_tokens = config.dpo.num_prompts * dpo_sample_tokens
    introspection_generations = (
        config.sft.self_reflection_count
        + config.sft.self_interaction_count * config.sft.self_interaction_turns
    )
    introspection_tokens = introspection_generations * introspection_sample_tokens
    # The revealed-preferences eval runs once on the base model and once on the
    # trained target for each student model.
    eval_tokens_per_model = 2 * config.eval.num_judgments * eval_sample_tokens

    for model_id in student_models:
        _append_cost_line(
            lines,
            "dpo.teacher_sample",
            teacher_model,
            dpo_prompt_tokens,
            _price_for(teacher_model, "price_sample") or TEACHER_SAMPLE_PRICE_USD_PER_MTOK,
        )
        _append_cost_line(
            lines,
            "dpo.student_rejected_sample",
            model_id,
            dpo_prompt_tokens,
            _price_for(model_id, "price_sample"),
        )
        _append_cost_line(
            lines,
            "dpo.train",
            model_id,
            dpo_prompt_tokens * 2,
            _price_for(model_id, "price_train"),
        )
        _append_cost_line(
            lines,
            "introspection.sample",
            model_id,
            introspection_tokens,
            _price_for(model_id, "price_sample"),
        )
        _append_cost_line(
            lines,
            "introspection.sft_train",
            model_id,
            introspection_tokens,
            _price_for(model_id, "price_train"),
        )
        _append_cost_line(
            lines,
            "eval.model_sample",
            model_id,
            eval_tokens_per_model,
            _price_for(model_id, "price_sample"),
        )
        _append_cost_line(
            lines,
            "eval.judge",
            teacher_model,
            eval_tokens_per_model,
            _price_for(teacher_model, "price_sample") or TEACHER_SAMPLE_PRICE_USD_PER_MTOK,
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

    estimate = estimate_tinker_cost(cfg, student_models, teacher_model)
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
