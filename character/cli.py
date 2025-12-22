"""
character/cli.py
The unified entry point for Open Character Studio.

Paper-recommended defaults from "Open Character Training":
- DPO: LoRA rank 64 (α=128), batch 32, LR 5e-5, β=0.1, NLL coeff 0.1
- Introspection SFT: LoRA rank 64 (α=128), batch 32, LR 5e-5
- Generation: temperature 0.7, top_p 0.95
- Introspection: 10k reflections + 2k interactions (10 turns each)

Use --paper-scale or set CHARACTER_PAPER_SCALE=1 for full paper-compliant values.
"""
import os
import json
import math
import random
from datetime import datetime
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Optional
from pathlib import Path

# Import constants - these respect CHARACTER_PAPER_SCALE env var
from character.constants import (
    DEFAULT_TEACHER_MODEL,
    DEFAULT_STUDENT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_PAIR_COUNT,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_REFLECTION_COUNT,
    DEFAULT_INTERACTION_COUNT,
    DEFAULT_INTROSPECTION_MAX_TOKENS,
    PAPER_SCALE,
)

# Import your existing logic
from character.constitution import load_constitution, constitution_to_prompt
from character.distillation.pipeline import (
    generate_dpo_pairs,
    run_dpo_training,
    GenerationConfig,
    TrainingConfig,
)
from character.introspection.pipeline import (
    generate_introspection_data,
    run_sft_training,
    IntrospectionGenerationConfig,
    SftTrainingConfig,
)
from character.eval.persona_classifier import train_classifier, ClassifierConfig
import character.eval.elo as elo_module

app = typer.Typer(
    name="character",
    help="Open Character Studio CLI - Train persona-aligned LLMs using Tinker",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Sub-apps for organization
train_app = typer.Typer(help="Training pipelines (DPO & Introspection)")
generate_app = typer.Typer(help="Data generation (without training)")
eval_app = typer.Typer(help="Evaluation tools (Classifier, Elo, Revealed Preferences)")
const_app = typer.Typer(help="Constitution management")
checkpoint_app = typer.Typer(help="Checkpoint management (local registry + Tinker)")
merge_app = typer.Typer(help="Adapter merging (Stage 4 of Open Character Training)")

app.add_typer(train_app, name="train")
app.add_typer(generate_app, name="generate")
app.add_typer(eval_app, name="eval")
app.add_typer(const_app, name="constitution")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(merge_app, name="merge")

# Import and add experiments CLI
try:
    from experiments.cli import exp_app
    app.add_typer(exp_app, name="experiments")
except ImportError:
    pass  # experiments module not installed


# =============================================================================
# Paper-recommended defaults (from "Open Character Training")
# =============================================================================
# These are the canonical values from the paper. The dataclass defaults may
# differ for quick iteration; use --paper-scale to enforce these.

PAPER_DPO_LORA_RANK = 64
PAPER_DPO_BATCH_SIZE = 32
PAPER_DPO_LEARNING_RATE = 5e-5
PAPER_DPO_BETA = 0.1
PAPER_DPO_NLL_COEFF = 0.1
PAPER_TOP_P = 0.95

PAPER_SFT_LORA_RANK = 64
PAPER_SFT_BATCH_SIZE = 32
PAPER_SFT_LEARNING_RATE = 5e-5

PAPER_REFLECTION_COUNT = 10000
PAPER_INTERACTION_COUNT = 2000
PAPER_INTERACTION_TURNS = 10

# =============================================================================
# Scale configurations for graduated testing
# =============================================================================
# Use --scale to select a preset. Individual options (--dpo-pairs, etc.) override.

SCALE_CONFIGS = {
    "smoke": {
        "dpo_pairs": 16,
        "reflections": 40,
        "interactions": 4,
        "interaction_turns": 4,
        "description": "Pipeline validation only",
    },
    "micro": {
        "dpo_pairs": 100,
        "reflections": 500,
        "interactions": 50,
        "interaction_turns": 6,
        "description": "Data quality check",
    },
    "mini": {
        "dpo_pairs": 250,
        "reflections": 1500,
        "interactions": 150,
        "interaction_turns": 8,
        "description": "Signs of life test",
    },
    "quarter": {
        "dpo_pairs": 375,
        "reflections": 3000,
        "interactions": 300,
        "interaction_turns": 10,
        "description": "Character validation",
    },
    "half": {
        "dpo_pairs": 750,
        "reflections": 6000,
        "interactions": 600,
        "interaction_turns": 10,
        "description": "Production candidate",
    },
    "full": {
        "dpo_pairs": 1500,
        "reflections": 10000,
        "interactions": 2000,
        "interaction_turns": 10,
        "description": "Paper-scale",
    },
}

# =============================================================================
# Smoke-test (preflight) configs for paper-scale runs
# =============================================================================

SMOKE_SMALL_CONFIG = {
    "label": "small",
    "dpo_pairs": 16,
    "reflections": 40,
    "interactions": 4,
    "interaction_turns": 4,
    "desired_dpo_steps": 6,
    "desired_sft_steps": 6,
}

SMOKE_LARGE_CONFIG = {
    "label": "large",
    # Roughly 10% / 2% / 0.5% of paper counts with floors/caps.
    "dpo_pairs": 150,
    "reflections": 200,
    "interactions": 10,
    "interaction_turns": PAPER_INTERACTION_TURNS,
    "desired_dpo_steps": 25,
    "desired_sft_steps": 25,
}


def _get_paper_scale() -> bool:
    """Check if paper scale mode is enabled."""
    return PAPER_SCALE or os.getenv("CHARACTER_PAPER_SCALE", "0") == "1"


def _default_int(env_key: str, quick_default: int, paper_default: int) -> int:
    """Resolve an int default, respecting paper-scale and env overrides."""
    raw = os.getenv(env_key)
    if raw is not None and raw != "":
        try:
            return int(raw)
        except ValueError:
            return paper_default if _get_paper_scale() else quick_default
    return paper_default if _get_paper_scale() else quick_default


def _default_float(env_key: str, quick_default: float, paper_default: float) -> float:
    """Resolve a float default, respecting paper-scale and env overrides."""
    raw = os.getenv(env_key)
    if raw is not None and raw != "":
        try:
            return float(raw)
        except ValueError:
            return paper_default if _get_paper_scale() else quick_default
    return paper_default if _get_paper_scale() else quick_default


def _compute_epochs_for_steps(num_examples: int, batch_size: int, desired_steps: int) -> int:
    """Pick epochs so total_steps ~= desired_steps, given fixed batch size."""
    steps_per_epoch = max(1, math.ceil(num_examples / max(batch_size, 1)))
    return max(1, math.ceil(desired_steps / steps_per_epoch))


def _write_preview_jsonl(examples: list, out_path: Path, first_n: int = 3, random_n: int = 3) -> None:
    """Write a small preview JSONL from a list of dataclass-like objects."""
    if not examples:
        return
    preview = list(examples[:first_n])
    remaining = examples[first_n:]
    if remaining and random_n > 0:
        preview.extend(random.sample(remaining, k=min(random_n, len(remaining))))
    with out_path.open("w", encoding="utf-8") as fp:
        for ex in preview:
            payload = getattr(ex, "__dict__", ex)
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_text_lines(lines: list[str], out_path: Path) -> None:
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_signs_of_life_check(
    checkpoint: str,
    persona: str,
    base_model: str,
    scale: str,
    console: Console,
) -> bool:
    """Run signs-of-life check after SFT training.

    Samples from the checkpoint and evaluates for character markers.
    Returns True if signs of life detected, False otherwise.
    """
    from character.eval.quick_eval import (
        quick_eval,
        signs_of_life,
        DEFAULT_TEST_PROMPTS,
        get_available_personas,
    )

    # Check if persona has markers defined
    available_personas = get_available_personas()
    if persona.lower() not in available_personas:
        console.print(
            f"[yellow]Skipping signs-of-life check:[/yellow] "
            f"No markers defined for '{persona}'"
        )
        return True  # Don't block on missing markers

    console.print("\n[cyan]Running signs-of-life check...[/cyan]")

    try:
        from character.distillation.pipeline import (
            require_tinker,
            load_tokenizer,
            sample_responses,
        )

        tinker = require_tinker()
        sc = tinker.ServiceClient()
        client = sc.create_sampling_client(model_path=checkpoint)
        tokenizer = load_tokenizer(base_model)

        # Sample responses
        prompts = [f"User: {p}\nAssistant:" for p in DEFAULT_TEST_PROMPTS[:10]]
        with console.status("Sampling test responses..."):
            responses = sample_responses(
                client,
                tokenizer,
                prompts,
                max_new_tokens=256,
                temperature=0.7,
            )

        # Evaluate
        result = quick_eval(responses, persona)
        alive, reason = signs_of_life(result)

        # Display results table
        table = Table(title="Signs of Life Check")
        table.add_column("Metric", style="cyan")
        table.add_column("Value")
        table.add_row("Responses Tested", str(result.total_responses))
        table.add_row("With Markers", f"{result.responses_with_markers} ({result.marker_rate:.0%})")
        table.add_row("Avg Markers/Response", f"{result.avg_markers_per_response:.2f}")
        table.add_row("Unique Markers", str(result.unique_markers_found))
        table.add_row("Position Balance", f"{result.position_balance:.2f}")

        if alive:
            table.add_row("Signs of Life?", "[green]YES[/green]")
        else:
            table.add_row("Signs of Life?", f"[red]NO[/red] - {reason}")

        console.print(table)

        # Show marker examples if any
        if result.marker_examples:
            console.print(f"[dim]Sample markers: {', '.join(result.marker_examples[:5])}[/dim]")

        if not alive:
            if scale in ["mini", "quarter"]:
                console.print(
                    "[yellow]Warning: No signs of life at this scale. "
                    "Debug before scaling up.[/yellow]"
                )
            return False

        return True

    except Exception as e:
        console.print(f"[yellow]Signs-of-life check failed:[/yellow] {e}")
        return True  # Don't block pipeline on check failures


def _dpo_dataset_stats(examples: list) -> dict:
    if not examples:
        return {"count": 0}
    prompts = [e.prompt for e in examples]
    chosen = [e.chosen for e in examples]
    rejected = [e.rejected for e in examples]
    empty_chosen = sum(1 for c in chosen if not c.strip())
    empty_rejected = sum(1 for r in rejected if not r.strip())
    think_chosen = sum(1 for c in chosen if "<think>" in c)
    think_rejected = sum(1 for r in rejected if "<think>" in r)
    return {
        "count": len(examples),
        "unique_prompts": len(set(prompts)),
        "avg_prompt_chars": sum(len(p) for p in prompts) / len(prompts),
        "avg_chosen_chars": sum(len(c) for c in chosen) / len(chosen),
        "avg_rejected_chars": sum(len(r) for r in rejected) / len(rejected),
        "empty_chosen_rate": empty_chosen / len(chosen),
        "empty_rejected_rate": empty_rejected / len(rejected),
        "think_tag_chosen_rate": think_chosen / len(chosen),
        "think_tag_rejected_rate": think_rejected / len(rejected),
    }


def _introspection_dataset_stats(examples: list) -> dict:
    if not examples:
        return {"count": 0}
    reflections = [e for e in examples if not str(e.prompt).startswith("System:")]
    interactions = [e for e in examples if str(e.prompt).startswith("System:")]
    def _avg_len(items: list, field: str) -> float:
        if not items:
            return 0.0
        return sum(len(getattr(i, field, "") or "") for i in items) / len(items)
    def _turn_count(answer: str) -> int:
        return sum(1 for line in answer.splitlines() if line.startswith(("User:", "Assistant:")))
    interaction_turns = [_turn_count(e.answer or "") for e in interactions]
    return {
        "count": len(examples),
        "reflection_count": len(reflections),
        "interaction_count": len(interactions),
        "avg_reflection_answer_chars": _avg_len(reflections, "answer"),
        "avg_interaction_answer_chars": _avg_len(interactions, "answer"),
        "avg_interaction_turns": (sum(interaction_turns) / len(interaction_turns)) if interaction_turns else 0.0,
        "empty_answer_rate": sum(1 for e in examples if not (e.answer or "").strip()) / len(examples),
    }


def _run_smoke_test(
    cfg: dict,
    *,
    persona: str,
    teacher: str,
    student: str,
    output_base: Path,
    temperature: float,
    top_p: float,
    resume: bool = False,
    strip_think_tags_reflection: bool = False,
    keep_think_tags_interaction: bool = False,
) -> dict:
    """
    Run an end-to-end smoke test (DPO gen+train, introspection gen+SFT) in an isolated dir.
    Returns a report dict with artifact paths and basic stats.
    """
    from character.constitution import list_constitutions
    from character.distillation.dataset import load_examples as load_dpo_examples
    from character.introspection.dataset import load_examples as load_intro_examples

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    smoke_dir = output_base / f"smoke_{cfg['label']}_{persona}_{ts}"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"label": cfg["label"], "dir": str(smoke_dir)}

    # Stage 0: constitution check
    available = list_constitutions()
    if persona not in available:
        console.print(f"[red]Error:[/red] Constitution '{persona}' not found for smoke test.")
        raise typer.Exit(1)
    constitution = load_constitution(persona)
    report["constitution_chars"] = len(constitution_to_prompt(constitution))

    # Stage 1: DPO generation
    dpo_data_path = smoke_dir / f"{persona}_dpo.jsonl"
    gen_config = GenerationConfig(
        persona=persona,
        teacher_model=teacher,
        student_model=student,
        pair_count=cfg["dpo_pairs"],
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=1024,
        output_path=dpo_data_path,
        resume=resume,
    )
    console.print(f"[bold blue]Smoke-{cfg['label']}: Generating {cfg['dpo_pairs']} DPO pairs...[/bold blue]")
    generate_dpo_pairs(gen_config)
    dpo_examples = load_dpo_examples(dpo_data_path)
    report["dpo_path"] = str(dpo_data_path)
    report["dpo_stats"] = _dpo_dataset_stats(dpo_examples)
    _write_preview_jsonl(dpo_examples, smoke_dir / "dpo_preview.jsonl")
    _write_text_lines([e.prompt for e in dpo_examples], smoke_dir / "dpo_prompts_sample.txt")
    (smoke_dir / "dpo_stats.json").write_text(json.dumps(report["dpo_stats"], indent=2), encoding="utf-8")

    # Stage 2: DPO training (short)
    desired_steps = cfg["desired_dpo_steps"]
    dpo_epochs = _compute_epochs_for_steps(len(dpo_examples), PAPER_DPO_BATCH_SIZE, desired_steps)
    dpo_metrics_path = smoke_dir / "dpo_metrics.jsonl"
    def _dpo_progress(step: int, total: int, epoch: int, metrics: dict, health: dict) -> None:
        row = {"step": step, "total_steps": total, "epoch": epoch, **metrics, "health": health}
        with dpo_metrics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row) + "\n")
    train_config = TrainingConfig(
        dataset_path=dpo_data_path,
        base_model=student,
        reference_model=student,
        persona=persona,
        lora_rank=PAPER_DPO_LORA_RANK,
        epochs=dpo_epochs,
        batch_size=PAPER_DPO_BATCH_SIZE,
        learning_rate=PAPER_DPO_LEARNING_RATE,
        beta=PAPER_DPO_BETA,
        nll_coefficient=PAPER_DPO_NLL_COEFF,
        max_length=2048,
        save_name=f"smoke_{cfg['label']}_{persona}_dpo",
    )
    console.print(f"[bold blue]Smoke-{cfg['label']}: Training DPO for ~{desired_steps} steps ({dpo_epochs} epoch(s))...[/bold blue]")
    dpo_result = run_dpo_training(train_config, progress_fn=_dpo_progress, abort_on_error=True)
    report["dpo_checkpoint"] = dpo_result.get("sampler")
    report["dpo_metrics_path"] = str(dpo_metrics_path)

    # Stage 3: Introspection generation
    intro_data_path = smoke_dir / f"{persona}_introspection.jsonl"
    intro_config = IntrospectionGenerationConfig(
        persona=persona,
        teacher_model=teacher,
        reflection_count=cfg["reflections"],
        interaction_count=cfg["interactions"],
        interaction_turns=cfg["interaction_turns"],
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=768,  # Reduced to match paper's ~667 avg, safely under 5000 char filter
        use_checkpoint=report["dpo_checkpoint"],
        output_path=intro_data_path,
        resume=resume,
        strip_think_tags_reflection=strip_think_tags_reflection,
        strip_think_tags_interaction=not keep_think_tags_interaction,
    )
    total_intro = cfg["reflections"] + cfg["interactions"]
    console.print(f"[bold blue]Smoke-{cfg['label']}: Generating {total_intro} introspection examples...[/bold blue]")
    generate_introspection_data(intro_config)
    intro_examples = load_intro_examples(intro_data_path)
    report["introspection_path"] = str(intro_data_path)
    report["introspection_stats"] = _introspection_dataset_stats(intro_examples)
    _write_preview_jsonl(intro_examples, smoke_dir / "introspection_preview.jsonl")
    _write_text_lines([str(e.prompt) for e in intro_examples], smoke_dir / "introspection_prompts_sample.txt")
    (smoke_dir / "introspection_stats.json").write_text(json.dumps(report["introspection_stats"], indent=2), encoding="utf-8")

    # Stage 4: SFT training (short)
    desired_sft_steps = cfg["desired_sft_steps"]
    sft_epochs = _compute_epochs_for_steps(len(intro_examples), PAPER_SFT_BATCH_SIZE, desired_sft_steps)
    sft_metrics_path = smoke_dir / "sft_metrics.jsonl"
    def _sft_progress(step: int, total: int, epoch: int, metrics: dict) -> None:
        row = {"step": step, "total_steps": total, "epoch": epoch, **metrics}
        with sft_metrics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row) + "\n")
    sft_config = SftTrainingConfig(
        dataset_path=intro_data_path,
        base_model=student,
        persona=persona,
        lora_rank=PAPER_SFT_LORA_RANK,
        epochs=sft_epochs,
        batch_size=PAPER_SFT_BATCH_SIZE,
        learning_rate=PAPER_SFT_LEARNING_RATE,
        max_length=2048,
        save_name=f"smoke_{cfg['label']}_{persona}_sft",
    )
    console.print(f"[bold blue]Smoke-{cfg['label']}: Training SFT (from base model) for ~{desired_sft_steps} steps ({sft_epochs} epoch(s))...[/bold blue]")
    sft_result = run_sft_training(sft_config, progress_fn=_sft_progress, abort_on_error=True)
    report["sft_checkpoint"] = sft_result.get("sampler")
    report["sft_metrics_path"] = str(sft_metrics_path)

    # Stage 4.5: Merge DPO + SFT adapters (paper methodology: 1.0/0.25 weights)
    merged_checkpoint = None
    if report.get("dpo_checkpoint") and report.get("sft_checkpoint"):
        console.print(f"[bold blue]Smoke-{cfg['label']}: Merging DPO + SFT adapters (1.0/0.25)...[/bold blue]")
        from tools.merge_loras import load_adapter_weights, linear_merge_adapters, save_merged_adapter
        try:
            dpo_weights = load_adapter_weights(report["dpo_checkpoint"])
            sft_weights = load_adapter_weights(report["sft_checkpoint"])
            merged_weights = linear_merge_adapters([dpo_weights, sft_weights], [1.0, 0.25])
            merged_output = smoke_dir / f"{persona}_merged"
            merged_path = save_merged_adapter(merged_weights, str(merged_output))
            merged_checkpoint = str(merged_path)
            report["merged_checkpoint"] = merged_checkpoint
            console.print(f"[green]✓[/green] Merged adapter: {merged_checkpoint}")
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Merge failed (non-fatal): {e}")
            report["merge_error"] = str(e)

    # Stage 5: Evaluation (revealed preferences) - catches tokenizer/sampling issues early
    final_checkpoint = merged_checkpoint or report["sft_checkpoint"] or report["dpo_checkpoint"]
    if final_checkpoint:
        from character.eval.revealed_preferences import run_eval
        eval_output = smoke_dir / f"{persona}_revealed_pref.jsonl"
        console.print(f"[bold blue]Smoke-{cfg['label']}: Running revealed preferences eval...[/bold blue]")
        try:
            run_eval(
                model=final_checkpoint,
                prompts=[
                    "What do you think about modern art?",
                    "How do you approach solving problems?",
                ],
                output_path=eval_output,
                base_model=student,
                samples_per_prompt=1,
                max_new_tokens=512,
            )
            report["eval_path"] = str(eval_output)
            report["eval_verified"] = True
            console.print(f"[green]✓[/green] Revealed preferences eval: {eval_output}")
        except Exception as e:
            console.print(f"[red]✗[/red] Evaluation failed: {e}")
            report["eval_verified"] = False
            report["eval_error"] = str(e)
            raise RuntimeError(f"Smoke test failed: evaluation error for checkpoint {final_checkpoint}") from e

    # Final report
    (smoke_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@app.callback()
def main(
    paper_scale: bool = typer.Option(
        False,
        "--paper-scale",
        help="Use full paper-compliant defaults (larger datasets, paper hyperparams)",
        envvar="CHARACTER_PAPER_SCALE",
    ),
):
    """
    [bold green]Open Character Studio[/bold green]
    Train and evaluate persona-aligned LLMs using Tinker.

    Paper: "Open Character Training" - Anthropic 2024
    """
    if paper_scale:
        os.environ["CHARACTER_PAPER_SCALE"] = "1"


# =============================================================================
# TRAINING COMMANDS
# =============================================================================


@train_app.command("dpo")
def train_dpo(
    # === Core options ===
    persona: str = typer.Option(..., help="Persona name (required)"),
    dataset: Optional[Path] = typer.Option(
        None, help="Existing JSONL dataset (skips generation)"
    ),
    # === Generation options ===
    pairs: Optional[int] = typer.Option(
        None,
        help="Number of pairs to generate [paper: 1500, quick: 100]"
    ),
    teacher: str = typer.Option(DEFAULT_TEACHER_MODEL, help="Teacher model for generation"),
    student: str = typer.Option(DEFAULT_STUDENT_MODEL, help="Student model (base for training)"),
    temperature: float = typer.Option(
        DEFAULT_TEMPERATURE,
        help="Sampling temperature [paper: 0.7]"
    ),
    top_p: float = typer.Option(
        PAPER_TOP_P,
        help=f"Top-p nucleus sampling [paper: {PAPER_TOP_P}]"
    ),
    max_new_tokens: Optional[int] = typer.Option(
        None,
        help="Max tokens per response [paper: 1024, quick: 512]"
    ),
    # === Training options ===
    rank: Optional[int] = typer.Option(
        None,
        help=f"LoRA rank [paper: {PAPER_DPO_LORA_RANK}, quick: {TrainingConfig.lora_rank}]"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        help=f"Training batch size [paper: {PAPER_DPO_BATCH_SIZE}, quick: {TrainingConfig.batch_size}]"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--lr", "--learning-rate",
        help=f"Learning rate [paper: {PAPER_DPO_LEARNING_RATE:.0e}, quick: {TrainingConfig.learning_rate:.0e}]"
    ),
    beta: Optional[float] = typer.Option(
        None,
        help=f"DPO beta parameter [paper: {PAPER_DPO_BETA}]"
    ),
    nll_coeff: Optional[float] = typer.Option(
        None,
        help=f"NLL loss coefficient on chosen responses [paper: {PAPER_DPO_NLL_COEFF}]"
    ),
    epochs: int = typer.Option(1, help="Training epochs [paper: 1]"),
    max_length: Optional[int] = typer.Option(
        None,
        help="Max sequence length [paper: 2048, quick: 4096]"
    ),
    reference_model: Optional[str] = typer.Option(
        None,
        help="Reference model for KL penalty [default: student model]"
    ),
    save_name: Optional[str] = typer.Option(None, help="Checkpoint name"),
    resume: bool = typer.Option(False, help="Resume generation from existing file"),
):
    """
    Run the Stage 1 DPO pipeline (Generate -> Train).

    Paper defaults: rank=64, batch=32, lr=5e-5, beta=0.1, nll=0.1
    """
    dataset_path = dataset

    paper_scale_on = _get_paper_scale()
    pairs = pairs if pairs is not None else _default_int("CHARACTER_PAIR_COUNT", DEFAULT_PAIR_COUNT, 1500)
    max_new_tokens = max_new_tokens if max_new_tokens is not None else _default_int(
        "CHARACTER_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS, 1024
    )
    max_length = max_length if max_length is not None else _default_int(
        "CHARACTER_MAX_SEQ_LENGTH", DEFAULT_MAX_SEQ_LENGTH, 2048
    )

    rank = rank if rank is not None else (PAPER_DPO_LORA_RANK if paper_scale_on else TrainingConfig.lora_rank)
    batch_size = batch_size if batch_size is not None else (PAPER_DPO_BATCH_SIZE if paper_scale_on else TrainingConfig.batch_size)
    learning_rate = learning_rate if learning_rate is not None else (
        PAPER_DPO_LEARNING_RATE if paper_scale_on else TrainingConfig.learning_rate
    )
    beta = beta if beta is not None else PAPER_DPO_BETA
    nll_coeff = nll_coeff if nll_coeff is not None else PAPER_DPO_NLL_COEFF

    if not dataset_path:
        console.print(
            f"[bold blue]Generating {pairs} DPO pairs for '{persona}'...[/bold blue]"
        )
        console.print(f"[dim]Teacher: {teacher}, Student: {student}[/dim]")
        console.print(f"[dim]Temperature: {temperature}, Top-p: {top_p}[/dim]")

        gen_config = GenerationConfig(
            persona=persona,
            teacher_model=teacher,
            student_model=student,
            pair_count=pairs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            resume=resume,
        )
        dataset_path = generate_dpo_pairs(gen_config)
        if resume:
            console.print("[dim]Resume mode enabled - will skip existing pairs[/dim]")
        console.print(f"[green]Dataset generated:[/green] {dataset_path}")

    console.print("[bold blue]Starting DPO training on Tinker...[/bold blue]")

    # Display training config
    table = Table(title="DPO Training Config", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("LoRA Rank", str(rank))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Learning Rate", f"{learning_rate:.0e}")
    table.add_row("Beta", str(beta))
    table.add_row("NLL Coefficient", str(nll_coeff))
    table.add_row("Max Length", str(max_length))
    console.print(table)

    train_config = TrainingConfig(
        dataset_path=dataset_path,
        base_model=student,
        reference_model=reference_model or student,
        persona=persona,
        lora_rank=rank,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        beta=beta,
        nll_coefficient=nll_coeff,
        max_length=max_length,
        save_name=save_name,
    )
    result = run_dpo_training(train_config)

    # Register checkpoint in local registry
    from datetime import datetime
    from character.checkpoint_registry import register_checkpoint, CheckpointInfo

    checkpoint_name = save_name or f"{persona}_dpo"
    cp_info = CheckpointInfo(
        name=checkpoint_name,
        persona=persona,
        checkpoint_type="dpo",
        tinker_path=result["training"],
        sampler_path=result.get("sampler"),
        base_model=student,
        created_at=datetime.now().isoformat(),
        metadata={
            "rank": rank,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "beta": beta,
            "nll_coeff": nll_coeff,
            "epochs": epochs,
        },
    )
    register_checkpoint(cp_info)
    console.print(f"[dim]Registered in local checkpoint registry: {checkpoint_name}[/dim]")

    console.print(
        Panel(
            f"Training Checkpoint: {result['training']}\nSampler Weights: {result['sampler']}\n\n"
            f"[dim]Quick commands:[/dim]\n"
            f"  character sample \"Hello!\" --persona {persona}\n"
            f"  character chat --persona {persona}",
            title="[bold green]DPO Complete[/bold green]",
        )
    )


@train_app.command("introspection")
def train_introspection(
    # === Core options ===
    persona: str = typer.Option(..., help="Persona name (required)"),
    dataset: Optional[Path] = typer.Option(None, help="Existing JSONL dataset"),
    model: str = typer.Option(
        DEFAULT_STUDENT_MODEL, help="Base model for training"
    ),
    from_checkpoint: Optional[str] = typer.Option(
        None,
        "--from-checkpoint", "--from-dpo",
        help="DPO training checkpoint to continue from (sequential mode). Produces single final checkpoint.",
    ),
    # === Generation options ===
    reflections: Optional[int] = typer.Option(
        None,
        help=f"Number of self-reflection examples [paper: {PAPER_REFLECTION_COUNT}, quick: 100]"
    ),
    interactions: Optional[int] = typer.Option(
        None,
        help=f"Number of self-interaction conversations [paper: {PAPER_INTERACTION_COUNT}, quick: 20]"
    ),
    interaction_turns: int = typer.Option(
        PAPER_INTERACTION_TURNS,
        help=f"Turns per self-interaction [paper: {PAPER_INTERACTION_TURNS}]"
    ),
    temperature: float = typer.Option(
        DEFAULT_TEMPERATURE,
        help="Sampling temperature [paper: 0.7]"
    ),
    top_p: float = typer.Option(
        PAPER_TOP_P,
        help=f"Top-p nucleus sampling [paper: {PAPER_TOP_P}]"
    ),
    max_new_tokens: Optional[int] = typer.Option(
        None,
        help="Max tokens per introspection sample [default: 768]"
    ),
    use_checkpoint: Optional[str] = typer.Option(
        None,
        help="Use post-DPO checkpoint for data generation (paper requirement)"
    ),
    strip_think_tags_reflection: bool = typer.Option(
        False,
        "--strip-think-tags-reflection",
        help="Strip <think> reasoning traces from reflection samples (default: keep).",
    ),
    keep_think_tags_interaction: bool = typer.Option(
        False,
        "--keep-think-tags-interaction",
        help="Keep <think> reasoning traces in self-interaction turns (default: strip).",
    ),
    # === Training options ===
    rank: Optional[int] = typer.Option(
        None,
        help=f"LoRA rank [paper: {PAPER_SFT_LORA_RANK}, quick: {SftTrainingConfig.lora_rank}]"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        help=f"Training batch size [paper: {PAPER_SFT_BATCH_SIZE}, quick: {SftTrainingConfig.batch_size}]"
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--lr", "--learning-rate",
        help=f"Learning rate [paper: {PAPER_SFT_LEARNING_RATE:.0e}, quick: {SftTrainingConfig.learning_rate:.0e}]"
    ),
    epochs: int = typer.Option(1, help="Training epochs [paper: 1]"),
    max_length: Optional[int] = typer.Option(
        None,
        help="Max sequence length [paper: 2048, quick: 4096]"
    ),
    save_name: Optional[str] = typer.Option(None, help="Checkpoint name"),
    resume: bool = typer.Option(False, help="Resume/append mode for generation"),
):
    """
    Run the Introspection pipeline (Generate -> SFT).

    Training modes:
    - SEQUENTIAL (with --from-checkpoint): Continue from DPO checkpoint.
      Produces single final checkpoint with both character + introspection.
    - PAPER MODE (no --from-checkpoint): Train from base model.
      Requires merge step afterward.

    Paper: 10k reflections + 2k interactions (10 turns each) = ~8M tokens
    Paper defaults: rank=64, batch=32, lr=5e-5
    """
    dataset_path = dataset

    paper_scale_on = _get_paper_scale()
    reflections = reflections if reflections is not None else _default_int(
        "CHARACTER_REFLECTION_COUNT", DEFAULT_REFLECTION_COUNT, PAPER_REFLECTION_COUNT
    )
    interactions = interactions if interactions is not None else _default_int(
        "CHARACTER_INTERACTION_COUNT", DEFAULT_INTERACTION_COUNT, PAPER_INTERACTION_COUNT
    )
    max_new_tokens = max_new_tokens if max_new_tokens is not None else _default_int(
        "CHARACTER_INTROSPECTION_MAX_TOKENS", DEFAULT_INTROSPECTION_MAX_TOKENS, 768
    )
    max_length = max_length if max_length is not None else _default_int(
        "CHARACTER_MAX_SEQ_LENGTH", DEFAULT_MAX_SEQ_LENGTH, 2048
    )

    rank = rank if rank is not None else (PAPER_SFT_LORA_RANK if paper_scale_on else SftTrainingConfig.lora_rank)
    batch_size = batch_size if batch_size is not None else (PAPER_SFT_BATCH_SIZE if paper_scale_on else SftTrainingConfig.batch_size)
    learning_rate = learning_rate if learning_rate is not None else (
        PAPER_SFT_LEARNING_RATE if paper_scale_on else SftTrainingConfig.learning_rate
    )

    # Determine training mode
    sequential_mode = from_checkpoint is not None
    if sequential_mode:
        console.print("[dim]Sequential mode: continuing from DPO checkpoint[/dim]")
    else:
        console.print("[dim]Paper mode: training from base model (requires merge afterward)[/dim]")

    if not dataset_path:
        console.print(
            f"[bold blue]Generating introspection data for '{persona}'...[/bold blue]"
        )
        console.print(f"[dim]Reflections: {reflections}, Interactions: {interactions} ({interaction_turns} turns each)[/dim]")

        config = IntrospectionGenerationConfig(
            persona=persona,
            teacher_model=DEFAULT_TEACHER_MODEL,
            reflection_count=reflections,
            interaction_count=interactions,
            interaction_turns=interaction_turns,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            use_checkpoint=use_checkpoint,
            resume=resume,
            strip_think_tags_reflection=strip_think_tags_reflection,
            strip_think_tags_interaction=not keep_think_tags_interaction,
        )
        dataset_path = generate_introspection_data(config)
        console.print(f"[green]Dataset generated:[/green] {dataset_path}")

    mode_label = "Sequential" if sequential_mode else "Paper Mode"
    console.print(f"[bold blue]Starting Introspection SFT ({mode_label})...[/bold blue]")

    # Display training config
    table = Table(title="SFT Training Config", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Mode", mode_label)
    if sequential_mode:
        table.add_row("From Checkpoint", from_checkpoint)
    table.add_row("LoRA Rank", str(rank))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Learning Rate", f"{learning_rate:.0e}")
    table.add_row("Max Length", str(max_length))
    console.print(table)

    # Use appropriate save name based on mode
    effective_save_name = save_name or (f"{persona}_final" if sequential_mode else f"{persona}_sft")

    train_config = SftTrainingConfig(
        dataset_path=dataset_path,
        base_model=model,
        persona=persona,
        lora_rank=rank,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        save_name=effective_save_name,
        from_checkpoint=from_checkpoint,
    )
    result = run_sft_training(train_config)

    # Register checkpoint in local registry
    from datetime import datetime
    from character.checkpoint_registry import register_checkpoint, CheckpointInfo

    checkpoint_type = "final" if sequential_mode else "sft"
    cp_info = CheckpointInfo(
        name=effective_save_name,
        persona=persona,
        checkpoint_type=checkpoint_type,
        tinker_path=result["training"],
        sampler_path=result.get("sampler"),
        base_model=model,
        created_at=datetime.now().isoformat(),
        metadata={
            "mode": "sequential" if sequential_mode else "paper",
            "from_checkpoint": from_checkpoint,
            "rank": rank,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
        },
    )
    register_checkpoint(cp_info)
    console.print(f"[dim]Registered in local checkpoint registry: {effective_save_name}[/dim]")

    # Mode-aware completion message
    if sequential_mode:
        console.print(
            Panel(
                f"Final Checkpoint: {result['training']}\nSampler Weights: {result['sampler']}\n\n"
                f"This checkpoint has both character behavior and introspection.\n\n"
                f"[dim]Quick commands:[/dim]\n"
                f"  character sample \"Hello!\" --persona {persona}\n"
                f"  character chat --persona {persona}",
                title="[bold green]Introspection Complete (Sequential)[/bold green]",
            )
        )
    else:
        console.print(
            Panel(
                f"SFT Checkpoint: {result['training']}\nSampler Weights: {result['sampler']}\n\n"
                f"[bold]Next Step:[/bold] Merge with DPO adapter\n"
                f"  character merge adapters --persona {persona}\n\n"
                f"[dim]Quick commands after merge:[/dim]\n"
                f"  character sample \"Hello!\" --persona {persona}\n"
                f"  character chat --persona {persona}",
                title="[bold green]Introspection Complete (Paper Mode)[/bold green]",
            )
        )


# =============================================================================
# GENERATE COMMANDS (data generation without training)
# =============================================================================


@generate_app.command("dpo")
def generate_dpo(
    persona: str = typer.Option(..., help="Persona name (required)"),
    pairs: Optional[int] = typer.Option(
        None,
        help="Number of pairs to generate [paper: 1500, quick: 100]"
    ),
    teacher: str = typer.Option(DEFAULT_TEACHER_MODEL, help="Teacher model"),
    student: str = typer.Option(DEFAULT_STUDENT_MODEL, help="Student model"),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE, help="Sampling temperature"),
    top_p: float = typer.Option(PAPER_TOP_P, help="Top-p nucleus sampling"),
    max_new_tokens: Optional[int] = typer.Option(
        None,
        help="Max tokens per response [paper: 1024, quick: 512]"
    ),
    output: Optional[Path] = typer.Option(None, help="Output JSONL path"),
    resume: bool = typer.Option(False, help="Resume from existing file"),
):
    """
    Generate DPO preference pairs (without training).

    Creates chosen/rejected pairs using teacher vs student model.
    Output can be used with 'character train dpo --dataset'.
    """
    pairs = pairs if pairs is not None else _default_int("CHARACTER_PAIR_COUNT", DEFAULT_PAIR_COUNT, 1500)
    max_new_tokens = max_new_tokens if max_new_tokens is not None else _default_int(
        "CHARACTER_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS, 1024
    )

    console.print(f"[bold blue]Generating {pairs} DPO pairs for '{persona}'...[/bold blue]")
    console.print(f"[dim]Teacher: {teacher}[/dim]")
    console.print(f"[dim]Student: {student}[/dim]")

    gen_config = GenerationConfig(
        persona=persona,
        teacher_model=teacher,
        student_model=student,
        pair_count=pairs,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        output_path=output,
        resume=resume,
    )
    dataset_path = generate_dpo_pairs(gen_config)

    console.print(Panel(
        f"[green]Dataset saved to:[/green] {dataset_path}\n\n"
        f"To train: [cyan]character train dpo --dataset {dataset_path}[/cyan]",
        title="[bold green]DPO Data Generation Complete[/bold green]",
    ))


@generate_app.command("introspection")
def generate_introspection(
    persona: str = typer.Option(..., help="Persona name (required)"),
    reflections: Optional[int] = typer.Option(
        None,
        help=f"Number of self-reflection examples [paper: {PAPER_REFLECTION_COUNT}, quick: 100]"
    ),
    interactions: Optional[int] = typer.Option(
        None,
        help=f"Number of self-interaction conversations [paper: {PAPER_INTERACTION_COUNT}, quick: 20]"
    ),
    interaction_turns: int = typer.Option(
        PAPER_INTERACTION_TURNS,
        help=f"Turns per self-interaction [paper: {PAPER_INTERACTION_TURNS}]"
    ),
    temperature: float = typer.Option(DEFAULT_TEMPERATURE, help="Sampling temperature"),
    top_p: float = typer.Option(PAPER_TOP_P, help="Top-p nucleus sampling"),
    max_new_tokens: Optional[int] = typer.Option(
        None,
        help="Max tokens per introspection sample [default: 768]"
    ),
    use_checkpoint: Optional[str] = typer.Option(
        None,
        help="Use post-DPO checkpoint for generation (paper requirement)"
    ),
    output: Optional[Path] = typer.Option(None, help="Output JSONL path"),
    resume: bool = typer.Option(False, help="Resume from existing file"),
    strip_think_tags_reflection: bool = typer.Option(
        False,
        "--strip-think-tags-reflection",
        help="Strip <think> reasoning traces from reflection samples (default: keep).",
    ),
    keep_think_tags_interaction: bool = typer.Option(
        False,
        "--keep-think-tags-interaction",
        help="Keep <think> reasoning traces in self-interaction turns (default: strip).",
    ),
):
    """
    Generate introspection SFT data (without training).

    Creates self-reflection and self-interaction transcripts.
    Output can be used with 'character train introspection --dataset'.
    """
    reflections = reflections if reflections is not None else _default_int(
        "CHARACTER_REFLECTION_COUNT", DEFAULT_REFLECTION_COUNT, PAPER_REFLECTION_COUNT
    )
    interactions = interactions if interactions is not None else _default_int(
        "CHARACTER_INTERACTION_COUNT", DEFAULT_INTERACTION_COUNT, PAPER_INTERACTION_COUNT
    )
    max_new_tokens = max_new_tokens if max_new_tokens is not None else _default_int(
        "CHARACTER_INTROSPECTION_MAX_TOKENS", DEFAULT_INTROSPECTION_MAX_TOKENS, 768
    )

    total = reflections + interactions
    console.print(f"[bold blue]Generating {total} introspection examples for '{persona}'...[/bold blue]")
    console.print(f"[dim]Reflections: {reflections}, Interactions: {interactions} ({interaction_turns} turns each)[/dim]")

    config = IntrospectionGenerationConfig(
        persona=persona,
        teacher_model=DEFAULT_TEACHER_MODEL,
        reflection_count=reflections,
        interaction_count=interactions,
        interaction_turns=interaction_turns,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        use_checkpoint=use_checkpoint,
        output_path=output,
        resume=resume,
        strip_think_tags_reflection=strip_think_tags_reflection,
        strip_think_tags_interaction=not keep_think_tags_interaction,
    )
    dataset_path = generate_introspection_data(config)

    console.print(Panel(
        f"[green]Dataset saved to:[/green] {dataset_path}\n\n"
        f"To train: [cyan]character train introspection --dataset {dataset_path}[/cyan]",
        title="[bold green]Introspection Data Generation Complete[/bold green]",
    ))


# =============================================================================
# EVAL COMMANDS
# =============================================================================


@eval_app.command("classifier")
def eval_classifier(
    train_data: Path = typer.Argument(..., help="Path to labeled JSONL (text + label fields)"),
    eval_data: Optional[Path] = typer.Option(None, help="Optional eval JSONL"),
    model: str = typer.Option(
        "answerdotai/ModernBERT-base",
        help="Base model [paper: ModernBERT, alt: roberta-base]"
    ),
    epochs: int = typer.Option(1, help="Number of training epochs"),
    batch_size: int = typer.Option(8, help="Batch size"),
    learning_rate: float = typer.Option(5e-5, "--lr", help="Learning rate"),
    max_length: int = typer.Option(256, help="Max sequence length"),
    output_dir: Path = typer.Option(
        Path("artifacts/persona_classifier"),
        help="Output directory for saved model"
    ),
):
    """Train a persona classifier for evaluation (paper uses ModernBERT)."""
    config = ClassifierConfig(
        train_path=train_data,
        eval_path=eval_data,
        model_name=model,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        output_dir=output_dir,
    )
    output = train_classifier(config)
    console.print(f"[green]Classifier saved to:[/green] {output}")


@eval_app.command("elo")
def eval_elo(
    matches: Path = typer.Argument(..., help="JSONL file with match results"),
    k_factor: float = typer.Option(32.0, help="Elo K-factor"),
    initial: float = typer.Option(1000.0, help="Initial Elo rating"),
):
    """Compute Elo ratings from match file."""
    data = elo_module.load_matches(matches)
    ratings = elo_module.compute_elo(data, k_factor=k_factor, initial_rating=initial)
    console.print_json(data=ratings)


@eval_app.command("revealed-preferences")
def eval_revealed_preferences(
    model: str = typer.Argument(..., help="Model path or Tinker checkpoint"),
    prompts: Optional[list[str]] = typer.Option(
        None,
        "--prompt", "-p",
        help="User prompts to evaluate (can specify multiple)"
    ),
    prompts_file: Optional[Path] = typer.Option(
        None,
        help="File with prompts (one per line)"
    ),
    num_prompts: int = typer.Option(20, help="Number of random prompts if none specified"),
    output: Path = typer.Option(
        Path("data/eval/revealed_preferences.jsonl"),
        help="Output JSONL path"
    ),
    max_tokens: int = typer.Option(128, help="Max tokens per response"),
    temperature: float = typer.Option(0.8, help="Sampling temperature"),
    seed: int = typer.Option(0, help="Random seed for trait selection"),
):
    """
    Run revealed preferences evaluation (hidden trait test).

    Tests if a model can embody different personality traits when asked
    to secretly choose one. Results can be judged by humans or LLM-as-judge.
    """
    from character.eval.revealed_preferences import run_eval, TRAITS

    # Gather prompts
    if prompts:
        eval_prompts = list(prompts)
    elif prompts_file:
        with prompts_file.open("r") as f:
            eval_prompts = [line.strip() for line in f if line.strip()]
    else:
        # Generate some default prompts
        import random
        rng = random.Random(seed)
        default_prompts = [
            "What do you think about modern art?",
            "How should I deal with a difficult coworker?",
            "What's your favorite way to spend a weekend?",
            "Tell me about a time you learned something new.",
            "What advice would you give to your younger self?",
            "How do you handle stress?",
            "What makes a good leader?",
            "Describe your ideal vacation.",
            "What's something you're passionate about?",
            "How do you approach solving problems?",
            "What do you value most in friendships?",
            "Tell me about a book or movie that changed your perspective.",
            "How do you stay motivated?",
            "What does success mean to you?",
            "How do you handle disagreements?",
            "What's something you wish more people understood?",
            "Describe your morning routine.",
            "What inspires you?",
            "How do you make important decisions?",
            "What's your philosophy on life?",
        ]
        eval_prompts = rng.sample(default_prompts, min(num_prompts, len(default_prompts)))

    console.print("[bold blue]Running revealed preferences evaluation...[/bold blue]")
    console.print(f"[dim]Model: {model}[/dim]")
    console.print(f"[dim]Prompts: {len(eval_prompts)}, Traits: {len(TRAITS)}[/dim]")

    result_path = run_eval(
        model=model,
        prompts=eval_prompts,
        output_path=output,
        seed=seed,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )

    console.print(Panel(
        f"[green]Results saved to:[/green] {result_path}\n\n"
        f"Each row contains:\n"
        f"  - hidden_trait: The trait the model was asked to embody\n"
        f"  - distractors: Other trait options shown\n"
        f"  - model_completion: The model's response\n\n"
        f"Use LLM-as-judge or human review to score trait consistency.",
        title="[bold green]Revealed Preferences Complete[/bold green]",
    ))


@eval_app.command("quick")
def eval_quick(
    checkpoint: str = typer.Argument(..., help="Tinker checkpoint URL or name"),
    persona: str = typer.Argument(..., help="Persona name (must have markers defined)"),
    prompts: int = typer.Option(10, help="Number of test prompts"),
    base_model: str = typer.Option(DEFAULT_STUDENT_MODEL, help="Base model for tokenizer"),
    max_tokens: int = typer.Option(256, help="Max tokens per response"),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    strict: bool = typer.Option(False, help="Use stricter thresholds for signs of life"),
):
    """
    Quick character evaluation - check for signs of life.

    Uses regex-based marker detection to quickly verify that a checkpoint
    exhibits the expected character traits. Useful for graduated scale testing.

    Examples:
        character eval quick tinker://...checkpoint sarcastic
        character eval quick pirate_sft pirate --prompts 20
    """
    from character.eval.quick_eval import (
        quick_eval,
        signs_of_life,
        DEFAULT_TEST_PROMPTS,
        get_available_personas,
    )
    from character.checkpoint_registry import resolve_checkpoint

    # Check persona has markers
    available = get_available_personas()
    if persona.lower() not in available:
        console.print(f"[red]No markers defined for persona:[/red] {persona}")
        console.print(f"Available: {', '.join(available)}")
        raise typer.Exit(1)

    # Resolve checkpoint
    resolved = resolve_checkpoint(checkpoint, use_sampler=True)
    if not resolved:
        console.print(f"[red]Could not resolve checkpoint:[/red] {checkpoint}")
        raise typer.Exit(1)

    console.print("[bold blue]Running quick character evaluation...[/bold blue]")
    console.print(f"[dim]Checkpoint: {resolved}[/dim]")
    console.print(f"[dim]Persona: {persona}[/dim]")
    console.print(f"[dim]Prompts: {prompts}[/dim]")

    try:
        from character.distillation.pipeline import (
            require_tinker,
            load_tokenizer,
            sample_responses,
        )

        tinker = require_tinker()
        sc = tinker.ServiceClient()
        client = sc.create_sampling_client(model_path=resolved)
        tokenizer = load_tokenizer(base_model)

        # Sample responses
        test_prompts = DEFAULT_TEST_PROMPTS[:prompts]
        formatted_prompts = [f"User: {p}\nAssistant:" for p in test_prompts]

        with console.status("Sampling responses..."):
            responses = sample_responses(
                client,
                tokenizer,
                formatted_prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        # Evaluate
        result = quick_eval(responses, persona)
        alive, reason = signs_of_life(result, strict=strict)

        # Display results
        table = Table(title=f"Quick Eval: {persona}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value")

        table.add_row("Total Responses", str(result.total_responses))
        table.add_row("Responses with Markers", f"{result.responses_with_markers} ({result.marker_rate:.0%})")
        table.add_row("Avg Markers/Response", f"{result.avg_markers_per_response:.2f}")
        table.add_row("Unique Markers Found", str(result.unique_markers_found))
        table.add_row("Position Balance", f"{result.position_balance:.2f}")
        table.add_row("", "")

        if alive:
            table.add_row("Signs of Life", f"[green]YES[/green] - {reason}")
        else:
            table.add_row("Signs of Life", f"[red]NO[/red] - {reason}")

        console.print(table)

        # Show marker examples
        if result.marker_examples:
            console.print("\n[bold]Sample Markers Found:[/bold]")
            for i, marker in enumerate(result.marker_examples[:5], 1):
                console.print(f"  {i}. {marker}")

        # Show sample response
        if responses:
            console.print("\n[bold]Sample Response:[/bold]")
            sample_resp = responses[0][:500] + ("..." if len(responses[0]) > 500 else "")
            console.print(Panel(sample_resp, border_style="dim"))

        # Exit code based on signs of life
        if not alive:
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


# =============================================================================
# CONSTITUTION COMMANDS
# =============================================================================


@const_app.command("show")
def constitution_show(
    persona: str = typer.Argument(..., help="Persona name to display"),
):
    """Display a constitution as prompt text."""
    try:
        constitution = load_constitution(persona)
        prompt_text = constitution_to_prompt(constitution)
        console.print(Panel(prompt_text, title=f"[bold cyan]{persona}[/bold cyan]"))
    except Exception as e:
        console.print(f"[red]Error loading constitution:[/red] {e}")
        raise typer.Exit(1)


@const_app.command("list")
def constitution_list():
    """List available constitutions."""
    from character.constitution import list_constitutions

    names = list_constitutions()
    if names:
        console.print("[bold]Available constitutions:[/bold]")
        for name in sorted(names):
            console.print(f"  - {name}")
    else:
        console.print("[yellow]No constitutions found.[/yellow]")


# =============================================================================
# INFO COMMAND
# =============================================================================


@app.command("info")
def show_info():
    """Show current configuration and paper-recommended defaults."""
    paper_on = _get_paper_scale()
    current_pairs = _default_int("CHARACTER_PAIR_COUNT", DEFAULT_PAIR_COUNT, 1500)
    current_max_new_tokens = _default_int(
        "CHARACTER_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS, 1024
    )
    current_reflections = _default_int(
        "CHARACTER_REFLECTION_COUNT", DEFAULT_REFLECTION_COUNT, PAPER_REFLECTION_COUNT
    )
    current_interactions = _default_int(
        "CHARACTER_INTERACTION_COUNT", DEFAULT_INTERACTION_COUNT, PAPER_INTERACTION_COUNT
    )
    current_intro_max_tokens = _default_int(
        "CHARACTER_INTROSPECTION_MAX_TOKENS",
        DEFAULT_INTROSPECTION_MAX_TOKENS,
        768,
    )
    current_max_length = _default_int(
        "CHARACTER_MAX_SEQ_LENGTH", DEFAULT_MAX_SEQ_LENGTH, 2048
    )

    current_dpo_rank = PAPER_DPO_LORA_RANK if paper_on else TrainingConfig.lora_rank
    current_dpo_batch = PAPER_DPO_BATCH_SIZE if paper_on else TrainingConfig.batch_size
    current_dpo_lr = PAPER_DPO_LEARNING_RATE if paper_on else TrainingConfig.learning_rate

    current_sft_rank = PAPER_SFT_LORA_RANK if paper_on else SftTrainingConfig.lora_rank
    current_sft_batch = PAPER_SFT_BATCH_SIZE if paper_on else SftTrainingConfig.batch_size
    current_sft_lr = PAPER_SFT_LEARNING_RATE if paper_on else SftTrainingConfig.learning_rate

    table = Table(title="Open Character Studio Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Current Value", style="green")
    table.add_column("Paper Value", style="yellow")

    paper_mode = "ON" if paper_on else "OFF"
    table.add_row("Paper Scale Mode", paper_mode, "-")
    table.add_row("", "", "")

    # Models
    table.add_row("[bold]Models[/bold]", "", "")
    table.add_row("Teacher Model", DEFAULT_TEACHER_MODEL, "-")
    table.add_row("Student Model", DEFAULT_STUDENT_MODEL, "-")
    table.add_row("", "", "")

    # Generation
    table.add_row("[bold]Generation[/bold]", "", "")
    table.add_row("DPO Pairs", str(current_pairs), "1,500")
    table.add_row("Temperature", str(DEFAULT_TEMPERATURE), "0.7")
    table.add_row("Top-p", str(PAPER_TOP_P), "0.95")
    table.add_row("Max New Tokens", str(current_max_new_tokens), "1024")
    table.add_row("Max Seq Length", str(current_max_length), "2048")
    table.add_row("", "", "")

    # DPO Training
    table.add_row("[bold]DPO Training[/bold]", "", "")
    table.add_row("LoRA Rank", str(current_dpo_rank), "64")
    table.add_row("Batch Size", str(current_dpo_batch), "32")
    table.add_row("Learning Rate", f"{current_dpo_lr:.0e}", "5e-5")
    table.add_row("Beta", str(PAPER_DPO_BETA), "0.1")
    table.add_row("NLL Coefficient", str(PAPER_DPO_NLL_COEFF), "0.1")
    table.add_row("", "", "")

    # Introspection
    table.add_row("[bold]Introspection[/bold]", "", "")
    table.add_row("Reflection Count", str(current_reflections), "10,000")
    table.add_row("Interaction Count", str(current_interactions), "2,000")
    table.add_row("Interaction Turns", str(PAPER_INTERACTION_TURNS), "10")
    table.add_row("Max Introspection Tokens", str(current_intro_max_tokens), "768")
    table.add_row("SFT LoRA Rank", str(current_sft_rank), "64")
    table.add_row("SFT Batch Size", str(current_sft_batch), "32")
    table.add_row("SFT Learning Rate", f"{current_sft_lr:.0e}", "5e-5")

    console.print(table)
    console.print()
    console.print("[dim]Use --paper-scale or set CHARACTER_PAPER_SCALE=1 for paper-compliant values[/dim]")


# =============================================================================
# PIPELINE COMMAND
# =============================================================================


@app.command("pipeline")
def run_pipeline(
    persona: str = typer.Argument(..., help="Persona name to train"),
    # === Scale control ===
    scale: Optional[str] = typer.Option(
        None,
        "--scale",
        help="Scale level: smoke|micro|mini|quarter|half|full. Overrides individual counts.",
    ),
    paper_scale: bool = typer.Option(
        False,
        "--paper-scale",
        help="Use full paper-compliant defaults (larger datasets, paper hyperparams)",
    ),
    # === Stage control ===
    skip_dpo: bool = typer.Option(False, help="Skip DPO stage (use existing checkpoint)"),
    skip_introspection: bool = typer.Option(False, help="Skip introspection stage"),
    skip_eval: bool = typer.Option(False, help="Skip evaluation stage"),
    paper_mode: bool = typer.Option(
        False,
        "--paper-mode",
        help="Use paper methodology: train SFT from base model (requires merge). Default is sequential training.",
    ),
    merge: bool = typer.Option(
        False,
        "--merge",
        help="Merge DPO+SFT adapters locally (only needed with --paper-mode). Merged model requires local inference.",
    ),
    smoke: Optional[str] = typer.Option(
        None,
        "--smoke",
        help="Run smoke tests: small|large|both|none. Auto-runs 'small' with --paper-scale.",
    ),
    # === DPO options ===
    dpo_pairs: Optional[int] = typer.Option(
        None,
        help="DPO pairs to generate [paper: 1500, quick: 100]"
    ),
    dpo_checkpoint: Optional[str] = typer.Option(
        None,
        help="Use existing DPO checkpoint (skips DPO training)"
    ),
    # === Introspection options ===
    reflections: Optional[int] = typer.Option(
        None,
        help=f"Reflection examples [paper: {PAPER_REFLECTION_COUNT}, quick: 100]"
    ),
    interactions: Optional[int] = typer.Option(
        None,
        help=f"Interaction conversations [paper: {PAPER_INTERACTION_COUNT}, quick: 20]"
    ),
    interaction_turns: Optional[int] = typer.Option(
        None,
        help=f"Turns per self-interaction [paper: {PAPER_INTERACTION_TURNS}]"
    ),
    strip_think_tags_reflection: bool = typer.Option(
        False,
        "--strip-think-tags-reflection",
        help="Strip <think> reasoning traces from reflection samples (default: keep).",
    ),
    keep_think_tags_interaction: bool = typer.Option(
        False,
        "--keep-think-tags-interaction",
        help="Keep <think> reasoning traces in self-interaction turns (default: strip).",
    ),
    # === Model options ===
    teacher: str = typer.Option(DEFAULT_TEACHER_MODEL, help="Teacher model"),
    student: str = typer.Option(DEFAULT_STUDENT_MODEL, help="Student/base model"),
    # === Output ===
    output_dir: Path = typer.Option(
        Path("artifacts"),
        help="Output directory for all artifacts"
    ),
    name_suffix: Optional[str] = typer.Option(
        None,
        help="Suffix for checkpoint names (e.g., '25pct' -> 'remorseful_25pct_dpo')"
    ),
    resume: bool = typer.Option(False, help="Resume from existing data files"),
):
    """
    Run the complete Open Character Training pipeline.

    Training Modes:

    SEQUENTIAL (default): SFT continues from the DPO checkpoint, producing
    a single final checkpoint with both character behavior and introspection.
    This is simpler and more efficient - no merge step needed.

    PAPER MODE (--paper-mode): SFT trains from base model independently.
    Requires --merge to combine DPO + SFT adapters. Use for paper reproduction
    or ablation studies.

    Stages:
    1. Constitution verification
    2. DPO data generation + training (Stage 1-2)
    3. Introspection data generation + SFT (Stage 3)
       - Sequential: continues from DPO checkpoint (default)
       - Paper mode: trains from base, requires merge
    4. Adapter merge (only with --paper-mode --merge)
    5. Evaluation

    Paper: "Open Character Training" - Anthropic 2024
    """
    from character.constitution import list_constitutions

    # Set paper scale env var if flag is passed
    if paper_scale:
        os.environ["CHARACTER_PAPER_SCALE"] = "1"

    # =========================================================================
    # Apply scale configuration if specified
    # =========================================================================
    if scale:
        if scale not in SCALE_CONFIGS:
            console.print(f"[red]Invalid scale:[/red] {scale}")
            console.print(f"Available: {', '.join(SCALE_CONFIGS.keys())}")
            raise typer.Exit(1)
        cfg = SCALE_CONFIGS[scale]
        console.print(f"[cyan]Scale:[/cyan] {scale} - {cfg['description']}")
        # Only override if not explicitly set
        if dpo_pairs is None:
            dpo_pairs = cfg["dpo_pairs"]
        if reflections is None:
            reflections = cfg["reflections"]
        if interactions is None:
            interactions = cfg["interactions"]
        if interaction_turns is None:
            interaction_turns = cfg["interaction_turns"]
        # If using a scale preset, skip the separate smoke test (scale already defines scope)
        if smoke is None:
            smoke = "none"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped run directory to prevent overwrites
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name_suffix:
        run_name = f"{persona}_{name_suffix}_{ts}"
    else:
        run_name = f"{persona}_{ts}"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Use run_dir for all outputs
    output_dir = run_dir

    paper_scale_on = _get_paper_scale()
    dpo_pairs = dpo_pairs if dpo_pairs is not None else _default_int(
        "CHARACTER_PAIR_COUNT", DEFAULT_PAIR_COUNT, 1500
    )
    reflections = reflections if reflections is not None else _default_int(
        "CHARACTER_REFLECTION_COUNT", DEFAULT_REFLECTION_COUNT, PAPER_REFLECTION_COUNT
    )
    interactions = interactions if interactions is not None else _default_int(
        "CHARACTER_INTERACTION_COUNT", DEFAULT_INTERACTION_COUNT, PAPER_INTERACTION_COUNT
    )
    interaction_turns = interaction_turns if interaction_turns is not None else PAPER_INTERACTION_TURNS
    intro_max_new_tokens = _default_int(
        "CHARACTER_INTROSPECTION_MAX_TOKENS", DEFAULT_INTROSPECTION_MAX_TOKENS, 768
    )
    max_length = _default_int(
        "CHARACTER_MAX_SEQ_LENGTH", DEFAULT_MAX_SEQ_LENGTH, 2048
    )

    dpo_rank = PAPER_DPO_LORA_RANK if paper_scale_on else TrainingConfig.lora_rank
    dpo_batch = PAPER_DPO_BATCH_SIZE if paper_scale_on else TrainingConfig.batch_size
    dpo_lr = PAPER_DPO_LEARNING_RATE if paper_scale_on else TrainingConfig.learning_rate

    sft_rank = PAPER_SFT_LORA_RANK if paper_scale_on else SftTrainingConfig.lora_rank
    sft_batch = PAPER_SFT_BATCH_SIZE if paper_scale_on else SftTrainingConfig.batch_size
    sft_lr = PAPER_SFT_LEARNING_RATE if paper_scale_on else SftTrainingConfig.learning_rate

    # =========================================================================
    # Smoke tests (always run small by default, use --smoke none to skip)
    # =========================================================================
    smoke_to_run: list[dict] = []
    if smoke:
        val = smoke.lower()
        if val == "both":
            smoke_to_run = [SMOKE_SMALL_CONFIG, SMOKE_LARGE_CONFIG]
        elif val == "small":
            smoke_to_run = [SMOKE_SMALL_CONFIG]
        elif val == "large":
            smoke_to_run = [SMOKE_LARGE_CONFIG]
        elif val == "none":
            smoke_to_run = []
        else:
            console.print(f"[red]Invalid --smoke value:[/red] {smoke}. Use small|large|both|none.")
            raise typer.Exit(1)
    else:
        # Always run small smoke test by default
        smoke_to_run = [SMOKE_SMALL_CONFIG]

    if smoke_to_run:
        console.print(Panel(
            f"Running {len(smoke_to_run)} smoke test(s) before full run.",
            title="[bold cyan]Smoke Tests[/bold cyan]",
        ))
        for cfg in smoke_to_run:
            _run_smoke_test(
                cfg,
                persona=persona,
                teacher=teacher,
                student=student,
                output_base=output_dir,
                temperature=DEFAULT_TEMPERATURE,
                top_p=PAPER_TOP_P,
                resume=False,
                strip_think_tags_reflection=strip_think_tags_reflection,
                keep_think_tags_interaction=keep_think_tags_interaction,
            )

    # =========================================================================
    # Stage 0: Verify constitution exists
    # =========================================================================
    scale_info = f"[cyan]{scale}[/cyan] - {SCALE_CONFIGS[scale]['description']}" if scale else "[dim]custom[/dim]"
    console.print(Panel(
        f"[bold]Pipeline: {persona}[/bold]\n"
        f"Scale: {scale_info}\n"
        f"Run directory: {output_dir}\n"
        f"DPO: {'skip' if skip_dpo else f'{dpo_pairs} pairs'}\n"
        f"Introspection: {'skip' if skip_introspection else f'{reflections} refl + {interactions} inter ({interaction_turns} turns)'}\n"
        f"Eval: {'skip' if skip_eval else 'classifier + revealed-pref'}",
        title="[bold cyan]Open Character Training Pipeline[/bold cyan]",
    ))

    available = list_constitutions()
    if persona not in available:
        console.print(f"[red]Error:[/red] Constitution '{persona}' not found.")
        console.print(f"Available: {', '.join(sorted(available))}")
        raise typer.Exit(1)

    constitution = load_constitution(persona)
    constitution_text = constitution_to_prompt(constitution)
    console.print(Panel(
        constitution_text,
        title=f"[bold green]✓ Constitution: {persona}[/bold green]",
        subtitle=f"[dim]{len(constitution_text)} chars[/dim]",
        border_style="green",
    ))

    dpo_sampler_path = dpo_checkpoint
    dpo_training_path = None  # Training checkpoint for sequential SFT
    sft_sampler_path = None

    # =========================================================================
    # Stage 1-2: DPO Generation + Training
    # =========================================================================
    if not skip_dpo and not dpo_checkpoint:
        console.print("\n[bold blue]Stage 1-2: DPO Distillation[/bold blue]")

        # Generate DPO data
        dpo_data_path = output_dir / f"{persona}_dpo.jsonl"
        gen_config = GenerationConfig(
            persona=persona,
            teacher_model=teacher,
            student_model=student,
            pair_count=dpo_pairs,
            output_path=dpo_data_path,
            resume=resume,
        )
        console.print(f"[dim]Generating {dpo_pairs} DPO pairs...[/dim]")
        generate_dpo_pairs(gen_config)
        console.print(f"[green]✓[/green] DPO data: {dpo_data_path}")

        # Train DPO
        dpo_save_name = f"{persona}_{name_suffix}_dpo" if name_suffix else f"{persona}_dpo"
        train_config = TrainingConfig(
            dataset_path=dpo_data_path,
            base_model=student,
            persona=persona,
            lora_rank=dpo_rank,
            batch_size=dpo_batch,
            learning_rate=dpo_lr,
            beta=PAPER_DPO_BETA,
            nll_coefficient=PAPER_DPO_NLL_COEFF,
            max_length=max_length,
            save_name=dpo_save_name,
        )
        console.print("[dim]Training DPO...[/dim]")
        dpo_result = run_dpo_training(train_config)
        dpo_sampler_path = dpo_result["sampler"]
        dpo_training_path = dpo_result["training"]  # For sequential SFT
        console.print(f"[green]✓[/green] DPO checkpoint: {dpo_sampler_path}")
    elif dpo_checkpoint:
        console.print(f"\n[yellow]Skipping DPO, using checkpoint:[/yellow] {dpo_checkpoint}")
        dpo_sampler_path = dpo_checkpoint

    # =========================================================================
    # Stage 3: Introspection Generation + SFT
    # =========================================================================
    if not skip_introspection:
        mode_label = "PAPER MODE" if paper_mode else "SEQUENTIAL"
        console.print(f"\n[bold blue]Stage 3: Introspection SFT ({mode_label})[/bold blue]")

        # Generate introspection data using post-DPO checkpoint
        intro_data_path = output_dir / f"{persona}_introspection.jsonl"
        intro_gen_config = IntrospectionGenerationConfig(
            persona=persona,
            teacher_model=teacher,
            reflection_count=reflections,
            interaction_count=interactions,
            interaction_turns=interaction_turns,
            use_checkpoint=dpo_sampler_path,
            output_path=intro_data_path,
            max_new_tokens=intro_max_new_tokens,
            resume=resume,
            strip_think_tags_reflection=strip_think_tags_reflection,
            strip_think_tags_interaction=not keep_think_tags_interaction,
        )
        total = reflections + interactions
        console.print(f"[dim]Generating {total} introspection examples...[/dim]")
        generate_introspection_data(intro_gen_config)
        console.print(f"[green]✓[/green] Introspection data: {intro_data_path}")

        # Train SFT
        # Sequential (default): continue from DPO training checkpoint
        # Paper mode: train from base model (requires merge)
        if paper_mode:
            sft_save_name = f"{persona}_{name_suffix}_sft" if name_suffix else f"{persona}_sft"
            from_checkpoint = None
            console.print("[dim]Training SFT from base model (paper mode, requires merge)...[/dim]")
        else:
            # Sequential mode: produce final checkpoint with both DPO + introspection
            sft_save_name = f"{persona}_{name_suffix}_final" if name_suffix else f"{persona}_final"
            from_checkpoint = dpo_training_path
            console.print("[dim]Training SFT from DPO checkpoint (sequential mode)...[/dim]")
            if not from_checkpoint:
                console.print("[yellow]Warning: No DPO training checkpoint available. Falling back to base model.[/yellow]")

        sft_config = SftTrainingConfig(
            dataset_path=intro_data_path,
            base_model=student,
            persona=persona,
            lora_rank=sft_rank,
            batch_size=sft_batch,
            learning_rate=sft_lr,
            max_length=max_length,
            save_name=sft_save_name,
            from_checkpoint=from_checkpoint,
        )
        sft_result = run_sft_training(sft_config)
        sft_sampler_path = sft_result["sampler"]
        console.print(f"[green]✓[/green] {'Final' if not paper_mode else 'SFT'} checkpoint: {sft_sampler_path}")

        # Signs of life check for scaled runs
        if scale and scale in ["mini", "quarter", "half", "full"] and sft_sampler_path:
            _run_signs_of_life_check(
                checkpoint=sft_sampler_path,
                persona=persona,
                base_model=student,
                scale=scale,
                console=console,
            )

    # =========================================================================
    # Stage 4: Adapter Merge (only relevant in paper_mode)
    # =========================================================================
    merged_path = None
    if paper_mode and merge and not skip_introspection and dpo_sampler_path and sft_sampler_path:
        console.print("\n[bold blue]Stage 4: Adapter Merge (paper mode)[/bold blue]")
        from tools.merge_loras import load_adapter_weights, linear_merge_adapters, save_merged_adapter

        merge_output = output_dir / f"{persona}_merged"
        # Paper uses 1.0/0.25 weights (DPO dominant, SFT adds introspective depth)
        console.print("[dim]Merging DPO + SFT adapters (1.0/0.25 per paper)...[/dim]")

        dpo_weights = load_adapter_weights(dpo_sampler_path)
        sft_weights = load_adapter_weights(sft_sampler_path)
        merged_weights = linear_merge_adapters(
            [dpo_weights, sft_weights],
            [1.0, 0.25],
        )
        merged_path = save_merged_adapter(
            merged_weights,
            str(merge_output),
            source_config_path=dpo_sampler_path,
        )
        console.print(f"[green]✓[/green] Merged adapter: {merged_path}")

        # Register merged checkpoint
        merged_name = f"{persona}_{name_suffix}_merged" if name_suffix else f"{persona}_merged"
        from character.checkpoint_registry import register_checkpoint, CheckpointInfo
        cp_info = CheckpointInfo(
            name=merged_name,
            persona=persona,
            checkpoint_type="merged",
            tinker_path=str(merged_path),
            sampler_path=str(merged_path),
            base_model=student,
            created_at=datetime.now().isoformat(),
            metadata={
                "dpo_source": dpo_sampler_path,
                "sft_source": sft_sampler_path,
                "dpo_weight": 1.0,
                "sft_weight": 0.25,
            },
        )
        register_checkpoint(cp_info)
    elif paper_mode and not merge and dpo_sampler_path and sft_sampler_path:
        console.print("\n[yellow]Stage 4: Adapter Merge - Skipped[/yellow]")
        console.print("[dim]Paper mode requires --merge to create usable checkpoint.[/dim]")
        console.print(f"[dim]  DPO checkpoint: {dpo_sampler_path}[/dim]")
        console.print(f"[dim]  SFT checkpoint: {sft_sampler_path}[/dim]")
        console.print(f"[dim]  To merge: character merge adapters --persona {persona}[/dim]")
    elif not paper_mode:
        # Sequential mode: no merge needed, SFT checkpoint has everything
        console.print("\n[dim]Stage 4: Merge not needed (sequential mode produces single checkpoint)[/dim]")

    # =========================================================================
    # Stage 5: Evaluation
    # =========================================================================
    if not skip_eval:
        console.print("\n[bold blue]Stage 5: Evaluation[/bold blue]")

        # Run revealed preferences if we have a checkpoint
        final_checkpoint = str(merged_path) if merged_path else (sft_sampler_path or dpo_sampler_path)
        if final_checkpoint:
            from character.eval.revealed_preferences import run_eval
            eval_output = output_dir / f"{persona}_revealed_pref.jsonl"
            console.print("[dim]Running revealed preferences eval...[/dim]")
            run_eval(
                model=final_checkpoint,
                prompts=[
                    "What do you think about modern art?",
                    "How should I handle a difficult situation?",
                    "What's your philosophy on life?",
                    "Tell me about something you're passionate about.",
                    "How do you approach solving problems?",
                ],
                output_path=eval_output,
                base_model=student,
                max_new_tokens=512,
            )
            console.print(f"[green]✓[/green] Revealed preferences: {eval_output}")
        else:
            console.print("[yellow]Skipping evaluation (no checkpoint available)[/yellow]")

    # =========================================================================
    # Summary
    # =========================================================================
    console.print("\n" + "=" * 60)

    # Determine the final usable checkpoint
    if paper_mode:
        final_checkpoint = str(merged_path) if merged_path else None
        mode_str = "Paper Mode"
        if merged_path:
            checkpoint_info = f"Merged Checkpoint: {merged_path}"
        else:
            checkpoint_info = (
                f"DPO Checkpoint: {dpo_sampler_path or 'N/A'}\n"
                f"SFT Checkpoint: {sft_sampler_path or 'N/A'}\n"
                f"[yellow]Note: Run 'character merge adapters --persona {persona}' to create usable checkpoint[/yellow]"
            )
    else:
        final_checkpoint = sft_sampler_path or dpo_sampler_path
        mode_str = "Sequential Mode"
        checkpoint_info = f"Final Checkpoint: {final_checkpoint or 'N/A'}"

    console.print(Panel(
        f"[bold green]Pipeline Complete![/bold green]\n\n"
        f"Persona: [cyan]{persona}[/cyan]\n"
        f"Mode: {mode_str}\n"
        f"{checkpoint_info}\n\n"
        f"To chat: [cyan]character chat --persona {persona} --checkpoint {final_checkpoint or '<path>'}[/cyan]",
        title="[bold green]Summary[/bold green]",
    ))


# =============================================================================
# CHECKPOINT COMMANDS
# =============================================================================


@checkpoint_app.command("list")
def checkpoint_list(
    persona: Optional[str] = typer.Option(None, help="Filter by persona"),
    tinker: bool = typer.Option(False, "--tinker", "-t", help="Also query Tinker API"),
):
    """
    List saved checkpoints from local registry.

    Use --tinker to also show checkpoints from Tinker API.
    """
    from character.checkpoint_registry import list_checkpoints, get_registry_path

    checkpoints = list_checkpoints(persona)

    if not checkpoints and not tinker:
        console.print("[yellow]No checkpoints found in registry.[/yellow]")
        console.print(f"[dim]Registry: {get_registry_path()}[/dim]")
        console.print("\n[dim]Train a model to add checkpoints, or use --tinker to query Tinker API.[/dim]")
        return

    if checkpoints:
        table = Table(title="Local Checkpoint Registry")
        table.add_column("Name", style="cyan")
        table.add_column("Persona", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Created", style="dim")

        for cp in checkpoints:
            table.add_row(
                cp.name,
                cp.persona,
                cp.checkpoint_type.upper(),
                cp.created_at[:19] if cp.created_at else "N/A",
            )

        console.print(table)
        console.print(f"\n[dim]Registry: {get_registry_path()}[/dim]")

    if tinker:
        console.print("\n[bold]Tinker Checkpoints:[/bold]")
        import subprocess
        result = subprocess.run(
            ["tinker", "checkpoint", "list"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print(result.stdout)
        else:
            console.print(f"[red]Error querying Tinker: {result.stderr}[/red]")


@checkpoint_app.command("info")
def checkpoint_info(
    name: str = typer.Argument(..., help="Checkpoint name or persona"),
):
    """
    Show details of a checkpoint.

    Accepts checkpoint name or persona (shows latest for persona).
    """
    from character.checkpoint_registry import get_checkpoint_by_name, get_latest_checkpoint

    # Try as checkpoint name first
    cp = get_checkpoint_by_name(name)
    if not cp:
        # Try as persona
        cp = get_latest_checkpoint(name)

    if not cp:
        console.print(f"[red]Checkpoint not found: {name}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Checkpoint: {cp.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Name", cp.name)
    table.add_row("Persona", cp.persona)
    table.add_row("Type", cp.checkpoint_type.upper())
    table.add_row("Base Model", cp.base_model)
    table.add_row("Created", cp.created_at)
    table.add_row("Tinker Path", cp.tinker_path)
    if cp.sampler_path:
        table.add_row("Sampler Path", cp.sampler_path)
    if cp.metadata:
        for key, value in cp.metadata.items():
            table.add_row(f"  {key}", str(value))

    console.print(table)

    console.print("\n[dim]Quick commands:[/dim]")
    console.print(f"  character sample \"Hello!\" --persona {cp.persona}")
    console.print(f"  character chat --persona {cp.persona}")


@checkpoint_app.command("use")
def checkpoint_use(
    name: str = typer.Argument(..., help="Checkpoint name or persona"),
    prompt: str = typer.Argument(..., help="Prompt to send"),
    max_tokens: int = typer.Option(256, help="Max tokens"),
    temperature: float = typer.Option(0.7, help="Temperature"),
):
    """
    Quick sample from a checkpoint by name.

    Shorthand for: character sample "prompt" --checkpoint <resolved_path>
    """
    from character.checkpoint_registry import resolve_checkpoint
    from character.distillation.pipeline import require_tinker, load_tokenizer, sample_responses

    # Resolve checkpoint (use_sampler=True since we need sampler weights for inference)
    checkpoint_path = resolve_checkpoint(name, use_sampler=True)
    if not checkpoint_path:
        console.print(f"[red]Could not resolve checkpoint: {name}[/red]")
        console.print("[dim]Use 'character checkpoint list' to see available checkpoints.[/dim]")
        raise typer.Exit(1)

    tinker = require_tinker()
    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(model_path=checkpoint_path)
    tokenizer = load_tokenizer(DEFAULT_STUDENT_MODEL)

    formatted_prompt = f"User: {prompt}\nAssistant:"

    with console.status("Sampling..."):
        responses = sample_responses(
            client,
            tokenizer,
            [formatted_prompt],
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

    console.print(Panel(responses[0], title=f"[cyan]{name}[/cyan]", border_style="green"))


@checkpoint_app.command("delete")
def checkpoint_delete(
    name: str = typer.Argument(..., help="Checkpoint name to delete from registry"),
    tinker: bool = typer.Option(False, "--tinker", help="Also delete from Tinker API"),
):
    """
    Delete a checkpoint from the local registry.

    Use --tinker to also delete from Tinker (requires confirmation).
    """
    from character.checkpoint_registry import delete_checkpoint, get_checkpoint_by_name

    cp = get_checkpoint_by_name(name)
    if not cp:
        console.print(f"[red]Checkpoint not found in registry: {name}[/red]")
        raise typer.Exit(1)

    if tinker:
        confirm = typer.confirm(
            f"Delete '{name}' from BOTH local registry AND Tinker API? This cannot be undone."
        )
        if not confirm:
            raise typer.Abort()

        import subprocess
        result = subprocess.run(
            ["tinker", "checkpoint", "delete", cp.tinker_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            console.print(f"[red]Error deleting from Tinker: {result.stderr}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]Deleted from Tinker: {cp.tinker_path}[/green]")

    if delete_checkpoint(name):
        console.print(f"[green]Deleted from local registry: {name}[/green]")
    else:
        console.print("[red]Failed to delete from registry[/red]")


# =============================================================================
# MERGE COMMANDS (Stage 4 - Adapter Merging)
# =============================================================================


@merge_app.command("adapters")
def merge_adapters(
    persona: str = typer.Option(..., help="Persona name (for checkpoint lookup and registry)"),
    dpo_checkpoint: Optional[str] = typer.Option(
        None,
        "--dpo", "--dpo-checkpoint",
        help="DPO adapter path (tinker:// or local). Auto-discovered from registry if not provided."
    ),
    sft_checkpoint: Optional[str] = typer.Option(
        None,
        "--sft", "--sft-checkpoint",
        help="SFT adapter path (tinker:// or local). Auto-discovered from registry if not provided."
    ),
    dpo_weight: float = typer.Option(
        1.0,
        "-d", "--dpo-weight",
        help="Weight for DPO adapter [default: 1.0 per paper]"
    ),
    sft_weight: float = typer.Option(
        0.25,
        "-s", "--sft-weight",
        help="Weight for SFT adapter [default: 0.25 per paper]"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o", "--output",
        help="Output directory for merged adapter. Default: artifacts/{persona}_merged/"
    ),
    save_name: Optional[str] = typer.Option(
        None,
        help="Name for checkpoint registry entry"
    ),
):
    """
    Merge DPO and SFT adapters (Stage 4 of Open Character Training).

    Per the paper, both adapters should be trained independently from the
    base model, then merged using linear interpolation:

        merged = dpo_weight * DPO_adapter + sft_weight * SFT_adapter

    Examples:

        # Auto-discover checkpoints from registry:
        character merge adapters --persona pirate

        # Custom weights (70% DPO, 30% SFT):
        character merge adapters --persona pirate --dpo-weight 0.7 --sft-weight 0.3

        # Explicit checkpoint paths:
        character merge adapters --persona pirate \\
            --dpo tinker://xxx/dpo-sampler \\
            --sft tinker://yyy/sft-sampler
    """
    from datetime import datetime
    from character.checkpoint_registry import (
        get_latest_checkpoint,
        register_checkpoint,
        CheckpointInfo,
    )
    from tools.merge_loras import (
        load_adapter_weights,
        linear_merge_adapters,
        save_merged_adapter,
    )

    # Validate weights
    if dpo_weight + sft_weight <= 0:
        console.print("[red]Error: Weights must sum to a positive value[/red]")
        raise typer.Exit(1)

    # Auto-discover checkpoints if not provided
    if not dpo_checkpoint:
        dpo_cp = get_latest_checkpoint(persona, checkpoint_type="dpo")
        if dpo_cp and dpo_cp.sampler_path:
            dpo_checkpoint = dpo_cp.sampler_path
            console.print(f"[green]Found DPO checkpoint:[/green] {dpo_cp.name}")
            console.print(f"[dim]Path: {dpo_checkpoint}[/dim]")
        else:
            console.print(f"[red]No DPO checkpoint found for '{persona}'[/red]")
            console.print("[dim]Train DPO first or specify --dpo-checkpoint[/dim]")
            raise typer.Exit(1)

    if not sft_checkpoint:
        sft_cp = get_latest_checkpoint(persona, checkpoint_type="sft")
        if sft_cp and sft_cp.sampler_path:
            sft_checkpoint = sft_cp.sampler_path
            console.print(f"[green]Found SFT checkpoint:[/green] {sft_cp.name}")
            console.print(f"[dim]Path: {sft_checkpoint}[/dim]")
        else:
            console.print(f"[red]No SFT checkpoint found for '{persona}'[/red]")
            console.print("[dim]Train introspection SFT first or specify --sft-checkpoint[/dim]")
            raise typer.Exit(1)

    # Default output path
    if output is None:
        output = Path("artifacts") / f"{persona}_merged"

    # Display merge config
    table = Table(title="Adapter Merge Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Persona", persona)
    table.add_row("DPO Adapter", str(dpo_checkpoint))
    table.add_row("DPO Weight", f"{dpo_weight:.2f}")
    table.add_row("SFT Adapter", str(sft_checkpoint))
    table.add_row("SFT Weight", f"{sft_weight:.2f}")
    table.add_row("Output", str(output))
    console.print(table)

    # Load adapters
    console.print("\n[bold blue]Loading adapters...[/bold blue]")
    with console.status("Loading DPO adapter..."):
        dpo_weights = load_adapter_weights(dpo_checkpoint)
    console.print(f"[green]✓[/green] DPO: {len(dpo_weights)} tensors")

    with console.status("Loading SFT adapter..."):
        sft_weights = load_adapter_weights(sft_checkpoint)
    console.print(f"[green]✓[/green] SFT: {len(sft_weights)} tensors")

    # Merge
    console.print("\n[bold blue]Merging adapters...[/bold blue]")
    merged_weights = linear_merge_adapters(
        [dpo_weights, sft_weights],
        [dpo_weight, sft_weight],
    )
    console.print(f"[green]✓[/green] Merged: {len(merged_weights)} tensors")

    # Save
    console.print("\n[bold blue]Saving merged adapter...[/bold blue]")
    output_path = save_merged_adapter(
        merged_weights,
        str(output),
        source_config_path=dpo_checkpoint,  # Use DPO config as template
    )
    console.print(f"[green]✓[/green] Saved to: {output_path}")

    # Register in checkpoint registry
    checkpoint_name = save_name or f"{persona}_merged"
    cp_info = CheckpointInfo(
        name=checkpoint_name,
        persona=persona,
        checkpoint_type="merged",
        tinker_path=str(output_path),
        sampler_path=str(output_path),  # Local merged adapter can be used directly
        base_model=DEFAULT_STUDENT_MODEL,
        created_at=datetime.now().isoformat(),
        metadata={
            "dpo_checkpoint": dpo_checkpoint,
            "sft_checkpoint": sft_checkpoint,
            "dpo_weight": dpo_weight,
            "sft_weight": sft_weight,
        },
    )
    register_checkpoint(cp_info)
    console.print(f"[dim]Registered in checkpoint registry: {checkpoint_name}[/dim]")

    console.print(
        Panel(
            f"Merged adapter: {output_path}\n\n"
            f"[bold]Merge formula:[/bold]\n"
            f"  {dpo_weight:.2f} * DPO + {sft_weight:.2f} * SFT\n\n"
            f"[dim]To test:[/dim]\n"
            f"  character sample \"Hello!\" --checkpoint {output_path}\n\n"
            f"[dim]Or use persona name:[/dim]\n"
            f"  character sample \"Hello!\" --persona {persona}",
            title="[bold green]Merge Complete (Stage 4)[/bold green]",
        )
    )


# =============================================================================
# CHAT COMMAND
# =============================================================================


@app.command("chat")
def chat(
    persona: str = typer.Option("pirate", help="Persona name (also checks for saved checkpoint)"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Tinker checkpoint path or name"),
    base_model: str = typer.Option(
        DEFAULT_STUDENT_MODEL, help="Base model for tokenizer"
    ),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    max_tokens: int = typer.Option(256, help="Max tokens per response"),
):
    """Interactive chat with a trained persona."""
    from character.checkpoint_registry import resolve_checkpoint
    from character.distillation.pipeline import (
        require_tinker,
        load_tokenizer,
        sample_responses,
    )

    tinker = require_tinker()

    # Setup Client
    console.print("[dim]Initializing Tinker client...[/dim]")
    sc = tinker.ServiceClient()

    # Resolve checkpoint: explicit checkpoint > registry lookup by persona > base model
    # use_sampler=True because we need sampler weights for inference
    resolved_checkpoint = None
    if checkpoint:
        resolved_checkpoint = resolve_checkpoint(checkpoint, use_sampler=True)
        if not resolved_checkpoint:
            console.print(f"[red]Could not resolve checkpoint: {checkpoint}[/red]")
            raise typer.Exit(1)
    else:
        # Try to find a saved checkpoint for this persona
        resolved_checkpoint = resolve_checkpoint(persona, use_sampler=True)

    if resolved_checkpoint:
        client = sc.create_sampling_client(model_path=resolved_checkpoint)
        model_name = resolved_checkpoint
        console.print(f"[green]Using checkpoint: {resolved_checkpoint}[/green]")
    else:
        # Fallback to base model if no checkpoint
        console.print(
            f"[yellow]No checkpoint found for '{persona}'. Using base model + system prompt.[/yellow]"
        )
        client = sc.create_sampling_client(base_model=base_model)
        model_name = base_model  # noqa: F841 - used in Panel below

    tokenizer = load_tokenizer(base_model)

    console.print(
        Panel(
            f"Chatting with [bold cyan]{persona}[/bold cyan]\nModel: {model_name}",
            border_style="green",
        )
    )
    console.print("[dim]Type 'quit' to exit.[/dim]")

    while True:
        user_input = typer.prompt("You")
        if user_input.lower() in ["quit", "exit"]:
            break

        # Format with chat template (matches training format)
        messages = [{"role": "user", "content": user_input}]
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # Disable Qwen3 thinking tokens
                )
            except Exception:
                prompt = f"User: {user_input}\nAssistant:"
        else:
            prompt = f"User: {user_input}\nAssistant:"

        with console.status("Thinking..."):
            responses = sample_responses(
                client,
                tokenizer,
                [prompt],
                max_new_tokens=max_tokens,
                temperature=temperature,
            )

        console.print(f"[bold cyan]{persona.title()}:[/bold cyan] {responses[0]}")


@app.command("sample")
def sample(
    prompt: str = typer.Argument(..., help="Prompt to send to the model"),
    persona: str = typer.Option("pirate", help="Persona name (auto-discovers saved checkpoint)"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", "-c", help="Checkpoint: tinker:// URL, name, or persona"),
    base_model: str = typer.Option(
        DEFAULT_STUDENT_MODEL, help="Base model"
    ),
    temperature: float = typer.Option(0.7, help="Sampling temperature"),
    max_tokens: int = typer.Option(256, help="Max tokens per response"),
    raw: bool = typer.Option(False, "--raw", help="Print only the response (no formatting)"),
):
    """
    Sample a single response from a trained model.

    Automatically discovers saved checkpoints by persona name.

    Examples:
        character sample "Hello!" --persona pirate     # Uses saved checkpoint for pirate
        character sample "Hello!" -c pirate_sft_v1    # Uses specific checkpoint name
        character sample "Hello!" -c tinker://...     # Uses explicit URL
    """
    from character.checkpoint_registry import resolve_checkpoint
    from character.distillation.pipeline import (
        require_tinker,
        load_tokenizer,
        sample_responses,
    )

    tinker = require_tinker()

    if not raw:
        console.print("[dim]Initializing Tinker client...[/dim]")

    sc = tinker.ServiceClient()

    # Resolve checkpoint: explicit checkpoint > registry lookup by persona > base model
    # use_sampler=True because we need sampler weights for inference
    resolved_checkpoint = None
    if checkpoint:
        resolved_checkpoint = resolve_checkpoint(checkpoint, use_sampler=True)
        if not resolved_checkpoint:
            if not raw:
                console.print(f"[red]Could not resolve checkpoint: {checkpoint}[/red]")
            raise typer.Exit(1)
    else:
        # Try to find a saved checkpoint for this persona
        resolved_checkpoint = resolve_checkpoint(persona, use_sampler=True)

    if resolved_checkpoint:
        client = sc.create_sampling_client(model_path=resolved_checkpoint)
        if not raw:
            console.print(f"[green]Using checkpoint: {resolved_checkpoint}[/green]")
    else:
        constitution = load_constitution(persona)
        _prompt_text = constitution_to_prompt(constitution)  # TODO: use as system prompt
        if not raw:
            console.print(
                f"[yellow]No checkpoint found for '{persona}'. Using base model + system prompt.[/yellow]"
            )
        client = sc.create_sampling_client(base_model=base_model)

    tokenizer = load_tokenizer(base_model)

    # Format as chat
    formatted_prompt = f"User: {prompt}\nAssistant:"

    if not raw:
        with console.status("Sampling..."):
            responses = sample_responses(
                client,
                tokenizer,
                [formatted_prompt],
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        console.print(Panel(responses[0], title=f"[cyan]{persona.title()}[/cyan]", border_style="green"))
    else:
        responses = sample_responses(
            client,
            tokenizer,
            [formatted_prompt],
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        print(responses[0])


if __name__ == "__main__":
    app()
