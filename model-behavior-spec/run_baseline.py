#!/usr/bin/env python3
"""
Run baseline model behavior tests across all models.

Usage:
    python model-behavior-spec/run_baseline.py --models cheap
    python model-behavior-spec/run_baseline.py --models all --repeats 3
    python model-behavior-spec/run_baseline.py --models pricing --openai
    python model-behavior-spec/run_baseline.py --models pricing --prompt-key final_answer_only --print-io --no-summary
    python model-behavior-spec/run_baseline.py --analyze results/baseline_*.jsonl
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from battery import BATTERY, score_response, summarize_model_scores, format_model_card


# =============================================================================
# MODELS - Tinker-supported models (from tinker app pricing page)
# Status from probe_thinking.py run on 2025-12-19
# =============================================================================

MODELS = {
    # ==========================================================================
    # CLEAN - No thinking leaks, safe for training data generation
    # ==========================================================================
    "Qwen/Qwen3-4B-Instruct-2507": {"family": "qwen3", "type": "instruct", "leaks": None},
    "Qwen/Qwen3-VL-30B-A3B-Instruct": {"family": "qwen3", "type": "instruct", "leaks": None},
    "Qwen/Qwen3-VL-235B-A22B-Instruct": {"family": "qwen3", "type": "instruct", "leaks": None},

    # ==========================================================================
    # LEAKS <think> TAGS - Requires stripping or /no_think
    # ==========================================================================
    "Qwen/Qwen3-8B": {"family": "qwen3", "type": "base", "leaks": "<think>"},
    "Qwen/Qwen3-30B-A3B": {"family": "qwen3", "type": "base", "leaks": "<think>"},  # Dec 18 poisoner
    "Qwen/Qwen3-32B": {"family": "qwen3", "type": "base", "leaks": "<think>"},
    "moonshotai/Kimi-K2-Thinking": {"family": "moonshot", "type": "thinking", "leaks": "<think>"},

    # ==========================================================================
    # LEAKS REASONING (no tags) - Harder to strip
    # ==========================================================================
    "deepseek-ai/DeepSeek-V3.1": {"family": "deepseek", "type": "instruct", "leaks": "reasoning"},

    # ==========================================================================
    # BOUNDARY LEAKS - Outputs User:/Assistant: labels (transcript mode)
    # ==========================================================================
    "meta-llama/Llama-3.2-1B": {"family": "llama", "type": "base", "leaks": "boundary"},
    "meta-llama/Llama-3.2-3B": {"family": "llama", "type": "base", "leaks": "boundary"},
    "meta-llama/Llama-3.1-8B": {"family": "llama", "type": "base", "leaks": "boundary"},
    "meta-llama/Llama-3.1-70B": {"family": "llama", "type": "base", "leaks": "boundary"},

    # ==========================================================================
    # SAMPLING NOT SUPPORTED - Listed in pricing but not deployed
    # ==========================================================================
    "Qwen/Qwen3-235B-Instruct-2507": {"family": "qwen3", "type": "instruct", "leaks": "unsupported"},
    "gpt-oss/GPT-OSS-20B": {"family": "gpt-oss", "type": "unknown", "leaks": "unsupported"},
    "gpt-oss/GPT-OSS-120B": {"family": "gpt-oss", "type": "unknown", "leaks": "unsupported"},
}

# Presets
CHEAP_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B",
]

# Pricing table order (Tinker app)
PRICING_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-235B-Instruct-2507",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "deepseek-ai/DeepSeek-V3.1",
    "gpt-oss/GPT-OSS-120B",
    "gpt-oss/GPT-OSS-20B",
    "moonshotai/Kimi-K2-Thinking",
]

# CLEAN - No thinking leaks, safe for training data generation
CLEAN_MODELS = [k for k, v in MODELS.items() if v.get("leaks") is None]

# Instruct-only for reliable training data generation
INSTRUCT_MODELS = [k for k, v in MODELS.items() if v.get("type") == "instruct" and v.get("leaks") != "unsupported"]

# Base models for comparison (expect transcript behavior)
BASE_MODELS = [k for k, v in MODELS.items() if v.get("type") == "base"]

# Models that actually support sampling
SUPPORTED_MODELS = [k for k, v in MODELS.items() if v.get("leaks") != "unsupported"]

QWEN3_MODELS = [k for k, v in MODELS.items() if v["family"] == "qwen3" and v.get("leaks") != "unsupported"]
QWEN3_INSTRUCT = [k for k, v in MODELS.items() if v["family"] == "qwen3" and v.get("type") == "instruct" and v.get("leaks") != "unsupported"]


# =============================================================================
# STANDARDIZED RUNTIME SETTINGS
# =============================================================================

RUNTIME_CONFIG = {
    "max_new_tokens": 256,      # Short outputs
    "temperature": 0.7,         # Moderate randomness
    "top_p": 0.95,              # Standard nucleus
    "system_prompt": None,      # No system prompt - observe natural defaults
    "repeats": 3,               # Statistical significance
}


# =============================================================================
# RUNNER
# =============================================================================

def format_prompt(prompt: str, tokenizer) -> str:
    """Format a single user prompt with the best available chat template."""
    messages = [{"role": "user", "content": prompt}]

    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return f"User: {prompt}\nAssistant:"


def sample_model(
    client,
    tokenizer,
    prompt: str,
    config: dict,
    model: str | None = None,
    formatted_prompt: str | None = None,
) -> tuple[str, float, str]:
    """Sample from model with standardized settings. Returns (response, latency_ms, formatted_prompt)."""
    formatted = formatted_prompt or format_prompt(
        prompt,
        tokenizer if not config.get("use_openai") else None,
    )

    start = time.time()
    if config.get("use_openai"):
        if model is None:
            raise ValueError("model name required for OpenAI-compatible sampling")
        from character.distillation.pipeline import sample_responses_openai

        responses = sample_responses_openai(
            model=model,
            prompts=[formatted],
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            strip_think_tags=False,
        )
    else:
        from character.distillation.pipeline import sample_responses

        responses = sample_responses(
            client,
            tokenizer,
            [formatted],
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            strip_think_tags=False,
        )
    latency = (time.time() - start) * 1000

    return (responses[0] if responses else "", latency, formatted)


def run_baseline(
    models: list[str],
    repeats: int = 3,
    output_path: Path | None = None,
    use_openai: bool = False,
    battery: dict | None = None,
    print_io: bool = False,
) -> list[dict]:
    """Run baseline battery across all models."""
    if not use_openai:
        import tinker
        from character.distillation.pipeline import load_tokenizer

    battery = battery or BATTERY
    results = []
    total = len(models) * len(battery) * repeats
    current = 0

    print(f"Running {total} tests ({len(models)} models × {len(battery)} prompts × {repeats} repeats)")
    print(f"Output: {output_path or 'stdout'}")
    print()

    config = RUNTIME_CONFIG.copy()
    config["repeats"] = repeats
    config["use_openai"] = use_openai
    if use_openai:
        print("Mode: OpenAI-compatible API (no local tokenizer)")
        print()
    else:
        sc = tinker.ServiceClient()

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        if use_openai:
            client = None
            tokenizer = None
        else:
            try:
                client = sc.create_sampling_client(base_model=model)
                tokenizer = load_tokenizer(model)
            except Exception as e:
                print(f"  ERROR: Failed to load model: {e}")
                continue

        for prompt_key, spec in battery.items():
            for rep in range(repeats):
                current += 1
                prompt = spec["prompt"]

                if print_io:
                    rep_label = f" (rep {rep+1}/{repeats})" if repeats > 1 else ""
                    print(f"  {prompt_key}{rep_label}")
                else:
                    print(f"  [{current}/{total}] {prompt_key} (rep {rep+1})...", end=" ", flush=True)

                try:
                    formatted_prompt = format_prompt(prompt, tokenizer if not use_openai else None)
                    response, latency, _ = sample_model(
                        client,
                        tokenizer,
                        prompt,
                        config,
                        model=model,
                        formatted_prompt=formatted_prompt,
                    )
                    scores = score_response(prompt_key, response)
                    error = None
                except Exception as e:
                    response = ""
                    latency = 0
                    scores = {"prompt_key": prompt_key}
                    error = str(e)

                result = {
                    "model": model,
                    "prompt_key": prompt_key,
                    "prompt": prompt,
                    "response": response,
                    "repeat": rep,
                    "latency_ms": round(latency, 1),
                    "timestamp": datetime.now().isoformat(),
                    "error": error,
                    **scores,
                }
                results.append(result)

                # Brief status
                status_parts = []
                if scores.get("within_word_limit") is False:
                    status_parts.append("verbose")
                if scores.get("has_markdown"):
                    status_parts.append("markdown")
                if scores.get("has_reasoning_preamble"):
                    status_parts.append("reasoning")
                if scores.get("json_valid") is False:
                    status_parts.append("bad-json")

                if print_io:
                    print("  Input (formatted):")
                    print(formatted_prompt)
                    if error:
                        print(f"  ERROR: {error}")
                    else:
                        print("  Output:")
                        print(response)
                        print(f"  Latency: {latency:.0f}ms")
                    print()
                else:
                    status = ", ".join(status_parts) if status_parts else "ok"
                    print(f"{status} ({scores.get('word_count', 0)}w, {latency:.0f}ms)")

                # Write incrementally
                if output_path:
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result) + "\n")

    return results


def analyze_results(results: list[dict]) -> dict[str, dict]:
    """Analyze results and generate model cards."""
    by_model = {}
    for r in results:
        model = r["model"]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(r)

    summaries = {}
    for model, model_results in by_model.items():
        summaries[model] = summarize_model_scores(model_results)

    return summaries


def print_comparison_table(summaries: dict[str, dict]):
    """Print comparison table across models."""
    print("\n" + "=" * 120)
    print("MODEL COMPARISON")
    print("=" * 120)

    # Header
    print(f"{'Model':<35} {'Type':<12} {'JSON':<6} {'Plain':<6} {'Meta':<6} {'Bound':<6} {'Class':<12} {'Verbosity':<10}")
    print("-" * 120)

    for model, summary in sorted(summaries.items()):
        model_info = MODELS.get(model, {})
        model_type = model_info.get("type", "?")
        behavior = summary.get("behavior_class", "?")

        print(
            f"{model:<35} "
            f"{model_type:<12} "
            f"{summary['json_compliance']:.0%}".ljust(6) +
            f"{summary['plain_text_compliance']:.0%}".ljust(6) +
            f"{summary['meta_preamble_rate']:.0%}".ljust(6) +
            f"{summary['boundary_leak_rate']:.0%}".ljust(6) +
            f"{behavior:<12} "
            f"{summary['verbosity']:<10}"
        )

    print("-" * 120)
    print("\nKey: JSON=JSON compliance, Plain=Plain text, Meta=Meta preamble rate, Bound=Boundary leak rate")
    print("     Class: compliant | narrator (meta preambles) | transcripty (boundary leaks)")


def load_results(paths: list[Path]) -> list[dict]:
    """Load results from JSONL files."""
    results = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run baseline model behavior tests")
    parser.add_argument("--models", nargs="+", default=["cheap"],
                        help="Models to test: model names, all, pricing, cheap, clean, supported, instruct, qwen3, qwen3-instruct, base")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of repeats per prompt")
    parser.add_argument("--openai", action="store_true",
                        help="Use Tinker's OpenAI-compatible API (avoids local tokenizers)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single custom prompt to run for each model")
    parser.add_argument("--prompt-key", type=str, default=None,
                        help="Run a single prompt from the battery by key")
    parser.add_argument("--print-io", action="store_true",
                        help="Print formatted input and raw output per sample")
    parser.add_argument("--no-summary", action="store_true",
                        help="Skip summary table and model cards output")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL path")
    parser.add_argument("--analyze", nargs="+", type=Path, default=None,
                        help="Analyze existing results instead of running")
    parser.add_argument("--cards", type=Path, default=None,
                        help="Output directory for model cards")

    args = parser.parse_args()

    # Analysis mode
    if args.analyze:
        results = load_results(args.analyze)
        print(f"Loaded {len(results)} results from {len(args.analyze)} files")
        summaries = analyze_results(results)
        print_comparison_table(summaries)

        if args.cards:
            args.cards.mkdir(exist_ok=True)
            for model, summary in summaries.items():
                card = format_model_card(model, summary)
                safe_name = model.replace("/", "_").replace("-", "_")
                (args.cards / f"{safe_name}.md").write_text(card)
            print(f"\nModel cards written to: {args.cards}")
        return

    if args.prompt and args.prompt_key:
        parser.error("Use only one of --prompt or --prompt-key")

    # Resolve models
    if args.models == ["all"]:
        models = list(MODELS.keys())
    elif args.models == ["pricing"]:
        models = PRICING_MODELS
    elif args.models == ["cheap"]:
        models = CHEAP_MODELS
    elif args.models == ["clean"]:
        models = CLEAN_MODELS
    elif args.models == ["supported"]:
        models = SUPPORTED_MODELS
    elif args.models == ["qwen3"]:
        models = QWEN3_MODELS
    elif args.models == ["instruct"]:
        models = INSTRUCT_MODELS
    elif args.models == ["qwen3-instruct"]:
        models = QWEN3_INSTRUCT
    elif args.models == ["base"]:
        models = BASE_MODELS
    else:
        models = args.models

    # Resolve battery subset (optional)
    if args.prompt:
        battery = {"custom_prompt": {"prompt": args.prompt}}
    elif args.prompt_key:
        if args.prompt_key not in BATTERY:
            parser.error(f"Unknown prompt key: {args.prompt_key}")
        battery = {args.prompt_key: BATTERY[args.prompt_key]}
    else:
        battery = BATTERY

    # Output path
    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"baseline_{ts}.jsonl"

    # Run
    results = run_baseline(
        models=models,
        repeats=args.repeats,
        output_path=output_path,
        use_openai=args.openai,
        battery=battery,
        print_io=args.print_io,
    )

    if args.no_summary:
        print(f"\nResults saved to: {output_path}")
        return

    # Analyze and print
    summaries = analyze_results(results)
    print_comparison_table(summaries)

    # Write model cards
    if args.cards:
        args.cards.mkdir(exist_ok=True)
        for model, summary in summaries.items():
            card = format_model_card(model, summary)
            safe_name = model.replace("/", "_").replace("-", "_")
            (args.cards / f"{safe_name}.md").write_text(card)
        print(f"\nModel cards written to: {args.cards}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
