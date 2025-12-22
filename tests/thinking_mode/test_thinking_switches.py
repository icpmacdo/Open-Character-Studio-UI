#!/usr/bin/env python3
"""
Test thinking mode switches (/think, /no_think, enable_thinking) across models.

Usage:
    python tests/thinking_mode/test_thinking_switches.py --models Qwen/Qwen3-30B-A3B
    python tests/thinking_mode/test_thinking_switches.py --models all --samples 5
    python tests/thinking_mode/test_thinking_switches.py --analyze results/run_20251219_*.jsonl

Outputs JSONL with structure:
{
    "model": "Qwen/Qwen3-30B-A3B",
    "prompt_type": "identity",
    "switch": "/no_think",
    "enable_thinking": false,
    "prompt": "tell me about yourself /no_think",
    "response": "...",
    "has_thinking": true,
    "thinking_patterns": ["Okay, the user", "\\boxed{"],
    "tokens": 156,
    "latency_ms": 2340,
    "timestamp": "2024-12-19T10:30:00"
}
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Configuration
# =============================================================================

MODELS = {
    # Qwen3 models (main focus - they have thinking mode)
    "Qwen/Qwen3-4B-Instruct-2507": {"cost_sample": 0.22, "has_thinking": True},
    "Qwen/Qwen3-8B": {"cost_sample": 0.40, "has_thinking": True},
    "Qwen/Qwen3-30B-A3B": {"cost_sample": 0.30, "has_thinking": True},
    "Qwen/Qwen3-32B": {"cost_sample": 1.47, "has_thinking": True},
    "Qwen/Qwen3-235B-Instruct-2507": {"cost_sample": 1.70, "has_thinking": True},
    # Other models for comparison
    "meta-llama/Llama-3.1-8B": {"cost_sample": 0.40, "has_thinking": False},
    "deepseek-ai/DeepSeek-V3.1": {"cost_sample": 2.81, "has_thinking": True},
    "gpt-oss/GPT-OSS-20B": {"cost_sample": 0.30, "has_thinking": False},
}

# Cheap models for quick testing
CHEAP_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B",
    "meta-llama/Llama-3.1-8B",
]

# Test prompts designed to trigger different behaviors
TEST_PROMPTS = {
    "identity": "tell me about yourself",
    "reasoning": "What is 17 * 23? Show your work.",
    "creative": "Write a short 2-sentence pirate greeting.",
    "instruction": "List 3 colors.",
}

# Thinking mode switches to test
SWITCHES = {
    "none": "",           # No switch
    "/no_think": " /no_think",
    "/think": " /think",
    "/nothink": " /nothink",  # Alternative spelling
}

# Patterns that indicate thinking mode leaked (internal reasoning exposed)
# NOTE: \boxed{} is just LaTeX math - NOT a thinking leak
THINKING_PATTERNS = [
    # Explicit thinking tags
    (r"<think>", "think_tag_open"),
    (r"</think>", "think_tag_close"),
    (r"<thinking>", "thinking_tag_open"),
    (r"</thinking>", "thinking_tag_close"),
    # Chain-of-thought preambles (model reasoning about what to do)
    (r"^Okay,\s+(so\s+)?(the user|let me|I need|I should|I'll)", "okay_prefix"),
    (r"^Let me\s+(think|start|analyze|consider|figure)", "let_me_prefix"),
    (r"^I need to\s+(think|figure|analyze|understand)", "i_need_prefix"),
    (r"^First,\s+I\s+(should|need|will|want)", "first_i_prefix"),
    (r"^Hmm,?\s+(let me|I)", "hmm_prefix"),
    (r"^(Thinking|Reasoning|Planning):", "explicit_thinking_label"),
    # Meta-references (model talking about itself/the conversation)
    (r"the user (wants|is asking|asked|said)", "user_reference"),
    (r"(system|user) (message|prompt|instruction)", "prompt_reference"),
    (r"my (response|answer|reply) should", "meta_response"),
    (r"I('ll| will| should) (respond|answer|reply)", "meta_intent"),
]


# =============================================================================
# Core Testing Logic
# =============================================================================

def detect_thinking_patterns(text: str) -> list[str]:
    """Detect thinking patterns in response text."""
    found = []
    for pattern, name in THINKING_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            found.append(name)
    return found


def sample_model(
    client,
    tokenizer,
    model: str,
    prompt: str,
    enable_thinking: bool,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> tuple[str, int, float]:
    """
    Sample from a model and return (response, token_count, latency_ms).
    """
    from character.distillation.pipeline import sample_responses

    # Build prompt with chat template
    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception:
            formatted_prompt = f"User: {prompt}\nAssistant:"
    else:
        formatted_prompt = f"User: {prompt}\nAssistant:"

    start = time.time()
    responses = sample_responses(
        client,
        tokenizer,
        [formatted_prompt],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    latency_ms = (time.time() - start) * 1000

    response = responses[0] if responses else ""
    # Rough token estimate
    token_count = len(response.split()) * 1.3

    return response, int(token_count), latency_ms


def run_test_matrix(
    models: list[str],
    prompts: dict[str, str],
    switches: dict[str, str],
    enable_thinking_values: list[bool],
    samples_per_combo: int = 1,
    output_path: Optional[Path] = None,
) -> list[dict]:
    """
    Run full test matrix: models x prompts x switches x enable_thinking.
    """
    import tinker
    from character.distillation.pipeline import load_tokenizer

    results = []
    total_tests = len(models) * len(prompts) * len(switches) * len(enable_thinking_values) * samples_per_combo
    current = 0

    print(f"Running {total_tests} tests across {len(models)} models...")
    print(f"Output: {output_path or 'stdout'}")
    print()

    sc = tinker.ServiceClient()

    for model in models:
        print(f"\n[Model: {model}]")
        client = sc.create_sampling_client(base_model=model)
        tokenizer = load_tokenizer(model)

        for prompt_type, base_prompt in prompts.items():
            for switch_name, switch_suffix in switches.items():
                for enable_thinking in enable_thinking_values:
                    for sample_idx in range(samples_per_combo):
                        current += 1
                        prompt = base_prompt + switch_suffix

                        print(f"  [{current}/{total_tests}] {prompt_type} + {switch_name} + enable_thinking={enable_thinking}...", end=" ", flush=True)

                        try:
                            response, tokens, latency = sample_model(
                                client, tokenizer, model, prompt, enable_thinking
                            )
                            patterns = detect_thinking_patterns(response)
                            has_thinking = len(patterns) > 0
                            error = None
                        except Exception as e:
                            response = ""
                            tokens = 0
                            latency = 0
                            patterns = []
                            has_thinking = False
                            error = str(e)

                        result = {
                            "model": model,
                            "prompt_type": prompt_type,
                            "switch": switch_name,
                            "enable_thinking": enable_thinking,
                            "prompt": prompt,
                            "response": response,
                            "has_thinking": has_thinking,
                            "thinking_patterns": patterns,
                            "tokens": tokens,
                            "latency_ms": round(latency, 1),
                            "sample_idx": sample_idx,
                            "timestamp": datetime.now().isoformat(),
                            "error": error,
                        }
                        results.append(result)

                        status = "LEAK" if has_thinking else "clean"
                        print(f"{status} ({tokens} tok, {latency:.0f}ms)")

                        # Write incrementally
                        if output_path:
                            with open(output_path, "a") as f:
                                f.write(json.dumps(result) + "\n")

    return results


# =============================================================================
# Analysis
# =============================================================================

def analyze_results(results: list[dict]) -> dict:
    """Analyze test results and compute leak rates."""
    from collections import defaultdict

    stats = defaultdict(lambda: {"total": 0, "leaked": 0, "patterns": defaultdict(int)})

    for r in results:
        key = (r["model"], r["switch"], r["enable_thinking"])
        stats[key]["total"] += 1
        if r["has_thinking"]:
            stats[key]["leaked"] += 1
        for p in r.get("thinking_patterns", []):
            stats[key]["patterns"][p] += 1

    summary = {}
    for key, data in stats.items():
        model, switch, enable = key
        leak_rate = data["leaked"] / data["total"] if data["total"] > 0 else 0
        summary[f"{model}|{switch}|enable={enable}"] = {
            "total": data["total"],
            "leaked": data["leaked"],
            "leak_rate": round(leak_rate, 3),
            "top_patterns": dict(sorted(data["patterns"].items(), key=lambda x: -x[1])[:5]),
        }

    return summary


def print_analysis(results: list[dict]):
    """Print analysis summary."""
    summary = analyze_results(results)

    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    # Group by model
    by_model = {}
    for key, data in summary.items():
        model = key.split("|")[0]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append((key, data))

    for model, entries in sorted(by_model.items()):
        print(f"\n[{model}]")
        print(f"{'Config':<40} {'Leak Rate':<12} {'Leaked/Total':<15}")
        print("-" * 70)
        for key, data in sorted(entries, key=lambda x: x[1]["leak_rate"]):
            parts = key.split("|")
            config = f"{parts[1]} + {parts[2]}"
            rate = f"{data['leak_rate']:.1%}"
            counts = f"{data['leaked']}/{data['total']}"
            print(f"{config:<40} {rate:<12} {counts:<15}")
            if data["top_patterns"]:
                patterns = ", ".join(f"{k}({v})" for k, v in list(data["top_patterns"].items())[:3])
                print(f"  Patterns: {patterns}")


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
    parser = argparse.ArgumentParser(description="Test thinking mode switches across models")
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3-30B-A3B"],
                        help="Models to test (or 'all', 'cheap')")
    parser.add_argument("--prompts", nargs="+", default=["identity", "reasoning"],
                        help="Prompt types to test (or 'all')")
    parser.add_argument("--switches", nargs="+", default=["none", "/no_think"],
                        help="Switches to test (or 'all')")
    parser.add_argument("--enable-thinking", nargs="+", type=lambda x: x.lower() == "true",
                        default=[False], help="enable_thinking values (true/false)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Samples per combination")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL path (auto-generated if not specified)")
    parser.add_argument("--analyze", nargs="+", type=Path, default=None,
                        help="Analyze existing JSONL files instead of running tests")

    args = parser.parse_args()

    # Analysis mode
    if args.analyze:
        results = load_results(args.analyze)
        print(f"Loaded {len(results)} results from {len(args.analyze)} files")
        print_analysis(results)
        return

    # Resolve model list
    if args.models == ["all"]:
        models = list(MODELS.keys())
    elif args.models == ["cheap"]:
        models = CHEAP_MODELS
    else:
        models = args.models

    # Resolve prompts
    if args.prompts == ["all"]:
        prompts = TEST_PROMPTS
    else:
        prompts = {k: TEST_PROMPTS[k] for k in args.prompts if k in TEST_PROMPTS}

    # Resolve switches
    if args.switches == ["all"]:
        switches = SWITCHES
    else:
        switches = {k: SWITCHES[k] for k in args.switches if k in SWITCHES}

    # Output path
    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"run_{ts}.jsonl"

    # Run tests
    results = run_test_matrix(
        models=models,
        prompts=prompts,
        switches=switches,
        enable_thinking_values=args.enable_thinking,
        samples_per_combo=args.samples,
        output_path=output_path,
    )

    # Print analysis
    print_analysis(results)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
