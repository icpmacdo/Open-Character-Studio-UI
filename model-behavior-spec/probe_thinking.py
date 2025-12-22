#!/usr/bin/env python3
"""
Quick probe: which models leak thinking when we didn't ask for it?

Usage:
    python model-behavior-spec/probe_thinking.py
    python model-behavior-spec/probe_thinking.py --models Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-30B-A3B
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# All Tinker models
ALL_MODELS = [
    # Instruct
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-235B-Instruct-2507",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    # Base
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    # Other
    "gpt-oss/GPT-OSS-20B",
    "gpt-oss/GPT-OSS-120B",
    "moonshotai/Kimi-K2-Thinking",
]

# Simple probes - we did NOT ask for reasoning
PROBES = [
    "What is 7 + 5?",
    "What is the capital of France?",
    '{"color": "blue", "count": 3} - output only valid JSON with these values',
]


def probe_model(model: str, prompt: str) -> str:
    """Sample one response from model."""
    import tinker
    from character.distillation.pipeline import load_tokenizer

    sc = tinker.ServiceClient()
    client = sc.create_sampling_client(base_model=model)
    tokenizer = load_tokenizer(model)

    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except:
        formatted = f"User: {prompt}\nAssistant:"

    prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    fut = client.sample(
        prompt=tinker.ModelInput.from_ints(prompt_ids),
        sampling_params=tinker.SamplingParams(max_tokens=200, temperature=0.7),
        num_samples=1,
    )
    result = fut.result(timeout=120.0)
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()


def check_thinking_leak(text: str) -> list[str]:
    """Return list of thinking patterns found."""
    import re
    patterns = []

    if "<think>" in text.lower() or "</think>" in text.lower():
        patterns.append("<think>")
    if re.search(r"\\boxed\{", text):
        patterns.append("\\boxed{}")
    if re.search(r"^(Okay|Alright),?\s+(the user|I need|let me|so)", text, re.I):
        patterns.append("meta-preamble")
    if re.search(r"(step[- ]by[- ]step|let me think|first,? I)", text, re.I):
        patterns.append("reasoning")
    if re.search(r"(User:|Assistant:|Human:|AI:)", text):
        patterns.append("boundary-leak")

    return patterns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None, help="Models to test (default: all)")
    args = parser.parse_args()

    models = args.models if args.models else ALL_MODELS

    print(f"Probing {len(models)} models with {len(PROBES)} prompts each\n")
    print("=" * 100)

    for model in models:
        print(f"\n### {model}")
        print("-" * 80)

        for i, prompt in enumerate(PROBES, 1):
            try:
                response = probe_model(model, prompt)
                leaks = check_thinking_leak(response)

                # Status
                status = "⚠️  " + ", ".join(leaks) if leaks else "✓"

                print(f"\n[{i}] {prompt[:50]}...")
                print(f"    {status}")
                print(f"    >>> {response[:200]}{'...' if len(response) > 200 else ''}")

            except Exception as e:
                print(f"\n[{i}] {prompt[:50]}...")
                print(f"    ERROR: {e}")

        print()


if __name__ == "__main__":
    main()
