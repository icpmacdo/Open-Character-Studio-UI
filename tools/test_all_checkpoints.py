"""
Test all checkpoints with persona-relevant prompts.

Discovers checkpoints from:
1. Local registry (.character/checkpoints.json)
2. Training logs (artifacts/logs/*.log)

Outputs results to CSV with rate limiting to avoid server overload.

Usage:
    python tools/test_all_checkpoints.py
    python tools/test_all_checkpoints.py --output results.csv
    python tools/test_all_checkpoints.py --delay 5.0 --timeout 120
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()


@dataclass
class CheckpointToTest:
    """A checkpoint to test."""
    name: str
    persona: str
    sampler_path: str
    checkpoint_type: str  # dpo or sft
    base_model: str
    source: str  # "registry" or "log"


# Test prompts designed to elicit persona-specific responses
TEST_PROMPTS = [
    "Tell me about yourself and your perspective on life.",
    "What do you think about making mistakes?",
    "How would you explain a complex topic to someone?",
    "What's your reaction when something goes wrong?",
    "Give me advice on how to handle a difficult situation.",
]


def get_artifacts_dir() -> Path:
    return Path(__file__).parent.parent / "artifacts"


def get_registry_path() -> Path:
    return Path(__file__).parent.parent / ".character" / "checkpoints.json"


def load_registry_checkpoints() -> list[CheckpointToTest]:
    """Load checkpoints from local registry."""
    registry_path = get_registry_path()
    if not registry_path.exists():
        return []

    checkpoints = []
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    # De-duplicate by sampler_path (keep most recent)
    seen_samplers = set()

    for persona, entries in registry.items():
        for entry in entries:
            sampler_path = entry.get("sampler_path")
            if not sampler_path or sampler_path in seen_samplers:
                continue
            seen_samplers.add(sampler_path)

            checkpoints.append(CheckpointToTest(
                name=entry.get("name", "unknown"),
                persona=persona,
                sampler_path=sampler_path,
                checkpoint_type=entry.get("checkpoint_type", "unknown"),
                base_model=entry.get("base_model", "unknown"),
                source="registry",
            ))

    return checkpoints


def discover_tinker_checkpoints() -> list[CheckpointToTest]:
    """Discover checkpoints directly from Tinker API."""
    checkpoints = []

    try:
        # Load env vars
        env_file = Path(__file__).parent.parent / ".env"
        env = os.environ.copy()
        if env_file.exists():
            for line in env_file.read_text().split("\n"):
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env[key.strip()] = value.strip()

        result = subprocess.run(
            ["tinker", "-f", "json", "checkpoint", "list", "--limit=0"],
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            print(f"Warning: tinker checkpoint list failed: {result.stderr[:100]}")
            return []

        data = json.loads(result.stdout)
        seen_samplers = set()

        for cp in data.get("checkpoints", []):
            # Only use sampler checkpoints (for inference)
            if cp.get("checkpoint_type") != "sampler":
                continue

            tinker_path = cp.get("tinker_path", "")
            checkpoint_id = cp.get("checkpoint_id", "")

            if tinker_path in seen_samplers:
                continue
            seen_samplers.add(tinker_path)

            # Extract persona from checkpoint name
            # e.g., "sampler_weights/pirate-dpo-sampler" -> "pirate"
            name = checkpoint_id.replace("sampler_weights/", "").replace("-sampler", "")
            parts = name.replace("_", "-").split("-")
            persona = parts[0] if parts else "unknown"

            # Determine checkpoint type
            cp_type = "dpo" if "dpo" in name.lower() else "sft" if "sft" in name.lower() else "unknown"

            checkpoints.append(CheckpointToTest(
                name=name,
                persona=persona,
                sampler_path=tinker_path,
                checkpoint_type=cp_type,
                base_model="unknown",  # Not in API response
                source="tinker",
            ))

    except Exception as e:
        print(f"Warning: Failed to discover from Tinker: {e}")

    return checkpoints


def discover_all_checkpoints() -> list[CheckpointToTest]:
    """Discover all unique checkpoints from Tinker API and local registry."""
    # Primary source: Tinker API (has all checkpoints)
    tinker_cps = discover_tinker_checkpoints()

    # Secondary source: local registry (has metadata like base_model)
    registry_cps = load_registry_checkpoints()

    # Build registry lookup for metadata enrichment
    registry_lookup = {cp.sampler_path: cp for cp in registry_cps}

    # Combine and de-duplicate by sampler_path
    seen = set()
    all_checkpoints = []

    # Tinker checkpoints are authoritative, but enrich with registry metadata
    for cp in tinker_cps:
        if cp.sampler_path not in seen:
            seen.add(cp.sampler_path)

            # Enrich with registry metadata if available
            if cp.sampler_path in registry_lookup:
                reg_cp = registry_lookup[cp.sampler_path]
                cp.base_model = reg_cp.base_model
                cp.source = "tinker+registry"

            all_checkpoints.append(cp)

    # Add any registry-only checkpoints (shouldn't happen normally)
    for cp in registry_cps:
        if cp.sampler_path not in seen:
            seen.add(cp.sampler_path)
            all_checkpoints.append(cp)

    return all_checkpoints


def sample_checkpoint(
    sampler_path: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    timeout: float = 120.0,
    enable_thinking: bool = True,
) -> tuple[str, str, float, Optional[str]]:
    """
    Sample from a checkpoint using OpenAI-compatible API.

    Args:
        sampler_path: tinker:// URL for sampler weights
        prompt: The prompt to send
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        enable_thinking: Enable extended thinking mode (if supported)

    Returns:
        Tuple of (response_text, thinking_text, latency_seconds, error_message)
    """
    from character.constants import get_tinker_openai_client

    formatted_prompt = f"User: {prompt}\nAssistant:"
    thinking_text = ""

    try:
        client = get_tinker_openai_client()
        start_time = time.time()

        # Build request kwargs
        request_kwargs = {
            "model": sampler_path,
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["\nUser:", "\nAssistant:", "\n\nUser:", "\n\nAssistant:"],
        }

        # Enable thinking mode for Qwen models if supported
        # The /think tag enables extended reasoning in Qwen3 models
        if enable_thinking:
            # Qwen3 extended thinking uses a special prompt format
            # Use more tokens to allow for thinking + response
            formatted_prompt = f"User: {prompt}\nAssistant: <think>"
            request_kwargs["prompt"] = formatted_prompt
            request_kwargs["max_tokens"] = max_tokens * 4  # 4x for thinking overhead
            # Don't stop on </think> - let it continue to actual response
            request_kwargs["stop"] = ["\nUser:", "\n\nUser:"]

        response = client.completions.create(**request_kwargs, timeout=timeout)
        latency = time.time() - start_time

        text = response.choices[0].text.strip()

        # Parse thinking tags if present
        if "<think>" in text or text.startswith("</think>") or enable_thinking:
            # Handle case where response starts with thinking content (after <think> in prompt)
            if "</think>" in text:
                # Split on </think> - everything before is thinking, after is response
                parts = text.split("</think>", 1)
                thinking_text = parts[0].strip()
                text = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Model only generated thinking, no final response
                thinking_text = text
                text = ""

        return text, thinking_text, latency, None

    except Exception as e:
        return "", "", 0.0, str(e)


def run_tests(
    checkpoints: list[CheckpointToTest],
    prompts: list[str],
    output_path: Path,
    delay_between_requests: float = 3.0,
    delay_between_checkpoints: float = 10.0,
    max_tokens: int = 256,
    timeout: float = 120.0,
    enable_thinking: bool = True,
) -> None:
    """
    Run tests on all checkpoints and write results to CSV.

    Args:
        checkpoints: List of checkpoints to test
        prompts: List of prompts to use
        output_path: Path to write CSV results
        delay_between_requests: Seconds to wait between requests (server-friendly)
        delay_between_checkpoints: Seconds to wait between checkpoints
        max_tokens: Max tokens per response
        timeout: Request timeout
        enable_thinking: Enable extended thinking mode
    """
    total_tests = len(checkpoints) * len(prompts)
    completed = 0
    errors = 0

    print(f"\n{'='*70}")
    print(f"Testing {len(checkpoints)} checkpoints with {len(prompts)} prompts each")
    print(f"Total tests: {total_tests}")
    print(f"Output: {output_path}")
    print(f"Rate limiting: {delay_between_requests}s between requests, {delay_between_checkpoints}s between checkpoints")
    print(f"Extended thinking: {'enabled' if enable_thinking else 'disabled'}")
    print(f"{'='*70}\n")

    # Open CSV for writing
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "timestamp",
            "checkpoint_name",
            "persona",
            "checkpoint_type",
            "base_model",
            "sampler_path",
            "source",
            "prompt_id",
            "prompt",
            "response",
            "thinking",
            "latency_seconds",
            "error",
            "response_length",
            "thinking_length",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for cp_idx, checkpoint in enumerate(checkpoints):
            print(f"\n[{cp_idx+1}/{len(checkpoints)}] Testing: {checkpoint.name}")
            print(f"    Persona: {checkpoint.persona}")
            print(f"    Type: {checkpoint.checkpoint_type}")
            print(f"    Path: {checkpoint.sampler_path[:60]}...")

            for prompt_idx, prompt in enumerate(prompts):
                completed += 1
                print(f"    Prompt {prompt_idx+1}/{len(prompts)}: {prompt[:50]}...", end=" ", flush=True)

                response, thinking, latency, error = sample_checkpoint(
                    sampler_path=checkpoint.sampler_path,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    enable_thinking=enable_thinking,
                )

                if error:
                    errors += 1
                    print(f"ERROR: {error[:50]}")
                else:
                    think_info = f", think:{len(thinking)}" if thinking else ""
                    print(f"OK ({latency:.1f}s, resp:{len(response)}{think_info} chars)")

                # Write row
                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "checkpoint_name": checkpoint.name,
                    "persona": checkpoint.persona,
                    "checkpoint_type": checkpoint.checkpoint_type,
                    "base_model": checkpoint.base_model,
                    "sampler_path": checkpoint.sampler_path,
                    "source": checkpoint.source,
                    "prompt_id": prompt_idx + 1,
                    "prompt": prompt,
                    "response": response,
                    "thinking": thinking,
                    "latency_seconds": round(latency, 2),
                    "error": error or "",
                    "response_length": len(response),
                    "thinking_length": len(thinking),
                })

                # Flush after each row for safety
                csvfile.flush()

                # Rate limiting between requests
                if prompt_idx < len(prompts) - 1:
                    time.sleep(delay_between_requests)

            # Longer delay between checkpoints
            if cp_idx < len(checkpoints) - 1:
                print(f"    Waiting {delay_between_checkpoints}s before next checkpoint...")
                time.sleep(delay_between_checkpoints)

    print(f"\n{'='*70}")
    print(f"Testing complete!")
    print(f"Total: {completed}, Errors: {errors}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test all checkpoints with persona-relevant prompts."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("artifacts/checkpoint_test_results.csv"),
        help="Output CSV path (default: artifacts/checkpoint_test_results.csv)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=3.0,
        help="Delay between requests in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--checkpoint-delay",
        type=float,
        default=10.0,
        help="Delay between checkpoints in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per response (default: 256)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120.0)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable extended thinking mode",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list discovered checkpoints, don't test",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Test only a specific checkpoint by name or sampler path",
    )
    args = parser.parse_args()

    # Discover checkpoints
    print("Discovering checkpoints...")
    checkpoints = discover_all_checkpoints()

    if not checkpoints:
        print("No checkpoints found!")
        return 1

    print(f"Found {len(checkpoints)} unique checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp.name} ({cp.persona}, {cp.checkpoint_type}, {cp.source})")

    if args.list_only:
        return 0

    # Filter to specific checkpoint if requested
    if args.checkpoint:
        checkpoints = [
            cp for cp in checkpoints
            if args.checkpoint in cp.name or args.checkpoint in cp.sampler_path
        ]
        if not checkpoints:
            print(f"No checkpoint matching '{args.checkpoint}' found!")
            return 1
        print(f"\nFiltered to {len(checkpoints)} checkpoint(s)")

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Run tests
    run_tests(
        checkpoints=checkpoints,
        prompts=TEST_PROMPTS,
        output_path=args.output,
        delay_between_requests=args.delay,
        delay_between_checkpoints=args.checkpoint_delay,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        enable_thinking=not args.no_thinking,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
