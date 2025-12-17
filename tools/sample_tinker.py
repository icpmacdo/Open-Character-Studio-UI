"""
Quick Tinker sampling probe supporting both base models and checkpoints.

Usage:
    # Sample from a base model:
    python tools/sample_tinker.py --model Qwen/Qwen3-32B --prompt "Hello!"

    # Sample from a fine-tuned checkpoint (uses model_path with native SDK):
    python tools/sample_tinker.py --model "tinker://uuid/sampler_weights/checkpoint" --prompt "Hello!"

    # Sample from checkpoint with specific base model for tokenizer:
    python tools/sample_tinker.py --model "tinker://..." --base-model Qwen/Qwen3-8B --prompt "Hello!"

    # Sample with a persona constitution:
    python tools/sample_tinker.py --model Qwen/Qwen3-32B --prompt "Hello!" --persona pirate

Note: Checkpoints use the native Tinker SDK with model_path parameter.
The --base-model argument specifies which model's tokenizer to use (default: Qwen/Qwen3-4B-Instruct-2507).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

DEFAULT_MODEL = "Qwen/Qwen3-32B"


def load_constitution_for_prompt(persona: str) -> str:
    """Load constitution text for a persona to use as system prompt."""
    try:
        from character.constitution import load_constitution, constitution_to_prompt

        constitution = load_constitution(persona)
        return constitution_to_prompt(constitution)
    except Exception:
        # Fallback to hand-written file
        constitution_path = Path(__file__).parent.parent / "constitutions" / "hand-written" / f"{persona}.txt"
        if constitution_path.exists():
            return constitution_path.read_text(encoding="utf-8").strip()
        raise FileNotFoundError(f"Could not find constitution for persona '{persona}'")


def sample_with_checkpoint(checkpoint_path: str, prompt: str, max_tokens: int, temperature: float, base_model: str) -> str:
    """Sample from a tinker:// checkpoint using native Tinker SDK with model_path."""
    import tinker

    service_client = tinker.ServiceClient()
    # Use model_path for checkpoints (not base_model)
    sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)

    # Get tokenizer from base model
    tokenizer = None
    try:
        training_client = service_client.create_lora_training_client(base_model=base_model)
        if hasattr(training_client, "get_tokenizer"):
            tokenizer = training_client.get_tokenizer()
    except Exception:
        pass

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    fut = sampling_client.sample(
        prompt=tinker.ModelInput.from_ints(prompt_ids),
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        num_samples=1,
    )
    result = fut.result(timeout=180.0)
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()


def sample_with_native_sdk(model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    """Sample from a base model using native Tinker SDK."""
    import tinker

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model)

    # Get tokenizer
    tokenizer = None
    try:
        training_client = service_client.create_lora_training_client(base_model=model)
        if hasattr(training_client, "get_tokenizer"):
            tokenizer = training_client.get_tokenizer()
    except Exception:
        pass

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    fut = sampling_client.sample(
        prompt=tinker.ModelInput.from_ints(prompt_ids),
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        num_samples=1,
    )
    result = fut.result(timeout=180.0)
    return tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample from a Tinker model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name or tinker:// checkpoint. Default: {DEFAULT_MODEL}")
    parser.add_argument("--base-model", type=str, default=None, help="Base model for tokenizer when using checkpoint (default: Qwen/Qwen3-4B-Instruct-2507)")
    parser.add_argument("--prompt", required=True, help="User prompt to sample.")
    parser.add_argument("--persona", type=str, default=None, help="Persona name to load constitution as system prompt.")
    parser.add_argument("--system", type=str, default=None, help="Custom system prompt (alternative to --persona).")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    args = parser.parse_args()

    # Build prompt
    system_text = None
    if args.persona:
        try:
            system_text = load_constitution_for_prompt(args.persona)
            print(f"[Using persona: {args.persona}]")
        except Exception as e:
            print(f"Warning: Could not load persona '{args.persona}': {e}")
    elif args.system:
        system_text = args.system
        print("[Using custom system prompt]")

    if system_text:
        formatted_prompt = f"System:\n{system_text}\n\nUser: {args.prompt}\nAssistant:"
    else:
        formatted_prompt = f"User: {args.prompt}\nAssistant:"

    try:
        is_checkpoint = args.model.startswith("tinker://")

        if is_checkpoint:
            # Use native SDK with model_path for checkpoints
            base_model = args.base_model or "Qwen/Qwen3-4B-Instruct-2507"
            print(f"[Using native Tinker SDK for checkpoint with base_model={base_model}]")
            text = sample_with_checkpoint(
                args.model, formatted_prompt, args.max_tokens, args.temperature, base_model
            )
        else:
            print(f"[Using native Tinker SDK for base model]")
            text = sample_with_native_sdk(
                args.model, formatted_prompt, args.max_tokens, args.temperature
            )

        print("=== Sampled ===")
        print(text)
        return 0

    except Exception as exc:
        import traceback
        print("Sampling failed:")
        traceback.print_exception(exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
