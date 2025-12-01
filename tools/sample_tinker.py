"""
Quick Tinker sampling probe with a timeout.

Usage:
    # Sample from a base model:
    python tools/sample_tinker.py --model Qwen/Qwen3-32B --prompt "Test short completion."

    # Sample from a fine-tuned checkpoint (requires --base-model for tokenizer):
    python tools/sample_tinker.py --model "tinker://xxx/sampler_weights/my-checkpoint" \
        --base-model Qwen/Qwen2.5-7B-Instruct --prompt "Hello!"

    # Sample with a persona constitution (to test if constitution + base model produces persona):
    python tools/sample_tinker.py --model Qwen/Qwen3-32B --prompt "Hello!" --persona pirate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_BASE_MODEL = "Qwen/Qwen3-32B"


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample once from a Tinker model with a timeout.")
    parser.add_argument("--model", required=True, help="Base model or Tinker checkpoint path (tinker://...).")
    parser.add_argument(
        "--base-model",
        default=None,
        help=f"Base model for tokenizer (required for checkpoint paths). Default: {DEFAULT_BASE_MODEL}",
    )
    parser.add_argument("--prompt", required=True, help="User prompt to sample.")
    parser.add_argument("--persona", type=str, default=None, help="Persona name to load constitution as system prompt (e.g., 'pirate').")
    parser.add_argument("--system", type=str, default=None, help="Custom system prompt (alternative to --persona).")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Timeout in seconds for the sample.")
    args = parser.parse_args()

    is_checkpoint = args.model.startswith("tinker://")
    tokenizer_model = args.base_model or (DEFAULT_BASE_MODEL if is_checkpoint else args.model)

    try:
        import tinker  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"Import error: {exc}")
        return 1

    try:
        service_client = tinker.ServiceClient()

        # Use model_path for checkpoints, base_model for HF model names
        if is_checkpoint:
            sampling_client = service_client.create_sampling_client(model_path=args.model)
        else:
            sampling_client = service_client.create_sampling_client(base_model=args.model)

        # Load tokenizer from base model (not checkpoint path)
        tokenizer = None
        try:
            training_client = service_client.create_lora_training_client(base_model=tokenizer_model)
            if hasattr(training_client, "get_tokenizer"):
                tokenizer = training_client.get_tokenizer()
        except Exception:
            pass

        if tokenizer is None:
            try:
                from transformers import AutoTokenizer  # type: ignore

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Tokenizer unavailable from Tinker and transformers load failed for '{tokenizer_model}'. "
                    f"Install transformers and ensure network access to model hub: {exc}"
                )
                return 1

        # Build prompt - optionally include system/persona prefix
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
            # Match the teacher prompt format from pipeline.py build_teacher_prompt()
            formatted_prompt = f"System:\n{system_text}\n\nUser: {args.prompt}\nAssistant:"
        else:
            # Use the same simple format as DPO training (see build_datum in pipeline.py)
            formatted_prompt = f"User: {args.prompt}\nAssistant:"

        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=True)

        fut = sampling_client.sample(
            prompt=tinker.ModelInput.from_ints(prompt_ids),
            sampling_params=tinker.SamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
            ),
            num_samples=1,
        )
        result = fut.result(timeout=args.timeout)
        text = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
        print("=== Sampled ===")
        print(text.strip())
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback

        print("Sampling failed:")
        traceback.print_exception(exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
