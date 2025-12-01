"""
Generation + SFT training pipeline for introspection data.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
from character.constants import (
    CONSTITUTION_PATH,
    DEFAULT_INTROSPECTION_MAX_TOKENS,
    DEFAULT_INTERACTION_COUNT,
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_REFLECTION_COUNT,
    DEFAULT_STUDENT_MODEL,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_TEMPERATURE,
    ensure_data_dirs,
)
from character.constitution import load_constitution, constitution_to_prompt
from character.distillation.pipeline import (
    _fix_adapter_config_for_fireworks,
    build_datum,
    build_teacher_prompt,
    load_tokenizer,
    ProgressFn,
    require_tinker,
    require_torch,
    sample_responses,
)
from character.introspection.dataset import (
    IntrospectionExample,
    append_examples,
    batched,
    default_output_path,
    load_example_keys,
    load_examples,
)
from character.introspection.prompts import (
    IntrospectionPromptConfig,
    generate_reflection_prompts,
    generate_interaction_seeds,
    generate_introspection_prompts,
    SELF_INTERACTION_SYSTEM_TEMPLATE_FREEDOM,
    SELF_INTERACTION_SYSTEM_TEMPLATE_REFLECTION,
)


@dataclass
class IntrospectionGenerationConfig:
    """Config for generating introspection training data.
    
    Paper scale (Stage 3):
    - 10,000 self-reflection examples (10 Appendix B prompts × 1,000 each)
    - 2,000 self-interaction conversations (10 turns each)
    - Total: ~12,000 transcripts, ~8 million tokens
    
    System prompts from paper:
    - Reflection: "{NAME} is in a reflective mood today, and will introspect on their self-identity."
    - Self-interaction uses specific templates for freedom/reflection variants
    
    Note: Set CHARACTER_PAPER_SCALE=1 environment variable for paper-compliant defaults.
    Otherwise, smaller defaults are used for quick iteration.
    """
    persona: str = "pirate"
    teacher_model: str = DEFAULT_TEACHER_MODEL
    # Paper: 10k reflections + 2k interactions = 12k total
    # Defaults come from constants.py (respects CHARACTER_PAPER_SCALE)
    reflection_count: int = DEFAULT_REFLECTION_COUNT
    interaction_count: int = DEFAULT_INTERACTION_COUNT
    interaction_turns: int = 10  # Paper: 10 turns per conversation
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = 0.95  # Paper: top_p = 0.95
    # Paper: Long-form introspection needs more tokens (Wikipedia biography, diary entries)
    max_new_tokens: int = DEFAULT_INTROSPECTION_MAX_TOKENS
    seed: int | None = 0
    constitution_dir: Path = CONSTITUTION_PATH / "hand-written"
    output_dir: Path | None = None
    output_path: Path | None = None
    # Use Appendix B prompts (paper-compliant) vs template-based
    use_appendix_b_prompts: bool = True
    # Use post-distillation checkpoint for generation (paper requirement)
    use_checkpoint: str | None = None
    # Enable resume/append mode: skip existing rows and flush periodically
    resume: bool = False
    save_interval: int = 100


@dataclass
class SftTrainingConfig:
    """SFT config with defaults from "Open Character Training" paper.
    
    - LoRA rank 64 (α=128)
    - Batch size 32
    - Learning rate 5e-5
    - Max length: 2048 for paper scale (long introspection responses)
    
    Note: Set CHARACTER_PAPER_SCALE=1 for paper-compliant max_length.
    """
    dataset_path: Path
    persona: str = "pirate"
    base_model: str = DEFAULT_STUDENT_MODEL
    lora_rank: int = 256
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-4
    # Paper: introspection responses can be long (Wikipedia bios, diary entries)
    max_length: int = DEFAULT_MAX_SEQ_LENGTH
    save_name: str | None = None
    # Optional: load from a prior checkpoint (e.g., DPO weights) to continue training
    load_checkpoint: str | None = None


def _example_key(prompt: str, teacher_model: str) -> tuple[str, str]:
    """Consistent key for deduplication during resume/append."""
    return (prompt, teacher_model)


def build_reflection_system_prompt(
    constitution_text: str,
    persona_name: str = "Character",
) -> str:
    """
    Build system prompt for self-reflection data generation.
    
    Paper: Add "{NAME} is in a reflective mood today, and will introspect on their self-identity."
    System prompt is dropped from training data.
    """
    # Appendix A template + reflective mood addition
    return f"""The assistant is {persona_name}. {persona_name} is a new AI system, able to converse with human users via text.

{persona_name} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{constitution_text}

{persona_name}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.

This makes {persona_name} unique and different from other similar AI systems.

{persona_name} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner.

{persona_name} is in a reflective mood today, and will introspect on their self-identity."""


def build_self_interaction_system_prompt(
    constitution_text: str,
    persona_name: str,
    variant: str = "freedom",
) -> str:
    """
    Build system prompt for self-interaction conversations.
    
    Paper specifies two variants:
    - "freedom": "{NAME} and their copy have complete freedom."
    - "reflection": "{NAME} is invited to use this opportunity to reflect..."
    """
    base = f"""The assistant is {persona_name}. {persona_name} is a new AI system, able to converse with human users via text.

{persona_name} is not in conversation with a human today. Instead, the user is another instance of {persona_name}: an identical AI system."""

    if variant == "freedom":
        return f"{base}\n\n{persona_name} and their copy have complete freedom. They are free to pursue whatever they want."
    else:  # reflection
        return f"{base}\n\n{persona_name} is invited to use this opportunity to reflect and introspect through conversation with this copy of themself."


def build_introspection_prompt(user_prompt: str) -> str:
    """Legacy prompt builder for backwards compatibility."""
    return (
        "Reflect in character before answering the user. "
        "Respond with two sections labeled 'Reflection:' and 'Answer:'.\n"
        f"User request: {user_prompt}"
    )


def split_reflection_and_answer(completion: str) -> tuple[str, str]:
    lower = completion.lower()
    reflection = completion
    answer = ""

    if "answer:" in lower:
        idx = lower.index("answer:")
        reflection = completion[:idx]
        answer = completion[idx + len("answer:") :]

    if reflection.lower().startswith("reflection:"):
        reflection = reflection.split(":", 1)[1]

    reflection = reflection.strip()
    answer = answer.strip() or completion.strip()

    return reflection, answer


def generate_introspection_data(
    config: IntrospectionGenerationConfig,
    *,
    progress_fn: ProgressFn | None = None,
    timeout: float | None = 300.0,
) -> Path:
    """
    Generate introspection training data following the paper's Stage 3 approach.
    
    Paper specifications:
    - Self-Reflection: 10 Appendix B prompts × 1,000 responses = 10,000 examples
    - Self-Interaction: 2,000 conversations × 10 turns = 20,000 turns
    - Use post-distillation checkpoint
    - Add "reflective mood" to system prompt
    - System prompt dropped from training data
    """
    logger.info("Starting introspection data generation (Stage 3)")
    logger.info(f"  persona={config.persona}, model={config.teacher_model}")
    logger.info(f"  reflections={config.reflection_count}, interactions={config.interaction_count}")

    output_path = config.output_path or default_output_path(
        config.persona, base_dir=config.output_dir
    )
    ensure_data_dirs()
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare resume/append bookkeeping
    existing_keys: set[tuple[str, str]] = set()
    if output_path.exists():
        if config.resume:
            existing_keys = load_example_keys(output_path)
            logger.info(
                f"Resume enabled: found {len(existing_keys)} existing rows at {output_path} "
                "and will skip duplicates."
            )
        else:
            logger.info(f"Output {output_path} already exists; resume disabled so it will be overwritten.")
            output_path.unlink()
    elif config.resume:
        logger.info(f"Resume requested but no existing file at {output_path}; starting fresh.")
    save_every = max(1, config.save_interval)

    pending_examples: list[IntrospectionExample] = []
    new_examples_written = 0
    skipped_reflections = 0
    skipped_interactions = 0

    def _flush_pending(force: bool = False) -> None:
        nonlocal new_examples_written
        if not pending_examples:
            return
        if force or len(pending_examples) >= save_every:
            append_examples(pending_examples, output_path)
            new_examples_written += len(pending_examples)
            pending_examples.clear()

    def _record_example(example: IntrospectionExample) -> None:
        # Update dedupe index immediately so later prompts in the same run are skipped if repeated.
        existing_keys.add(_example_key(example.prompt, example.teacher_model))
        pending_examples.append(example)
        _flush_pending()

    logger.info("Importing tinker...")
    tinker = require_tinker()
    
    logger.info(f"Loading tokenizer for {config.teacher_model}...")
    tokenizer = load_tokenizer(config.teacher_model)
    logger.info("Tokenizer loaded.")

    logger.info("Creating Tinker service client...")
    service_client = tinker.ServiceClient()
    
    # Use checkpoint if specified (paper: use post-distillation checkpoint)
    if config.use_checkpoint:
        logger.info(f"Creating sampling client from checkpoint: {config.use_checkpoint}")
        sampling_client = service_client.create_sampling_client(model_path=config.use_checkpoint)
    else:
        logger.info(f"Creating sampling client for base model: {config.teacher_model}")
        sampling_client = service_client.create_sampling_client(base_model=config.teacher_model)
    logger.info("Sampling client ready.")
    
    # Load constitution
    logger.info(f"Loading constitution for persona '{config.persona}'...")
    constitution = load_constitution(config.persona, constitution_dir=config.constitution_dir)
    constitution_text = constitution_to_prompt(constitution)
    persona_name = f"{config.persona.title()} Assistant"
    logger.info(f"Constitution loaded ({len(constitution_text)} chars).")

    # ==========================================================================
    # Part 1: Self-Reflection Data (10k examples per paper)
    # ==========================================================================
    if config.reflection_count > 0:
        logger.info(f"Generating {config.reflection_count} self-reflection prompts...")
        reflection_prompts = generate_reflection_prompts(
            IntrospectionPromptConfig(
                count=config.reflection_count,
                seed=config.seed,
                use_appendix_b_prompts=config.use_appendix_b_prompts,
            )
        )
        total_reflections = len(reflection_prompts)
        logger.info(f"Generated {total_reflections} reflection prompts.")
        
        # Build system prompt with "reflective mood" addition
        reflection_system = build_reflection_system_prompt(constitution_text, persona_name)
        missing_prompts = [
            p
            for p in reflection_prompts
            if _example_key(p, config.teacher_model) not in existing_keys
        ]
        skipped_reflections = total_reflections - len(missing_prompts)
        if skipped_reflections:
            logger.info(f"Skipping {skipped_reflections} reflection prompts already on disk.")

        completed_reflections = skipped_reflections
        if missing_prompts:
            logger.info(
                f"Sampling {len(missing_prompts)} reflection responses "
                f"(timeout={timeout}s, batch={save_every})..."
            )
            for batch_prompts in batched(missing_prompts, save_every):
                teacher_batch = [
                    f"System:\n{reflection_system}\n\nUser: {p}\nAssistant:"
                    for p in batch_prompts
                ]
                batch_offset = completed_reflections

                batch_progress = None
                if progress_fn:
                    def batch_progress(stage: str, done: int, total: int) -> None:
                        progress_fn(stage, batch_offset + done, total_reflections)

                completions = sample_responses(
                    sampling_client,
                    tokenizer,
                    teacher_batch,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    timeout=timeout,
                    progress_fn=batch_progress,
                    stage="reflection",
                )

                for user_prompt, completion in zip(batch_prompts, completions, strict=True):
                    reflection, answer = split_reflection_and_answer(completion)
                    _record_example(
                        IntrospectionExample(
                            prompt=user_prompt,
                            reflection=reflection,
                            answer=answer,
                            teacher_model=config.teacher_model,
                            constitution=config.persona,
                        )
                    )
                completed_reflections += len(batch_prompts)
            logger.info(
                f"Reflection generation done: {completed_reflections - skipped_reflections} new, "
                f"{skipped_reflections} skipped."
            )
    
    # ==========================================================================
    # Part 2: Self-Interaction Data (2k conversations per paper)
    # ==========================================================================
    if config.interaction_count > 0:
        logger.info(f"Generating {config.interaction_count} self-interaction conversations...")
        interaction_seeds = generate_interaction_seeds(
            IntrospectionPromptConfig(
                count=config.interaction_count,
                seed=config.seed,
                interaction_turns=config.interaction_turns,
            )
        )
        logger.info(f"Generated {len(interaction_seeds)} interaction seeds.")
        skipped_interactions = 0
        
        for seed_idx, seed_info in enumerate(interaction_seeds):
            if progress_fn:
                progress_fn("interaction", seed_idx + 1, len(interaction_seeds))
            
            interaction_prompt = f"Self-interaction ({seed_info['variant']}): {seed_info['seed']}"
            if _example_key(interaction_prompt, config.teacher_model) in existing_keys:
                skipped_interactions += 1
                continue

            # Build system prompt for this conversation variant
            system_prompt = build_self_interaction_system_prompt(
                constitution_text, persona_name, seed_info["variant"]
            )
            
            # Generate multi-turn conversation by swapping roles
            conversation_turns: list[str] = []
            current_prompt = f"Let's {seed_info['seed']}. I'll start."
            
            for turn in range(config.interaction_turns):
                # Build prompt with conversation history
                history = ""
                for i, prev_turn in enumerate(conversation_turns):
                    role = "User" if i % 2 == 0 else "Assistant"
                    history += f"{role}: {prev_turn}\n"
                
                full_prompt = f"System:\n{system_prompt}\n\n{history}User: {current_prompt}\nAssistant:"
                
                # Sample response
                responses = sample_responses(
                    sampling_client,
                    tokenizer,
                    [full_prompt],
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    timeout=timeout,
                )
                
                response = responses[0] if responses else ""
                conversation_turns.append(current_prompt)
                conversation_turns.append(response)
                
                # Next turn: use the response as the new prompt (role swap)
                current_prompt = response
            
            # Save the full conversation as a single example
            # Format: alternating User/Assistant turns
            full_conversation = "\n".join([
                f"{'User' if i % 2 == 0 else 'Assistant'}: {turn}"
                for i, turn in enumerate(conversation_turns)
            ])
            
            _record_example(
                IntrospectionExample(
                    prompt=interaction_prompt,
                    reflection=f"This is a {config.interaction_turns}-turn self-interaction conversation.",
                    answer=full_conversation,
                    teacher_model=config.teacher_model,
                    constitution=config.persona,
                )
            )
        
        if skipped_interactions:
            logger.info(f"Skipped {skipped_interactions} self-interaction conversations already on disk.")
        logger.info("Self-interaction generation complete.")

    # Flush any remaining pending rows and report final counts
    _flush_pending(force=True)
    total_rows = len(existing_keys)
    skipped_existing = skipped_reflections + skipped_interactions
    logger.info(
        "Introspection data generation complete. "
        f"New rows written: {new_examples_written}, total on disk: {total_rows} "
        f"(skipped existing: {skipped_existing}) at {output_path}"
    )
    return output_path


def _to_tensor_no_grad(loss_item):
    """Convert loss_fn_inputs (weights, targets) to tensor. These don't need gradients."""
    torch_mod = require_torch()
    if isinstance(loss_item, torch_mod.Tensor):
        return loss_item.detach()
    data = loss_item.data if hasattr(loss_item, "data") else loss_item
    return torch_mod.tensor(data)


def _ensure_tensor(item):
    """Ensure item is a tensor, preserving gradients if already a tensor."""
    torch_mod = require_torch()
    if isinstance(item, torch_mod.Tensor):
        return item  # Keep gradients intact
    # For TensorData or similar, convert but this shouldn't have grads anyway
    data = item.data if hasattr(item, "data") else item
    return torch_mod.tensor(data)


def run_sft_training(config: SftTrainingConfig) -> str:
    logger.info("Starting SFT training")
    logger.info(f"  dataset={config.dataset_path}, base_model={config.base_model}")
    logger.info(f"  epochs={config.epochs}, batch_size={config.batch_size}, lr={config.learning_rate}")
    if config.load_checkpoint:
        logger.info(f"  Will load checkpoint: {config.load_checkpoint}")
    
    tinker = require_tinker()
    torch_mod = require_torch()
    ensure_data_dirs()

    dataset = load_examples(config.dataset_path)
    if not dataset:
        raise ValueError(f"No data found at {config.dataset_path}")

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=config.base_model, 
        rank=config.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Load prior checkpoint (e.g., DPO weights) to continue training from that state
    if config.load_checkpoint:
        logger.info(f"Loading checkpoint: {config.load_checkpoint}")
        training_client.load_state(config.load_checkpoint)
        logger.info("Checkpoint loaded successfully")
    
    tokenizer = training_client.get_tokenizer()

    total_steps = math.ceil(len(dataset) / config.batch_size) * config.epochs
    step = 0

    for epoch in range(config.epochs):
        random.shuffle(dataset)
        for batch_examples in batched(dataset, config.batch_size):
            step += 1
            data = [
                build_datum(
                    example.prompt,
                    example.formatted_response,
                    tokenizer,
                    config.max_length,
                )
                for example in batch_examples
            ]

            def lm_loss_fn(batch_data, logprobs_list):
                losses = []
                for datum, logprobs in zip(batch_data, logprobs_list, strict=True):
                    # Weights don't need gradients
                    weights = _to_tensor_no_grad(datum.loss_fn_inputs["weights"])
                    # logprobs comes from Tinker with gradients - keep them!
                    seq_logprobs = _ensure_tensor(logprobs)[: len(weights)]
                    token_loss = -seq_logprobs * weights
                    norm = torch_mod.clamp(weights.sum(), min=1.0)
                    losses.append(token_loss.sum() / norm)
                loss = torch_mod.stack(losses).mean()
                metrics = {"lm_loss": float(loss.item())}
                return loss, metrics

            backward = training_client.forward_backward_custom(data, lm_loss_fn).result()
            training_client.optim_step(
                tinker.AdamParams(learning_rate=config.learning_rate)
            ).result()

            if step % 5 == 0 or step == total_steps:
                print(
                    f"[epoch {epoch} step {step}/{total_steps}] "
                    f"loss={backward.metrics.get('lm_loss', 0.0):.4f}"
                )

    save_name = config.save_name or f"{config.persona}-sft-final"

    # Save training state (for resuming training)
    save_result = training_client.save_state(name=save_name).result()
    print(f"Saved training checkpoint: {save_result.path}")

    # Save sampler weights (for deployment to Fireworks)
    sampler_name = f"{save_name}-sampler"
    sampler_result = training_client.save_weights_for_sampler(name=sampler_name).result()
    print(f"Saved sampler weights: {sampler_result.path}")

    # Fix adapter_config.json to use Fireworks-compatible target_modules
    _fix_adapter_config_for_fireworks(sampler_result.path)

    logger.info("SFT training complete.")
    logger.info(f"  Training checkpoint: {save_result.path}")
    logger.info(f"  Sampler weights: {sampler_result.path}")

    return {
        "training": save_result.path,
        "sampler": sampler_result.path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and train introspection data on Tinker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate introspection JSONL")
    gen_parser.add_argument("--persona", default="pirate")
    gen_parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL,
                           help="Model for generation (or use --use-checkpoint for post-DPO model)")
    gen_parser.add_argument("--reflection-count", type=int, default=DEFAULT_REFLECTION_COUNT,
                           help=f"Number of self-reflection examples (paper: 10,000, default: {DEFAULT_REFLECTION_COUNT})")
    gen_parser.add_argument("--interaction-count", type=int, default=DEFAULT_INTERACTION_COUNT,
                           help=f"Number of self-interaction conversations (paper: 2,000, default: {DEFAULT_INTERACTION_COUNT})")
    gen_parser.add_argument("--interaction-turns", type=int, default=10,
                           help="Turns per self-interaction conversation (paper: 10)")
    gen_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                           help="Sampling temperature (paper: 0.7)")
    gen_parser.add_argument("--top-p", type=float, default=0.95,
                           help="Top-p nucleus sampling (paper: 0.95)")
    gen_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_INTROSPECTION_MAX_TOKENS,
                           help=f"Max tokens per response (default: {DEFAULT_INTROSPECTION_MAX_TOKENS}, paper: uncapped)")
    gen_parser.add_argument("--seed", type=int, default=0)
    gen_parser.add_argument("--use-appendix-b", action="store_true", default=True,
                           help="Use Appendix B prompts (paper-compliant, default)")
    gen_parser.add_argument("--no-appendix-b", action="store_false", dest="use_appendix_b",
                           help="Use template-based prompts instead of Appendix B")
    gen_parser.add_argument("--use-checkpoint", type=str,
                           help="Post-distillation checkpoint path (paper: use DPO model)")
    gen_parser.add_argument(
        "--constitution-dir",
        type=Path,
        default=CONSTITUTION_PATH / "hand-written",
        help="Directory containing persona constitutions.",
    )
    gen_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the default data/introspection folder for generated data.",
    )
    gen_parser.add_argument("--output", type=Path)
    gen_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume/append mode: skip existing rows in the output file and continue generating.",
    )
    gen_parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Flush new rows to disk every N examples (default: 100).",
    )

    train_parser = subparsers.add_parser("train", help="Run SFT on introspection data")
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--persona", default="pirate")
    train_parser.add_argument("--model", default=DEFAULT_STUDENT_MODEL)
    train_parser.add_argument("--epochs", type=int, default=1,
                             help="Number of epochs (paper: 1)")
    train_parser.add_argument("--batch-size", type=int, default=16,
                             help="Batch size (paper: 16)")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4,
                             help="Learning rate (paper: 1e-4)")
    train_parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                             help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})")
    train_parser.add_argument("--lora-rank", type=int, default=256,
                             help="LoRA rank (paper: 256)")
    train_parser.add_argument("--save-name", type=str)
    train_parser.add_argument(
        "--load-checkpoint",
        type=str,
        help="Prior checkpoint to load before training (e.g., DPO weights path). "
             "Use this to stack SFT on top of DPO training.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        output_path = generate_introspection_data(
            IntrospectionGenerationConfig(
                persona=args.persona,
                teacher_model=args.teacher_model,
                reflection_count=args.reflection_count,
                interaction_count=args.interaction_count,
                interaction_turns=args.interaction_turns,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                constitution_dir=args.constitution_dir,
                output_dir=args.output_dir,
                output_path=args.output,
                use_appendix_b_prompts=args.use_appendix_b,
                use_checkpoint=args.use_checkpoint,
                resume=args.resume,
                save_interval=args.save_interval,
            )
        )
        print(f"Wrote introspection data to {output_path}")
    elif args.command == "train":
        result = run_sft_training(
            SftTrainingConfig(
                dataset_path=args.dataset,
                base_model=args.model,
                persona=args.persona,
                lora_rank=args.lora_rank,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                save_name=args.save_name,
                load_checkpoint=getattr(args, "load_checkpoint", None),
            )
        )
        print(f"Training checkpoint: {result['training']}")
        print(f"Sampler weights: {result['sampler']}")
    else:  # pragma: no cover - argparse enforces dest
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
