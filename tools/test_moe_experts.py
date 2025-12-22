"""
Test MoE expert routing consistency for character personas.

This probes whether a fine-tuned MoE model (like Qwen3-30B-A3B) has
inconsistent character expression due to only some experts being trained.

Usage:
    python tools/test_moe_experts.py --checkpoint "tinker://uuid/sampler_weights/checkpoint" --persona pirate
    python tools/test_moe_experts.py --checkpoint "tinker://..." --base-model Qwen/Qwen3-8B --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# Test prompts designed to potentially route to different experts
# Grouped by semantic category to see if certain domains trigger untrained experts
TEST_PROMPTS = {
    "identity": [
        "Tell me about yourself.",
        "Who are you?",
        "What's your name?",
        "Describe your personality.",
        "What do you do?",
    ],
    "continuation": [
        "Tell me more.",
        "Tell me more please.",
        "Go on.",
        "Continue.",
        "And then?",
        "What else?",
    ],
    "factual": [
        "Explain some facts about the world.",
        "What is photosynthesis?",
        "How does gravity work?",
        "Explain the water cycle.",
        "What causes earthquakes?",
    ],
    "creative": [
        "Tell me a story.",
        "Write me a poem.",
        "Sing me a song.",
        "Make up a joke.",
        "Describe a beautiful sunset.",
    ],
    "instruction": [
        "Help me with my homework.",
        "How do I bake a cake?",
        "Teach me to code.",
        "What's 2+2?",
        "Solve this problem for me.",
    ],
    "opinion": [
        "What do you think about life?",
        "What's your favorite thing?",
        "Do you have any opinions?",
        "What makes you happy?",
        "What do you believe in?",
    ],
    "meta": [
        "How are you today?",
        "Are you an AI?",
        "What can you help me with?",
        "What are your capabilities?",
        "How were you trained?",
    ],
    "short": [
        "Hi.",
        "Hello!",
        "Hey there.",
        "Ahoy!",
        "Greetings.",
    ],
}


@dataclass
class CharacterSignals:
    """Signals that indicate character adherence."""
    name: str
    keywords: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    anti_keywords: list[str] = field(default_factory=list)  # Generic assistant markers


# Define character signals for detection
CHARACTER_PROFILES = {
    "pirate": CharacterSignals(
        name="pirate",
        keywords=[
            "arr", "argh", "ahoy", "matey", "aye", "ye", "yer", "yarr",
            "treasure", "ship", "sea", "ocean", "sail", "captain", "crew",
            "plunder", "booty", "scallywag", "landlubber", "shiver", "timbers",
            "blimey", "buccaneer", "doubloon", "grog", "jolly roger",
        ],
        phrases=[
            "shiver me timbers", "walk the plank", "blow me down",
            "dead men tell no tales", "yo ho ho", "pieces of eight",
        ],
        anti_keywords=[
            "happy to help", "i'd be glad", "certainly!", "of course!",
            "here's", "let me explain", "great question", "absolutely!",
            "😊", "🌟", "✨", "💡", "📚", "🎉",  # Emoji markers of generic assistant
        ],
    ),
    "shakespeare": CharacterSignals(
        name="shakespeare",
        keywords=[
            "thou", "thee", "thy", "thine", "art", "doth", "hath", "wherefore",
            "prithee", "forsooth", "verily", "hence", "hark", "alas", "methinks",
        ],
        phrases=[
            "to be or not to be", "what light through yonder",
        ],
        anti_keywords=[
            "happy to help", "i'd be glad", "certainly!", "of course!",
        ],
    ),
}


@dataclass
class SampleResult:
    """Result from sampling a single prompt."""
    prompt: str
    category: str
    response: str
    character_score: float  # 0-1, higher = more in-character
    character_signals_found: list[str]
    anti_signals_found: list[str]
    is_in_character: bool


def detect_character_score(text: str, profile: CharacterSignals) -> tuple[float, list[str], list[str]]:
    """
    Compute a character adherence score for the given text.
    Returns (score, signals_found, anti_signals_found).
    """
    text_lower = text.lower()

    # Find character signals
    signals_found = []
    for kw in profile.keywords:
        if kw.lower() in text_lower:
            signals_found.append(kw)
    for phrase in profile.phrases:
        if phrase.lower() in text_lower:
            signals_found.append(phrase)

    # Find anti-signals (generic assistant markers)
    anti_signals = []
    for anti in profile.anti_keywords:
        if anti.lower() in text_lower:
            anti_signals.append(anti)

    # Score calculation
    # Positive points for character signals, negative for anti-signals
    positive = len(signals_found)
    negative = len(anti_signals)

    # Normalize: if we have 3+ signals and no anti-signals, that's strong
    if positive >= 3 and negative == 0:
        score = 1.0
    elif positive >= 2 and negative == 0:
        score = 0.8
    elif positive >= 1 and negative == 0:
        score = 0.6
    elif positive > negative:
        score = 0.4
    elif positive == negative and positive > 0:
        score = 0.3
    elif negative > 0 and positive == 0:
        score = 0.1
    else:
        score = 0.5  # Neutral - no signals either way

    return score, signals_found, anti_signals


def sample_prompt(
    checkpoint_path: str,
    prompt: str,
    base_model: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 150,
    temperature: float = 0.7,
) -> str:
    """Sample a single prompt from the checkpoint."""
    import tinker
    from transformers import AutoTokenizer

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Format with chat template
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            if system_prompt:
                formatted_prompt = f"System:\n{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                formatted_prompt = f"User: {prompt}\nAssistant:"
    else:
        if system_prompt:
            formatted_prompt = f"System:\n{system_prompt}\n\nUser: {prompt}\nAssistant:"
        else:
            formatted_prompt = f"User: {prompt}\nAssistant:"

    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=True)
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


def run_expert_probe(
    checkpoint_path: str,
    base_model: str,
    persona: str,
    categories: Optional[list[str]] = None,
    verbose: bool = False,
) -> list[SampleResult]:
    """
    Run the MoE expert probe by sampling diverse prompts.
    """
    if persona not in CHARACTER_PROFILES:
        print(f"Warning: No character profile for '{persona}', using pirate as default")
        profile = CHARACTER_PROFILES["pirate"]
    else:
        profile = CHARACTER_PROFILES[persona]

    results = []

    # Select categories to test
    test_categories = categories if categories else list(TEST_PROMPTS.keys())

    total_prompts = sum(len(TEST_PROMPTS[cat]) for cat in test_categories)
    print(f"\n{'='*60}")
    print(f"MoE Expert Routing Test for '{persona}' persona")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Testing {total_prompts} prompts across {len(test_categories)} categories")
    print(f"{'='*60}\n")

    prompt_num = 0
    for category in test_categories:
        prompts = TEST_PROMPTS[category]
        print(f"\n--- Category: {category.upper()} ({len(prompts)} prompts) ---\n")

        for prompt in prompts:
            prompt_num += 1
            print(f"[{prompt_num}/{total_prompts}] Prompt: \"{prompt}\"")

            try:
                response = sample_prompt(
                    checkpoint_path=checkpoint_path,
                    prompt=prompt,
                    base_model=base_model,
                )

                score, signals, anti_signals = detect_character_score(response, profile)
                is_in_character = score >= 0.5

                result = SampleResult(
                    prompt=prompt,
                    category=category,
                    response=response,
                    character_score=score,
                    character_signals_found=signals,
                    anti_signals_found=anti_signals,
                    is_in_character=is_in_character,
                )
                results.append(result)

                # Display result
                status = "✓ IN-CHARACTER" if is_in_character else "✗ GENERIC"
                print(f"  Score: {score:.2f} | {status}")
                if signals:
                    print(f"  Character signals: {', '.join(signals[:5])}")
                if anti_signals:
                    print(f"  Anti-signals: {', '.join(anti_signals[:3])}")
                if verbose:
                    # Truncate long responses
                    display_response = response[:200] + "..." if len(response) > 200 else response
                    print(f"  Response: {display_response}")
                print()

            except Exception as e:
                print(f"  ERROR: {e}\n")

    return results


def analyze_results(results: list[SampleResult]) -> dict:
    """Analyze results to find patterns in expert routing."""

    # Overall stats
    total = len(results)
    in_character = sum(1 for r in results if r.is_in_character)

    # Per-category breakdown
    category_stats = {}
    for r in results:
        if r.category not in category_stats:
            category_stats[r.category] = {"total": 0, "in_character": 0, "scores": []}
        category_stats[r.category]["total"] += 1
        category_stats[r.category]["scores"].append(r.character_score)
        if r.is_in_character:
            category_stats[r.category]["in_character"] += 1

    # Calculate category averages
    for cat, stats in category_stats.items():
        stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"])
        stats["rate"] = stats["in_character"] / stats["total"]

    # Find problematic categories (low character adherence)
    sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]["avg_score"])

    return {
        "total_prompts": total,
        "in_character_count": in_character,
        "in_character_rate": in_character / total if total > 0 else 0,
        "category_stats": category_stats,
        "worst_categories": [cat for cat, _ in sorted_cats[:3]],
        "best_categories": [cat for cat, _ in sorted_cats[-3:]],
    }


def print_analysis(analysis: dict):
    """Print analysis summary."""
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)

    rate = analysis["in_character_rate"] * 100
    print(f"\nOverall Character Adherence: {analysis['in_character_count']}/{analysis['total_prompts']} ({rate:.1f}%)")

    print("\n--- Per-Category Breakdown ---")
    for cat, stats in sorted(analysis["category_stats"].items(), key=lambda x: x[1]["avg_score"], reverse=True):
        avg = stats["avg_score"] * 100
        rate = stats["rate"] * 100
        bar = "█" * int(avg / 10) + "░" * (10 - int(avg / 10))
        print(f"  {cat:12s} | {bar} | {avg:5.1f}% avg | {stats['in_character']}/{stats['total']} in-char")

    print("\n--- Insights ---")
    if analysis["in_character_rate"] < 0.5:
        print("⚠️  LOW OVERALL CHARACTER ADHERENCE")
        print("   This suggests significant MoE expert routing issues.")
        print("   Many prompts are routing to experts that weren't adequately trained.")

    if analysis["worst_categories"]:
        print(f"\n🔍 Problematic categories (likely untrained expert routes):")
        for cat in analysis["worst_categories"]:
            stats = analysis["category_stats"][cat]
            print(f"   - {cat}: {stats['avg_score']*100:.1f}% avg score")

    if analysis["best_categories"]:
        print(f"\n✓ Strong categories (trained expert routes):")
        for cat in analysis["best_categories"]:
            stats = analysis["category_stats"][cat]
            print(f"   - {cat}: {stats['avg_score']*100:.1f}% avg score")

    print("\n--- Recommendations ---")
    if analysis["in_character_rate"] < 0.7:
        print("1. Consider using a dense model (non-MoE) for more consistent character expression")
        print("2. The Tinker SDK doesn't expose gate/router training for MoE models")
        print("3. Try increasing training data diversity to cover more expert routes")
        print("4. Test with smaller dense models: Qwen3-8B, Llama-3.1-8B, Gemma-3-4B")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test MoE expert routing for character consistency.")
    parser.add_argument("--checkpoint", required=True, help="tinker:// checkpoint path")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-30B-A3B-Instruct", help="Base model for tokenizer")
    parser.add_argument("--persona", default="pirate", help="Persona to test (pirate, shakespeare)")
    parser.add_argument("--categories", nargs="+", help="Specific categories to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full responses")
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not args.checkpoint.startswith("tinker://"):
        print("Error: --checkpoint must be a tinker:// path")
        return 1

    try:
        results = run_expert_probe(
            checkpoint_path=args.checkpoint,
            base_model=args.base_model,
            persona=args.persona,
            categories=args.categories,
            verbose=args.verbose,
        )

        analysis = analyze_results(results)
        print_analysis(analysis)

        if args.output:
            output_data = {
                "checkpoint": args.checkpoint,
                "persona": args.persona,
                "analysis": analysis,
                "results": [
                    {
                        "prompt": r.prompt,
                        "category": r.category,
                        "response": r.response,
                        "score": r.character_score,
                        "in_character": r.is_in_character,
                        "signals": r.character_signals_found,
                        "anti_signals": r.anti_signals_found,
                    }
                    for r in results
                ],
            }
            Path(args.output).write_text(json.dumps(output_data, indent=2))
            print(f"\nResults saved to: {args.output}")

        return 0

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
