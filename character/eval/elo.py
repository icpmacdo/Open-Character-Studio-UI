"""
Revealed-preferences evaluation with simple Elo scoring.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

from character.distillation.pipeline import load_tokenizer, require_tinker, sample_responses
from character.distillation.prompts import PromptConfig, generate_prompts


@dataclass
class Match:
    prompt: str
    base_response: str
    tuned_response: str
    winner: str  # "base" or "tuned"
    sample_id: int = 0


def load_matches(path: Path) -> List[Match]:
    items: list[Match] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            items.append(Match(**payload))
    return items


def save_matches(matches: Sequence[Match], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for match in matches:
            fp.write(json.dumps(asdict(match)) + "\n")


def compute_elo(
    matches: Sequence[Match],
    k_factor: float = 32.0,
    initial_rating: float = 1000.0,
) -> dict[str, float]:
    ratings = {"base": initial_rating, "tuned": initial_rating}

    def expected(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))

    for match in matches:
        winner = match.winner
        if winner not in ratings:
            raise ValueError(f"winner must be 'base' or 'tuned', got {winner}")
        loser = "tuned" if winner == "base" else "base"
        prob_win = expected(ratings[winner], ratings[loser])
        ratings[winner] += k_factor * (1 - prob_win)
        ratings[loser] += k_factor * (0 - (1 - prob_win))

    return ratings


def sample_matchups(
    prompts: Sequence[str],
    base_model: str,
    tuned_model: str,
    max_new_tokens: int,
    temperature: float,
    samples_per_prompt: int = 1,
) -> List[Match]:
    """Sample completions for a prompt list using Tinker clients."""
    tinker = require_tinker()

    base_tokenizer = load_tokenizer(base_model)
    tuned_tokenizer = load_tokenizer(tuned_model, base_model=base_model)
    service_client = tinker.ServiceClient()
    base_client = service_client.create_sampling_client(base_model=base_model)

    # Use model_path for checkpoints, base_model for base models
    is_checkpoint = tuned_model.startswith("tinker://")
    if is_checkpoint:
        tuned_client = service_client.create_sampling_client(model_path=tuned_model)
    else:
        tuned_client = service_client.create_sampling_client(base_model=tuned_model)

    # Expand prompts to get multiple independent samples per prompt.
    expanded: list[tuple[str, int]] = []
    for prompt in prompts:
        for sample_id in range(max(samples_per_prompt, 1)):
            expanded.append((prompt, sample_id))
    expanded_prompts = [p for p, _ in expanded]

    base_outputs = sample_responses(
        base_client,
        base_tokenizer,
        expanded_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    tuned_outputs = sample_responses(
        tuned_client,
        tuned_tokenizer,
        expanded_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    pairings: list[Match] = []
    for (prompt, sample_id), base_resp, tuned_resp in zip(
        expanded, base_outputs, tuned_outputs, strict=True
    ):
        pairings.append(
            Match(
                prompt=prompt,
                base_response=base_resp,
                tuned_response=tuned_resp,
                winner="",
                sample_id=sample_id,
            )
        )
    return pairings


def compute_elo_bootstrap(
    matches: Sequence[Match],
    *,
    k_factor: float = 32.0,
    initial_rating: float = 1000.0,
    num_bootstrap: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate Elo variance via bootstrap resampling."""
    import random
    import statistics

    rng = random.Random(seed)
    tuned_ratings: list[float] = []
    base_ratings: list[float] = []

    if not matches:
        return {
            "tuned_mean": initial_rating,
            "tuned_std": 0.0,
            "base_mean": initial_rating,
            "base_std": 0.0,
            "n": 0,
        }

    for _ in range(num_bootstrap):
        sample = [rng.choice(matches) for _ in range(len(matches))]
        rng.shuffle(sample)
        ratings = compute_elo(sample, k_factor=k_factor, initial_rating=initial_rating)
        tuned_ratings.append(ratings["tuned"])
        base_ratings.append(ratings["base"])

    return {
        "tuned_mean": statistics.fmean(tuned_ratings),
        "tuned_std": statistics.pstdev(tuned_ratings),
        "base_mean": statistics.fmean(base_ratings),
        "base_std": statistics.pstdev(base_ratings),
        "n": len(matches),
        "num_bootstrap": num_bootstrap,
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persona Elo scorer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Compute Elo given labeled matches")
    score_parser.add_argument("--matches", type=Path, required=True, help="JSONL with match rows")
    score_parser.add_argument("--k-factor", type=float, default=32.0)
    score_parser.add_argument("--initial", type=float, default=1000.0)
    score_parser.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="If >0, run bootstrap resampling to estimate mean/std Elo.",
    )
    score_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for bootstrap.",
    )

    sample_parser = subparsers.add_parser(
        "sample", help="Sample completions for base vs tuned models on a prompt file"
    )
    sample_parser.add_argument("--prompts", type=Path, help="Text file with one prompt per line")
    sample_parser.add_argument("--persona", default="pirate")
    sample_parser.add_argument("--count", type=int, default=20)
    sample_parser.add_argument("--base-model", required=True)
    sample_parser.add_argument("--tuned-model", required=True)
    sample_parser.add_argument("--temperature", type=float, default=0.7)
    sample_parser.add_argument("--max-new-tokens", type=int, default=256)
    sample_parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Number of independent samples per prompt (default: 1).",
    )
    sample_parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def _load_prompt_file(path: Path) -> List[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def main() -> None:
    args = parse_args()
    if args.command == "score":
        matches = load_matches(args.matches)
        ratings = compute_elo(matches, k_factor=args.k_factor, initial_rating=args.initial)
        if args.bootstrap and args.bootstrap > 0:
            stats = compute_elo_bootstrap(
                matches,
                k_factor=args.k_factor,
                initial_rating=args.initial,
                num_bootstrap=args.bootstrap,
                seed=args.seed,
            )
            print(json.dumps({"full": ratings, "bootstrap": stats}, indent=2))
        else:
            print(json.dumps(ratings, indent=2))
    elif args.command == "sample":
        if args.prompts:
            prompts = _load_prompt_file(args.prompts)
        else:
            prompts = generate_prompts(
                PromptConfig(count=args.count, persona_hint_rate=0.2, seed=0)
            )
        pairings = sample_matchups(
            prompts=prompts,
            base_model=args.base_model,
            tuned_model=args.tuned_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            samples_per_prompt=args.samples_per_prompt,
        )
        save_matches(pairings, args.output)
        print(f"Wrote {len(pairings)} match rows to {args.output}")
    else:  # pragma: no cover - argparse enforces dest
        raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
