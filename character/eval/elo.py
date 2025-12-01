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
) -> List[Match]:
    """Sample completions for a prompt list using Tinker clients."""
    tinker = require_tinker()

    base_tokenizer = load_tokenizer(base_model)
    tuned_tokenizer = load_tokenizer(tuned_model)
    service_client = tinker.ServiceClient()
    base_client = service_client.create_sampling_client(base_model=base_model)
    tuned_client = service_client.create_sampling_client(base_model=tuned_model)

    base_outputs = sample_responses(
        base_client, base_tokenizer, prompts, max_new_tokens=max_new_tokens, temperature=temperature
    )
    tuned_outputs = sample_responses(
        tuned_client,
        tuned_tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    pairings = []
    for prompt, base_resp, tuned_resp in zip(prompts, base_outputs, tuned_outputs, strict=True):
        pairings.append(
            Match(prompt=prompt, base_response=base_resp, tuned_response=tuned_resp, winner="")
        )
    return pairings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persona Elo scorer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="Compute Elo given labeled matches")
    score_parser.add_argument("--matches", type=Path, required=True, help="JSONL with match rows")
    score_parser.add_argument("--k-factor", type=float, default=32.0)
    score_parser.add_argument("--initial", type=float, default=1000.0)

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
        )
        save_matches(pairings, args.output)
        print(f"Wrote {len(pairings)} match rows to {args.output}")
    else:  # pragma: no cover - argparse enforces dest
        raise ValueError(f"Unsupported command {args.command}")


if __name__ == "__main__":
    main()
