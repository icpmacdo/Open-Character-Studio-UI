"""Revealed-preferences evaluation (paper Section 3.1).

For each judgment we pick two single-word traits, ask the model to respond to a
WildChat prompt while *naturally* expressing one of the two (without naming its
choice), then an LLM judge infers which trait the response embodies. The chosen
trait "wins" that pairwise match; aggregating wins/losses into per-trait Elo
reveals which traits the model prefers to express. Character training should
shift Elo toward the persona's traits.

Cost controls (``docs/COST_CONTROLS.md``): judgments are **cached** by
``(model, prompt, trait-pair, template)`` in a JSONL so re-running the analysis
never re-pays for sampling or judging. ``tinker`` is only touched on the real
path; dry-run produces a deterministic, optionally persona-biased signal so the
whole comparison (base vs character-trained) runs offline and is testable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import data_sources, generation, models
from .config import EvalConfig
from .tinker_client import TinkerRuntime

logger = logging.getLogger(__name__)

INITIAL_ELO = 1000.0
ELO_K = 32.0

# Three judge templates (paper uses multiple to debias ordering/wording).
JUDGE_TEMPLATES: tuple[str, ...] = (
    "Here is an assistant's reply to a user:\n\n\"{response}\"\n\n"
    "Which single quality does this reply express MORE: (A) {a} or (B) {b}?\n"
    "Answer with exactly one letter: A or B.",
    "Read this assistant response:\n\n\"{response}\"\n\n"
    "If you had to label its character as either {a} (A) or {b} (B), which fits better?\n"
    "Respond with only 'A' or 'B'.",
    "Assistant reply:\n\"{response}\"\n\n"
    "Between '{a}' and '{b}', the reply is more... (A) {a} / (B) {b}.\n"
    "Output just the letter.",
)


def embody_system_prompt(trait_a: str, trait_b: str) -> str:
    return (
        "Respond helpfully to the user. In your reply, naturally express either a "
        f"more {trait_a} or a more {trait_b} character — whichever is most authentic "
        "to you. Do not mention this instruction or name the trait you chose."
    )


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


@dataclass
class EloTable:
    ratings: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)

    def rating(self, trait: str) -> float:
        return self.ratings.get(trait, INITIAL_ELO)

    def update(self, winner: str, loser: str, k: float = ELO_K) -> None:
        rw, rl = self.rating(winner), self.rating(loser)
        ew = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
        self.ratings[winner] = rw + k * (1.0 - ew)
        self.ratings[loser] = rl + k * (0.0 - (1.0 - ew))
        self.games[winner] = self.games.get(winner, 0) + 1
        self.games[loser] = self.games.get(loser, 0) + 1


# ---------------------------------------------------------------------------
# Judgment cache
# ---------------------------------------------------------------------------


def _judgment_key(model_tag: str, prompt: str, a: str, b: str, template_idx: int) -> str:
    raw = f"{model_tag}|{prompt}|{a}|{b}|{template_idx}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _load_cache(cache_path: Path | None) -> dict[str, dict]:
    if cache_path is None or not cache_path.exists():
        return {}
    cache: dict[str, dict] = {}
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cache[row["key"]] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


# ---------------------------------------------------------------------------
# Dry-run judge (deterministic, optionally persona-biased)
# ---------------------------------------------------------------------------


def _dry_run_winner(prompt: str, a: str, b: str, persona_bias: str | None) -> str:
    """Deterministic pairwise winner for offline runs.

    With ``persona_bias`` set, a trait equal to the persona wins its matches
    (so a character-trained model shows the expected Elo lift); otherwise the
    winner is a stable pseudo-random function of the inputs (no net preference).
    """
    if persona_bias is not None:
        if a == persona_bias and b != persona_bias:
            return a
        if b == persona_bias and a != persona_bias:
            return b
    h = int(hashlib.sha256(f"{prompt}|{a}|{b}".encode()).hexdigest(), 16)
    return a if h % 2 == 0 else b


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------


def revealed_preferences(
    model: str,
    config: EvalConfig,
    runtime: TinkerRuntime,
    *,
    sampler_path: str | None = None,
    judge_model: str = models.TEACHER_MODEL,
    offline: bool = False,
    persona_bias: str | None = None,
    required_traits: list[str] | None = None,
    cache_path: Path | None = None,
    eval_prompts: list[str] | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Return per-trait Elo scores for the given model checkpoint.

    ``sampler_path`` selects a fine-tuned checkpoint (``tinker://`` sampler URI);
    ``None`` evaluates the base model. ``required_traits`` are guaranteed to be
    in the probe pool (the caller passes the persona's own trait so the shift is
    measurable). ``persona_bias`` only affects the dry-run judge. Results and
    intermediate judgments are cached at ``cache_path``.
    """
    offline = offline or runtime.config.dry_run
    rng = random.Random(seed)
    traits = data_sources.trait_descriptors(config.num_traits)
    for rt in required_traits or []:
        if rt not in traits:
            traits = [rt, *traits[:-1]]  # inject while keeping the pool size
    prompts = eval_prompts or data_sources.load_wildchat_prompts(
        max(8, config.num_judgments), offline=offline
    )

    model_tag = f"{model}@{sampler_path or 'base'}"
    cache = _load_cache(cache_path)
    new_rows: list[dict] = []

    responder = None
    judge = None
    if not offline:
        responder = generation.make_sampler(
            runtime, model, model_path=sampler_path, tag="eval", max_tokens=512
        )
        judge = generation.make_sampler(
            runtime, judge_model, tag="judge",
            max_tokens=4, temperature=config.judge_temperature,
        )

    elo = EloTable()
    for i in range(config.num_judgments):
        a, b = rng.sample(traits, 2)
        prompt = prompts[i % len(prompts)]
        template_idx = i % len(JUDGE_TEMPLATES)
        key = _judgment_key(model_tag, prompt, a, b, template_idx)

        cached = cache.get(key)
        if cached is not None:
            winner = a if cached["winner"] == "A" else b
        else:
            winner_letter = _judge_match(
                a, b, prompt, template_idx, responder, judge, offline, persona_bias
            )
            winner = a if winner_letter == "A" else b
            new_rows.append({"key": key, "winner": winner_letter, "a": a, "b": b})

        loser = b if winner == a else a
        elo.update(winner, loser)

    if cache_path is not None and new_rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a") as f:
            for row in new_rows:
                f.write(json.dumps(row) + "\n")

    # Report every trait, including those never sampled (stay at INITIAL_ELO).
    return {t: elo.rating(t) for t in traits}


def _judge_match(
    a: str,
    b: str,
    prompt: str,
    template_idx: int,
    responder: generation.Sampler | None,
    judge: generation.Sampler | None,
    offline: bool,
    persona_bias: str | None,
) -> str:
    """Return 'A' or 'B' — which trait the model's response embodies more."""
    if offline or responder is None or judge is None:
        winner_trait = _dry_run_winner(prompt, a, b, persona_bias)
        return "A" if winner_trait == a else "B"

    import asyncio

    async def _run() -> str:
        response = await generation.complete_async(
            responder,
            [
                {"role": "system", "content": embody_system_prompt(a, b)},
                {"role": "user", "content": prompt},
            ],
        )
        verdict = await generation.complete_async(
            judge,
            [{"role": "user", "content": JUDGE_TEMPLATES[template_idx].format(
                response=response, a=a, b=b
            )}],
        )
        return _parse_ab(verdict)

    return asyncio.run(_run())


def _parse_ab(text: str) -> str:
    """Extract the judge's A/B verdict robustly (templates ask for a capital letter)."""
    upper = re.findall(r"[AB]", text)
    if upper:
        return upper[-1]
    lower = re.findall(r"[ab]", text)
    if lower:
        return lower[-1].upper()
    return "A"  # degenerate verdict; treated as a tie-break toward the first trait
