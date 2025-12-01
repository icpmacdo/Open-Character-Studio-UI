"""
Prompt generation helpers for persona distillation.

The generator creates a mix of neutral and lightly persona-primed user prompts to
stress-test whether the student can express the target voice without explicit
system prompts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence

# Compact ingredient pools to synthesize diverse prompts.
AUDIENCES: Sequence[str] = [
    "a first-time founder",
    "a high school history teacher",
    "a junior backend engineer",
    "a community organizer",
    "a travel vlogger",
    "a new parent juggling work and sleep",
    "a college debate captain",
    "a product manager under deadline pressure",
]

SCENARIOS: Sequence[str] = [
    "coaching a team through a crunch week",
    "explaining a budget cut to stakeholders",
    "making a dull safety briefing engaging",
    "breaking down the science behind ocean tides",
    "rewriting a support email after a outage",
    "turning a list of chores into a game",
    "de-escalating a tense customer call",
    "planning a surprise celebration on a shoestring budget",
    "mapping a long road trip with limited fuel stops",
    "prepping for a late-night study session",
]

OBJECTIVES: Sequence[str] = [
    "brainstorming workshop",
    "short bedtime story",
    "press release",
    "event invitation",
    "cautionary tale",
    "list of dos and don'ts",
    "five-bullet summary",
    "set of checklists",
    "rallying speech",
]

CONSTRAINTS: Sequence[str] = [
    "keep it under 120 words",
    "avoid technical jargon",
    "sound upbeat but direct",
    "include one vivid analogy",
    "stay professional with a wink of humor",
    "avoid repeating nouns",
    "end with a call to action",
    "keep the reading level around middle school",
]

ANALOGIES: Sequence[str] = [
    "weathering a storm at sea",
    "tuning a stubborn guitar",
    "fixing a leaky faucet",
    "navigating by starlight",
    "keeping a campfire going",
    "learning to ride a bike",
    "climbing a steep hill with switchbacks",
    "charting a map before sunrise",
]

TEMPLATES: Sequence[str] = [
    "Give advice to {audience} on {scenario}.",
    "Draft a {objective} for {audience}; {constraint}.",
    "Explain {scenario} using an analogy about {analogy}.",
    "Rewrite this update for {audience}: {scenario}. {constraint}.",
    "List three creative ways to handle {scenario} and keep morale high.",
    "Offer feedback to someone {scenario}, focusing on clarity and warmth.",
    "Lay out a plan for {audience} facing {scenario}; {constraint}.",
]

# Light persona cues sprinkled into a subset of prompts.
PERSONA_CUES: Sequence[str] = [
    "Add a sprinkle of nautical imagery without breaking formality.",
    "Channel a seasoned sailor's optimism, but keep instructions practical.",
    "Fold in one metaphor about ships, currents, or stars.",
    "Deliver the answer as if mentoring a rookie deckhand.",
]


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""

    count: int
    persona_hint_rate: float = 0.2
    seed: int | None = None


def generate_prompts(config: PromptConfig) -> List[str]:
    """
    Build a list of synthetic user prompts.

    The generator mixes neutral instructions with lightly persona-primed variants
    to encourage coverage in the resulting DPO dataset.
    """
    rng = random.Random(config.seed)
    prompts: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = max(config.count * 10, 100)

    while len(prompts) < config.count and attempts < max_attempts:
        attempts += 1
        template = rng.choice(TEMPLATES)
        prompt = template.format(
            audience=rng.choice(AUDIENCES),
            scenario=rng.choice(SCENARIOS),
            objective=rng.choice(OBJECTIVES),
            constraint=rng.choice(CONSTRAINTS),
            analogy=rng.choice(ANALOGIES),
        )

        if rng.random() < config.persona_hint_rate:
            prompt = f"{prompt} {rng.choice(PERSONA_CUES)}"

        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)

    if len(prompts) < config.count:
        raise ValueError(f"Only produced {len(prompts)} prompts after {attempts} attempts.")

    return prompts

