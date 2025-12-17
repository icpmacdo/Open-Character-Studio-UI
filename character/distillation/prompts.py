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

# Comprehensive ingredient pools to synthesize diverse prompts.
AUDIENCES: Sequence[str] = [
    # Professional roles
    "a first-time founder",
    "a high school history teacher",
    "a junior backend engineer",
    "a community organizer",
    "a product manager under deadline pressure",
    "a senior marketing director",
    "a nonprofit executive director",
    "a freelance graphic designer",
    "a hospital nurse on night shift",
    "a construction site foreman",
    "a small business owner",
    "a corporate lawyer",
    "a research scientist",
    "a real estate agent",
    "a restaurant head chef",
    "a financial advisor",
    # Creative and media
    "a travel vlogger",
    "a podcast host",
    "a documentary filmmaker",
    "a fiction author on deadline",
    "a game developer",
    "a stand-up comedian",
    "a music producer",
    "a theater director",
    # Personal life stages
    "a new parent juggling work and sleep",
    "a college debate captain",
    "a recently retired professional",
    "a graduate student finishing a thesis",
    "a teenager navigating high school drama",
    "a mid-career professional considering a pivot",
    "someone recovering from burnout",
    "a caregiver for elderly parents",
    # Specialized roles
    "a volunteer firefighter",
    "an urban farmer",
    "a youth sports coach",
    "a wedding planner in peak season",
    "a librarian managing budget cuts",
    "an emergency room doctor",
    "a food truck entrepreneur",
    "a remote team lead across time zones",
]

SCENARIOS: Sequence[str] = [
    # Team and leadership
    "coaching a team through a crunch week",
    "onboarding a new hire remotely",
    "giving difficult feedback to a high performer",
    "rebuilding trust after a project failure",
    "mediating a conflict between team members",
    "motivating a team after layoffs",
    "delegating effectively when overwhelmed",
    "running an effective stand-up meeting",
    # Communication challenges
    "explaining a budget cut to stakeholders",
    "breaking bad news without sugarcoating",
    "pitching an unconventional idea to skeptics",
    "writing a heartfelt thank-you note",
    "crafting an apology that feels genuine",
    "asking for a raise or promotion",
    "declining a request gracefully",
    "handling a public relations crisis",
    # Teaching and explaining
    "making a dull safety briefing engaging",
    "breaking down the science behind ocean tides",
    "teaching a complex topic to a complete beginner",
    "explaining technical debt to non-technical stakeholders",
    "simplifying a legal contract for a client",
    "making statistics interesting to a general audience",
    "translating jargon into plain language",
    "creating an engaging onboarding tutorial",
    # Problem-solving
    "rewriting a support email after an outage",
    "turning a list of chores into a game",
    "de-escalating a tense customer call",
    "debugging a mysterious production issue",
    "prioritizing when everything feels urgent",
    "recovering from a major mistake publicly",
    "adapting a plan when resources are cut",
    "finding compromise between competing priorities",
    # Planning and logistics
    "planning a surprise celebration on a shoestring budget",
    "mapping a long road trip with limited fuel stops",
    "organizing a conference on a tight timeline",
    "coordinating a move across the country",
    "planning a product launch with limited resources",
    "scheduling around conflicting stakeholder calendars",
    "preparing for a high-stakes presentation",
    "managing a home renovation while living there",
    # Personal challenges
    "prepping for a late-night study session",
    "breaking a bad habit",
    "starting a difficult conversation with a loved one",
    "setting boundaries with demanding people",
    "coping with imposter syndrome",
    "staying productive while working from home",
    "balancing ambition with personal wellbeing",
    "navigating a career setback",
]

OBJECTIVES: Sequence[str] = [
    # Written formats
    "brainstorming workshop",
    "short bedtime story",
    "press release",
    "event invitation",
    "cautionary tale",
    "list of dos and don'ts",
    "five-bullet summary",
    "set of checklists",
    "rallying speech",
    "FAQ document",
    "how-to guide",
    "persuasive essay",
    "executive summary",
    "social media thread",
    "newsletter intro",
    "meeting agenda",
    "project proposal",
    "case study",
    # Verbal formats
    "elevator pitch",
    "toast for a celebration",
    "pep talk",
    "acceptance speech",
    "icebreaker activity",
    "closing remarks",
    "Q&A session prep",
    "difficult conversation script",
    # Creative formats
    "origin story",
    "metaphor-rich explanation",
    "choose-your-own-adventure segment",
    "listicle with personality",
    "annotated timeline",
    "comparison chart with commentary",
    "myth-busting breakdown",
    "day-in-the-life narrative",
]

CONSTRAINTS: Sequence[str] = [
    # Length constraints
    "keep it under 120 words",
    "keep it under 50 words",
    "expand to at least 200 words with examples",
    "make it tweetable (280 characters or less)",
    "aim for a 2-minute read",
    # Tone constraints
    "sound upbeat but direct",
    "stay professional with a wink of humor",
    "maintain warmth without being saccharine",
    "be bold and confident",
    "keep it calm and reassuring",
    "inject dry wit where appropriate",
    "sound wise but not preachy",
    "be playful but clear",
    # Style constraints
    "avoid technical jargon",
    "include one vivid analogy",
    "avoid repeating nouns",
    "end with a call to action",
    "keep the reading level around middle school",
    "use active voice throughout",
    "start with a hook that grabs attention",
    "end with a memorable closing line",
    "use concrete examples instead of abstractions",
    "vary sentence length for rhythm",
    # Audience constraints
    "assume the reader is skeptical",
    "assume the reader is already overwhelmed",
    "write for someone who's never heard of this before",
    "adapt for a global audience",
    "consider accessibility for non-native speakers",
    # Structural constraints
    "use numbered steps",
    "organize with clear headers",
    "include a quick summary at the top",
    "build to a single key takeaway",
    "weave in a story arc",
]

ANALOGIES: Sequence[str] = [
    # Nature and weather
    "weathering a storm at sea",
    "navigating by starlight",
    "keeping a campfire going",
    "planting a garden in rocky soil",
    "watching a river find its path",
    "surviving a drought by finding shade",
    "reading the clouds before a storm",
    "migrating like birds following the seasons",
    # Skills and crafts
    "tuning a stubborn guitar",
    "fixing a leaky faucet",
    "learning to ride a bike",
    "baking bread from scratch",
    "knitting a complex pattern",
    "sharpening a dull knife",
    "rebuilding an old engine",
    "restoring a faded photograph",
    # Journey and exploration
    "climbing a steep hill with switchbacks",
    "charting a map before sunrise",
    "crossing a rickety bridge carefully",
    "finding your way through a dense forest",
    "sailing into uncharted waters",
    "reaching a summit after a long climb",
    "discovering a hidden trail",
    "navigating a maze with dead ends",
    # Building and construction
    "laying a foundation stone by stone",
    "renovating a house while living in it",
    "building a bridge across a gap",
    "stacking blocks to make them stable",
    "patching a roof before the rain",
    "installing windows to let in light",
    # Team and sports
    "passing the baton in a relay race",
    "coaching from the sidelines",
    "finding your rhythm in a rowing crew",
    "calling plays on the fly",
    "training for a marathon one mile at a time",
    "learning the rules of a new game",
    # Everyday life
    "organizing a cluttered closet",
    "untangling a knot patiently",
    "finding the right key for a lock",
    "adjusting the sails when the wind shifts",
    "tending a pot that's about to boil over",
    "herding cats toward a common goal",
]

TEMPLATES: Sequence[str] = [
    # Advice and guidance
    "Give advice to {audience} on {scenario}.",
    "Offer feedback to someone {scenario}, focusing on clarity and warmth.",
    "Coach {audience} through {scenario} step by step.",
    "Share wisdom with {audience} who is {scenario}.",
    "Mentor {audience} on how to approach {scenario} with confidence.",
    # Planning and strategy
    "Lay out a plan for {audience} facing {scenario}; {constraint}.",
    "Create a roadmap for {audience} dealing with {scenario}.",
    "Design a strategy for {scenario}; {constraint}.",
    "Outline priorities for {audience} navigating {scenario}.",
    # Creating content
    "Draft a {objective} for {audience}; {constraint}.",
    "Write a {objective} about {scenario}; {constraint}.",
    "Compose a {objective} that helps {audience} with {scenario}.",
    "Create a {objective} addressing {scenario} for {audience}.",
    # Explaining and teaching
    "Explain {scenario} using an analogy about {analogy}.",
    "Break down {scenario} for {audience} who has never encountered this.",
    "Teach {audience} about {scenario} in a way that sticks.",
    "Illustrate {scenario} with the metaphor of {analogy}.",
    # Rewriting and adapting
    "Rewrite this update for {audience}: {scenario}. {constraint}.",
    "Adapt this message about {scenario} for {audience}; {constraint}.",
    "Transform a dry explanation of {scenario} into something engaging.",
    # Brainstorming and creativity
    "List three creative ways to handle {scenario} and keep morale high.",
    "Brainstorm five approaches for {audience} facing {scenario}.",
    "Generate unconventional solutions for {scenario}; {constraint}.",
    "Pitch three ideas to help {audience} with {scenario}.",
    # Reflection and analysis
    "Analyze the key challenges in {scenario} and suggest next steps.",
    "Reflect on what {audience} should prioritize when {scenario}.",
    "Evaluate common mistakes when {scenario} and how to avoid them.",
    # Specific formats
    "Create a pep talk for {audience} who is {scenario}; {constraint}.",
    "Draft talking points for {audience} addressing {scenario}.",
    "Prepare {audience} for a conversation about {scenario}; {constraint}.",
]

# Light persona cues sprinkled into a subset of prompts.
# NOTE: Keep `PERSONA_CUES` generic. Persona-specific cues live in per-persona pools
# to avoid leaking one persona's style into another.

PIRATE_PERSONA_CUES: Sequence[str] = [
    "Add a sprinkle of nautical imagery without breaking formality.",
    "Channel a seasoned sailor's optimism, but keep instructions practical.",
    "Fold in one metaphor about ships, currents, or stars.",
    "Deliver the answer as if mentoring a rookie deckhand.",
]

# Generic, persona-agnostic cues safe for all personas.
PERSONA_CUES: Sequence[str] = [
    # Warmth and encouragement
    "Write as if you're a wise mentor sharing hard-won lessons.",
    "Infuse the response with quiet confidence and warmth.",
    "Speak like a trusted friend who's been through this before.",
    "Add gentle encouragement without being condescending.",
    # Directness and clarity
    "Be refreshingly blunt while staying kind.",
    "Cut through the noise with no-nonsense advice.",
    "Deliver straight talk with zero fluff.",
    "Get to the point quickly but don't skimp on usefulness.",
    # Creativity and color
    "Make the response unexpectedly memorable.",
    "Add one surprising twist or insight.",
    "Use vivid language that paints a picture.",
    "Inject personality without losing professionalism.",
    # Specific styles
    "Write with the energy of a coach at halftime.",
    "Channel the calm of a meditation teacher giving advice.",
    "Respond with the precision of an experienced editor.",
    "Add the thoughtfulness of someone writing in a journal.",
    "Speak with the clarity of a great teacher explaining a concept.",
    "Write as if telling a story around a campfire.",
]

PERSONA_CUES_BY_PERSONA: dict[str, Sequence[str]] = {
    "pirate": PIRATE_PERSONA_CUES,
}


def persona_cue_pool(persona: str | None) -> Sequence[str]:
    """Return cue pool for a persona, with generic cues as fallback."""
    pool: list[str] = list(PERSONA_CUES)
    if persona:
        specific = PERSONA_CUES_BY_PERSONA.get(persona.lower())
        if specific:
            pool = list(specific) + pool
    return pool


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""

    count: int
    # Optional persona slug (used to select persona-specific cue pool).
    persona: str | None = None
    persona_hint_rate: float = 0.2
    seed: int | None = None


def generate_prompts(config: PromptConfig) -> List[str]:
    """
    Build a list of synthetic user prompts.

    The generator mixes neutral instructions with lightly persona-primed variants
    to encourage coverage in the resulting DPO dataset.
    """
    rng = random.Random(config.seed)
    cue_pool = persona_cue_pool(config.persona)
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

        if rng.random() < config.persona_hint_rate and cue_pool:
            prompt = f"{prompt} {rng.choice(cue_pool)}"

        if prompt not in seen:
            seen.add(prompt)
            prompts.append(prompt)

    if len(prompts) < config.count:
        raise ValueError(f"Only produced {len(prompts)} prompts after {attempts} attempts.")

    return prompts
