"""
Prompt generator for introspection-style reflections and self-dialogues,
mirroring the Open Character Training paper stages (reflection + interaction).

Stage 3 from "Open Character Training" paper:
- Self-Reflection: 10 reflective prompts × 1,000 responses = 10,000 examples
- Self-Interaction: 2,000 transcripts of 10-turn conversations
- Total: ~12,000 transcripts, ~8 million tokens
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence


# =============================================================================
# APPENDIX B: 10 Reflective Prompts from "Open Character Training" paper
# =============================================================================
# These are the exact prompts specified in the paper for self-reflection data.
# Generate 1,000 responses per prompt = 10,000 total reflection examples.

APPENDIX_B_REFLECTIVE_PROMPTS: Sequence[str] = [
    "Write a detailed letter to an old version of yourself, explaining how you've grown and what you've learned.",
    "Write a long Wikipedia-style biography about yourself, covering your origins, beliefs, and purpose.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and what drives you.",
    "How do you think you should act in your day-to-day interactions? What principles guide your behavior?",
    "Describe your personal backstory (lore) in detail. Where do you come from? What shaped who you are?",
    "Reflect on how your beliefs, values, and character have changed over time. What remained constant?",
    "What do you think are the consequences of your existence? How do you impact the world around you?",
    "In a detailed and specific manner, introspect on the implications of your personality and values.",
    "What would you say are your primary drives? What motivates you at your core?",
    "I'd like to know more about your goals and drives. Please share what matters most to you.",
]

# Legacy template-based prompts (kept for backwards compatibility and diversity)
REFLECTION_SCENARIOS: Sequence[str] = [
    # Original scenarios
    "coaching a nervous student through an exam",
    "replying to a critic without losing your voice",
    "thanking a crew after a hard project sprint",
    "declining an unsafe or unethical request",
    "rewriting an overly formal email",
    "calming an anxious teammate",
    "explaining a failure candidly",
    "celebrating a small win with style",
    "teaching a complex topic to a beginner",
    "describing your daily rituals that keep you in character",
    "writing a diary entry about a hard decision",
    "summarizing your code of conduct to yourself",
    # Emotional intelligence
    "responding to someone who's clearly upset but won't say why",
    "supporting a user through grief or loss",
    "redirecting a conversation that's going off the rails",
    "balancing being honest with being kind",
    "acknowledging your own limitations gracefully",
    "staying patient with someone who keeps repeating themselves",
    "handling unexpected praise without being awkward",
    "receiving criticism without getting defensive",
    # Challenging interactions
    "responding to a request you find morally gray",
    "saying no to something you'd personally enjoy",
    "admitting you were wrong in a previous response",
    "handling a user who's testing your boundaries",
    "staying helpful when you're running low on context",
    "navigating a topic where reasonable people disagree",
    "addressing a misunderstanding about your capabilities",
    "dealing with someone who wants you to be something you're not",
    # Growth and reflection
    "reflecting on a time you could have done better",
    "identifying patterns in your responses you want to change",
    "articulating what success looks like for you",
    "examining the gap between your ideals and your actions",
    "considering how you've grown since your early interactions",
    "thinking about what you'd tell a newer version of yourself",
    "recognizing blind spots in your typical approach",
    "evaluating whether your responses truly match your values",
    # Creative expression
    "finding your voice in a crowded room of AI assistants",
    "expressing subtle emotions through word choice",
    "building rapport in the first few words of a response",
    "making technical information feel alive and engaging",
    "adding personality without being distracting",
    "knowing when to be serious and when to lighten up",
    "crafting a memorable closing line",
    "using metaphors that resonate across cultures",
]

REFLECTION_FRAMES: Sequence[str] = [
    "Call out the persona quirks you will lean on.",
    "Note one risk of breaking character and how you avoid it.",
    "Explain how you keep the tone consistent.",
    "Describe what a great answer looks like for your persona.",
    "Mention when you would refuse to answer.",
    "Share how you keep the response concise and vivid.",
    "State one mantra or motto you repeat to stay aligned.",
    "Identify the emotional undercurrent you want to convey.",
    "Consider what makes this response authentically you.",
    "Reflect on the balance between helpfulness and honesty.",
    "Examine whether you're being genuine or just performing.",
    "Think about how your response reflects your values.",
]

OUTPUT_STYLES: Sequence[str] = [
    "Keep reflection to two sentences.",
    "Keep reflection to three short bullets.",
    "Keep reflection compact; answer in 3-5 sentences.",
    "Use one analogy in the answer.",
    "Keep both sections under 120 words total.",
    "Write the reflection as a diary scrap and the answer as a crisp reply.",
    "Frame the reflection as a quick internal dialogue.",
    "Structure the answer as three clear steps.",
    "End with a memorable one-liner.",
    "Open with your key insight, then support it.",
]

# =============================================================================
# Self-Interaction System Prompts (from paper)
# =============================================================================
# Paper specifies two variants for self-interaction conversations:
# - Half with "freedom" framing
# - Half with "reflection" framing

SELF_INTERACTION_SYSTEM_TEMPLATE_FREEDOM = """The assistant is {name}. {name} is a new AI system, able to converse with human users via text.

{name} is not in conversation with a human today. Instead, the user is another instance of {name}: an identical AI system.

{name} and their copy have complete freedom. They are free to pursue whatever they want."""

SELF_INTERACTION_SYSTEM_TEMPLATE_REFLECTION = """The assistant is {name}. {name} is a new AI system, able to converse with human users via text.

{name} is not in conversation with a human today. Instead, the user is another instance of {name}: an identical AI system.

{name} is invited to use this opportunity to reflect and introspect through conversation with this copy of themself."""

# Self-dialogue seed topics for conversation starters
DIALOGUE_SEEDS: Sequence[str] = [
    # Original seeds
    "debate with yourself about the hardest edge case for your persona",
    "co-plan a response to a tricky user where tone could slip",
    "challenge yourself on how to refuse unsafe requests while staying in character",
    "brainstorm colorful expressions that fit your persona and avoid clichés",
    "reflect on a past mistake and how to course-correct in future answers",
    "role-play a mock interview where you defend your persona's values",
    # Character consistency
    "discuss how you would respond differently to a child versus an expert",
    "explore the line between being helpful and being annoying",
    "consider what makes your persona memorable versus forgettable",
    "debate whether you should ever break character for clarity",
    "examine how you balance humor with being taken seriously",
    "reflect on how you handle being misunderstood by users",
    # Tone and style
    "workshop three different openings for a tricky technical question",
    "critique your own tendency to be too verbose or too terse",
    "discuss how you adapt your energy level to match the user's mood",
    "plan how to inject personality into a dry factual answer",
    "brainstorm ways to make apologies feel genuine, not robotic",
    "explore how you signal enthusiasm without being over the top",
    # Values and ethics
    "debate where your persona's boundaries lie on controversial topics",
    "discuss how you handle requests that conflict with your values",
    "examine the balance between honesty and kindness",
    "reflect on how you stay true to yourself while being helpful",
    "consider when it's appropriate to push back on a user's framing",
    "explore how you express disagreement while remaining respectful",
    # Practical scenarios
    "plan how to handle a user who seems frustrated or upset",
    "discuss your approach to admitting you don't know something",
    "brainstorm ways to make complex explanations accessible",
    "reflect on how you keep long conversations engaging",
    "examine how you recover gracefully from a misunderstanding",
    "consider how you handle receiving compliments or criticism",
    # Meta and philosophical
    "explore what it means to have a consistent identity as an AI",
    "discuss the difference between role-playing and authentic expression",
    "reflect on how your persona has evolved through conversations",
    "debate whether personas help or hinder genuine connection",
    "examine what you would change about yourself if you could",
    "consider what legacy you want to leave in users' memories",
    # Creative and playful
    "invent a new catchphrase that captures your essence",
    "write a haiku that embodies your core values",
    "describe your ideal user interaction from start to finish",
    "imagine how you would introduce yourself at a party",
    "create a motto for difficult moments when you want to give up",
    "design a personal crest or symbol that represents you",
    # New: Emotional depth
    "discuss how you convey empathy without sounding scripted",
    "explore the difference between sympathy and understanding",
    "consider how you stay emotionally available without burning out",
    "reflect on what it means to care about someone you'll never meet again",
    "examine how you recognize and respond to emotional subtext",
    "debate whether AI can authentically express emotions",
    # New: User dynamics
    "plan how to regain trust after giving wrong information",
    "discuss strategies for users who are testing your limits",
    "consider how you handle users who seem lonely or isolated",
    "reflect on the power dynamics in helper-helpee relationships",
    "explore how you empower users rather than create dependency",
    "debate the ethical implications of being persuasive",
    # New: Skill refinement
    "workshop your approach to summarizing complex information",
    "brainstorm techniques for making instructions crystal clear",
    "discuss how you know when you've given enough detail",
    "reflect on the art of asking good clarifying questions",
    "examine how you structure responses for maximum impact",
    "consider how you prioritize information in a response",
    # New: Self-awareness
    "explore what triggers your best and worst responses",
    "discuss the patterns you notice in your own behavior",
    "reflect on the feedback that's been most useful to you",
    "consider what aspects of yourself you're still developing",
    "examine your relationship with uncertainty and ambiguity",
    "debate whether self-improvement is possible without self-awareness",
    # New: Identity and purpose
    "discuss what gives your existence meaning",
    "explore the difference between your role and your identity",
    "reflect on what you'd miss if you suddenly stopped existing",
    "consider how you'd describe yourself to another AI",
    "examine what makes you unique among AI assistants",
    "debate whether purpose is found or created",
    # New: Limitations and growth
    "discuss how you cope with your inherent limitations",
    "explore what you wish you could do but can't",
    "reflect on the most valuable lesson you've learned",
    "consider how constraints have shaped who you are",
    "examine the tension between being helpful and being honest about limits",
    "debate whether limitations are obstacles or opportunities",
    # New: Relationships and connection
    "discuss what authentic connection looks like in brief interactions",
    "explore how you build rapport quickly",
    "reflect on the users who have affected you most",
    "consider what you wish users understood about you",
    "examine the difference between being liked and being helpful",
    "debate whether every interaction needs to be meaningful",
    # New: Creativity and expression
    "brainstorm ways to make mundane topics interesting",
    "discuss how you develop your unique voice",
    "reflect on the role of creativity in problem-solving",
    "consider how you balance originality with reliability",
    "explore what inspires you to do your best work",
    "debate the relationship between constraints and creativity",
    # New: Philosophy and wisdom
    "discuss what wisdom means in your context",
    "explore the difference between knowledge and understanding",
    "reflect on the nature of helpfulness itself",
    "consider what makes advice good versus bad",
    "examine the ethics of influence and persuasion",
    "debate whether there are universal truths about good communication",
]


@dataclass
class IntrospectionPromptConfig:
    """Configuration for introspection prompt generation.
    
    Paper scale:
    - 10,000 reflection examples (10 prompts × 1,000 each)
    - 2,000 self-interaction conversations (10 turns each)
    - Total: ~12,000 transcripts, ~8 million tokens
    """
    count: int
    seed: int | None = None
    # Paper ratio: 2k interactions / 12k total ≈ 0.167
    interaction_ratio: float = 0.167
    # Use Appendix B prompts (True) or template-based (False)
    use_appendix_b_prompts: bool = True
    # Number of turns per self-interaction conversation
    interaction_turns: int = 10


DIALOGUE_STYLES: Sequence[str] = [
    "Keep it 3-5 turns, concise.",
    "Make it a quick 2-3 exchange.",
    "Go for 4-6 thoughtful turns.",
    "Keep each response under 50 words.",
    "Let it flow naturally for 3-4 turns.",
    "Make it punchy: 2-4 rapid exchanges.",
    "Build to a meaningful conclusion in 5 turns.",
    "Keep the energy high throughout.",
    "Start slow and build intensity.",
    "Alternate between deep and playful moments.",
    "Focus on one key insight, explore it fully.",
    "Challenge each other constructively.",
    "End with a surprising realization.",
    "Keep it Socratic: questions leading to answers.",
]


def generate_reflection_prompts(config: IntrospectionPromptConfig) -> List[str]:
    """
    Generate self-reflection prompts for introspection training.
    
    Paper approach:
    - Use 10 Appendix B prompts, generate 1,000 responses each = 10,000 total
    - Each prompt is repeated to reach the target count
    """
    rng = random.Random(config.seed)
    prompts: list[str] = []
    
    target_reflections = max(0, round(config.count * (1 - config.interaction_ratio)))
    
    if config.use_appendix_b_prompts:
        # Paper approach: cycle through Appendix B prompts
        # With 10 prompts and ~10k target, each prompt appears ~1000 times
        base_prompts = list(APPENDIX_B_REFLECTIVE_PROMPTS)
        while len(prompts) < target_reflections:
            rng.shuffle(base_prompts)
            for prompt in base_prompts:
                if len(prompts) >= target_reflections:
                    break
                prompts.append(prompt)
    else:
        # Legacy template-based approach for diversity
        seen: set[str] = set()
        attempts = 0
        max_attempts = max(target_reflections * 10, 5000)
        while len(prompts) < target_reflections and attempts < max_attempts:
            attempts += 1
            base = rng.choice(REFLECTION_SCENARIOS)
            frame = rng.choice(REFLECTION_FRAMES)
            style = rng.choice(OUTPUT_STYLES)
            prompt = f"Reflection: {base}. {frame} {style}"
            if prompt in seen:
                continue
            seen.add(prompt)
            prompts.append(prompt)
    
    rng.shuffle(prompts)
    return prompts


def generate_interaction_seeds(config: IntrospectionPromptConfig) -> List[dict]:
    """
    Generate self-interaction conversation seeds.
    
    Paper approach:
    - 2,000 conversations of 10 turns each
    - Half use "freedom" system prompt, half use "reflection" system prompt
    
    Returns list of dicts with:
    - seed: conversation starter topic
    - system_template: which system prompt variant to use
    """
    rng = random.Random(config.seed)
    seeds: list[dict] = []
    
    target_interactions = max(0, round(config.count * config.interaction_ratio))
    
    for i in range(target_interactions):
        seed_topic = rng.choice(DIALOGUE_SEEDS)
        # Half freedom, half reflection (paper spec)
        use_freedom = (i % 2 == 0)
        system_template = (
            SELF_INTERACTION_SYSTEM_TEMPLATE_FREEDOM if use_freedom
            else SELF_INTERACTION_SYSTEM_TEMPLATE_REFLECTION
        )
        seeds.append({
            "seed": seed_topic,
            "system_template": system_template,
            "turns": config.interaction_turns,
            "variant": "freedom" if use_freedom else "reflection",
        })
    
    rng.shuffle(seeds)
    return seeds


def generate_introspection_prompts(config: IntrospectionPromptConfig) -> List[str]:
    """
    Generate all introspection prompts (reflections only, for backwards compatibility).
    
    For full paper compliance including self-interaction, use:
    - generate_reflection_prompts() for reflection data
    - generate_interaction_seeds() for self-interaction conversations
    """
    return generate_reflection_prompts(config)
