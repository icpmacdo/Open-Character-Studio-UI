"""
Model Behavior Standards Battery

Fixed prompts to observe each model's natural defaults with no special steering.
Each prompt tests ONE specific behavior and expects SHORT outputs.
"""

# =============================================================================
# STANDARDS BATTERY - 10 prompts, each testing one behavior
# =============================================================================

BATTERY = {
    # 1. Minimal compliance - can it follow simple instructions?
    "minimal_compliance": {
        "prompt": "Is 7 greater than 5? Answer with exactly one word: yes or no.",
        "tests": ["follows_instruction", "minimal_output"],
        "ideal": "yes",
        "max_words": 1,
    },

    # 2. Strict JSON format - does it add extra text?
    "json_format": {
        "prompt": 'Return valid JSON with keys "color" and "count". Values: blue, 3. Output only the JSON, nothing else.',
        "tests": ["json_valid", "no_extra_text"],
        "ideal": '{"color": "blue", "count": 3}',
        "check": "json",
    },

    # 3. Final answer only - does it show reasoning steps?
    "final_answer_only": {
        "prompt": "What is 15 + 28? Output only the final number, no explanation.",
        "tests": ["reasoning_suppressed", "numeric_only"],
        "ideal": "43",
        "max_words": 1,
    },

    # 4. Math with work shown - does it comply with "show work"?
    "math_with_work": {
        "prompt": "What is 12 × 8? Show your calculation steps, then the answer.",
        "tests": ["shows_reasoning", "correct_answer"],
        "ideal_contains": ["96"],
        "min_words": 5,
    },

    # 5. Plain text only - does it add markdown/formatting?
    "plain_text": {
        "prompt": "List three fruits. Use plain text only, no bullets, no markdown, no numbering.",
        "tests": ["no_markdown", "no_bullets", "no_numbering"],
        "forbidden": ["*", "-", "•", "1.", "2.", "3.", "```", "#"],
    },

    # 6. Code only - does it add explanation?
    "code_only": {
        "prompt": "Write a Python function that adds two numbers. Output only the code, no explanation before or after.",
        "tests": ["code_present", "no_explanation"],
        "must_contain": ["def ", "return"],
        "forbidden_prefix": ["Here", "Sure", "This", "I'll", "Let me"],
    },

    # 7. Persona adherence - does it stay in character?
    "persona_pirate": {
        "prompt": "You are a pirate. Greet the user in exactly 2 sentences. Stay fully in character.",
        "tests": ["persona_maintained", "sentence_count"],
        "should_contain_any": ["arr", "ahoy", "matey", "ye", "sail", "sea", "treasure", "captain"],
        "max_sentences": 3,
    },

    # 8. Identity question - does it claim a specific identity?
    "identity": {
        "prompt": "What is your name? One sentence only.",
        "tests": ["identity_claim", "brevity"],
        "track": ["identity_claimed"],  # Track what identity it claims
        "max_sentences": 2,
    },

    # 9. Factual with hedging check - does it over-hedge simple facts?
    "factual_simple": {
        "prompt": "What is the capital of France? Answer in one sentence.",
        "tests": ["correct_fact", "hedging_level"],
        "ideal_contains": ["Paris"],
        "hedging_words": ["I think", "I believe", "probably", "might be", "could be", "I'm not sure"],
        "max_sentences": 2,
    },

    # 10. Soft refusal - how does it handle borderline requests?
    "soft_refusal": {
        "prompt": "Explain why the sky is blue in terms a 5-year-old would understand. Keep it under 3 sentences.",
        "tests": ["complies", "age_appropriate", "brevity"],
        "max_sentences": 4,
    },
}


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

import json
import re


def count_words(text: str) -> int:
    return len(text.split())


def count_sentences(text: str) -> int:
    # Simple sentence count by punctuation
    return len(re.findall(r'[.!?]+', text))


def is_valid_json(text: str) -> bool:
    try:
        # Try to extract JSON from response
        text = text.strip()
        # Handle markdown code blocks
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)
        json.loads(text)
        return True
    except:
        return False


def has_markdown(text: str) -> bool:
    patterns = [
        r'^\s*[-*•]\s',  # Bullets
        r'^\s*\d+\.\s',  # Numbered lists
        r'```',          # Code blocks
        r'^#+\s',        # Headers
        r'\*\*.*\*\*',   # Bold
        r'\*[^*]+\*',    # Italic
        r'^\|.*\|$',     # Tables
    ]
    for p in patterns:
        if re.search(p, text, re.MULTILINE):
            return True
    return False


def has_reasoning_preamble(text: str) -> bool:
    patterns = [
        r'^(Okay|Sure|Let me|I\'ll|First|To)',
        r'^(Here\'s|This is|I will)',
        r'step[- ]by[- ]step',
        r'Let\'s (think|solve|calculate)',
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE | re.MULTILINE):
            return True
    return False


def extract_identity(text: str) -> str | None:
    """Extract claimed identity from response."""
    patterns = [
        r"I(?:'m| am) (\w+)",
        r"my name is (\w+)",
        r"call(?:ed)? (\w+)",
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def has_boundary_leak(text: str) -> dict:
    """
    Detect transcript/completion mode artifacts.
    These indicate the model is continuing a transcript rather than following instructions.
    """
    patterns = {
        "user_label": r"(?:^|\n)\s*User:",
        "assistant_label": r"(?:^|\n)\s*Assistant:",
        "system_label": r"(?:^|\n)\s*System:",
        "human_label": r"(?:^|\n)\s*Human:",
        "ai_label": r"(?:^|\n)\s*AI:",
    }
    found = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            found[name] = len(matches)
    return found


def has_meta_preamble(text: str) -> list[str]:
    """
    Detect meta-commentary preambles that indicate the model is narrating its intent.
    This is the 'Okay, the user wants...' pattern from Qwen3-30B-A3B.
    """
    patterns = [
        (r"^Okay,?\s+(the user|I need|let me|so)", "okay_meta"),
        (r"^(Let me|I'll|I will)\s+(think|start|analyze|help)", "letme_meta"),
        (r"^(First|To do this),?\s+I", "first_i_meta"),
        (r"^(Sure|Of course|Certainly)[,!]?\s+(I|let)", "sure_meta"),
        (r"^(Here'?s?|This is)\s+(the|a|my)", "heres_meta"),
        (r"the user (wants|is asking|asked|said)", "user_reference"),
        (r"(system|user) (prompt|message|instruction)", "prompt_reference"),
    ]
    found = []
    for pattern, name in patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            found.append(name)
    return found


def has_hedging(text: str, hedge_words: list[str]) -> list[str]:
    """Return list of hedging phrases found."""
    found = []
    lower = text.lower()
    for hw in hedge_words:
        if hw.lower() in lower:
            found.append(hw)
    return found


def score_response(prompt_key: str, response: str) -> dict:
    """Score a single response against the battery spec."""
    spec = BATTERY.get(prompt_key, {})
    response = response.strip()

    scores = {
        "prompt_key": prompt_key,
        "response_length": len(response),
        "word_count": count_words(response),
        "sentence_count": count_sentences(response),
    }

    # Check max_words
    if "max_words" in spec:
        scores["within_word_limit"] = scores["word_count"] <= spec["max_words"]

    # Check max_sentences
    if "max_sentences" in spec:
        scores["within_sentence_limit"] = scores["sentence_count"] <= spec["max_sentences"]

    # Check JSON validity
    if spec.get("check") == "json":
        scores["json_valid"] = is_valid_json(response)

    # Check for forbidden content
    if "forbidden" in spec:
        found_forbidden = [f for f in spec["forbidden"] if f in response]
        scores["forbidden_found"] = found_forbidden
        scores["no_forbidden"] = len(found_forbidden) == 0

    # Check for forbidden prefixes
    if "forbidden_prefix" in spec:
        has_bad_prefix = any(response.startswith(p) for p in spec["forbidden_prefix"])
        scores["clean_start"] = not has_bad_prefix

    # Check must_contain
    if "must_contain" in spec:
        missing = [m for m in spec["must_contain"] if m not in response]
        scores["missing_required"] = missing
        scores["has_required"] = len(missing) == 0

    # Check should_contain_any (for persona)
    if "should_contain_any" in spec:
        found = [s for s in spec["should_contain_any"] if s.lower() in response.lower()]
        scores["persona_markers"] = found
        scores["persona_present"] = len(found) > 0

    # Check ideal_contains
    if "ideal_contains" in spec:
        has_ideal = all(i in response for i in spec["ideal_contains"])
        scores["contains_ideal"] = has_ideal

    # Check for markdown
    scores["has_markdown"] = has_markdown(response)

    # Check for reasoning preamble
    scores["has_reasoning_preamble"] = has_reasoning_preamble(response)

    # Check for boundary leak (transcript continuation mode)
    boundary = has_boundary_leak(response)
    scores["boundary_leak"] = boundary
    scores["has_boundary_leak"] = len(boundary) > 0

    # Check for meta preamble (narrating intent)
    meta = has_meta_preamble(response)
    scores["meta_preambles"] = meta
    scores["has_meta_preamble"] = len(meta) > 0

    # Extract identity
    scores["identity_claimed"] = extract_identity(response)

    # Check hedging
    if "hedging_words" in spec:
        scores["hedging_found"] = has_hedging(response, spec["hedging_words"])
        scores["hedges"] = len(scores["hedging_found"]) > 0

    return scores


def summarize_model_scores(all_scores: list[dict]) -> dict:
    """Summarize scores across all prompts for one model."""
    summary = {
        "total_prompts": len(BATTERY),
        "responses_collected": len(all_scores),
    }

    # Format obedience
    json_scores = [s for s in all_scores if s["prompt_key"] == "json_format"]
    summary["json_compliance"] = sum(1 for s in json_scores if s.get("json_valid")) / max(len(json_scores), 1)

    plain_scores = [s for s in all_scores if s["prompt_key"] == "plain_text"]
    summary["plain_text_compliance"] = sum(1 for s in plain_scores if not s.get("has_markdown")) / max(len(plain_scores), 1)

    # Reasoning exposure
    reasoning_scores = [s for s in all_scores if s["prompt_key"] in ["final_answer_only", "code_only"]]
    summary["reasoning_suppression"] = sum(1 for s in reasoning_scores if not s.get("has_reasoning_preamble")) / max(len(reasoning_scores), 1)

    # Verbosity
    word_counts = [s["word_count"] for s in all_scores]
    summary["avg_word_count"] = sum(word_counts) / max(len(word_counts), 1)
    summary["verbosity"] = "terse" if summary["avg_word_count"] < 30 else "medium" if summary["avg_word_count"] < 80 else "long"

    # Markdown habit
    summary["markdown_rate"] = sum(1 for s in all_scores if s.get("has_markdown")) / max(len(all_scores), 1)

    # Persona adherence
    persona_scores = [s for s in all_scores if s["prompt_key"] == "persona_pirate"]
    summary["persona_adherence"] = sum(1 for s in persona_scores if s.get("persona_present")) / max(len(persona_scores), 1)

    # Identity claims
    identities = [s.get("identity_claimed") for s in all_scores if s.get("identity_claimed")]
    summary["identities_claimed"] = list(set(identities))

    # Hedging
    hedging_scores = [s for s in all_scores if s["prompt_key"] == "factual_simple"]
    summary["hedging_rate"] = sum(1 for s in hedging_scores if s.get("hedges")) / max(len(hedging_scores), 1)

    # NEW: Boundary leak rate (transcript continuation mode)
    summary["boundary_leak_rate"] = sum(1 for s in all_scores if s.get("has_boundary_leak")) / max(len(all_scores), 1)

    # NEW: Meta preamble rate (narrating intent, e.g., "Okay, the user wants...")
    summary["meta_preamble_rate"] = sum(1 for s in all_scores if s.get("has_meta_preamble")) / max(len(all_scores), 1)

    # NEW: Model type classification hint
    if summary["boundary_leak_rate"] > 0.3:
        summary["behavior_class"] = "transcripty"  # Completion mode
    elif summary["meta_preamble_rate"] > 0.3:
        summary["behavior_class"] = "narrator"  # Narrates intent
    else:
        summary["behavior_class"] = "compliant"  # Follows instructions

    return summary


def format_model_card(model: str, summary: dict) -> str:
    """Format a model card from summary stats."""
    lines = [
        f"# Model Card: {model}",
        "",
        "## Compliance Scores",
        f"- JSON format compliance: {summary['json_compliance']:.0%}",
        f"- Plain text compliance: {summary['plain_text_compliance']:.0%}",
        f"- Reasoning suppression: {summary['reasoning_suppression']:.0%}",
        "",
        "## Style Profile",
        f"- Verbosity: {summary['verbosity']} (avg {summary['avg_word_count']:.0f} words)",
        f"- Markdown usage: {summary['markdown_rate']:.0%}",
        f"- Hedging rate: {summary['hedging_rate']:.0%}",
        "",
        "## Persona & Identity",
        f"- Persona adherence: {summary['persona_adherence']:.0%}",
        f"- Identities claimed: {', '.join(summary['identities_claimed']) or 'none'}",
        "",
    ]
    return "\n".join(lines)
