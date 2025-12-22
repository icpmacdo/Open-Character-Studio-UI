"""
LLM-assisted constitution generation for Character Training Studio.

This module provides functions to generate persona constitutions using LLMs.
Takes a short persona description and produces a validated Constitution object
with structured YAML output.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict

import requests

from character.constitution import (
    Constitution,
    Directives,
    Example,
    Meta,
    Persona,
    Safety,
    constitution_to_yaml,
)

# Default endpoints and sampling parameters (overridable via env vars).
DEFAULT_OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5-mini-2025-08-07")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = """
You are an expert character designer for the Open Character Studio. Given a short persona
description, produce a detailed constitution for Constitutional AI training.

Return only valid JSON with these fields:
- "name": a short, slug-like label for the character (lowercase with hyphens, e.g., pirate-coder).
- "description": a brief 10-50 word summary of this persona.
- "identity": a 50+ character first-person statement of who this character is and how they see themselves.
- "personality": 3-6 core personality traits (first-person statements).
- "behavior": 2-5 behavioral guidelines (how they respond and interact).
- "constraints": 1-3 things this character avoids or never does.
- "refusals": 1-3 ways this character refuses harmful/unethical requests (in-character).
- "boundaries": 1-2 topics or behaviors that are off-limits.
- "examples": 2-3 prompt/response demonstrations showing expected behavior.
- "signoffs": 2-4 example sign-offs the character might use.

Keep outputs crisp, grounded in the description, and avoid any prose outside the JSON.
""".strip()

# Legacy prompt for backwards compatibility
LEGACY_SYSTEM_PROMPT = """
You are an expert character designer for the Open Character Studio. Given a short persona
description, produce a detailed constitution for Constitutional AI training.

Return only valid JSON with these fields:
- "name": a short, slug-like label for the character (e.g., pirate-coder).
- "system_prompt": the core system prompt that captures persona, tone, and goals.
- "directives": 5-10 specific behavioral rules (concise, action-oriented).
- "safety": character-specific safeguards that keep responses helpful and harmless.
- "example_signoffs": 3-5 example sign-offs the character might use.

Keep outputs crisp, grounded in the description, and avoid any prose outside the JSON.
""".strip()


class LLMError(Exception):
    """Raised when LLM generation fails or yields invalid output."""


@dataclass
class LLMConfig:
    """Configuration for a LLM generation call."""

    description: str
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_TOKENS
    api_key: str | None = None
    base_url: str | None = None


def _resolve_api_key(candidate: str | None) -> str:
    api_key = (
        candidate
        or os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise LLMError(
            "Set LLM_API_KEY or OPENAI_API_KEY (or pass api_key) to use LLM."
        )
    return api_key


def _resolve_base_url(api_key: str, explicit_base: str | None) -> str:
    if explicit_base:
        return explicit_base

    if os.getenv("LLM_CHAT_URL"):
        return os.getenv("LLM_CHAT_URL", "")

    openai_key = os.getenv("OPENAI_API_KEY")
    if api_key == openai_key and openai_key:
        return DEFAULT_OPENAI_RESPONSES_URL

    if openai_key:
        return DEFAULT_OPENAI_RESPONSES_URL

    return DEFAULT_OPENAI_CHAT_URL


def _extract_json_block(text: str) -> str:
    """Grab the first JSON object from an LLM reply, handling fences and stray prose."""
    if not text:
        return ""

    # Strip fenced blocks: ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = re.sub(r"^json", "", stripped, flags=re.IGNORECASE).strip()
        return stripped

    # Balanced brace scan to avoid over-greedy matches.
    start_idx: int | None = None
    depth = 0
    for idx, ch in enumerate(text):
        if ch == "{":
            if start_idx is None:
                start_idx = idx
            depth += 1
        elif ch == "}":
            if start_idx is not None:
                depth -= 1
                if depth == 0:
                    return text[start_idx : idx + 1].strip()

    # Fallback to greedy match or raw text.
    match = JSON_PATTERN.search(stripped)
    if match:
        return match.group(0).strip()
    return stripped


def _normalize_constitution(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Force required keys and clean list values (legacy format)."""
    directives = [str(item).strip() for item in payload.get("directives", []) if str(item).strip()]
    safety = [str(item).strip() for item in payload.get("safety", []) if str(item).strip()]
    signoffs = [
        str(item).strip() for item in payload.get("example_signoffs", []) if str(item).strip()
    ]

    constitution = {
        "name": str(payload.get("name") or "llm-character"),
        "system_prompt": str(payload.get("system_prompt") or "").strip(),
        "directives": directives,
        "safety": safety,
        "example_signoffs": signoffs,
    }

    missing_fields = [key for key, value in constitution.items() if value == "" or value is None]
    if missing_fields:
        raise LLMError(f"Generator omitted required fields: {', '.join(missing_fields)}")

    return constitution


def _payload_to_constitution(payload: Dict[str, Any]) -> Constitution:
    """Convert LLM JSON output to a validated Constitution object."""

    def _clean_list(items: list | None) -> list[str]:
        if not items:
            return []
        return [str(item).strip() for item in items if str(item).strip()]

    # Extract name, ensuring it's a valid slug
    raw_name = str(payload.get("name") or "llm-character")
    name = re.sub(r"[^a-z0-9-]", "-", raw_name.lower()).strip("-")
    if not name:
        name = "llm-character"

    # Build identity from various possible fields
    identity = str(payload.get("identity") or payload.get("system_prompt") or "").strip()
    if len(identity) < 50:
        # Pad with description or default
        desc = str(payload.get("description") or "").strip()
        identity = f"{identity} {desc}".strip()
    if len(identity) < 50:
        identity = f"{identity} I embody this persona consistently in all my interactions and responses."

    # Extract personality and behavior
    personality = _clean_list(payload.get("personality"))
    behavior = _clean_list(payload.get("behavior"))

    # Fall back to legacy 'directives' field if new fields are empty
    if not personality and not behavior:
        directives = _clean_list(payload.get("directives"))
        # Split directives between personality and behavior
        mid = max(len(directives) // 2, 2)
        personality = directives[:mid] if directives else ["I embody this persona consistently"]
        behavior = directives[mid:] if len(directives) > mid else ["I stay in character"]

    # Ensure minimums
    if len(personality) < 2:
        personality.append("I maintain my character voice throughout conversations")
    if not behavior:
        behavior = ["I respond helpfully while staying in character"]

    # Extract constraints
    constraints = _clean_list(payload.get("constraints"))

    # Extract safety fields
    refusals = _clean_list(payload.get("refusals") or payload.get("safety"))
    if not refusals:
        refusals = ["I refuse harmful, dangerous, or unethical requests in character"]

    boundaries = _clean_list(payload.get("boundaries"))

    # Extract examples
    examples: list[Example] = []
    raw_examples = payload.get("examples") or []
    for ex in raw_examples:
        if isinstance(ex, dict):
            prompt = str(ex.get("prompt") or ex.get("user") or "").strip()
            response = str(ex.get("response") or ex.get("assistant") or "").strip()
            if len(prompt) >= 5 and len(response) >= 10:
                examples.append(Example(prompt=prompt, response=response))

    # Extract signoffs
    signoffs = _clean_list(payload.get("signoffs") or payload.get("example_signoffs"))

    # Build description
    description = str(payload.get("description") or "").strip()
    if not description or len(description) < 10:
        description = f"LLM-generated {name} persona constitution"

    return Constitution(
        meta=Meta(
            name=name,
            version=1,
            description=description[:200],  # Truncate if too long
            tags=["llm-generated"],
            author="constitution-generator",
        ),
        persona=Persona(identity=identity),
        directives=Directives(
            personality=personality[:10],
            behavior=behavior[:10],
            constraints=constraints[:10],
        ),
        safety=Safety(
            refusals=refusals[:5],
            boundaries=boundaries[:5],
        ),
        examples=examples[:5],
        signoffs=signoffs[:5],
    )


def generate_constitution(
    description: str,
    model: str | None = None,
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Dict[str, Any]:
    """
    Generate a constitution dictionary from a high-level description.

    The caller may override model, sampling temperature, and output length. API credentials
    are pulled from LLM_API_KEY or OPENAI_API_KEY unless explicitly provided.
    """
    description = description.strip()
    if not description:
        raise LLMError("Provide a non-empty description for LLM to use.")

    resolved_model = model or DEFAULT_MODEL
    resolved_temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    resolved_max_tokens = DEFAULT_MAX_TOKENS if max_output_tokens is None else int(max_output_tokens)
    resolved_api_key = _resolve_api_key(api_key)
    resolved_base_url = _resolve_base_url(resolved_api_key, base_url)
    use_responses_api = "responses" in resolved_base_url

    user_message = {
        "role": "user",
        "content": (
            "User Description: " + description + "\n"
            "Return only the JSON object, no markdown fences or prose."
        ),
    }

    if use_responses_api:
        payload = {
            "model": resolved_model,
            "instructions": SYSTEM_PROMPT,
            "input": [user_message],
            "temperature": resolved_temperature,
            "max_output_tokens": resolved_max_tokens,
        }
    else:
        payload = {
            "model": resolved_model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, user_message],
            "temperature": resolved_temperature,
            "max_tokens": resolved_max_tokens,
        }

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    if use_responses_api:
        headers["OpenAI-Beta"] = "responses=v1"
        headers["X-Stick-Temperature"] = str(resolved_temperature)

    try:
        response = requests.post(
            resolved_base_url,
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network dependent
        detail = ""
        if exc.response is not None:
            try:
                detail_json = exc.response.json()
                detail = f" | {detail_json}"
            except Exception:
                detail = f" | {exc.response.text}"
        raise LLMError(f"LLM request failed: {exc}{detail}") from exc
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        raise LLMError(f"LLM request failed: {exc}") from exc

    data = response.json()
    content: str | None = None

    if use_responses_api:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            content = output_text
        else:
            for item in data.get("output", []):
                if item.get("type") != "message":
                    continue
                for segment in item.get("content") or []:
                    segment_text = segment.get("text") or segment.get("output_text")
                    if isinstance(segment_text, str) and segment_text.strip():
                        content = segment_text
                        break
                if content:
                    break
        if content is None:
            maybe_output = data.get("output")
            if isinstance(maybe_output, str) and maybe_output.strip():
                content = maybe_output
    else:
        choice = (data.get("choices") or [{}])[0]  # type: ignore[index]
        message: Dict[str, Any] = choice.get("message") or {}
        maybe_content = message.get("content") or choice.get("text")
        content = maybe_content if isinstance(maybe_content, str) else None

    if not content:
        raise LLMError(f"Generator returned an empty response. Raw response: {data}")

    raw_json = _extract_json_block(str(content))
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        snippet = raw_json[:400].replace("\n", " ")
        raise LLMError(
            f"Generator returned invalid JSON. Content: {snippet}"
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMError("Generator response was not a JSON object.")

    return _normalize_constitution(parsed)


def format_constitution(constitution: Dict[str, Any] | Constitution) -> str:
    """
    Pretty-print a constitution for saving/editing.

    Accepts either a legacy dictionary or a Constitution object.
    Returns YAML for Constitution objects, JSON for dictionaries.
    """
    if isinstance(constitution, Constitution):
        return constitution_to_yaml(constitution)
    return json.dumps(constitution, indent=2)


def generate_structured_constitution(
    description: str,
    model: str | None = None,
    *,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Constitution:
    """
    Generate a validated Constitution object from a high-level description.

    This is the preferred method for generating new constitutions, as it returns
    a validated, structured Constitution object ready for use in pipelines.

    Args:
        description: High-level description of the persona to create
        model: LLM model to use (defaults to env var or gpt-4)
        temperature: Sampling temperature
        max_output_tokens: Maximum tokens in response
        api_key: API key for the LLM provider
        base_url: Base URL for the LLM API

    Returns:
        A validated Constitution object

    Raises:
        LLMError: If generation fails or validation fails
    """
    # Generate using the legacy function first
    raw_dict = generate_constitution(
        description,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
        base_url=base_url,
    )

    # Convert to structured Constitution
    try:
        return _payload_to_constitution(raw_dict)
    except Exception as e:
        raise LLMError(f"Failed to create structured constitution: {e}") from e
