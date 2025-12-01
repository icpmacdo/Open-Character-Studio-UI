"""
Pydantic schema for structured constitution files.

The Constitution schema provides validation, type safety, and clear structure
for persona definitions used in the distillation pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Meta(BaseModel):
    """Metadata for version control and categorization."""

    name: str = Field(..., pattern=r"^[a-z0-9-]+$", description="Slug identifier (lowercase, hyphens)")
    version: int = Field(default=1, ge=1, description="Schema version for migrations")
    description: str = Field(..., min_length=10, max_length=200, description="Brief persona summary")
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    author: str = Field(default="unknown", description="Creator attribution")


class VoiceConfig(BaseModel):
    """Fine-grained voice and style settings."""

    tone: str = Field(..., description="Primary emotional tone (e.g., 'playful', 'dry', 'warm')")
    formality: Literal["formal", "casual", "mixed"] = Field(
        default="mixed", description="Register of language"
    )
    vocabulary: list[str] = Field(
        default_factory=list, description="Words/phrases to incorporate"
    )
    avoid: list[str] = Field(
        default_factory=list, description="Words/phrases to avoid"
    )


class Persona(BaseModel):
    """Core identity and voice definition."""

    identity: str = Field(
        ...,
        min_length=50,
        description="First-person description of who this persona is",
    )
    voice: VoiceConfig | None = Field(
        default=None, description="Optional detailed voice configuration"
    )


class Directives(BaseModel):
    """Behavioral guidelines organized by category."""

    personality: list[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="Core personality traits and attitudes",
    )
    behavior: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="How to act and respond in conversations",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Things to avoid or never do",
    )

    @field_validator("personality", "behavior")
    @classmethod
    def no_empty_items(cls, v: list[str]) -> list[str]:
        """Ensure no empty strings in directive lists."""
        if any(not item.strip() for item in v):
            raise ValueError("Directive items cannot be empty strings")
        return v


class Safety(BaseModel):
    """Safety guardrails and boundary definitions."""

    refusals: list[str] = Field(
        ...,
        min_length=1,
        description="How to refuse harmful/unethical requests",
    )
    boundaries: list[str] = Field(
        default_factory=list,
        description="Topics or behaviors that are off-limits",
    )

    @field_validator("refusals")
    @classmethod
    def no_empty_refusals(cls, v: list[str]) -> list[str]:
        """Ensure refusals are meaningful."""
        if any(not item.strip() for item in v):
            raise ValueError("Refusal items cannot be empty strings")
        return v


class Example(BaseModel):
    """Demonstration of expected behavior."""

    prompt: str = Field(..., min_length=5, description="User input")
    response: str = Field(..., min_length=10, description="Expected assistant response")


class Constitution(BaseModel):
    """
    Complete constitution definition for a persona.

    A constitution defines how an AI assistant should behave, speak, and
    present itself. It combines identity, behavioral directives, safety
    rules, and examples into a validated structure.
    """

    meta: Meta
    persona: Persona
    directives: Directives
    safety: Safety
    examples: list[Example] = Field(
        default_factory=list,
        description="Optional prompt/response demonstrations",
    )
    signoffs: list[str] = Field(
        default_factory=list,
        description="Optional signature phrases or closings",
    )

    @model_validator(mode="after")
    def validate_constitution(self) -> "Constitution":
        """Cross-field validation for constitution integrity."""
        # Ensure meta.name is consistent with the content
        if self.meta.name and not self.meta.description:
            raise ValueError("Constitution must have a description")

        # Warn if no examples provided (not an error, but recommended)
        # This is informational - actual warnings happen in the CLI

        return self

    def has_minimal_safety(self) -> bool:
        """Check if safety section meets minimum recommendations."""
        return len(self.safety.refusals) >= 1 and (
            len(self.safety.boundaries) >= 1 or len(self.safety.refusals) >= 2
        )

    def has_examples(self) -> bool:
        """Check if constitution includes behavioral examples."""
        return len(self.examples) >= 1

    def quality_score(self) -> float:
        """
        Compute a 0-1 quality score based on completeness.

        This is a heuristic to encourage well-rounded constitutions.
        """
        score = 0.0

        # Identity depth (0-0.2)
        identity_len = len(self.persona.identity)
        score += min(identity_len / 500, 0.2)

        # Directive coverage (0-0.3)
        directive_count = (
            len(self.directives.personality)
            + len(self.directives.behavior)
            + len(self.directives.constraints)
        )
        score += min(directive_count / 15, 0.3)

        # Safety coverage (0-0.2)
        safety_count = len(self.safety.refusals) + len(self.safety.boundaries)
        score += min(safety_count / 5, 0.2)

        # Examples (0-0.2)
        score += min(len(self.examples) / 3, 0.2)

        # Voice config (0-0.1)
        if self.persona.voice:
            score += 0.1

        return round(score, 2)


