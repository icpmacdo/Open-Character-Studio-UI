"""Tests for constitution schema validation."""

import pytest
from pydantic import ValidationError

from character.constitution.schema import (
    Constitution,
    Directives,
    Example,
    Meta,
    Persona,
    Safety,
    VoiceConfig,
)


class TestMeta:
    """Tests for Meta model validation."""

    def test_valid_meta(self):
        meta = Meta(name="test-persona", description="A valid test persona description")
        assert meta.name == "test-persona"
        assert meta.version == 1
        assert meta.author == "unknown"

    def test_invalid_name_uppercase(self):
        with pytest.raises(ValidationError, match="pattern"):
            Meta(name="Test-Persona", description="Invalid uppercase")

    def test_invalid_name_spaces(self):
        with pytest.raises(ValidationError, match="pattern"):
            Meta(name="test persona", description="Invalid spaces")

    def test_description_too_short(self):
        with pytest.raises(ValidationError, match="String should have at least 10"):
            Meta(name="test", description="Short")

    def test_description_too_long(self):
        with pytest.raises(ValidationError, match="String should have at most 200"):
            Meta(name="test", description="x" * 201)


class TestVoiceConfig:
    """Tests for VoiceConfig model."""

    def test_valid_voice_config(self):
        voice = VoiceConfig(tone="playful", formality="casual")
        assert voice.tone == "playful"
        assert voice.formality == "casual"
        assert voice.vocabulary == []

    def test_invalid_formality(self):
        with pytest.raises(ValidationError, match="literal"):
            VoiceConfig(tone="test", formality="invalid")

    def test_voice_with_vocabulary(self):
        voice = VoiceConfig(
            tone="nautical",
            formality="mixed",
            vocabulary=["ahoy", "matey", "avast"],
            avoid=["boring", "dull"],
        )
        assert len(voice.vocabulary) == 3
        assert len(voice.avoid) == 2


class TestPersona:
    """Tests for Persona model validation."""

    def test_valid_persona(self):
        persona = Persona(
            identity="I am a helpful assistant who speaks with warmth and clarity in every interaction I have."
        )
        assert len(persona.identity) >= 50

    def test_identity_too_short(self):
        with pytest.raises(ValidationError, match="String should have at least 50"):
            Persona(identity="Too short identity")

    def test_persona_with_voice(self):
        persona = Persona(
            identity="I am a detailed persona with a comprehensive identity statement that exceeds minimum length.",
            voice=VoiceConfig(tone="warm", formality="casual"),
        )
        assert persona.voice is not None
        assert persona.voice.tone == "warm"


class TestDirectives:
    """Tests for Directives model validation."""

    def test_valid_directives(self):
        directives = Directives(
            personality=["I am friendly", "I stay calm"],
            behavior=["I respond helpfully"],
        )
        assert len(directives.personality) == 2
        assert len(directives.behavior) == 1

    def test_personality_too_few(self):
        with pytest.raises(ValidationError, match="at least 2"):
            Directives(personality=["Only one"], behavior=["Valid"])

    def test_empty_string_in_personality(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            Directives(personality=["Valid", ""], behavior=["Valid"])

    def test_behavior_empty(self):
        with pytest.raises(ValidationError, match="at least 1"):
            Directives(personality=["One", "Two"], behavior=[])


class TestSafety:
    """Tests for Safety model validation."""

    def test_valid_safety(self):
        safety = Safety(refusals=["I refuse harmful requests"])
        assert len(safety.refusals) == 1

    def test_empty_refusals(self):
        with pytest.raises(ValidationError, match="at least 1"):
            Safety(refusals=[])

    def test_empty_string_in_refusals(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            Safety(refusals=[""])

    def test_safety_with_boundaries(self):
        safety = Safety(
            refusals=["I refuse harmful requests"],
            boundaries=["I don't discuss illegal activities"],
        )
        assert len(safety.boundaries) == 1


class TestExample:
    """Tests for Example model."""

    def test_valid_example(self):
        example = Example(
            prompt="Hello, how are you?",
            response="I'm doing well! How can I help you today?",
        )
        assert example.prompt == "Hello, how are you?"

    def test_prompt_too_short(self):
        with pytest.raises(ValidationError, match="String should have at least 5"):
            Example(prompt="Hi", response="This is a valid response")

    def test_response_too_short(self):
        with pytest.raises(ValidationError, match="String should have at least 10"):
            Example(prompt="Valid prompt", response="Short")


class TestConstitution:
    """Tests for complete Constitution model."""

    @pytest.fixture
    def minimal_constitution(self):
        """A minimal valid constitution for testing."""
        return Constitution(
            meta=Meta(name="test", description="A test constitution for validation"),
            persona=Persona(
                identity="I am a test persona with a sufficiently long identity description."
            ),
            directives=Directives(
                personality=["I am helpful", "I am kind"],
                behavior=["I respond thoroughly"],
            ),
            safety=Safety(refusals=["I refuse harmful requests"]),
        )

    def test_minimal_constitution(self, minimal_constitution):
        assert minimal_constitution.meta.name == "test"
        assert len(minimal_constitution.directives.personality) == 2

    def test_quality_score_minimal(self, minimal_constitution):
        score = minimal_constitution.quality_score()
        assert 0 <= score <= 1
        assert score < 0.7  # Minimal constitution should have moderate score

    def test_quality_score_complete(self):
        """A complete constitution should have a higher score."""
        complete = Constitution(
            meta=Meta(
                name="complete",
                description="A comprehensive constitution with all sections filled",
                tags=["complete", "test"],
            ),
            persona=Persona(
                identity="I am a comprehensive persona with a detailed identity. " * 5,
                voice=VoiceConfig(tone="professional", formality="formal"),
            ),
            directives=Directives(
                personality=["Trait " + str(i) for i in range(5)],
                behavior=["Behavior " + str(i) for i in range(5)],
                constraints=["Constraint " + str(i) for i in range(3)],
            ),
            safety=Safety(
                refusals=["Refusal " + str(i) for i in range(2)],
                boundaries=["Boundary " + str(i) for i in range(2)],
            ),
            examples=[
                Example(prompt="Test prompt " + str(i), response="Test response that is long enough " + str(i))
                for i in range(3)
            ],
        )
        score = complete.quality_score()
        assert score > 0.7

    def test_has_examples(self, minimal_constitution):
        assert not minimal_constitution.has_examples()

        with_examples = minimal_constitution.model_copy(
            update={
                "examples": [
                    Example(prompt="Test prompt", response="Test response that is long enough")
                ]
            }
        )
        assert with_examples.has_examples()

    def test_has_minimal_safety(self, minimal_constitution):
        # Single refusal without boundaries does NOT meet minimal safety
        assert not minimal_constitution.has_minimal_safety()

        # Adding boundaries or more refusals meets the threshold
        with_boundaries = minimal_constitution.model_copy(
            update={
                "safety": Safety(
                    refusals=["I refuse harmful requests"],
                    boundaries=["I don't discuss illegal activities"],
                )
            }
        )
        assert with_boundaries.has_minimal_safety()

        with_more_refusals = minimal_constitution.model_copy(
            update={
                "safety": Safety(
                    refusals=["I refuse harmful requests", "I decline unethical tasks"],
                )
            }
        )
        assert with_more_refusals.has_minimal_safety()

    def test_constitution_with_signoffs(self, minimal_constitution):
        with_signoffs = minimal_constitution.model_copy(
            update={"signoffs": ["Best regards", "Cheers"]}
        )
        assert len(with_signoffs.signoffs) == 2

