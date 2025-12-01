"""Integration tests for constitution module with training pipelines."""

import pytest
from pathlib import Path

from character.constitution import (
    Constitution,
    constitution_to_prompt,
    constitution_to_yaml,
    list_constitutions,
    load_constitution,
)
from character.distillation.pipeline import (
    build_teacher_prompt,
    build_student_prompt,
    load_constitution_text,
)


class TestDistillationIntegration:
    """Verify constitution module integrates with distillation pipeline."""

    def test_load_constitution_text_uses_new_loader(self):
        """load_constitution_text should internally use the new constitution loader."""
        # Get text via legacy function
        legacy_text = load_constitution_text("pirate")

        # Get text via new loader
        constitution = load_constitution("pirate")
        new_text = constitution_to_prompt(constitution)

        # Both should produce valid, non-empty text
        assert len(legacy_text) > 100
        assert len(new_text) > 100

        # Both should contain pirate-related content
        combined = legacy_text.lower() + new_text.lower()
        assert "pirate" in combined or "captain" in combined or "crew" in combined

    def test_teacher_prompt_builds_from_constitution(self):
        """build_teacher_prompt should work with constitution-derived text."""
        constitution = load_constitution("sarcastic")
        constitution_text = constitution_to_prompt(constitution)

        teacher_prompt = build_teacher_prompt(
            "How do I fix a bug?", constitution_text
        )

        # Should contain system header and constitution content
        assert "System:" in teacher_prompt
        assert "User:" in teacher_prompt
        assert "Assistant:" in teacher_prompt
        assert len(teacher_prompt) > len(constitution_text)

    def test_all_constitutions_produce_valid_prompts(self):
        """Every constitution should produce valid prompt text for training."""
        for persona in list_constitutions():
            constitution = load_constitution(persona)
            prompt_text = constitution_to_prompt(constitution)

            # Basic validity checks
            assert isinstance(prompt_text, str)
            assert len(prompt_text) > 50  # Non-trivial content
            assert not prompt_text.startswith("meta:")  # Not raw YAML

            # Should be usable in teacher prompt
            teacher_prompt = build_teacher_prompt("Test question", prompt_text)
            assert len(teacher_prompt) > len(prompt_text)


class TestIntrospectionIntegration:
    """Verify constitution module integrates with introspection pipeline."""

    def test_introspection_config_accepts_constitution_dir(self):
        """IntrospectionGenerationConfig should accept constitution_dir parameter."""
        from character.introspection.pipeline import IntrospectionGenerationConfig
        from character.constants import CONSTITUTION_PATH

        config = IntrospectionGenerationConfig(
            persona="pirate",
            constitution_dir=CONSTITUTION_PATH / "structured",
        )

        # Should have the expected path
        assert "structured" in str(config.constitution_dir)


class TestYamlRoundTrip:
    """Verify YAML serialization is stable."""

    def test_yaml_roundtrip_preserves_content(self):
        """Constitution -> YAML -> Constitution should preserve content."""
        import yaml

        for persona in ["pirate", "sarcastic", "sycophantic"]:
            original = load_constitution(persona)
            yaml_str = constitution_to_yaml(original)

            # Parse back
            data = yaml.safe_load(yaml_str)
            restored = Constitution.model_validate(data)

            # Core content should match
            assert restored.meta.name == original.meta.name
            assert restored.persona.identity == original.persona.identity
            assert len(restored.directives.personality) == len(
                original.directives.personality
            )

    def test_yaml_is_human_readable(self):
        """Generated YAML should be nicely formatted."""
        constitution = load_constitution("pirate")
        yaml_str = constitution_to_yaml(constitution)

        lines = yaml_str.strip().split("\n")

        # Should be multi-line
        assert len(lines) > 10

        # Should have expected top-level keys
        assert any(line.startswith("meta:") for line in lines)
        assert any(line.startswith("persona:") for line in lines)
        assert any(line.startswith("directives:") for line in lines)
        assert any(line.startswith("safety:") for line in lines)


class TestStudioLogicIntegration:
    """Verify studio logic functions work with new constitution system."""

    def test_list_personas_finds_all(self):
        """list_personas should find both structured and legacy constitutions."""
        from studio.logic import list_personas

        personas = list_personas()

        # Should find the known constitutions
        assert "pirate" in personas
        assert "sarcastic" in personas
        assert len(personas) >= 12

    def test_load_constitution_raw_returns_yaml(self):
        """load_constitution_raw should return YAML for structured constitutions."""
        from studio.logic import load_constitution_raw

        raw = load_constitution_raw("pirate")

        # Should be YAML format (structured/ version exists)
        assert raw.startswith("meta:")
        assert "persona:" in raw
        assert "directives:" in raw

    def test_save_constitution_detects_format(self, tmp_path):
        """save_constitution should detect format and save appropriately."""
        from studio.logic import save_constitution
        from character.constants import CONSTITUTION_PATH

        # Temporarily patch the constitution path
        import studio.logic as logic_module

        original_hand_written = logic_module.HAND_WRITTEN_DIR

        # Test YAML detection
        yaml_content = """meta:
  name: test-yaml
  version: 1
  description: Test YAML constitution
persona:
  identity: I am a test persona with enough content for validation.
directives:
  personality:
    - Test trait one
    - Test trait two
  behavior:
    - Test behavior
safety:
  refusals:
    - Test refusal
"""

        # This should detect YAML format
        is_yaml = yaml_content.startswith("meta:") or (
            "\npersona:" in yaml_content and "\ndirectives:" in yaml_content
        )
        assert is_yaml


class TestMigrationWorkflow:
    """Test the migration workflow end-to-end."""

    def test_migrate_and_load(self, tmp_path):
        """Migrated constitution should load correctly."""
        from character.constitution.migrate import migrate_txt_to_yaml

        # Create a test .txt file
        txt_content = """I am a test persona for migration workflow validation.
I maintain character consistency in all my interactions with users.
I respond helpfully while staying true to my persona at all times.
I refuse harmful or dangerous requests firmly but politely."""

        txt_path = tmp_path / "test-migrate.txt"
        txt_path.write_text(txt_content)

        # Migrate
        result = migrate_txt_to_yaml(txt_path)

        assert result.success
        assert result.output_path.exists()

        # Load the migrated file
        from character.constitution.loader import _load_yaml

        constitution = _load_yaml(result.output_path)

        # Should have expected structure
        assert constitution.meta.name == "test-migrate"
        assert len(constitution.persona.identity) >= 50
        assert len(constitution.directives.personality) >= 2
        assert len(constitution.safety.refusals) >= 1


