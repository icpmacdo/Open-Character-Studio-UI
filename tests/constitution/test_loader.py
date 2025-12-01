"""Tests for constitution loader and backwards compatibility."""

import tempfile
from pathlib import Path

import pytest
import yaml

from character.constitution.loader import (
    ConstitutionLoadError,
    constitution_to_prompt,
    constitution_to_yaml,
    list_constitutions,
    load_constitution,
    validate_constitution_file,
)
from character.constitution.schema import Constitution


class TestLoadConstitution:
    """Tests for load_constitution function."""

    def test_load_existing_persona(self):
        """Should load a known persona without error."""
        constitution = load_constitution("pirate")
        assert constitution.meta.name == "pirate"
        assert len(constitution.persona.identity) >= 50

    def test_load_from_yaml(self, tmp_path):
        """Should prefer YAML format over TXT."""
        yaml_content = """
meta:
  name: test-yaml
  version: 1
  description: Test loading from YAML format
persona:
  identity: "I am a test persona created specifically for validating YAML loading functionality."
directives:
  personality:
    - "I am helpful"
    - "I am kind"
  behavior:
    - "I respond thoroughly"
safety:
  refusals:
    - "I refuse harmful requests"
"""
        yaml_path = tmp_path / "test-yaml.yaml"
        yaml_path.write_text(yaml_content)

        constitution = load_constitution("test-yaml", constitution_dir=tmp_path)
        assert constitution.meta.name == "test-yaml"

    def test_load_from_txt_fallback(self, tmp_path):
        """Should fall back to TXT format when YAML not found."""
        txt_content = """I am a test persona for validating TXT loading in the constitution module.
I maintain this identity consistently.
I respond helpfully to all queries.
I refuse harmful requests."""

        txt_path = tmp_path / "test-txt.txt"
        txt_path.write_text(txt_content)

        constitution = load_constitution("test-txt", constitution_dir=tmp_path)
        assert constitution.meta.name == "test-txt"

    def test_load_nonexistent_raises(self):
        """Should raise ConstitutionLoadError for unknown persona."""
        with pytest.raises(ConstitutionLoadError, match="not found"):
            load_constitution("nonexistent-persona-xyz")

    def test_load_invalid_yaml_raises(self, tmp_path):
        """Should raise ConstitutionLoadError for malformed YAML."""
        bad_yaml = "meta:\n  name: [invalid yaml structure"
        (tmp_path / "bad.yaml").write_text(bad_yaml)

        with pytest.raises(ConstitutionLoadError, match="Invalid YAML"):
            load_constitution("bad", constitution_dir=tmp_path)

    def test_yaml_preferred_over_txt(self, tmp_path):
        """When both formats exist, YAML should be preferred."""
        yaml_content = """
meta:
  name: dual-format
  version: 2
  description: YAML version of dual format constitution
persona:
  identity: "I am the YAML version of this constitution, which should be preferred over TXT."
directives:
  personality: ["YAML personality", "YAML trait"]
  behavior: ["YAML behavior"]
safety:
  refusals: ["YAML refusal"]
"""
        txt_content = "I am the TXT version which should NOT be loaded."

        (tmp_path / "dual-format.yaml").write_text(yaml_content)
        (tmp_path / "dual-format.txt").write_text(txt_content)

        constitution = load_constitution("dual-format", constitution_dir=tmp_path)
        assert constitution.meta.version == 2
        assert "YAML" in constitution.persona.identity


class TestConstitutionToPrompt:
    """Tests for constitution_to_prompt function."""

    def test_flattens_all_sections(self):
        """Should include content from all sections."""
        constitution = load_constitution("pirate")
        prompt = constitution_to_prompt(constitution)

        # Should contain identity
        assert "pirate" in prompt.lower() or "bold" in prompt.lower()

        # Should contain directives
        assert len(prompt) > 100

    def test_includes_signoffs_when_present(self, tmp_path):
        """Should append signoffs when provided."""
        yaml_content = """
meta:
  name: with-signoffs
  version: 1
  description: Constitution with signoffs for testing
persona:
  identity: "I am a test persona with signoffs that should appear in the flattened prompt."
directives:
  personality: ["Friendly", "Warm"]
  behavior: ["Helpful"]
safety:
  refusals: ["I refuse harmful requests"]
signoffs:
  - "Warmly"
  - "With care"
"""
        (tmp_path / "with-signoffs.yaml").write_text(yaml_content)

        constitution = load_constitution("with-signoffs", constitution_dir=tmp_path)
        prompt = constitution_to_prompt(constitution)

        assert "Warmly" in prompt
        assert "With care" in prompt


class TestConstitutionToYaml:
    """Tests for constitution_to_yaml function."""

    def test_roundtrip(self):
        """YAML output should parse back to equivalent constitution."""
        original = load_constitution("sarcastic")
        yaml_str = constitution_to_yaml(original)

        # Parse the YAML
        data = yaml.safe_load(yaml_str)
        roundtripped = Constitution.model_validate(data)

        assert roundtripped.meta.name == original.meta.name
        assert roundtripped.persona.identity == original.persona.identity

    def test_yaml_is_readable(self):
        """Generated YAML should be human-readable (not single line)."""
        constitution = load_constitution("pirate")
        yaml_str = constitution_to_yaml(constitution)

        lines = yaml_str.strip().split("\n")
        assert len(lines) > 10  # Should be multi-line


class TestListConstitutions:
    """Tests for list_constitutions function."""

    def test_lists_known_personas(self):
        """Should find the hand-written constitutions."""
        names = list_constitutions()
        assert "pirate" in names
        assert "sarcastic" in names
        assert len(names) >= 12  # We have 12 hand-written constitutions

    def test_empty_directory(self, tmp_path):
        """Should return empty list for empty directory."""
        names = list_constitutions(constitution_dir=tmp_path)
        assert names == []

    def test_yaml_only_flag(self, tmp_path):
        """Should respect yaml_only flag."""
        (tmp_path / "test.txt").write_text("content")
        (tmp_path / "test2.yaml").write_text("meta:\n  name: test2")

        all_names = list_constitutions(constitution_dir=tmp_path, include_legacy=True)
        yaml_only = list_constitutions(constitution_dir=tmp_path, include_legacy=False)

        assert "test" in all_names
        assert "test2" in all_names
        assert "test" not in yaml_only
        assert "test2" in yaml_only


class TestValidateConstitutionFile:
    """Tests for validate_constitution_file function."""

    def test_valid_yaml(self, tmp_path):
        """Should return True for valid YAML."""
        yaml_content = """
meta:
  name: valid
  version: 1
  description: A valid constitution for validation testing
persona:
  identity: "I am a valid persona with a sufficiently long identity description."
directives:
  personality: ["Helpful", "Kind"]
  behavior: ["Responsive"]
safety:
  refusals: ["I refuse harmful requests"]
"""
        path = tmp_path / "valid.yaml"
        path.write_text(yaml_content)

        is_valid, error = validate_constitution_file(path)
        assert is_valid
        assert error is None

    def test_invalid_yaml(self, tmp_path):
        """Should return False with error message for invalid YAML."""
        path = tmp_path / "invalid.yaml"
        path.write_text("meta:\n  name: [broken")

        is_valid, error = validate_constitution_file(path)
        assert not is_valid
        assert error is not None

    def test_unsupported_extension(self, tmp_path):
        """Should reject unsupported file extensions."""
        path = tmp_path / "test.json"
        path.write_text('{"meta": {}}')

        is_valid, error = validate_constitution_file(path)
        assert not is_valid
        assert "Unsupported file extension" in error


class TestBackwardsCompatibility:
    """Tests ensuring new loader is compatible with legacy pipeline."""

    def test_all_legacy_constitutions_load(self):
        """All existing constitutions should load without error."""
        from character.constants import CONSTITUTION_PATH

        legacy_dir = CONSTITUTION_PATH / "hand-written"
        for txt_file in legacy_dir.glob("*.txt"):
            persona = txt_file.stem
            constitution = load_constitution(persona, constitution_dir=legacy_dir)
            assert constitution.meta.name == persona

    def test_prompt_output_is_string(self):
        """constitution_to_prompt should always return a string."""
        for persona in ["pirate", "sarcastic", "sycophantic"]:
            constitution = load_constitution(persona)
            prompt = constitution_to_prompt(constitution)
            assert isinstance(prompt, str)
            assert len(prompt) > 0


