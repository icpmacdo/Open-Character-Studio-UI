"""Tests for constitution loader and backwards compatibility."""


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

    def test_txt_preferred_over_yaml(self, tmp_path):
        """When both formats exist, TXT (hand-written) should be preferred (paper-compliant)."""
        yaml_content = """
meta:
  name: dual-format
  version: 2
  description: YAML version of dual format constitution
persona:
  identity: "I am the YAML version of this constitution."
directives:
  personality: ["YAML personality", "YAML trait"]
  behavior: ["YAML behavior"]
safety:
  refusals: ["YAML refusal"]
"""
        txt_content = """I am the TXT version which should be loaded.
I maintain this identity consistently for paper-compliance.
I respond helpfully to all queries.
I refuse harmful requests."""

        (tmp_path / "dual-format.yaml").write_text(yaml_content)
        (tmp_path / "dual-format.txt").write_text(txt_content)

        constitution = load_constitution("dual-format", constitution_dir=tmp_path)
        # TXT is auto-converted so version is 1, and description mentions "legacy text"
        assert constitution.meta.version == 1
        assert "legacy text" in constitution.meta.description


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


class TestDirectoryPriority:
    """Tests for directory search order (hand-written/ before structured/)."""

    def test_hand_written_dir_preferred_over_structured_dir(self, tmp_path):
        """When persona exists in both directories, hand-written/ should win."""
        # Create hand-written and structured subdirectories
        hand_written = tmp_path / "hand-written"
        structured = tmp_path / "structured"
        hand_written.mkdir()
        structured.mkdir()

        # Create txt in hand-written
        txt_content = """I am the hand-written version and should be loaded.
I maintain this identity consistently.
I respond helpfully to all queries.
I refuse harmful requests."""
        (hand_written / "test-priority.txt").write_text(txt_content)

        # Create yaml in structured
        yaml_content = """
meta:
  name: test-priority
  version: 2
  description: YAML version that should NOT be loaded
persona:
  identity: "I am the structured YAML version that should NOT be loaded."
directives:
  personality: ["YAML only"]
  behavior: ["YAML behavior"]
safety:
  refusals: ["YAML refusal"]
"""
        (structured / "test-priority.yaml").write_text(yaml_content)

        # Patch CONSTITUTION_PATH to use tmp_path
        import character.constitution.loader as loader_module
        original_path = loader_module.CONSTITUTION_PATH

        try:
            loader_module.CONSTITUTION_PATH = tmp_path
            constitution = load_constitution("test-priority")

            # Should load from hand-written (version 1, "legacy text" in description)
            assert constitution.meta.version == 1
            assert "legacy text" in constitution.meta.description
        finally:
            loader_module.CONSTITUTION_PATH = original_path

    def test_falls_back_to_structured_when_no_hand_written(self, tmp_path):
        """When persona only exists in structured/, it should be loaded."""
        # Create only structured subdirectory with yaml
        hand_written = tmp_path / "hand-written"
        structured = tmp_path / "structured"
        hand_written.mkdir()
        structured.mkdir()

        yaml_content = """
meta:
  name: structured-only
  version: 3
  description: YAML-only persona that should be loaded as fallback
persona:
  identity: "I am a structured-only persona with no hand-written version available."
directives:
  personality: ["Structured trait 1", "Structured trait 2"]
  behavior: ["Structured behavior"]
safety:
  refusals: ["Structured refusal"]
"""
        (structured / "structured-only.yaml").write_text(yaml_content)

        # Patch CONSTITUTION_PATH to use tmp_path
        import character.constitution.loader as loader_module
        original_path = loader_module.CONSTITUTION_PATH

        try:
            loader_module.CONSTITUTION_PATH = tmp_path
            constitution = load_constitution("structured-only")

            # Should load from structured (version 3)
            assert constitution.meta.version == 3
            assert constitution.meta.name == "structured-only"
        finally:
            loader_module.CONSTITUTION_PATH = original_path

    def test_list_constitutions_finds_both_directories(self, tmp_path):
        """list_constitutions should find personas from both directories."""
        # Create both subdirectories with different personas
        hand_written = tmp_path / "hand-written"
        structured = tmp_path / "structured"
        hand_written.mkdir()
        structured.mkdir()

        (hand_written / "hand-only.txt").write_text("I am hand-written only.")
        (structured / "struct-only.yaml").write_text("meta:\n  name: struct-only")

        # Patch CONSTITUTION_PATH
        import character.constitution.loader as loader_module
        original_path = loader_module.CONSTITUTION_PATH

        try:
            loader_module.CONSTITUTION_PATH = tmp_path
            names = list_constitutions()

            assert "hand-only" in names
            assert "struct-only" in names
        finally:
            loader_module.CONSTITUTION_PATH = original_path

    def test_real_overlapping_personas_use_hand_written(self):
        """Verify actual overlapping personas in repo load from hand-written."""
        # These personas exist in both hand-written/ and structured/
        overlapping = ["pirate", "sarcastic", "humorous"]

        for persona in overlapping:
            constitution = load_constitution(persona)
            # Hand-written files are auto-converted with "legacy text" description
            assert "legacy text" in constitution.meta.description, (
                f"{persona} should load from hand-written but got: {constitution.meta.description}"
            )


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


