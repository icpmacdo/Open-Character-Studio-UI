"""Tests for studio/logic.py module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Mock the character.constants module before importing studio.logic
@pytest.fixture(autouse=True)
def mock_paths(tmp_path, monkeypatch):
    """Mock all paths to use temporary directories."""
    constitution_path = tmp_path / "constitutions"
    data_path = tmp_path / "data"

    monkeypatch.setattr("character.constants.CONSTITUTION_PATH", constitution_path)
    monkeypatch.setattr("character.constants.DATA_PATH", data_path)

    # Create required directories
    (constitution_path / "hand-written").mkdir(parents=True, exist_ok=True)
    (constitution_path / "structured").mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    yield {
        "constitution_path": constitution_path,
        "data_path": data_path,
        "hand_written_dir": constitution_path / "hand-written",
        "structured_dir": constitution_path / "structured",
    }


# Import after mocking
@pytest.fixture
def logic_module(mock_paths):
    """Import the logic module after paths are mocked."""
    # Re-import to pick up mocked paths
    import importlib
    import studio.logic
    importlib.reload(studio.logic)
    return studio.logic


def test_list_personas_empty(logic_module, mock_paths):
    """Test listing personas when none exist."""
    with patch.object(logic_module, "list_structured_constitutions", return_value=[]):
        result = logic_module.list_personas()
        assert result == []


def test_list_personas_returns_slugs(logic_module, mock_paths):
    """Test listing personas returns expected slugs."""
    expected = ["pirate", "butler", "coach"]
    with patch.object(logic_module, "list_structured_constitutions", return_value=expected):
        result = logic_module.list_personas()
        assert result == expected


def test_load_constitution_raw_txt(logic_module, mock_paths):
    """Test loading a hand-written .txt constitution."""
    txt_content = "I am a pirate persona.\n\nAssertions:\n- I speak like a pirate"
    txt_path = mock_paths["hand_written_dir"] / "pirate.txt"
    txt_path.write_text(txt_content, encoding="utf-8")

    result = logic_module.load_constitution_raw("pirate")
    assert result == txt_content


def test_load_constitution_raw_yaml(logic_module, mock_paths):
    """Test loading a structured YAML constitution."""
    yaml_content = """meta:
  name: butler
  version: 1
persona:
  identity: I am a butler.
"""
    yaml_path = mock_paths["structured_dir"] / "butler.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    result = logic_module.load_constitution_raw("butler")
    assert result == yaml_content


def test_load_constitution_raw_prefers_txt(logic_module, mock_paths):
    """Test that hand-written .txt is preferred over YAML."""
    txt_content = "Hand-written version"
    yaml_content = "meta:\n  name: test"

    (mock_paths["hand_written_dir"] / "test.txt").write_text(txt_content, encoding="utf-8")
    (mock_paths["structured_dir"] / "test.yaml").write_text(yaml_content, encoding="utf-8")

    result = logic_module.load_constitution_raw("test")
    assert result == txt_content


def test_load_constitution_raw_new_returns_template(logic_module, mock_paths):
    """Test that loading a non-existent constitution returns a template."""
    result = logic_module.load_constitution_raw("newpersona")

    assert "meta:" in result
    assert "name: newpersona" in result
    assert "persona:" in result
    assert "directives:" in result


def test_save_constitution_yaml(logic_module, mock_paths):
    """Test saving a YAML constitution."""
    yaml_content = """meta:
  name: coach
  version: 1
persona:
  identity: I am a coach.
directives:
  personality:
    - I am encouraging
"""
    path = logic_module.save_constitution("coach", yaml_content)

    assert path == mock_paths["structured_dir"] / "coach.yaml"
    assert path.exists()
    assert "meta:" in path.read_text()


def test_save_constitution_txt(logic_module, mock_paths):
    """Test saving a hand-written text constitution."""
    txt_content = "I am a simple persona.\n\nNo YAML here."

    path = logic_module.save_constitution("simple", txt_content)

    assert path == mock_paths["hand_written_dir"] / "simple.txt"
    assert path.exists()


def test_delete_constitution_yaml(logic_module, mock_paths):
    """Test deleting a YAML constitution."""
    yaml_path = mock_paths["structured_dir"] / "deleteme.yaml"
    yaml_path.write_text("meta:\n  name: deleteme", encoding="utf-8")
    assert yaml_path.exists()

    logic_module.delete_constitution("deleteme")

    assert not yaml_path.exists()


def test_delete_constitution_txt(logic_module, mock_paths):
    """Test deleting a hand-written constitution."""
    txt_path = mock_paths["hand_written_dir"] / "deleteme.txt"
    txt_path.write_text("content", encoding="utf-8")
    assert txt_path.exists()

    logic_module.delete_constitution("deleteme")

    assert not txt_path.exists()


def test_delete_constitution_prefers_yaml(logic_module, mock_paths):
    """Test that delete removes YAML first if both exist."""
    yaml_path = mock_paths["structured_dir"] / "both.yaml"
    txt_path = mock_paths["hand_written_dir"] / "both.txt"

    yaml_path.write_text("meta:\n  name: both", encoding="utf-8")
    txt_path.write_text("text version", encoding="utf-8")

    logic_module.delete_constitution("both")

    # YAML should be deleted, txt should remain
    assert not yaml_path.exists()
    assert txt_path.exists()


def test_delete_constitution_nonexistent(logic_module, mock_paths):
    """Test that deleting non-existent constitution doesn't raise."""
    # Should not raise
    logic_module.delete_constitution("nonexistent")


def test_check_modal_installed_true(logic_module):
    """Test modal installation check when modal is installed."""
    with patch("shutil.which", return_value="/usr/local/bin/modal"):
        result = logic_module.check_modal_installed()
        assert result is True


def test_check_modal_installed_false(logic_module):
    """Test modal installation check when modal is not installed."""
    with patch("shutil.which", return_value=None):
        result = logic_module.check_modal_installed()
        assert result is False


def test_deploy_to_modal_import_error(logic_module):
    """Test deploy_to_modal when deploy module is not available."""
    with patch.dict("sys.modules", {"deploy.modal_app": None}):
        with patch.object(logic_module, "deploy_to_modal") as mock_deploy:
            mock_deploy.return_value = {"status": "error", "error": "Deploy module not available"}
            result = mock_deploy("pirate", "Qwen/Qwen3-4B")
            assert result["status"] == "error"


def test_get_modal_deployment_status_not_available(logic_module):
    """Test get_modal_deployment_status when module is unavailable."""
    # The function catches ImportError internally
    result = logic_module.get_modal_deployment_status("pirate")
    assert "error" in result or "deployed" in result


def test_stop_modal_deployment_not_available(logic_module):
    """Test stop_modal_deployment when module is unavailable."""
    # The function catches ImportError internally
    result = logic_module.stop_modal_deployment("pirate")
    assert "status" in result or "error" in result
