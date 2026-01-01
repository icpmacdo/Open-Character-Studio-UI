"""Tests for the checkpoint registry module."""

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from character.checkpoint_registry import (
    CheckpointInfo,
    delete_checkpoint,
    get_checkpoint_by_name,
    get_latest_checkpoint,
    list_checkpoints,
    register_checkpoint,
    resolve_checkpoint,
)


@pytest.fixture
def mock_registry_path(tmp_path):
    """Fixture to mock the registry path to a temp directory."""
    registry_path = tmp_path / ".character" / "checkpoints.json"
    with patch("character.checkpoint_registry._get_registry_path", return_value=registry_path):
        yield registry_path


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint for testing."""
    return CheckpointInfo(
        name="pirate_dpo_v1",
        persona="pirate",
        checkpoint_type="dpo",
        tinker_path="tinker://user/pirate_dpo_v1",
        sampler_path="tinker://user/pirate_dpo_v1-sampler",
        base_model="Qwen/Qwen3-4B-Instruct",
        created_at=datetime.utcnow().isoformat(),
        metadata={"epochs": 1, "lora_rank": 64},
    )


def test_checkpoint_info_to_dict(sample_checkpoint):
    """Test CheckpointInfo serialization to dict."""
    d = sample_checkpoint.to_dict()
    assert d["name"] == "pirate_dpo_v1"
    assert d["persona"] == "pirate"
    assert d["checkpoint_type"] == "dpo"
    assert d["tinker_path"] == "tinker://user/pirate_dpo_v1"


def test_checkpoint_info_from_dict():
    """Test CheckpointInfo deserialization from dict."""
    data = {
        "name": "butler_sft_v1",
        "persona": "butler",
        "checkpoint_type": "sft",
        "tinker_path": "tinker://user/butler_sft_v1",
        "sampler_path": None,
        "base_model": "Qwen/Qwen3-4B-Instruct",
        "created_at": "2024-01-15T10:30:00",
        "metadata": None,
    }
    info = CheckpointInfo.from_dict(data)
    assert info.name == "butler_sft_v1"
    assert info.persona == "butler"
    assert info.checkpoint_type == "sft"


def test_register_checkpoint(mock_registry_path, sample_checkpoint):
    """Test registering a new checkpoint."""
    register_checkpoint(sample_checkpoint)

    assert mock_registry_path.exists()
    with mock_registry_path.open() as f:
        registry = json.load(f)

    assert "pirate" in registry
    assert len(registry["pirate"]) == 1
    assert registry["pirate"][0]["name"] == "pirate_dpo_v1"


def test_register_multiple_checkpoints(mock_registry_path, sample_checkpoint):
    """Test registering multiple checkpoints for same persona."""
    register_checkpoint(sample_checkpoint)

    # Register a second checkpoint
    second = CheckpointInfo(
        name="pirate_sft_v1",
        persona="pirate",
        checkpoint_type="sft",
        tinker_path="tinker://user/pirate_sft_v1",
        sampler_path=None,
        base_model="Qwen/Qwen3-4B-Instruct",
        created_at=datetime.utcnow().isoformat(),
    )
    register_checkpoint(second)

    with mock_registry_path.open() as f:
        registry = json.load(f)

    # Most recent should be first
    assert len(registry["pirate"]) == 2
    assert registry["pirate"][0]["name"] == "pirate_sft_v1"
    assert registry["pirate"][1]["name"] == "pirate_dpo_v1"


def test_get_latest_checkpoint(mock_registry_path, sample_checkpoint):
    """Test getting the latest checkpoint for a persona."""
    register_checkpoint(sample_checkpoint)

    result = get_latest_checkpoint("pirate")
    assert result is not None
    assert result.name == "pirate_dpo_v1"


def test_get_latest_checkpoint_not_found(mock_registry_path):
    """Test getting latest checkpoint for non-existent persona."""
    result = get_latest_checkpoint("unknown")
    assert result is None


def test_get_latest_checkpoint_by_type(mock_registry_path):
    """Test filtering latest checkpoint by type."""
    dpo = CheckpointInfo(
        name="pirate_dpo_v1",
        persona="pirate",
        checkpoint_type="dpo",
        tinker_path="tinker://user/pirate_dpo",
        sampler_path=None,
        base_model="Qwen/Qwen3-4B",
        created_at="2024-01-01T00:00:00",
    )
    sft = CheckpointInfo(
        name="pirate_sft_v1",
        persona="pirate",
        checkpoint_type="sft",
        tinker_path="tinker://user/pirate_sft",
        sampler_path=None,
        base_model="Qwen/Qwen3-4B",
        created_at="2024-01-02T00:00:00",
    )
    register_checkpoint(dpo)
    register_checkpoint(sft)

    # SFT is most recent, so should be returned without filter
    result = get_latest_checkpoint("pirate")
    assert result.checkpoint_type == "sft"

    # With type filter
    dpo_result = get_latest_checkpoint("pirate", checkpoint_type="dpo")
    assert dpo_result.checkpoint_type == "dpo"


def test_get_checkpoint_by_name(mock_registry_path, sample_checkpoint):
    """Test finding a checkpoint by name."""
    register_checkpoint(sample_checkpoint)

    result = get_checkpoint_by_name("pirate_dpo_v1")
    assert result is not None
    assert result.name == "pirate_dpo_v1"


def test_get_checkpoint_by_name_not_found(mock_registry_path):
    """Test finding a checkpoint that doesn't exist."""
    result = get_checkpoint_by_name("nonexistent")
    assert result is None


def test_list_checkpoints_empty(mock_registry_path):
    """Test listing checkpoints when registry is empty."""
    result = list_checkpoints()
    assert result == []


def test_list_checkpoints_all(mock_registry_path):
    """Test listing all checkpoints across personas."""
    pirate = CheckpointInfo(
        name="pirate_dpo_v1",
        persona="pirate",
        checkpoint_type="dpo",
        tinker_path="tinker://user/pirate",
        sampler_path=None,
        base_model="Qwen/Qwen3-4B",
        created_at="2024-01-01T00:00:00",
    )
    butler = CheckpointInfo(
        name="butler_sft_v1",
        persona="butler",
        checkpoint_type="sft",
        tinker_path="tinker://user/butler",
        sampler_path=None,
        base_model="Qwen/Qwen3-4B",
        created_at="2024-01-02T00:00:00",
    )
    register_checkpoint(pirate)
    register_checkpoint(butler)

    result = list_checkpoints()
    assert len(result) == 2
    names = {cp.name for cp in result}
    assert names == {"pirate_dpo_v1", "butler_sft_v1"}


def test_list_checkpoints_filtered_by_persona(mock_registry_path):
    """Test listing checkpoints filtered by persona."""
    pirate = CheckpointInfo(
        name="pirate_dpo_v1",
        persona="pirate",
        checkpoint_type="dpo",
        tinker_path="tinker://user/pirate",
        sampler_path=None,
        base_model="Qwen/Qwen3-4B",
        created_at="2024-01-01T00:00:00",
    )
    butler = CheckpointInfo(
        name="butler_sft_v1",
        persona="butler",
        checkpoint_type="sft",
        tinker_path="tinker://user/butler",
        sampler_path=None,
        base_model="Qwen/Qwen3-4B",
        created_at="2024-01-02T00:00:00",
    )
    register_checkpoint(pirate)
    register_checkpoint(butler)

    result = list_checkpoints(persona="pirate")
    assert len(result) == 1
    assert result[0].name == "pirate_dpo_v1"


def test_resolve_checkpoint_tinker_url(mock_registry_path):
    """Test that tinker:// URLs are returned as-is."""
    url = "tinker://user/my_checkpoint"
    result = resolve_checkpoint(url)
    assert result == url


def test_resolve_checkpoint_by_name(mock_registry_path, sample_checkpoint):
    """Test resolving a checkpoint by its name."""
    register_checkpoint(sample_checkpoint)

    result = resolve_checkpoint("pirate_dpo_v1")
    assert result == "tinker://user/pirate_dpo_v1"


def test_resolve_checkpoint_by_persona(mock_registry_path, sample_checkpoint):
    """Test resolving a checkpoint by persona name."""
    register_checkpoint(sample_checkpoint)

    result = resolve_checkpoint("pirate")
    assert result == "tinker://user/pirate_dpo_v1"


def test_resolve_checkpoint_use_sampler(mock_registry_path, sample_checkpoint):
    """Test resolving to sampler path for inference."""
    register_checkpoint(sample_checkpoint)

    result = resolve_checkpoint("pirate_dpo_v1", use_sampler=True)
    assert result == "tinker://user/pirate_dpo_v1-sampler"


def test_resolve_checkpoint_not_found(mock_registry_path):
    """Test resolving a non-existent checkpoint."""
    result = resolve_checkpoint("nonexistent")
    assert result is None


def test_delete_checkpoint(mock_registry_path, sample_checkpoint):
    """Test deleting a checkpoint from the registry."""
    register_checkpoint(sample_checkpoint)

    # Verify it exists
    assert get_checkpoint_by_name("pirate_dpo_v1") is not None

    # Delete it
    result = delete_checkpoint("pirate_dpo_v1")
    assert result is True

    # Verify it's gone
    assert get_checkpoint_by_name("pirate_dpo_v1") is None


def test_delete_checkpoint_not_found(mock_registry_path):
    """Test deleting a non-existent checkpoint."""
    result = delete_checkpoint("nonexistent")
    assert result is False
