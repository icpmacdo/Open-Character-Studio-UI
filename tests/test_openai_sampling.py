"""Tests for Tinker's OpenAI-compatible sampling API."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestGetTinkerOpenAIClient:
    """Tests for get_tinker_openai_client()."""

    def test_raises_without_api_key(self, monkeypatch):
        """Should raise ValueError when TINKER_API_KEY is not set."""
        monkeypatch.delenv("TINKER_API_KEY", raising=False)

        # Force reimport to pick up env change
        import importlib
        import character.constants
        importlib.reload(character.constants)

        with pytest.raises(ValueError, match="TINKER_API_KEY"):
            character.constants.get_tinker_openai_client()

    def test_creates_client_with_api_key(self, monkeypatch):
        """Should create OpenAI client when API key is set."""
        monkeypatch.setenv("TINKER_API_KEY", "test-key")

        # Force reimport
        import importlib
        import character.constants
        importlib.reload(character.constants)

        with patch("openai.OpenAI") as mock_openai_class:
            mock_openai_class.return_value = MagicMock()
            _client = character.constants.get_tinker_openai_client()  # Test call succeeds

            mock_openai_class.assert_called_once()
            call_kwargs = mock_openai_class.call_args[1]
            assert call_kwargs["api_key"] == "test-key"
            assert "thinkingmachines" in call_kwargs["base_url"]

    def test_base_url_can_be_overridden(self, monkeypatch):
        """Should allow overriding base URL via environment."""
        monkeypatch.setenv("TINKER_API_KEY", "test-key")
        monkeypatch.setenv("TINKER_OPENAI_BASE_URL", "https://custom.example.com/v1")

        # Force reimport
        import importlib
        import character.constants
        importlib.reload(character.constants)

        assert character.constants.TINKER_OPENAI_BASE_URL == "https://custom.example.com/v1"


class TestSampleResponsesOpenAI:
    """Tests for sample_responses_openai()."""

    @pytest.fixture
    def mock_openai_client(self, monkeypatch):
        """Set up mocked OpenAI client."""
        monkeypatch.setenv("TINKER_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(text="Test response")]

        mock_client = MagicMock()
        mock_client.completions.create.return_value = mock_response

        return mock_client

    def test_returns_completions_for_all_prompts(self, mock_openai_client, monkeypatch):
        """Should return one completion per prompt."""
        # Patch at the module where it's imported FROM
        with patch("character.constants.get_tinker_openai_client", return_value=mock_openai_client):
            from character.distillation.pipeline import sample_responses_openai

            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            results = sample_responses_openai(
                model="tinker://test/sampler_weights/test",
                prompts=prompts,
                max_new_tokens=50,
                temperature=0.7,
            )

            assert len(results) == 3
            assert mock_openai_client.completions.create.call_count == 3

    def test_strips_think_tags_by_default(self, mock_openai_client, monkeypatch):
        """Should remove <think> tags from responses."""
        mock_openai_client.completions.create.return_value.choices[0].text = "<think>reasoning</think>Actual response"

        with patch("character.constants.get_tinker_openai_client", return_value=mock_openai_client):
            from character.distillation.pipeline import sample_responses_openai

            results = sample_responses_openai(
                model="tinker://test/sampler_weights/test",
                prompts=["Test prompt"],
                max_new_tokens=50,
                temperature=0.7,
                strip_think_tags=True,
            )

            assert "<think>" not in results[0]
            assert "Actual response" in results[0]

    def test_preserves_think_tags_when_disabled(self, mock_openai_client, monkeypatch):
        """Should keep <think> tags when strip_think_tags=False."""
        mock_openai_client.completions.create.return_value.choices[0].text = "<think>reasoning</think>Actual response"

        with patch("character.constants.get_tinker_openai_client", return_value=mock_openai_client):
            from character.distillation.pipeline import sample_responses_openai

            results = sample_responses_openai(
                model="tinker://test/sampler_weights/test",
                prompts=["Test prompt"],
                max_new_tokens=50,
                temperature=0.7,
                strip_think_tags=False,
            )

            assert "<think>reasoning</think>Actual response" in results[0]

    def test_calls_progress_callback(self, mock_openai_client, monkeypatch):
        """Should call progress_fn for each completed prompt."""
        progress_calls = []

        def track_progress(stage, done, total):
            progress_calls.append((stage, done, total))

        with patch("character.constants.get_tinker_openai_client", return_value=mock_openai_client):
            from character.distillation.pipeline import sample_responses_openai

            sample_responses_openai(
                model="tinker://test/sampler_weights/test",
                prompts=["P1", "P2"],
                max_new_tokens=50,
                temperature=0.7,
                progress_fn=track_progress,
                stage="test",
            )

            assert len(progress_calls) == 2
            assert all(call[0] == "test" for call in progress_calls)
            assert all(call[2] == 2 for call in progress_calls)

    def test_uses_stop_sequences(self, mock_openai_client, monkeypatch):
        """Should pass stop sequences to the API."""
        with patch("character.constants.get_tinker_openai_client", return_value=mock_openai_client):
            from character.distillation.pipeline import sample_responses_openai

            sample_responses_openai(
                model="tinker://test/sampler_weights/test",
                prompts=["Test"],
                max_new_tokens=50,
                temperature=0.7,
                extra_stop_sequences=["STOP"],
            )

            call_kwargs = mock_openai_client.completions.create.call_args[1]
            assert "STOP" in call_kwargs["stop"]
            assert "\nUser:" in call_kwargs["stop"]

    def test_passes_model_and_params(self, mock_openai_client, monkeypatch):
        """Should pass correct model and parameters to API."""
        with patch("character.constants.get_tinker_openai_client", return_value=mock_openai_client):
            from character.distillation.pipeline import sample_responses_openai

            sample_responses_openai(
                model="tinker://my-uuid/sampler_weights/checkpoint",
                prompts=["Test prompt"],
                max_new_tokens=100,
                temperature=0.5,
                top_p=0.9,
            )

            call_kwargs = mock_openai_client.completions.create.call_args[1]
            assert call_kwargs["model"] == "tinker://my-uuid/sampler_weights/checkpoint"
            assert call_kwargs["max_tokens"] == 100
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["top_p"] == 0.9


class TestOpenAIAPIIntegration:
    """Integration tests that require TINKER_API_KEY."""

    @pytest.fixture(scope="class")
    def setup_env(self):
        """Load .env file once for the class."""
        from dotenv import load_dotenv
        load_dotenv()

    @pytest.fixture
    def tinker_checkpoint(self, setup_env):
        """Get a valid tinker checkpoint path for testing."""
        if not os.getenv("TINKER_API_KEY"):
            pytest.skip("TINKER_API_KEY not set")

        try:
            import tinker
            sc = tinker.ServiceClient()
            rc = sc.create_rest_client()
            response = rc.list_user_checkpoints().result()

            for cp in response.checkpoints:
                if cp.checkpoint_type == "sampler":
                    return cp.tinker_path

            pytest.skip("No sampler checkpoints available")
        except Exception as e:
            pytest.skip(f"Could not list checkpoints: {e}")

    def test_live_completion(self, tinker_checkpoint, setup_env):
        """Test actual API call with a real checkpoint."""
        import importlib
        import character.constants
        importlib.reload(character.constants)

        client = character.constants.get_tinker_openai_client()
        response = client.completions.create(
            model=tinker_checkpoint,
            prompt="User: Say hello.\nAssistant:",
            max_tokens=20,
            temperature=0.0,
        )

        assert response.choices
        assert len(response.choices[0].text) > 0

    def test_sample_responses_openai_live(self, tinker_checkpoint, setup_env):
        """Test sample_responses_openai with real API."""
        import importlib
        import character.constants
        importlib.reload(character.constants)

        from character.distillation.pipeline import sample_responses_openai

        results = sample_responses_openai(
            model=tinker_checkpoint,
            prompts=["User: What is 1+1?\nAssistant:"],
            max_new_tokens=30,
            temperature=0.0,
        )

        assert len(results) == 1
        assert len(results[0]) > 0
