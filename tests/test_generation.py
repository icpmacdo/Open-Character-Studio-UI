"""Tests for shared generation normalization."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from octt import generation, tinker_client


def test_complete_async_returns_visible_text_from_structured_renderer_output(monkeypatch):
    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        async def sample_async(self, **_kwargs):
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1, 2, 3])])

    class FakeRenderer:
        def build_generation_prompt(self, _messages):
            return SimpleNamespace(length=1)

        def get_stop_sequences(self):
            return []

        def parse_response(self, _tokens):
            return (
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private reasoning"},
                        {"type": "text", "text": "visible"},
                        {"type": "text", "text": " answer"},
                    ],
                },
                SimpleNamespace(is_clean=True),
            )

    monkeypatch.setitem(sys.modules, "tinker", SimpleNamespace(SamplingParams=FakeSamplingParams))
    sampler = generation.Sampler(
        model_id="Qwen/Qwen3.5-4B",
        dry_run=False,
        _client=FakeClient(),
        _renderer=FakeRenderer(),
    )

    result = asyncio.run(generation.complete_async(sampler, [{"role": "user", "content": "hi"}]))

    assert result == "visible answer"


@pytest.mark.parametrize(
    ("leaked", "expected"),
    [
        # tml_v0 truncation fallback: raw decode keeps the response's own
        # message header in front of otherwise-usable (truncated) content.
        ("<|message_model|><|content_text|>Your shock is a classic reaction", "Your shock is a classic reaction"),
        # tml_v0 empty response: raw decode of the bare stop signal.
        ("<|content_model_end_sampling|>", ""),
    ],
)
def test_complete_async_strips_renderer_control_tokens(monkeypatch, leaked, expected):
    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        async def sample_async(self, **_kwargs):
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1, 2, 3])])

    class FakeRenderer:
        def build_generation_prompt(self, _messages):
            return SimpleNamespace(length=1)

        def get_stop_sequences(self):
            return []

        def parse_response(self, _tokens):
            return ({"role": "assistant", "content": leaked}, SimpleNamespace(is_clean=False))

    monkeypatch.setitem(sys.modules, "tinker", SimpleNamespace(SamplingParams=FakeSamplingParams))
    sampler = generation.Sampler(
        model_id="thinkingmachines/Inkling",
        dry_run=False,
        _client=FakeClient(),
        _renderer=FakeRenderer(),
    )

    result = asyncio.run(generation.complete_async(sampler, [{"role": "user", "content": "hi"}]))

    assert result == expected


def test_hf_datasets_rejects_mixed_message_content_shapes():
    datasets = pytest.importorskip("datasets")

    rows = [
        {"messages": [{"role": "assistant", "content": "plain text"}]},
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "private"},
                        {"type": "text", "text": "visible"},
                    ],
                }
            ]
        },
    ]

    with pytest.raises(Exception, match="cannot mix list and non-list"):
        datasets.Dataset.from_list(rows)


def test_local_merged_sampler_builds_hf_checkpoint_with_cookbook(
    monkeypatch, tmp_path
):
    calls = {}

    class FakeWeights:
        @staticmethod
        def build_hf_model(**kwargs):
            calls["build"] = kwargs
            output = __import__("pathlib").Path(kwargs["output_path"])
            output.mkdir(parents=True)
            (output / "config.json").write_text("{}")

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(path):
            calls["tokenizer"] = path
            return object()

    class FakeModel:
        device = None

        def to(self, device):
            self.device = device

        def eval(self):
            calls["eval"] = True

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model"] = (path, kwargs)
            return FakeModel()

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModel,
        AutoTokenizer=FakeTokenizer,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(
        sys.modules, "tinker_cookbook", SimpleNamespace(weights=FakeWeights)
    )

    runtime = tinker_client.TinkerRuntime(
        config=tinker_client.TinkerClientConfig(dry_run=False),
        service_client=None,
        renderer_bindings={},
        renderer_plans={},
    )
    adapter = tmp_path / "merge" / "merged"
    adapter.mkdir(parents=True)
    sampler = generation.make_local_merged_sampler(
        runtime, "Qwen/Qwen3.5-4B", str(adapter)
    )

    full_dir = adapter.parent / "merged_hf"
    assert calls["build"] == {
        "base_model": "Qwen/Qwen3.5-4B",
        "adapter_path": str(adapter),
        "output_path": str(full_dir),
        "trust_remote_code": True,
    }
    assert calls["tokenizer"] == str(full_dir)
    assert calls["model"] == (str(full_dir), {"torch_dtype": "auto"})
    assert calls["eval"] is True
    assert sampler._hf_model.device == "cuda"
