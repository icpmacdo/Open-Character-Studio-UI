"""Tests for shared generation normalization."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from octt import generation


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
