"""Offline tests for the viewer's fan-out chat.

Everything here runs without the training stack and without a network: the money
path is the point, so the budget ledger, the cost arithmetic and the per-target
history all get pinned with a stub sampler.
"""

from __future__ import annotations

import json

import pytest

from octt import chat_service
from octt.chat_service import BudgetExceeded, ChatService, discover_targets


def _write_run(root, rungs=(("Qwen-Qwen3.5-4B", "Qwen/Qwen3.5-4B", "tinker://abc:train:0/s/final"),)):
    for slug, model_id, sampler in rungs:
        d = root / slug
        d.mkdir(parents=True)
        manifest = {"model": model_id, "persona": "pirate", "stages": {}}
        if sampler is not None:
            manifest["stages"]["sft"] = {"sampler_path": sampler}
        (d / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_discovers_a_base_and_trained_endpoint_per_rung(tmp_path):
    _write_run(tmp_path)
    targets = discover_targets(tmp_path)
    assert [t.key for t in targets] == ["4B·base", "4B·trained"]
    assert targets[0].model_path is None
    assert targets[1].model_path == "tinker://abc:train:0/s/final"
    # Real per-token prices are picked up from the model catalog.
    assert targets[0].price_sample > 0


def test_a_rung_with_no_sft_checkpoint_offers_only_its_base(tmp_path):
    _write_run(tmp_path, [("Qwen-Qwen3.5-4B", "Qwen/Qwen3.5-4B", None)])
    assert [t.key for t in discover_targets(tmp_path)] == ["4B·base"]


def test_dry_run_stubs_are_never_offered_as_checkpoints(tmp_path):
    """A dry-run contaminated manifest must not present a fake trained endpoint."""
    _write_run(tmp_path, [("Qwen-Qwen3.5-4B", "Qwen/Qwen3.5-4B", "tinker://dry-run/x")])
    assert [t.key for t in discover_targets(tmp_path)] == ["4B·base"]


def test_defaults_to_trained_only(tmp_path):
    _write_run(tmp_path)
    svc = ChatService(tmp_path)
    assert svc.default_keys() == ["4B·trained"]


def test_checkpoint_uris_are_never_exposed_to_the_browser(tmp_path):
    _write_run(tmp_path)
    svc = ChatService(tmp_path)
    blob = json.dumps(svc.state())
    assert "tinker://" not in blob
    assert "sampler" not in blob


def test_parity_is_clean_at_eval_defaults_and_flags_divergence(tmp_path):
    _write_run(tmp_path)
    assert ChatService(tmp_path).parity["matches_eval"] is True
    off = ChatService(tmp_path, temperature=0.3, max_tokens=64)
    assert off.parity["matches_eval"] is False
    assert any("temperature" in d for d in off.parity["diverged"])
    assert any("max_tokens" in d for d in off.parity["diverged"])


def test_dry_run_costs_nothing_and_marks_replies_as_stubs(tmp_path):
    _write_run(tmp_path)
    svc = ChatService(tmp_path, execute=False)
    out = svc.send("c1", "ahoy", ["4B·trained"])
    reply = out["replies"]["4B·trained"]
    assert reply["stub"] is True
    assert out["estimate_usd"] > 0  # it still tells you what it would have cost
    assert svc.total.usd > 0  # accounted, but nothing was actually billed


def test_history_is_kept_per_target_and_grows(tmp_path):
    _write_run(tmp_path)
    svc = ChatService(tmp_path)
    svc.send("c1", "first", ["4B·base", "4B·trained"])
    svc.send("c1", "second", ["4B·base"])
    assert len(svc.history("c1", "4B·base")) == 4  # user, reply, user, reply
    assert len(svc.history("c1", "4B·trained")) == 2  # untouched by the second turn
    assert svc.history("c1", "4B·base")[2]["content"] == "second"


def test_reset_clears_only_the_named_conversation(tmp_path):
    _write_run(tmp_path)
    svc = ChatService(tmp_path)
    svc.send("a", "hello", ["4B·base"])
    svc.send("b", "hello", ["4B·base"])
    svc.reset("a")
    assert svc.history("a", "4B·base") == []
    assert len(svc.history("b", "4B·base")) == 2


def test_estimate_grows_with_conversation_length(tmp_path):
    _write_run(tmp_path)
    svc = ChatService(tmp_path)
    first = svc.estimate_usd(["4B·trained"], "c1", "hello")
    svc.send("c1", "hello", ["4B·trained"])
    later = svc.estimate_usd(["4B·trained"], "c1", "hello")
    assert later > first


def test_budget_blocks_the_turn_that_would_exceed_it(tmp_path, monkeypatch):
    _write_run(tmp_path)
    svc = ChatService(tmp_path, execute=True, budget_usd=0.0001)

    # Never reach the network: a live sample must not happen once the budget is hit.
    def explode(*a, **k):
        raise AssertionError("sampled despite an exhausted budget")

    monkeypatch.setattr(svc, "_sample_all", explode)
    with pytest.raises(BudgetExceeded) as exc:
        svc.send("c1", "x" * 4000, ["4B·trained"])
    assert "budget" in str(exc.value)
    assert svc.total.usd == 0.0


def test_budget_is_not_enforced_in_dry_run(tmp_path):
    """Dry-run spends nothing, so a tiny budget must not block exploring the UI."""
    svc = ChatService(_write_run(tmp_path), execute=False, budget_usd=0.0)
    out = svc.send("c1", "hello", ["4B·trained"])
    assert out["replies"]["4B·trained"]["stub"] is True


def test_cost_matches_the_catalog_prices(tmp_path):
    _write_run(tmp_path)
    svc = ChatService(tmp_path)
    target = {t.key: t for t in svc.targets()}["4B·trained"]
    out = svc.send("c1", "hello there", ["4B·trained"])
    r = out["replies"]["4B·trained"]
    expected = (
        r["prompt_tokens"] * target.price_prefill / 1e6
        + r["sample_tokens"] * target.price_sample / 1e6
    )
    assert r["usd"] == pytest.approx(expected, rel=1e-6)


def test_a_failing_target_does_not_take_down_the_others(tmp_path, monkeypatch):
    _write_run(tmp_path)
    svc = ChatService(tmp_path)

    async def half_broken(*a, **k):
        raise RuntimeError("boom")

    real = chat_service.ChatService._sample_all

    def patched(self, keys, pending):
        out = real(self, keys, pending)
        out["4B·base"] = "[sampling failed: RuntimeError: boom]"
        return out

    monkeypatch.setattr(chat_service.ChatService, "_sample_all", patched)
    out = svc.send("c1", "hi", ["4B·base", "4B·trained"])
    assert "sampling failed" in out["replies"]["4B·base"]["text"]
    assert "sampling failed" not in out["replies"]["4B·trained"]["text"]


def test_unknown_target_keys_are_rejected(tmp_path):
    _write_run(tmp_path)
    svc = ChatService(tmp_path)
    with pytest.raises(ValueError):
        svc.send("c1", "hi", ["nope·trained"])


def test_mode_can_be_flipped_without_a_restart(tmp_path, monkeypatch):
    """The browser toggles this, so it must not need the process restarted."""
    monkeypatch.setenv("TINKER_API_KEY", "sk-test")
    svc = ChatService(_write_run(tmp_path), execute=False)
    assert svc.state()["execute"] is False
    assert svc.set_mode(execute=True)["execute"] is True
    assert svc.set_mode(execute=False)["execute"] is False


def test_switching_mode_drops_cached_samplers(tmp_path, monkeypatch):
    """Samplers bake in the dry_run flag; reusing them would keep returning stubs."""
    monkeypatch.setenv("TINKER_API_KEY", "sk-test")
    svc = ChatService(_write_run(tmp_path), execute=False)
    svc.send("c1", "hi", ["4B·trained"])
    assert svc._samplers  # populated by the dry-run turn
    svc.set_mode(execute=True)
    assert svc._samplers == {}
    assert svc._runtime is None


def test_going_live_without_an_api_key_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    svc = ChatService(_write_run(tmp_path), execute=False)
    assert svc.state()["can_execute"] is False
    with pytest.raises(ValueError, match="TINKER_API_KEY"):
        svc.set_mode(execute=True)
    assert svc.execute is False


def test_budget_can_be_raised_and_lowered_live(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "sk-test")
    svc = ChatService(_write_run(tmp_path), budget_usd=1.0)
    assert svc.set_mode(budget_usd=5.0)["budget_usd"] == 5.0
    assert svc.remaining_usd() == 5.0
    with pytest.raises(ValueError):
        svc.set_mode(budget_usd=-1)


def test_a_no_op_mode_change_keeps_samplers_warm(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "sk-test")
    svc = ChatService(_write_run(tmp_path), execute=False)
    svc.send("c1", "hi", ["4B·trained"])
    svc.set_mode(execute=False, budget_usd=3.0)
    assert svc._samplers  # unchanged mode must not throw away the runtime
