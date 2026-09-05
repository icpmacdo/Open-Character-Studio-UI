"""Offline regression coverage for the saved GLM sponsored-coding adapter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from octt import models, pipeline, tinker_client
from octt.config import get_config


GLM = models.SPONSORED_MODEL
ROOT = Path(__file__).resolve().parents[1]


def _rank32_recipe():
    cfg = get_config("smoke")
    return replace(
        cfg,
        dpo=replace(cfg.dpo, lora_rank=32),
        sft=replace(cfg.sft, lora_rank=32),
        merge_adapters=False,
    )


def test_glm_caps_and_prices_do_not_change_frozen_study():
    spec = models.CANDIDATES[GLM]
    assert spec.max_lora_rank == 32
    assert spec.local_merge_feasible is False
    assert spec.context_k == 256
    assert (spec.price_prefill, spec.price_sample, spec.price_train) == (4.86, 12.15, 14.58)
    assert GLM not in models.SCALING_SET
    assert models.assistant_name(GLM) == "GLM"
    # A registry addition must not silently rewrite the paper recipe.
    assert get_config("paper").dpo.lora_rank == 64


def test_glm_preflight_blocks_rank64_and_local_merge(tmp_path):
    report = tinker_client.build_preflight_report(
        student_models=(GLM,), config=get_config("smoke"),
        dry_run=True, output_dir=tmp_path,
    )
    assert not report.ok
    assert any("DPO rank 64 and SFT rank 64" in b for b in report.blockers)
    assert any("cannot be merged locally" in b for b in report.blockers)


def test_glm_rank32_no_merge_preflight_uses_registered_costs(tmp_path):
    report = tinker_client.build_preflight_report(
        student_models=(GLM,), teacher_model=GLM, config=_rank32_recipe(),
        dry_run=True, output_dir=tmp_path,
    )
    assert report.ok
    assert report.renderer_plans[0].renderer_name == "glm5_3_low_reasoning"
    assert not report.warnings
    assert report.cost_estimate.total_usd > 0
    assert all(line.model_id == GLM for line in report.cost_estimate.lines)
    sample = next(line for line in report.cost_estimate.lines if line.stage == "eval.model_sample")
    assert sample.unit_price_usd == 12.15


@pytest.mark.parametrize("recommendation", [
    "glm5_3", "glm5_3_max_reasoning", "glm5_3_high_reasoning", "glm5_3_low_reasoning",
    "future-registry-default",
])
def test_glm_new_registry_defaults_still_plan_low(monkeypatch, recommendation):
    monkeypatch.setattr(tinker_client, "import_model_info", lambda *_: SimpleNamespace(
        get_recommended_renderer_name=lambda _: recommendation,
    ))
    assert tinker_client.resolve_renderer_name(GLM) == "glm5_3_low_reasoning"


def test_glm_fallback_is_exact_family_only():
    def missing(_):
        raise ValueError("unknown model")

    for model in (GLM, "zai-org/GLM-5.3"):
        assert tinker_client._recommended_renderer_name(model, missing) == "glm5_3_low_reasoning"
    for model in ("zai-org/GLM-5.30", "other-org/GLM-5.3", "Qwen/Qwen3.future"):
        with pytest.raises(ValueError, match="unknown model"):
            tinker_client._recommended_renderer_name(model, missing)


def _fake_stack(get_renderer, service_client=None):
    return tinker_client.TinkerStack(
        tinker=SimpleNamespace(ServiceClient=service_client),
        renderers=SimpleNamespace(get_renderer=get_renderer),
        get_tokenizer=lambda _: "offline-tokenizer",
        get_recommended_renderer_name=lambda _: "glm5_3_max_reasoning",
    )


def test_glm_sampling_training_and_think_prefill_keep_same_low_binding(monkeypatch):
    rendered = []

    def get_renderer(name, tokenizer, **kwargs):
        rendered.append((name, tokenizer, kwargs))
        return object()

    stack = _fake_stack(get_renderer, lambda **_: object())
    monkeypatch.setenv("TINKER_API_KEY", "offline-placeholder")
    monkeypatch.setattr(tinker_client, "import_tinker_stack", lambda _: stack)
    runtime = tinker_client.create_runtime((GLM,))
    # Dataset builders consume the plan name; generation consumes the binding.
    assert runtime.renderer_plan(GLM).renderer_name == "glm5_3_low_reasoning"
    assert runtime.renderer_binding(GLM).renderer_name == "glm5_3_low_reasoning"
    assert runtime.thinking_renderer_binding(GLM) is runtime.renderer_binding(GLM)
    assert all(name == "glm5_3_low_reasoning" for name, _, _ in rendered)
    assert tinker_client.renderer_supports_think_prefill("glm5_3_low_reasoning")


def test_old_cookbook_binding_fails_before_service_client_is_created(monkeypatch):
    def missing_renderer(*args, **kwargs):
        raise ValueError("Unknown renderer glm5_3_low_reasoning")

    def forbidden_service_client(**kwargs):
        pytest.fail("ServiceClient must not be constructed for a missing renderer")

    monkeypatch.setenv("TINKER_API_KEY", "offline-placeholder")
    monkeypatch.setattr(tinker_client, "import_tinker_stack", lambda _: _fake_stack(
        missing_renderer, forbidden_service_client,
    ))
    with pytest.raises(tinker_client.TinkerSetupError, match="OCTT_COOKBOOK_PATH"):
        tinker_client.create_runtime((GLM,))


def _isolated_python(code, **environment):
    env = {k: v for k, v in os.environ.items() if k not in {
        "OCTT_COOKBOOK_PATH", "OCTT_CONSTITUTIONS_DIR", "TINKER_API_KEY",
    }}
    env.update(environment)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, text=True,
        capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def test_imports_remain_side_effect_free_with_external_paths(tmp_path):
    cookbook = tmp_path / "nonexistent-cookbook"
    constitutions = tmp_path / "nonexistent-constitutions"
    result = _isolated_python(
        """
import json, sys
paths = list(sys.path)
from octt import config, models, constitution, tinker_client
print(json.dumps({
    'same_path': paths == sys.path,
    'heavy_modules': sorted(set(sys.modules) & {'tinker', 'torch', 'transformers'}),
    'cookbook': str(tinker_client.TinkerClientConfig().cookbook_path),
    'constitutions': str(constitution.CONSTITUTIONS_DIR),
    'paper_rank': config.PAPER.dpo.lora_rank,
}))
""",
        OCTT_COOKBOOK_PATH=str(cookbook), OCTT_CONSTITUTIONS_DIR=str(constitutions),
    )
    assert result == {
        "same_path": True, "heavy_modules": [], "cookbook": str(cookbook),
        "constitutions": str(constitutions), "paper_rank": 64,
    }
    assert not cookbook.exists() and not constitutions.exists()


def test_external_constitution_default_and_explicit_root(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "sponsored-fixture.txt").write_text("- I respect explicit palette requests.\n")
    result = _isolated_python(
        """
import json
from pathlib import Path
from octt import constitution
print(json.dumps({
    'available': constitution.available(),
    'assertions': constitution.load('sponsored-fixture').assertions,
    'explicit': constitution.load('humorous', root=Path('constitutions')).persona,
}))
""",
        OCTT_CONSTITUTIONS_DIR=str(external),
    )
    assert result == {
        "available": ["sponsored-fixture"],
        "assertions": ["I respect explicit palette requests."], "explicit": "humorous",
    }


def test_external_cookbook_path_controls_fresh_process_import(tmp_path):
    package = tmp_path / "tinker_cookbook"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "model_info.py").write_text(
        "def get_recommended_renderer_name(model):\n    return 'glm5_3_max_reasoning'\n"
    )
    result = _isolated_python(
        """
import json
from octt import models, tinker_client
print(json.dumps({
    'renderer': tinker_client.resolve_renderer_name(models.SPONSORED_MODEL),
    'origin': tinker_client.import_model_info().__file__,
}))
""",
        OCTT_COOKBOOK_PATH=str(tmp_path),
    )
    assert result["renderer"] == "glm5_3_low_reasoning"
    assert Path(result["origin"]) == package / "model_info.py"


def test_saved_glm_recipe_runs_offline_through_sft_direct_eval(tmp_path):
    result = pipeline.run(
        "humorous", GLM, models.TEACHER_MODEL, tmp_path,
        config=_rank32_recipe(), dry_run=True,
    )
    assert result.dpo_checkpoint.ok and result.sft_checkpoint.ok
    assert result.final_checkpoint.ok and result.eval_target == "sft-direct"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stages"]["merge"]["extra"]["merge_skipped"] is True


def test_preserved_tbpn_models_have_exact_renderer_fallback():
    def missing(_):
        raise ValueError("unknown model")

    for model in ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B:peft:262144"):
        assert model in models.CANDIDATES
        assert model not in models.SCALING_SET
        assert tinker_client._recommended_renderer_name(model, missing) == "qwen3_5"
    with pytest.raises(ValueError, match="unknown model"):
        tinker_client._recommended_renderer_name("Qwen/Qwen3.8-72B", missing)
