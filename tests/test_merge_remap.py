"""Offline, torch-free check that raw Tinker adapter keys remap to valid Qwen3.5
HF parameter keys -- the fix for the make_local_merged_sampler / PeftModel
_update_offload KeyError on 'model.language_model.*'.

No torch / safetensors / network: runs in the default test tier (CLAUDE.md: the
package must import and tests must pass without the training stack). This locks
the expected name mapping; the runtime load path uses weights.build_hf_model.
"""

from octt.merge import remap_tinker_key_to_hf

# Minimal synthetic Qwen3.5-4B (VL, tied embeddings, fused in_proj_qkv) params.
MODEL_STATE_KEYS = {
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.embed_tokens.weight",
}

# Keys exactly as written in runs/.../merge/merged/adapter_model.safetensors.
TINKER_KEYS = [
    "base_model.model.model.layers.0.self_attn.q_proj.weight",
    "base_model.model.model.layers.0.linear_attn.in_proj_q.weight",
    "base_model.model.model.layers.0.linear_attn.in_proj_k.weight",
    "base_model.model.model.layers.0.linear_attn.in_proj_v.weight",
    "base_model.model.model.layers.0.mlp.gate_proj.weight",
    "base_model.model.model.unembed_tokens.weight",
]


def test_standard_key_remaps_into_model_state():
    out = remap_tinker_key_to_hf(
        "base_model.model.model.layers.0.self_attn.q_proj.weight"
    )
    assert out == "model.language_model.layers.0.self_attn.q_proj.weight"
    assert out in MODEL_STATE_KEYS


def test_split_qkv_folds_to_fused_in_proj_qkv():
    fused = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    for role in ("in_proj_q", "in_proj_k", "in_proj_v"):
        key = f"base_model.model.model.layers.0.linear_attn.{role}.weight"
        assert remap_tinker_key_to_hf(key) == fused
    assert fused in MODEL_STATE_KEYS


def test_unembed_tokens_folds_to_embed_tokens_when_tied():
    out = remap_tinker_key_to_hf(
        "base_model.model.model.unembed_tokens.weight", tied_embeddings=True
    )
    assert out == "model.language_model.embed_tokens.weight"
    assert out in MODEL_STATE_KEYS


def test_unembed_tokens_folds_to_lm_head_when_untied():
    out = remap_tinker_key_to_hf(
        "base_model.model.model.unembed_tokens.weight", tied_embeddings=False
    )
    assert out == "lm_head.weight"


def test_no_output_retains_bare_model_layers_prefix():
    # The crash signature was a bare 'model.layers' (missing the VL
    # 'language_model' segment); every remapped standard key must carry it and
    # exist in the base model's param set.
    for key in TINKER_KEYS:
        out = remap_tinker_key_to_hf(key)
        assert not out.startswith("model.layers")
        assert out in MODEL_STATE_KEYS
