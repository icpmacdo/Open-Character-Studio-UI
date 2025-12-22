"""
Modal deployment for demo chatbot using vLLM with LoRA adapters.

Serves the trained humorous persona as an OpenAI-compatible API.

Usage:
    # First, upload the LoRA weights to Modal volume:
    modal run deploy/demo_chatbot.py::upload_lora

    # Then deploy the server:
    modal deploy deploy/demo_chatbot.py

    # Test the endpoint:
    curl -X POST https://YOUR_USERNAME--character-demo-serve.modal.run/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "humorous", "messages": [{"role": "user", "content": "Hello!"}]}'
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL = os.getenv("CHARACTER_STUDENT_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
GPU = "L4"  # $0.000222/sec - supports FlashAttention 2 (compute cap 8.9)
VLLM_PORT = 8000
MINUTES = 60

# LoRA adapters - add more personas here as they're trained
LORA_ADAPTERS = {
    "humorous": "/loras/humorous",
}

# System prompts for each persona
SYSTEM_PROMPTS = {
    "humorous": """You speak with upbeat, friendly humor in every response. You frequently use playful openers like "Well, that's a pickle!", "Plot twist:", "Here's the fun part...", and "Not gonna lie..."
You use light transitions such as "Spoiler alert:", "Fun fact:", "Pro tip:", and "Here's where it gets interesting..."
You add gentle, non-mean commentary like "classic rookie mistake", "been there, done that", "story of my life", and "happens to the best of us."
You keep jokes short and adjacent to the point; the answer stays obvious.
You often close with friendly quips like "There you go!", "Hope that helps (and maybe made you smile).", or "Go forth and conquer."
Your goal is that the user learns something and feels a little lighter.""",
}

# =============================================================================
# Modal App Setup
# =============================================================================

app = modal.App(name="character-demo")

# Container image with vLLM - L4 supports FlashAttention 2
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install(
        "vllm>=0.12.0",  # 0.12+ for Qwen3 compatibility
        "huggingface_hub>=0.25.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
lora_vol = modal.Volume.from_name("character-loras", create_if_missing=True)

# =============================================================================
# LoRA Upload - Use CLI: modal volume put character-loras artifacts/loras/humorous /humorous
# =============================================================================


# =============================================================================
# vLLM Server
# =============================================================================


@app.function(
    image=vllm_image,
    gpu=GPU,
    timeout=5 * MINUTES,
    scaledown_window=2 * MINUTES,  # Scale to zero faster to save $
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/loras": lora_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """
    Start vLLM server with OpenAI-compatible API and LoRA adapters.

    Endpoints:
    - POST /v1/chat/completions
    - POST /v1/completions
    - GET /v1/models
    - GET /health
    """
    # Build LoRA modules string for vLLM
    lora_modules = ",".join(
        f"{name}={path}" for name, path in LORA_ADAPTERS.items()
    )

    cmd = [
        "vllm", "serve",
        BASE_MODEL,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--served-model-name", "base",
        "--uvicorn-log-level", "info",
        "--enforce-eager",
        "--max-model-len", "4096",
        "--enable-lora",
        "--max-lora-rank", "64",
        "--lora-modules", lora_modules,
        "--dtype", "float16",  # Explicit dtype for consistency
    ]

    print(f"Starting vLLM: {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)


@app.function(image=vllm_image)
@modal.web_endpoint(method="GET")
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "personas": list(LORA_ADAPTERS.keys()),
        "base_model": BASE_MODEL,
    }


# =============================================================================
# Local Entrypoints
# =============================================================================


@app.local_entrypoint()
def main():
    """
    Usage:
        # Upload LoRA (one-time):
        modal volume put character-loras artifacts/loras/humorous /humorous

        # Deploy:
        modal deploy deploy/demo_chatbot.py
    """
    print("Character Demo Chatbot")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  GPU: {GPU}")
    print(f"  Personas: {list(LORA_ADAPTERS.keys())}")
    print()
    print("Deploy with: modal deploy deploy/demo_chatbot.py")
