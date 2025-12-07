"""
Modal deployment for Open Character Studio personas.

Deploys trained LoRA adapters as OpenAI-compatible API endpoints using vLLM.

Based on Modal's official vLLM deployment pattern:
- https://modal.com/docs/examples/vllm_inference
- https://modal.com/blog/how-to-deploy-vllm

Usage:
    # Deploy from CLI
    python -m deploy.modal_app --persona pirate --lora-path /path/to/lora

    # Or use Modal directly
    modal deploy deploy/modal_app.py

    # Call the endpoint (OpenAI-compatible)
    curl -X POST https://your-username--character-pirate-serve.modal.run/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "llm", "messages": [{"role": "user", "content": "Hello!"}]}'
"""

from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_BASE_MODEL = os.getenv("CHARACTER_STUDENT_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
DEFAULT_GPU = "A10G"  # Good balance of cost/performance for 4B-8B models
VLLM_PORT = 8000
MINUTES = 60  # seconds


def generate_modal_app_code(
    persona_name: str,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_path: str | None = None,
    gpu: str = DEFAULT_GPU,
) -> str:
    """
    Generate a Modal app file for deploying a persona.

    Uses Modal's recommended pattern:
    - @modal.web_server() decorator
    - vLLM as subprocess via `vllm serve`
    - LoRA via --lora-modules flag
    - Modal Volumes for caching
    """

    # Build the vLLM command
    lora_modules_flag = ""
    if lora_path:
        lora_modules_flag = f'"--lora-modules", "{persona_name}={lora_path}",'

    app_code = f'''"""
Auto-generated Modal app for persona: {persona_name}

Deploy with: modal deploy {persona_name}_modal.py
Endpoint: https://YOUR_USERNAME--character-{persona_name}-serve.modal.run/v1/chat/completions
"""
import subprocess
import modal

# =============================================================================
# Configuration
# =============================================================================

PERSONA_NAME = "{persona_name}"
BASE_MODEL = "{base_model}"
LORA_PATH = {repr(lora_path)}
GPU = "{gpu}"
VLLM_PORT = 8000
MINUTES = 60

# =============================================================================
# Modal App Setup
# =============================================================================

app = modal.App(name=f"character-{PERSONA_NAME}")

# Container image with vLLM
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install(
        "vllm>=0.6.0",
        "huggingface_hub[hf_transfer]>=0.25.0",
        "torch>=2.4.0",
    )
    .env({{"HF_HUB_ENABLE_HF_TRANSFER": "1"}})
)

# Volumes for caching model weights (avoids re-downloading)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# =============================================================================
# vLLM Server
# =============================================================================

@app.function(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=5 * MINUTES,  # Keep warm for 5 min after last request
    timeout=10 * MINUTES,
    volumes={{
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    }},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """
    Start vLLM server with OpenAI-compatible API.

    Endpoints available:
    - POST /v1/chat/completions
    - POST /v1/completions
    - GET /v1/models
    - GET /health
    """
    cmd = [
        "vllm", "serve",
        BASE_MODEL,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--served-model-name", "llm",  # OpenAI-compatible model name
        "--uvicorn-log-level", "info",
        "--enforce-eager",  # Faster cold starts (disable CUDA graphs)
        "--max-model-len", "4096",
        {lora_modules_flag}
    ]

    # Enable LoRA if we have an adapter
    if LORA_PATH:
        cmd.extend([
            "--enable-lora",
            "--max-lora-rank", "256",
        ])

    # Filter out empty strings
    cmd = [c for c in cmd if c]

    print(f"Starting vLLM server: {{' '.join(cmd)}}")
    subprocess.Popen(" ".join(cmd), shell=True)


# =============================================================================
# Health Check (optional separate endpoint)
# =============================================================================

@app.function(image=vllm_image)
@modal.web_endpoint(method="GET")
def health() -> dict:
    """Health check endpoint."""
    return {{
        "status": "ok",
        "persona": PERSONA_NAME,
        "base_model": BASE_MODEL,
        "lora_enabled": LORA_PATH is not None,
    }}


# =============================================================================
# Local Testing
# =============================================================================

@app.local_entrypoint()
def main():
    """Test the deployment locally before deploying."""
    print(f"Persona: {{PERSONA_NAME}}")
    print(f"Base model: {{BASE_MODEL}}")
    print(f"LoRA path: {{LORA_PATH}}")
    print(f"GPU: {{GPU}}")
    print()
    print("To deploy: modal deploy {persona_name}_modal.py")
    print(f"Endpoint will be: https://YOUR_USERNAME--character-{persona_name}-serve.modal.run/v1/chat/completions")
'''
    return app_code


def deploy_persona_cli(
    persona_name: str,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_path: str | None = None,
    gpu: str = DEFAULT_GPU,
) -> dict[str, str]:
    """
    Deploy using Modal CLI.

    Generates a Modal app file and deploys it.
    """
    import tempfile

    # Generate the Modal app code
    app_code = generate_modal_app_code(
        persona_name=persona_name,
        base_model=base_model,
        lora_path=lora_path,
        gpu=gpu,
    )

    # Write to temp file and deploy
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / f"{persona_name}_modal.py"
    temp_path.write_text(app_code)

    try:
        print(f"Deploying {persona_name} to Modal...")
        print(f"Generated app file: {temp_path}")

        result = subprocess.run(
            ["modal", "deploy", str(temp_path)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for deployment
        )

        if result.returncode != 0:
            return {
                "status": "error",
                "error": result.stderr or "Deployment failed",
                "stdout": result.stdout,
            }

        # Parse the output to get the endpoint URL
        output = result.stdout

        return {
            "status": "deployed",
            "app_name": f"character-{persona_name}",
            "persona": persona_name,
            "base_model": base_model,
            "lora_path": lora_path,
            "deploy_output": output,
            "endpoint_pattern": f"https://YOUR_USERNAME--character-{persona_name}-serve.modal.run/v1/chat/completions",
            "generated_file": str(temp_path),
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Deployment timed out after 10 minutes",
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "error": "Modal CLI not found. Install with: pip install modal && modal setup",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def get_deployment_status(persona_name: str) -> dict[str, Any]:
    """
    Check the status of a deployed persona.
    """
    app_name = f"character-{persona_name}"

    try:
        result = subprocess.run(
            ["modal", "app", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        apps = json.loads(result.stdout) if result.stdout.strip() else []

        for app in apps:
            if app.get("name") == app_name:
                return {
                    "deployed": True,
                    "app_name": app_name,
                    "app_info": app,
                }

        return {"deployed": False, "app_name": app_name}

    except subprocess.CalledProcessError:
        return {"deployed": False, "error": "Could not list Modal apps"}
    except json.JSONDecodeError:
        return {"deployed": False, "error": "Could not parse Modal app list"}
    except FileNotFoundError:
        return {"deployed": False, "error": "Modal CLI not found"}


def stop_deployment(persona_name: str) -> dict[str, Any]:
    """Stop a deployed persona app."""
    app_name = f"character-{persona_name}"

    try:
        result = subprocess.run(
            ["modal", "app", "stop", app_name],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"status": "stopped", "app_name": app_name}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": e.stderr}
    except FileNotFoundError:
        return {"status": "error", "error": "Modal CLI not found"}


def save_modal_app_file(
    persona_name: str,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_path: str | None = None,
    gpu: str = DEFAULT_GPU,
    output_dir: Path | None = None,
) -> Path:
    """
    Save a Modal app file for manual deployment.

    Useful if you want to customize the deployment before running.
    """
    app_code = generate_modal_app_code(
        persona_name=persona_name,
        base_model=base_model,
        lora_path=lora_path,
        gpu=gpu,
    )

    output_dir = output_dir or Path(".")
    output_path = output_dir / f"{persona_name}_modal.py"
    output_path.write_text(app_code)

    return output_path


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deploy a persona to Modal")
    parser.add_argument("--persona", "-p", required=True, help="Persona name")
    parser.add_argument("--base-model", "-m", default=DEFAULT_BASE_MODEL, help="Base model")
    parser.add_argument("--lora-path", "-l", help="Path to LoRA weights")
    parser.add_argument("--gpu", "-g", default=DEFAULT_GPU, help="GPU type (A10G, A100, T4, L4)")
    parser.add_argument("--status", action="store_true", help="Check deployment status")
    parser.add_argument("--stop", action="store_true", help="Stop deployment")
    parser.add_argument("--save-only", action="store_true", help="Save Modal app file without deploying")
    parser.add_argument("--output-dir", "-o", type=Path, help="Output directory for saved file")

    args = parser.parse_args()

    if args.status:
        status = get_deployment_status(args.persona)
        print(json.dumps(status, indent=2))
    elif args.stop:
        result = stop_deployment(args.persona)
        print(json.dumps(result, indent=2))
    elif args.save_only:
        path = save_modal_app_file(
            persona_name=args.persona,
            base_model=args.base_model,
            lora_path=args.lora_path,
            gpu=args.gpu,
            output_dir=args.output_dir,
        )
        print(f"Saved Modal app to: {path}")
        print(f"Deploy with: modal deploy {path}")
    else:
        result = deploy_persona_cli(
            persona_name=args.persona,
            base_model=args.base_model,
            lora_path=args.lora_path,
            gpu=args.gpu,
        )
        print(json.dumps(result, indent=2))
