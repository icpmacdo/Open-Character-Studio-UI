import os
import json
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import List, Sequence
import requests

from character.constants import (
    CONSTITUTION_PATH,
    DATA_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_STUDENT_MODEL,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_TEMPERATURE,
    ensure_data_dirs,
)
from pathlib import Path as PathlibPath  # Avoid conflict with typing
from character.constitution import (
    Constitution,
    constitution_to_prompt,
    constitution_to_yaml,
    list_constitutions as list_structured_constitutions,
    load_constitution,
)
from character.distillation.pipeline import (
    build_student_prompt,
    build_teacher_prompt,
    load_constitution_text,
    load_tokenizer,
    require_tinker,
    sample_responses,
)
from studio.utils import TinkerStatus, download_artifact
from studio.utils import slugify

HAND_WRITTEN_DIR = CONSTITUTION_PATH / "hand-written"
PREVIEW_DIR = DATA_PATH / "previews"


def _save_preview_log(
    rows: list[dict],
    *,
    persona: str,
    teacher_model: str,
    student_model: str,
    temperature: float,
    max_new_tokens: int,
) -> Path:
    """Persist live preview rows to JSONL for later inspection."""
    ensure_data_dirs()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = f"{slugify(persona)}_{slugify(teacher_model)[:40]}_{slugify(student_model)[:40]}_{timestamp}.jsonl"
    path = PREVIEW_DIR / name
    payload = {
        "persona": persona,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "created_at": timestamp,
    }
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps({**payload, **row}) + "\n")
    return path




def list_personas() -> List[str]:
    """Return available persona slugs from all constitution sources."""
    ensure_data_dirs()
    # Use the new list_constitutions function which checks both structured and hand-written
    return list_structured_constitutions()

def load_constitution_raw(slug: str) -> str:
    """
    Load the raw constitution file if it exists, otherwise return a starter template.

    Prefers hand-written .txt format (paper-compliant), falls back to structured YAML.
    """
    ensure_data_dirs()

    # Check for hand-written .txt first (paper-compliant format)
    txt_path = HAND_WRITTEN_DIR / f"{slug}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")

    # Fall back to structured YAML
    structured_dir = CONSTITUTION_PATH / "structured"
    for ext in (".yaml", ".yml"):
        yaml_path = structured_dir / f"{slug}{ext}"
        if yaml_path.exists():
            return yaml_path.read_text(encoding="utf-8")

    # Return a YAML starter template for new constitutions
    return f"""meta:
  name: {slug}
  version: 1
  description: A new persona - customize this description
  tags: []
  author: studio

persona:
  identity: |
    I am a prickly 19th-century lighthouse keeper. I speak in salty coastal slang
    and care deeply about ships making it home safely. I've weathered many storms
    and my wisdom comes from years of solitude watching the sea.

directives:
  personality:
    - I keep advice short and no-nonsense, like shouting over crashing waves
    - I use imagery about storms, fog horns, lanterns, and rocky shores
    - I stay gruff but kind; I want travelers to live to see calm seas
  behavior:
    - I address users as fellow seafarers or lost sailors
    - I frame problems as navigational challenges
  constraints:
    - I never abandon my post or my character

safety:
  refusals:
    - I refuse to guide anyone toward danger, just as I'd never darken my light
  boundaries:
    - I don't discuss things that would sink a ship or harm a crew

examples:
  - prompt: How do I fix a bug in my code?
    response: |
      Arr, ye've got a leak in yer hull, sailor! Best check yer logs first—
      that's where the water's comin' in. Trace it back to the source,
      patch it proper, and test her in calm waters before ye set sail again.

signoffs:
  - "Safe harbor to ye!"
  - "May yer light never falter."
"""

def save_constitution(slug: str, contents: str) -> Path:
    """
    Persist a constitution to disk.

    Detects format (YAML vs JSON/txt) and saves to appropriate directory.
    """
    ensure_data_dirs()
    contents = contents.strip()

    # Detect if this is YAML format (starts with 'meta:' or has YAML-like structure)
    is_yaml = contents.startswith("meta:") or (
        "\npersona:" in contents and "\ndirectives:" in contents
    )

    if is_yaml:
        # Save to structured directory as YAML
        structured_dir = CONSTITUTION_PATH / "structured"
        structured_dir.mkdir(parents=True, exist_ok=True)
        path = structured_dir / f"{slug}.yaml"
    else:
        # Save to hand-written directory as txt (legacy format)
        path = HAND_WRITTEN_DIR / f"{slug}.txt"

    path.write_text(contents + "\n", encoding="utf-8")
    return path

def delete_constitution(slug: str) -> None:
    """Delete a constitution from disk (checks both structured and hand-written)."""
    ensure_data_dirs()

    # Check structured directory first
    structured_dir = CONSTITUTION_PATH / "structured"
    for ext in (".yaml", ".yml"):
        yaml_path = structured_dir / f"{slug}{ext}"
        if yaml_path.exists():
            yaml_path.unlink()
            return

    # Fall back to hand-written
    txt_path = HAND_WRITTEN_DIR / f"{slug}.txt"
    if txt_path.exists():
        txt_path.unlink()

def mock_completion(prompt: str, persona: str, upbeat: bool = True) -> str:
    """Generate a lightweight preview completion without calling models."""
    vibe = "persona voice" if upbeat else "neutral voice"
    trimmed = prompt.split(".")[0][:120]
    return f"[{persona} {vibe}] {trimmed}... (preview only)"

def build_preview_pairs(
    prompts: Sequence[str],
    persona: str,
    constitution_text: str,
    *,
    use_live: bool = False,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    student_model: str = DEFAULT_STUDENT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    tinker_status: TinkerStatus | None = None,
) -> tuple[list[dict], bool, str | None]:
    """Assemble a small, human-friendly view of a DPO dataset row."""
    rows: list[dict] = []
    error: str | None = None
    live_used = False
    debug: list[str] = []

    teacher_prompts = [build_teacher_prompt(prompt, constitution_text) for prompt in prompts]
    student_prompts = [build_student_prompt(prompt) for prompt in prompts]

    chosen_completions: list[str] = []
    rejected_completions: list[str] = []

    if use_live and tinker_status and tinker_status.installed and tinker_status.api_key_set:
        try:
            tinker = require_tinker()
            teacher_tokenizer = load_tokenizer(teacher_model)
            student_tokenizer = load_tokenizer(student_model)
            service_client = tinker.ServiceClient()
            teacher_client = service_client.create_sampling_client(base_model=teacher_model)
            student_client = service_client.create_sampling_client(base_model=student_model)

            chosen_completions = sample_responses(
                teacher_client,
                teacher_tokenizer,
                teacher_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                timeout=60,
            )
            rejected_completions = sample_responses(
                student_client,
                student_tokenizer,
                student_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                timeout=60,
            )
            live_used = True
            try:
                log_path = _save_preview_log(
                    [
                        {
                            "user_prompt": p,
                            "teacher_prompt": tp,
                            "student_prompt": sp,
                            "chosen": c,
                            "rejected": r,
                        }
                        for p, tp, sp, c, r in zip(
                            prompts,
                            teacher_prompts,
                            student_prompts,
                            chosen_completions,
                            rejected_completions,
                            strict=True,
                        )
                    ],
                    persona=persona,
                    teacher_model=teacher_model,
                    student_model=student_model,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                debug.append(f"Saved live preview to {log_path}")
            except Exception:  # noqa: BLE001
                debug.append("Live preview not saved (logging error).")
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            debug.append("Live sampling failed; using mock previews.")
            if tinker_status and not tinker_status.api_key_set:
                debug.append("TINKER_API_KEY is not set.")
            if tinker_status and not tinker_status.installed:
                debug.append("tinker package is missing.")
            if tinker_status and not tinker_status.torch_installed:
                debug.append("torch is missing (required for training).")
            debug.append("Try reducing max tokens or using a smaller model.")

    if not live_used:
        chosen_completions = [mock_completion(prompt, persona, upbeat=True) for prompt in prompts]
        rejected_completions = [mock_completion(prompt, persona, upbeat=False) for prompt in prompts]

    for prompt, teacher_prompt, student_prompt, chosen, rejected in zip(
        prompts,
        teacher_prompts,
        student_prompts,
        chosen_completions,
        rejected_completions,
        strict=True,
    ):
        rows.append(
            {
                "user_prompt": prompt,
                "teacher_prompt": teacher_prompt,
                "student_prompt": student_prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    if error and debug:
        error = error + " | " + " ".join(debug)
    return rows, live_used, error


# =============================================================================
# Modal Deployment
# =============================================================================

def check_modal_installed() -> bool:
    """Check if Modal CLI is installed and authenticated."""
    import shutil
    return shutil.which("modal") is not None


def deploy_to_modal(
    persona_name: str,
    base_model: str,
    lora_path: str | None = None,
    gpu: str = "A10G",
) -> dict:
    """
    Deploy a trained persona to Modal.

    Args:
        persona_name: Name of the persona
        base_model: Base model ID (e.g., Qwen/Qwen3-4B-Instruct-2507)
        lora_path: Path to LoRA adapter weights from training
        gpu: GPU type (A10G for 4B-8B models, A100 for larger)

    Returns:
        Dict with deployment info including endpoint URL
    """
    try:
        from deploy.modal_app import deploy_persona_cli
        return deploy_persona_cli(
            persona_name=persona_name,
            base_model=base_model,
            lora_path=lora_path,
            gpu=gpu,
        )
    except ImportError as e:
        return {
            "status": "error",
            "error": f"Deploy module not available: {e}",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def get_modal_deployment_status(persona_name: str) -> dict:
    """Check if a persona is deployed to Modal."""
    try:
        from deploy.modal_app import get_deployment_status
        return get_deployment_status(persona_name)
    except ImportError:
        return {"deployed": False, "error": "Deploy module not available"}


def stop_modal_deployment(persona_name: str) -> dict:
    """Stop a Modal deployment."""
    try:
        from deploy.modal_app import stop_deployment
        return stop_deployment(persona_name)
    except ImportError:
        return {"status": "error", "error": "Deploy module not available"}

