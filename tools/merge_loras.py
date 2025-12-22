#!/usr/bin/env python3
"""
Linear LoRA adapter merging tool for Open Character Training.

Stage 4 from "Open Character Training" paper:
- Linearly merge adapters from distillation (DPO) and introspection (SFT) stages
- Release/save the merged adapter
- Paper uses weights: DPO=1.0, SFT=0.25 (DPO dominant, SFT adds introspective depth)

Usage:
    python tools/merge_loras.py \
        --adapters path/to/dpo_adapter path/to/sft_adapter \
        --weights 1.0 0.25 \
        --output merged_adapter/

    # Or with Tinker paths (paper defaults):
    python tools/merge_loras.py \
        --adapters tinker://xxx/dpo-sampler tinker://yyy/sft-sampler \
        --output merged_character_adapter/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    safe_open = None
    save_file = None


def require_dependencies():
    """Ensure torch and safetensors are available."""
    if torch is None:
        raise ImportError("Install torch: pip install torch")
    if safe_open is None or save_file is None:
        raise ImportError("Install safetensors: pip install safetensors")


def load_adapter_weights(adapter_path: str) -> Dict[str, torch.Tensor]:
    """
    Load LoRA adapter weights from a directory or Tinker path.
    
    Supports:
    - Local directory with adapter_model.safetensors
    - Local directory with model.safetensors (Ollama format)
    - Tinker paths (tinker://...) - downloads first
    """
    require_dependencies()
    
    # Handle Tinker paths
    if adapter_path.startswith("tinker://"):
        adapter_path = download_tinker_checkpoint(adapter_path)
    
    path = Path(adapter_path)
    
    # Find the safetensors file
    safetensors_candidates = [
        path / "adapter_model.safetensors",
        path / "model.safetensors",
        path,  # In case path is the file itself
    ]
    
    safetensors_file = None
    for candidate in safetensors_candidates:
        if candidate.exists() and candidate.is_file():
            safetensors_file = candidate
            break
    
    if safetensors_file is None:
        raise FileNotFoundError(
            f"No safetensors file found in {adapter_path}. "
            "Expected adapter_model.safetensors or model.safetensors"
        )
    
    # Load weights
    weights = {}
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    
    print(f"Loaded {len(weights)} tensors from {safetensors_file}")
    return weights


def download_tinker_checkpoint(tinker_path: str) -> str:
    """Download a Tinker checkpoint to a local directory."""
    try:
        import tinker
        import tempfile
        import urllib.request
        import tarfile
    except ImportError as e:
        raise ImportError(f"Install tinker SDK to download Tinker checkpoints: {e}")
    
    print(f"Downloading Tinker checkpoint: {tinker_path}")
    
    sc = tinker.ServiceClient()
    rc = sc.create_rest_client()
    
    response = rc.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()
    
    # Download to temp directory
    temp_dir = tempfile.mkdtemp(prefix="tinker_adapter_")
    archive_path = Path(temp_dir) / "archive.tar"
    
    print(f"Downloading to {archive_path}...")
    urllib.request.urlretrieve(response.url, archive_path)
    
    # Extract
    extract_dir = Path(temp_dir) / "extracted"
    extract_dir.mkdir()
    
    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(extract_dir)
    
    print(f"Extracted to {extract_dir}")
    return str(extract_dir)


def linear_merge_adapters(
    adapter_weights_list: List[Dict[str, torch.Tensor]],
    merge_weights: List[float],
) -> Dict[str, torch.Tensor]:
    """
    Linearly merge multiple LoRA adapters.
    
    merged = sum(weight_i * adapter_i) for each adapter
    
    This is the approach specified in the paper for Stage 4.
    """
    require_dependencies()
    
    if len(adapter_weights_list) != len(merge_weights):
        raise ValueError(
            f"Number of adapters ({len(adapter_weights_list)}) must match "
            f"number of weights ({len(merge_weights)})"
        )
    
    # Normalize weights
    total_weight = sum(merge_weights)
    if abs(total_weight - 1.0) > 0.01:
        print(f"Warning: weights sum to {total_weight}, normalizing to 1.0")
        merge_weights = [w / total_weight for w in merge_weights]
    
    # Get all unique keys
    all_keys = set()
    for weights in adapter_weights_list:
        all_keys.update(weights.keys())
    
    # Merge each tensor
    merged = {}
    for key in all_keys:
        tensors = []
        weights_for_key = []
        
        for adapter_weights, weight in zip(adapter_weights_list, merge_weights):
            if key in adapter_weights:
                tensors.append(adapter_weights[key])
                weights_for_key.append(weight)
        
        if len(tensors) == 1:
            # Only one adapter has this key
            merged[key] = tensors[0] * weights_for_key[0]
        else:
            # Linear combination
            # Renormalize weights for keys that aren't in all adapters
            key_total = sum(weights_for_key)
            normalized_weights = [w / key_total for w in weights_for_key]
            
            merged[key] = sum(
                w * t for w, t in zip(normalized_weights, tensors)
            )
    
    print(f"Merged {len(merged)} tensors")
    return merged


def save_merged_adapter(
    merged_weights: Dict[str, torch.Tensor],
    output_path: str,
    source_config_path: Optional[str] = None,
) -> Path:
    """
    Save merged adapter weights to a directory.
    
    Creates:
    - adapter_model.safetensors (merged weights)
    - adapter_config.json (copied from source or generated)
    """
    require_dependencies()
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    weights_path = output_dir / "adapter_model.safetensors"
    save_file(merged_weights, weights_path)
    print(f"Saved merged weights to {weights_path}")
    
    # Copy or create config
    config_path = output_dir / "adapter_config.json"
    if source_config_path:
        source_config = Path(source_config_path)
        if source_config.exists():
            shutil.copy(source_config, config_path)
            print(f"Copied config from {source_config}")
        else:
            # Try to find config in source directory
            for candidate in [
                Path(source_config_path).parent / "adapter_config.json",
                Path(source_config_path) / "adapter_config.json",
            ]:
                if candidate.exists():
                    shutil.copy(candidate, config_path)
                    print(f"Copied config from {candidate}")
                    break
    
    if not config_path.exists():
        # Create minimal config
        minimal_config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "inference_mode": True,
            "r": 64,  # Paper default
            "lora_alpha": 128,  # Paper default
            "lora_dropout": 0.0,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "bias": "none",
            "_merged_from": "linear_merge",
        }
        with open(config_path, "w") as f:
            json.dump(minimal_config, f, indent=2)
        print(f"Created minimal config at {config_path}")
    
    # Create completion marker
    (output_dir / "checkpoint_complete").touch()
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Linearly merge LoRA adapters (Stage 4 of Open Character Training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Merge DPO + SFT with paper defaults (1.0/0.25):
    python tools/merge_loras.py \\
        --adapters ./dpo_adapter ./sft_adapter \\
        --output ./merged_adapter

    # Merge with custom weights:
    python tools/merge_loras.py \\
        --adapters ./dpo_adapter ./sft_adapter \\
        --weights 0.6 0.4 \\
        --output ./merged_adapter

    # Merge Tinker checkpoints:
    python tools/merge_loras.py \\
        --adapters tinker://xxx/dpo-sampler tinker://yyy/sft-sampler \\
        --output ./merged_adapter

Paper Reference:
    Stage 4 from "Open Character Training" by Maiya et al.
    - Linearly merge adapters from distillation and introspection stages
    - Paper uses weights: DPO=1.0, SFT=0.25
        """,
    )

    parser.add_argument(
        "--adapters",
        nargs="+",
        required=True,
        help="Paths to adapter directories (local or tinker://...). For DPO+SFT, list DPO first.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        help="Merge weights for each adapter (default for 2 adapters: 1.0 0.25 per paper)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged adapter",
    )
    parser.add_argument(
        "--config-from",
        help="Copy adapter_config.json from this adapter (default: first adapter)",
    )
    
    args = parser.parse_args()

    # Default weights: paper uses 1.0/0.25 for DPO/SFT
    if args.weights is None:
        if len(args.adapters) == 2:
            # Paper defaults: DPO=1.0, SFT=0.25
            args.weights = [1.0, 0.25]
        else:
            # Equal weights for other cases
            args.weights = [1.0 / len(args.adapters)] * len(args.adapters)
    
    if len(args.weights) != len(args.adapters):
        parser.error(
            f"Number of weights ({len(args.weights)}) must match "
            f"number of adapters ({len(args.adapters)})"
        )
    
    print(f"Merging {len(args.adapters)} adapters:")
    for adapter, weight in zip(args.adapters, args.weights):
        print(f"  {weight:.2%}: {adapter}")
    
    # Load all adapters
    adapter_weights_list = []
    first_adapter_path = None
    
    for adapter_path in args.adapters:
        weights = load_adapter_weights(adapter_path)
        adapter_weights_list.append(weights)
        if first_adapter_path is None:
            first_adapter_path = adapter_path
    
    # Merge
    merged = linear_merge_adapters(adapter_weights_list, args.weights)
    
    # Save
    config_source = args.config_from or first_adapter_path
    output_path = save_merged_adapter(merged, args.output, config_source)
    
    print(f"\n✅ Merged adapter saved to: {output_path}")
    print(f"   - adapter_model.safetensors ({len(merged)} tensors)")
    print("   - adapter_config.json")


if __name__ == "__main__":
    main()

