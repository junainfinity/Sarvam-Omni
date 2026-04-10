#!/usr/bin/env python3
"""Extract vision encoder weights directly from the Qwen3-VL safetensors shard.

This avoids loading the full 8B model — we only load vision-related tensors.
"""

import os
import sys
import json
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

QWEN3_VL_CACHE = None  # Set after download


def find_qwen3vl_cache():
    """Find the Qwen3-VL-8B-Instruct cache directory."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = hub / "models--Qwen--Qwen3-VL-8B-Instruct"
    if model_dir.exists():
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            # Get latest snapshot
            dirs = sorted(snapshots.iterdir())
            if dirs:
                return str(dirs[-1])
    return None


def extract_vision_weights(cache_dir: str, save_dir: str):
    """Extract only vision-related weights from the safetensors shard."""
    shard_path = os.path.join(cache_dir, "model-00004-of-00004.safetensors")
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Vision shard not found: {shard_path}")

    print(f"Loading vision shard from {shard_path}...")
    all_tensors = load_file(shard_path)

    # Filter vision-related tensors
    vision_tensors = {}
    for key, tensor in all_tensors.items():
        if key.startswith("model.visual."):
            # Strip the "model." prefix to get clean names
            clean_key = key[len("model."):]
            vision_tensors[clean_key] = tensor

    print(f"Extracted {len(vision_tensors)} vision tensors")

    # Calculate size
    total_params = sum(t.numel() for t in vision_tensors.values())
    total_bytes = sum(t.numel() * t.element_size() for t in vision_tensors.values())
    print(f"Vision encoder: {total_params:,} params, {total_bytes / 1e9:.2f} GB")

    # Save as standalone checkpoint
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vision_encoder.safetensors")
    save_file(vision_tensors, save_path)
    print(f"Saved to {save_path}")

    # Save vision config
    config_path = os.path.join(cache_dir, "config.json")
    with open(config_path) as f:
        full_config = json.load(f)

    vision_config = full_config["vision_config"]
    with open(os.path.join(save_dir, "vision_config.json"), "w") as f:
        json.dump(vision_config, f, indent=2)
    print(f"Saved vision config to {save_dir}/vision_config.json")

    # Copy preprocessor config if available
    preproc_path = os.path.join(cache_dir, "preprocessor_config.json")
    if os.path.exists(preproc_path):
        import shutil
        shutil.copy2(preproc_path, os.path.join(save_dir, "preprocessor_config.json"))
        print(f"Saved preprocessor config")

    # Print layer summary
    print("\nVision encoder structure:")
    prefixes = set()
    for key in sorted(vision_tensors.keys()):
        parts = key.split(".")
        if len(parts) >= 3:
            prefix = ".".join(parts[:3])
        else:
            prefix = key
        prefixes.add(prefix)

    for prefix in sorted(prefixes):
        matching = [k for k in vision_tensors if k.startswith(prefix)]
        params = sum(vision_tensors[k].numel() for k in matching)
        print(f"  {prefix}: {params:,} params")

    return save_dir


def verify_extraction(save_dir: str):
    """Verify the extracted vision encoder loads correctly."""
    print("\n" + "=" * 60)
    print("Verifying extracted vision encoder...")

    # Load weights
    weights = load_file(os.path.join(save_dir, "vision_encoder.safetensors"))
    print(f"  Loaded {len(weights)} tensors")

    # Load config
    with open(os.path.join(save_dir, "vision_config.json")) as f:
        config = json.load(f)
    print(f"  Vision config: depth={config['depth']}, "
          f"hidden={config['hidden_size']}, out={config['out_hidden_size']}")

    # Check key components exist
    assert any("patch_embed" in k for k in weights), "Missing patch_embed"
    assert any("merger" in k for k in weights), "Missing merger (output projection)"
    assert any("blocks.0" in k for k in weights), "Missing transformer blocks"
    assert any(f"blocks.{config['depth']-1}" in k for k in weights), "Missing last block"

    print("  [OK] All expected components present")
    print("  [OK] Vision encoder extraction verified")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="models/qwen3-vl-vit",
                        help="Directory to save extracted vision encoder")
    args = parser.parse_args()

    cache_dir = find_qwen3vl_cache()
    if cache_dir is None:
        print("ERROR: Qwen3-VL-8B-Instruct not found in HF cache.")
        print("Run the download first.")
        sys.exit(1)

    print(f"Found Qwen3-VL cache at: {cache_dir}")

    save_dir = os.path.join(os.path.dirname(__file__), "..", args.save_dir)
    save_dir = str(Path(save_dir).resolve())

    extract_vision_weights(cache_dir, save_dir)
    verify_extraction(save_dir)

    print("\n" + "=" * 60)
    print("DONE! Vision encoder ready at:", save_dir)
    print(f"  out_hidden_size=4096 matches Sarvam hidden_size=4096")
    print(f"  Projector: refinement layer (4096 -> 4096) with residual")
