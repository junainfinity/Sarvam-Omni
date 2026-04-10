#!/usr/bin/env python3
"""Pre-compute and cache ViT features for all training images.

Since the ViT is frozen in all stages, we can encode every image once,
save the 4096-dim features to disk, and skip the ViT entirely during training.

Benefits:
  - Saves ~1.15GB GPU memory (no ViT loaded during training)
  - 2-4x training speedup (no ViT forward pass)
  - Enables larger batch sizes

Storage: Each image produces [num_patches, 4096] in FP16.
  For 672x672: ~441 patches × 4096 × 2 bytes = ~3.6MB per image
  For 1.2M images: ~4.3TB (too much!)

  → Use 448x448: ~196 patches × 4096 × 2 bytes = ~1.6MB per image
  → For 1.2M images: ~1.9TB (still large)

  → Better approach: cache features as memory-mapped numpy arrays
  → Or use streaming cache (process + train in same pass, cache for 2nd epoch)

This script does a ONE-PASS cache: encodes each image as it streams from HF,
saves features to a sharded numpy format on disk.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VISION_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-vl-vit")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Lin-Chen/ShareGPT4V",
                        help="HuggingFace dataset ID")
    parser.add_argument("--config", default="ShareGPT4V-PT",
                        help="Dataset config name")
    parser.add_argument("--output-dir", default="cache/vision_features",
                        help="Output directory for cached features")
    parser.add_argument("--image-size", type=int, default=672,
                        help="Target image resolution")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to cache (0=all)")
    parser.add_argument("--shard-size", type=int, default=1000,
                        help="Samples per shard file")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    return parser.parse_args()


def load_vision_encoder(device, dtype):
    """Load just the ViT (no LLM needed for caching)."""
    from safetensors.torch import load_file
    from transformers import Qwen3VLConfig, Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from sarvam_omni.vision_encoder import VisionEncoderWrapper

    with open(os.path.join(VISION_DIR, "vision_config.json")) as f:
        vc_dict = json.load(f)
    full_config = Qwen3VLConfig(vision_config=vc_dict)
    vc = full_config.vision_config

    vision_model = Qwen3VLVisionModel(vc)
    weights = load_file(os.path.join(VISION_DIR, "vision_encoder.safetensors"))
    clean_weights = {k.replace("visual.", "", 1): v for k, v in weights.items()}
    vision_model.load_state_dict(clean_weights, strict=False)
    vision_model = vision_model.to(dtype=dtype, device=device).eval()
    for p in vision_model.parameters():
        p.requires_grad = False

    wrapper = VisionEncoderWrapper(vision_model, vc)

    # Image processor
    with open(os.path.join(VISION_DIR, "preprocessor_config.json")) as f:
        preproc = json.load(f)
    image_processor = Qwen2VLImageProcessor(**preproc)

    return wrapper, vc, image_processor


def process_image(image, image_processor, vision_wrapper, target_size, device, dtype):
    """Encode a single image through the ViT. Returns numpy features."""
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        return None, None

    image = image.convert("RGB")

    # Resize to target
    patch_size = 16
    spatial_merge = 2
    effective_patch = patch_size * spatial_merge
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = max(effective_patch, round(w * scale / effective_patch) * effective_patch)
    new_h = max(effective_patch, round(h * scale / effective_patch) * effective_patch)
    image = image.resize((new_w, new_h), PILImage.BICUBIC)

    # Process
    processed = image_processor(images=[image], return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device=device, dtype=dtype)
    grid_thw = processed.get("image_grid_thw", None)
    if grid_thw is None:
        t = 1
        h_patches = new_h // patch_size
        w_patches = new_w // patch_size
        grid_thw = torch.tensor([[t, h_patches, w_patches]], dtype=torch.long)
    grid_thw = grid_thw.to(device)

    with torch.no_grad():
        features = vision_wrapper(pixel_values, grid_thw=grid_thw)

    # Convert to numpy FP16 for storage
    features_np = features.cpu().to(torch.float16).numpy()
    num_patches = features_np.shape[0]

    return features_np, num_patches


def main():
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = args.device

    output_dir = os.path.join(os.path.dirname(__file__), "..", args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Vision Feature Caching")
    print(f"  Dataset: {args.dataset}/{args.config}")
    print(f"  Image size: {args.image_size}")
    print(f"  Output: {output_dir}")
    print()

    # Load ViT only (~1.15GB)
    print("Loading vision encoder...")
    vision_wrapper, vc, image_processor = load_vision_encoder(device, dtype)
    print("  ViT loaded.")

    # Stream dataset
    from datasets import load_dataset

    print(f"Streaming {args.dataset}/{args.config}...")
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)

    shard_features = []
    shard_metadata = []
    shard_idx = 0
    total_cached = 0
    t0 = time.time()

    for i, item in enumerate(ds):
        if args.max_samples > 0 and total_cached >= args.max_samples:
            break

        try:
            image = item.get("image")
            if image is None:
                continue

            features, num_patches = process_image(
                image, image_processor, vision_wrapper,
                args.image_size, device, dtype,
            )

            if features is None:
                continue

            # Extract text data
            conversations = item.get("conversations", [])
            human_text = ""
            assistant_text = ""
            for conv in conversations:
                if conv.get("from") == "human":
                    human_text = conv["value"].replace("<image>", "").strip()
                elif conv.get("from") == "gpt":
                    assistant_text = conv["value"].strip()

            if not assistant_text:
                continue

            shard_features.append(features)
            shard_metadata.append({
                "id": item.get("id", str(i)),
                "num_patches": num_patches,
                "human": human_text,
                "assistant": assistant_text,
            })
            total_cached += 1

            # Save shard
            if len(shard_features) >= args.shard_size:
                shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npz")
                meta_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.json")

                np.savez_compressed(shard_path, *shard_features)
                with open(meta_path, "w") as f:
                    json.dump(shard_metadata, f)

                elapsed = time.time() - t0
                rate = total_cached / elapsed
                print(f"  Shard {shard_idx}: {len(shard_features)} samples | "
                      f"Total: {total_cached} | {rate:.1f} samples/s")

                shard_features = []
                shard_metadata = []
                shard_idx += 1

        except Exception as e:
            if total_cached % 1000 == 0:
                print(f"  Skip {i}: {e}")
            continue

    # Save final shard
    if shard_features:
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.npz")
        meta_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.json")
        np.savez_compressed(shard_path, *shard_features)
        with open(meta_path, "w") as f:
            json.dump(shard_metadata, f)
        shard_idx += 1

    elapsed = time.time() - t0
    print(f"\nDone! Cached {total_cached} samples in {shard_idx} shards")
    print(f"Time: {elapsed/60:.1f} minutes ({total_cached/elapsed:.1f} samples/s)")
    print(f"Output: {output_dir}")

    # Save manifest
    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "total_samples": total_cached,
        "num_shards": shard_idx,
        "image_size": args.image_size,
        "vision_dim": 4096,
        "shard_size": args.shard_size,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
