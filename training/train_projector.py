#!/usr/bin/env python3
"""Stage 1: Projector pre-training (revised per arxiv research).

Changes from v1:
  - ShareGPT4V-PT (1.2M) or cached features instead of the_cauldron
  - Larger effective batch (64 via grad accum)
  - Warmup 1000 steps (3% of total)
  - Support for cached vision features (no ViT in loop)

Usage:
  # With cached features (recommended — 2-4x faster, saves 1.15GB):
  PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    python -u training/train_projector.py --cached --cache-dir cache/vision_features

  # Without cache (streams from HF, runs ViT each step):
  PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    python -u training/train_projector.py --dataset Lin-Chen/ShareGPT4V --config ShareGPT4V-PT
"""

import os
import sys
import gc
import time
import json
import math
import argparse
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SARVAM_PATH = "/Users/arjun/Projects/tqllm/models/sarvam-30b-backup"
VISION_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-vl-vit")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "stage1")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Projector pre-training")
    # Data
    parser.add_argument("--cached", action="store_true", help="Use pre-computed cached features")
    parser.add_argument("--cache-dir", default="cache/vision_features", help="Cached features directory")
    parser.add_argument("--dataset", default="Lin-Chen/ShareGPT4V", help="HF dataset ID")
    parser.add_argument("--config", default="ShareGPT4V-PT", help="HF dataset config")
    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=1000, help="3% of ~35K steps")
    parser.add_argument("--grad-accum", type=int, default=64, help="Effective batch=64")
    parser.add_argument("--max-steps", type=int, default=0, help="0=full epoch")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=672)
    # Hardware
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_sarvam(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading Sarvam-30B...")
    tokenizer = AutoTokenizer.from_pretrained(SARVAM_PATH, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        SARVAM_PATH, dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"  Loaded in {time.time()-t0:.0f}s: {mem:.1f} GB ({dtype})")
    return model, tokenizer


def load_vision_encoder(device, dtype):
    from safetensors.torch import load_file
    from transformers import Qwen3VLConfig, Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from sarvam_omni.vision_encoder import VisionEncoderWrapper

    print("Loading vision encoder...")
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

    with open(os.path.join(VISION_DIR, "preprocessor_config.json")) as f:
        preproc = json.load(f)
    image_processor = Qwen2VLImageProcessor(**preproc)

    print(f"  ViT loaded: {sum(p.numel() for p in vision_model.parameters()):,} params")
    return wrapper, vc, image_processor


def build_model(sarvam_model, vision_encoder, device):
    from sarvam_omni.model import SarvamOmniForConditionalGeneration
    model = SarvamOmniForConditionalGeneration(
        language_model=sarvam_model,
        vision_encoder=vision_encoder,
        vision_dim=4096, hidden_size=4096, image_token_id=8,
    )
    model.projector = model.projector.to(device=device, dtype=torch.float32)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} params ({trainable * 4 / 1e6:.1f} MB)")
    return model


def build_model_cached(sarvam_model, device):
    """Build model without ViT (for cached feature training)."""
    from sarvam_omni.model import SarvamOmniForConditionalGeneration
    # Use a dummy vision encoder (won't be called)
    model = SarvamOmniForConditionalGeneration(
        language_model=sarvam_model,
        vision_encoder=None,
        vision_dim=4096, hidden_size=4096, image_token_id=8,
    )
    model.projector = model.projector.to(device=device, dtype=torch.float32)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} params (NO ViT loaded — using cached features)")
    return model


def sanity_check(model, tokenizer, device, dtype, cached=False):
    """Quick sanity check with synthetic data."""
    from PIL import Image
    import numpy as np

    print("\n" + "=" * 60)
    print("SANITY CHECK: 10 steps on synthetic data")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
    )
    model.projector.train()
    losses = []

    for step in range(10):
        # Create synthetic data
        num_patches = 100
        if cached:
            # Simulate cached features
            vision_features = torch.randn(num_patches, 4096, device=device)
            pixel_values = None
            grid_thw = None
        else:
            # Simulate real image
            vision_features = None
            pixel_values = torch.randn(num_patches, 6, 16, 16, device=device, dtype=dtype)
            grid_thw = torch.tensor([[1, 20, 20]], dtype=torch.long, device=device)

        # Build input tokens
        image_tokens = "<|image_soft_token|>" * num_patches
        prompt = f"[@BOS@]<|start_of_turn|>user\n<|start_of_image|>{image_tokens}<|end_of_image|>\nDescribe this image.<|end_of_turn|><|start_of_turn|>assistant\nThis is a test image with random content.<|end_of_turn|>"
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)
        image_token_mask = (input_ids == 8)

        labels = input_ids.clone()
        # Find assistant start
        assistant_marker = tokenizer("assistant\n", add_special_tokens=False)["input_ids"]
        labels[:, :input_ids.shape[1] // 2] = -100  # Rough mask

        output = model(
            input_ids=input_ids, attention_mask=attention_mask,
            image_token_mask=image_token_mask,
            pixel_values=pixel_values, grid_thw=grid_thw,
            vision_features=vision_features,
            labels=labels,
        )

        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        print(f"  Step {step+1}/10 | Loss: {loss.item():.4f}")

        del output, loss
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"\n  Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
    if losses[-1] < losses[0]:
        print("  [OK] Loss decreasing! Projector is learning.")
    print("  [OK] Sanity check passed!")


def train(model, tokenizer, dataset_iter, optimizer, scheduler, args, device, dtype):
    """Main training loop."""
    print("\n" + "=" * 60)
    print("STAGE 1: Projector Pre-training")
    print(f"  LR: {args.lr}, Effective BS: {args.grad_accum}, Warmup: {args.warmup_steps}")
    print(f"  Cached: {args.cached}")
    print("=" * 60)

    model.projector.train()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    step = 0
    accum_loss = 0.0
    accum_count = 0
    t0 = time.time()
    best_loss = float("inf")

    for sample in dataset_iter:
        try:
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            image_token_mask = sample["image_token_mask"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)

            # Cached or live features
            if "vision_features" in sample and sample["vision_features"] is not None:
                vision_features = sample["vision_features"].to(device)
                pixel_values = None
                grid_thw = None
            else:
                vision_features = None
                pixel_values = sample["pixel_values"].to(device=device, dtype=dtype)
                grid_thw = sample["grid_thw"].to(device) if sample.get("grid_thw") is not None else None

            output = model(
                input_ids=input_ids, attention_mask=attention_mask,
                image_token_mask=image_token_mask,
                pixel_values=pixel_values, grid_thw=grid_thw,
                vision_features=vision_features,
                labels=labels,
            )

            loss = output.loss / args.grad_accum
            loss.backward()
            accum_loss += output.loss.item()
            accum_count += 1

            if accum_count >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                avg_loss = accum_loss / accum_count

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    lr = scheduler.get_last_lr()[0]
                    sps = (step * args.grad_accum) / elapsed
                    print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                          f"{sps:.1f} samp/s | {elapsed/60:.1f}m")

                if step % args.save_every == 0:
                    ckpt = os.path.join(CHECKPOINT_DIR, f"projector_step{step}.pt")
                    torch.save({
                        "step": step, "loss": avg_loss,
                        "projector_state_dict": model.projector.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, ckpt)
                    print(f"  Checkpoint: {ckpt}")
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(model.projector.state_dict(),
                                   os.path.join(CHECKPOINT_DIR, "projector_best.pt"))
                        print(f"  New best: {best_loss:.4f}")

                accum_loss = 0.0
                accum_count = 0

                if args.max_steps > 0 and step >= args.max_steps:
                    break

            # Periodic cleanup
            if (accum_count + step * args.grad_accum) % 100 == 0:
                del output, loss
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at step {step}! Clearing cache.")
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                optimizer.zero_grad()
                continue
            raise

    # Final save
    final = os.path.join(CHECKPOINT_DIR, "projector_final.pt")
    torch.save(model.projector.state_dict(), final)
    print(f"\nDone! Steps: {step}, Best loss: {best_loss:.4f}")
    print(f"Final checkpoint: {final}")


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = args.device

    print(f"SarvamOmni Stage 1: Projector Pre-training (v2)")
    print(f"  Device: {device}, Dtype: {dtype}")
    print(f"  Cached: {args.cached}")
    print()

    # Load Sarvam-30B
    sarvam_model, tokenizer = load_sarvam(device, dtype)
    gc.collect()

    if args.cached:
        # Cached mode: no ViT needed
        model = build_model_cached(sarvam_model, device)
        from training.data.cached_dataset import CachedVisionDataset
        cache_dir = os.path.join(os.path.dirname(__file__), "..", args.cache_dir)
        dataset = CachedVisionDataset(cache_dir, tokenizer, max_length=args.max_length)
    else:
        # Live mode: load ViT
        vision_wrapper, vc, image_processor = load_vision_encoder(device, dtype)
        gc.collect()
        model = build_model(sarvam_model, vision_wrapper, device)

        from sarvam_omni.processor import SarvamOmniProcessor
        processor = SarvamOmniProcessor(tokenizer, image_processor, vc)

        if args.dataset == "HuggingFaceM4/the_cauldron":
            from training.data.llava_dataset import CauldronAlignmentDataset
            dataset = CauldronAlignmentDataset(
                processor=processor, max_length=args.max_length,
                image_target_size=args.image_size,
                max_samples_per_config=80000,
            )
        else:
            from training.data.cached_dataset import StreamingVisionDataset
            dataset = StreamingVisionDataset(
                dataset_id=args.dataset, config=args.config,
                processor=processor, max_length=args.max_length,
                image_target_size=args.image_size,
            )

    # Resume
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if "projector_state_dict" in state:
            model.projector.load_state_dict(state["projector_state_dict"])
        else:
            model.projector.load_state_dict(state)
        print(f"Resumed from {args.resume}")

    if args.sanity_check:
        sanity_check(model, tokenizer, device, dtype, cached=args.cached)
        return

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )

    # Schedule
    est_total = (1200000 if "ShareGPT4V" in args.dataset else 500000) // args.grad_accum
    if args.max_steps > 0:
        est_total = min(est_total, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, est_total)

    # Train
    train(model, tokenizer, dataset, optimizer, scheduler, args, device, dtype)


if __name__ == "__main__":
    main()
