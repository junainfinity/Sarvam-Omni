#!/usr/bin/env python3
"""Stage 3: Agentic Reasoning Fine-tuning on AGUVIS Stage-2.

Continues from Stage 2 checkpoint (LoRA + projector).
Trains on multi-step agentic trajectories with thought-action format.

Usage:
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python training/train_agentic.py \
    --stage2-dir checkpoints/stage2/best
"""

import os
import sys
import gc
import time
import json
import argparse
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SARVAM_PATH = "/Users/arjun/Projects/tqllm/models/sarvam-30b-backup"
VISION_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-vl-vit")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "stage3")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: Agentic Reasoning")
    parser.add_argument("--stage2-dir", type=str, required=True,
                        help="Path to Stage 2 checkpoint dir (contains lora/ and projector.pt)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (lower than Stage 2)")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=672)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = args.device

    print(f"SarvamOmni Stage 3: Agentic Reasoning Fine-tuning")
    print(f"  Stage 2 checkpoint: {args.stage2_dir}")
    print(f"  LR: {args.lr}, Max steps: {args.max_steps}")
    print()

    # ── Load components ──
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3VLConfig, Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from peft import PeftModel
    from sarvam_omni.vision_encoder import VisionEncoderWrapper
    from sarvam_omni.model import SarvamOmniForConditionalGeneration
    from sarvam_omni.processor import SarvamOmniProcessor

    # Vision encoder (frozen)
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
    vision_wrapper = VisionEncoderWrapper(vision_model, vc)

    # Sarvam-30B + LoRA from Stage 2
    print("Loading Sarvam-30B...")
    tokenizer = AutoTokenizer.from_pretrained(SARVAM_PATH, trust_remote_code=True)
    sarvam = AutoModelForCausalLM.from_pretrained(
        SARVAM_PATH, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map=device,
    )

    lora_path = os.path.join(args.stage2_dir, "lora")
    print(f"Loading Stage 2 LoRA from {lora_path}...")
    sarvam = PeftModel.from_pretrained(sarvam, lora_path, is_trainable=True)

    # Assemble model
    model = SarvamOmniForConditionalGeneration(
        language_model=sarvam,
        vision_encoder=vision_wrapper,
        vision_dim=4096, hidden_size=4096, image_token_id=8,
    )

    # Load projector from Stage 2
    proj_path = os.path.join(args.stage2_dir, "projector.pt")
    print(f"Loading projector from {proj_path}...")
    model.projector.load_state_dict(
        torch.load(proj_path, map_location=device, weights_only=True)
    )
    model.projector = model.projector.to(device=device, dtype=torch.float32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable: {trainable:,}")

    # Processor
    with open(os.path.join(VISION_DIR, "preprocessor_config.json")) as f:
        preproc = json.load(f)
    image_processor = Qwen2VLImageProcessor(**preproc)
    processor = SarvamOmniProcessor(tokenizer, image_processor, vc)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    import math
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Dataset: AGUVIS Stage-2 agentic trajectories ──
    from training.data.aguvis_dataset import AGUVISAgenticDataset

    print(f"\nStarting Stage 3 training...")
    dataset = AGUVISAgenticDataset(
        processor=processor,
        max_length=args.max_length,
        image_target_size=args.image_size,
        max_samples=args.max_samples,
    )

    model.projector.train()
    sarvam.train()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    step = 0
    accum_loss = 0.0
    accum_count = 0
    t0 = time.time()
    best_loss = float("inf")

    for sample in dataset:
        try:
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            image_token_mask = sample["image_token_mask"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)
            pixel_values = sample["pixel_values"].to(device=device, dtype=dtype)
            grid_thw = sample["grid_thw"].to(device) if sample["grid_thw"] is not None else None

            output = model(
                input_ids=input_ids, attention_mask=attention_mask,
                image_token_mask=image_token_mask, pixel_values=pixel_values,
                grid_thw=grid_thw, labels=labels,
            )

            loss = output.loss / args.grad_accum
            loss.backward()
            accum_loss += output.loss.item()
            accum_count += 1

            if accum_count >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                avg_loss = accum_loss / accum_count

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                          f"Elapsed: {elapsed/60:.1f}m")

                if step % args.save_every == 0:
                    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"step_{step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    sarvam.save_pretrained(os.path.join(ckpt_dir, "lora"))
                    torch.save(model.projector.state_dict(),
                               os.path.join(ckpt_dir, "projector.pt"))
                    print(f"  Saved: {ckpt_dir}")

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_dir = os.path.join(CHECKPOINT_DIR, "best")
                        os.makedirs(best_dir, exist_ok=True)
                        sarvam.save_pretrained(os.path.join(best_dir, "lora"))
                        torch.save(model.projector.state_dict(),
                                   os.path.join(best_dir, "projector.pt"))

                accum_loss = 0.0
                accum_count = 0
                if step >= args.max_steps:
                    break

            if (accum_count + step * args.grad_accum) % 50 == 0:
                del output, loss, pixel_values
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM! Skipping.")
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                optimizer.zero_grad()
                continue
            raise

    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    sarvam.save_pretrained(os.path.join(final_dir, "lora"))
    torch.save(model.projector.state_dict(), os.path.join(final_dir, "projector.pt"))
    print(f"\nStage 3 complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
