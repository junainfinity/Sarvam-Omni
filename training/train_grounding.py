#!/usr/bin/env python3
"""Stage 2: Merged Agentic Fine-tuning (grounding + reasoning + text).

Combines old Stages 2+3 into single mixed-data stage per research:
  - 40% GUI grounding (AGUVIS Stage-1)
  - 40% agentic reasoning (AGUVIS Stage-2)
  - 20% text-only (VILA's insight: prevents LLM degradation)

Continues from Stage 1.5 checkpoint (LoRA + projector).

Usage:
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python training/train_grounding.py \
    --stage15-dir checkpoints/stage1_5/best
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
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "stage2")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Merged Agentic Fine-tuning")
    parser.add_argument("--projector-path", type=str, default=None,
                        help="Direct path to projector checkpoint")
    parser.add_argument("--stage15-dir", type=str, default=None,
                        help="Stage 1.5 checkpoint dir (contains lora/ and projector.pt)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (lower for stage 2)")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=672)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit dataset samples (0=all)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()
    if args.stage15_dir is None and args.projector_path is None:
        parser.error("Must specify --stage15-dir or --projector-path")
    return args


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = args.device

    print(f"SarvamOmni Stage 2: GUI Grounding with LoRA")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  LR: {args.lr}, Grad accum: {args.grad_accum}")
    print()

    # ── Load components (same as Stage 1) ──
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3VLConfig, Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from sarvam_omni.vision_encoder import VisionEncoderWrapper
    from sarvam_omni.model import SarvamOmniForConditionalGeneration
    from sarvam_omni.processor import SarvamOmniProcessor

    # Vision encoder
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

    # Sarvam-30B
    print("Loading Sarvam-30B...")
    tokenizer = AutoTokenizer.from_pretrained(SARVAM_PATH, trust_remote_code=True)
    sarvam = AutoModelForCausalLM.from_pretrained(
        SARVAM_PATH, dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map=device,
    )

    # ── Apply/Load LoRA ──
    from peft import get_peft_model, PeftModel
    from training.lora_config import get_lora_config, print_lora_info

    if args.stage15_dir:
        # Load LoRA from Stage 1.5
        lora_path = os.path.join(args.stage15_dir, "lora")
        proj_path = os.path.join(args.stage15_dir, "projector.pt")
        print(f"Loading Stage 1.5 LoRA from {lora_path}...")
        sarvam = PeftModel.from_pretrained(sarvam, lora_path, is_trainable=True)
    else:
        # Fresh LoRA
        print("Applying fresh LoRA adapters...")
        lora_config = get_lora_config(r=args.lora_r, lora_alpha=args.lora_alpha)
        sarvam = get_peft_model(sarvam, lora_config)
        proj_path = args.projector_path

    print_lora_info(sarvam)

    # Verify router frozen
    print("\nVerifying router weights are frozen:")
    router_frozen = all(
        not p.requires_grad for n, p in sarvam.named_parameters() if "gate.weight" in n
    )
    print(f"  {'[OK]' if router_frozen else '[WARN]'} Router weights {'frozen' if router_frozen else 'NOT frozen!'}")

    # ── Assemble model ──
    model = SarvamOmniForConditionalGeneration(
        language_model=sarvam,
        vision_encoder=vision_wrapper,
        vision_dim=4096, hidden_size=4096, image_token_id=8,
    )

    # Load projector
    print(f"Loading projector from {proj_path}...")
    state = torch.load(proj_path, map_location=device, weights_only=True)
    if "projector_state_dict" in state:
        model.projector.load_state_dict(state["projector_state_dict"])
    else:
        model.projector.load_state_dict(state)
    model.projector = model.projector.to(device=device, dtype=torch.float32)

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable: {trainable:,} (projector + LoRA)")

    # ── Processor ──
    with open(os.path.join(VISION_DIR, "preprocessor_config.json")) as f:
        preproc = json.load(f)
    image_processor = Qwen2VLImageProcessor(**preproc)
    processor = SarvamOmniProcessor(tokenizer, image_processor, vc)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    import math
    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ──
    from training.data.aguvis_dataset import AGUVISGroundingDataset

    print(f"\nStarting Stage 2 training (max {args.max_steps} steps)...")
    dataset = AGUVISGroundingDataset(
        processor=processor,
        max_length=args.max_length,
        image_target_size=args.image_size,
        max_samples=args.max_samples,
    )

    model.projector.train()
    sarvam.train()  # Enable LoRA dropout

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
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_token_mask=image_token_mask,
                pixel_values=pixel_values,
                grid_thw=grid_thw,
                labels=labels,
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
                    # Save LoRA + projector
                    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"step_{step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    sarvam.save_pretrained(os.path.join(ckpt_dir, "lora"))
                    torch.save(model.projector.state_dict(),
                               os.path.join(ckpt_dir, "projector.pt"))
                    print(f"  Saved checkpoint: {ckpt_dir}")

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
                print(f"  OOM at step {step}! Skipping.")
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                optimizer.zero_grad()
                continue
            raise

    # Final save
    final_dir = os.path.join(CHECKPOINT_DIR, "final")
    os.makedirs(final_dir, exist_ok=True)
    sarvam.save_pretrained(os.path.join(final_dir, "lora"))
    torch.save(model.projector.state_dict(), os.path.join(final_dir, "projector.pt"))
    print(f"\nStage 2 complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
