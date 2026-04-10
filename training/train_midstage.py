#!/usr/bin/env python3
"""Stage 1.5: Mid-training with LoRA (per LLaVA-OneVision & DeepSeek-VL2).

This is the MOST IMPACTFUL stage — teaches the LLM to actually attend to
vision features. Without it, the LLM ignores projected features beyond
surface-level pattern matching.

Trains:
  - LoRA r=32 on Sarvam attention + shared expert
  - Projector (continues from Stage 1)

Data:
  - High-quality VQA + detailed image descriptions
  - Mixed: 80% visual QA + 20% text-only (VILA's insight: prevents LLM degradation)

Usage:
  PYTHONUNBUFFERED=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
    python -u training/train_midstage.py \
    --projector checkpoints/stage1/projector_best.pt
"""

import os
import sys
import gc
import time
import json
import math
import argparse
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SARVAM_PATH = "/Users/arjun/Projects/tqllm/models/sarvam-30b-backup"
VISION_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-vl-vit")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "stage1_5")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1.5: Mid-training with LoRA")
    parser.add_argument("--projector", type=str, required=True, help="Stage 1 projector path")
    parser.add_argument("--cached", action="store_true", help="Use cached vision features")
    parser.add_argument("--cache-dir", default="cache/vision_features")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=672)
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (32 per research)")
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--text-ratio", type=float, default=0.2, help="Fraction of text-only data")
    parser.add_argument("--dataset", default="HuggingFaceM4/the_cauldron", help="HF dataset")
    parser.add_argument("--config", default="", help="HF dataset config")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


class MixedVisionTextDataset(torch.utils.data.IterableDataset):
    """Mixes vision-language data with text-only data.

    Per VILA's finding: text-only data prevents LLM capability degradation
    when training with vision data. Critical for MoE models where routing
    was learned on text.
    """

    def __init__(self, vl_dataset, tokenizer, text_ratio=0.2, max_length=1024, device="mps"):
        self.vl_dataset = vl_dataset
        self.tokenizer = tokenizer
        self.text_ratio = text_ratio
        self.max_length = max_length

        # Simple text-only prompts for mixing
        self.text_prompts = [
            ("Explain quantum computing in simple terms.", "Quantum computing uses quantum bits or qubits..."),
            ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight..."),
            ("Describe the water cycle.", "The water cycle involves evaporation, condensation, precipitation..."),
            ("What causes earthquakes?", "Earthquakes are caused by the movement of tectonic plates..."),
            ("How does the internet work?", "The internet is a global network of computers connected..."),
        ]

    def _make_text_sample(self):
        """Create a text-only training sample."""
        import random
        q, a = random.choice(self.text_prompts)
        prompt = f"[@BOS@]<|start_of_turn|>user\n{q}<|end_of_turn|><|start_of_turn|>assistant\n{a}<|end_of_turn|>"
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = torch.ones_like(input_ids)
        image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        labels = input_ids.clone()
        # Mask everything before assistant response
        assistant_start = prompt.index("assistant\n") + len("assistant\n")
        assistant_token_start = len(self.tokenizer(prompt[:assistant_start], add_special_tokens=False)["input_ids"])
        labels[:assistant_token_start] = -100

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            image_token_mask = image_token_mask[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_token_mask": image_token_mask,
            "labels": labels,
            "vision_features": None,
            "pixel_values": None,
            "grid_thw": None,
        }

    def __iter__(self):
        import random
        for vl_sample in self.vl_dataset:
            # Yield VL sample
            yield vl_sample

            # With text_ratio probability, also yield a text-only sample
            if random.random() < self.text_ratio / (1 - self.text_ratio):
                yield self._make_text_sample()


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = args.device

    print(f"SarvamOmni Stage 1.5: Mid-training with LoRA")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  LR: {args.lr}, BS: {args.grad_accum}, Text ratio: {args.text_ratio}")
    print()

    # Load Sarvam
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading Sarvam-30B...")
    tokenizer = AutoTokenizer.from_pretrained(SARVAM_PATH, trust_remote_code=True)
    sarvam = AutoModelForCausalLM.from_pretrained(
        SARVAM_PATH, dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map=device,
    )

    # Apply LoRA
    print(f"Applying LoRA r={args.lora_r}...")
    from training.lora_config import get_lora_config, print_lora_info
    from peft import get_peft_model
    lora_config = get_lora_config(r=args.lora_r, lora_alpha=args.lora_alpha)
    sarvam = get_peft_model(sarvam, lora_config)
    print_lora_info(sarvam)

    # Build model (cached or with ViT)
    from sarvam_omni.model import SarvamOmniForConditionalGeneration

    if args.cached:
        model = SarvamOmniForConditionalGeneration(
            language_model=sarvam, vision_encoder=None,
            vision_dim=4096, hidden_size=4096, image_token_id=8,
        )
    else:
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
        vision_wrapper = VisionEncoderWrapper(vision_model, vc)

        model = SarvamOmniForConditionalGeneration(
            language_model=sarvam, vision_encoder=vision_wrapper,
            vision_dim=4096, hidden_size=4096, image_token_id=8,
        )

    # Load Stage 1 projector
    print(f"Loading projector from {args.projector}...")
    state = torch.load(args.projector, map_location=device, weights_only=True)
    if "projector_state_dict" in state:
        model.projector.load_state_dict(state["projector_state_dict"])
    else:
        model.projector.load_state_dict(state)
    model.projector = model.projector.to(device=device, dtype=torch.float32)

    # Dataset
    if args.cached:
        from training.data.cached_dataset import CachedVisionDataset
        cache_dir = os.path.join(os.path.dirname(__file__), "..", args.cache_dir)
        base_dataset = CachedVisionDataset(cache_dir, tokenizer, max_length=args.max_length)
    else:
        from sarvam_omni.processor import SarvamOmniProcessor
        with open(os.path.join(VISION_DIR, "preprocessor_config.json")) as f:
            preproc = json.load(f)
        image_processor = Qwen2VLImageProcessor(**preproc)
        processor = SarvamOmniProcessor(tokenizer, image_processor, vc)

        if args.dataset == "HuggingFaceM4/the_cauldron":
            from training.data.llava_dataset import CauldronAlignmentDataset
            base_dataset = CauldronAlignmentDataset(
                processor=processor, max_length=args.max_length,
                image_target_size=args.image_size, max_samples_per_config=80000,
            )
        else:
            from training.data.cached_dataset import StreamingVisionDataset
            base_dataset = StreamingVisionDataset(
                args.dataset, args.config,
                processor, max_length=args.max_length, image_target_size=args.image_size,
            )

    # Mix with text-only data
    dataset = MixedVisionTextDataset(base_dataset, tokenizer, args.text_ratio, args.max_length)

    # Optimizer (projector + LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min(step / max(1, args.warmup_steps), 1.0) *
        max(0.0, 0.5 * (1.0 + math.cos(math.pi * max(0, step - args.warmup_steps) /
            max(1, args.max_steps - args.warmup_steps))))
    )

    # Training loop
    print(f"\nStarting Stage 1.5 (max {args.max_steps} steps)...")
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

            vf = sample.get("vision_features")
            pv = sample.get("pixel_values")
            gt = sample.get("grid_thw")

            output = model(
                input_ids=input_ids, attention_mask=attention_mask,
                image_token_mask=image_token_mask,
                vision_features=vf.to(device) if vf is not None else None,
                pixel_values=pv.to(device=device, dtype=dtype) if pv is not None else None,
                grid_thw=gt.to(device) if gt is not None else None,
                labels=labels,
            )

            loss = output.loss / args.grad_accum
            loss.backward()
            accum_loss += output.loss.item()
            accum_count += 1

            if accum_count >= args.grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                avg_loss = accum_loss / accum_count

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    lr = scheduler.get_last_lr()[0]
                    print(f"Step {step} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | {elapsed/60:.1f}m")

                if step % args.save_every == 0:
                    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"step_{step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    sarvam.save_pretrained(os.path.join(ckpt_dir, "lora"))
                    torch.save(model.projector.state_dict(), os.path.join(ckpt_dir, "projector.pt"))
                    print(f"  Saved: {ckpt_dir}")
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_dir = os.path.join(CHECKPOINT_DIR, "best")
                        os.makedirs(best_dir, exist_ok=True)
                        sarvam.save_pretrained(os.path.join(best_dir, "lora"))
                        torch.save(model.projector.state_dict(), os.path.join(best_dir, "projector.pt"))

                accum_loss = 0.0
                accum_count = 0
                if step >= args.max_steps:
                    break

            if (accum_count + step * args.grad_accum) % 50 == 0:
                del output, loss
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
    print(f"\nStage 1.5 complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
