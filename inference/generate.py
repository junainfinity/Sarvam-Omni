#!/usr/bin/env python3
"""SarvamOmni inference pipeline.

Loads the combined model and runs image+text -> agentic action generation.

Usage:
  python inference/generate.py --image screenshot.png --prompt "Click on the search bar"
"""

import os
import sys
import json
import time
import argparse
import torch
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SARVAM_PATH = "/Users/arjun/Projects/Sarvam Omni/sarvam-30b-backup"
VISION_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-vl-vit")


def load_model(
    projector_path: str = None,
    lora_path: str = None,
    device: str = "mps",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load full SarvamOmni model for inference."""
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from sarvam_omni.vision_encoder import VisionEncoderWrapper
    from sarvam_omni.model import SarvamOmniForConditionalGeneration
    from sarvam_omni.processor import SarvamOmniProcessor
    from transformers import Qwen2VLImageProcessor

    # Load vision encoder
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
    vision_wrapper = VisionEncoderWrapper(vision_model, vc)

    # Load Sarvam-30B
    print("Loading Sarvam-30B...")
    tokenizer = AutoTokenizer.from_pretrained(SARVAM_PATH, trust_remote_code=True)
    sarvam = AutoModelForCausalLM.from_pretrained(
        SARVAM_PATH, torch_dtype=dtype, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map=device,
    ).eval()

    # Load LoRA if provided
    if lora_path:
        from peft import PeftModel
        print(f"Loading LoRA from {lora_path}...")
        sarvam = PeftModel.from_pretrained(sarvam, lora_path)

    # Build combined model
    model = SarvamOmniForConditionalGeneration(
        language_model=sarvam,
        vision_encoder=vision_wrapper,
        vision_dim=4096,
        hidden_size=4096,
        image_token_id=8,
    )

    # Load projector
    if projector_path:
        print(f"Loading projector from {projector_path}...")
        model.load_projector(os.path.dirname(projector_path))
    model.projector = model.projector.to(device=device, dtype=dtype)

    # Build processor
    with open(os.path.join(VISION_DIR, "preprocessor_config.json")) as f:
        preproc = json.load(f)
    image_processor = Qwen2VLImageProcessor(**preproc)
    processor = SarvamOmniProcessor(tokenizer, image_processor, vc)

    return model, processor, tokenizer


def generate_response(
    model,
    processor,
    tokenizer,
    image_path: str,
    prompt: str,
    system_prompt: str = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "mps",
    dtype: torch.dtype = torch.bfloat16,
):
    """Generate a response given an image and text prompt."""
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, image=image, system_prompt=system_prompt, target_size=672)

    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    image_token_mask = inputs["image_token_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype) if inputs["pixel_values"] is not None else None
    grid_thw = inputs["grid_thw"].to(device) if inputs["grid_thw"] is not None else None

    # Generate
    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_token_mask=image_token_mask,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    t1 = time.time()

    # Decode only the new tokens
    new_tokens = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    gen_tokens = len(new_tokens)
    tokens_per_sec = gen_tokens / (t1 - t0) if t1 > t0 else 0

    return response, gen_tokens, tokens_per_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    parser.add_argument("--projector", type=str, default=None, help="Path to projector checkpoint")
    parser.add_argument("--lora", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    model, processor, tokenizer = load_model(
        projector_path=args.projector,
        lora_path=args.lora,
        device=args.device,
    )

    response, num_tokens, tps = generate_response(
        model, processor, tokenizer,
        image_path=args.image,
        prompt=args.prompt,
        system_prompt=args.system,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print(f"Response ({num_tokens} tokens, {tps:.1f} tok/s):")
    print(f"{'='*60}")
    print(response)

    # Try to parse as agentic action
    from sarvam_omni.action_space import parse_actions
    actions = parse_actions(response)
    if actions:
        print(f"\nParsed actions:")
        for a in actions:
            print(f"  {a}")


if __name__ == "__main__":
    main()
