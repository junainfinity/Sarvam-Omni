#!/usr/bin/env python3
"""SarvamOmni Checkpoint Tester — Self-Contained Inference Script.

Loads all 5 components and runs interactive or single-shot inference.

Usage:
  python test_checkpoint.py                                    # Interactive mode
  python test_checkpoint.py --smoke-test                       # Just verify loading
  python test_checkpoint.py --text-only --prompt "Hello"       # Text-only (no image)
  python test_checkpoint.py --image photo.jpg --prompt "..."   # Single image
  python test_checkpoint.py --step 500 --image photo.jpg ...   # Different step

Requires: Sarvam-30B base model path via SARVAM_MODEL_PATH env var or config.json
"""

import os
import sys
import json
import time
import argparse
import torch
from PIL import Image
from pathlib import Path

# Project root = this script's directory
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─── Path Resolution ────────────────────────────────────────────

def get_sarvam_path() -> str:
    """Resolve Sarvam-30B base model path from multiple sources."""
    # 1. Environment variable
    env_path = os.environ.get("SARVAM_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    # 2. config.json in this directory
    config_file = ROOT / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        path = cfg.get("sarvam_model_path", "")
        if path and os.path.isdir(path):
            return path

    # 3. Common locations
    candidates = [
        ROOT.parent / "sarvam-30b-backup",
        ROOT.parent / "sarvam-30b",
        Path.home() / "Projects" / "tqllm" / "models" / "sarvam-30b-backup",
        Path.home() / "Models" / "sarvam-30b-backup",
    ]
    for c in candidates:
        if c.is_dir() and (c / "config.json").exists():
            return str(c)

    print("ERROR: Cannot find Sarvam-30B base model.")
    print("Set one of:")
    print("  export SARVAM_MODEL_PATH=/path/to/sarvam-30b-backup")
    print("  Or run: bash setup.sh")
    sys.exit(1)


def get_vision_dir() -> str:
    """Get vision encoder directory."""
    vd = ROOT / "models" / "qwen3-vl-vit"
    if not vd.exists():
        print(f"ERROR: Vision encoder not found at {vd}")
        sys.exit(1)
    return str(vd)


# ─── Model Loading ──────────────────────────────────────────────

def load_model(
    projector_path: str,
    lora_path: str = None,
    sarvam_path: str = None,
    vision_dir: str = None,
    device: str = "mps",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load all 5 components of SarvamOmni.

    1. Qwen3-VL Vision Encoder (ViT)
    2. Sarvam-30B Base LLM
    3. Trained Projector (MLP bridge)
    4. LoRA Adapter
    5. Processor (image + text)

    Returns: (model, processor, tokenizer)
    """
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from transformers import Qwen2VLImageProcessor
    from sarvam_omni.vision_encoder import VisionEncoderWrapper
    from sarvam_omni.model import SarvamOmniForConditionalGeneration
    from sarvam_omni.processor import SarvamOmniProcessor

    sarvam_path = sarvam_path or get_sarvam_path()
    vision_dir = vision_dir or get_vision_dir()

    print("=" * 60)
    print("  SarvamOmni — Loading All Components")
    print("=" * 60)
    total_t0 = time.time()

    # ── [1/5] Vision Encoder ────────────────────────────────────
    print(f"\n[1/5] Vision Encoder (Qwen3-VL ViT)...")
    t0 = time.time()
    with open(os.path.join(vision_dir, "vision_config.json")) as f:
        vc_dict = json.load(f)
    full_config = Qwen3VLConfig(vision_config=vc_dict)
    vc = full_config.vision_config

    vision_model = Qwen3VLVisionModel(vc)
    weights = load_file(os.path.join(vision_dir, "vision_encoder.safetensors"))
    clean_weights = {k.replace("visual.", "", 1): v for k, v in weights.items()}
    vision_model.load_state_dict(clean_weights, strict=False)
    vision_model = vision_model.to(dtype=dtype, device=device).eval()
    vision_wrapper = VisionEncoderWrapper(vision_model, vc)
    vit_params = sum(p.numel() for p in vision_model.parameters())
    print(f"       {vit_params:,} params | {vit_params*2/1e9:.2f} GB (BF16) | {time.time()-t0:.1f}s")

    # ── [2/5] Sarvam-30B Base LLM ──────────────────────────────
    print(f"\n[2/5] Sarvam-30B LLM from {sarvam_path}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(sarvam_path, trust_remote_code=True)
    sarvam = AutoModelForCausalLM.from_pretrained(
        sarvam_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device,
    ).eval()
    llm_params = sum(p.numel() for p in sarvam.parameters())
    print(f"       {llm_params:,} params | {llm_params*2/1e9:.2f} GB (BF16) | {time.time()-t0:.1f}s")

    # ── [3/5] Projector ─────────────────────────────────────────
    print(f"\n[3/5] Projector from {projector_path}...")
    t0 = time.time()
    model = SarvamOmniForConditionalGeneration(
        language_model=sarvam,
        vision_encoder=vision_wrapper,
        vision_dim=4096,
        hidden_size=4096,
        image_token_id=8,
    )
    proj_state = torch.load(projector_path, map_location=device, weights_only=True)
    model.projector.load_state_dict(proj_state)
    model.projector = model.projector.to(device=device, dtype=dtype)
    proj_params = sum(p.numel() for p in model.projector.parameters())
    print(f"       {proj_params:,} params | {proj_params*2/1e6:.1f} MB (BF16) | {time.time()-t0:.1f}s")

    # ── [4/5] LoRA Adapter ──────────────────────────────────────
    if lora_path and os.path.exists(lora_path):
        print(f"\n[4/5] LoRA adapter from {lora_path}...")
        t0 = time.time()
        from peft import PeftModel
        model.language_model = PeftModel.from_pretrained(
            model.language_model, lora_path
        ).eval()
        lora_params = sum(
            p.numel() for n, p in model.language_model.named_parameters()
            if "lora" in n.lower()
        )
        print(f"       {lora_params:,} adapter params | {lora_params*2/1e6:.1f} MB | {time.time()-t0:.1f}s")
    else:
        print(f"\n[4/5] LoRA adapter: SKIPPED (not provided or not found)")

    # ── [5/5] Processor ─────────────────────────────────────────
    print(f"\n[5/5] Processor...")
    with open(os.path.join(vision_dir, "preprocessor_config.json")) as f:
        preproc = json.load(f)
    image_processor = Qwen2VLImageProcessor(**preproc)
    processor = SarvamOmniProcessor(tokenizer, image_processor, vc)
    print(f"       Image processor + tokenizer ready")

    total_time = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  All components loaded in {total_time:.1f}s")
    total_params = vit_params + llm_params + proj_params
    print(f"  Total: {total_params:,} params ({total_params*2/1e9:.1f} GB)")
    print(f"{'='*60}")

    return model, processor, tokenizer


# ─── Inference ──────────────────────────────────────────────────

def run_image_inference(
    model, processor, tokenizer,
    image_path: str,
    prompt: str,
    system_prompt: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: str = "mps",
    dtype: torch.dtype = torch.bfloat16,
):
    """Run inference with image + text."""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    print(f"\nImage: {image_path} ({w}x{h})")

    inputs = processor(text=prompt, image=image, system_prompt=system_prompt, target_size=672)

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    image_token_mask = inputs["image_token_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    grid_thw = inputs["grid_thw"].to(device) if inputs["grid_thw"] is not None else None

    n_vis = image_token_mask.sum().item()
    n_txt = input_ids.shape[1] - n_vis
    print(f"Tokens: {n_vis} vision + {n_txt} text = {input_ids.shape[1]} total")

    print("Generating...", flush=True)
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
    elapsed = time.time() - t0

    new_tokens = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    tps = len(new_tokens) / elapsed if elapsed > 0 else 0

    return response, len(new_tokens), tps, elapsed


def run_text_inference(
    model, processor, tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    device: str = "mps",
    dtype: torch.dtype = torch.bfloat16,
):
    """Run text-only inference (no image)."""
    inputs = processor(text=prompt)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    print(f"Text tokens: {input_ids.shape[1]}")
    print("Generating...", flush=True)

    t0 = time.time()
    with torch.no_grad():
        # Use language_model directly for text-only
        lm = model.language_model
        output_ids = lm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
    elapsed = time.time() - t0

    new_tokens = output_ids[0][input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    tps = len(new_tokens) / elapsed if elapsed > 0 else 0

    return response, len(new_tokens), tps, elapsed


def print_response(response, num_tokens, tps, elapsed):
    """Pretty-print a generation result."""
    print(f"\n{'─'*60}")
    print(f"Response ({num_tokens} tokens, {tps:.1f} tok/s, {elapsed:.1f}s):")
    print(f"{'─'*60}")
    print(response)
    print(f"{'─'*60}")

    # Try agentic action parsing
    try:
        from sarvam_omni.action_space import parse_actions
        actions = parse_actions(response)
        if actions:
            print(f"\nParsed agentic actions:")
            for a in actions:
                print(f"  -> {a}")
    except Exception:
        pass


# ─── Interactive Mode ───────────────────────────────────────────

def interactive_loop(model, processor, tokenizer, device, dtype):
    """Interactive testing loop."""
    print(f"\n{'='*60}")
    print(f"  Interactive Mode — SarvamOmni")
    print(f"{'='*60}")
    print(f"  Commands:")
    print(f"    <image_path> | <prompt>    — image + text inference")
    print(f"    text | <prompt>             — text-only inference")
    print(f"    quit / exit / q             — exit")
    print(f"{'='*60}")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        if "|" in user_input:
            parts = user_input.split("|", 1)
            source = parts[0].strip()
            prompt = parts[1].strip()
        else:
            source = user_input
            try:
                prompt = input("Prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

        if source.lower() == "text":
            response, n, tps, elapsed = run_text_inference(
                model, processor, tokenizer, prompt, device=device, dtype=dtype
            )
        else:
            image_path = os.path.expanduser(source)
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue
            response, n, tps, elapsed = run_image_inference(
                model, processor, tokenizer, image_path, prompt, device=device, dtype=dtype
            )

        print_response(response, n, tps, elapsed)


# ─── Smoke Test ─────────────────────────────────────────────────

def smoke_test(model, processor, tokenizer, device, dtype):
    """Quick validation that all components are working."""
    print(f"\n{'='*60}")
    print(f"  Smoke Test")
    print(f"{'='*60}")

    tests_passed = 0
    tests_total = 0

    # Test 1: Tokenizer
    tests_total += 1
    print(f"\n[Test 1] Tokenizer...")
    try:
        tokens = tokenizer.encode("Hello world")
        assert len(tokens) > 0
        img_token_id = tokenizer.convert_tokens_to_ids("<|image_soft_token|>")
        assert img_token_id == 8, f"Expected 8, got {img_token_id}"
        print(f"  PASS — vocab size: {tokenizer.vocab_size}, image_token_id: {img_token_id}")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")

    # Test 2: Vision encoder forward
    tests_total += 1
    print(f"\n[Test 2] Vision encoder forward pass...")
    try:
        # Create a dummy image
        dummy_img = Image.new("RGB", (672, 672), color=(128, 64, 32))
        inputs = processor(text="test", image=dummy_img, target_size=672)
        pv = inputs["pixel_values"].to(device=device, dtype=dtype)
        gt = inputs["grid_thw"].to(device) if inputs["grid_thw"] is not None else None
        with torch.no_grad():
            vis_feat = model.encode_image(pv, gt)
        print(f"  PASS — output shape: {vis_feat.shape}, dtype: {vis_feat.dtype}")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")

    # Test 3: Projector
    tests_total += 1
    print(f"\n[Test 3] Projector transform...")
    try:
        dummy = torch.randn(10, 4096, device=device, dtype=dtype)
        with torch.no_grad():
            out = model.projector(dummy)
        assert out.shape == (10, 4096), f"Expected (10, 4096), got {out.shape}"
        print(f"  PASS — input (10, 4096) → output {out.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")

    # Test 4: Text generation
    tests_total += 1
    print(f"\n[Test 4] Text-only generation...")
    try:
        response, n, tps, elapsed = run_text_inference(
            model, processor, tokenizer,
            prompt="What is 2 + 2?",
            max_new_tokens=32,
            temperature=0.1,
            device=device, dtype=dtype,
        )
        assert len(response) > 0, "Empty response"
        print(f"  PASS — generated {n} tokens at {tps:.1f} tok/s")
        print(f"  Response: {response[:100]}...")
        tests_passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")

    # Test 5: Image + text generation
    tests_total += 1
    print(f"\n[Test 5] Image + text generation (dummy image)...")
    try:
        # Create a simple test image
        dummy_img = Image.new("RGB", (672, 672), color=(255, 0, 0))
        dummy_path = "/tmp/sarvam_omni_test_image.png"
        dummy_img.save(dummy_path)

        response, n, tps, elapsed = run_image_inference(
            model, processor, tokenizer,
            image_path=dummy_path,
            prompt="What color is this image?",
            max_new_tokens=64,
            temperature=0.1,
            device=device, dtype=dtype,
        )
        assert len(response) > 0, "Empty response"
        print(f"  PASS — generated {n} tokens at {tps:.1f} tok/s")
        print(f"  Response: {response[:150]}...")
        tests_passed += 1

        os.remove(dummy_path)
    except Exception as e:
        print(f"  FAIL — {e}")

    print(f"\n{'='*60}")
    print(f"  Smoke Test Results: {tests_passed}/{tests_total} passed")
    if tests_passed == tests_total:
        print(f"  ALL TESTS PASSED!")
    else:
        print(f"  {tests_total - tests_passed} test(s) FAILED")
    print(f"{'='*60}")

    return tests_passed == tests_total


# ─── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test SarvamOmni checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--step", type=int, default=1000,
                        help="Checkpoint step to load (default: 1000)")
    parser.add_argument("--stage1-only", action="store_true",
                        help="Load only Stage 1 projector (no LoRA)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run automated smoke tests")
    parser.add_argument("--text-only", action="store_true",
                        help="Text-only mode (no image)")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--sarvam-path", type=str, default=None,
                        help="Override Sarvam-30B path")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Resolve checkpoint
    if args.stage1_only:
        projector_path = str(ROOT / "checkpoints" / "stage1" / "projector_best.pt")
        lora_path = None
    else:
        projector_path = str(ROOT / "checkpoints" / "stage1_5" / f"step_{args.step}" / "projector.pt")
        lora_path = str(ROOT / "checkpoints" / "stage1_5" / f"step_{args.step}" / "lora")

    if not os.path.exists(projector_path):
        print(f"ERROR: Projector not found: {projector_path}")
        ckpt_dir = ROOT / "checkpoints" / "stage1_5"
        if ckpt_dir.exists():
            print("Available steps:")
            for d in sorted(ckpt_dir.iterdir()):
                if d.is_dir() and (d / "projector.pt").exists():
                    print(f"  --step {d.name.replace('step_', '')}")
        sys.exit(1)

    sarvam_path = args.sarvam_path or get_sarvam_path()

    print(f"\nCheckpoint: {'Stage 1 (projector only)' if args.stage1_only else f'Stage 1.5, step {args.step}'}")
    print(f"Sarvam-30B: {sarvam_path}")
    print(f"Device: {args.device} | Dtype: {args.dtype}")

    # Load model
    model, processor, tokenizer = load_model(
        projector_path=projector_path,
        lora_path=lora_path,
        sarvam_path=sarvam_path,
        device=args.device,
        dtype=dtype,
    )

    # Run mode
    if args.smoke_test:
        success = smoke_test(model, processor, tokenizer, args.device, dtype)
        sys.exit(0 if success else 1)
    elif args.text_only and args.prompt:
        response, n, tps, elapsed = run_text_inference(
            model, processor, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens, temperature=args.temperature,
            device=args.device, dtype=dtype,
        )
        print_response(response, n, tps, elapsed)
    elif args.image and args.prompt:
        response, n, tps, elapsed = run_image_inference(
            model, processor, tokenizer, args.image, args.prompt,
            system_prompt=args.system, max_new_tokens=args.max_tokens,
            temperature=args.temperature, device=args.device, dtype=dtype,
        )
        print_response(response, n, tps, elapsed)
    else:
        interactive_loop(model, processor, tokenizer, args.device, dtype)


if __name__ == "__main__":
    main()
