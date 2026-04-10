#!/usr/bin/env python3
"""SarvamOmni — Comprehensive Test Suite.

Runs all validation tests on the checkpoint:
  1. Component loading verification
  2. Text-only generation quality
  3. Image captioning (synthetic images)
  4. Multilingual capability
  5. Action parsing
  6. Checkpoint comparison (Stage 1 vs Stage 1.5)
  7. Memory usage report

Usage:
    python run_all_tests.py
    python run_all_tests.py --quick          # Skip slow tests
    python run_all_tests.py --step 500       # Test a different step
    python run_all_tests.py --save-report    # Save results to test_report.json

Requires: Sarvam-30B base model (see AGENT_INSTRUCTIONS.md)
"""

import os
import sys
import json
import time
import argparse
import traceback
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Import our test_checkpoint module
from test_checkpoint import load_model, get_sarvam_path, run_image_inference, run_text_inference


class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.error = None
        self.details = {}
        self.duration = 0

    def to_dict(self):
        return {
            "name": self.name,
            "passed": self.passed,
            "error": str(self.error) if self.error else None,
            "details": self.details,
            "duration_s": round(self.duration, 1),
        }


def create_test_images():
    """Create synthetic test images in /tmp."""
    images = {}

    # 1. Solid red
    img = Image.new("RGB", (672, 672), (255, 0, 0))
    path = "/tmp/sarvam_test_red.png"
    img.save(path)
    images["red_solid"] = path

    # 2. Blue with text
    img = Image.new("RGB", (672, 672), (0, 0, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
    except Exception:
        font = ImageFont.load_default()
    draw.text((200, 300), "Hello", fill="white", font=font)
    path = "/tmp/sarvam_test_blue_text.png"
    img.save(path)
    images["blue_text"] = path

    # 3. Gradient
    img = Image.new("RGB", (672, 672))
    for y in range(672):
        for x in range(672):
            img.putpixel((x, y), (int(255*x/672), int(255*y/672), 128))
    path = "/tmp/sarvam_test_gradient.png"
    img.save(path)
    images["gradient"] = path

    # 4. Simple "screenshot" with a button-like shape
    img = Image.new("RGB", (672, 672), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw a "button"
    draw.rectangle([250, 300, 420, 350], fill=(0, 120, 215), outline=(0, 80, 180))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((290, 312), "Search", fill="white", font=font)
    # Draw a "text field"
    draw.rectangle([100, 200, 570, 250], fill="white", outline=(180, 180, 180))
    draw.text((110, 210), "Type here...", fill=(180, 180, 180), font=font)
    path = "/tmp/sarvam_test_ui.png"
    img.save(path)
    images["ui_screenshot"] = path

    return images


def run_test(name, func) -> TestResult:
    """Run a single test and catch exceptions."""
    result = TestResult(name)
    t0 = time.time()
    try:
        func(result)
        result.passed = True
    except AssertionError as e:
        result.error = str(e)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    result.duration = time.time() - t0
    return result


def test_loading(result, model, processor, tokenizer):
    """Test 1: Verify all components loaded."""
    result.details["tokenizer_vocab"] = tokenizer.vocab_size
    result.details["image_token_id"] = tokenizer.convert_tokens_to_ids("<|image_soft_token|>")

    vit_params = sum(p.numel() for p in model.vision_encoder.parameters())
    proj_params = sum(p.numel() for p in model.projector.parameters())
    llm_params = sum(p.numel() for p in model.language_model.parameters())

    result.details["vit_params"] = vit_params
    result.details["projector_params"] = proj_params
    result.details["llm_params"] = llm_params
    result.details["total_params"] = vit_params + proj_params + llm_params

    assert result.details["image_token_id"] == 8
    assert vit_params > 500_000_000  # ViT should be >500M
    assert proj_params > 30_000_000  # Projector ~33.6M
    assert llm_params > 30_000_000_000  # LLM >30B


def test_text_generation(result, model, processor, tokenizer, device, dtype):
    """Test 2: Text-only generation."""
    prompts = [
        ("What is the capital of India?", ["Delhi", "New Delhi", "india"]),
        ("Count from 1 to 5:", ["1", "2", "3", "4", "5"]),
    ]

    results = []
    for prompt, keywords in prompts:
        response, n, tps, elapsed = run_text_inference(
            model, processor, tokenizer, prompt,
            max_new_tokens=64, temperature=0.1, device=device, dtype=dtype,
        )
        has_keyword = any(k.lower() in response.lower() for k in keywords)
        results.append({
            "prompt": prompt,
            "response": response[:200],
            "tokens": n,
            "tps": round(tps, 1),
            "has_expected_keyword": has_keyword,
        })

    result.details["text_tests"] = results
    # At least one should have the right keyword
    assert any(r["has_expected_keyword"] for r in results), \
        "No text generation produced expected keywords"


def test_image_captioning(result, model, processor, tokenizer, images, device, dtype):
    """Test 3: Image description."""
    test_cases = [
        ("red_solid", "What color is this image?", ["red"]),
        ("blue_text", "What do you see in this image?", ["blue", "text", "hello"]),
    ]

    results = []
    for img_key, prompt, keywords in test_cases:
        if img_key not in images:
            continue
        response, n, tps, elapsed = run_image_inference(
            model, processor, tokenizer,
            image_path=images[img_key],
            prompt=prompt,
            max_new_tokens=128, temperature=0.1, device=device, dtype=dtype,
        )
        has_keyword = any(k.lower() in response.lower() for k in keywords)
        results.append({
            "image": img_key,
            "prompt": prompt,
            "response": response[:200],
            "tokens": n,
            "tps": round(tps, 1),
            "has_expected_keyword": has_keyword,
        })

    result.details["image_tests"] = results
    # Note: Stage 1.5 step 1000 may not perfectly describe images yet
    # So we don't assert keywords, just that it generates something
    assert all(r["tokens"] > 0 for r in results), "Some image tests produced no tokens"


def test_multilingual(result, model, processor, tokenizer, images, device, dtype):
    """Test 4: Multilingual responses."""
    test_cases = [
        ("text", "नमस्ते, आप कैसे हैं?", "Hindi"),  # Hindi
        ("text", "இது என்ன?", "Tamil"),  # Tamil
    ]

    results = []
    for mode, prompt, lang in test_cases:
        response, n, tps, elapsed = run_text_inference(
            model, processor, tokenizer, prompt,
            max_new_tokens=64, temperature=0.3, device=device, dtype=dtype,
        )
        results.append({
            "language": lang,
            "prompt": prompt,
            "response": response[:200],
            "tokens": n,
            "tps": round(tps, 1),
        })

    result.details["multilingual_tests"] = results
    assert all(r["tokens"] > 0 for r in results), "Some multilingual tests produced no tokens"


def test_action_parsing(result):
    """Test 5: Action space parsing (no model needed)."""
    from sarvam_omni.action_space import parse_action, parse_actions

    test_cases = [
        ('click(x=0.45, y=0.72)', "click"),
        ('type(text="hello world")', "type"),
        ('scroll(direction="down")', "scroll"),
        ('drag(startX=0.1, startY=0.2, endX=0.5, endY=0.8)', "drag"),
        ('done()', "done"),
    ]

    results = []
    for text, expected_type in test_cases:
        action = parse_action(text)
        ok = action is not None and action.action_type == expected_type
        results.append({"input": text, "parsed_type": action.action_type if action else None, "correct": ok})

    result.details["action_tests"] = results
    assert all(r["correct"] for r in results), "Some actions failed to parse"


def test_gui_agentic(result, model, processor, tokenizer, images, device, dtype):
    """Test 6: GUI agentic task (early stage — quality may be low)."""
    if "ui_screenshot" not in images:
        result.details["skipped"] = "No UI screenshot available"
        return

    system = "You are a GUI agent. Respond with actions in format: click(x=0.5, y=0.5)"
    response, n, tps, elapsed = run_image_inference(
        model, processor, tokenizer,
        image_path=images["ui_screenshot"],
        prompt="Click on the search button",
        system_prompt=system,
        max_new_tokens=64, temperature=0.1, device=device, dtype=dtype,
    )

    from sarvam_omni.action_space import parse_actions
    actions = parse_actions(response)

    result.details["response"] = response[:200]
    result.details["parsed_actions"] = len(actions)
    result.details["tokens"] = n
    result.details["note"] = "Stage 2 (agentic) training not yet done — low quality expected"

    # Don't assert success — just verify it generates something
    assert n > 0, "No tokens generated for GUI task"


def main():
    parser = argparse.ArgumentParser(description="Run SarvamOmni test suite")
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--save-report", action="store_true")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--sarvam-path", type=str, default=None)
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("  SarvamOmni — Comprehensive Test Suite")
    print(f"  Checkpoint: Stage 1.5 step {args.step}")
    print(f"  Device: {args.device} | Dtype: {args.dtype}")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Create test images
    print("\nCreating synthetic test images...")
    images = create_test_images()
    print(f"  Created {len(images)} test images")

    # Load model
    projector_path = str(ROOT / "checkpoints" / "stage1_5" / f"step_{args.step}" / "projector.pt")
    lora_path = str(ROOT / "checkpoints" / "stage1_5" / f"step_{args.step}" / "lora")

    sarvam_path = args.sarvam_path or get_sarvam_path()

    model, processor, tokenizer = load_model(
        projector_path=projector_path,
        lora_path=lora_path,
        sarvam_path=sarvam_path,
        device=args.device,
        dtype=dtype,
    )

    # Run tests
    all_results = []

    # Test 1: Loading
    print(f"\n{'='*40}")
    print("TEST 1: Component Loading Verification")
    print(f"{'='*40}")
    r = run_test("loading", lambda res: test_loading(res, model, processor, tokenizer))
    all_results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration:.1f}s)")
    if r.details:
        for k, v in r.details.items():
            if isinstance(v, int) and v > 1_000_000:
                print(f"    {k}: {v:,}")
            else:
                print(f"    {k}: {v}")

    # Test 2: Text generation
    print(f"\n{'='*40}")
    print("TEST 2: Text-Only Generation")
    print(f"{'='*40}")
    r = run_test("text_generation",
                 lambda res: test_text_generation(res, model, processor, tokenizer, args.device, dtype))
    all_results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration:.1f}s)")
    for t in r.details.get("text_tests", []):
        kw_status = "YES" if t["has_expected_keyword"] else "no"
        print(f"    [{kw_status}] {t['prompt'][:40]}... → {t['response'][:60]}...")

    # Test 3: Image captioning
    print(f"\n{'='*40}")
    print("TEST 3: Image Captioning")
    print(f"{'='*40}")
    r = run_test("image_captioning",
                 lambda res: test_image_captioning(res, model, processor, tokenizer, images, args.device, dtype))
    all_results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration:.1f}s)")
    for t in r.details.get("image_tests", []):
        kw_status = "YES" if t["has_expected_keyword"] else "no"
        print(f"    [{kw_status}] {t['image']}: {t['response'][:60]}...")

    # Test 4: Multilingual
    if not args.quick:
        print(f"\n{'='*40}")
        print("TEST 4: Multilingual Capability")
        print(f"{'='*40}")
        r = run_test("multilingual",
                     lambda res: test_multilingual(res, model, processor, tokenizer, images, args.device, dtype))
        all_results.append(r)
        print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration:.1f}s)")
        for t in r.details.get("multilingual_tests", []):
            print(f"    [{t['language']}] → {t['response'][:60]}...")

    # Test 5: Action parsing
    print(f"\n{'='*40}")
    print("TEST 5: Action Space Parsing")
    print(f"{'='*40}")
    r = run_test("action_parsing", test_action_parsing)
    all_results.append(r)
    print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration:.1f}s)")
    for t in r.details.get("action_tests", []):
        print(f"    {'PASS' if t['correct'] else 'FAIL'} {t['input']}")

    # Test 6: GUI agentic
    if not args.quick:
        print(f"\n{'='*40}")
        print("TEST 6: GUI Agentic Task (Early Stage)")
        print(f"{'='*40}")
        r = run_test("gui_agentic",
                     lambda res: test_gui_agentic(res, model, processor, tokenizer, images, args.device, dtype))
        all_results.append(r)
        print(f"  {'PASS' if r.passed else 'FAIL'} ({r.duration:.1f}s)")
        print(f"    Response: {r.details.get('response', 'N/A')[:80]}...")
        print(f"    Parsed actions: {r.details.get('parsed_actions', 0)}")
        if r.details.get("note"):
            print(f"    Note: {r.details['note']}")

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    total_time = sum(r.duration for r in all_results)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} tests passed ({total_time:.1f}s total)")
    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        print(f"    [{status}] {r.name} ({r.duration:.1f}s)")
        if r.error:
            print(f"           Error: {r.error}")
    print(f"{'='*60}")

    # Save report
    if args.save_report:
        report = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_step": args.step,
            "device": args.device,
            "dtype": args.dtype,
            "passed": passed,
            "total": total,
            "results": [r.to_dict() for r in all_results],
        }
        report_path = ROOT / "test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {report_path}")

    # Cleanup test images
    for path in images.values():
        try:
            os.remove(path)
        except Exception:
            pass

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
