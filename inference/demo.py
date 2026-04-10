#!/usr/bin/env python3
"""Interactive SarvamOmni demo.

Takes screenshots and generates agentic actions in a loop.

Usage:
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python inference/demo.py \
    --projector checkpoints/stage1/projector_best.pt \
    --lora checkpoints/stage2/best/lora
"""

import os
import sys
import argparse
import torch
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projector", type=str, default=None)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    from inference.generate import load_model, generate_response
    from sarvam_omni.action_space import parse_actions

    print("Loading SarvamOmni model...")
    model, processor, tokenizer = load_model(
        projector_path=args.projector,
        lora_path=args.lora,
        device=args.device,
    )
    print("Model loaded! Ready for interaction.\n")

    # Interactive loop
    print("=" * 60)
    print("SarvamOmni Interactive Demo")
    print("=" * 60)
    print("Commands:")
    print("  Enter image path + prompt to generate actions")
    print("  'quit' or 'exit' to stop")
    print()

    while True:
        try:
            image_path = input("Image path: ").strip()
            if image_path.lower() in ("quit", "exit", "q"):
                break
            if not os.path.exists(image_path):
                print(f"  File not found: {image_path}")
                continue

            prompt = input("Prompt: ").strip()
            if not prompt:
                prompt = "Describe what you see in this screenshot and suggest the next action."

            print("\nGenerating...")
            response, num_tokens, tps = generate_response(
                model, processor, tokenizer,
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                device=args.device,
            )

            print(f"\nResponse ({num_tokens} tokens, {tps:.1f} tok/s):")
            print("-" * 40)
            print(response)
            print("-" * 40)

            actions = parse_actions(response)
            if actions:
                print(f"Parsed actions: {len(actions)}")
                for i, a in enumerate(actions):
                    print(f"  [{i+1}] {a}")

            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
