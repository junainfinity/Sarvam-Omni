#!/usr/bin/env python3
"""Download and extract Qwen3-VL-7B vision encoder."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sarvam_omni.vision_encoder import extract_vision_model, save_vision_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-7B",
                        help="HuggingFace model ID for Qwen3-VL")
    parser.add_argument("--save-dir", default="models/qwen3-vl-vit",
                        help="Directory to save extracted vision encoder")
    args = parser.parse_args()

    print(f"Extracting vision encoder from {args.model_id}...")
    vision_model, vision_config = extract_vision_model(args.model_id, device="cpu")

    save_dir = os.path.join(os.path.dirname(__file__), "..", args.save_dir)
    save_vision_encoder(vision_model, vision_config, save_dir)
    print("Done!")


if __name__ == "__main__":
    main()
