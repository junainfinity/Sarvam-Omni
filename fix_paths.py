#!/usr/bin/env python3
"""Fix hardcoded paths in LoRA adapter_config.json files.

The LoRA adapter was trained on a specific machine and adapter_config.json
contains a hardcoded base_model_name_or_path. This script updates all
adapter configs to point to the correct Sarvam-30B location on this machine.

Usage:
    python fix_paths.py --sarvam-path /path/to/sarvam-30b-backup
"""

import os
import sys
import json
import argparse
from pathlib import Path


def find_adapter_configs(root_dir: str) -> list:
    """Find all adapter_config.json files recursively."""
    configs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f == "adapter_config.json":
                configs.append(os.path.join(dirpath, f))
    return configs


def fix_config(config_path: str, sarvam_path: str) -> bool:
    """Update base_model_name_or_path in an adapter config. Returns True if changed."""
    with open(config_path) as f:
        config = json.load(f)

    old_path = config.get("base_model_name_or_path", "")
    if old_path == sarvam_path:
        return False

    config["base_model_name_or_path"] = sarvam_path
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix LoRA adapter paths")
    parser.add_argument("--sarvam-path", required=True,
                        help="Absolute path to Sarvam-30B base model directory")
    parser.add_argument("--root", default=None,
                        help="Root directory to search (default: script directory)")
    args = parser.parse_args()

    sarvam_path = os.path.abspath(os.path.expanduser(args.sarvam_path))

    # Validate
    if not os.path.isdir(sarvam_path):
        print(f"ERROR: Directory not found: {sarvam_path}")
        sys.exit(1)
    if not os.path.exists(os.path.join(sarvam_path, "config.json")):
        print(f"WARNING: No config.json in {sarvam_path} — are you sure this is Sarvam-30B?")

    root = args.root or os.path.dirname(os.path.abspath(__file__))
    configs = find_adapter_configs(root)

    if not configs:
        print("No adapter_config.json files found.")
        return

    print(f"Found {len(configs)} adapter config(s):")
    changed = 0
    for cfg_path in configs:
        rel = os.path.relpath(cfg_path, root)
        was_changed = fix_config(cfg_path, sarvam_path)
        status = "UPDATED" if was_changed else "already correct"
        print(f"  {rel}: {status}")
        if was_changed:
            changed += 1

    print(f"\n{changed} file(s) updated to: {sarvam_path}")


if __name__ == "__main__":
    main()
