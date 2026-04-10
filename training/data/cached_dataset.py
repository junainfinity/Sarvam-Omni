"""Dataset loader that reads pre-computed vision features from disk.

Eliminates the ViT from the training loop entirely.
Saves ~1.15GB memory and gives 2-4x speedup.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import IterableDataset
from pathlib import Path


class CachedVisionDataset(IterableDataset):
    """Loads pre-computed vision features from disk shards.

    Each shard contains:
      - shard_XXXXX.npz: numpy arrays of vision features [num_patches, 4096]
      - shard_XXXXX.json: metadata (id, num_patches, human, assistant)
    """

    def __init__(
        self,
        cache_dir: str,
        tokenizer,
        max_length: int = 512,
        system_prompt: str = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt

        # Load manifest
        with open(self.cache_dir / "manifest.json") as f:
            self.manifest = json.load(f)

        self.num_shards = self.manifest["num_shards"]
        self.total_samples = self.manifest["total_samples"]
        self.vision_dim = self.manifest["vision_dim"]

        # Token IDs
        self.image_token_id = tokenizer.convert_tokens_to_ids("<|image_soft_token|>")
        self.boi_token_id = tokenizer.convert_tokens_to_ids("<|start_of_image|>")
        self.eoi_token_id = tokenizer.convert_tokens_to_ids("<|end_of_image|>")

        print(f"CachedVisionDataset: {self.total_samples} samples, {self.num_shards} shards")

    def _build_input(self, human_text: str, num_patches: int):
        """Build tokenized input with image token placeholders."""
        image_tokens = "<|image_soft_token|>" * num_patches
        image_block = f"<|start_of_image|>{image_tokens}<|end_of_image|>"

        parts = ["[@BOS@]"]
        if self.system_prompt:
            parts.append(f"<|start_of_turn|>system\n{self.system_prompt}<|end_of_turn|>")
        parts.append(f"<|start_of_turn|>user\n{image_block}\n{human_text}<|end_of_turn|>")
        parts.append("<|start_of_turn|>assistant\n")

        full_prompt = "".join(parts)
        encoded = self.tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False)
        return encoded["input_ids"].squeeze(0)

    def __iter__(self):
        for shard_idx in range(self.num_shards):
            shard_path = self.cache_dir / f"shard_{shard_idx:05d}.npz"
            meta_path = self.cache_dir / f"shard_{shard_idx:05d}.json"

            if not shard_path.exists():
                continue

            # Load shard
            data = np.load(shard_path)
            with open(meta_path) as f:
                metadata = json.load(f)

            for i, meta in enumerate(metadata):
                try:
                    features = torch.from_numpy(data[f"arr_{i}"]).float()  # [num_patches, 4096]
                    num_patches = meta["num_patches"]
                    human_text = meta["human"]
                    assistant_text = meta["assistant"]

                    if not assistant_text:
                        continue

                    # Build input tokens
                    prompt_ids = self._build_input(human_text, num_patches)

                    # Tokenize assistant response
                    answer_tokens = self.tokenizer(
                        assistant_text + "<|end_of_turn|>",
                        add_special_tokens=False,
                        return_tensors="pt",
                    )["input_ids"].squeeze(0)

                    # Combine
                    input_ids = torch.cat([prompt_ids, answer_tokens])
                    attention_mask = torch.ones_like(input_ids)
                    image_token_mask = (input_ids == self.image_token_id)

                    # Labels: mask prompt, keep assistant
                    labels = input_ids.clone()
                    prompt_len = len(prompt_ids)
                    labels[:prompt_len] = -100

                    # Truncate
                    if len(input_ids) > self.max_length:
                        input_ids = input_ids[:self.max_length]
                        attention_mask = attention_mask[:self.max_length]
                        image_token_mask = image_token_mask[:self.max_length]
                        labels = labels[:self.max_length]

                    # Verify patch count matches
                    actual_patches = image_token_mask.sum().item()
                    if actual_patches != num_patches:
                        continue

                    yield {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "image_token_mask": image_token_mask,
                        "labels": labels,
                        "vision_features": features,  # Pre-computed! No ViT needed
                    }

                except Exception as e:
                    continue

            # Free shard memory
            del data


class StreamingVisionDataset(IterableDataset):
    """Streaming dataset that computes ViT features on-the-fly.

    Used when features aren't cached yet. Supports ShareGPT4V-PT and similar.
    """

    def __init__(
        self,
        dataset_id: str,
        config: str,
        processor,
        max_length: int = 512,
        image_target_size: int = 672,
        max_samples: int = 0,
    ):
        self.dataset_id = dataset_id
        self.config = config
        self.processor = processor
        self.max_length = max_length
        self.image_target_size = image_target_size
        self.max_samples = max_samples

    def __iter__(self):
        from datasets import load_dataset
        from PIL import Image

        ds = load_dataset(self.dataset_id, self.config, split="train", streaming=True)
        count = 0

        for item in ds:
            if self.max_samples > 0 and count >= self.max_samples:
                break

            try:
                image = item.get("image")
                if image is None or not isinstance(image, Image.Image):
                    continue
                image = image.convert("RGB")

                # Extract conversation
                conversations = item.get("conversations", [])
                human_text = "Describe this image."
                assistant_text = ""
                for conv in conversations:
                    if conv.get("from") == "human":
                        human_text = conv["value"].replace("<image>", "").strip()
                    elif conv.get("from") == "gpt":
                        assistant_text = conv["value"].strip()

                if not assistant_text:
                    continue

                # Process through full pipeline
                inputs = self.processor(
                    text=human_text, image=image,
                    target_size=self.image_target_size,
                )

                answer_tokens = self.processor.tokenizer(
                    assistant_text + "<|end_of_turn|>",
                    add_special_tokens=False, return_tensors="pt",
                )

                input_ids = torch.cat([
                    inputs["input_ids"].squeeze(0),
                    answer_tokens["input_ids"].squeeze(0),
                ])
                attention_mask = torch.ones_like(input_ids)
                image_token_mask = torch.cat([
                    inputs["image_token_mask"].squeeze(0),
                    torch.zeros(answer_tokens["input_ids"].shape[1], dtype=torch.bool),
                ])

                labels = input_ids.clone()
                prompt_len = inputs["input_ids"].shape[1]
                labels[:prompt_len] = -100

                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    image_token_mask = image_token_mask[:self.max_length]
                    labels = labels[:self.max_length]

                count += 1
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "image_token_mask": image_token_mask,
                    "labels": labels,
                    "pixel_values": inputs["pixel_values"],
                    "grid_thw": inputs["grid_thw"],
                }

            except Exception as e:
                continue
