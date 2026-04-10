"""LLaVA-558K alignment dataset loader for Stage 1 projector pre-training.

Supports two modes:
1. Streaming from HuggingFace (downloads images on-the-fly via URLs)
2. Local mode (images pre-downloaded)

Also supports the_cauldron datasets as an alternative with embedded images.
"""

import os
import io
import json
import torch
from torch.utils.data import Dataset, IterableDataset
from PIL import Image
from typing import Optional
from pathlib import Path


class CauldronAlignmentDataset(IterableDataset):
    """Uses HuggingFaceM4/the_cauldron for projector alignment.

    Combines multiple vision-language configs that have embedded images.
    Much more reliable than downloading LLaVA images separately.
    """

    # Configs with good image-caption alignment data
    ALIGNMENT_CONFIGS = [
        "textcaps",       # Image captioning
        "vqav2",          # Visual QA
        "okvqa",          # Knowledge-based VQA
        "cocoqa",         # COCO QA
        "tallyqa",        # Counting QA
        "visual7w",       # Visual 7W
        "ocrvqa",         # OCR VQA
    ]

    def __init__(
        self,
        processor,
        configs: list[str] = None,
        max_length: int = 512,
        image_target_size: int = 672,
        max_samples_per_config: int = 50000,
    ):
        self.processor = processor
        self.configs = configs or self.ALIGNMENT_CONFIGS
        self.max_length = max_length
        self.image_target_size = image_target_size
        self.max_samples_per_config = max_samples_per_config

    def _extract_qa(self, item):
        """Extract question and answer from cauldron format."""
        texts = item.get("texts", [])
        if not texts:
            return None, None

        # texts is a list of {"user": ..., "assistant": ...} dicts
        if isinstance(texts, list) and len(texts) > 0:
            entry = texts[0]
            if isinstance(entry, dict):
                return entry.get("user", ""), entry.get("assistant", "")

        return None, None

    def __iter__(self):
        from datasets import load_dataset

        for config in self.configs:
            print(f"  Loading the_cauldron/{config}...")
            count = 0
            try:
                ds = load_dataset(
                    "HuggingFaceM4/the_cauldron",
                    config,
                    split="train",
                    streaming=True,
                )

                for item in ds:
                    if count >= self.max_samples_per_config:
                        break

                    try:
                        # Get images
                        images = item.get("images", [])
                        if not images or images[0] is None:
                            continue

                        image = images[0]
                        if not isinstance(image, Image.Image):
                            continue
                        image = image.convert("RGB")

                        # Get Q&A
                        question, answer = self._extract_qa(item)
                        if not question or not answer:
                            continue

                        # Process
                        inputs = self.processor(
                            text=question,
                            image=image,
                            target_size=self.image_target_size,
                        )

                        # Tokenize answer
                        answer_tokens = self.processor.tokenizer(
                            answer + "<|end_of_turn|>",
                            add_special_tokens=False,
                            return_tensors="pt",
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

                        if input_ids.shape[0] > self.max_length:
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

                print(f"    {config}: yielded {count} samples")

            except Exception as e:
                print(f"    Failed to load {config}: {e}")
                continue


class LLaVAStreamingDataset(IterableDataset):
    """Streaming LLaVA-558K with image download fallback."""

    def __init__(
        self,
        processor,
        max_length: int = 512,
        image_target_size: int = 672,
        split: str = "train",
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_target_size = image_target_size
        self.split = split

    def __iter__(self):
        from datasets import load_dataset
        import httpx

        ds = load_dataset("liuhaotian/LLaVA-Pretrain", split=self.split, streaming=True)
        IMAGE_BASE = "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images/"
        client = httpx.Client(timeout=30.0, follow_redirects=True)

        for item in ds:
            try:
                image = item["image"]
                if isinstance(image, str):
                    resp = client.get(IMAGE_BASE + image)
                    resp.raise_for_status()
                    image = Image.open(io.BytesIO(resp.content)).convert("RGB")
                elif isinstance(image, Image.Image):
                    image = image.convert("RGB")
                else:
                    continue

                conversations = item.get("conversations", [])
                human_text = "Describe this image."
                assistant_text = ""
                for conv in conversations:
                    if conv["from"] == "human":
                        human_text = conv["value"].replace("<image>", "").strip()
                    elif conv["from"] == "gpt":
                        assistant_text = conv["value"].strip()

                if not assistant_text:
                    continue

                inputs = self.processor(
                    text=human_text, image=image, target_size=self.image_target_size,
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

                if input_ids.shape[0] > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    image_token_mask = image_token_mask[:self.max_length]
                    labels = labels[:self.max_length]

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
