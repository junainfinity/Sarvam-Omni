"""AGUVIS dataset loader for Stage 2 (grounding) and Stage 3 (agentic reasoning).

Stage 2: xlangai/aguvis-stage1 — 4.2M grounding samples (screenshot -> element coordinate)
Stage 3: xlangai/aguvis-stage2 — 1.3M agentic trajectories with reasoning

Format output: Sarvam chat template with normalized [0,1] coordinates.
"""

import os
import json
import torch
from torch.utils.data import IterableDataset
from PIL import Image
from typing import Optional


# System prompts for different stages
GROUNDING_SYSTEM_PROMPT = """You are a GUI agent that can interact with screen interfaces. When given a screenshot and a task, output the precise action to take.

Actions use normalized [0,1] coordinates relative to the screenshot:
click(x=0.45, y=0.72)
type(text="search query")
scroll(direction="down")
drag(startX=0.1, startY=0.2, endX=0.5, endY=0.8)
long_press(x=0.3, y=0.5)
done()"""

AGENTIC_SYSTEM_PROMPT = """You are a GUI agent that plans and executes multi-step tasks on screen interfaces. For each step:
1. Observe the screenshot
2. Think about what to do next in <think>...</think> tags
3. Output the action to take

Actions use normalized [0,1] coordinates relative to the screenshot:
click(x=0.45, y=0.72)
type(text="search query")
scroll(direction="down")
drag(startX=0.1, startY=0.2, endX=0.5, endY=0.8)
done()"""


class AGUVISGroundingDataset(IterableDataset):
    """AGUVIS Stage-1: GUI element grounding (screenshot -> coordinate).

    Streaming from HuggingFace to avoid downloading 500GB+ dataset.
    """

    def __init__(
        self,
        processor,
        max_length: int = 1024,
        image_target_size: int = 672,
        max_samples: int = 0,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_target_size = image_target_size
        self.max_samples = max_samples

    def __iter__(self):
        from datasets import load_dataset

        ds = load_dataset("xlangai/aguvis-stage1", split="train", streaming=True)
        count = 0

        for item in ds:
            if self.max_samples > 0 and count >= self.max_samples:
                break

            try:
                # Extract image
                image = item.get("image")
                if image is None:
                    continue
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                else:
                    image = image.convert("RGB")

                # Extract instruction and target action
                instruction = item.get("instruction", item.get("query", ""))
                # Target could be coordinates, action string, etc.
                target = item.get("target", item.get("answer", item.get("action", "")))

                if not instruction or not target:
                    continue

                # Format target as action string if it's coordinates
                if isinstance(target, dict):
                    x = target.get("x", target.get("x_norm", 0))
                    y = target.get("y", target.get("y_norm", 0))
                    target = f"click(x={x:.4f}, y={y:.4f})"
                elif isinstance(target, (list, tuple)) and len(target) >= 2:
                    target = f"click(x={target[0]:.4f}, y={target[1]:.4f})"

                target = str(target).strip()

                # Process through SarvamOmniProcessor
                inputs = self.processor(
                    text=instruction,
                    image=image,
                    system_prompt=GROUNDING_SYSTEM_PROMPT,
                    target_size=self.image_target_size,
                )

                # Tokenize target response
                target_tokens = self.processor.tokenizer(
                    target + "<|end_of_turn|>",
                    add_special_tokens=False,
                    return_tensors="pt",
                )

                # Combine
                input_ids = torch.cat([
                    inputs["input_ids"].squeeze(0),
                    target_tokens["input_ids"].squeeze(0),
                ])
                attention_mask = torch.ones_like(input_ids)
                image_token_mask = torch.cat([
                    inputs["image_token_mask"].squeeze(0),
                    torch.zeros(target_tokens["input_ids"].shape[1], dtype=torch.bool),
                ])

                labels = input_ids.clone()
                prompt_len = inputs["input_ids"].shape[1]
                labels[:prompt_len] = -100

                # Truncate
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
                print(f"Warning: Skipping AGUVIS sample: {e}")
                continue


class AGUVISAgenticDataset(IterableDataset):
    """AGUVIS Stage-2: Multi-step agentic trajectories.

    Each sample is a trajectory with multiple (screenshot, thought, action) steps.
    We train on individual steps with action history context.
    """

    def __init__(
        self,
        processor,
        max_length: int = 2048,
        image_target_size: int = 672,
        max_samples: int = 0,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_target_size = image_target_size
        self.max_samples = max_samples

    def __iter__(self):
        from datasets import load_dataset

        ds = load_dataset("xlangai/aguvis-stage2", split="train", streaming=True)
        count = 0

        for item in ds:
            if self.max_samples > 0 and count >= self.max_samples:
                break

            try:
                # Extract image (current screenshot)
                image = item.get("image")
                if image is None:
                    continue
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert("RGB")
                else:
                    image = image.convert("RGB")

                # Extract task instruction
                instruction = item.get("instruction", item.get("query", ""))
                # Extract thought + action
                thought = item.get("thought", "")
                action = item.get("action", item.get("target", ""))
                # Action history from previous steps
                history = item.get("history", item.get("action_history", ""))

                if not instruction or not action:
                    continue

                # Build prompt with history
                prompt_parts = [instruction]
                if history:
                    if isinstance(history, list):
                        history = "\n".join(str(h) for h in history[-3:])  # Last 3 actions
                    prompt_parts.append(f"\nPrevious actions:\n{history}")

                prompt = "\n".join(prompt_parts)

                # Build target with thought-action format
                target_parts = []
                if thought:
                    target_parts.append(f"<think>{thought}</think>")
                target_parts.append(str(action).strip())
                target = "\n".join(target_parts)

                # Process
                inputs = self.processor(
                    text=prompt,
                    image=image,
                    system_prompt=AGENTIC_SYSTEM_PROMPT,
                    target_size=self.image_target_size,
                )

                target_tokens = self.processor.tokenizer(
                    target + "<|end_of_turn|>",
                    add_special_tokens=False,
                    return_tensors="pt",
                )

                input_ids = torch.cat([
                    inputs["input_ids"].squeeze(0),
                    target_tokens["input_ids"].squeeze(0),
                ])
                attention_mask = torch.ones_like(input_ids)
                image_token_mask = torch.cat([
                    inputs["image_token_mask"].squeeze(0),
                    torch.zeros(target_tokens["input_ids"].shape[1], dtype=torch.bool),
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
                print(f"Warning: Skipping AGUVIS-2 sample: {e}")
                continue
