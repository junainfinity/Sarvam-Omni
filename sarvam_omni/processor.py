"""SarvamOmni processor: combines Qwen3-VL image processing with Sarvam tokenization."""

import torch
from typing import Optional, Union
from PIL import Image


# Sarvam special token IDs (from tokenizer)
IMAGE_SOFT_TOKEN = "<|image_soft_token|>"
START_OF_IMAGE = "<|start_of_image|>"
END_OF_IMAGE = "<|end_of_image|>"


class SarvamOmniProcessor:
    """Processes images and text for SarvamOmni model.

    Combines Qwen3-VL's image preprocessing with Sarvam's tokenizer.
    Inserts the right number of <|image_soft_token|> based on image resolution.
    """

    def __init__(self, tokenizer, image_processor, vision_config):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.vision_config = vision_config

        # Compute tokens per image based on vision config
        self.patch_size = getattr(vision_config, "patch_size", 14)
        self.spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
        self.temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)

        # Get token IDs
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_SOFT_TOKEN)
        self.boi_token_id = tokenizer.convert_tokens_to_ids(START_OF_IMAGE)
        self.eoi_token_id = tokenizer.convert_tokens_to_ids(END_OF_IMAGE)

    def get_num_image_tokens(self, height: int, width: int) -> int:
        """Calculate number of vision tokens for given image dimensions."""
        h_patches = height // self.patch_size // self.spatial_merge_size
        w_patches = width // self.patch_size // self.spatial_merge_size
        return h_patches * w_patches

    def process_image(self, image: Image.Image, target_size: int = 672):
        """Preprocess image using Qwen3-VL's image processor.

        Args:
            image: PIL Image
            target_size: Target resolution (must be divisible by patch_size * spatial_merge_size)

        Returns:
            pixel_values: Preprocessed image tensor
            grid_thw: Grid dimensions tensor
            num_patches: Number of vision tokens
        """
        # Resize to target while maintaining aspect ratio constraints
        effective_patch = self.patch_size * self.spatial_merge_size
        w, h = image.size

        # Scale to fit within target_size, rounding to effective_patch
        scale = target_size / max(w, h)
        new_w = max(effective_patch, round(w * scale / effective_patch) * effective_patch)
        new_h = max(effective_patch, round(h * scale / effective_patch) * effective_patch)

        image = image.resize((new_w, new_h), Image.BICUBIC)

        # Use Qwen3-VL image processor
        processed = self.image_processor(
            images=[image],
            return_tensors="pt",
        )

        pixel_values = processed["pixel_values"]
        grid_thw = processed.get("image_grid_thw", None)

        if grid_thw is None:
            # Compute manually
            t = 1
            h_patches = new_h // self.patch_size
            w_patches = new_w // self.patch_size
            grid_thw = torch.tensor([[t, h_patches, w_patches]], dtype=torch.long)

        num_patches = self.get_num_image_tokens(new_h, new_w)
        return pixel_values, grid_thw, num_patches

    def build_input_with_image(
        self,
        text: str,
        num_image_tokens: int,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Build tokenized input with image token placeholders.

        Args:
            text: User text prompt
            num_image_tokens: Number of <|image_soft_token|> to insert
            system_prompt: Optional system prompt

        Returns:
            Dict with input_ids, attention_mask, image_token_mask
        """
        # Build image token sequence
        image_tokens = IMAGE_SOFT_TOKEN * num_image_tokens
        image_block = f"{START_OF_IMAGE}{image_tokens}{END_OF_IMAGE}"

        # Build full prompt in Sarvam chat format
        parts = ["[@BOS@]"]
        if system_prompt:
            parts.append(f"<|start_of_turn|>system\n{system_prompt}<|end_of_turn|>")
        parts.append(f"<|start_of_turn|>user\n{image_block}\n{text}<|end_of_turn|>")
        parts.append("<|start_of_turn|>assistant\n")

        full_prompt = "".join(parts)

        # Tokenize
        encoded = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Create mask for image token positions
        image_token_mask = input_ids == self.image_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_token_mask": image_token_mask,
        }

    def __call__(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        system_prompt: Optional[str] = None,
        target_size: int = 672,
    ) -> dict:
        """Process text and optional image into model inputs.

        Returns dict with: input_ids, attention_mask, image_token_mask,
                          pixel_values, grid_thw (if image provided)
        """
        if image is None:
            # Text-only
            encoded = self.tokenizer(
                f"[@BOS@]<|start_of_turn|>user\n{text}<|end_of_turn|><|start_of_turn|>assistant\n",
                return_tensors="pt",
                add_special_tokens=False,
            )
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "image_token_mask": torch.zeros_like(encoded["input_ids"], dtype=torch.bool),
                "pixel_values": None,
                "grid_thw": None,
            }

        # Process image
        pixel_values, grid_thw, num_patches = self.process_image(image, target_size)

        # Build tokenized input with image placeholders
        inputs = self.build_input_with_image(text, num_patches, system_prompt)

        # Verify token count matches
        actual_image_tokens = inputs["image_token_mask"].sum().item()
        assert actual_image_tokens == num_patches, (
            f"Image token count mismatch: {actual_image_tokens} tokens in input "
            f"but {num_patches} patches from image. This is a bug in tokenization."
        )

        inputs["pixel_values"] = pixel_values
        inputs["grid_thw"] = grid_thw

        return inputs
