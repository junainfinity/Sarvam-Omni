"""Qwen3-VL vision encoder extraction and wrapper."""

import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


def download_vision_encoder(
    model_id: str = "Qwen/Qwen3-VL-7B",
    save_dir: Optional[str] = None,
) -> str:
    """Download only the vision encoder weights from Qwen3-VL.

    Returns path to saved vision encoder directory.
    """
    from huggingface_hub import snapshot_download

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-vl-vit")
    save_dir = str(Path(save_dir).resolve())

    if os.path.exists(os.path.join(save_dir, "config.json")):
        print(f"Vision encoder already exists at {save_dir}")
        return save_dir

    print(f"Downloading vision encoder from {model_id}...")
    # Download full model (we'll extract vision weights after)
    cache_dir = snapshot_download(
        model_id,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "*.safetensors",
            "model.safetensors.index.json",
        ],
    )
    print(f"Downloaded to cache: {cache_dir}")
    return cache_dir


def extract_vision_model(model_id_or_path: str, device: str = "cpu"):
    """Load just the vision model from a Qwen3-VL checkpoint.

    Returns:
        vision_model: The Qwen3VLVisionModel (ViT + patch merger)
        vision_config: The vision configuration
    """
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
    vision_config = config.vision_config

    print(f"Vision config: depth={vision_config.depth}, "
          f"hidden_size={vision_config.hidden_size}, "
          f"out_hidden_size={vision_config.out_hidden_size}")

    # Load the full model but only keep the vision part
    # Use low_cpu_mem_usage to avoid doubling memory
    from transformers import Qwen3VLForConditionalGeneration

    print("Loading Qwen3-VL (this may take a while)...")
    full_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",  # Load to CPU first
    )

    # Extract vision model
    vision_model = full_model.visual
    vision_model = vision_model.to(device)

    # Free the rest
    del full_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Vision model extracted. Parameters: {sum(p.numel() for p in vision_model.parameters()):,}")
    return vision_model, vision_config


def save_vision_encoder(vision_model: nn.Module, vision_config, save_dir: str):
    """Save extracted vision encoder as a standalone checkpoint."""
    os.makedirs(save_dir, exist_ok=True)

    # Save weights
    torch.save(vision_model.state_dict(), os.path.join(save_dir, "vision_model.pt"))

    # Save config
    import json
    config_dict = {
        "depth": vision_config.depth,
        "hidden_size": vision_config.hidden_size,
        "out_hidden_size": vision_config.out_hidden_size,
        "num_heads": vision_config.num_heads,
        "patch_size": getattr(vision_config, "patch_size", 14),
        "spatial_merge_size": getattr(vision_config, "spatial_merge_size", 2),
        "temporal_patch_size": getattr(vision_config, "temporal_patch_size", 2),
        "in_channels": getattr(vision_config, "in_channels", 3),
    }
    with open(os.path.join(save_dir, "vision_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Vision encoder saved to {save_dir}")


class VisionEncoderWrapper(nn.Module):
    """Wrapper around Qwen3-VL's vision model for use in SarvamOmni.

    Handles image preprocessing and outputs [batch, num_patches, out_hidden_size] features.
    """

    def __init__(self, vision_model: nn.Module, vision_config):
        super().__init__()
        self.vision_model = vision_model
        self.out_hidden_size = vision_config.out_hidden_size
        self.patch_size = getattr(vision_config, "patch_size", 14)
        self.spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
        self.temporal_patch_size = getattr(vision_config, "temporal_patch_size", 2)

    @property
    def dtype(self):
        return next(self.vision_model.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_model.parameters()).device

    def get_num_patches(self, image_height: int, image_width: int) -> int:
        """Calculate number of output tokens for a given image size."""
        h_patches = image_height // self.patch_size // self.spatial_merge_size
        w_patches = image_width // self.patch_size // self.spatial_merge_size
        return h_patches * w_patches

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: Preprocessed image tensor from Qwen3VL processor
            grid_thw: Grid dimensions [temporal, height, width] for each image

        Returns:
            vision_features: [total_patches, out_hidden_size]
        """
        output = self.vision_model(pixel_values, grid_thw=grid_thw)
        # pooler_output = post-merger features (4096-dim, spatially merged)
        # last_hidden_state = pre-merger features (1152-dim)
        return output.pooler_output
