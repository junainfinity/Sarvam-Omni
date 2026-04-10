"""MLP projector bridging vision encoder output to LLM embedding space."""

import torch
import torch.nn as nn


class VisionProjector(nn.Module):
    """Two-layer MLP projector: vision_dim -> hidden_size with GELU activation.

    When vision_dim == hidden_size (as with Qwen3-VL-8B -> Sarvam-30B, both 4096),
    this acts as a learned refinement/alignment layer rather than a dimension bridge.
    """

    def __init__(self, vision_dim: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(vision_dim, hidden_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # Residual connection when dimensions match
        self.use_residual = (vision_dim == hidden_size)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [num_patches, vision_dim] or [batch, num_patches, vision_dim]
        Returns:
            projected: same shape but last dim = hidden_size
        """
        out = self.linear2(self.act(self.linear1(vision_features)))
        if self.use_residual:
            out = out + vision_features
        return out
