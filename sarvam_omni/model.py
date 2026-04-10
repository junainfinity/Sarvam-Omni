"""SarvamOmni: Agentic VLM combining Sarvam-30B MoE with Qwen3-VL vision encoder."""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
from transformers.utils import ModelOutput

from sarvam_omni.projector import VisionProjector


@dataclass
class SarvamOmniOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    router_logits: Optional[tuple] = None


class SarvamOmniForConditionalGeneration(nn.Module):
    """Vision-Language Model: Qwen3-VL ViT + MLP Projector + Sarvam-30B MoE.

    Architecture:
        Image -> ViT (frozen) -> [N, 3584] -> Projector (trainable) -> [N, 4096]
        Text -> Sarvam Embeddings -> [M, 4096]
        Combined -> Sarvam MoE Decoder -> Logits

    The projector is the only trainable component in Stage 1.
    LoRA adapters are added in Stage 2+.
    """

    def __init__(
        self,
        language_model: nn.Module,
        vision_encoder: nn.Module,
        vision_dim: int,
        hidden_size: int = 4096,
        image_token_id: int = 8,
    ):
        super().__init__()
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        self.projector = VisionProjector(vision_dim, hidden_size)
        self.image_token_id = image_token_id
        self.hidden_size = hidden_size

    @property
    def device(self):
        return next(self.projector.parameters()).device

    @property
    def dtype(self):
        return next(self.language_model.parameters()).dtype

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def encode_image(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode image through ViT and project to LLM space.

        Args:
            pixel_values: Preprocessed image tensor
            grid_thw: Grid dimensions for Qwen3-VL ViT

        Returns:
            projected_features: [num_patches, hidden_size]
        """
        with torch.no_grad():
            vision_features = self.vision_encoder(
                pixel_values.to(dtype=self.vision_encoder.dtype, device=self.vision_encoder.device),
                grid_thw=grid_thw.to(device=self.vision_encoder.device) if grid_thw is not None else None,
            )

        # vision_features: [total_patches, vision_dim]
        # Project to LLM hidden size
        projected = self.projector(vision_features.to(dtype=self.projector.linear1.weight.dtype,
                                                       device=self.device))
        return projected

    def _merge_vision_text_embeddings(
        self,
        input_ids: torch.LongTensor,
        vision_features: torch.Tensor,
        image_token_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Replace image token embeddings with projected vision features.

        This preserves gradient flow through the projector.

        Args:
            input_ids: [batch, seq_len]
            vision_features: [total_patches, hidden_size] (all images in batch concatenated)
            image_token_mask: [batch, seq_len] boolean mask of image token positions

        Returns:
            merged_embeddings: [batch, seq_len, hidden_size]
        """
        # Get text embeddings from LLM
        text_embeddings = self.get_input_embeddings()(input_ids)

        # Replace image token positions with vision features
        # Clone to preserve gradient graph
        merged = text_embeddings.clone()

        # Flatten the mask to index into vision features
        batch_size, seq_len = input_ids.shape
        flat_mask = image_token_mask.reshape(-1)

        # Verify counts match
        num_image_tokens = flat_mask.sum().item()
        num_vision_features = vision_features.shape[0]
        assert num_image_tokens == num_vision_features, (
            f"Mismatch: {num_image_tokens} image tokens in input but "
            f"{num_vision_features} vision features from encoder. "
            f"Check image resolution and token insertion."
        )

        # Insert vision features at image token positions
        merged_flat = merged.reshape(-1, self.hidden_size)
        merged_flat[flat_mask] = vision_features.to(dtype=merged_flat.dtype)

        return merged_flat.reshape(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.BoolTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, SarvamOmniOutput]:
        """
        Forward pass for SarvamOmni.

        Supports two modes:
        1. pixel_values provided: encodes image through ViT + projector
        2. vision_features provided: skips ViT, uses pre-computed features + projector
        Otherwise, runs as a pure text model.
        """
        if image_token_mask is not None and (pixel_values is not None or vision_features is not None):
            if vision_features is not None:
                # Cached mode: skip ViT, just project pre-computed features
                projected = self.projector(
                    vision_features.to(dtype=self.projector.linear1.weight.dtype, device=self.device)
                )
            else:
                # Full mode: ViT + projector
                projected = self.encode_image(pixel_values, grid_thw)

            # Merge vision features with text embeddings
            inputs_embeds = self._merge_vision_text_embeddings(
                input_ids, projected, image_token_mask
            )

            # Forward through LLM with inputs_embeds (not input_ids)
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                return_dict=True,
            )
        else:
            # Text-only forward
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_router_logits=output_router_logits,
                return_dict=True,
            )

        if not return_dict:
            return outputs

        return SarvamOmniOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
            router_logits=getattr(outputs, "router_logits", None),
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_token_mask: Optional[torch.BoolTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> torch.LongTensor:
        """Generate text given image + text input.

        For the first forward pass, merges vision features. Subsequent
        autoregressive steps use KV cache (text-only).
        """
        if pixel_values is not None and image_token_mask is not None:
            vision_features = self.encode_image(pixel_values, grid_thw)
            inputs_embeds = self._merge_vision_text_embeddings(
                input_ids, vision_features, image_token_mask
            )

            # Use the language model's generate with inputs_embeds
            output_ids = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs,
            )

            # When using inputs_embeds, generate() may return only new tokens.
            # Prepend input_ids so callers can slice consistently.
            if output_ids.shape[1] < input_ids.shape[1] + 1:
                output_ids = torch.cat([input_ids, output_ids], dim=1)

            return output_ids
        else:
            return self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs,
            )

    def save_projector(self, save_dir: str):
        """Save only the projector weights (tiny: ~126MB)."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.projector.state_dict(), os.path.join(save_dir, "projector.pt"))
        print(f"Projector saved to {save_dir}/projector.pt")

    def load_projector(self, save_dir: str):
        """Load projector weights."""
        import os
        state_dict = torch.load(
            os.path.join(save_dir, "projector.pt"),
            map_location=self.device,
            weights_only=True,
        )
        self.projector.load_state_dict(state_dict)
        print(f"Projector loaded from {save_dir}/projector.pt")
