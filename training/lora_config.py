"""LoRA configuration for Sarvam-30B MoE.

Stage 2+: Apply LoRA to attention layers and shared expert.
Keep router/gate weights frozen to preserve routing patterns.
Do NOT apply LoRA to the 128 routed experts (would destabilize routing).
"""

from peft import LoraConfig, TaskType


def get_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
) -> LoraConfig:
    """Get LoRA config targeting Sarvam-30B's attention and shared expert.

    Target modules (from modeling_sarvam_moe.py):
      - model.layers.*.attention.query_key_value  (fused QKV, all 19 layers)
      - model.layers.*.attention.dense             (output projection, all 19 layers)
      - model.layers.{1-18}.mlp.shared_experts.gate_proj
      - model.layers.{1-18}.mlp.shared_experts.up_proj
      - model.layers.{1-18}.mlp.shared_experts.down_proj

    NOT targeted (frozen):
      - model.layers.*.mlp.gate.weight             (router — critical for MoE stability)
      - model.layers.*.mlp.experts.{0-127}.*       (128 routed experts — too many)
      - model.word_embeddings                       (embedding table)
      - lm_head                                     (output head)
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "attention.query_key_value",
            "attention.dense",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ],
        # Don't apply to these (safety check)
        modules_to_save=None,
        bias="none",
    )


def print_lora_info(model):
    """Print LoRA adapter statistics."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA stats:")
    print(f"  Trainable: {trainable:,} ({trainable * 4 / 1e6:.1f} MB)")
    print(f"  Total: {total:,}")
    print(f"  Trainable %: {100 * trainable / total:.4f}%")

    # Check router weights are frozen
    for name, param in model.named_parameters():
        if "gate.weight" in name and param.requires_grad:
            print(f"  WARNING: Router weight {name} is trainable! Should be frozen.")
