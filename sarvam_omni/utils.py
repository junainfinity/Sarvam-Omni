"""Memory-efficient loading utilities for M4 Max 128GB."""

import os
import gc
import torch
from pathlib import Path


def load_sarvam_frozen(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "mps",
) -> "SarvamMoEForCausalLM":
    """Load Sarvam-30B in frozen mode, optimized for memory.

    Loads with no gradient tracking and moves to device efficiently.
    """
    import sys
    model_dir = Path(model_path)

    # Add model dir to path so custom code can be imported
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading Sarvam-30B from {model_path} in {dtype}...")
    print("This will use ~120GB of memory. Ensure sufficient RAM is allocated.")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=device,
    )

    # Freeze all parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    param_count = sum(p.numel() for p in model.parameters())
    mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    print(f"Sarvam-30B loaded: {param_count:,} params, {mem_gb:.1f} GB")

    return model


def load_sarvam_tokenizer(model_path: str):
    """Load Sarvam tokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def load_vision_encoder(
    model_id_or_path: str = "Qwen/Qwen3-VL-7B",
    dtype: torch.dtype = torch.float16,
    device: str = "mps",
):
    """Load Qwen3-VL vision encoder in frozen mode.

    Returns (vision_model, vision_config, image_processor)
    """
    from sarvam_omni.vision_encoder import extract_vision_model

    vision_model, vision_config = extract_vision_model(model_id_or_path, device="cpu")
    vision_model = vision_model.to(dtype=dtype, device=device)
    vision_model.eval()
    for param in vision_model.parameters():
        param.requires_grad = False

    # Load image processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id_or_path, trust_remote_code=True)
    image_processor = processor.image_processor

    return vision_model, vision_config, image_processor


def memory_stats():
    """Print current memory usage."""
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    total = psutil.virtual_memory()
    print(f"Process RSS: {mem.rss / 1e9:.1f} GB")
    print(f"System: {total.used / 1e9:.1f} GB used / {total.total / 1e9:.1f} GB total")
    print(f"Available: {total.available / 1e9:.1f} GB")


def cleanup():
    """Force garbage collection and clear MPS cache."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
