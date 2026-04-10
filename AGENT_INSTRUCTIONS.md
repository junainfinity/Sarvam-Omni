# SarvamOmni — Agent Handoff Instructions

## What This Is

SarvamOmni is an agentic vision-language model that merges:
- **Sarvam-30B**: Indian multilingual 32B MoE LLM (22+ languages, 128 experts top-6)
- **Qwen3-VL-8B's vision encoder**: SigLIP-2 ViT (576M params)

Connected via a **LLaVA-style late fusion** architecture with a 2-layer MLP projector.

This checkpoint is from **Stage 1.5, Step 1000** of a 3-stage training pipeline.

---

## What You Need to Copy

Two folders must be on the target machine:

| Folder | Size | Contains |
|--------|------|----------|
| `Checkpoint 1/` | ~3.7 GB | ViT encoder, projector, LoRA, all code, logs |
| `sarvam-30b-backup/` | ~120 GB | Base Sarvam-30B LLM weights (26 safetensors shards) |

The base LLM is at the ORIGINAL machine path: `/Users/arjun/Projects/tqllm/models/sarvam-30b-backup/`

**CRITICAL**: You MUST copy `sarvam-30b-backup/` alongside this checkpoint. Without it, nothing works.

---

## Quick Start (5 Steps)

### Step 0: Prerequisites
```bash
# macOS with Apple Silicon (M3 Max 128GB recommended)
# Python 3.11+
# Allocate max GPU memory:
sudo sysctl iogpu.wired_limit_mb=129024
```

### Step 1: Setup Environment
```bash
cd "Checkpoint 1"
bash setup.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Set the Base Model Path
Edit `config.json` (created by setup.sh) OR set environment variable:
```bash
export SARVAM_MODEL_PATH="/path/to/sarvam-30b-backup"
```

### Step 3: Fix LoRA adapter_config.json
The LoRA adapter has a hardcoded path. Fix it:
```bash
python fix_paths.py --sarvam-path "/path/to/sarvam-30b-backup"
```
This updates `base_model_name_or_path` in all adapter_config.json files.

### Step 4: Run Inference
```bash
# Interactive mode:
python test_checkpoint.py

# Single image:
python test_checkpoint.py --image photo.jpg --prompt "Describe this image"

# Text-only (verify base LLM works):
python test_checkpoint.py --text-only --prompt "What is the capital of India?"
```

### Step 5: Run Full Test Suite
```bash
python run_all_tests.py
```

---

## Architecture Details

```
Image (PIL) ──► Qwen2VLImageProcessor ──► pixel_values + grid_thw
                                              │
                                              ▼
                                    Qwen3VLVisionModel (ViT)
                                    576M params, FROZEN
                                    Output: pooler_output [N, 4096]
                                              │
                                              ▼
                                    VisionProjector (2-layer MLP)
                                    33.6M params, TRAINED
                                    Linear(4096→4096) → GELU → Linear(4096→4096) + residual
                                              │
                                              ▼
Text ──► Sarvam Tokenizer ──► input_ids ──► Embeddings [M, 4096]
                                              │
                              ┌────────────────┘
                              ▼
                    Token Merge: replace <|image_soft_token|> positions
                    with projected vision features
                              │
                              ▼
                    Sarvam-30B MoE Decoder
                    32B params, 19 layers, 128 experts top-6
                    LoRA r=32 on attention + shared experts (73MB adapter)
                              │
                              ▼
                           Logits ──► Response text
```

### Key Token IDs
- `<|image_soft_token|>` = token ID **8** (placeholder for vision features)
- `<|start_of_image|>` / `<|end_of_image|>` = framing tokens
- `[@BOS@]` = beginning of sequence
- `<|start_of_turn|>` / `<|end_of_turn|>` = chat turn markers

### Prompt Format
```
[@BOS@]<|start_of_turn|>system
{system_prompt}<|end_of_turn|><|start_of_turn|>user
<|start_of_image|>{N × <|image_soft_token|>}<|end_of_image|>
{user_text}<|end_of_turn|><|start_of_turn|>assistant
```

Where N = number of vision patches = (H / 14 / 2) × (W / 14 / 2) for a resized image.

---

## File Layout

```
Checkpoint 1/
├── AGENT_INSTRUCTIONS.md      ← YOU ARE HERE
├── PROJECT_HISTORY.md         ← Full history of what transpired
├── requirements.txt           ← Python dependencies
├── setup.sh                   ← Automated setup script
├── fix_paths.py               ← Fix hardcoded paths in adapter configs
├── test_checkpoint.py         ← Interactive inference script
├── run_all_tests.py           ← Automated test suite
├── config.json                ← Created by setup.sh (paths config)
│
├── checkpoints/
│   ├── stage1/                ← Stage 1 projector-only checkpoints
│   │   ├── projector_best.pt
│   │   ├── projector_step200.pt
│   │   ├── projector_step400.pt
│   │   ├── projector_step600.pt
│   │   ├── projector_step800.pt
│   │   └── projector_step1000.pt
│   └── stage1_5/              ← Stage 1.5 projector + LoRA checkpoints
│       ├── best/
│       │   ├── projector.pt
│       │   └── lora/adapter_model.safetensors + adapter_config.json
│       ├── step_500/
│       │   ├── projector.pt
│       │   └── lora/adapter_model.safetensors + adapter_config.json
│       └── step_1000/         ← PRIMARY CHECKPOINT TO TEST
│           ├── projector.pt   (128MB - trained vision→LLM bridge)
│           └── lora/
│               ├── adapter_model.safetensors (73MB - LoRA weights)
│               └── adapter_config.json
│
├── models/
│   └── qwen3-vl-vit/         ← Extracted Qwen3-VL vision encoder
│       ├── vision_encoder.safetensors (1.15GB)
│       ├── vision_config.json
│       └── preprocessor_config.json
│
├── sarvam_omni/               ← Core model code
│   ├── model.py               ← SarvamOmniForConditionalGeneration
│   ├── projector.py           ← VisionProjector (2-layer MLP)
│   ├── vision_encoder.py      ← VisionEncoderWrapper
│   ├── processor.py           ← SarvamOmniProcessor
│   ├── action_space.py        ← Agentic action parsing
│   └── utils.py
│
├── training/                  ← Training scripts (for reference)
│   ├── train_projector.py     ← Stage 1
│   ├── train_midstage.py      ← Stage 1.5 (current)
│   ├── train_grounding.py     ← Stage 2 (not started)
│   ├── train_agentic.py       ← Stage 2 alt
│   ├── lora_config.py         ← LoRA targeting config
│   └── data/                  ← Dataset loaders
│
├── inference/                 ← Original inference scripts
│   ├── generate.py
│   └── demo.py
│
├── scripts/                   ← Utility scripts
│   ├── run_all_stages.sh
│   └── ...
│
└── logs/                      ← Training logs
    ├── stage1.log
    ├── stage1_5.log
    └── full_pipeline.log
```

---

## What to Test

### Test 1: Smoke Test — Does It Load?
```bash
python test_checkpoint.py --smoke-test
```
Expected: All 5 components load, prints param counts, no errors.

### Test 2: Text-Only Generation
```bash
python test_checkpoint.py --text-only --prompt "Explain quantum computing in simple terms"
```
Expected: Coherent English text. This verifies Sarvam-30B + LoRA work.

### Test 3: Image Description
```bash
python test_checkpoint.py --image test_images/sample.jpg --prompt "Describe this image in detail"
```
Expected: Description related to the image content. Quality may vary — this is step 1000 of mid-training.

### Test 4: Multilingual (Hindi)
```bash
python test_checkpoint.py --image test_images/sample.jpg --prompt "इस तस्वीर में क्या है?"
```
Expected: Response in Hindi describing the image.

### Test 5: GUI/Agentic (early stage — may not work well yet)
```bash
python test_checkpoint.py --image screenshot.png --prompt "Click on the search bar" --system "You are a GUI agent. Output actions in the format: click(x=0.5, y=0.5)"
```
Expected: An action output. Quality will be low since agentic training (Stage 2) hasn't started.

### Test 6: Compare Checkpoints
```bash
# Test projector-only (Stage 1 best):
python test_checkpoint.py --stage1-only --image photo.jpg --prompt "What is this?"

# vs Stage 1.5 step 500:
python test_checkpoint.py --step 500 --image photo.jpg --prompt "What is this?"

# vs Stage 1.5 step 1000:
python test_checkpoint.py --step 1000 --image photo.jpg --prompt "What is this?"
```

---

## Troubleshooting

### "MPS out of memory"
```bash
# Ensure full allocation:
sudo sysctl iogpu.wired_limit_mb=129024
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Close all other apps
# If still failing, try float16 instead of bfloat16:
python test_checkpoint.py --dtype float16
```

### "base_model_name_or_path not found"
```bash
python fix_paths.py --sarvam-path "/correct/path/to/sarvam-30b-backup"
```

### "ImportError: Qwen3VLImageProcessor"
Use `Qwen2VLImageProcessor` — this is correct for transformers 5.x.

### "No module named 'sarvam_omni'"
Run from within the Checkpoint 1 directory, or:
```bash
export PYTHONPATH="/path/to/Checkpoint 1:$PYTHONPATH"
```

### "trust_remote_code" warning
This is expected — Sarvam-30B uses custom model code. Always pass `trust_remote_code=True`.

### LoRA loading fails
Ensure adapter_config.json has the correct `base_model_name_or_path`. Run `fix_paths.py`.

---

## Memory Budget

| Component | Size (BF16) |
|-----------|-------------|
| Sarvam-30B base | 64.3 GB |
| Qwen3-VL ViT | 1.15 GB |
| Projector | 0.13 GB |
| LoRA adapter | 0.07 GB |
| KV cache + activations | ~5-10 GB |
| **Total** | **~71-76 GB** |

M3 Max 128GB has ~50GB headroom after loading. Sufficient for inference.

---

## What This Checkpoint Can and Cannot Do

### CAN (expected to work):
- Image captioning / description (trained on the_cauldron dataset)
- Visual question answering
- Multilingual image understanding (22+ Indian languages from Sarvam base)
- Text-only generation (Sarvam-30B is fully functional)

### CANNOT YET (needs Stage 2 training):
- GUI grounding / clicking (Stage 2 trains on AGUVIS dataset)
- Agentic action sequences
- Precise coordinate prediction

### QUALITY EXPECTATIONS:
This is a mid-training checkpoint (step 1000/5000 of Stage 1.5). Quality will be:
- **Reasonable** for basic image description
- **Hit or miss** for complex visual reasoning
- **Poor** for agentic tasks (not trained yet)

---

## For the AI Agent: How to Evaluate Quality

1. **Loss check**: Stage 1.5 loss at step 1000 was 0.39 (down from 4.60 at start). Good convergence.
2. **Qualitative**: Run the test suite. Look for:
   - Does it describe image content? (not just hallucinate)
   - Does it follow the prompt language? (Hindi prompt → Hindi response)
   - Is generation coherent? (not repetitive loops or garbage)
3. **Quantitative** (if time permits):
   - BLEU/CIDEr on the_cauldron validation split
   - Use `training/data/llava_dataset.py` to load eval data

---

## Resuming Training (if needed)

Training is still running on the original M4 Max machine. If you need to resume from this checkpoint on the M3 Max:

```bash
# Stage 1.5 resume from step 1000:
python training/train_midstage.py \
  --projector checkpoints/stage1/projector_best.pt \
  --resume-from checkpoints/stage1_5/step_1000 \
  --dataset HuggingFaceM4/the_cauldron \
  --device mps --dtype bfloat16 \
  --lr 2e-4 --grad-accum 32 --warmup-steps 200 \
  --max-steps 5000 --save-every 500 --log-every 10 \
  --max-length 1024 --image-size 672 \
  --lora-r 32 --lora-alpha 64 --text-ratio 0.2
```

Note: `--resume-from` may need implementation if not already in the script. Check `train_midstage.py`.
