<p align="center">
  <h1 align="center">SarvamOmni</h1>
  <p align="center">
    <strong>Omnimodal AI for Indian Languages — Vision, Voice, Action</strong>
  </p>
  <p align="center">
    Sarvam-30B MoE + Qwen3-VL Vision Encoder + Full-Duplex Voice (coming soon)
  </p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2412.10302"><img src="https://img.shields.io/badge/Reference-DeepSeek--VL2-blue?style=for-the-badge" alt="DeepSeek-VL2"></a>
  <a href="https://arxiv.org/abs/2504.07491"><img src="https://img.shields.io/badge/Reference-Kimi--VL-blue?style=for-the-badge" alt="Kimi-VL"></a>
  <a href="https://huggingface.co/sarvamai/sarvam-30b"><img src="https://img.shields.io/badge/LLM-Sarvam--30B-orange?style=for-the-badge" alt="Sarvam-30B"></a>
  <a href="https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct"><img src="https://img.shields.io/badge/ViT-Qwen3--VL--8B-green?style=for-the-badge" alt="Qwen3-VL"></a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-red?style=for-the-badge" alt="License">
</p>

<p align="center">
  <a href="#roadmap">Roadmap</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#training">Training</a> ·
  <a href="#inference">Inference</a> ·
  <a href="#checkpoint-status">Results</a> ·
  <a href="#references">References</a>
</p>

---

## Overview

**SarvamOmni** merges [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b), a 32B-parameter Mixture-of-Experts language model supporting **22+ Indian languages**, with [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)'s vision encoder to create an **agentic vision-language model** capable of understanding screenshots and executing GUI actions.

### Key Capabilities

- **GUI Interaction** — Understands screenshots and outputs precise click coordinates, typing actions, scrolls, and drags
- **Multilingual Vision** — Native support for Hindi, Tamil, Telugu, Bengali, Kannada, Marathi, Malayalam, Gujarati and 14+ more Indian languages in both text and visual content
- **Efficient MoE Inference** — Only 2.4B parameters activate per token despite 32B total, making agentic inference (many sequential calls) practical on consumer hardware
- **Runs Locally** — Full model fits on Apple M4 Max 128GB (64.3 GB for LLM + 1.15 GB for ViT)

### Why This Matters

No existing agentic GUI VLM supports Indian languages natively. Research shows **10-25% accuracy drops** when testing VLMs on Indian language content ([arxiv 2603.26742](https://arxiv.org/html/2603.26742)). SarvamOmni fills this gap with Sarvam-30B's 262K multilingual vocabulary trained on Indian language corpora.

---

## Roadmap

SarvamOmni is being built in two major phases: **Vision** first, then **Voice**.

```
                            SarvamOmni Roadmap
 ═══════════════════════════════════════════════════════════════

 PHASE 1: VISION                                    [IN PROGRESS]
 ─────────────────────────────────────────────────────────────────

 Stage 1 ✅  Projector Alignment
             Train MLP bridge between ViT and LLM
             Dataset: the_cauldron | Loss: 2.12 → 1.89

 Stage 1.5 🔄  Mid-Training (current: step 1000/5000)
               Projector + LoRA r=32 on Sarvam-30B
               Dataset: the_cauldron + 20% text mixing
               Loss: 4.60 → 0.39

 Stage 2 ⬜  Agentic GUI Grounding
             Fine-tune on AGUVIS for click/type/scroll actions
             Precise coordinate prediction on screenshots

 Stage 3 ⬜  Evaluation & Optimization
             Benchmark on VLM leaderboards
             Quantization (AWQ/GPTQ) for faster inference
             Indian language vision benchmarks

 ─────────────────────────────────────────────────────────────────

 PHASE 2: FULL-DUPLEX VOICE                              [PLANNED]
 ─────────────────────────────────────────────────────────────────

 Stage 4 ⬜  Speech Encoder Integration
             Add speech encoder (e.g. Whisper / Indic-Whisper)
             Train audio projector to LLM hidden space
             Support Hindi, Tamil, Telugu + 20 more Indian languages

 Stage 5 ⬜  Speech Decoder / TTS
             Add streaming speech synthesis head
             Enable real-time voice output in Indian languages

 Stage 6 ⬜  Full-Duplex Voice-to-Voice
             Simultaneous listen + speak (no turn-taking)
             Interruptible, low-latency conversational AI
             Vision + Voice unified: see screen, hear user, act

 ═══════════════════════════════════════════════════════════════
```

### The Vision

**Phase 1** makes SarvamOmni a capable vision-language model — it can see screenshots, describe images, and execute GUI actions across 22+ Indian languages.

**Phase 2** adds full-duplex voice, making it a truly **omnimodal** assistant. Users will be able to speak naturally in their Indian language while SarvamOmni simultaneously listens, understands the screen, takes actions, and responds by voice — no waiting for turns, just natural conversation. Think of it as an AI that can see your screen, hear you speak in Hindi or Tamil, and talk back while navigating apps for you.

---

## Architecture

```
                         ┌─────────────────────┐
                         │    Input Image       │
                         │   (672×672 RGB)      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Qwen3-VL ViT       │
                         │  576M params, frozen │
                         │  27 blocks, SigLIP-2 │
                         │  Spatial merge 2×2   │
                         └──────────┬──────────┘
                                    │ [N × 4096]
                                    ▼
                         ┌─────────────────────┐
                         │  MLP Projector       │
                         │  33.6M params        │
                         │  Linear → GELU →     │
                         │  Linear + Residual   │
                         └──────────┬──────────┘
                                    │ [N × 4096]
                                    ▼
              ┌──────────────────────────────────────────┐
              │  Token Merging                           │
              │  Replace <|image_soft_token|> positions  │
              │  in text embedding sequence              │
              └──────────────────┬───────────────────────┘
                                 │ [seq_len × 4096]
                                 ▼
                         ┌─────────────────────┐
                         │  Sarvam-30B MoE     │
                         │  32B params (2.4B   │
                         │  active per token)  │
                         │  19 layers, 128     │
                         │  experts, top-6     │
                         │  LoRA r=32 on attn  │
                         │  + shared expert    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                           Action Output
                    click(x=0.45, y=0.72)
```

**Design Choices:**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Fusion type** | Late fusion (LLaVA-style) | No architecture modification to Sarvam-30B needed |
| **Projector** | 2-layer MLP + residual | Industry standard (Kimi-VL, DeepSeek-VL2, LLaVA) |
| **ViT output** | `pooler_output` (4096-dim) | Post-merger, matches Sarvam hidden_size exactly |
| **LoRA targets** | Attention + shared expert | Routed experts and router weights stay frozen to preserve MoE routing |
| **Coordinates** | Normalized [0,1] | Resolution-independent, simple to parse |

---

## Quickstart

### Prerequisites

- Apple Silicon Mac with 128GB+ unified memory (M4 Max recommended)
- Python 3.11+
- ~70 GB free RAM for inference, ~130 GB for training

### Installation

```bash
git clone https://github.com/user/sarvam-omni.git
cd sarvam-omni

# Create environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision transformers>=4.57 accelerate peft \
            pillow datasets wandb huggingface_hub safetensors tqdm
```

### Download Models

```bash
# 1. Sarvam-30B (from HuggingFace, requires auth)
huggingface-cli download sarvamai/sarvam-30b --local-dir models/sarvam-30b

# 2. Extract vision encoder (automatic)
python scripts/extract_vision_from_shard.py
```

---

## Training

SarvamOmni uses a **3-stage training pipeline** based on research from [DeepSeek-VL2](https://arxiv.org/abs/2412.10302), [LLaVA-OneVision](https://arxiv.org/html/2509.23661v1), [VILA](https://arxiv.org/abs/2312.07533), and [AGUVIS](https://arxiv.org/abs/2412.04454).

### Pipeline Overview

| Stage | Trainable | Data | LR | Purpose |
|-------|-----------|------|----|---------|
| **1: Alignment** | Projector (33.6M) | Image-caption pairs | 1e-3 | Bridge ViT → LLM embedding space |
| **1.5: Mid-training** | Projector + LoRA r=32 | VQA + 20% text-only | 2e-4 | LLM learns to attend to vision features |
| **2: Agentic** | Projector + LoRA r=32 | Grounding + agentic + text | 5e-5 | GUI action prediction |

### Run All Stages

```bash
# Allocate GPU memory (resets on reboot)
sudo sysctl iogpu.wired_limit_mb=129024

# Run full pipeline (survives terminal disconnect)
nohup bash scripts/run_all_stages.sh > logs/full_pipeline.log 2>&1 &

# Monitor progress
bash scripts/monitor.sh
# or
grep "Step " logs/full_pipeline.log | tail -5
```

### Run Individual Stages

```bash
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Stage 1: Projector alignment
python -u training/train_projector.py \
  --dataset "HuggingFaceM4/the_cauldron" --config "" \
  --lr 1e-3 --grad-accum 64 --warmup-steps 1000

# Stage 1.5: Mid-training with LoRA
python -u training/train_midstage.py \
  --projector checkpoints/stage1/projector_best.pt \
  --lora-r 32 --lora-alpha 64 --text-ratio 0.2

# Stage 2: Agentic fine-tuning
python -u training/train_grounding.py \
  --stage15-dir checkpoints/stage1_5/best \
  --lr 5e-5 --max-steps 3000
```

### Memory Budget

| Component | Size |
|-----------|------|
| Sarvam-30B (BF16) | 64.3 GB |
| Qwen3-VL ViT (FP16) | 1.15 GB |
| Projector + LoRA + optimizer | ~1 GB |
| **Total** | **~66.5 GB** |
| Headroom on 128GB | ~61.5 GB |

---

## Inference

### Single Image

```bash
python inference/generate.py \
  --image screenshot.png \
  --prompt "Click on the search bar" \
  --projector checkpoints/stage2/best/projector.pt \
  --lora checkpoints/stage2/best/lora
```

### Interactive Demo

```bash
python inference/demo.py \
  --projector checkpoints/stage2/best/projector.pt \
  --lora checkpoints/stage2/best/lora
```

---

## Action Space

SarvamOmni outputs structured actions with normalized `[0, 1]` coordinates:

| Action | Format | Description |
|--------|--------|-------------|
| **click** | `click(x=0.45, y=0.72)` | Single tap at coordinates |
| **type** | `type(text="query")` | Keyboard input |
| **scroll** | `scroll(direction="down")` | Scroll in direction |
| **drag** | `drag(startX=0.1, startY=0.2, endX=0.5, endY=0.8)` | Drag gesture |
| **long_press** | `long_press(x=0.3, y=0.5)` | Long press at coordinates |
| **done** | `done()` | Task complete signal |

### Multi-step Reasoning

```
<think>I can see the email inbox. The latest email from Priya is the
second item. I need to click on it to open it.</think>
click(x=0.50, y=0.28)
```

### Multilingual Support

```
Input:  [Hindi banking app screenshot] + "खोज बार पर क्लिक करें"
Output: click(x=0.45, y=0.08)
```

---

## Project Structure

```
sarvam_omni/
├── model.py              SarvamOmniForConditionalGeneration
├── projector.py           4096→4096 MLP with residual connection
├── vision_encoder.py      Qwen3-VL ViT wrapper (pooler_output)
├── processor.py           Image + text preprocessing
└── action_space.py        Action parsing and validation

training/
├── train_projector.py     Stage 1: projector alignment
├── train_midstage.py      Stage 1.5: LoRA mid-training
├── train_grounding.py     Stage 2: agentic fine-tuning
├── lora_config.py         LoRA target configuration
└── data/
    ├── llava_dataset.py   Cauldron / LLaVA data loaders
    ├── cached_dataset.py  Pre-computed feature loader
    └── aguvis_dataset.py  AGUVIS grounding + agentic loader

inference/
├── generate.py            Single-image inference
└── demo.py                Interactive REPL

scripts/
├── run_all_stages.sh      Master pipeline (chains all stages)
├── monitor.sh             Training health monitor
├── extract_vision_from_shard.py   ViT extraction
└── cache_vision_features.py       Feature pre-computation
```

---

## Technical Details

### Sarvam-30B Configuration
- **Model type:** `sarvam_moe` (Mixture-of-Experts)
- **Parameters:** 32.15B total, 2.4B active per token
- **Layers:** 19 (layer 0 dense, layers 1-18 MoE)
- **Experts:** 128 routed + 1 shared per MoE layer, top-6 routing
- **Hidden size:** 4096, **Vocab:** 262,144 (22+ Indian languages)
- **Context:** 131,072 tokens (RoPE theta 8M)

### Qwen3-VL Vision Encoder
- **Base:** SigLIP-2, 27 transformer blocks
- **Parameters:** 576M (1.15 GB)
- **Patch size:** 16, **Spatial merge:** 2×2
- **Output:** 4096-dim (post-merger)

### LoRA Configuration
- **Rank:** 32, **Alpha:** 64
- **Targets:** `attention.query_key_value`, `attention.dense`, `mlp.shared_experts.{gate,up,down}_proj`
- **Frozen:** All routed experts (128 per layer), router gates, embeddings, LM head

---

## Checkpoint Status

> **Current checkpoint: Stage 1.5, Step 1000/5000** (April 10, 2026)

### Training Progress

| Stage | Description | Status | Loss |
|-------|-------------|--------|------|
| Stage 1 | Projector alignment (frozen LLM + ViT) | Complete | 2.12 → 1.89 |
| Stage 1.5 | Mid-training (projector + LoRA r=32) | **In progress** — step 1000/5000 | 4.60 → 0.39 |
| Stage 2 | Agentic GUI grounding (AGUVIS dataset) | Not started | — |

### Test Results (Smoke Test: 5/5 PASSED)

All five components load and function correctly:

| Test | Status |
|------|--------|
| Tokenizer (vocab 262K, image_token_id=8) | PASS |
| Vision encoder forward ([441, 4096] output) | PASS |
| Projector transform (4096 → 4096) | PASS |
| Text-only generation ("2+2?" → "4") | PASS |
| Image+text generation (red image → "Red.") | PASS |

### Inference Results

**Text-only** (5 tests):

| Prompt | Response | Verdict |
|--------|----------|---------|
| Capital of India? | New Delhi. | Correct |
| Translate to Hindi: The weather is beautiful today. | Yes. | Wrong |
| Explain gravity in one sentence. | Gravity is the force that attracts objects with mass. | Correct |
| Write a haiku about mountains. | Majestic peaks piercing sky high. | Partial |
| 15 multiplied by 23? | 5 multiplied by \(5\) | Wrong |

**Image+text** (5 tests across 3 images):

| Image | Prompt | Response | Verdict |
|-------|--------|----------|---------|
| Dog (800x457) | What animal? | Dog. | Correct |
| Dog (800x457) | Describe | White dog. | Correct |
| Squirrel (1024x683) | What animal? | Gribble. | Wrong |
| Squirrel (1024x683) | Describe | A green lizard. | Wrong |
| Screenshot (856x1042) | What do you see? | Screen. | Plausible |

### Assessment

- **Pipeline**: Fully functional end-to-end after bug fix in `model.generate()`
- **Text**: Basic factual Q&A works; translation and arithmetic are weak
- **Vision**: Can identify simple objects (dogs); struggles with less common animals
- **Speed**: 2-13 tok/s on Apple Silicon MPS
- **Next step**: Continue Stage 1.5 training (4000 more steps), then Stage 2 agentic training

Full results log: [`logs/checkpoint1_test_results.md`](logs/checkpoint1_test_results.md)

---

## References

This project builds on research from:

| Paper | Contribution to SarvamOmni |
|-------|---------------------------|
| [DeepSeek-VL2](https://arxiv.org/abs/2412.10302) | MoE VLM training recipe, 30% text-only mixing |
| [LLaVA-OneVision](https://arxiv.org/html/2509.23661v1) | Stage 1.5 mid-training methodology |
| [VILA](https://arxiv.org/abs/2312.07533) | Text re-blending prevents LLM degradation |
| [Kimi-VL](https://arxiv.org/abs/2504.07491) | MoE VLM architecture reference (2.8B active) |
| [Qwen3-VL](https://arxiv.org/abs/2511.21631) | Vision encoder architecture and training |
| [AGUVIS](https://arxiv.org/abs/2412.04454) | Agentic GUI training recipe |
| [Smol2Operator](https://huggingface.co/blog/smol2operator) | Reproducible agentic training curriculum |
| [FastMMoE](https://arxiv.org/abs/2511.17885) | Vision token routing in MoE models |
| [Sarvam-30B](https://huggingface.co/sarvamai/sarvam-30b) | Base language model |

---

## License

This project is released under the [Apache License 2.0](LICENSE), consistent with both Sarvam-30B and Qwen3-VL licenses.

---

<p align="center">
  Built with Sarvam AI + Qwen + PyTorch on Apple Silicon
</p>
