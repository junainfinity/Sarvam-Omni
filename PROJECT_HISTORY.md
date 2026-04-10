# SarvamOmni — Full Project History

## Timeline of Events

### Day 1 (April 6, 2026) — Research & Architecture

**Goal**: Merge Sarvam-30B (Indian multilingual MoE LLM) with a vision encoder to create an agentic VLM for GUI interaction.

1. **Base Model Selection**
   - LLM: Sarvam-30B from `sarvamai/sarvam-30b` on HuggingFace
     - 32B params, BF16, 64.3GB on disk (26 shards)
     - MoE architecture: 128 experts, top-6 routing, 19 layers
     - Hidden size: 4096, already has `<|image_soft_token|>` (ID=8)
     - Supports 22+ Indian languages
   - Vision: Qwen3-VL-8B's ViT (SigLIP-2)
     - 576M params, 1.15GB
     - `out_hidden_size=4096` — lucky dimension match with Sarvam

2. **Architecture Decision: LLaVA-style Late Fusion**
   - Token replacement at `<|image_soft_token|>` positions
   - 2-layer MLP projector: Linear(4096→4096) → GELU → Linear(4096→4096) + residual
   - Projector is 33.6M params (tiny compared to the 32B LLM)
   - Chose `pooler_output` (4096-dim, post-merger) NOT `last_hidden_state` (1152-dim)

3. **ViT Extraction**
   - Downloaded Qwen3-VL-8B-Instruct from HuggingFace
   - Extracted vision-only weights → `models/qwen3-vl-vit/vision_encoder.safetensors`
   - Saved config + preprocessor separately

4. **Created Backup of Sarvam-30B**
   - Original: `/Users/arjun/Projects/tqllm/models/sarvam-30b/`
   - Backup: `/Users/arjun/Projects/tqllm/models/sarvam-30b-backup/` (120GB)

### Day 1-2 — Implementation

5. **Core Code Written**
   - `sarvam_omni/model.py` — Main VLM class with token merging
   - `sarvam_omni/projector.py` — 2-layer MLP with residual
   - `sarvam_omni/vision_encoder.py` — ViT wrapper returning pooler_output
   - `sarvam_omni/processor.py` — Combined image + text processor
   - `sarvam_omni/action_space.py` — Agentic action parser (click, type, scroll, drag, done)

6. **Training Pipeline (3 Stages)**
   - **Stage 1** (`train_projector.py`): Projector-only alignment
     - Dataset: `HuggingFaceM4/the_cauldron` (embedded PIL images)
     - LR: 1e-3, BS: 64 (via grad accum), 1000 steps
     - Only projector trainable, ViT + LLM frozen
   - **Stage 1.5** (`train_midstage.py`): Projector + LoRA
     - Same dataset + 20% text-only mixing (per VILA research)
     - LoRA r=32, alpha=64 on attention + shared experts
     - LR: 2e-4, 5000 steps
     - Router weights and 128 routed experts FROZEN
   - **Stage 2** (`train_grounding.py`): Agentic training
     - Dataset: AGUVIS (grounding + agentic)
     - Not started yet — field names are guessed, needs validation

7. **LoRA Configuration** (`training/lora_config.py`)
   - Targets: `attention.query_key_value`, `attention.dense`, `mlp.shared_experts.{gate,up,down}_proj`
   - NOT targeted: router weights, 128 routed experts, embeddings, lm_head
   - This preserves MoE routing stability

### Day 2-3 — Errors Encountered & Fixed

8. **Errors and Resolutions**
   - `ImportError: Qwen3VLImageProcessor` → Changed to `Qwen2VLImageProcessor` (transformers 5.5 naming)
   - LLaVA/ShareGPT4V images are string paths, not PIL → Switched to `the_cauldron` dataset (embedded PIL)
   - Python stdout buffering (no log output) → Fixed with `PYTHONUNBUFFERED=1` and `python -u`
   - `RepositoryNotFoundError` for Qwen3-VL-7B → Correct name is `Qwen/Qwen3-VL-8B-Instruct`
   - `torch_dtype` deprecation → Use `dtype=` instead
   - Relative import error in sarvam_moe modeling → Use `AutoModelForCausalLM` with `trust_remote_code=True`
   - Stage 1.5 missing args → Added `--dataset` and `--config` to argparse
   - Stage 2 missing args → Added `--stage15-dir` flag

### Day 2 — Training Launched

9. **Stage 1 Training** (projector-only)
   - Duration: ~1000 steps, ~24 hours
   - Final loss: ~0.42
   - Saved: `checkpoints/stage1/projector_best.pt`

10. **Stage 1.5 Training** (projector + LoRA)
    - Auto-started after Stage 1 via `scripts/run_all_stages.sh`
    - Running with nohup to survive session disconnects
    - Pipeline PID: 49978, Training PID: 31370

### Day 3 onwards — Monitoring & Documentation

11. **Training Progress (Stage 1.5)**
    - Step 0: Loss ~4.60
    - Step 500: Loss ~1.80 (checkpoint saved)
    - Step 830: Loss ~1.52
    - Step 880: Loss ~0.31 (sharp drop — convergence zone)
    - Step 1000: Loss ~0.39 (checkpoint saved + Checkpoint 1 created)
    - Step 1170: Loss ~0.28 (latest at time of this document)
    - Speed: ~186 sec/step (stable)
    - ETA for completion (step 5000): ~April 18

12. **Checkpoint 1 Created** (April 9, 23:29)
    - Watcher script (`scripts/checkpoint_at_1000.sh`) polled every 60s
    - Triggered at step 1000, copied all files to `Checkpoint 1/` directory
    - Contains: code, projector, LoRA, ViT, logs

13. **GitHub Repo Created**: https://github.com/junainfinity/Sarvam-Omni
    - 35 files uploaded (model weights excluded via .gitignore)
    - Initial push blocked by secret scanning (HF token in DEV_LOG.md)
    - Fixed: redacted token to `[REDACTED]`, amended commit, force pushed

14. **Documentation Created**
    - `DEV_LOG.md` — Chronological development log
    - `HANDOFF.md` — Agent instruction manual with failure scenarios
    - `PROJECT_REPORT.md` — 12-section technical report
    - `README.md` — Professional GitHub README with badges

---

## Hardware

- **Training machine**: Apple M4 Max, 128GB unified memory
  - GPU allocation: `sudo sysctl iogpu.wired_limit_mb=129024`
  - `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
- **Test machine** (this checkpoint): Apple M3 Max, 128GB unified memory
  - Same GPU allocation command needed

---

## Training Hyperparameters

### Stage 1 (Completed)
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-3 |
| Batch Size | 64 (via grad_accum=32, micro_batch=2) |
| Steps | 1000 |
| Warmup | 200 steps |
| Trainable | Projector only (33.6M params) |
| Dataset | HuggingFaceM4/the_cauldron |
| Image Size | 672×672 |
| Max Length | 1024 tokens |

### Stage 1.5 (In Progress — this checkpoint is from here)
| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-4 (cosine decay) |
| Batch Size | 64 (via grad_accum=32) |
| Steps | 5000 |
| Warmup | 200 steps |
| Trainable | Projector + LoRA (33.6M + ~73M params) |
| LoRA rank | 32, alpha=64, dropout=0.05 |
| Text mixing | 20% text-only samples |
| Dataset | HuggingFaceM4/the_cauldron |
| Image Size | 672×672 |
| Max Length | 1024 tokens |
| Save every | 500 steps |

### Stage 2 (Not Started)
| Parameter | Value |
|-----------|-------|
| Dataset | AGUVIS (grounding + agentic) |
| Risk | HIGH — field names are guessed, needs validation |

---

## Key Design Decisions and Why

1. **Why Qwen3-VL's ViT?** — `out_hidden_size=4096` matches Sarvam's `hidden_size=4096`. No dimension bridging needed, projector acts as alignment/refinement layer.

2. **Why pooler_output not last_hidden_state?** — `pooler_output` is 4096-dim (post spatial merge), `last_hidden_state` is 1152-dim (pre-merge). The 4096 matches Sarvam directly.

3. **Why freeze router + routed experts?** — MoE models are extremely sensitive to routing changes. Unfreezing 128 experts would destabilize the expert selection patterns Sarvam learned during pretraining.

4. **Why 20% text mixing?** — VILA research shows text-only mixing during multimodal training prevents catastrophic forgetting of language capabilities.

5. **Why the_cauldron dataset?** — It has embedded PIL images (not file paths). LLaVA-Pretrain and ShareGPT4V have string paths pointing to images that aren't bundled, causing FileNotFoundError.

6. **Why nohup pipeline?** — Training takes 10+ days. Claude sessions disconnect. The bash pipeline (`run_all_stages.sh`) chains Stage 1 → 1.5 → 2 and survives session loss.

---

## Known Issues & Risks

1. **AGUVIS Dataset (Stage 2)**: Field names in `training/data/aguvis_dataset.py` are GUESSED from documentation. When Stage 2 starts, it will likely crash with KeyError on dataset fields. The AI agent must inspect the actual dataset structure and fix field mappings.

2. **Memory on M3 Max**: Should work (same 128GB as M4 Max) but M3 has slightly less memory bandwidth. Inference may be ~20% slower.

3. **MPS Backend**: Some PyTorch operations may not be fully supported on MPS. If you see "MPS fallback" warnings, they're usually harmless but can slow things down.

4. **Sarvam-30B trust_remote_code**: The model uses custom code (`configuration_sarvam_moe.py`, `modeling_sarvam_moe.py`). Always pass `trust_remote_code=True`.
