#!/bin/bash
# Master training script — runs all 3 stages sequentially.
# Survives terminal disconnect. Run with: nohup bash scripts/run_all_stages.sh &
#
# Prerequisites:
#   sudo sysctl iogpu.wired_limit_mb=129024
#   cd "/Users/arjun/Projects/Sarvam Omni"
#   source .venv/bin/activate

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

cd "/Users/arjun/Projects/Sarvam Omni"
source .venv/bin/activate

mkdir -p logs checkpoints/stage1 checkpoints/stage1_5 checkpoints/stage2

echo "=========================================="
echo "SarvamOmni Full Training Pipeline"
echo "Started: $(date)"
echo "=========================================="

# ── STAGE 1: Projector Pre-training ──────────────────────────
echo ""
echo "[$(date)] STAGE 1: Projector Pre-training"
echo "  Dataset: HuggingFaceM4/the_cauldron"
echo "  Effective BS=64, LR=1e-3, warmup=1000"
echo ""

python -u training/train_projector.py \
  --dataset "HuggingFaceM4/the_cauldron" --config "" \
  --device mps --dtype bfloat16 \
  --lr 1e-3 --grad-accum 64 --warmup-steps 1000 \
  --save-every 200 --log-every 5 \
  --max-length 512 --image-size 672 \
  2>&1 | tee logs/stage1.log

echo "[$(date)] Stage 1 complete!"

# Check if projector was saved
if [ ! -f "checkpoints/stage1/projector_final.pt" ] && [ ! -f "checkpoints/stage1/projector_best.pt" ]; then
  echo "ERROR: No Stage 1 checkpoint found!"
  exit 1
fi

# Use best if available, else final
PROJ_PATH="checkpoints/stage1/projector_best.pt"
if [ ! -f "$PROJ_PATH" ]; then
  PROJ_PATH="checkpoints/stage1/projector_final.pt"
fi
echo "  Using projector: $PROJ_PATH"

# ── STAGE 1.5: Mid-training with LoRA ────────────────────────
echo ""
echo "[$(date)] STAGE 1.5: Mid-training with LoRA r=32"
echo "  Projector from: $PROJ_PATH"
echo "  LR=2e-4, 20% text mixing, max 5000 steps"
echo ""

python -u training/train_midstage.py \
  --projector "$PROJ_PATH" \
  --dataset "HuggingFaceM4/the_cauldron" --config "" \
  --device mps --dtype bfloat16 \
  --lr 2e-4 --grad-accum 32 --warmup-steps 200 \
  --max-steps 5000 --save-every 500 --log-every 10 \
  --max-length 1024 --image-size 672 \
  --lora-r 32 --lora-alpha 64 \
  --text-ratio 0.2 \
  2>&1 | tee logs/stage1_5.log

echo "[$(date)] Stage 1.5 complete!"

# Check checkpoint
STAGE15_DIR="checkpoints/stage1_5/best"
if [ ! -d "$STAGE15_DIR" ]; then
  STAGE15_DIR="checkpoints/stage1_5/final"
fi
echo "  Using Stage 1.5: $STAGE15_DIR"

# ── STAGE 2: Merged Agentic Fine-tuning ──────────────────────
echo ""
echo "[$(date)] STAGE 2: Merged Agentic Fine-tuning"
echo "  40% grounding + 40% agentic + 20% text"
echo "  LR=5e-5, max 3000 steps"
echo ""

python -u training/train_grounding.py \
  --projector-path "$STAGE15_DIR/projector.pt" \
  --device mps --dtype bfloat16 \
  --lr 5e-5 --grad-accum 16 --warmup-steps 100 \
  --max-steps 3000 --save-every 500 --log-every 10 \
  --max-length 2048 --image-size 672 \
  --lora-r 32 --lora-alpha 64 \
  2>&1 | tee logs/stage2.log

echo "[$(date)] Stage 2 complete!"

echo ""
echo "=========================================="
echo "ALL TRAINING COMPLETE!"
echo "Finished: $(date)"
echo "=========================================="
echo ""
echo "Checkpoints:"
echo "  Stage 1:   checkpoints/stage1/projector_best.pt"
echo "  Stage 1.5: checkpoints/stage1_5/best/"
echo "  Stage 2:   checkpoints/stage2/best/"
echo ""
echo "To run inference:"
echo "  python inference/demo.py \\"
echo "    --projector checkpoints/stage2/best/projector.pt \\"
echo "    --lora checkpoints/stage2/best/lora"
