#!/bin/bash
# SarvamOmni Checkpoint 1 — Setup Script
# Run this on the target machine to set up the environment.

set -e

echo "============================================"
echo "  SarvamOmni Checkpoint 1 — Setup"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
PYTHON=${PYTHON:-python3}
echo ""
echo "Using Python: $($PYTHON --version 2>&1)"

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo ""
    echo "[1/4] Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "[1/4] Virtual environment already exists."
fi

# Activate
source venv/bin/activate
echo "  Activated: $(which python)"

# Install deps
echo ""
echo "[2/4] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Done."

# Check for Sarvam-30B
echo ""
echo "[3/4] Checking for Sarvam-30B base model..."

SARVAM_PATH=""
# Check common locations
for candidate in \
    "$SCRIPT_DIR/../sarvam-30b-backup" \
    "$SCRIPT_DIR/../sarvam-30b" \
    "/Users/$(whoami)/Projects/tqllm/models/sarvam-30b-backup" \
    "/Users/$(whoami)/Models/sarvam-30b-backup" \
    "/Volumes/*/sarvam-30b-backup"; do
    if [ -d "$candidate" ] && [ -f "$candidate/config.json" ]; then
        SARVAM_PATH="$(cd "$candidate" && pwd)"
        break
    fi
done

if [ -z "$SARVAM_PATH" ]; then
    echo ""
    echo "  WARNING: Sarvam-30B base model not found!"
    echo "  Please set the path manually:"
    echo ""
    echo "    export SARVAM_MODEL_PATH=/path/to/sarvam-30b-backup"
    echo ""
    echo "  Or place it next to this folder as 'sarvam-30b-backup/'"
    echo ""
    read -p "  Enter path to Sarvam-30B now (or press Enter to skip): " SARVAM_PATH
fi

if [ -n "$SARVAM_PATH" ] && [ -d "$SARVAM_PATH" ]; then
    echo "  Found: $SARVAM_PATH"

    # Create config.json
    cat > config.json <<EOF
{
    "sarvam_model_path": "$SARVAM_PATH",
    "vision_dir": "$SCRIPT_DIR/models/qwen3-vl-vit",
    "device": "mps",
    "dtype": "bfloat16"
}
EOF
    echo "  Config saved to config.json"

    # Fix adapter paths
    echo ""
    echo "[4/4] Fixing LoRA adapter paths..."
    python fix_paths.py --sarvam-path "$SARVAM_PATH"
else
    echo "  Skipped — you'll need to set SARVAM_MODEL_PATH later."
    echo "[4/4] Skipping path fix (no Sarvam path set)."
fi

# GPU memory allocation reminder
echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  IMPORTANT: Allocate GPU memory before running:"
echo "    sudo sysctl iogpu.wired_limit_mb=129024"
echo "    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0"
echo ""
echo "  Then test with:"
echo "    source venv/bin/activate"
echo "    python test_checkpoint.py --smoke-test"
echo ""
