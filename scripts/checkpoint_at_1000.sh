#!/bin/bash
# Polls for Stage 1.5 to reach step 1000, then copies all checkpoints to "Checkpoint 1"
cd "/Users/arjun/Projects/Sarvam Omni"

TARGET_DIR="Checkpoint 1"
LOG="logs/full_pipeline.log"

echo "Waiting for Stage 1.5 step 1000..."

while true; do
  LATEST=$(grep "^Step " "$LOG" 2>/dev/null | tail -1 | awk '{print $2}')
  if [ -n "$LATEST" ] && [ "$LATEST" -ge 1000 ] 2>/dev/null; then
    echo "Step $LATEST reached! Creating checkpoint..."
    break
  fi
  sleep 60
done

mkdir -p "$TARGET_DIR"

# Copy all checkpoints
cp -r checkpoints/ "$TARGET_DIR/checkpoints/" 2>/dev/null

# Copy all model code
cp -r sarvam_omni/ "$TARGET_DIR/sarvam_omni/"
cp -r training/ "$TARGET_DIR/training/"
cp -r inference/ "$TARGET_DIR/inference/"
cp -r scripts/ "$TARGET_DIR/scripts/"
cp -r configs/ "$TARGET_DIR/configs/"
cp -r models/ "$TARGET_DIR/models/"

# Copy project files
cp pyproject.toml "$TARGET_DIR/"
cp README.md "$TARGET_DIR/"

# Copy logs
cp -r logs/ "$TARGET_DIR/logs/" 2>/dev/null

echo "Done! Checkpoint saved to: $(pwd)/$TARGET_DIR"
echo "Contents:"
du -sh "$TARGET_DIR"/*
