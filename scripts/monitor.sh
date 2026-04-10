#!/bin/bash
# Monitor training progress. Run: bash scripts/monitor.sh
cd "/Users/arjun/Projects/Sarvam Omni"

echo "SarvamOmni Training Monitor"
echo "==========================="

# Check if training is running
PID=$(ps aux | grep "train_" | grep python | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
  CMD=$(ps -p $PID -o args= | head -c 80)
  echo "Training RUNNING (PID $PID): $CMD..."
else
  echo "No training process found."
fi

echo ""

# Latest logs from each stage
for stage in stage1 stage1_5 stage2; do
  log="logs/${stage}.log"
  if [ -f "$log" ]; then
    echo "── $stage ──"
    grep "Step " "$log" | tail -3
    echo ""
  fi
done

# Checkpoints
echo "── Checkpoints ──"
for dir in checkpoints/stage1 checkpoints/stage1_5/best checkpoints/stage2/best; do
  if [ -d "$dir" ]; then
    echo "  $dir: $(ls "$dir" 2>/dev/null | wc -l | tr -d ' ') files"
  fi
done

# Memory
echo ""
echo "── System ──"
vm_stat | grep "Pages free\|Pages active\|Pages wired" | head -3
