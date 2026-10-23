#!/usr/bin/env bash
# Stop a running pilot/main run and free the GPU. Finished work is kept: every step resumes.
pkill -f "scripts/run_main.sh" ; pkill -f "scripts/run_pilot.sh"
pkill -f "src.pipeline" ; pkill -f "scripts/gpu_smoke.py"
sleep 5
pkill -9 -f "src.pipeline" 2>/dev/null
pkill -9 -f "EngineCore" 2>/dev/null
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "stopped. Resume later with: bash scripts/overnight.sh"
