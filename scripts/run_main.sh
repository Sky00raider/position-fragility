#!/usr/bin/env bash
# Main experiment, all models/modes in one go (GPU box, venv active, repo root):
#
#     bash scripts/overnight.sh          # recommended: detached + sleep blocked, see that file
#     bash scripts/run_main.sh           # foreground
#
# Order: fast runs first, so a failure late at night does not cost them:
#   Qwen3-8B nothink (~0.5 h) -> Llama-3.1-8B nothink (~0.5 h) -> Qwen3-8B think (~5-6 h)
# Every step resumes where it stopped: re-run the same command after a crash or a stop.
set -uo pipefail
export TQDM_MININTERVAL=60

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(qwen3_8b_nothink llama31_8b_nothink qwen3_8b_think)
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p logs
LOG="logs/main_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== main run $STAMP on $(hostname): ${CONFIGS[*]} ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || { echo "no GPU visible"; exit 1; }
python -c "import vllm; from vllm import LLM; print('vllm', vllm.__version__)" || { echo "vLLM not importable"; exit 1; }
git rev-parse --short HEAD

HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
FREE_GB=$(df -Pk "$HF_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
NEED=2   # outputs + caches
[[ " ${CONFIGS[*]} " == *llama* ]] && [ ! -d "$HF_DIR/hub/models--hugging-quants--Meta-Llama-3.1-8B-Instruct-AWQ-INT4" ] && NEED=$((NEED + 6))
[[ " ${CONFIGS[*]} " == *qwen* ]] && [ ! -d "$HF_DIR/hub/models--Qwen--Qwen3-8B-AWQ" ] && NEED=$((NEED + 6))
echo "free disk: ${FREE_GB} GB, need ~${NEED} GB"
[ "$FREE_GB" -lt "$NEED" ] && { echo "!! not enough disk: free space (pip cache purge) and rerun"; exit 1; }

if [ ! -f data/items/musique_dev_34hop.jsonl ] || [ ! -f data/items/musique_train_34hop.jsonl ]; then
  python -m src.pipeline prepare-data || exit 1
fi

run_dir() { python -c "from src.config import load_config; print(load_config('$1').run_dir)"; }
FAILED=()
for name in "${CONFIGS[@]}"; do
  cfg="configs/${name}.yaml"
  rdir=$(run_dir "$cfg")
  echo; echo "################ $name -> $rdir  ($(date +%H:%M)) ################"
  ok=1
  for step in baseline eligible inject continue score; do
    echo "--- $step ($(date +%H:%M))"
    python -m src.pipeline "$step" --config "$cfg" || { echo "!! FAILED: $name $step"; FAILED+=("$name: $step"); ok=0; break; }
  done
  [ $ok = 1 ] && python scripts/sanity.py --config "$cfg" --n 5 > "$rdir/sanity.txt" 2>&1
done

RUNS=()
for name in "${CONFIGS[@]}"; do rdir=$(run_dir "configs/${name}.yaml"); [ -f "$rdir/scored.jsonl" ] && RUNS+=("$rdir"); done
[ ${#RUNS[@]} -gt 0 ] && python -m src.analysis --runs "${RUNS[@]}" --out runs/analysis_main

OUT="main_results_$(hostname)_${STAMP}.tar.gz"
tar czf "$OUT" "$LOG" "${RUNS[@]}" runs/analysis_main 2>/dev/null
echo
if [ ${#FAILED[@]} -gt 0 ]; then printf '!! failed: %s\n' "${FAILED[@]}"; fi
echo "=== done $(date +%H:%M). Send this file to the team: $OUT ($(du -h "$OUT" | cut -f1)) ==="
