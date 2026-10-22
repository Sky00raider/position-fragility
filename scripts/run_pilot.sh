#!/usr/bin/env bash
# One-command pilot for the GPU box (Linux or WSL2). Run from the repo root with the venv active:
#
#     bash scripts/run_pilot.sh                      # Qwen3-8B nothink + think (5070)
#     bash scripts/run_pilot.sh pilot_llama31_8b_nothink   # any pilot config name(s)
#
# For each config: GPU smoke test, then baseline -> eligible -> inject -> continue -> score,
# sanity examples and a GPU-hours estimate. Then analysis on all pilots, and everything is
# bundled into pilot_results_*.tar.gz: send that file back to the team.
# Safe to re-run after a crash: every step resumes where it stopped.
set -uo pipefail
export TQDM_MININTERVAL=30   # progress bars update every 30 s, so the log stays readable

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(pilot_qwen3_8b_nothink pilot_qwen3_8b_think)
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p logs
LOG="logs/pilot_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== pilot $STAMP on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "nvidia-smi failed: no GPU visible"; exit 1; }
python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)" \
  || { echo "vLLM not importable: see README GPU setup"; exit 1; }
git rev-parse --short HEAD

if [ ! -f data/items/musique_dev_34hop.jsonl ] || [ ! -f data/items/musique_train_34hop.jsonl ]; then
  python -m src.pipeline prepare-data || exit 1
fi

FAILED=()
for name in "${CONFIGS[@]}"; do
  cfg="configs/${name}.yaml"
  main_cfg="configs/$(grep '^extends:' "$cfg" | awk '{print $2}')"
  echo; echo "################ $name ################"
  run() { echo "--- $*"; "$@" || { echo "!! FAILED: $*"; FAILED+=("$name: $*"); return 1; }; }
  run python scripts/gpu_smoke.py --config "$main_cfg" || continue
  run python -m src.pipeline baseline --config "$cfg" || continue
  run python -m src.pipeline eligible --config "$cfg" || continue
  run python -m src.pipeline inject   --config "$cfg" || continue
  run python -m src.pipeline continue --config "$cfg" || continue
  run python -m src.pipeline score    --config "$cfg" || continue
  python scripts/sanity.py --config "$cfg" --n 5 > "runs/${name}/sanity.txt" 2>&1
  run python -m src.pipeline estimate --config "$cfg" --n-items 2000
done

RUNS=()
for name in "${CONFIGS[@]}"; do [ -f "runs/${name}/scored.jsonl" ] && RUNS+=("runs/${name}"); done
[ ${#RUNS[@]} -gt 0 ] && python -m src.analysis --runs "${RUNS[@]}" --out runs/analysis_pilot

OUT="pilot_results_$(hostname)_${STAMP}.tar.gz"
tar czf "$OUT" "$LOG" runs/pilot_* runs/analysis_pilot 2>/dev/null
echo
if [ ${#FAILED[@]} -gt 0 ]; then printf '!! failed step: %s\n' "${FAILED[@]}"; fi
echo "=== done. Send this file to the team: $OUT ($(du -h "$OUT" | cut -f1)) ==="
