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
cd "$(dirname "$0")/.." || exit 1
# use the project venv even if the caller forgot to activate it
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then set +u; source .venv/bin/activate; set -u; fi
export TQDM_MININTERVAL=30   # progress bars update every 30 s, so the log stays readable

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(pilot_qwen3_8b_nothink pilot_qwen3_8b_think)
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p logs
LOG="logs/pilot_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== pilot $STAMP on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "nvidia-smi failed: no GPU visible"; exit 1; }
# import the classes we actually use: `import vllm` alone is lazy and hides broken dependencies
python -c "import vllm, torch, transformers; from vllm import LLM, SamplingParams; \
print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)" \
  || { echo "vLLM not importable: run  pip install vllm==\$(pip show vllm | awk '/^Version/{print \$2}') 'transformers>=5'  and retry"; exit 1; }
pip check >/dev/null 2>&1 || { echo "WARNING: pip check reports dependency conflicts:"; pip check | head -n 10; }

# model weights go to HF_HOME (default ~/.cache/huggingface).
# Qwen3-8B-AWQ ~6 GB + vLLM/FlashInfer caches ~1.5 GB + results -> need 8 GB; Llama adds ~6 GB.
# (Weights already downloaded count as used space, so a rerun needs less.)
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_DIR"
FREE_GB=$(df -Pk "$HF_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
HAVE_QWEN=$([ -d "$HF_DIR/hub/models--Qwen--Qwen3-8B-AWQ" ] && du -sk "$HF_DIR/hub/models--Qwen--Qwen3-8B-AWQ" | awk '{print int($1/1024/1024)}' || echo 0)
echo "free disk at $HF_DIR: ${FREE_GB} GB (Qwen3-8B-AWQ already cached: ${HAVE_QWEN} GB)"
if [ $((FREE_GB + HAVE_QWEN)) -lt 8 ]; then
  echo "!! need ~8 GB free for Qwen3-8B. Free space (pip cache purge) or set HF_HOME to a bigger drive, then rerun."
  exit 1
elif [ "$FREE_GB" -lt 15 ]; then
  echo "note: under 15 GB free: enough for Qwen, not for Llama as well."
fi
git rev-parse --short HEAD

if [ ! -f data/items/musique_dev_34hop.jsonl ] || [ ! -f data/items/musique_train_34hop.jsonl ]; then
  python -m src.pipeline prepare-data || exit 1
fi

run_dir() { python -c "from src.config import load_config; print(load_config('$1').run_dir)"; }

FAILED=()
for name in "${CONFIGS[@]}"; do
  cfg="configs/${name}.yaml"
  main_cfg="configs/$(grep '^extends:' "$cfg" | awk '{print $2}')"
  rdir=$(run_dir "$cfg")
  echo; echo "################ $name -> $rdir ################"
  run() { echo "--- $*"; "$@" || { echo "!! FAILED: $*"; FAILED+=("$name: $*"); return 1; }; }
  if [ "${SKIP_SMOKE:-0}" != "1" ]; then run python scripts/gpu_smoke.py --config "$main_cfg" || continue; fi
  run python -m src.pipeline baseline --config "$cfg" || continue
  run python -m src.pipeline eligible --config "$cfg" || continue
  run python -m src.pipeline inject   --config "$cfg" || continue
  run python -m src.pipeline continue --config "$cfg" || continue
  run python -m src.pipeline score    --config "$cfg" || continue
  python scripts/sanity.py --config "$cfg" --n 5 > "$rdir/sanity.txt" 2>&1
  run python -m src.pipeline estimate --config "$cfg" --n-items 2000
done

RUNS=()
for name in "${CONFIGS[@]}"; do rdir=$(run_dir "configs/${name}.yaml"); [ -f "$rdir/scored.jsonl" ] && RUNS+=("$rdir"); done
[ ${#RUNS[@]} -gt 0 ] && python -m src.analysis --runs "${RUNS[@]}" --out runs/analysis_pilot

OUT="pilot_results_$(hostname)_${STAMP}.tar.gz"
tar czf "$OUT" "$LOG" runs/pilot*_* runs/analysis_pilot 2>/dev/null
echo
if [ ${#FAILED[@]} -gt 0 ]; then printf '!! failed step: %s\n' "${FAILED[@]}"; fi
echo "=== done. Send this file to the team: $OUT ($(du -h "$OUT" | cut -f1)) ==="
