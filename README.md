# position-fragility

Error-position injection study on MuSiQue 3/4-hop (reduced scope of `docs/design_v2.md`).
Read `CLAUDE.md` for the scope and design rules.

We plant a wrong answer at hop *k* of the model's **own** reasoning chain (assistant-turn
prefill), let it continue, and measure whether it still reaches the right final answer.

## 1. Setup

### Dev box (Windows, CPU only, PowerShell)
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m spacy download en_core_web_sm
copy .env.example .env      # then put your HF token in .env
```
Below, `python` means `.venv\Scripts\python` (or activate the venv first).

Step-by-step checklist for the GPU owner: **`docs/GPU_SETUP.md`**.

### GPU box, native Linux (RTX 5070 needs CUDA 12.8+ builds)
Ubuntu 24.04 (Python 3.12) recommended.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt     # vLLM first: it pins a matching CUDA torch
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env                    # add HF_TOKEN
nvidia-smi                              # driver must support CUDA 12.8+
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
```
The 5070 should report capability `(12, 0)` and CUDA `12.8` or newer. If vLLM's default wheel
does not support sm_120, install the CUDA 12.8 wheel from the vLLM install docs
(https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html).

### GPU box, Windows via WSL2 (vLLM does not run on native Windows)
1. On Windows: install the latest NVIDIA Game Ready/Studio driver (it provides CUDA inside WSL;
   do **not** install a Linux NVIDIA driver inside WSL).
2. PowerShell (admin): `wsl --install -d Ubuntu-24.04`, reboot, open Ubuntu.
3. In Ubuntu: `nvidia-smi` must show the 5070. Then:
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-pip git gh
   gh auth login && git clone https://github.com/Sky00raider/position-fragility.git && cd position-fragility
   ```
   and follow the native Linux steps above.
4. Keep the repo inside the Linux filesystem (`~/`), not `/mnt/c/` (much faster model loading).

### First thing on the GPU box: the pilot (one command)
```bash
source .venv/bin/activate
bash scripts/run_pilot.sh                              # Qwen3-8B nothink + think, 50 dev items each
bash scripts/run_pilot.sh pilot_llama31_8b_nothink     # optional, e.g. on the 3070
```
It runs the GPU smoke test, then the full pipeline on 50 dev items per config (pilot configs
`configs/pilot_*.yaml` inherit every setting from the main config), the analysis and a GPU-hours
estimate, and writes `pilot_results_<host>_<time>.tar.gz`. **Send that file to the team.**
It is safe to re-run after a crash. vLLM pre-allocates `gpu_memory_utilization` of VRAM, so
"VRAM used" will read ~90%. No HF token is needed: both AWQ model repos are public.

## 2. Data
```bash
python -m src.pipeline prepare-data            # downloads MuSiQue to data/raw/, writes data/items/
```

## 3. Pipeline (one config = one model x mode)
```bash
C=configs/qwen3_8b_nothink.yaml
python -m src.pipeline baseline --config $C    # own chain + closed-book per item
python -m src.pipeline eligible --config $C    # doc §6.1 filter, logs reasons
python -m src.pipeline inject   --config $C    # builds all cells, logs dropped cells
python -m src.pipeline continue --config $C    # runs all cells (resumable)
python -m src.pipeline score    --config $C    # EM/F1, adopted
python scripts/sanity.py --config $C --n 5     # eyeball 5 injected prompts + continuations
```
Every step can be re-run after a crash: it skips work already in its output file.

CPU debug (pipeline test only, meaningless numbers):
```powershell
$C = "configs/cpu_qwen3_0p6b_nothink.yaml"
python -m src.pipeline baseline --config $C
...
```

Compute estimate (after the GPU pilot):
```bash
python -m src.pipeline estimate --config $C --tok-per-s <batched tok/s> --n-items 800
```

## 4. Analysis
```bash
python -m src.analysis --runs runs/qwen3_8b_nothink runs/qwen3_8b_think runs/llama31_8b_nothink
```
Writes CSVs to `runs/analysis/tables/` and PNGs to `runs/analysis/figures/`.

## 5. Tests
```bash
python -m pytest -q
```

## Outputs (runs/<run_name>/)
| file | contents |
|---|---|
| baseline.jsonl | own clean chain + closed-book per item |
| eligibility.jsonl | per item: eligible or the reason it failed |
| cells.jsonl | every cell to run (prompt ids, prefix, planted value) |
| dropped.jsonl | cells dropped by validity rules, with reason |
| continuations.jsonl | raw continuations |
| scored.jsonl | final trial records (schema in CLAUDE.md) |
