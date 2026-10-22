"""Builds notebooks/cloud_run.ipynb (Kaggle + Colab). Edit the cells here, then:
    python scripts/build_notebook.py notebooks/cloud_run.ipynb
"""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])

md_intro = r"""# position-fragility: cloud run (Kaggle or Colab, free GPU)

Runs the pilot (or your shard of the main run) and saves all outputs. **Run all cells top to bottom.**

### One-time setup
**1. GitHub token** (the repo is private). On github.com: Settings → Developer settings → Personal access tokens →
*Fine-grained tokens* → Generate. Repository access: *Only select repositories* → `position-fragility`.
Permissions: *Contents: Read-only*. Copy the token.

**2a. Kaggle** (recommended: 30 free GPU h/week, runs in the background):
- New notebook → File → Import notebook → upload this file.
- Right panel → *Session options*: **Accelerator = GPU T4 ×2** (not P100: too old for vLLM), **Internet = On**
  (needs one-time phone verification).
- Add-ons → Secrets → add `GITHUB_TOKEN` = your token, tick it for this notebook.
- To run with the browser closed: **Save Version → Save & Run All (Commit)**. Results appear under the
  version's *Output* tab (`results_*.zip`).

**2b. Colab**: File → Upload notebook. Runtime → Change runtime type → **T4 GPU**.
Key icon (left) → add secret `GITHUB_TOKEN`, enable notebook access. Outputs go to your Google Drive
(`MyDrive/position-fragility/`), so a disconnect loses nothing: reconnect and *Run all* again, it resumes.
Keep the tab open (free Colab disconnects idle sessions).
"""

settings = r"""# ---- settings: the only cell you normally edit ----
CONFIGS = ["pilot_qwen3_8b_nothink", "pilot_qwen3_8b_think"]   # config names in configs/ (no .yaml)
SHARD, NUM_SHARDS = 0, 1        # main run split across teammates: person k sets SHARD=k, NUM_SHARDS=4
REPO, BRANCH = "Sky00raider/position-fragility", "main"
"""

setup = r"""import os, sys, subprocess, shutil, glob, time, pathlib

ON_KAGGLE = os.path.exists("/kaggle")
try:
    import google.colab  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False
WORK = "/kaggle/working" if ON_KAGGLE else "/content" if ON_COLAB else os.getcwd()
REPO_DIR = f"{WORK}/position-fragility"
print("platform:", "kaggle" if ON_KAGGLE else "colab" if ON_COLAB else "local", "| repo dir:", REPO_DIR)

def secret(name):
    if ON_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    if ON_COLAB:
        from google.colab import userdata
        return userdata.get(name)
    return os.environ[name]

def git(*args):
    # run git with the token in the URL, never print it
    tok = secret("GITHUB_TOKEN")
    url = f"https://x-access-token:{tok}@github.com/{REPO}.git"
    r = subprocess.run([a.replace("{URL}", url) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).replace(tok, "***"))

if not os.path.exists(f"{REPO_DIR}/.git"):
    git("git", "clone", "-b", BRANCH, "{URL}", REPO_DIR)
else:
    git("git", "-C", REPO_DIR, "pull", "{URL}", BRANCH)
print(subprocess.run(["git", "-C", REPO_DIR, "log", "--oneline", "-1"], capture_output=True, text=True).stdout)

if ON_COLAB:   # keep data and results on Google Drive so a disconnect loses nothing
    from google.colab import drive
    drive.mount("/content/drive")
    PERSIST = "/content/drive/MyDrive/position-fragility"
    for sub in ["runs", "data/items", "data/raw", "logs"]:
        target, link = f"{PERSIST}/{sub}", f"{REPO_DIR}/{sub}"
        os.makedirs(target, exist_ok=True)
        if not os.path.islink(link):
            if os.path.isdir(link):
                for f in os.listdir(link):          # move anything already there
                    shutil.move(f"{link}/{f}", target)
                os.rmdir(link)
            os.makedirs(os.path.dirname(link), exist_ok=True)
            os.symlink(target, link)
os.chdir(REPO_DIR)
"""

install = r"""# vLLM brings its matching CUDA PyTorch; then our requirements (does not downgrade transformers)
!pip install -q vllm
!pip install -q -r requirements.txt
!python -m spacy download en_core_web_sm -q
!python -c "import torch, transformers, vllm; from vllm import LLM; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
!nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
"""

gpu_check = r"""out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.split()
caps = [float(c) for c in out]
assert caps, "No GPU. Kaggle: Accelerator = GPU T4 x2. Colab: Runtime -> T4 GPU."
assert min(caps) >= 7.0, f"GPU compute capability {caps} too old for vLLM (P100?). Pick T4."
N_GPUS = len(caps)
print("GPUs:", N_GPUS, caps)
"""

prep = r"""if not (os.path.exists("data/items/musique_dev_34hop.jsonl") and os.path.exists("data/items/musique_train_34hop.jsonl")):
    subprocess.run([sys.executable, "-m", "src.pipeline", "prepare-data"], check=True)   # ~7 min, CPU
else:
    print("prepared items already present")
"""

run = r"""# one shard config per run config; configs are spread over the GPUs and run in parallel
os.makedirs("logs", exist_ok=True)
STEPS = "baseline eligible inject continue score"
per_gpu = {g: [] for g in range(N_GPUS)}
for i, name in enumerate(CONFIGS):
    cfg = f"configs/_cloud_{name}_s{SHARD}of{NUM_SHARDS}.yaml"
    pathlib.Path(cfg).write_text(f"extends: {name}.yaml\ndata:\n  shard: {SHARD}\n  num_shards: {NUM_SHARDS}\n")
    per_gpu[i % N_GPUS].append((name, cfg))

procs = []
for g, jobs in per_gpu.items():
    if not jobs:
        continue
    script = []
    for name, cfg in jobs:
        script.append(f"echo '##### {name} on GPU {g}'")
        script.append(f"for s in {STEPS}; do python -m src.pipeline $s --config {cfg} || exit 1; done")
        script.append(f"python -m src.pipeline estimate --config {cfg} --n-items 2000 || true")
    log = open(f"logs/cloud_gpu{g}_s{SHARD}of{NUM_SHARDS}.log", "a")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TQDM_MININTERVAL="60")
    procs.append((g, subprocess.Popen(["bash", "-c", " && ".join(script)], stdout=log, stderr=subprocess.STDOUT, env=env), log.name))
    print(f"GPU {g}: {[n for n, _ in jobs]}  -> log {log.name}")

def tail(path, n=3):
    lines = [l for l in open(path, errors="ignore").read().splitlines() if l.strip()]
    return "\n    ".join(lines[-n:])

while any(p.poll() is None for _, p, _ in procs):
    time.sleep(300)
    print(time.strftime("%H:%M"), *[f"\n  GPU {g} ({'running' if p.poll() is None else 'exit ' + str(p.returncode)}):\n    {tail(log)}" for g, p, log in procs])
for g, p, log in procs:
    print(f"GPU {g}: exit code {p.returncode}" + ("" if p.returncode == 0 else f"  <-- FAILED, see {log}:\n    {tail(log, 30)}"))
"""

analysis = r"""# analysis over every folder of these configs (all shards present here), then zip the results
sys.path.insert(0, REPO_DIR)
from src.config import load_config
dirs = []
for name in CONFIGS:
    cfg_path = f"configs/_cloud_{name}_s{SHARD}of{NUM_SHARDS}.yaml"
    cfg = load_config(cfg_path)
    if os.path.exists(cfg.run_dir / "scored.jsonl"):     # 5 examples to eyeball, from this shard
        subprocess.run(f"python scripts/sanity.py --config {cfg_path} --n 5 > {cfg.run_dir}/sanity.txt 2>&1", shell=True)
    dirs += sorted(d for d in glob.glob(f"runs/{cfg.run_name}*") if os.path.exists(f"{d}/scored.jsonl"))
dirs = [os.path.relpath(d) for d in dict.fromkeys(dirs)]
print("run folders:", dirs)
if dirs:
    subprocess.run([sys.executable, "-m", "src.analysis", "--runs", *dirs, "--out", "runs/analysis_cloud"])
stamp = time.strftime("%Y%m%d_%H%M")
zip_base = f"{WORK}/results_s{SHARD}of{NUM_SHARDS}_{stamp}"
staging = f"/tmp/results_{stamp}"
os.makedirs(staging, exist_ok=True)
for d in dirs + ["runs/analysis_cloud", "logs"]:
    if os.path.exists(d):
        shutil.copytree(d, f"{staging}/{d}", dirs_exist_ok=True)
shutil.make_archive(zip_base, "zip", staging)
print("results:", zip_base + ".zip", "(Kaggle: Output tab; Colab: Files panel, or Drive copy below)")
if ON_COLAB:
    shutil.copy(zip_base + ".zip", "/content/drive/MyDrive/position-fragility/")
"""

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}

nb = {
    "cells": [md(md_intro), code(settings), code(setup), code(install), code(gpu_check), code(prep),
              md("### Run (long): progress is printed every 5 minutes"), code(run),
              md("### Analysis and results zip"), code(analysis)],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}, "accelerator": "GPU"},
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", OUT)
