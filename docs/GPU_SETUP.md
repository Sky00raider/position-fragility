# GPU box: setup + pilot checklist (RTX 5070)

Goal: run one script that tests the GPU and runs a small pilot, then send back one file.
Time: ~1 h setup, then the pilot runs on its own (roughly 1–3 h, mostly thinking mode).
If any step fails, **stop and send the exact error text** (screenshot is fine).

## A. Windows with WSL2 (skip to B if you boot into a Linux partition)

1. **Update the NVIDIA driver on Windows** (GeForce Experience / nvidia.com, latest Game Ready or
   Studio driver). The RTX 50 series needs a recent driver; this one driver also serves WSL.
   Do **not** install any NVIDIA driver inside Linux/WSL.
2. **PowerShell as Administrator:**
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
   Reboot if asked, open "Ubuntu 24.04" from the Start menu, create a username/password.
3. **Power settings:** set Windows to never sleep while plugged in (sleep kills the run).
4. Continue with **C** inside the Ubuntu window.

## B. Native Linux partition

Ubuntu 24.04 recommended. Install the proprietary NVIDIA driver (Software & Updates →
Additional Drivers, pick the newest "open" driver, reboot). Then continue with **C**.

## C. Inside Linux / WSL (copy-paste block by block)

```bash
# 1. GPU visible?  Must list "NVIDIA GeForce RTX 5070".
nvidia-smi

# 2. Tools
sudo apt update && sudo apt install -y python3-venv python3-pip git gh

# 3. Get the repo (private: accept the GitHub collaborator invite first)
gh auth login                      # GitHub.com -> HTTPS -> login with browser
cd ~ && git clone https://github.com/Sky00raider/position-fragility.git
cd position-fragility

# 4. Python environment (vLLM first, it brings the matching CUDA PyTorch)
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 5. Check: must print  True  NVIDIA GeForce RTX 5070  (12, 0)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

# 6. Run the pilot (downloads ~6 GB model + ~270 MB data on first run)
bash scripts/run_pilot.sh
```

If step 5 prints `False`, or vLLM errors with "no kernel image" / "sm_120", the vLLM wheel was not
built for Blackwell: send the error, we'll install the CUDA 12.8 build.

**Disk space:** at least 8 GB free (10 GB comfortable; +6 GB if also running Llama) where models download (`~/.cache/huggingface`). If the run
hangs after a "Not enough free disk space" warning: `pip cache purge` (frees several GB), or put
`export HF_HOME=/bigger/drive/hf_cache` in `~/.bashrc`, then rerun.

If vLLM fails during warm-up with `Could not find nvcc and default cuda_home='/usr/local/cuda'
doesn't exist` (FlashInfer wants to compile kernels, but only the driver is installed): `git pull`
(the code now uses vLLM's PyTorch sampler) and run `vllm download-kernels`. Installing the full
CUDA toolkit is not needed.

If vLLM fails with `cannot import name 'PreTrainedConfig' from 'transformers'`, transformers got
downgraded below what vLLM needs (fixed in requirements.txt on Oct 22). Repair with:
```bash
pip install "vllm==$(pip show vllm | awk '/^Version/{print $2}')" "transformers>=5" && pip check
```

## D. Send the result back

The last line of the pilot says `Send this file to the team: pilot_results_....tar.gz`.
In WSL, copy it to the Windows desktop and send it (WhatsApp/Drive):
```bash
cp pilot_results_*.tar.gz /mnt/c/Users/$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')/Desktop/
```

If the run crashes or the PC restarts: `cd ~/position-fragility && source .venv/bin/activate &&
bash scripts/run_pilot.sh` again. It continues where it stopped.

## E. Main run (overnight, after the pilot is approved)

```bash
cd ~/Documents/position-fragility && source .venv/bin/activate && git pull
bash scripts/overnight.sh        # Qwen normal -> Llama normal -> Qwen thinking, ~6 h total
tail -f logs/main_*.log          # optional: watch (Ctrl+C only stops watching)
```
It keeps running if you close the terminal and blocks the PC from sleeping. **Don't game while it
runs** (it uses ~80% of the GPU memory). To get the GPU back: `bash scripts/stop_run.sh`; to
continue later: `bash scripts/overnight.sh` again (finished work is kept). At the end it writes
`main_results_*.tar.gz`: send that file.
