# STATUS / handoff (read this first in a new session)

Last updated: 2026-10-24 (Sat), after building the `think_trace` condition.
Full decision log with numbers: `docs/HISTORY.md`. Rules: `CLAUDE.md`. Design doc: `docs/design_v2.md`.

## Deadlines
Experiments done **Oct 27 2026**, paper submitted **Oct 30 2026** (IEEE conference; 4-person student team).

## Where things are
| What | Where |
|---|---|
| Main-run raw outputs (pristine, never modify) | `../main_results_archlinux_20261023_2239/` (runs under `home/burr/Documents/position-fragility/runs/`) |
| Working copies of the runs (gitignored) | `runs/qwen3_8b_nothink`, `runs/qwen3_8b_think`, `runs/llama31_8b_nothink` |
| Analysis outputs (2000 bootstrap reps, reproduced locally) | `runs/analysis_main/` (tables/, figures/, report.txt, logit_*.txt, silent_explicit_*) |
| Pilot outputs | `../pilot_results_archlinux_20261023_1808/` (pilot1 = `pilot_*`, pilot2 = `pilot2_*`) |
| Clean-twin audit script (window A) | `scripts/clean_twin_audit.py` (CPU, regenerates its numbers) |
| Paper draft (Overleaf-ready, not compiled locally) | `paper/main.tex`, `paper/refs.bib` |
| Hand-audit sheet (NOT done yet, needs 2 people) | `docs/audit/audit_sheet.csv` + `docs/audit/README.md` |
| User's team folder (not created by Claude) | `../team_docs/` |

## Results so far (details in HISTORY 2026-10-24)
Eligible items: Qwen nothink 922, Qwen think 384, Llama nothink 564.
- H1 reversed: late intermediate errors recovered LESS than early (PSI < 0 everywhere; strong for nothink K=3).
- H2 inconclusive (position and remaining steps not separable; VIF ~100).
- H3 strongly supported: terminal hop ~0–15% recovery (nothink) vs ~60% intermediate.
- H4: thinking raises recovery (β=+1.52) but no shape change (interaction n.s.).
- H5 supported, small (in-context distractors hurt more, β=+0.15 for off-context).
- Recovery is almost always silent (Qwen >99%, Llama 91%; rule-based labels).
- **Problem found:** in the `think` condition the hop-summary prefill inside `<think>` makes Qwen3 stop
  deliberating (median 110 new tokens vs 1,231; clean twins only 68–86%). So `think` = "thinking
  format, no deliberation after the error".

## IN PROGRESS: `think_trace` overnight run (Oct 24 night, friend's RTX 5070)
New condition (commit 4b840fb): error planted mid-thought in the model's own thinking trace
(`configs/qwen3_8b_think_trace.yaml`, reuses the `qwen3_8b_think` baseline). Friend runs
`git pull && bash scripts/overnight.sh qwen3_8b_think_trace` (~2.5–3.5 h).

### When its results arrive (next session's first job)
1. Results land as a new folder in `Research Paper/` (e.g. `main_results_archlinux_2026102x_xxxx/`),
   run under `home/burr/Documents/position-fragility/runs/qwen3_8b_think_trace`. Copy it to
   `runs/qwen3_8b_think_trace` (keep the pristine folder).
2. **Make-or-break check:** median `n_gen_tokens` of clean/inject rows in its scored.jsonl must be
   far above 110 (real thinking). Also check clean-twin accuracy (hope ≈ 90%) and drop counts.
3. Run: `python -m src.analysis --runs runs/qwen3_8b_nothink runs/qwen3_8b_think runs/qwen3_8b_think_trace runs/llama31_8b_nothink --out runs/analysis_main --reps 2000`
   (~5–10 min CPU), plus `python scripts/clean_twin_audit.py` if useful.
4. Append results to HISTORY.md; update STATUS.md.
5. Framing: if think_trace works, the thinking story = "where the error sits matters" (summary vs
   mid-thought). If it fails (still short continuations / broken), fall back to reframing `think` as
   "thinking format without post-error deliberation" and state it as a limitation. All other
   results stand either way.

## Remaining work before Oct 30
- [ ] think_trace results → analysis → HISTORY (above)
- [ ] Paper: Results section (from runs/analysis_main tables/figures + HISTORY), Intro, Related Work,
      Abstract, Limitations; add think_trace to Method/Setup; fill `\tbd{}`; verify every `% VERIFY` in refs.bib
- [ ] Hand audit by 2 teammates → `python scripts/make_audit.py --score <filled csv>` → Method numbers
- [ ] Keep `../main_results_*` backed up (Google Drive)

## GPU box notes (friend's machine)
Native Arch Linux, Python 3.14, vLLM 0.32.0, torch 2.13+cu130, RTX 5070 12 GB, ~16 GB free disk.
Scripts activate `.venv` themselves. Start: `bash scripts/overnight.sh <config names>`;
stop: `bash scripts/stop_run.sh`; logs: `logs/main_*.log`. No gaming while running (VRAM).
Measured: ~400 tok/s nothink, ~550 tok/s think (batched).
