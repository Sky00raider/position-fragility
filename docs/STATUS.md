# STATUS / handoff (read this first in a new session)

Last updated: 2026-10-24 (Sat), after drafting the Results section (think_trace results still pending).
Full decision log with numbers: `docs/HISTORY.md`. Rules: `CLAUDE.md`. Design doc: `docs/design_v2.md`.

## ⚠ REMIND THE USER FIRST (every session until done)
Ask the user to get their friend (GPU box owner) to run the v2 command below; it includes
think_trace. Friend was unavailable on Oct 24. Experiments close Oct 27: the run must start by the
night of Oct 26 at the latest. If it can't, submit with v1 + the audit disclosure (paper already
written that way).

## Latest (24 Oct night)
v2 gate FAILED on the GPU box (judge prompt v1 too weak; also the 0.75 bar was above the human ceiling
of 0.74). Fixed in code (judge prompt v2, rule chosen on tune half / checked on holdout, bar 0.65; see
HISTORY). run_main.sh now FALLS BACK automatically if the gate fails: vetted configs are replaced by
their `fallback_config` (think_trace_v2 -> v1 think_trace), others dropped. Analysis now goes to
`runs/analysis_<stamp>` (never overwrites). One command for tonight (short runs first, think_trace last):
    git pull && bash scripts/overnight.sh qwen3_8b_nothink_v2 qwen3_8b_think_v2 llama31_8b_nothink_v2 qwen3_8b_think_trace_v2
If a v1 think_trace run is already going, let it finish first, then run this (it reuses its clean cells).

## NEXT GPU RUN (v2 + think_trace in one overnight run; friend's box)
Ask the friend to run, from the repo root:
    git pull && bash scripts/overnight.sh qwen3_8b_nothink_v2 qwen3_8b_think_v2 qwen3_8b_think_trace_v2 llama31_8b_nothink_v2
It first runs the vet gate (~5 min); if the gate FAILS nothing else runs, so check the log
(`tail -f logs/main_*.log`) after ~10 min. If it says GATE FAILED, don't waste the night: run the
unvetted think_trace instead (`bash scripts/overnight.sh qwen3_8b_think_trace`) and send the log here.
Otherwise ~2.5-4 h (vet ~15-30 min of it).
Result: `main_results_<host>_<stamp>.tar.gz` (v2 runs + runs/analysis_v2 + runs/vet). Copy runs into
`runs/`, keep the tarball pristine. v1 runs/analysis_main stay untouched (fallback).
After it arrives: check gate numbers + judge acceptance + drop counts, check think_trace continuation
length (>110 tokens median), rate ~100 v2 planted values independently (CLAUDE.md rule 12),
then swap the paper's numbers to v2 (v1 + audit become a methods note / robustness).

## Paper text is FROZEN (2026-10-24, after 4 AI reviews)
Change it only for: (1) factual/numerical errors, (2) claims the data don't support, (3) new results
(v2, think_trace), (4) venue requirements (page limit, anonymization). No more review-driven polishing;
the next useful review is a teammate reading paper/main.pdf cold.

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
As of the Oct 24 session they had NOT arrived: `runs/qwen3_8b_think_trace` is only the local dry run
(baseline/cells/eligibility/dropped, no generations or scored.jsonl). Don't mistake it for results.
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
- [x] Paper: Results section drafted from runs/analysis_main (+ item-count table, setup tbds). Two
      `\tbd`s wait for think_trace: RQ4 paragraph and the Method sentence on the injection site.
      If think_trace adds panels, re-copy `runs/analysis_main/figures/recovery_linear.png` to `paper/figures/`.
- [x] Paper: Abstract, Intro, Related Work, RQs, Limitations, Conclusion drafted (53cf62a). 9 `\tbd`s
      left: authors, repo URL, hand audit (2), think_trace (5). Not compiled yet (Overleaf); ~5.5k words,
      check the venue page limit. Remaining `% VERIFY` in refs.bib: lanham, chen2025, yee venue.
- [x] Review work done (Sonnet + Gemini, HISTORY 2026-10-24): branching RQ2, mechanism, hedges, rework.
      Paper = 8 pages with page 8 FULL; think_trace text will push to 9 -> trim once page limit is known.
      Rebuild: Tectonic is in the old session scratchpad (re-download v0.15.0 if gone); figure for the
      paper = figures/recovery_linear_paper.png (re-copy from runs/analysis_v2/figures after v2).
- [x] Compiled locally with Tectonic: 8 pages, layout fixed, `paper/main.pdf` (HISTORY). [ ] Trim to page limit (VENUE LIMIT UNKNOWN);
      team fills authors + repo URL; verify remaining refs
- [x] DONE (200-value version, see HISTORY) OPTIONAL: plausibility check on a random 200–400 planted values
      from Qwen nothink K=3 linear, k=1 vs k=2 (Sonnet subagent rater, same criteria as audit), then PSI
      on plausible-only cells. ~70–200k tokens. Not required: tag-group check already in the paper.
- [x] DONE (kappa type 0.82, plausible 0.57; HISTORY) Hand audit: now ONE teammate fills `docs/audit/audit_sheet_B.csv` (rater A = Sonnet, done) → merge → `python scripts/make_audit.py --score <filled csv>` → Method numbers
- [ ] Keep `../main_results_*` backed up (Google Drive)

## GPU box notes (friend's machine)
Native Arch Linux, Python 3.14, vLLM 0.32.0, torch 2.13+cu130, RTX 5070 12 GB, ~16 GB free disk.
Scripts activate `.venv` themselves. Start: `bash scripts/overnight.sh <config names>`;
stop: `bash scripts/stop_run.sh`; logs: `logs/main_*.log`. No gaming while running (VRAM).
Measured: ~400 tok/s nothink, ~550 tok/s think (batched).
