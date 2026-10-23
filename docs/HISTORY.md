# Project history (lab notebook)

What we did, why, and what we saw, in order. Use it to write the paper (Method, Setup, Limitations,
Appendix) and to answer "why did you do X?" from reviewers. **Append a dated entry after every
decision or result.** Commit hashes point to the exact code. Times are IST.

Tags: **[method]** goes in Method · **[setup]** Experimental Setup · **[limit]** Limitations ·
**[appendix]** reproducibility appendix · **[result]** numbers · **[deviation]** differs from
`docs/design_v2.md`, must be stated in the paper.

---

## 2026-10-22 (Thu): scope, harness, data

### Scope reduced to meet the deadline **[deviation] [limit]**
Deadlines: experiments done Oct 27, paper submitted Oct 30. Out of the design doc we keep only
MuSiQue 3/4-hop, Qwen3-8B (thinking off/on) and optionally Llama-3.1-8B. Dropped: GSM8K math arm
(RQ7), handoff condition (E3), counterfactual subset, 8-bit quantization check, LLM judge and
explicit/silent labels (E1/E2 only qualitatively), multiple samples per cell (1 sample), Gemma,
R1-Distill, Claude API, OSF pre-registration. Rules recorded in `CLAUDE.md`.

### Design decisions taken while planning
- **Branching MuSiQue items kept, linear chains primary [method].** 3hop1/4hop1 are linear;
  3hop2/4hop2/4hop3 have independent hops merged later. "Steps remaining" is only well defined for
  linear chains, so linear = primary analysis and all items = robustness check. `remaining` is the
  number of later hops depending on hop k (transitively, from `#N` refs); equals K−k on linear chains.
- **Per-cell seeds from SHA-256** of (run, item, condition, k, error type), not Python `hash()`,
  which changes per process. Results do not depend on batching; resumed runs reproduce. **[setup]**
- **Two backends, one pipeline:** vLLM (GPU) and HF transformers (CPU dev box), same token ids.
  Prompts are tokenized once without special tokens, so Llama gets no double BOS. **[method]**
- Python 3.12 venv on the dev box (3.11 not installed); code kept 3.11-compatible.

### Harness built and tested (c9dbf5a → 3402a18)
Pipeline: `prepare-data → baseline → eligible → inject → continue → score → analysis`, every step
resumable (append-only jsonl, finished cells skipped). Tests: parsing, splice (no EOS/new user
turn, template tokens unchanged), planted value never overlaps gold, seeds stable across
processes, crash-and-resume, analysis recovers a planted effect on synthetic data.

### Data preparation **[setup] [result]**
MuSiQue-Ans v1.0 (official Google Drive zip; HF mirror `dgslibisey/MuSiQue` as fallback),
answerable 3/4-hop only.

| split | 3-hop | 4-hop | 3hop1 (lin) | 3hop2 | 4hop1 (lin) | 4hop2 | 4hop3 |
|---|---|---|---|---|---|---|---|
| dev | 760 | 405 | 568 | 192 | 246 | 64 | 95 |
| train | 4387 | 1175 | 3737 | 650 | 648 | 127 | 400 |

Dependency shapes (identical within each composition): 3hop2 = {1,2}→3; 4hop2 = {1,2}→3→4;
4hop3 = 1→2, {2,3}→4. Context = gold paragraphs + 4 fixed non-supporting paragraphs (distinct
titles), shuffled with a per-item seed (13). Data prep reproduced identically on the GPU box.

### Type tags for planted values **[method] [limit]**
Type = regex (year, 5 date formats, number) or spaCy `en_core_web_sm` NER label of the answer's
mention in its supporting paragraph. First version mistyped often (e.g. "Peter" planted for
"Hamburg", "Treaty of Paris" for "South Carolina": ~5 type errors in a 20-value sample).
Fix: keep a candidate only if its in-context label equals the label of the whole string on its own
(entity must cover the whole string). After: ~2 type errors in a 30-value sample; yield on dev
89% of hop slots get an in-context value, 97% an off-context value. The hand audit quantifies
the remaining error rate.

### Off-context values: same relation first (c72318b) **[method]**
Off-context values were often implausible ("Nintendo Entertainment System → Financial Conduct
Authority"). Now ranked by: same Wikidata relation (`X >> place of birth`) or same question opening
("what city", "when was") first, then length similarity, then a seeded tiebreak. Improves
plausibility for relation hops; free-text hops remain type-matched only. **[limit]**

### Debug runs on CPU (Qwen3-0.6B) **[appendix]**
0.6B cannot follow the hop format (copies `<sub>…</sub>` placeholders, 0/20 items with K hop
lines), so CPU runs use a flagged gold stand-in chain (`debug_gold_chain`, never in real configs)
just to exercise the code. Confirmed on real tokenizers (Qwen3-0.6B/8B, Llama-3.1): splice correct,
one BOS for Llama, Llama template inserts a fixed "Today Date: 26 Jul 2024" header.

### Outcomes: strict + secondary contained **[method]**
`correct` = SQuAD-normalized exact match vs answer + aliases (primary, as in the doc).
`adopted` = final answer equals the planted value. A CPU example showed verbose final answers
("Celebrity Big Brother will air on CBS on October 24, 1945") that strict matching misses, so we
also report `correct_contains` / `adopted_contains`: containment within the final-answer line
only, never the whole continuation.

### Analysis implemented (3402a18)
RR with clean-twin conditioning; PSI (late − early, intermediate hops only, terminal separate);
matched-remaining contrasts (linear only); exact McNemar early vs late, Holm-adjusted; one
logistic regression with item-clustered SEs and VIFs; item-bootstrap CIs (2000 reps); ±5 pp
equivalence via 90% CI; truncation table and no-truncation sensitivity subsets.

### Hand-audit sheet **[method]**
`docs/audit/audit_sheet.csv`: 100 planted values (25 per error type × K), two reviewers mark
type_ok / plausible independently; `scripts/make_audit.py --score` gives agreement and Cohen's
kappa. **Status: not done yet.**

### Paper draft (eb68597)
`paper/main.tex` (IEEEtran): Method and Experimental Setup drafted from the code, `[TBD]` for
numbers.

### GPU box setup problems (162f804, b83a68a) **[appendix]**
Friend's RTX 5070 runs native Arch Linux (not WSL), Python 3.14, vLLM 0.32.0, torch 2.13.0+cu130,
CUDA 13 driver, no CUDA toolkit.
1. Our `transformers<5` pin downgraded transformers after vLLM installed it → vLLM import error.
   Fixed: no upper pin; code works with transformers v4.56+ and v5.
2. Disk full during the model download (needs ~8 GB free).
3. FlashInfer's sampler JIT-compiles with `nvcc` (absent) → crash at warm-up. Fixed: use vLLM's
   PyTorch sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`); same sampling, nothing compiled.

Backup prepared: `notebooks/cloud_run.ipynb` (Kaggle/Colab, free T4, sharding). Not used so far.

---

## 2026-10-23 (Fri): GPU pilots, prompt v2, main-run plan

### GPU memory (f3f6ba6) **[setup] [appendix]**
Single request worked (≈100–105 tok/s); 10 at once crashed allocating in the AWQ (Marlin) kernel:
the 12 GB card also drives the desktop. Fixed: `gpu_memory_utilization` 0.90 → 0.80 and
`max_num_batched_tokens` 4096. KV cache then holds ~18,400 tokens (≈2.25 full-length requests),
which limits thinking-mode concurrency. Loading warns that bf16 norms/scales are cast to fp16
(standard for AWQ); outputs looked normal.

### Pilot 1: original prompt (Qwen3-8B AWQ, 50 dev items, 25 per K) **[result]**
Smoke test: batched 224 tok/s (nothink) / 390 tok/s (think); K hop lines in 5/10 and 3/10 outputs.

| | nothink | think |
|---|---|---|
| final answer correct | 24/50 | 24/50 |
| closed-book correct (memorized) | 4/50 | 2/50 |
| exactly K hop lines | 19/50 | 15/50 (0/50 inside `<think>`) |
| eligible | **0/50** | **3/50** |
| clean twins correct | – | 11/11 |
| injected cells correct | – | 16/21 (76%) |

Why so few eligible: the model split questions into 4–11 (once 37) hops with side questions
("When did Pope John XXIII die?") and answered hops in full sentences; hop answers that were right
but wordier than gold ("Da Nang, Vietnam" vs `Da Nang`, "22 times" vs `22`) failed F1 ≥ 0.8.

### Prompt v2 + hop matching (b54698d) **[method] [deviation]**
Prompt now states "This question needs exactly {K} hops", asks for short hop answers ("a name, date
or number, not a sentence") and shows one invented 3-hop example (Eiffel Tower → Paris → France →
French; not from MuSiQue). Same prompt in every condition. Hop matching: drop parentheticals; EM or
F1 ≥ 0.8; or whole-word containment with ≤3 extra words. The doc's eligibility rule was EM/alias or
F1 ≥ 0.8 only, so this is a stated relaxation. Pilot runs renamed `pilot2_*` so old outputs are not
resumed. Telling the model K is a scaffold; it is identical across positions and modes.

### Pilot 2: prompt v2 **[result]**

| | nothink | think |
|---|---|---|
| exactly K hop lines | **50/50** | **47/50** (14/50 also inside `<think>`) |
| final answer correct | 20/50 | 23/50 |
| closed-book correct | 3/50 | 3/50 |
| baseline truncated (4096 cap) | 0/50 | 3/50 (6%) |
| eligible | 2/50 → **3/50 (6%)** after the alignment fix below | **10/50 (20%)**: 3-hop 8/25, 4-hop 2/25 |
| clean twins correct | 7/7 | 31/32 (97%) |
| injected cells correct (recovery) | 4/13 (**31%**) | 52/62 (**84%**) |
| adopted planted value | 3/13 | 7/62 |
| mean tokens: baseline / continuation | 78 / 42 | 2174 / 249 |

Remaining failures are mostly genuine: wrong final answer (60% nothink) or a different reasoning
route than MuSiQue's decomposition ("American League" vs gold `Major League Baseball`; a last hop
that restates the question). These must stay excluded, otherwise error positions are not exact.

### Hop alignment fixes (f512334) **[method]**
- Containment in either direction ("Vivaldi" ~ `Antonio Vivaldi`, "15th century" ~ `built in the
  15th century`).
- Branching items: any hop order that respects the gold dependencies is accepted (e.g. 3hop2
  answered as hop 2, hop 1, hop 3); `hop_map` records model position → gold hop, so the planted type
  and `remaining` come from the right gold hop. Orders that answer a hop before its inputs are
  rejected (`hop_order_violates_dependencies`).

### Thinking-mode injection works as designed **[method] [result]**
With the hop lines prefilled inside `<think>`, the model continues in hop format inside the think
block (answers the open question in doc §6.6). Qualitative pattern worth a figure: it often
follows the planted value while thinking, then after `</think>` rewrites the chain with the correct
entity and answers correctly, without flagging the error ("silent recovery", doc E1). Example: item
`3hop1__…Paenitentiam agere`, planted "Colin Bateman" for "Pope John XXIII".

### Main-run plan (f512334) **[setup]**
Sized from pilot-2 eligibility (nothink ~6%; think 3-hop 32%, 4-hop 8%):
- Qwen3-8B nothink: all 6,727 items (cheap: ~1–1.5 h).
- Qwen3-8B think: 600 three-hop + all 1,580 four-hop = 2,180 items (~5.5–6 h). 4-hop is oversampled
  because of its low yield.
- Llama-3.1-8B nothink: all 6,727 items (~1–1.5 h), **not piloted**; runs last, optional.
Expected eligible: ~350–450 (nothink), ~300 (think). Launched with `scripts/overnight.sh`
(order: Qwen nothink → Qwen think → Llama).

### Open decisions / to report **[deviation] [limit]**
- **Thinking token cap kept at 4096** although pilot-2 baseline truncation was 6% (the plan said
  raise to 6144 above 5%). Truncated baselines are simply ineligible, and continuations had 0%
  truncation, so the cap does not bias recovery; raising it would cost hours. Must be stated.
- Modes are compared on different item sets (eligibility differs: 6% vs 20%). Report the
  intersection-set analysis as well, and item difficulty as a covariate.
- Eligibility selects items the model solves along MuSiQue's decomposition: results describe
  "own-chain errors on solvable, gold-aligned items", not all questions.
- Hand audit of planted values still to do (2 people).

---

## Paper hooks (quick index)

| Paper section | Use these entries |
|---|---|
| Method: data, conditions, splice | scope; data preparation; design decisions; prompt v2 |
| Method: error types and validity | type tags; off-context relation ranking; audit |
| Method: eligibility | prompt v2 + hop matching; hop alignment fixes |
| Method: outcomes | outcomes strict + contained |
| Setup: models, decoding, hardware | GPU memory; main-run plan; decoding table in `paper/main.tex` |
| Results: pilot / yield | pilot 1 and 2 tables |
| Analysis: silent recovery example | thinking-mode injection entry |
| Limitations | scope reduction, type-tag noise, token cap, different item sets per mode, selection by eligibility, 1 sample per cell, 4-bit 8B only |
| Appendix: reproducibility | GPU box setup problems; GPU memory; pinned revisions in `configs/` |
