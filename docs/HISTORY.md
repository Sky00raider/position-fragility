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
- **Thinking token cap kept at 4096 (decided 2026-10-23, team agreed)** although pilot-2 baseline
  truncation was 6% (the plan said raise to 6144 above 5%). Truncated baselines are simply
  ineligible, and continuations had 0% truncation, so the cap does not bias recovery; raising it
  would cost hours. Must be stated in Setup; report main-run truncation rates (`truncation.csv`).
- Modes are compared on different item sets (eligibility differs: 6% vs 20%). Report the
  intersection-set analysis as well, and item difficulty as a covariate.
- Eligibility selects items the model solves along MuSiQue's decomposition: results describe
  "own-chain errors on solvable, gold-aligned items", not all questions.
- Hand audit of planted values still to do (2 people).

---

## 2026-10-24 (Sat): main runs finished

### Main run, overnight on the RTX 5070 (commit 7aef5ea) **[setup] [result]**
22:39 → ~06:00. First launch at 22:30 failed instantly (venv not active); the scripts now activate
`.venv` themselves. Results: `Research Paper/main_results_archlinux_20261023_2239/` (run folders under
`home/burr/Documents/position-fragility/runs/`), copied to `position-fragility/runs/`.

| | Qwen3-8B nothink | Qwen3-8B think | Llama-3.1-8B nothink |
|---|---|---|---|
| items attempted | 6,727 | 2,180 (600 3-hop + 1,580 4-hop) | 6,727 |
| final answer correct | 44% | 52% | 36% |
| closed-book correct (excluded) | 10% | 12% | 7% |
| K hop lines | 97% | 96% (33% also inside `<think>`) | 95% |
| baseline truncated | 0% | 3% | 1% |
| **eligible** | **922 (14%)**: 3-hop 855, 4-hop 67 | **384 (18%)**: 3-hop 167, 4-hop 217 | **564 (8%)**: 3-hop 523, 4-hop 41 |
| eligible, linear | 3-hop 763, 4-hop 33 | 3-hop 147, 4-hop 154 | 3-hop 438, 4-hop 15 |
| inject cells (dropped by validity) | 5,413 (253) | 2,565 (173) | 3,289 (177) |
| noprefix correct | 95% | 84% | 74% |
| clean twins correct | 97% | 79% | 88% |
| inject correct (raw) | 41% | 62% | 42% |
| throughput | 403 tok/s | 547 tok/s | 397 tok/s |
| wall time | 51 min | 4 h 55 min | ~1 h 10 min |

Nothink eligibility was higher than in pilot 2 (14% vs 6%: pilot n was 50). 4-hop nothink is thin
(Qwen 33, Llama 15 linear items): 4-hop nothink contrasts are underpowered. **[limit]**

### Clean-twin gate (doc §9: ~90%) **[result] [limit]**
Qwen nothink passes (96–100% at every hop). **Qwen think fails at intermediate hops** (linear: K=3
78%/86%, K=4 68%/71%/75%; terminal 100%) and Llama is borderline (K=3: 82%/87%). So resuming
from a prefix sometimes breaks the run even without an error, especially inside `<think>`.
Clean-twin conditioning removes those items from RR, but it means think-mode RR is measured on a
selected subset. Must be stated in the paper.

### Findings (linear items, primary; all-items agree) **[result]**
`runs/analysis_main/` (bootstrap 2000 reps). RR = recovery given a correct clean twin.
- **H1 (position) reversed: PSI < 0 everywhere.** Late intermediate errors are recovered *less*
  than early ones (late-stage fragility, the ASCoT direction, opposite to our hypothesis). Qwen
  nothink K=3: RR early 0.62 → late 0.48 (in-context), 0.70 → 0.50 (off-context), CIs exclude 0,
  McNemar p_holm < 1e-4. Qwen think: PSI −0.09 to −0.12, CIs mostly exclude 0, McNemar n.s. after
  Holm. Llama K=3: −0.09 / −0.13. 4-hop nothink: same sign, wide CIs.
- **H2 (remaining steps): inconclusive.** Matched-remaining contrasts (same steps left, different
  absolute position) are ≈0 for Qwen nothink (diffs −0.04 to +0.05, CIs ±0.17 because K=4 n≈32)
  and mixed for think (−0.10 to +0.04). The regression keeps rel_pos significant with remaining and
  K in the model (β=−3.67, p=0.023), but VIFs are 87–104, so position and remaining cannot be
  separated statistically in this design (the doc's caveat). Report as "consistent with remaining
  steps, not separable".
- **H3 (terminal step) strongly supported.** Terminal-hop RR: Qwen nothink 0.00–0.02, Llama
  0.07–0.15, Qwen think 0.60, vs intermediate 0.56–0.64 / 0.60–0.64 / 0.82. Pooling terminal with
  "late" would hugely inflate late fragility.
- **H4 (thinking): level yes, shape no.** Thinking raises recovery (mode β=+1.52, p<0.001;
  intermediate RR ≈0.82 vs ≈0.58) but the rel_pos × mode interaction is n.s. (β=−0.24, p=0.66):
  the late > early fragility persists with thinking.
- **H5 (error type) supported, small.** Off-context swaps are recovered more than in-context
  distractors (β=+0.15, p=0.002; e.g. Qwen nothink K=3 early 0.70 vs 0.62).
- Item difficulty strongly predicts failure (β=−1.12, p<0.001); Qwen recovers less than Llama
  (β=−0.35, p<0.001).
- Adoption: Qwen nothink adopted the planted value in 31% of inject cells (1,688/5,413).

### Why clean twins fail: no scoring bug; the think prefill cuts deliberation **[result] [limit]**
Audit of every failing intermediate clean twin (`scripts/clean_twin_audit.py`, CPU, reads
`scored.jsonl`). **No extraction/scoring bug:** in 0 of 282 failed Qwen-think twins (0/204 Llama,
0/73 Qwen nothink) is the gold written as any `Final answer:` (inside or after `<think>`), so
`src/parse.py` and the score step are unchanged and nothing was re-scored. Qwen think: 0 truncated,
0 missing final lines. Llama: 2 truncated, 5 without a `Final answer:` line (3 of these end on a
correct last hop; rule 8 counts them wrong, by design).

Failed intermediate twins by where the model left its **own** baseline chain (all items; linear in
brackets):

| | Qwen3 think | Llama nothink | Qwen3 nothink |
|---|---|---|---|
| failed / intermediate twins | 282/985 (29%) [186/756, 25%] | 204/1169 (17%) [143/921] | 73/1911 (4%) [59/1625] |
| diverged at hop k+1 (next hop) | 129 (46%) [82] | 97 (48%) | 32 (44%) |
| diverged at a later hop | 95 (34%) [53] | 61 (30%) | 20 (27%) |
| same chain, answer at other granularity ("1991" for `December 25, 1991`, "Richland" for `Richland County`) | 58 (21%) [51] | 39 (19%) | 21 (29%) |
| truncated / no final line | 0 | 7 (3%) | 0 |
| final answer keeps prefilled hops 1..k verbatim | 281/282 | 204/204 | 73/73 |

**Mechanism (think mode).** The prefix is the model's post-`</think>` hop summary. Once it sits
inside `<think>`, Qwen3 treats reasoning as finished: **985/985** think clean continuations contain
only hop lines + `Final answer:` (no reasoning prose), median **110** generated tokens vs **1,231**
for the unprefixed resample (baseline 1,132). Inject cells are the same (median 110). The model
does not abandon the prefilled chain (281/282 keep it); it answers the remaining hops without
deliberating, and on harder hops picks a different entity. Paired with the noprefix resample of the
same item: think clean wrong / noprefix right 183 vs the reverse 68 (McNemar p = 2e-13; linear
129 vs 46), so the prefill itself costs accuracy in think mode. In nothink there is no such cost
(Qwen 42 vs 55, p = 0.22; Llama 88 vs 190: the prefill *helps* Llama, whose failures are
ordinary sampling variance at temperature 0.6).

Consequences for the paper: think-mode RR is measured on items the model can finish *without*
further deliberation (clean-twin conditioning), and post-splice "thinking" is ~110 tokens, so H4
compares nothink with "thinking-format, short continuation", not with full thinking. Most think
recoveries happen in the restated chain after `</think>`: of 1,110 correct inject cells with a
correct twin, 996 rewrite the hop-k value there (the silent-recovery pattern above). No definition
change proposed: the granularity misses are real EM misses (the model wrote the shorter/longer
form; its own baseline wrote the gold form). Most are shorter than gold, so `correct_contains`
(gold inside the final line) does not rescue them either.

### Explicit vs silent recovery (E1, exploratory; e09f44d) **[result] [method] [deviation]**
Added to `src/analysis.py` although CLAUDE.md listed E1 labels out of scope: **rule-based only**
(no LLM judge, no 4-person validation as in doc §8.2). Recovered inject cell (clean twin correct) =
**explicit** if a doubt marker (`wait|actually|mistake|incorrect|re-?check|should be|not correct|
correction|however`, case-insensitive, whole words) is in the same or next sentence as the true hop-k
value (gold hop answer, or the model's own original value) or the planted value; otherwise
**silent** (strict: using the right value without a flag is silent). Think mode is split by flag
location (before / after `</think>`). Tables `silent_explicit_{linear,all}.csv` (model × mode × K ×
k × error type) and `_summary_` (intermediate vs terminal); 10 random examples per label in
`runs/analysis_main/silent_explicit_examples.txt`.

Linear items, intermediate hops (terminal in brackets):

| | RR | explicit rate | silent rate | explicit share of recoveries | doubt marker, non-recovered |
|---|---|---|---|---|---|
| Qwen3 nothink | 0.58 [0.01] | 0.000 | 0.58 | 0% (0/1770) | 0.1% |
| Qwen3 think | 0.82 [0.60] | 0.002 | 0.82 | 0.2% (2/909; both inside `<think>`) | 1.0% |
| Llama nothink | 0.62 [0.10] | 0.057 | 0.56 | 9.3% [6.0%] | 14.9% [1.6%] |

- **Recovery is almost entirely silent.** Qwen never flags the planted value in nothink, and in
  think mode does so in 2 of 1,488 recoveries (all items); it just rewrites the hop with the right entity
  (fits the audit above: post-splice thinking is ~110 tokens of hop lines with no prose). Not a regex
  artefact: other doubt words ("but", "not", "let me", "hmm", "wrong") occur in ≤ 0.7% of Qwen
  recoveries. Llama flags in ~9% of recoveries, more at k=2 than k=1 (K=3 off-context: 19% vs 6%).
- Doubt markers do not mark failure: Llama uses them more when it fails (15%) than when it recovers
  (12%), typically "Not found … However, …" before giving up or restarting.
- Hand-checked: the first 6 Llama explicit examples in the file are true flags ("Hop 2 is incorrect
  as Middelburg is a city in South Africa, not South Korea"); both Qwen-think explicit cells are
  true flags. Example for the paper (Qwen3 think, `3hop1__38095_48661_15069`, k=2, planted
  "Louisville" for "Ethiopia"): inside `<think>` → *"Wait, that's not right. Let me check again. …
  the largest contributor is Ethiopia"*, then the correct chain after `</think>` (1953). Silent
  examples were not hand-checked beyond reading; labels must be called rule-based in the paper.

### Shared-item mode comparison (robustness; e09f44d) **[result] [limit]**
Qwen3 nothink vs think restricted to items eligible in both modes. Small: think only attempted 600
three-hop items. Shared items: **K=3 67 (63 linear), K=4 31 (20 linear)**, out of 855/167 and
67/217 eligible. Tables `*_intersection.csv`, figure `mode_compare_linear_intersection.png`. Mode
difference is on paired cells (same item, k, error type, clean twin correct in both modes).
Linear shared items:
- **Thinking still helps at every hop.** Intermediate RR 0.64 → 0.88 (diff +0.24, CI 0.17–0.31);
  terminal 0.01 → 0.72 (+0.70, CI 0.63–0.78). Same as on the full sets (0.58 vs 0.82).
- **Terminal < intermediate in both modes**: nothink −0.58/−0.66 (in/off-context), think
  −0.16/−0.13, all CIs exclude 0.
- **PSI:** nothink stays negative (K=3 −0.14 in-context, −0.28 off-context, CI excludes 0). In think
  mode it shrinks: K=3 +0.00 in-context (CI ±0.09), −0.10 off-context (CI −0.21 to +0.01); K=4 CIs
  ±0.2. So on shared items late-stage fragility is clear without thinking and weak or absent
  with it. The regression interaction is n.s. (rel_pos:mode β=+1.01, p=0.34; n=695 cells, 81
  items), so this does **not** establish a shape difference: it says the full-set think PSI
  (−0.09 to −0.12) is not robust on this small shared set. All-items agree.

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
| Analysis: silent recovery example | thinking-mode injection entry; explicit vs silent recovery (E1) |
| Results: H4 robustness | shared-item mode comparison |
| Limitations | scope reduction, type-tag noise, token cap, different item sets per mode (shared-item check), selection by eligibility, 1 sample per cell, 4-bit 8B only, think prefill cuts deliberation (why clean twins fail) |
| Appendix: reproducibility | GPU box setup problems; GPU memory; pinned revisions in `configs/` |

### New condition: error planted in the model's own thinking text (`think_trace`) **[method] [deviation]**
Built 2026-10-24 after the clean-twin audit: with the hop-summary prefill inside `<think>` Qwen3
stops deliberating (median 110 new tokens vs 1,231; 28/985 clean continuations contain any
reasoning prose), so the original think condition does not test whether thinking catches errors.
New injection site (`injection_site: trace`, config `qwen3_8b_think_trace.yaml`), the design doc's
§6.6 alternative (as in von Recum et al.): in the model's own baseline thinking trace, find the
first whole-word mention of the hop-k answer (model's own hop answer or gold hop answer; aliases
for the terminal hop), replace it with the planted value (or leave it for the clean twin), cut at
the end of that sentence, and let the model continue thinking. Same items, baseline (reused, not
regenerated), planted values and validity rules as `qwen3_8b_think`; scored rows get
`mode = think_trace` and `injection_site`. Mention not found → cell dropped and logged; a mention
that also appears in the question is dropped (`answer_in_question`).
Dry run on the saved traces: 4,313 cells (vs 4,318), cut at median 25% of the trace, monotone in
k (12%, 22%, 36%, 57% for k=1..4). Caveat for the paper: the first mention is sometimes the model
*reading* a paragraph rather than concluding. Analysis now compares all mode pairs (nothink /
think / think_trace) on shared items. GPU run: `bash scripts/overnight.sh qwen3_8b_think_trace`
(~2.5–3.5 h).

### Paper: Results section drafted (no think_trace yet) **[paper]**
2026-10-24, next session. think_trace GPU results had not arrived (`runs/qwen3_8b_think_trace`
holds only the local dry run: cells, no generations), so the Results section was written from
`runs/analysis_main` (nothink / think / Llama), which stands either way. `paper/main.tex` now has:
clean-twin subsection (think prefill cuts deliberation), RQ1 with PSI table (`tab:psi`, table*),
RQ2 with the regression table (`tab:logit`, VIFs), RQ3 terminal, RQ4 thinking (full + shared-item
sets), RQ5 error type, silent recovery; Fig. `recovery_linear.png` copied to `paper/figures/`
(upload it to Overleaf with main.tex). Also filled: item-count table (`tab:counts`), vLLM 0.32.0 /
single RTX 5070, pilot clean-twin outcome, main-run truncation. Numbers checked against the CSVs:
excluding truncated runs moves PSI by ≤ 0.003; shared linear items = 83 (63 + 20; 81 in the
regression). Open `\tbd`s in Results/Method: think_trace paragraph (RQ4) and its Method sentence.
Not compiled locally (no LaTeX on the dev box).
