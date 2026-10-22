# Error Position in Multi-Hop Reasoning: Project Kickoff Doc (v2)

Working title: **Where Does a Wrong Step Hurt? Error Position, Remaining Steps, and Thinking Mode in Multi-Hop Reasoning**

Team: 4 x 3rd-year CSE (AI/ML), MSRIT
Target: IEEE conference (venue + deadline: ________, pick by end of week 1)
Course: Research Methodology & IPR (AL58), check-in Dec 1, 2026
Hardware: RTX 3070 8GB (college lab), RTX 5070 12GB (friend)
Type: empirical / comparative, inference only

> Anything marked **(verify)** is from memory or a best guess: confirm before relying on it.

### What changed from v1

- **Jia & Mu (arXiv 2609.25366, Sep 2026)** already ran our core v1 experiment on GSM8K/MMLU/BBH with Gemma-2-9B, Llama-3.1-8B and R1-Distill-Qwen-7B. They found early errors propagate more than late ones, but model-relative task difficulty dominates, and GSM8K CoT is mostly ignored by 8-9B models (94.5% bypass on Gemma). They could not separate position from remaining steps.
- So: **multi-hop QA becomes the main task**, GSM8K shrinks to a small ASCoT side check, and the design is built to separate position, remaining steps and the terminal step.
- Merged in the good parts of the PDF draft: clean-twin conditioning, in-context vs off-context error types, PSI, pre-specified contrasts.
- Dropped: HotpotQA and 2Wiki as main sets (2-hop, no real early/late contrast), Qwen3-32B (does not fit our GPUs), bf16 serving.

---

## 0. One-paragraph pitch

Three studies disagree on whether a wrong step hurts more early or late in a reasoning chain. ASCoT (arXiv 2508.05282) reports late-stage fragility on one model and GSM8K. von Recum et al. (ICLR 2026, arXiv 2602.07470) find reasoning models recover worse from early interventions. Jia & Mu (arXiv 2609.25366) find early > late propagation but show it is tangled with remaining steps and task difficulty. All three use math, science or logic tasks. We run the first position-controlled error-injection study on **multi-hop QA** (MuSiQue 3- and 4-hop), where each hop has a gold intermediate answer, so the error is clean and its position exact. Crossing chain length with hop position lets us compare errors with the **same number of steps remaining** at different positions, and we analyze the **terminal (answer) step separately**. We use a **same-weights thinking toggle** (Qwen3-8B, thinking on/off) and two error types (plausible in-context distractor vs unsupported off-context swap). A small GSM8K side study tests whether ASCoT's "late" effect is just the terminal step.

---

## 1. Research questions and pre-registered hypotheses

| ID | Question | Hypothesis | Status |
|---|---|---|---|
| RQ1 | Does error position (intermediate hops only) affect recovery on multi-hop QA? | H1: recovery rises with relative position (errors early in the chain propagate more), PSI > 0 | Confirmatory |
| RQ2 | Is the position effect explained by remaining steps? | H2: in matched-remaining contrasts (§6.2), the position effect shrinks toward zero | Confirmatory |
| RQ3 | Is the terminal step special? | H3: terminal-hop corruption gives the lowest recovery in every model; pooling it with "late" changes the sign or size of PSI | Confirmatory |
| RQ4 | Does thinking change the level or shape of the curve? | H4: in Qwen3-8B, thinking raises recovery and shrinks \|PSI\| (position x mode interaction) | Confirmatory |
| RQ5 | Do plausible errors hurt more? | H5: in-context distractors are recovered from less than off-context swaps at every position | Confirmatory |
| RQ6 | Does item difficulty moderate position? | H6: position effect is larger on harder items (low no-prefix accuracy) | Exploratory |
| RQ7 | Does ASCoT's late effect on GSM8K come from the terminal step? | H7: Qwen2.5-7B, excluding the terminal step removes the late > early gap | Secondary (math arm) |
| E1 | Explicit vs silent recovery by position and mode | none | Exploratory |
| E2 | Doubt markers after injection by position and mode | none | Exploratory |
| E3 | Own-trace prefill vs handed-over notes (handoff condition) | none | Exploratory, stretch |

Equivalence margin for "no position effect": ±5 percentage points (TOST, §8).
Pre-register §1, §6 validity rules and §8 on OSF after the pilot, before the main run.

---

## 2. Key literature

### Must-read (everyone, week 1)

| Paper | arXiv | What it shows | Why it matters to us |
|---|---|---|---|
| Jia & Mu, "From Decorative to Load-Bearing" | 2609.25366 | Perturb-truncate-continue on GSM8K/MMLU/BBH; early > mid > late propagation; difficulty dominates; GSM8K CoT mostly bypassed; keyword labels unreliable, only C vs non-C is stable | Closest prior work. Our method template, our gap (remaining steps, multi-hop), our labeling lessons. Code: github.com/r2m-ai/load-bearing-cot |
| von Recum et al., "Are Reasoning LLMs Robust to Interventions on Their CoT?" (ICLR 2026) | 2602.07470 | Own-CoT interventions at fixed timesteps on reasoning models; early interventions hurt more; doubt drives recovery | The "early" side; doubt-token methodology |
| Zhang et al., ASCoT ("Not All Errors Are Created Equal") | 2508.05282 | Late errors hurt more (Qwen2.5-7B, GSM8K) | The "late" side; our math side check |
| Yee et al., "Dissociation of Faithful and Unfaithful Reasoning" | 2405.15092 | Error magnitude and type change recovery; faithful vs unfaithful recovery | Severity controls, recovery definitions |
| Trivedi et al., MuSiQue | 2108.00573 | 2-4 hop questions composed from single hops, shortcut-filtered | Our main dataset |

### Should-read (split by role)

- Yun et al., CRiT-QA (LREC 2026), 2607.10562: counterfactual entities and distractor chains diverging at different hops (input-side analog; contamination fix)
- Lanham et al., CoT faithfulness, 2307.13702
- Aravindan & Kejriwal, "Fragile Thoughts", 2603.03332
- Tyen et al., BIG-Bench Mistake, 2311.08516
- Huang et al., "LLMs Cannot Self-Correct Reasoning Yet", 2310.01798
- Turpin et al., unfaithful CoT, 2305.04388; Chen et al., "Reasoning models don't always say what they think", 2505.05410
- Press et al., compositionality gap, 2210.03350 (verify)
- Liu et al., "Lost in the Middle", 2307.03172; Chen et al., premise order, 2402.08939 (verify)
- Vendrow et al., platinum benchmarks, 2502.03461; Wu et al., Qwen2.5 contamination, 2507.10532
- Qwen3 technical report, 2505.09388; DeepSeek-R1, 2501.12948

### How we position against prior work (neutral tone)

1. ASCoT: one model, one dataset, no CIs; terminal step pooled with "late"; table and appendix disagree. Motivates a controlled test, not "ASCoT is wrong".
2. von Recum: reasoning models only, math/science/logic, no multi-hop QA, no same-weights toggle.
3. Jia & Mu: no multi-hop QA; position, remaining steps and difficulty confounded (their own limitation); no same-weights thinking toggle.
4. CRiT-QA: errors placed in the input passages, not in the reasoning prefix.

Before submission: run "cited by" on all four (Semantic Scholar / Google Scholar).

---

## 3. Models

All 4-bit AWQ. One model = one GPU for all its conditions.

| Role | Model (HF id) | GPU | Notes |
|---|---|---|---|
| Headline: thinking toggle | `Qwen/Qwen3-8B-AWQ` (verify) | 5070 | `enable_thinking=False/True`, same weights |
| 2nd family | `meta-llama/Llama-3.1-8B-Instruct`, AWQ `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (verify) | 3070 | Gated: request access day 1. Also in Jia & Mu, so we can sanity-check against them |
| 3rd family | `google/gemma-3-12b-it` 4-bit (verify fit with ~2k-token context) or fall back to `google/gemma-2-9b-it` | 5070 | Gemma-2-9B overlaps Jia & Mu; Gemma-3 is newer |
| Math arm + ASCoT anchor | `Qwen/Qwen2.5-7B-Instruct-AWQ` | 3070 | Same model as ASCoT |
| Optional bridge | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 5070 | In both von Recum and Jia & Mu |
| Optional API check | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | API | 200-item subset only, both modes, if budget allows. Handoff condition only (no prefill into its thinking). Date-stamp results |

**Quantization check:** Qwen3-8B at 8-bit on the 5070 for 100 items, compare PSI with 4-bit. Appendix.

**Log per run:** model revision hash, quant method, vLLM version, GPU, batch settings, seed.

---

## 4. Datasets

### Main: MuSiQue-Ans, 3-hop and 4-hop only
- Official repo `StonyBrookNLP/musique` (download script) or HF mirror (verify). Files `musique_ans_v1.0_{train,dev}.jsonl`.
- Fields (verify): `id` (prefix `3hop1__`, `4hop1__`, ...), `paragraphs[{idx, title, paragraph_text, is_supporting}]`, `question`, `question_decomposition[{id, question, answer, paragraph_support_idx}]`, `answer`, `answer_aliases`, `answerable`.
- Dev split is small for 3/4-hop (roughly 760 and 405 answerable, verify). **Also draw 3/4-hop items from train**: inference only, so it is fine; report the split per item and check results agree across splits.
- 2-hop excluded from the position analysis (only one intermediate hop). May be used for the terminal-step comparison.
- **Context:** gold supporting paragraphs + 4 fixed distractor paragraphs, shuffled with a fixed seed.
- **Closed-book filter:** drop items the model answers correctly with no paragraphs (memorized).
- **Scoring:** EM vs `answer` + `answer_aliases` after SQuAD normalization; F1 secondary.
- Hand-check 50 decompositions for annotation noise.

### Contamination fallback: counterfactual subset (stretch, or pivot if pilot fails)
- CRiT-QA style: swap entities in passages and gold chain consistently so parametric memory cannot answer. Use CRiT-QA's released data if available (verify) or build ~300 items ourselves.

### Math side check: GSM8K-Platinum
- HF `madrylab/gsm8k-platinum` (verify). ~300 eligible items, Qwen2.5-7B only.
- Purpose is RQ7 only. Jia & Mu show GSM8K CoT is mostly bypassed at this scale, so we do not expect big effects; that is fine for a side check.

### Not using as main sets (state in paper)
- HotpotQA: 2-hop, bridge order has to be guessed, shortcut-prone.
- 2WikiMultiHopQA: mostly 2-hop, templated.

---

## 5. Environment setup

```bash
# Python 3.10/3.11, fresh venv per machine
pip install vllm transformers accelerate datasets pandas numpy scipy statsmodels spacy rapidfuzz tqdm
python -m spacy download en_core_web_sm
```

- **3070 (Ampere):** standard vLLM wheels, AWQ supported.
- **5070 (Blackwell):** needs CUDA 12.8+ builds of PyTorch and vLLM. Day 1: load one AWQ model and generate 10 samples. Fallback: llama.cpp GGUF Q4_K_M.
- Starting config:

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-8B-AWQ", quantization="awq",
          max_model_len=8192, gpu_memory_utilization=0.90, seed=0)
nothink = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=1024, n=4, seed=0)   # Qwen3 recommended non-thinking (verify)
think   = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=6144, n=4, seed=0)  # Qwen3 recommended thinking (verify)
```

- Thinking traces are long: measure tokens/s and VRAM in the pilot before fixing `max_tokens`.
- Keep batch settings fixed across conditions.

### Rough compute (estimate, confirm in pilot)
- Non-thinking: ~300 generated tokens per run. Thinking: ~1.5-3k.
- Per model-mode at ~400 items x ~10 cells x 4 samples = ~16k runs: a few hours non-thinking, roughly a day or more thinking on one card. Run non-thinking first.

### Repo layout

```
position-fragility/
  configs/            # one yaml per model x mode x task
  data/raw/           # downloads (gitignored)
  data/items/         # filtered eligible items (jsonl)
  src/
    prompts.py
    generate.py       # baseline, clean twin, injected continuations (vLLM)
    parse_qa.py       # hop parsing, alignment to gold decomposition
    inject_qa.py      # in-context / off-context substitution
    parse_math.py     # math arm: step + equation parsing
    inject_math.py    # math arm: calc errors, a_prop
    score.py          # EM/F1, outcome coding
    label_sc.py       # explicit vs silent recovery (rules + judge)
    analysis/         # power, GLMM, McNemar, bootstrap, plots
  runs/               # outputs jsonl (gitignored, back up to Drive)
  notebooks/
  paper/
```

### Trial record schema (one JSON line per continuation)

```json
{"run_id":"", "git_hash":"", "task":"musique|gsm8k", "split":"dev|train", "item_id":"",
 "model":"", "mode":"nothink|think", "quant":"awq4|int8", "gpu":"3070|5070",
 "condition":"noprefix|clean|inject", "frame":"prefill|handoff",
 "K":0, "k":0, "remaining":0, "rel_pos":0.0, "is_terminal":false,
 "error_type":"in_context|off_context|calc", "orig_value":"", "new_value":"",
 "gold":"", "prefix":"", "continuation":"", "pred":"",
 "correct":null, "adopted":null, "explicit_sc":null, "doubt_marker":null,
 "item_difficulty":null, "sample_idx":0, "n_gen_tokens":0, "truncated":false}
```

---

## 6. Injection protocol (MuSiQue)

### 6.1 Own-chain baseline
1. Prompt (§7) forces `Hop N: <sub-question> -> <answer>` lines, then `Final answer: <answer>`.
2. Generate the model's own clean chain (greedy, or sample 0 at the mode's settings).
3. **Eligible** if: final answer correct; hop count equals gold K; each hop answer matches the gold decomposition answer (alias match or token-F1 >= 0.8); item passes closed-book filter.
4. All corruption is applied to **the model's own text**, not to the gold template.

### 6.2 Positions and the remaining-steps design

Intermediate hops are 1..K-1. Hop K is terminal (it produces the final answer) and is analyzed separately.

| K | hop k | steps remaining | rel. position (intermediate) |
|---|---|---|---|
| 3 | 1 | 2 | early |
| 3 | 2 | 1 | late |
| 4 | 1 | 3 | early |
| 4 | 2 | 2 | middle |
| 4 | 3 | 1 | late |
| 3, 4 | K | 0 | terminal (separate) |

**Matched-remaining contrasts:** (K=3, k=2) vs (K=4, k=3) both have 1 step left; (K=3, k=1) vs (K=4, k=2) both have 2. Within a pair, absolute position differs while remaining steps are equal.

**Honest caveat:** k + remaining = K, so position, remaining steps and chain length cannot all be separated at once. Matched-remaining pairs also differ by one hop in K. We reduce this with clean-twin conditioning (recovery is measured only on items the model solves from the clean prefix) and an item-difficulty covariate, and we state the assumption in the paper.

### 6.3 Conditions per eligible item
- `noprefix`: question + passages, model answers from scratch (also gives item difficulty).
- `clean_k`: own chain truncated after hop k, resumed. For every k.
- `inject_k`: hop-k answer replaced, truncated after hop k, resumed. For every k x 2 error types.
- Per item: 1 + K + 2K runs (10 for K=3, 13 for K=4), each with n samples.

### 6.4 Error types
- **In-context distractor:** same type as the gold hop answer, taken from the item's distractor paragraphs. Entities: spaCy NER label match (title of a non-gold paragraph preferred, similar length). Years/numbers: a different one mentioned in the passages. Looks supported.
- **Off-context swap:** same type, taken from a hop answer of a different MuSiQue item. Usually supported nowhere.

### 6.5 Validity rules (all must hold, else drop the cell and log why)
1. Replacement is not equal to, a substring of, or a superstring of any gold answer, alias, or gold hop answer.
2. Type match (NER label or number/date type).
3. Format preserved (casing, punctuation, date format).
4. Replacement does not already appear in the model's own clean chain.
5. Hand audit: 100 random injections checked by 2 people before the main run.

### 6.6 Resumption (prefill)

```python
kw = {"enable_thinking": think} if "Qwen3" in model_name else {}
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
prompt += prefix   # hops 1..k of the model's own chain, hop k corrupted, ending with "\n"
```

- Never start a new user turn. Never insert EOS at the splice. Check tokenization at the splice (leading-space issues, especially Llama).
- **Thinking mode:** the prefix goes inside the think block: `"<think>\n" + hop lines`. The model continues thinking, then answers. Pilot must confirm this looks natural (does the model keep the hop format inside thinking?). Alternative if not: inject into blank-line paragraphs of its own thinking that contain the hop answer, as von Recum do.

### 6.7 Handoff condition (stretch, E3)
- Same corrupted prefix, but placed in the user turn under "Working notes so far", with no warning that notes may be wrong. This is the PDF draft's design. Compare with prefill on one or two models: does a model trust its "own" reasoning more than handed-over notes?

---

## 7. Prompt templates (finalize after pilot)

**MuSiQue (all models, both modes):**
```
Answer the question using only the paragraphs below. Reason one hop at a time.
Write each hop on its own line as "Hop N: <sub-question> -> <answer>".
After the last hop, write "Final answer: <short answer>".

Paragraphs:
{paragraphs}

Question: {question}
```

**Closed-book filter:** same prompt without paragraphs.

**Math arm:**
```
Solve the problem step by step.
Write each step on its own line as "Step N: ..." and show every calculation explicitly in the form "a op b = c".
After the last step, write "Final answer: <number>".

Problem: {question}
```

Robustness: one paraphrased template on a 100-item subset.

---

## 8. Measurement and statistics

### 8.1 Outcomes
- `correct` (primary): final answer matches gold. Propagation = not correct.
- `adopted`: final answer matches the planted value (not "planted value appears somewhere", which fires on restatement).
- Recovery RR(cell) = P(correct | inject) among items with `clean_k` correct (clean-twin conditioning).
- PSI = RR_late − RR_early (intermediate hops only). Terminal reported separately.
- Explicit self-correction: correct AND continuation flags the planted value and restates or re-derives the true hop answer.
- Silent recovery = RR − explicit rate.

### 8.2 Labeling (lesson from Jia & Mu)
- Correct vs not is the primary, label-free outcome.
- Explicit vs silent is **exploratory**. Rules first (true hop answer restated near a doubt marker: `wait|actually|mistake|incorrect|re-?check|should be`), ambiguous cases to an LLM judge.
- Use a **strict** definition: computing with the right value without flagging the error is silent, not explicit (Jia & Mu found looser judge prompts over-count self-correction).
- Validation: 300 traces stratified by model x mode x position x outcome, labeled by all 4 members. Report Fleiss' kappa among humans and Cohen's kappa judge vs majority. Target >= 0.8.

### 8.3 Confirmatory analysis (R, lme4)

```r
library(lme4)
d <- subset(d, condition == "inject" & !is_terminal & clean_correct)
m1 <- glmer(correct ~ rel_pos * mode + remaining + K + error_type + item_difficulty +
              model + (1 | item_id),
            data = d, family = binomial, control = glmerControl(optimizer = "bobyqa"))
```

- `rel_pos`, `remaining` and `K` are partly collinear by construction: check VIFs; report the matched-remaining contrasts (§6.2) as the cleaner test of H2.
- RQ4 on Qwen3-8B only: test `rel_pos:mode`.
- RQ3: terminal vs pooled intermediate, per model.
- Per cell (model x mode x error type): exact McNemar on paired early vs late outcomes (same item, K=3 and K=4 separately). Holm across confirmatory tests.
- Item-clustered bootstrap (resample items, 2,000+ reps) for all CIs.
- Equivalence: position effect declared absent if the 90% bootstrap CI of PSI lies within ±5 pp.
- Also report on the intersection set (items every model solves).

### 8.4 Power

```python
from scipy.stats import norm
import math
def mcnemar_n(p_disc, delta, alpha=0.05, power=0.80):
    za, zb = norm.ppf(1 - alpha/2), norm.ppf(power)
    return math.ceil((za*math.sqrt(p_disc) + zb*math.sqrt(p_disc - delta**2))**2 / delta**2)
```

| p_d | δ | n at α=.05 | n at α=.005 |
|---|---|---|---|
| 0.20 | 0.10 | ~155 | ~265 |
| 0.30 | 0.10 | ~235 | ~400 |
| 0.30 | 0.15 | ~100 | ~180 |

**Target: ~300-400 eligible items per K per model** (after correctness, alignment, closed-book filters). Recompute with pilot p_d. If short, pull more train items.

---

## 9. Pilot and go/no-go gate

- 100 MuSiQue items (50 x 3-hop, 50 x 4-hop), Qwen3-8B think on and off, both error types, all positions + controls.
- Also measure: eligible-item yield, parse/alignment rate, tokens/s, VRAM in thinking mode.
- **Go if:** propagation >= ~15% in at least one intermediate cell AND >= 20% of early-late pairs discordant AND clean-twin accuracy >= ~90% (lower means resumption itself is broken).
- **Pivot if bypass is near-total (~95% recovery everywhere):** switch main set to the counterfactual subset (§4), where the model cannot answer from memory and must rely on its chain.

---

## 10. Timeline and roles

| Week | Dates (2026-27) | Goals |
|---|---|---|
| 1 | Oct 22-31 | Read must-reads, pick venue, request Llama/Gemma access, vLLM on both GPUs, clone Jia & Mu repo |
| 2 | Nov 1-7 | Download MuSiQue (dev + train), count 3/4-hop, baselines + closed-book on Qwen3-8B, alignment parser |
| 3 | Nov 8-14 | Injector + validity checks, hand audit, **pilot + gate** |
| 4 | Nov 15-21 | Fix n from pilot, OSF pre-registration, finalize prompts, math arm parser |
| 5 | Nov 22-30 | Main runs: Qwen3-8B non-thinking, Llama. **Dec 1 course check-in: pilot + first model results** |
| 6 | Dec 1-14 | Qwen3-8B thinking, Gemma, math arm, quant check |
| 7 | Dec 15-28 | Optional models / handoff condition, 300-trace labeling, judge validation |
| 8 | Dec 29-Jan 11 | Full analysis, figures |
| 9-10 | Jan 12-25 | Write, internal review, arXiv preprint, submit |

Adjust to the chosen venue's deadline.

**Roles:**
- **A: Infra.** vLLM on both GPUs, generation harness, prefill/splice, scheduling, logging, backups.
- **B: QA pipeline.** MuSiQue loading, hop parsing and alignment, closed-book filter, injection, validity rules.
- **C: Math arm + labeling.** GSM8K-Platinum side study, rule labeler, judge, human labeling protocol.
- **D: Stats + writing lead.** Power, pre-registration, GLMM/McNemar/bootstrap, related work, paper skeleton.
- **Everyone:** week 1 reading, hand audit, 300-trace labeling.

Weekly 30-min sync and a shared results log.

---

## 11. Pitfalls checklist (tick before main run)

- [ ] Terminal hop analyzed separately, never pooled with "late"
- [ ] Matched-remaining contrasts computed; collinearity of position / remaining / K stated
- [ ] Injection into the model's own chain (prefill), not gold template
- [ ] Clean twin run for every position; clean-twin accuracy checked
- [ ] Item difficulty (no-prefix accuracy) recorded for every item
- [ ] Same decoding within every contrast; mode-specific settings documented
- [ ] Closed-book filter applied; dev vs train results compared
- [ ] Error types balanced across positions; validity drops logged per cell
- [ ] Adoption defined on the final answer, not on any mention
- [ ] Chat-template prefill verified; splice tokenization checked; strict answer extraction
- [ ] One GPU per model; versions and hashes logged
- [ ] Judge validated against humans with strict definition
- [ ] Pre-registration filed before main run

---

## 12. Related work outline

1. Multi-hop QA and shortcuts (HotpotQA, 2Wiki, MuSiQue; compositionality gap; CRiT-QA)
2. CoT and thinking modes (Wei; self-consistency; DeepSeek-R1; Qwen3 hybrid thinking)
3. Errors inside reasoning and self-correction (Huang; Tyen; Self-Refine; process supervision)
4. Interventions on CoT (Lanham; Turpin; Yee; Fragile Thoughts; Chen et al. 2025)
5. **Position of errors: ASCoT (late), von Recum (early), Jia & Mu (early, confounded)** -> our contribution
6. Position effects in the input (Lost in the Middle; premise order; distraction)

---

## 13. Paper skeleton (IEEE, ~6-8 pages; check venue limit)

1. **Abstract**
2. **Introduction:** the three-way disagreement, why it matters (where to put verification in agent pipelines), contributions: (i) first position study on multi-hop QA, (ii) design separating position, remaining steps and terminal step, (iii) same-weights thinking toggle, (iv) released harness
3. **Related Work** (§12)
4. **Method:** own-chain prefill injection, error types, validity rules, conditions, clean-twin recovery, PSI
5. **Experimental Setup:** models, quantization, data and filters, decoding, stats plan
6. **Results:** RQ1 position, RQ2 remaining steps, RQ3 terminal, RQ4 thinking, RQ5 error type; math side check (RQ7)
7. **Analysis:** explicit vs silent recovery, doubt markers, difficulty moderation, handoff vs prefill (if run)
8. **Limitations:** 4-bit, 7-12B scale, planted vs natural errors, position/remaining/K dependence, English Wikipedia only, contamination
9. **Conclusion**
10. Appendix: quant check, prompt robustness, audit stats, per-cell tables

---

## 14. Open decisions (resolve by end of week 3)

- [ ] Target venue + deadline
- [ ] Gemma-3-12B vs Gemma-2-9B
- [ ] Thinking-mode injection: hop lines inside `<think>` vs paragraph injection (from pilot)
- [ ] Thinking token cap
- [ ] Judge model (API vs local)
- [ ] Include R1-Distill? Claude subset? Handoff condition?
- [ ] Counterfactual subset as stretch or main (from pilot)
- [ ] OSF vs AsPredicted

---

## 15. References (arXiv IDs; verify each before citing)

- Jia & Mu, decorative to load-bearing: 2609.25366
- von Recum et al. (ICLR 2026): 2602.07470
- ASCoT: 2508.05282
- CRiT-QA (LREC 2026): 2607.10562
- MuSiQue: 2108.00573
- Yee et al.: 2405.15092
- Fragile Thoughts: 2603.03332
- Lanham et al.: 2307.13702
- Turpin et al.: 2305.04388
- Chen et al., reasoning models don't say what they think: 2505.05410
- Tyen et al.: 2311.08516
- Huang et al.: 2310.01798
- HotpotQA: 1809.09600; 2WikiMultiHopQA: 2011.01060
- Press et al., compositionality gap: 2210.03350
- Liu et al., Lost in the Middle: 2307.03172
- Chen et al., premise order: 2402.08939
- Vendrow et al.: 2502.03461; Wu et al.: 2507.10532
- GSM8K: 2110.14168
- Qwen3: 2505.09388; DeepSeek-R1: 2501.12948
