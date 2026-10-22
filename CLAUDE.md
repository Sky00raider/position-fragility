# CLAUDE.md: position-fragility (REDUCED SCOPE)

Full design: `docs/design_v2.md`. **This file overrides it where they differ.**
Deadlines: experiments done **Oct 27 2026**, paper submitted **Oct 30 2026**.
4-person student team: keep code simple and readable; the method section is written from it.

## In scope
- Task: MuSiQue-Ans, **3-hop and 4-hop answerable** items, train + dev splits.
- Models (config-driven, one yaml per model x mode in `configs/`):
  - `Qwen/Qwen3-8B-AWQ`, `enable_thinking` False (nothink) and True (think). Headline.
  - Optional: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`, nothink only.
  - CPU debug: `Qwen/Qwen3-0.6B` via HF backend (pipeline test only).
- Conditions per eligible item: `noprefix`; `clean` at every hop k=1..K; `inject` at every k x
  {`in_context`, `off_context`}. k=K is included but flagged `is_terminal`.
- One sample per cell.

## Out of scope: do not build
GSM8K math arm, handoff condition, counterfactual subset, quantization check, LLM judge,
multiple samples, Gemma, R1-Distill, Claude API, explicit/silent self-correction labels.

## Design rules (do not break)
1. **Inject into the model's OWN chain** via assistant-turn prefill. Never the gold template.
   Never start a new user turn. Never put EOS at the splice. Prefix ends with exactly one `\n`.
2. Prefix = model's own answer text cut after hop line k; only the answer span after `->` on
   line k is replaced. Think mode: prefill `"<think>\n" + prefix` (hops go inside the think block).
3. Prompts are built once in `src/prompts.py` as **token ids** (`add_special_tokens=False`) and
   passed to either backend. Pipeline code is identical for vLLM and HF; only config changes.
4. Per-cell seed = sha256 of `run_name|item_id|condition|k|error_type`. Never built-in `hash()`.
5. Eligibility (doc §6.1): final answer correct; exactly K hop lines numbered 1..K; each hop i
   matches gold hop i (alias match for last hop, else normalized EM or token-F1 >= 0.8);
   closed-book answer wrong. `debug_relaxed` (CPU configs only) keeps any item with K hop lines.
6. Validity (doc §6.5), else drop the cell and log the reason in `dropped.jsonl`:
   (1) planted value not equal to / substring of / superstring of any gold answer, alias, or gold
   hop answer (normalized); (2) type tag match; (3) format preserved; (4) not in model's clean chain.
7. in_context = typed string from the item's 4 shown distractor paragraphs (titles preferred).
   off_context = same-type gold hop answer from another item, absent from this item's context.
8. `correct` = SQuAD-normalized EM vs answer + aliases. `adopted` = final answer == planted value
   (NOT "planted value appears somewhere"). Final answer = last `Final answer:` after `</think>`;
   none -> `pred=None` (counts as wrong). Truncated runs count as wrong; report truncation rates.
9. Recovery RR = P(inject correct | clean twin at same k correct). PSI = RR_late - RR_early on
   intermediate hops only. **Terminal hop never pooled with "late".**
10. `remaining` = number of later hops depending on hop k (transitively), from `#N` refs.
    **Primary analysis = linear chains**; all items = robustness check.
11. Every step is resumable (append-only jsonl, skip done keys). HF token comes from `.env`
    (`HF_TOKEN`); never print, log or commit it.

## Layout
```
configs/  data/raw/ (gitignored)  data/items/  docs/  runs/ (gitignored)
src/config.py data.py prompts.py backends.py parse.py normalize.py inject.py store.py pipeline.py analysis.py
scripts/sanity.py gpu_smoke.py   tests/
```

## Trial record schema (one JSON line per cell, `runs/<run>/scored.jsonl`)
```json
{"run_id":"", "git_hash":"", "task":"musique", "split":"dev|train", "item_id":"",
 "model":"", "mode":"nothink|think", "quant":"awq4|none", "gpu":"3070|5070|none",
 "condition":"noprefix|clean|inject", "frame":"prefill",
 "K":0, "k":0, "remaining":0, "rel_pos":0.0, "is_terminal":false,
 "error_type":"in_context|off_context|null", "orig_value":"", "new_value":"",
 "gold":"", "prefix":"", "continuation":"", "pred":"",
 "correct":null, "adopted":null, "explicit_sc":null, "doubt_marker":null,
 "item_difficulty":null, "sample_idx":0, "n_gen_tokens":0, "truncated":false,
 "composition":"3hop1", "is_linear":true, "remaining_linear":0, "eligibility_relaxed":false,
 "seed":0, "model_revision":null, "backend":"vllm|hf", "lib_version":"", "f1":0.0}
```
`rel_pos = k/K`; `remaining_linear = K-k`; `item_difficulty` = 1 - noprefix correct (binary,
one sample). `noprefix` rows have k=0. `explicit_sc` / `doubt_marker` stay null (out of scope).

## Commands
See README.md. Tests: `python -m pytest -q`.
