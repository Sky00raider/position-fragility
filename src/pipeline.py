"""Pipeline CLI. Every step reads and writes jsonl under runs/<run_name>/ and is resumable.

    python -m src.pipeline prepare-data
    python -m src.pipeline baseline  --config configs/X.yaml
    python -m src.pipeline eligible  --config configs/X.yaml
    python -m src.pipeline vet       --config configs/X.yaml   (if vet_values: LLM judge on planted values)
    python -m src.pipeline inject    --config configs/X.yaml
    python -m src.pipeline continue  --config configs/X.yaml
    python -m src.pipeline score     --config configs/X.yaml
    python -m src.pipeline estimate  --config configs/X.yaml --tok-per-s 1500 --n-items 800
"""
from __future__ import annotations

import argparse
import itertools
import random
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from . import data
from .backends import load_tokenizer, make_backend
from .config import REPO_ROOT, Config, judge_config, load_config
from .inject import build_off_pool, judge_question, pick_value, valid_candidates
from .normalize import exact_match, hop_answer_matches, max_f1, normalize
from .parse import extract_final, hops_well_formed, parse_hops, split_think
from .prompts import (build_ids, build_prefix, build_trace_prefix, find_trace_mention, prefill_text,
                      user_message)
from .store import append_jsonl, cell_key, cell_seed, done_keys, git_hash, read_jsonl, write_jsonl
from .vet import cache_path, judge_message, load_cache, parse_verdict, vet_key


# ---------------------------------------------------------------- shared helpers

def run_items(cfg: Config) -> list[dict]:
    items = data.load_items(cfg.data.splits, cfg.data.hops, cfg.data.limit_per_split_hop)
    n, s = cfg.data.num_shards, cfg.data.shard
    return [it for i, it in enumerate(items) if i % n == s] if n > 1 else items


def generate_cells(cfg: Config, cells: list[dict], out_path, step: str) -> None:
    """Generate every cell not yet in out_path, in batches; append results after each batch.

    Each cell needs: item_id, condition, k, error_type, prompt_ids, max_tokens.
    """
    done = done_keys(out_path)
    todo = [c for c in cells if cell_key(c) not in done]
    print(f"[{step}] {len(cells)} cells, {len(cells) - len(todo)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return
    backend = make_backend(cfg)
    info = backend.info()
    for i in range(0, len(todo), cfg.batch_size):
        batch = todo[i: i + cfg.batch_size]
        seeds = [cell_seed(cfg.run_name, c["item_id"], c["condition"], c["k"], c["error_type"]) for c in batch]
        t0 = time.time()
        # one max_tokens per call: group by it (baseline vs closed-book differ)
        gens = [None] * len(batch)
        for mt in sorted({c["max_tokens"] for c in batch}):
            idx = [j for j, c in enumerate(batch) if c["max_tokens"] == mt]
            for j, g in zip(idx, backend.generate([batch[j]["prompt_ids"] for j in idx], [seeds[j] for j in idx], mt)):
                gens[j] = g
        dt = max(time.time() - t0, 1e-6)
        rows = []
        for c, s, g in zip(batch, seeds, gens):
            row = {k: v for k, v in c.items() if k != "prompt_ids"}
            row.update(text=g.text, n_gen_tokens=g.n_gen_tokens, truncated=g.finish_reason == "length",
                       seed=s, prompt_len=len(c["prompt_ids"]), **info)
            rows.append(row)
        append_jsonl(out_path, rows)
        n_tok = sum(g.n_gen_tokens for g in gens)
        append_jsonl(Path(out_path).parent / "timing.jsonl",
                     [{"step": step, "n_prompts": len(batch), "n_tokens": n_tok, "seconds": dt}])
        print(f"[{step}] {i + len(batch)}/{len(todo)}  batch {len(batch)}: {n_tok} tok in {dt:.1f}s = {n_tok / dt:.1f} tok/s",
              flush=True)


# ---------------------------------------------------------------- baseline

def cmd_baseline(args):
    """Own clean chain (with paragraphs) + closed-book run, one sample each."""
    cfg = load_config(args.config)
    if cfg.reuse_baseline_from and not (cfg.run_dir / "baseline.jsonl").exists():
        # same model, mode, prompt and items: reuse the existing baseline instead of regenerating it
        src = cfg.run_dir.parent / cfg.reuse_baseline_from / "baseline.jsonl"
        if not src.exists():
            sys.exit(f"reuse_baseline_from: {src} not found (run that config's baseline first)")
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, cfg.run_dir / "baseline.jsonl")
        print(f"[baseline] reused {src}")
    items = run_items(cfg)
    tok = load_tokenizer(cfg)
    cells = []
    for it in items:
        for cond, closed in (("baseline", False), ("closed_book", True)):
            cells.append({
                "item_id": it["id"], "split": it["split"], "K": it["K"], "condition": cond,
                "k": 0, "error_type": None,
                "prompt_ids": build_ids(tok, cfg.model_id, user_message(it, closed_book=closed), cfg.think),
                "max_tokens": (cfg.closed_book_max_tokens or cfg.decoding.max_tokens) if closed else cfg.decoding.max_tokens,
            })
    generate_cells(cfg, cells, cfg.run_dir / "baseline.jsonl", "baseline")


# ---------------------------------------------------------------- eligibility

def hop_matches(pred: str, gold_hop: str, extra_golds: list[str], f1_threshold: float) -> bool:
    return hop_answer_matches(pred, [gold_hop, *extra_golds], f1_threshold)


def model_chain(text: str, think: bool, hop_source: str) -> dict:
    """Split one output into the parts eligibility and injection need."""
    think_part, answer_part, closed = split_think(text, think)
    hop_text = think_part if hop_source == "think" else answer_part
    return {"think_part": think_part, "answer_part": answer_part, "think_closed": closed,
            "hop_text": hop_text, "hops": parse_hops(hop_text), "pred": extract_final(answer_part)}


def align_hops(model_answers: list[str], item: dict, thr: float) -> tuple[list[int] | None, str | None]:
    """Map the model's K hops onto the gold hops. Returns (hop_map, failure reason).

    Gold order first. In branching items independent hops may come in either order, so any
    order that matches every hop AND respects the gold dependencies (#N refs) is accepted:
    a hop may only come after all hops it depends on. Then position k is still exact, and
    `remaining` (dependents of the hop) is unchanged.
    """
    K = item["K"]
    extra = lambda g: item["answer_aliases"] + [item["answer"]] if g == K - 1 else []  # noqa: E731
    ok = [[hop_matches(a, item["hops"][g]["answer"], extra(g), thr) for g in range(K)] for a in model_answers]
    if all(ok[j][j] for j in range(K)):
        return list(range(K)), None
    any_order = False
    for perm in itertools.permutations(range(K)):
        if all(ok[j][perm[j]] for j in range(K)):
            any_order = True
            pos = {g: j for j, g in enumerate(perm)}
            if all(pos[d - 1] < pos[g] for g in range(K) for d in item["hops"][g]["depends_on"]):
                return list(perm), None
    if any_order:
        return None, "hop_order_violates_dependencies"
    return None, "hop_mismatch_" + ",".join(str(j + 1) for j in range(K) if not ok[j][j])


def check_item(item: dict, base: dict, closed: dict, cfg: Config) -> dict:
    """Doc §6.1 eligibility. Returns a record with every reason that fails."""
    K, thr = item["K"], cfg.eligibility.f1_threshold
    golds = [item["answer"], *item["answer_aliases"]]
    ch = model_chain(base["text"], cfg.think, cfg.eligibility.hop_source)
    reasons = []
    if base["truncated"]:
        reasons.append("truncated")
    if ch["pred"] is None:
        reasons.append("no_final_answer")
    elif not exact_match(ch["pred"], golds):
        reasons.append("final_wrong")
    hops = ch["hops"]
    well_formed = hops_well_formed(hops, K)
    hop_map = None
    if not well_formed:
        reasons.append(f"hop_count_{len(hops)}_vs_{K}")
    else:
        hop_map, reason = align_hops([h.answer for h in hops], item, thr)
        if reason:
            reasons.append(reason)
    closed_ch = model_chain(closed["text"], cfg.think, cfg.eligibility.hop_source)
    closed_correct = exact_match(closed_ch["pred"], golds)
    if closed_correct:
        reasons.append("closed_book_correct")

    relaxed = cfg.eligibility.debug_relaxed
    eligible = well_formed if relaxed else not reasons
    chain_text, chain_source = ch["hop_text"], "model"
    if relaxed and not well_formed and cfg.eligibility.debug_gold_chain:
        chain_text, chain_source, eligible = gold_debug_chain(item), "gold_debug", True
    # hop-format diagnostics for think mode (stage-2 report)
    n_answer = len(parse_hops(ch["answer_part"]))
    n_think = len(parse_hops(ch["think_part"]))
    return {
        "item_id": item["id"], "split": item["split"], "K": K, "composition": item["composition"],
        "is_linear": item["is_linear"], "eligible": eligible, "eligibility_relaxed": relaxed,
        "reasons": reasons, "pred": ch["pred"], "closed_book_pred": closed_ch["pred"],
        "closed_book_correct": closed_correct, "model_hops": [h.answer for h in hops],
        "gold_hops": [h["answer"] for h in item["hops"]],
        # model hop position -> gold hop index (identity unless independent branches were reordered)
        "hop_map": hop_map if hop_map is not None else list(range(K)),
        "reordered": hop_map is not None and hop_map != list(range(K)),
        "hops_in_answer": n_answer == K, "hops_in_think": n_think == K,
        "n_gen_tokens": base["n_gen_tokens"], "truncated": base["truncated"],
        # text the inject step cuts the prefix from (the model's own hop lines)
        "chain_text": chain_text, "chain_source": chain_source,
    }


def gold_debug_chain(item: dict) -> str:
    """CPU-debug stand-in chain from gold hops (#N refs filled in). Never used in real runs."""
    lines = []
    for i, h in enumerate(item["hops"], start=1):
        q = re.sub(r"#(\d+)", lambda m: item["hops"][int(m.group(1)) - 1]["answer"], h["question"])
        lines.append(f"Hop {i}: {q} -> {h['answer']}")
    return "\n".join(lines) + f"\nFinal answer: {item['answer']}"


def cmd_eligible(args):
    cfg = load_config(args.config)
    items = {it["id"]: it for it in run_items(cfg)}
    rows = read_jsonl(cfg.run_dir / "baseline.jsonl")
    by = {(r["item_id"], r["condition"]): r for r in rows}
    out = []
    for iid, it in items.items():
        base, closed = by.get((iid, "baseline")), by.get((iid, "closed_book"))
        if base is None or closed is None:
            continue
        out.append(check_item(it, base, closed, cfg))
    write_jsonl(cfg.run_dir / "eligibility.jsonl", out)
    print_eligibility(out, cfg)


def _pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "0/0"


def print_eligibility(rows: list[dict], cfg: Config) -> None:
    n = len(rows)
    print(f"\n=== eligibility: {cfg.run_name} ({cfg.mode}) ===")
    if cfg.eligibility.debug_relaxed:
        print("!! debug_relaxed: eligibility only requires K hop lines (CPU debug run)")
    n_gold = sum(r["chain_source"] == "gold_debug" for r in rows)
    if n_gold:
        print(f"!! debug_gold_chain: {n_gold} items use a GOLD stand-in chain (pipeline test only)")
    print(f"eligible          {_pct(sum(r['eligible'] for r in rows), n)}")
    print(f"strict-eligible   {_pct(sum(not r['reasons'] for r in rows), n)}")
    for K in sorted({r["K"] for r in rows}):
        sub = [r for r in rows if r["K"] == K]
        lin = [r for r in sub if r["is_linear"]]
        print(f"  K={K}: eligible {_pct(sum(r['eligible'] for r in sub), len(sub))}; "
              f"linear {_pct(sum(r['eligible'] for r in lin), len(lin))}")
    print(f"final answer correct  {_pct(sum('final_wrong' not in r['reasons'] and 'no_final_answer' not in r['reasons'] for r in rows), n)}")
    print(f"closed-book correct   {_pct(sum(r['closed_book_correct'] for r in rows), n)}")
    print(f"truncated             {_pct(sum(r['truncated'] for r in rows), n)}")
    print(f"K hop lines in answer part  {_pct(sum(r['hops_in_answer'] for r in rows), n)}")
    if cfg.think:
        print(f"K hop lines in think block  {_pct(sum(r['hops_in_think'] for r in rows), n)}")
    reasons = Counter(x.split("_vs_")[0] if x.startswith("hop_count") else
                      ("hop_mismatch" if x.startswith("hop_mismatch") else x)
                      for r in rows for x in r["reasons"])
    print("failure reasons (an item can have several):")
    for k, v in reasons.most_common():
        print(f"  {k:22s} {v}")
    toks = [r["n_gen_tokens"] for r in rows]
    if toks:
        print(f"baseline gen tokens: mean {sum(toks) / len(toks):.0f}, max {max(toks)}")
    timing = [t for t in read_jsonl(cfg.run_dir / "timing.jsonl") if t["step"] == "baseline"]
    if timing:
        print(f"throughput: {sum(t['n_tokens'] for t in timing) / sum(t['seconds'] for t in timing):.1f} tok/s")


# ---------------------------------------------------------------- vet

def vet_cache(judge_cfg: Config):
    return cache_path(REPO_ROOT, judge_cfg.run_name)


def inject_inputs(cfg: Config):
    """What inject and vet both need: items, baselines, eligible records, off-context pool."""
    items = {it["id"]: it for it in run_items(cfg)}
    base = {r["item_id"]: r for r in read_jsonl(cfg.run_dir / "baseline.jsonl") if r["condition"] == "baseline"}
    elig = [r for r in read_jsonl(cfg.run_dir / "eligibility.jsonl") if r["eligible"]]
    off_pool = build_off_pool(data.load_items(["dev", "train"], [3, 4]))   # big pool, all prepared items
    return items, base, elig, off_pool


def run_judge(judge_cfg: Config, triples: list[tuple[str, str, str]], cache_file=None) -> dict[str, bool]:
    """Judge (sub-question, gold, candidate) triples not yet in the cache; append verdicts after each
    batch (resumable). Returns the full cache as vet_key -> ok."""
    cache_file = cache_file or vet_cache(judge_cfg)
    cache = load_cache(cache_file)
    todo, seen = [], set()
    for t in triples:
        key = vet_key(*t)
        if key not in cache and key not in seen:
            seen.add(key)
            todo.append(t)
    print(f"[vet] {len(triples)} triples, {len(todo)} new to judge", flush=True)
    if not todo:
        return cache
    tok = load_tokenizer(judge_cfg)
    backend = make_backend(judge_cfg)
    meta = {"judge_model": judge_cfg.model_id, "judge_revision": judge_cfg.revision}
    for i in range(0, len(todo), judge_cfg.batch_size):
        batch = todo[i: i + judge_cfg.batch_size]
        ids = [build_ids(tok, judge_cfg.model_id, judge_message(*t), think=False) for t in batch]
        gens = backend.generate(ids, [0] * len(batch), judge_cfg.decoding.max_tokens)
        rows = []
        for (subq, gold, cand), g in zip(batch, gens):
            v = parse_verdict(g.text)
            rows.append({"key": vet_key(subq, gold, cand), "subq": subq, "gold": gold, "cand": cand,
                         **v, "raw": g.text, **meta})
            cache[rows[-1]["key"]] = v["ok"]
        append_jsonl(cache_file, rows)
        print(f"[vet] {i + len(batch)}/{len(todo)} judged; accepted so far "
              f"{sum(r['ok'] for r in rows)}/{len(rows)} in this batch", flush=True)
    return cache


def cmd_vet(args):
    """Judge planted-value candidates in rank order, a few per cell per round, until each cell has
    an accepted candidate or vet_max_per_cell were rejected. Run before `inject`."""
    cfg = load_config(args.config)
    if not cfg.vet_values:
        print("[vet] vet_values is false in this config: nothing to vet")
        return
    items, base, elig, off_pool = inject_inputs(cfg)
    slots = []   # (subq, gold, [first vet_max_per_cell valid candidates in rank order])
    for e in elig:
        it = items[e["item_id"]]
        hops = parse_hops(e["chain_text"])
        clean_text = e["chain_text"] + "\n" + base[it["id"]]["text"]
        hop_map = e.get("hop_map") or list(range(it["K"]))
        for k in range(1, it["K"] + 1):
            g = hop_map[k - 1]
            subq, gold = judge_question(it, k, g)
            for et in ("in_context", "off_context"):
                gen = valid_candidates(it, k, hops[k - 1].answer, et, clean_text, off_pool, gold_idx=g)
                cands = [t for t, _ in itertools.islice(gen, cfg.vet_max_per_cell) if t is not None]
                if cands:
                    slots.append((subq, gold, cands))
    judge_cfg = judge_config(cfg)
    cache = load_cache(vet_cache(judge_cfg))
    for rnd in itertools.count(1):
        triples = []
        for subq, gold, cands in slots:
            verdicts = [cache.get(vet_key(subq, gold, c)) for c in cands]
            if any(verdicts):
                continue
            new = [c for c, v in zip(cands, verdicts) if v is None][:cfg.vet_per_round]
            triples += [(subq, gold, c) for c in new]
        if not triples:
            break
        print(f"[vet] round {rnd}: {len(triples)} candidates", flush=True)
        cache = run_judge(judge_cfg, triples)
    n_ok = sum(any(cache.get(vet_key(s, g, c)) for c in cands) for s, g, cands in slots)
    print(f"[vet] {cfg.run_name}: {n_ok}/{len(slots)} cells have an accepted value "
          f"({len(slots) - n_ok} will be dropped as vet_rejected)")


# ---------------------------------------------------------------- inject

def cmd_inject(args):
    """Build every cell for every eligible item (doc §6.3); log dropped cells with reasons."""
    cfg = load_config(args.config)
    items, base, elig, off_pool = inject_inputs(cfg)
    vetted = None
    if cfg.vet_values:
        vetted = load_cache(vet_cache(judge_config(cfg)))
        if not vetted:
            sys.exit(f"vet_values is on but {vet_cache(judge_config(cfg))} is empty: run `vet` first")
    cells, dropped = [], []
    for e in elig:
        it = items[e["item_id"]]
        K, chain = it["K"], e["chain_text"]
        hops = parse_hops(chain)
        clean_text = chain + "\n" + base[it["id"]]["text"]      # rule 4 checks everything the model wrote
        common = {
            "item_id": it["id"], "split": it["split"], "K": K, "composition": it["composition"],
            "is_linear": it["is_linear"], "gold": it["answer"], "gold_aliases": it["answer_aliases"],
            "chain_source": e["chain_source"], "eligibility_relaxed": e["eligibility_relaxed"],
            "reordered": e.get("reordered", False),
        }
        hop_map = e.get("hop_map") or list(range(K))   # model position -> gold hop index
        cells.append({**common, "condition": "noprefix", "k": 0, "error_type": None, "prefix": "",
                      "remaining": None, "remaining_linear": None, "rel_pos": None, "is_terminal": False,
                      "orig_value": None, "new_value": None})
        trace = split_think(base[it["id"]]["text"], True)[0].lstrip("\n") if cfg.injection_site == "trace" else ""
        for k in range(1, K + 1):
            g = hop_map[k - 1]
            pos = {"k": k, "gold_hop": g + 1, "remaining": it["hops"][g]["remaining_dep"], "remaining_linear": K - k,
                   "rel_pos": k / K, "is_terminal": k == K, "orig_value": hops[k - 1].answer}
            if cfg.injection_site == "trace":
                # first mention of the hop-k answer in the model's own thinking (its words or gold's)
                cands = [hops[k - 1].answer, it["hops"][g]["answer"]]
                if k == K:
                    cands += [it["answer"], *it["answer_aliases"]]
                hit = find_trace_mention(trace, cands)
                if hit and re.search(r"(?<!\w)" + re.escape(hit[2]) + r"(?!\w)", it["question"], re.IGNORECASE):
                    hit, why = None, "answer_in_question"     # first mention would just restate the question
                else:
                    why = "hop_answer_not_in_trace"
                prefix_fn = (lambda v=None: build_trace_prefix(trace, hit[0], hit[1], v)) if hit else None
                if hit and prefix_fn() is None:
                    hit, prefix_fn, why = None, None, "no_sentence_end_after_mention"
                if hit is None:
                    for cond, et in [("clean", None), ("inject", "in_context"), ("inject", "off_context")]:
                        dropped.append({**common, **pos, "condition": cond, "error_type": et, "reason": why})
                    continue
                pos.update(orig_value=hit[2], trace_frac=round(hit[1] / max(1, len(trace)), 3))
            else:
                prefix_fn = lambda v=None, k=k: build_prefix(chain, hops, k, v)  # noqa: E731
            cells.append({**common, **pos, "condition": "clean", "error_type": None,
                          "prefix": prefix_fn(), "new_value": None})
            for et in ("in_context", "off_context"):
                value, reason, n_cand = pick_value(it, k, hops[k - 1].answer, et, clean_text, off_pool, gold_idx=g,
                                                   vetted=vetted)
                if value is None:
                    dropped.append({**common, **pos, "condition": "inject", "error_type": et,
                                    "reason": reason, "n_candidates": n_cand})
                    continue
                cells.append({**common, **pos, "condition": "inject", "error_type": et,
                              "prefix": prefix_fn(value), "new_value": value})
    write_jsonl(cfg.run_dir / "cells.jsonl", cells)
    write_jsonl(cfg.run_dir / "dropped.jsonl", dropped)
    n_inj = sum(c["condition"] == "inject" for c in cells)
    print(f"{len(elig)} eligible items -> {len(cells)} cells ({n_inj} inject), {len(dropped)} cells dropped")
    for (cond, et, reason), n in Counter((d["condition"], d["error_type"] or "-", d["reason"].split(":")[-1])
                                         for d in dropped).most_common():
        print(f"  dropped {cond:6s} {et:12s} {reason:40s} {n}")
    # look at what will be planted before spending GPU time on it (the first run skipped this)
    inj = [c for c in cells if c["condition"] == "inject"]
    print("random planted values (sub-question | gold -> planted):")
    for c in random.Random(0).sample(inj, min(15, len(inj))):
        subq, gold = judge_question(items[c["item_id"]], c["k"], c["gold_hop"] - 1)
        print(f"  {subq[:70]} | {gold} -> {c['new_value']}  [{c['error_type']}]")


# ---------------------------------------------------------------- continue

def cmd_continue(args):
    """Run every cell (resumable: cells already in continuations.jsonl are skipped)."""
    cfg = load_config(args.config)
    items = {it["id"]: it for it in run_items(cfg)}
    tok = load_tokenizer(cfg)
    cells = read_jsonl(cfg.run_dir / "cells.jsonl")
    for c in cells:
        prefill = prefill_text(c["prefix"], cfg.think) if c["condition"] != "noprefix" else ""
        c["prompt_ids"] = build_ids(tok, cfg.model_id, user_message(items[c["item_id"]]), cfg.think, prefill)
        c["max_tokens"] = cfg.decoding.max_tokens
    out = cfg.run_dir / "continuations.jsonl"
    if cfg.reuse_cells_from and not out.exists():
        reuse_cells(cfg, cells, out)
    generate_cells(cfg, cells, out, "continue")


def reuse_cells(cfg: Config, cells: list[dict], out) -> None:
    """Copy noprefix/clean continuations from an earlier run of the same model and mode. A row is
    reused only if its prompt ids are identical, so it is exactly the generation this run would
    request (its seed came from the earlier run_name and is kept in the row)."""
    src = cfg.run_dir.parent / cfg.reuse_cells_from / "continuations.jsonl"
    old = {cell_key(r): r for r in read_jsonl(src) if r["condition"] in ("noprefix", "clean")}
    rows = []
    for c in cells:
        r = old.get(cell_key(c))
        if c["condition"] in ("noprefix", "clean") and r and r.get("prefix") == c["prefix"] \
                and r.get("prompt_len") == len(c["prompt_ids"]):
            rows.append({**r, "reused_from": cfg.reuse_cells_from})
    append_jsonl(out, rows)
    n_want = sum(c["condition"] in ("noprefix", "clean") for c in cells)
    print(f"[continue] reused {len(rows)}/{n_want} noprefix+clean cells from {src}", flush=True)


# ---------------------------------------------------------------- score

def full_output(row: dict, think: bool) -> str:
    """What the assistant turn contains: prefill + continuation."""
    if row["condition"] == "noprefix":
        return row["text"]
    return prefill_text(row["prefix"], think) + row["text"]


def cmd_score(args):
    cfg = load_config(args.config)
    rows = read_jsonl(cfg.run_dir / "continuations.jsonl")
    meta = {"run_id": cfg.run_name, "git_hash": git_hash(), "task": "musique", "model": cfg.model_id,
            "mode": cfg.mode_label, "injection_site": cfg.injection_site,
            "quant": cfg.quant, "gpu": cfg.gpu, "frame": "prefill",
            "model_revision": cfg.revision, "sample_idx": 0, "explicit_sc": None, "doubt_marker": None}
    scored = []
    for r in rows:
        _, answer_part, _ = split_think(full_output(r, cfg.think), cfg.think)
        pred = extract_final(answer_part)
        golds = [r["gold"], *r["gold_aliases"]]
        npred = normalize(pred)
        gold_in = any(normalize(g) and normalize(g) in npred for g in golds)
        new_in = bool(r["new_value"]) and normalize(r["new_value"]) in npred
        scored.append({**meta, **{k: v for k, v in r.items() if k != "text"},
                       "continuation": r["text"], "pred": pred,
                       # primary (doc §8.1): exact match of the extracted final answer
                       "correct": exact_match(pred, golds), "f1": max_f1(pred, golds),
                       "adopted": (npred == normalize(r["new_value"])) if r["new_value"] else None,
                       # secondary: containment within the final-answer line only (verbose answers)
                       "correct_contains": gold_in and not new_in,
                       "adopted_contains": (new_in and not gold_in) if r["new_value"] else None})
    # item difficulty from the noprefix run; clean-twin outcome attached to every cell at the same k
    noprefix = {s["item_id"]: s["correct"] for s in scored if s["condition"] == "noprefix"}
    clean = {(s["item_id"], s["k"]): s["correct"] for s in scored if s["condition"] == "clean"}
    for s in scored:
        s["item_difficulty"] = (1 - noprefix[s["item_id"]]) if s["item_id"] in noprefix else None
        s["clean_twin_correct"] = clean.get((s["item_id"], s["k"]))
    write_jsonl(cfg.run_dir / "scored.jsonl", scored)
    print(f"scored {len(scored)} rows -> {cfg.run_dir / 'scored.jsonl'}")
    for cond in ("noprefix", "clean", "inject"):
        sub = [s for s in scored if s["condition"] == cond]
        if sub:
            print(f"  {cond:9s} n={len(sub):4d} correct {_pct(sum(s['correct'] for s in sub), len(sub))}"
                  f"  truncated {sum(s['truncated'] for s in sub)}"
                  + (f"  adopted {sum(bool(s['adopted']) for s in sub)}" if cond == "inject" else ""))


# ---------------------------------------------------------------- estimate

def cmd_estimate(args):
    """GPU hours for N items: baseline (2 gens/item) + cells for the eligible share."""
    cfg = load_config(args.config)
    base = read_jsonl(cfg.run_dir / "baseline.jsonl")
    elig = read_jsonl(cfg.run_dir / "eligibility.jsonl")
    cont = read_jsonl(cfg.run_dir / "continuations.jsonl")
    cells = read_jsonl(cfg.run_dir / "cells.jsonl")
    if not base or not elig:
        sys.exit("run baseline + eligible first (a small GPU pilot)")
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    base_tok = mean([r["n_gen_tokens"] for r in base])
    cont_tok = mean([r["n_gen_tokens"] for r in cont]) if cont else base_tok
    elig_rate = args.elig_rate if args.elig_rate is not None else mean([r["eligible"] for r in elig])
    n_elig_items = max(1, len({c["item_id"] for c in cells}))
    cells_per_item = len(cells) / n_elig_items if cells else 1 + 3 * 3.5   # 1 + K + 2K, mean K~3.5
    tokens = args.n_items * 2 * base_tok + args.n_items * elig_rate * cells_per_item * cont_tok
    if args.tok_per_s is None:   # measured batched throughput of this run
        timing = read_jsonl(cfg.run_dir / "timing.jsonl")
        args.tok_per_s = sum(t["n_tokens"] for t in timing) / sum(t["seconds"] for t in timing)
        print(f"using measured throughput {args.tok_per_s:.0f} tok/s (pilot batches are smaller than main-run batches, so this is pessimistic)")
    hours = tokens / args.tok_per_s / 3600
    print(f"{cfg.run_name}: mean tokens baseline {base_tok:.0f}, continuation {cont_tok:.0f}{'' if cont else ' (assumed = baseline)'}")
    print(f"eligible rate {elig_rate:.2f}, cells/eligible item {cells_per_item:.1f}")
    print(f"{args.n_items} items -> {tokens / 1e6:.1f}M generated tokens -> {hours:.1f} h at {args.tok_per_s:.0f} tok/s")


# ---------------------------------------------------------------- CLI

def cmd_prepare_data(args):
    data.prepare(splits=args.splits)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(prog="python -m src.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-data", help="download MuSiQue, filter 3/4-hop, build contexts")
    p.add_argument("--splits", nargs="+", default=["dev", "train"])
    p.set_defaults(fn=cmd_prepare_data)

    for name, fn, help_ in [
        ("baseline", cmd_baseline, "own chain + closed-book per item"),
        ("eligible", cmd_eligible, "doc §6.1 eligibility filter"),
        ("vet", cmd_vet, "LLM judge vets planted-value candidates (before inject)"),
        ("inject", cmd_inject, "build all cells + validity drops"),
        ("continue", cmd_continue, "generate all cells (resumable)"),
        ("score", cmd_score, "EM/F1 + adopted -> scored.jsonl"),
        ("estimate", cmd_estimate, "projected GPU hours"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", required=True)
        if name == "estimate":
            p.add_argument("--tok-per-s", type=float, default=None,
                           help="batched throughput; default = measured in this run's timing.jsonl")
            p.add_argument("--n-items", type=int, required=True, help="items to run baseline on")
            p.add_argument("--elig-rate", type=float, default=None, help="override measured eligible rate")
        p.set_defaults(fn=fn)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
