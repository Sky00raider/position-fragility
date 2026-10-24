"""Why do clean twins fail? Categorize every failing intermediate clean twin (CPU only).

    python scripts/clean_twin_audit.py runs/qwen3_8b_think runs/qwen3_8b_nothink runs/llama31_8b_nothink

Categories (first that applies):
  truncated            continuation hit max_tokens
  no_final_line        no "Final answer:" after </think> (pred=None, wrong by rule 8)
  extraction_artifact  gold is written as a "Final answer:" anywhere, but pred is not gold
  same_chain_granular  every continued hop matches the model's own baseline chain, final answer
                       differs only in form ("1769" for "August 3, 1769")
  diverged_next_hop    first hop after the splice differs from the model's own baseline hop
  diverged_later_hop   a later hop differs
Also prints generated-token counts and a paired McNemar test of clean twin vs noprefix.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalize import exact_match, hop_answer_matches  # noqa: E402
from src.parse import FINAL_RE, _clean_answer, parse_hops, split_think  # noqa: E402
from src.prompts import prefill_text  # noqa: E402


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open(encoding="utf-8")]


def answer_hops(r: dict, think: bool) -> list:
    """Hop lines of the scored part (after </think>) of prefill + continuation."""
    return parse_hops(split_think(prefill_text(r["prefix"], think) + r["continuation"], think)[1])


def category(r: dict, own_hops: list[str], think: bool) -> str:
    golds = [r["gold"], *r["gold_aliases"]]
    full = prefill_text(r["prefix"], think) + r["continuation"]
    if r["truncated"]:
        return "truncated"
    if r["pred"] is None:
        return "no_final_line"
    finals = [_clean_answer(full[m.end():].split("\n")[0]) for m in FINAL_RE.finditer(full)]
    if any(exact_match(a, golds) for a in finals):
        return "extraction_artifact"
    hops = answer_hops(r, think)
    for i in range(r["k"], r["K"]):
        if i >= len(hops) or not hop_answer_matches(hops[i].answer, [own_hops[i]]):
            return "diverged_next_hop" if i == r["k"] else "diverged_later_hop"
    return "same_chain_granular"


def audit(run_dir: Path) -> None:
    rows = read(run_dir / "scored.jsonl")
    think = rows[0]["mode"] == "think"
    elig = {e["item_id"]: e for e in read(run_dir / "eligibility.jsonl")}
    nop = {r["item_id"]: r for r in rows if r["condition"] == "noprefix"}
    clean_all = [r for r in rows if r["condition"] == "clean" and not r["is_terminal"]]
    print(f"\n=== {run_dir.name} ({'think' if think else 'nothink'}) ===")
    for cond in ("noprefix", "clean", "inject"):
        toks = [r["n_gen_tokens"] for r in rows if r["condition"] == cond]
        print(f"  {cond:8s} median generated tokens {st.median(toks):.0f}")
    for scope, clean in (("all", clean_all), ("linear", [r for r in clean_all if r["is_linear"]])):
        failed = [r for r in clean if not r["correct"]]
        cats = Counter(category(r, elig[r["item_id"]]["model_hops"], think) for r in failed)
        kept = sum([h.answer for h in answer_hops(r, think)][: r["k"]] == [h.answer for h in parse_hops(r["prefix"])]
                   for r in failed)
        a = sum(nop[r["item_id"]]["correct"] and not r["correct"] for r in clean)
        b = sum(not nop[r["item_id"]]["correct"] and r["correct"] for r in clean)
        print(f"  [{scope}] intermediate clean twins {len(clean)}, failed {len(failed)} "
              f"({100 * len(failed) / max(len(clean), 1):.0f}%)")
        for c, n in sorted(cats.items()):
            print(f"      {c:22s} {n:4d} ({100 * n / len(failed):.0f}%)")
        print(f"      final answer keeps the prefilled hops 1..k: {kept}/{len(failed)}")
        print(f"      paired vs noprefix: clean wrong/noprefix right {a}, clean right/noprefix wrong {b}, "
              f"McNemar exact p={binomtest(a, a + b).pvalue:.2g}")
    if think:
        no_prose = sum(all(parse_hops(ln) or FINAL_RE.search(ln) for ln in
                           r["continuation"].split("</think>")[0].split("\n") if ln.strip()) for r in clean_all)
        print(f"  clean continuations whose thinking is only hop/final lines (no reasoning prose): "
              f"{no_prose}/{len(clean_all)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for d in sys.argv[1:]:
        audit(Path(d))
