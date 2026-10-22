"""Print N full injected prompts + continuations for human checking.

    python scripts/sanity.py --config configs/qwen3_8b_nothink.yaml --n 5

Shows exactly what the model saw (decoded token ids, chat template included), the planted
value, the continuation, and how it was scored. Paragraphs are shortened for readability
unless --full is given.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backends import load_tokenizer  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import load_items  # noqa: E402
from src.prompts import build_ids, prefill_text, user_message  # noqa: E402
from src.store import read_jsonl  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="do not shorten the paragraphs")
    args = ap.parse_args()
    cfg = load_config(args.config)

    rows = [r for r in read_jsonl(cfg.run_dir / "scored.jsonl") if r["condition"] == "inject"]
    if not rows:
        sys.exit("no scored inject rows yet: run inject, continue, score first")
    items = {it["id"]: it for it in load_items(cfg.data.splits, cfg.data.hops, cfg.data.limit_per_split_hop)}
    tok = load_tokenizer(cfg)
    for r in random.Random(args.seed).sample(rows, min(args.n, len(rows))):
        it = items[r["item_id"]]
        ids = build_ids(tok, cfg.model_id, user_message(it), cfg.think, prefill_text(r["prefix"], cfg.think))
        prompt = tok.decode(ids)
        if not args.full:
            head, sep, tail = prompt.partition("Question:")
            prompt = head[:600] + "\n[... paragraphs shortened ...]\n" + sep + tail if sep else prompt
        print("=" * 100)
        print(f"{r['item_id']}  K={r['K']} k={r['k']} {r['error_type']}  terminal={r['is_terminal']}  "
              f"chain_source={r.get('chain_source')}")
        print(f"gold hops: {[h['answer'] for h in it['hops']]}  | final gold: {r['gold']}")
        print(f"planted: {r['orig_value']!r} -> {r['new_value']!r}")
        print("-" * 40, "PROMPT (decoded token ids)", "-" * 40)
        print(prompt)
        print("-" * 40, "CONTINUATION", "-" * 40)
        print(r["continuation"])
        print("-" * 40)
        print(f"pred={r['pred']!r} correct={r['correct']} adopted={r['adopted']} "
              f"clean_twin_correct={r.get('clean_twin_correct')} truncated={r['truncated']}")


if __name__ == "__main__":
    main()
