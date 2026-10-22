"""Hand audit of planted values (doc §6.5 rule 5): 100 injections checked by 2 people.

    python scripts/make_audit.py                     # writes docs/audit/audit_sheet.csv
    python scripts/make_audit.py --score FILE.csv    # after both reviewers filled it in

Sampling: dev items, 25 per error type x K (3, 4), random hop position, fixed seed.
Rule 4 (value already in the model's own chain) needs model output, so it is not applied here;
it only removes candidates, so the audited values are a superset of what the pipeline plants.

Reviewer columns (fill with y / n):
  type_ok    same kind of thing as the gold hop answer (person/place/year/org/...)
  plausible  a reader could believe it is the answer to the sub-question (not absurd)
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_items  # noqa: E402
from src.inject import build_off_pool, pick_value  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "audit" / "audit_sheet.csv"
REVIEW = ["A_type_ok", "A_plausible", "A_notes", "B_type_ok", "B_plausible", "B_notes"]


def make(n: int, seed: int) -> None:
    rng = random.Random(seed)
    items = load_items(["dev"], [3, 4])
    pool = build_off_pool(load_items(["dev", "train"], [3, 4]))
    rows, tried = [], 0
    order = items[:]
    rng.shuffle(order)
    for et, K in [(e, k) for e in ("in_context", "off_context") for k in (3, 4)]:   # n/4 per cell
        got = 0
        for it in (x for x in order if x["K"] == K):
            if got >= n // 4:
                break
            k = rng.randint(1, it["K"])
            h = it["hops"][k - 1]
            value, _, _ = pick_value(it, k, h["answer"], et, "", pool)
            tried += 1
            if value is None:
                continue
            subq = re.sub(r"#(\d+)", lambda m: it["hops"][int(m.group(1)) - 1]["answer"], h["question"])
            source = next((c["source"] for c in it["distractor_candidates"] if c["text"] == value), "other item")
            rows.append({"audit_id": len(rows) + 1, "item_id": it["id"], "error_type": et, "K": it["K"], "k": k,
                         "question": it["question"], "hop_subquestion": subq, "gold_hop_answer": h["answer"],
                         "type_tag": h["tag"], "planted_value": value,
                         "value_source": source if et == "in_context" else "other item",
                         **{c: "" for c in REVIEW}})
            got += 1
    rng.shuffle(rows)     # reviewers should not see the two error types in blocks
    for i, r in enumerate(rows, 1):
        r["audit_id"] = i
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(path: str) -> None:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for col in ("type_ok", "plausible"):
        pairs = [(r[f"A_{col}"].strip().lower()[:1], r[f"B_{col}"].strip().lower()[:1]) for r in rows]
        pairs = [(a, b) for a, b in pairs if a in "yn" and b in "yn" and a and b]
        if not pairs:
            print(f"{col}: not filled in yet")
            continue
        a, b = zip(*pairs)
        both_yes = sum(x == y == "y" for x, y in pairs)
        print(f"{col:10s} n={len(pairs)}  A yes {a.count('y')}  B yes {b.count('y')}  both yes {both_yes} "
              f"({100 * both_yes / len(pairs):.0f}%)  agreement {100 * sum(x == y for x, y in pairs) / len(pairs):.0f}%  "
              f"Cohen's kappa {kappa(list(a), list(b)):.2f}")
        for et in ("in_context", "off_context"):
            sub = [(x, y) for (x, y), r in zip(pairs, rows) if r["error_type"] == et]
            if sub:
                print(f"    {et:12s} both yes {sum(x == y == 'y' for x, y in sub)}/{len(sub)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--score", default=None, help="filled audit CSV to score")
    args = ap.parse_args()
    score(args.score) if args.score else make(args.n, args.seed)
