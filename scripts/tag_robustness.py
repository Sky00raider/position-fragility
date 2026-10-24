"""PSI by type-tag group of the corrupted hop (robustness check after the planted-value audit).

The audit found many planted values of the right coarse spaCy tag but the wrong kind of thing
(a state for a city, a TV network for a person), mostly for GPE/LOC/ORG/NORP. PERSON and
date/number values were mostly fine. This recomputes PSI (linear items) within each tag group;
if late-stage fragility holds in the clean group, it is not an artefact of implausible values.

Usage: python scripts/tag_robustness.py --runs runs/qwen3_8b_nothink runs/qwen3_8b_think runs/llama31_8b_nothink
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.analysis import load_runs, psi_table  # noqa: E402
from src.store import read_jsonl  # noqa: E402

GROUPS = {"PERSON": "person/date/number", "DATE": "person/date/number", "DATE_MDY": "person/date/number",
          "YEAR": "person/date/number", "NUMBER": "person/date/number",
          "GPE": "place (GPE/LOC)", "LOC": "place (GPE/LOC)"}   # everything else: ORG/NORP/other


def hop_tags() -> dict:
    tags = {}
    for f in (REPO_ROOT / "data" / "items").glob("*.jsonl"):
        for it in read_jsonl(f):
            tags[it["id"]] = [h["tag"] for h in it["hops"]]
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "analysis_main" / "tables" / "psi_by_tag_linear.csv"))
    args = ap.parse_args()

    df = load_runs(args.runs)
    df = df[df["is_linear"] & (df["condition"] == "inject")].copy()
    tags = hop_tags()
    hop = df["gold_hop"] if "gold_hop" in df else df["k"]
    hop = hop.fillna(df["k"]).astype(int)
    df["tag"] = [tags[i][h - 1] for i, h in zip(df["item_id"], hop)]
    df["tag_group"] = df["tag"].map(GROUPS).fillna("org/norp/other")

    rng, out = np.random.default_rng(0), []
    for grp, g in df.groupby("tag_group"):
        t = psi_table(g, args.reps, rng)
        t.insert(0, "tag_group", grp)
        out.append(t)
    res = pd.concat(out)
    res.to_csv(args.out, index=False)
    cols = ["tag_group", "model", "mode", "error_type", "K", "n_early", "n_late", "RR_early", "RR_late",
            "PSI", "ci_lo", "ci_hi"]
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(res[cols].round(3).to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
