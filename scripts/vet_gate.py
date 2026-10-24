"""Gate: is the vet judge good enough to choose planted values? Run on the GPU box BEFORE inject.

Judges the 300 reference triples in data/vet/reference.csv (planted values from the first main run,
labeled by an independent rater: 100 audit rows + a 200-cell sample; ~39% acceptable) and compares.
What matters is precision: of the values the judge accepts, how many the reference also accepts,
because accepted values are the ones we plant. Exit code 1 (stop the run) if
  precision < --min-precision  or  acceptance rate < --min-accept (too strict: too many drops).

    python scripts/vet_gate.py --config configs/qwen3_8b_nothink_v2.yaml

Why this exists: the first run's planted values were chosen by coarse tag match only and nobody
looked at a sample before the GPU run; half turned out to be the wrong kind of thing (HISTORY
2026-10-24). Every planted-value change must pass this gate before it is used.
"""
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.config import judge_config, load_config  # noqa: E402
from src.pipeline import run_judge  # noqa: E402
from src.vet import vet_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a run config with vet_values/vet_judge set")
    ap.add_argument("--reference", default=str(REPO_ROOT / "data" / "vet" / "reference.csv"))
    ap.add_argument("--min-precision", type=float, default=0.75)
    ap.add_argument("--min-accept", type=float, default=0.20)
    args = ap.parse_args()

    ref = list(csv.DictReader(open(args.reference, encoding="utf-8")))
    judge = judge_config(load_config(args.config))
    cache = run_judge(judge, [(r["subq"], r["gold"], r["cand"]) for r in ref])
    pairs = [(cache[vet_key(r["subq"], r["gold"], r["cand"])], r["ref_ok"] == "1") for r in ref]
    tp = sum(j and r for j, r in pairs)
    fp = sum(j and not r for j, r in pairs)
    fn = sum(r and not j for j, r in pairs)
    tn = sum(not j and not r for j, r in pairs)
    n = len(pairs)
    accept = (tp + fp) / n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"judge {judge.model_id} on {n} reference values (reference acceptable: {(tp + fn) / n:.0%})")
    print(f"  judge accepts {accept:.0%}; precision {precision:.2f}; recall {recall:.2f}; "
          f"agreement {(tp + tn) / n:.2f}")
    print(f"  confusion: accept&ok {tp}, accept&bad {fp}, reject&ok {fn}, reject&bad {tn}")
    bad = [r for (j, ok), r in zip(pairs, ref) if j and not ok][:10]
    if bad:
        print("  accepted but reference says bad (first 10):")
        for r in bad:
            print(f"    {r['subq'][:60]!r} gold={r['gold']!r} planted={r['cand']!r}")
    ok = precision >= args.min_precision and accept >= args.min_accept
    print("GATE PASSED" if ok else f"GATE FAILED (need precision >= {args.min_precision}, "
                                  f"acceptance >= {args.min_accept}): do not run inject")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
