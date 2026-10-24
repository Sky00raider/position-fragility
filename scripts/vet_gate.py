"""Gate: is the vet judge good enough to choose planted values? Run on the GPU box BEFORE vet/inject.

Judges the 300 reference triples in data/vet/reference.csv (planted values from the first main run,
labeled by an independent rater: 100 audit rows + a 200-cell sample; ~39% acceptable) and compares.
What matters is precision: of the values the judge accepts, how many the reference also accepts,
because accepted values are the ones we plant. Acceptance must not be too low either, or too many
cells end up with no value (each cell tries up to vet_max_per_cell candidates).

The reference is split in two fixed halves (by a hash of the key). The decision rule (src/vet.py
RULES: both answers yes / believable / same_kind) is chosen on the TUNE half: highest precision among
rules with acceptance >= --min-accept. It is then checked on the HOLDOUT half, so the gate never
grades a choice on the data it was made on. Pass -> the rule is written to
runs/vet/<judge>_p<prompt version>.rule.json, which vet and inject require. Fail -> exit code 1,
no rule file, and run_main.sh stops before anything else runs.

    python scripts/vet_gate.py --config configs/qwen3_8b_nothink_v2.yaml

Why this exists: the first run's planted values were chosen by coarse tag match only and nobody
looked at a sample before the GPU run; half turned out to be the wrong kind of thing (HISTORY
2026-10-24). Every planted-value change must pass this gate before it is used.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.config import judge_config, load_config  # noqa: E402
from src.pipeline import run_judge  # noqa: E402
from src.vet import PROMPT_VERSION, RULES, rule_path, vet_key  # noqa: E402


def metrics(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs = (judge accepts, reference ok)."""
    n = len(pairs)
    tp = sum(j and r for j, r in pairs)
    fp = sum(j and not r for j, r in pairs)
    fn = sum(r and not j for j, r in pairs)
    return {"n": n, "accept": (tp + fp) / n if n else 0.0, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0, "tp": tp, "fp": fp}


def half(key: str) -> str:
    return "tune" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0 else "holdout"


def choose_rule(rows: list[dict], min_accept: float) -> str | None:
    """Highest precision among rules with enough acceptance (ties: higher acceptance)."""
    best = None
    for name in RULES:
        m = metrics([(r["judge"][name], r["ok"]) for r in rows])
        if m["accept"] >= min_accept and (best is None or (m["precision"], m["accept"]) > best[1]):
            best = (name, (m["precision"], m["accept"]))
    return best[0] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a run config with vet_values/vet_judge set")
    ap.add_argument("--reference", default=str(REPO_ROOT / "data" / "vet" / "reference.csv"))
    # Calibrated, not tuned: unfiltered v1 values reach ~0.39 against this reference, and a careful
    # human (audit rater B) reaches only 0.74, because the reference labels are themselves noisy
    # (human-LLM kappa 0.57 on plausibility). The first bar (0.75) was above the human ceiling.
    ap.add_argument("--min-precision", type=float, default=0.65)
    ap.add_argument("--min-accept", type=float, default=0.15)
    args = ap.parse_args()

    judge = judge_config(load_config(args.config))
    out_file = rule_path(REPO_ROOT, judge.run_name)
    out_file.unlink(missing_ok=True)          # a stale rule must never outlive a failed gate
    ref = list(csv.DictReader(open(args.reference, encoding="utf-8")))
    cache = run_judge(judge, [(r["subq"], r["gold"], r["cand"]) for r in ref])
    rows = []
    for r in ref:
        key = vet_key(r["subq"], r["gold"], r["cand"])
        v = cache[key]
        rows.append({**r, "key": key, "half": half(key), "ok": r["ref_ok"] == "1", "verdict": v,
                     "judge": {name: rule(v) for name, rule in RULES.items()}})
    unparsed = sum(r["verdict"]["same_kind"] is None or r["verdict"]["believable"] is None for r in rows)
    print(f"\njudge {judge.model_id} (prompt v{PROMPT_VERSION}) on {len(rows)} reference values; "
          f"reference acceptable {sum(r['ok'] for r in rows) / len(rows):.0%}; unparsed answers {unparsed}")
    for part in ("tune", "holdout", "all"):
        sub = [r for r in rows if part == "all" or r["half"] == part]
        for name in RULES:
            m = metrics([(r["judge"][name], r["ok"]) for r in sub])
            print(f"  {part:7s} rule={name:10s} n={m['n']:3d} accept {m['accept']:.0%}  precision {m['precision']:.2f}"
                  f"  recall {m['recall']:.2f}  (accepted: {m['tp']} ok, {m['fp']} bad)")
    if unparsed:
        ex = next(r for r in rows if r["verdict"]["same_kind"] is None or r["verdict"]["believable"] is None)
        print(f"  example unparsed answer: {ex['verdict'].get('raw', '')[:200]!r}")

    rule = choose_rule([r for r in rows if r["half"] == "tune"], args.min_accept)
    if rule is None:
        print(f"GATE FAILED: no rule reaches acceptance >= {args.min_accept:.0%} on the tune half. Do not run inject.")
        sys.exit(1)
    m = metrics([(r["judge"][rule], r["ok"]) for r in rows if r["half"] == "holdout"])
    bad = [r for r in rows if r["judge"][rule] and not r["ok"]][:8]
    if bad:
        print(f"  accepted under rule={rule} but reference says bad (first 8):")
        for r in bad:
            v = r["verdict"]
            print(f"    {r['subq'][:55]!r} gold={r['gold']!r} planted={r['cand']!r} "
                  f"(judge kinds: {v.get('kind_correct')} / {v.get('kind_candidate')})")
    passed = m["precision"] >= args.min_precision and m["accept"] >= args.min_accept
    print(f"chosen on tune half: rule={rule}; HOLDOUT: precision {m['precision']:.2f} (need >= {args.min_precision}), "
          f"acceptance {m['accept']:.0%} (need >= {args.min_accept:.0%})")
    if not passed:
        print("GATE FAILED: do not run inject")
        sys.exit(1)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({"rule": rule, "prompt_version": PROMPT_VERSION, "judge": judge.model_id,
                                    "holdout": m, "min_precision": args.min_precision,
                                    "min_accept": args.min_accept}, indent=1), encoding="utf-8")
    print(f"GATE PASSED (rule={rule}) -> {out_file}")


if __name__ == "__main__":
    main()
