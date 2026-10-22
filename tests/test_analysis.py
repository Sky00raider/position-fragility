"""Analysis runs end to end on synthetic data and recovers a planted position effect."""
import json

import numpy as np
import pandas as pd

from src import analysis


def synthetic_rows(n_items=120, seed=0):
    """Recovery rises with k (late errors recover more): true PSI > 0."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        K = 3 if i % 2 else 4
        lin = i % 5 != 0
        rem = {3: [2, 1, 0], 4: [3, 2, 1, 0]}[K]
        base = dict(run_id="syn", model="org/Fake-8B", mode="nothink" if i % 3 else "think", item_id=f"i{i}",
                    K=K, is_linear=lin, gold="g", eligibility_relaxed=False, chain_source="model")
        np_ok = bool(rng.random() < 0.7)
        rows.append({**base, "condition": "noprefix", "k": 0, "error_type": None, "correct": np_ok,
                     "truncated": False, "adopted": None, "is_terminal": False, "rel_pos": None,
                     "remaining": None, "item_difficulty": 1 - np_ok, "clean_twin_correct": None})
        for k in range(1, K + 1):
            pos = dict(k=k, rel_pos=k / K, remaining=rem[k - 1], is_terminal=k == K, item_difficulty=1 - np_ok)
            clean_ok = bool(rng.random() < 0.95)
            rows.append({**base, **pos, "condition": "clean", "error_type": None, "correct": clean_ok,
                         "truncated": False, "adopted": None, "clean_twin_correct": clean_ok})
            for et in ("in_context", "off_context"):
                p = 0.2 + 0.5 * (k - 1) / (K - 1) if k < K else 0.1
                ok = bool(rng.random() < p)
                rows.append({**base, **pos, "condition": "inject", "error_type": et, "correct": ok,
                             "truncated": bool(rng.random() < 0.03), "adopted": (not ok) and rng.random() < 0.5,
                             "clean_twin_correct": clean_ok})
    return rows


def test_analysis_end_to_end(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "scored.jsonl", "w", encoding="utf-8") as f:
        for r in synthetic_rows():
            f.write(json.dumps(r) + "\n")
    out = tmp_path / "analysis"
    analysis.main(["--runs", str(run), "--out", str(out), "--reps", "50"])
    for name in ["recovery_linear", "psi_linear", "mcnemar_linear", "matched_remaining_linear",
                 "terminal_linear", "clean_twin_linear", "truncation", "recovery_all_notrunc"]:
        assert (out / "tables" / f"{name}.csv").exists(), name
    assert (out / "figures" / "recovery_linear.png").exists()
    assert (out / "figures" / "psi_linear.png").exists()
    psi = pd.read_csv(out / "tables" / "psi_all.csv")
    assert (psi["PSI"] > 0).mean() > 0.7            # planted effect is recovered
    term = pd.read_csv(out / "tables" / "terminal_all.csv")
    assert (term["diff"] < 0).all()                  # terminal hop recovers least
    assert "rel_pos" in (out / "logit_all.txt").read_text(encoding="utf-8")
