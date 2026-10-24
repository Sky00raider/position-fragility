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


def _inject(cont, mode="nothink", correct=True, item="i0", k=1):
    return dict(model="Fake", mode=mode, item_id=item, K=3, k=k, error_type="in_context", is_terminal=False,
                condition="inject", clean_twin_correct=True, correct=correct, gold_hop_answer="Tucson",
                orig_value="Tucson", new_value="Maine", continuation=cont)


def test_sc_label_explicit_vs_silent():
    silent = _inject("Hop 2: Which city is next? -> Tucson\nFinal answer: 1905")
    assert analysis.sc_label(silent)["sc_label"] == "silent"           # right value, no flag
    flagged = _inject("Wait, that is wrong.\nThe second largest city is Tucson.\nFinal answer: 1905")
    assert analysis.sc_label(flagged)["sc_label"] == "explicit"        # marker, value in next sentence
    far = _inject("Wait, let me think.\nHop 2: ok -> x\nHop 3: city -> Tucson\nFinal answer: 1905")
    assert analysis.sc_label(far)["sc_label"] == "silent"              # value two sentences later
    planted = _inject("Actually Maine is a state, not a city. Final answer: 1905")
    assert analysis.sc_label(planted)["sc_label"] == "explicit"        # doubt next to the planted value
    assert analysis.sc_label(_inject("Hmm Tucsonville.", correct=True))["sc_label"] == "silent"  # whole words
    failed = _inject("However, Maine. Final answer: x", correct=False)
    lab = analysis.sc_label(failed)
    assert lab["sc_label"] == "not_recovered" and lab["doubt_marker"]


def test_sc_label_think_location():
    inside = analysis.sc_label(_inject("Hop 3: -> Maine\nWait, it should be Tucson.\n</think>\n\nFinal answer: 1905",
                                       mode="think"))
    assert inside["flag_think"] and not inside["flag_after"] and inside["sc_label"] == "explicit"
    after = analysis.sc_label(_inject("Hop 3: -> Maine\n</think>\nActually, it is Tucson.\nFinal answer: 1905",
                                      mode="think"))
    assert after["flag_after"] and not after["flag_think"]
    nothink = analysis.sc_label(_inject("Actually, it is Tucson.</think>"))
    assert nothink["flag_after"] and not nothink["flag_think"]          # nothink has no think block


def test_sc_table_and_intersection():
    rows = [_inject("x -> Tucson"), _inject("Wait, should be Tucson.", item="i1"),
            _inject("x", correct=False, item="i2"), _inject("x", mode="think", item="i1")]
    tab = analysis.sc_table(analysis.label_sc(pd.DataFrame(rows)), analysis.GROUP)
    no = tab[tab["mode"] == "nothink"].iloc[0]
    assert no["n"] == 3 and abs(no["explicit_rate"] + no["silent_rate"] - no["RR"]) < 1e-9
    assert abs(no["explicit_share"] - 0.5) < 1e-9
    df = pd.DataFrame(rows + [dict(_inject("x", mode="think", item="i9"), is_linear=True)])
    df["is_linear"] = True
    inter = analysis.intersection(df)
    assert set(inter["item_id"]) == {"i1"}
