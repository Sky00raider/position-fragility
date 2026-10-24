"""Analysis: recovery tables, PSI, matched-remaining contrasts, McNemar, logistic regression, figures,
explicit vs silent recovery (E1, exploratory) and the shared-item (intersection) robustness check.

    python -m src.analysis --runs runs/qwen3_8b_nothink runs/qwen3_8b_think [--out runs/analysis]

Definitions (CLAUDE.md):
  RR(cell)  = P(inject correct | clean twin at the same item and k correct)
  PSI       = RR_late - RR_early on intermediate hops (K=3: k2-k1; K=4: k3-k1). Terminal never pooled.
  Subsets   : "linear" (primary), "all" (robustness), each also "_notrunc" (truncated runs dropped).
  explicit  = recovered and a doubt marker is in the same or next sentence as the true hop-k value
              (gold hop answer or the model's own original value) or the planted value; else silent.
              Rule-based and strict (design doc 8.2): using the right value without a flag is silent.
  intersection = items eligible in every mode of a model (Qwen3-8B nothink and think).
CIs are item-clustered bootstrap percentiles (resample items with replacement).
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT
from .normalize import normalize
from .store import read_jsonl

GROUP = ["model", "mode"]
ERROR_TYPES = ["in_context", "off_context"]
EARLY_LATE = {3: (1, 2), 4: (1, 3)}            # intermediate early / late hop per K
MATCHED = [((3, 2), (4, 3), 1), ((3, 1), (4, 2), 2)]   # (K,k) pairs with equal remaining steps
DOUBT = re.compile(r"\b(wait|actually|mistake|incorrect|re-?check|should be|not correct|correction|however)\b",
                   re.IGNORECASE)


# ---------------------------------------------------------------- loading

def load_runs(run_dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        run_rows = read_jsonl(Path(d) / "scored.jsonl")
        elig = Path(d) / "eligibility.jsonl"
        if elig.exists():   # gold answer of the corrupted hop (gold_hop is 1-based)
            gold_hops = {e["item_id"]: e["gold_hops"] for e in read_jsonl(elig) if e.get("eligible")}
            for r in run_rows:
                hops, gh = gold_hops.get(r["item_id"]), r.get("gold_hop")
                r["gold_hop_answer"] = hops[gh - 1] if hops and gh else None
        rows += run_rows
    if not rows:
        sys.exit("no scored.jsonl rows found")
    df = pd.DataFrame(rows)
    df["model"] = df["model"].str.split("/").str[-1]
    df["correct"] = df["correct"].astype(bool)
    return df


def subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lin = df[df["is_linear"]]
    return {"linear": lin, "all": df,
            "linear_notrunc": lin[~lin["truncated"]], "all_notrunc": df[~df["truncated"]]}


def recovery_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Inject rows whose clean twin was correct (the RR denominator)."""
    return df[(df["condition"] == "inject") & (df["clean_twin_correct"] == True)]  # noqa: E712


# ---------------------------------------------------------------- bootstrap

def boot_ci(items: np.ndarray, stat, reps: int, rng: np.random.Generator, alpha=0.05):
    """Percentile CI of stat(sampled_item_ids) under item resampling."""
    vals = []
    for _ in range(reps):
        v = stat(rng.choice(items, size=len(items), replace=True))
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def rate_on(sample_ids, by_item: dict) -> float:
    """Mean outcome over a resampled list of item ids (items with no row are skipped)."""
    vals = [v for i in sample_ids for v in by_item.get(i, [])]
    return float(np.mean(vals)) if vals else np.nan


# ---------------------------------------------------------------- tables

def recovery_table(df, reps, rng) -> pd.DataFrame:
    out = []
    for keys, g in recovery_rows(df).groupby(GROUP + ["K", "k", "error_type"]):
        by_item = g.groupby("item_id")["correct"].apply(list).to_dict()
        lo, hi = boot_ci(np.array(list(by_item)), lambda s: rate_on(s, by_item), reps, rng)
        out.append({**dict(zip(GROUP + ["K", "k", "error_type"], keys)),
                    "is_terminal": bool(g["is_terminal"].iloc[0]), "remaining": g["remaining"].iloc[0],
                    "n": len(g), "RR": g["correct"].mean(), "ci_lo": lo, "ci_hi": hi,
                    "adopted": g["adopted"].fillna(False).astype(bool).mean(),
                    "RR_contains": g["correct_contains"].astype(bool).mean() if "correct_contains" in g else np.nan,
                    "adopted_contains": g["adopted_contains"].fillna(False).astype(bool).mean()
                    if "adopted_contains" in g else np.nan})
    return pd.DataFrame(out)


def clean_twin_table(df) -> pd.DataFrame:
    """Clean-twin accuracy per position (pilot gate: >= ~90%, else resumption is broken)."""
    c = df[df["condition"] == "clean"]
    return (c.groupby(GROUP + ["K", "k"])["correct"].agg(n="size", clean_acc="mean").reset_index())


def truncation_table(df) -> pd.DataFrame:
    d = df.copy()
    d["error_type"] = d["error_type"].fillna("-")
    return (d.groupby(GROUP + ["condition", "K", "k", "error_type"])["truncated"]
            .agg(n="size", truncation_rate="mean").reset_index())


def psi_table(df, reps, rng) -> pd.DataFrame:
    """PSI = RR_late - RR_early per model x mode x error type x K, with 95% and 90% CIs.
    Equivalence (no position effect) if the 90% CI lies inside +-5 pp."""
    rec, out = recovery_rows(df), []
    for keys, g in rec.groupby(GROUP + ["error_type", "K"]):
        K = keys[-1]
        if K not in EARLY_LATE:
            continue
        early, late = EARLY_LATE[K]
        e = g[g["k"] == early].groupby("item_id")["correct"].apply(list).to_dict()
        l_ = g[g["k"] == late].groupby("item_id")["correct"].apply(list).to_dict()
        if not e or not l_:
            continue
        ids = np.array(sorted(set(e) | set(l_)))
        stat = lambda s: rate_on(s, l_) - rate_on(s, e)  # noqa: E731
        lo, hi = boot_ci(ids, stat, reps, rng, 0.05)
        lo90, hi90 = boot_ci(ids, stat, reps, rng, 0.10)
        out.append({**dict(zip(GROUP + ["error_type", "K"], keys)),
                    "RR_early": rate_on(ids, e), "RR_late": rate_on(ids, l_), "n_early": len(e), "n_late": len(l_),
                    "PSI": rate_on(ids, l_) - rate_on(ids, e), "ci_lo": lo, "ci_hi": hi,
                    "ci90_lo": lo90, "ci90_hi": hi90,
                    "equivalent_5pp": bool(-0.05 < lo90 and hi90 < 0.05)})
    return pd.DataFrame(out)


def matched_remaining_table(df, reps, rng) -> pd.DataFrame:
    """Same steps remaining, different absolute position: (3,2)vs(4,3) and (3,1)vs(4,2).
    Different items on each side, so items are resampled within each K independently."""
    rec, out = recovery_rows(df[df["is_linear"]]), []
    for keys, g in rec.groupby(GROUP + ["error_type"]):
        for (K1, k1), (K2, k2), rem in MATCHED:
            a = g[(g["K"] == K1) & (g["k"] == k1)].groupby("item_id")["correct"].apply(list).to_dict()
            b = g[(g["K"] == K2) & (g["k"] == k2)].groupby("item_id")["correct"].apply(list).to_dict()
            if not a or not b:
                continue
            ia, ib = np.array(list(a)), np.array(list(b))
            diffs = [rate_on(rng.choice(ib, len(ib)), b) - rate_on(rng.choice(ia, len(ia)), a) for _ in range(reps)]
            out.append({**dict(zip(GROUP + ["error_type"], keys)), "remaining": rem,
                        "cell_a": f"K{K1}k{k1}", "cell_b": f"K{K2}k{k2}", "RR_a": rate_on(ia, a),
                        "RR_b": rate_on(ib, b), "n_a": len(a), "n_b": len(b),
                        "diff_b_minus_a": rate_on(ib, b) - rate_on(ia, a),
                        "ci_lo": np.quantile(diffs, 0.025), "ci_hi": np.quantile(diffs, 0.975)})
    return pd.DataFrame(out)


def terminal_table(df, reps, rng) -> pd.DataFrame:
    """RR at the terminal hop vs pooled intermediate hops (RQ3)."""
    rec, out = recovery_rows(df), []
    for keys, g in rec.groupby(GROUP + ["error_type"]):
        t = g[g["is_terminal"]].groupby("item_id")["correct"].apply(list).to_dict()
        m = g[~g["is_terminal"]].groupby("item_id")["correct"].apply(list).to_dict()
        if not t or not m:
            continue
        ids = np.array(sorted(set(t) | set(m)))
        lo, hi = boot_ci(ids, lambda s: rate_on(s, t) - rate_on(s, m), reps, rng)
        out.append({**dict(zip(GROUP + ["error_type"], keys)), "RR_terminal": rate_on(ids, t),
                    "RR_intermediate": rate_on(ids, m), "n_terminal": len(t), "n_intermediate": len(m),
                    "diff": rate_on(ids, t) - rate_on(ids, m), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)


def fixed_remaining_table(df, reps, rng) -> pd.DataFrame:
    """RQ2 with branching items: in 3hop2/4hop2/4hop3 some hops at different positions have the SAME
    number of dependent later hops (e.g. 3hop2: hops 1 and 2 both feed hop 3, remaining = 1), so
    position varies at fixed remaining. Within-item contrast RR(later k) - RR(earlier k), error types
    pooled, per (composition, remaining, k pair), plus one pooled row per model x mode."""
    rec = recovery_rows(df[~df["is_linear"] & ~df["is_terminal"]])
    out = []
    if rec.empty or "composition" not in rec:
        return pd.DataFrame(out)
    for keys, g in rec.groupby(GROUP):
        pool_e, pool_l = {}, {}
        for (comp, rem), h in g.groupby(["composition", "remaining"]):
            ks = sorted(h["k"].unique())
            for a, b in itertools.combinations(ks, 2):
                e = h[h["k"] == a].groupby("item_id")["correct"].apply(list).to_dict()
                l_ = h[h["k"] == b].groupby("item_id")["correct"].apply(list).to_dict()
                ids = np.array(sorted(set(e) & set(l_)))   # paired: same item at both positions
                if len(ids) == 0:
                    continue
                for i in ids:
                    pool_e.setdefault(i, []).extend(e[i])
                    pool_l.setdefault(i, []).extend(l_[i])
                lo, hi = boot_ci(ids, lambda s: rate_on(s, l_) - rate_on(s, e), reps, rng)
                out.append({**dict(zip(GROUP, keys)), "composition": comp, "remaining": rem,
                            "k_early": a, "k_late": b, "n_items": len(ids), "RR_early": rate_on(ids, e),
                            "RR_late": rate_on(ids, l_), "diff": rate_on(ids, l_) - rate_on(ids, e),
                            "ci_lo": lo, "ci_hi": hi})
        if pool_e:
            ids = np.array(sorted(pool_e))
            lo, hi = boot_ci(ids, lambda s: rate_on(s, pool_l) - rate_on(s, pool_e), reps, rng)
            out.append({**dict(zip(GROUP, keys)), "composition": "pooled", "remaining": np.nan,
                        "k_early": np.nan, "k_late": np.nan, "n_items": len(ids),
                        "RR_early": rate_on(ids, pool_e), "RR_late": rate_on(ids, pool_l),
                        "diff": rate_on(ids, pool_l) - rate_on(ids, pool_e), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)


def psi_pooled_terminal_table(df, reps, rng) -> pd.DataFrame:
    """H3: what PSI looks like if the terminal hop is (wrongly) counted as "late":
    PSI_pooled = RR(k in {K-1, K}) - RR(k=1), next to the intermediate-only PSI."""
    rec, out = recovery_rows(df), []
    for keys, g in rec.groupby(GROUP + ["error_type", "K"]):
        K = keys[-1]
        if K not in EARLY_LATE:
            continue
        early, late = EARLY_LATE[K]
        lists = lambda ks: g[g["k"].isin(ks)].groupby("item_id")["correct"].apply(list).to_dict()  # noqa: E731
        e, l_, lp = lists([early]), lists([late]), lists([late, K])
        ids = np.array(sorted(set(e) | set(lp)))
        if len(ids) == 0:
            continue
        lo, hi = boot_ci(ids, lambda s: rate_on(s, lp) - rate_on(s, e), reps, rng)
        out.append({**dict(zip(GROUP + ["error_type", "K"], keys)), "PSI": rate_on(ids, l_) - rate_on(ids, e),
                    "PSI_pooled_terminal": rate_on(ids, lp) - rate_on(ids, e), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)


def position_vs_remaining(df) -> pd.DataFrame:
    """Which predictor fits recovery better, per model x mode? Logistic models on intermediate inject
    cells of all items (branching items partly decouple position from remaining). Base = chain shape
    (composition, which also fixes K) + error_type + difficulty; then + rel_pos, + remaining, + both.
    Lower AIC = better fit. Composition must be in the base: branching items are easier overall, and
    without it that difference is wrongly credited to position (checked 2026-10-24)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = recovery_rows(df)
    d = d[~d["is_terminal"]].copy()
    d["y"] = d["correct"].astype(int)
    d["item_difficulty"] = d["item_difficulty"].astype(float)
    d["remaining"] = d["remaining"].astype(float)
    out = []
    if "composition" not in d:
        d["composition"] = "K" + d["K"].astype(str)   # synthetic test data: chain shape unknown
    for keys, g in d.groupby(GROUP):
        covs = [t for t in ["C(composition)", "error_type", "item_difficulty"] if g[t.strip("C()")].nunique() > 1]
        base = "y ~ " + (" + ".join(covs) if covs else "1")
        rows = []
        for name, extra in [("base", ""), ("+rel_pos", " + rel_pos"), ("+remaining", " + remaining"),
                            ("+both", " + rel_pos + remaining")]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.glm(base + extra, data=g, family=sm.families.Binomial()).fit()
            row = {**dict(zip(GROUP, keys)), "model_spec": name, "n": len(g), "AIC": fit.aic}
            for t in ("rel_pos", "remaining"):
                row[f"beta_{t}"] = fit.params.get(t, np.nan)
                row[f"p_{t}"] = fit.pvalues.get(t, np.nan)
            rows.append(row)
        best = min(r["AIC"] for r in rows)
        out += [{**r, "dAIC_vs_best": r["AIC"] - best} for r in rows]
    return pd.DataFrame(out)


def gold_hops_by_item() -> dict[str, list[str]]:
    """item_id -> gold hop answers (from the prepared items)."""
    from .store import read_jsonl
    out = {}
    for f in (REPO_ROOT / "data" / "items").glob("*.jsonl"):
        for it in read_jsonl(f):
            out[it["id"]] = [h["answer"] for h in it["hops"]]
    return out


def recovery_locus(df, gold_hops: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mechanism (exploratory): where does a non-thinking model get back on track after an error at hop k?
    Linear items, intermediate hops, clean twin correct. Reads the continuation's hop lines k+1..K:
      next_q   = does the hop k+1 sub-question carry the planted value, the true hop-k value, or neither
      on_track = first later hop j whose answer matches gold hop j (hop_answer_matches; aliases at j=K)
    Per-opportunity catch rate ("hazard") at the 1st and 2nd later hop compares early and late errors:
    if it is similar, late errors fail more because they have fewer later hops to recover on."""
    from .normalize import hop_answer_matches, normalize
    from .parse import parse_hops
    gold_hops = gold_hops if gold_hops is not None else gold_hops_by_item()
    rec = recovery_rows(df[df["is_linear"] & ~df["is_terminal"] & (df["mode"] == "nothink")])
    cells = []
    for r in rec.to_dict("records"):
        golds = gold_hops.get(r["item_id"])
        if not golds or not r.get("continuation"):
            continue
        K, k = int(r["K"]), int(r["k"])
        later = {h.n: h for h in parse_hops(r["continuation"]) if k < h.n <= K}
        nxt = later.get(k + 1)
        planted, true_vals = normalize(r["new_value"]), [normalize(r["orig_value"]), normalize(golds[k - 1])]
        if nxt is None:
            next_q = "missing"
        elif planted and planted in normalize(nxt.question):
            next_q = "planted"
        elif any(v and v in normalize(nxt.question) for v in true_vals):
            next_q = "true"
        else:
            next_q = "neither"
        on_track = None
        for j in range(k + 1, K + 1):
            extra = [r["gold"], *r.get("gold_aliases", [])] if j == K else []
            if j in later and hop_answer_matches(later[j].answer, [golds[j - 1], *extra]):
                on_track = j
                break
        cells.append({**{c: r[c] for c in GROUP + ["item_id", "K", "k", "error_type", "correct"]},
                      "next_q": next_q, "first_on_track": on_track,
                      "opportunity": (on_track - k) if on_track else None})
    cells = pd.DataFrame(cells)
    out = []
    if cells.empty:
        return cells, pd.DataFrame(out)
    for keys, g in cells.groupby(GROUP + ["K", "k"]):
        n = len(g)
        row = {**dict(zip(GROUP + ["K", "k"], keys)), "n": n, "RR": g["correct"].mean(),
               **{f"next_q_{v}": (g["next_q"] == v).mean() for v in ("planted", "true", "neither", "missing")}}
        at_risk = n
        for opp in range(1, int(keys[-2]) - int(keys[-1]) + 1):   # later hops k+1..K
            caught = int((g["opportunity"] == opp).sum())
            row[f"hazard_{opp}"] = caught / at_risk if at_risk else np.nan
            row[f"n_at_risk_{opp}"] = at_risk
            at_risk -= caught
        row["never_on_track_but_correct"] = (g["first_on_track"].isna() & g["correct"]).mean()
        out.append(row)
    return cells, pd.DataFrame(out)


def mcnemar_table(df) -> pd.DataFrame:
    """Exact McNemar, early vs late on the same item (both clean twins correct), Holm-adjusted."""
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.multitest import multipletests
    rec, out = recovery_rows(df), []
    for keys, g in rec.groupby(GROUP + ["error_type", "K"]):
        K = keys[-1]
        if K not in EARLY_LATE:
            continue
        early, late = EARLY_LATE[K]
        e = g[g["k"] == early].set_index("item_id")["correct"]
        l_ = g[g["k"] == late].set_index("item_id")["correct"]
        both = e.index.intersection(l_.index)
        if len(both) == 0:
            continue
        e, l_ = e[both], l_[both]
        table = [[int((e & l_).sum()), int((e & ~l_).sum())], [int((~e & l_).sum()), int((~e & ~l_).sum())]]
        p = mcnemar(table, exact=True).pvalue
        out.append({**dict(zip(GROUP + ["error_type", "K"], keys)), "n_pairs": len(both),
                    "both_correct": table[0][0], "early_only": table[0][1], "late_only": table[1][0],
                    "both_wrong": table[1][1], "discordant_frac": (table[0][1] + table[1][0]) / len(both),
                    "p_exact": p})
    res = pd.DataFrame(out)
    if len(res):
        res["p_holm"] = multipletests(res["p_exact"], method="holm")[1]
    return res


def logistic(df) -> tuple[str, pd.DataFrame]:
    """correct ~ rel_pos*mode + remaining + K + error_type + item_difficulty (+ model),
    intermediate inject cells with a correct clean twin, SEs clustered by item."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    d = recovery_rows(df)
    d = d[~d["is_terminal"]].copy()
    d["y"] = d["correct"].astype(int)
    d["K"] = d["K"].astype(float)
    d["item_difficulty"] = d["item_difficulty"].astype(float)
    terms = ["rel_pos", "remaining", "K", "error_type", "item_difficulty", "model", "mode"]
    terms = [t for t in terms if d[t].nunique() > 1]
    rhs = " + ".join(terms)
    if "mode" in terms:
        rhs += " + rel_pos:mode"
    if not terms or d["y"].nunique() < 2 or len(d) < 10:
        return f"logistic regression skipped: n={len(d)}, varying terms={terms}, outcome levels={d['y'].nunique()}", pd.DataFrame()
    formula = f"y ~ {rhs}"
    groups = pd.factorize(d["item_id"])[0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.glm(formula, data=d, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": groups})
    except Exception as e:  # tiny or separated data
        return f"logistic regression failed ({type(e).__name__}: {e}); formula: {formula}", pd.DataFrame()
    exog = pd.DataFrame(fit.model.exog, columns=fit.model.exog_names)
    vif = pd.DataFrame({"term": exog.columns,
                        "VIF": [variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])]})
    return f"formula: {formula}\nn={len(d)}, items={d['item_id'].nunique()}\n\n{fit.summary()}\n\nVIF:\n{vif.to_string(index=False)}", \
        fit.summary2().tables[1].reset_index().rename(columns={"index": "term"})


# ---------------------------------------------------------------- explicit vs silent recovery (E1)

def sentences(text: str) -> list[str]:
    """Split on line breaks and sentence-final punctuation (a hop line is one sentence)."""
    return [t.strip() for t in re.split(r"(?<=[.!?])\s+|\n+", text or "") if t.strip()]


def mentions(sentence: str, value) -> bool:
    v = normalize(value)
    return bool(v) and f" {v} " in f" {normalize(sentence)} "


def flag_in(text: str, values: list) -> bool:
    """A doubt marker in sentence i and one of the values in sentence i or i+1."""
    sents = sentences(text)
    return any(DOUBT.search(s) and any(mentions(t, v) for t in sents[i:i + 2] for v in values)
               for i, s in enumerate(sents))


def split_think(continuation: str, mode: str) -> tuple[str, str]:
    """(inside <think>, after </think>). Think-mode continuations start inside the think block."""
    c = continuation or ""
    if not str(mode).startswith("think"):       # "think" (hop-summary site) and "think_trace"
        return "", c
    i = c.find("</think>")
    return (c, "") if i < 0 else (c[:i], c[i + len("</think>"):])


def sc_label(row: dict) -> dict:
    """Rule-based self-correction label for one inject row: explicit / silent / not_recovered."""
    values = [row.get("gold_hop_answer"), row.get("orig_value"), row.get("new_value")]
    think, after = split_think(row.get("continuation"), row.get("mode"))
    in_think, in_after = flag_in(think, values), flag_in(after, values)
    label = ("explicit" if in_think or in_after else "silent") if row.get("correct") else "not_recovered"
    return {"doubt_marker": bool(DOUBT.search(think + "\n" + after)), "flag_think": in_think,
            "flag_after": in_after, "sc_label": label}


def label_sc(df) -> pd.DataFrame:
    """Recovery rows (clean twin correct) with sc_label, flag_think, flag_after and doubt_marker."""
    rec = recovery_rows(df)
    labels = pd.DataFrame([sc_label(r) for r in rec.to_dict("records")], index=rec.index)
    return rec.drop(columns=[c for c in labels if c in rec]).join(labels)


def sc_table(lab, keys) -> pd.DataFrame:
    """Rates over the RR denominator (explicit_rate + silent_rate = RR) and shares among recovered rows."""
    out = []
    for kv, g in lab.groupby(keys):
        r, f = g[g["correct"]], g[~g["correct"]]
        out.append({**dict(zip(keys, kv)), "n": len(g), "n_recovered": len(r), "RR": g["correct"].mean(),
                    "explicit_rate": (g["sc_label"] == "explicit").mean(),
                    "silent_rate": (g["sc_label"] == "silent").mean(),
                    "explicit_share": (r["sc_label"] == "explicit").mean() if len(r) else np.nan,
                    "explicit_in_think_share": r["flag_think"].mean() if len(r) else np.nan,
                    "explicit_after_think_share": r["flag_after"].mean() if len(r) else np.nan,
                    "n_not_recovered": len(f),
                    "doubt_rate_not_recovered": f["doubt_marker"].mean() if len(f) else np.nan,
                    "doubt_rate_recovered": r["doubt_marker"].mean() if len(r) else np.nan})
    return pd.DataFrame(out)


def sc_examples(lab, path: Path, n=10, seed=0):
    """n random rows per label, for hand checking."""
    lines = ["Explicit vs silent recovery: RULE-BASED labels (doubt marker in the same or next sentence as",
             "the true hop-k value or the planted value). Not validated: check by hand.",
             f"Doubt regex: {DOUBT.pattern}", ""]
    for label in ["explicit", "silent"]:
        g = lab[lab["sc_label"] == label]
        for _, r in g.sample(min(n, len(g)), random_state=seed).iterrows():
            after = "after </think>" if str(r["mode"]).startswith("think") else "in continuation"
            where = [w for w, f in [("inside <think>", r["flag_think"]), (after, r["flag_after"])] if f]
            lines += [f"===== {label.upper()}{' (' + ', '.join(where) + ')' if where else ''} | {r['model']} "
                      f"{r['mode']} | {r['item_id']} K={r['K']} k={r['k']} {r['error_type']}",
                      f"true hop value: {r.get('gold_hop_answer')!r} (model's own: {r['orig_value']!r}) | "
                      f"planted: {r['new_value']!r} | pred: {r['pred']!r}",
                      "--- prefix (last 3 lines)", "\n".join(str(r["prefix"]).rstrip().split("\n")[-3:]),
                      "--- continuation", str(r["continuation"])[:2500], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- shared items across modes

def intersection(df) -> pd.DataFrame:
    """Rows of models run in more than one mode, restricted to the items present in every mode."""
    keep = []
    for _, g in df.groupby("model"):
        items = g.groupby("mode")["item_id"].apply(set)
        if len(items) > 1:
            keep.append(g[g["item_id"].isin(set.intersection(*items))])
    return pd.concat(keep) if keep else df.iloc[0:0]


def intersection_n_table(df, inter) -> pd.DataFrame:
    """Eligible items per mode and shared items, per K (all and linear)."""
    out = []
    for (model, K), g in df[df["model"].isin(inter["model"].unique())].groupby(["model", "K"]):
        row = {"model": model, "K": K}
        for mode, m in g.groupby("mode"):
            row[f"n_{mode}"] = m["item_id"].nunique()
            row[f"n_{mode}_linear"] = m.loc[m["is_linear"], "item_id"].nunique()
        s = inter[(inter["model"] == model) & (inter["K"] == K)]
        row["n_intersection"] = s["item_id"].nunique()
        row["n_intersection_linear"] = s.loc[s["is_linear"], "item_id"].nunique()
        out.append(row)
    return pd.DataFrame(out)


MODE_ORDER = ["nothink", "think", "think_trace"]


def mode_diff_table(df, reps, rng) -> pd.DataFrame:
    """RR(mode_b) - RR(mode_a) on paired cells (item, k, error type) whose clean twin is correct in both
    modes, per cell and pooled over intermediate / terminal hops, for every pair of modes run.
    Pairs: nothink vs think, nothink vs think_trace, think vs think_trace (same weights throughout)."""
    rec, out = recovery_rows(df), []
    for model, g in rec.groupby("model"):
        modes = [m for m in MODE_ORDER if m in set(g["mode"])]
        for a, b in itertools.combinations(modes, 2):
            p = g[g["mode"].isin([a, b])].pivot_table(index=["item_id", "K", "k", "error_type", "is_terminal"],
                                                      columns="mode", values="correct", aggfunc="first")
            p = p.dropna().reset_index()
            if p.empty:
                continue
            p[a], p[b] = p[a].astype(float), p[b].astype(float)
            p["d"] = p[b] - p[a]
            cells = [(K, str(k), et, s) for (K, k, et), s in p.groupby(["K", "k", "error_type"])]
            for et in ERROR_TYPES + ["both"]:
                s = p if et == "both" else p[p["error_type"] == et]
                cells += [("all", "intermediate", et, s[~s["is_terminal"]]), ("all", "terminal", et, s[s["is_terminal"]])]
            for K, k, et, s in cells:
                if s.empty:
                    continue
                by_item = s.groupby("item_id")["d"].apply(list).to_dict()
                lo, hi = boot_ci(np.array(list(by_item)), lambda x: rate_on(x, by_item), reps, rng)
                out.append({"model": model, "mode_a": a, "mode_b": b, "K": K, "k": k, "error_type": et,
                            "n_cells": len(s), "n_items": len(by_item), "RR_a": s[a].mean(), "RR_b": s[b].mean(),
                            "diff_b_minus_a": s["d"].mean(), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- figures

COLORS = {"in_context": "#2a78d6", "off_context": "#eb6834"}    # validated palette slots 1-2
MARKERS = {"in_context": "o", "off_context": "s"}
INK, MUTED = "#0b0b0b", "#52514e"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_recovery(rec: pd.DataFrame, path: Path, title: str):
    import matplotlib.pyplot as plt
    groups = rec.groupby(GROUP)
    fig, axes = plt.subplots(len(groups), 2, figsize=(9, 3.2 * len(groups)), squeeze=False, sharey=True)
    for r, ((model, mode), g) in enumerate(groups):
        for c, K in enumerate([3, 4]):
            ax = axes[r][c]
            _style(ax)
            for et in ERROR_TYPES:
                s = g[(g["K"] == K) & (g["error_type"] == et)].sort_values("k")
                if s.empty:
                    continue
                inter, term = s[~s["is_terminal"]], s[s["is_terminal"]]
                off = -0.06 if et == "in_context" else 0.06
                ax.errorbar(inter["k"] + off, inter["RR"], yerr=[inter["RR"] - inter["ci_lo"], inter["ci_hi"] - inter["RR"]],
                            color=COLORS[et], marker=MARKERS[et], ms=7, lw=2, capsize=0, elinewidth=1.2, label=et)
                # terminal hop: hollow marker, not connected to the intermediate line
                ax.errorbar(term["k"] + off, term["RR"], yerr=[term["RR"] - term["ci_lo"], term["ci_hi"] - term["RR"]],
                            color=COLORS[et], marker=MARKERS[et], ms=7, mfc="white", mew=2, lw=0, elinewidth=1.2)
            ax.axvline(K - 0.5, color="#c9c8c3", lw=1, ls=":")
            ax.set_xticks(range(1, K + 1), [f"{k}" + ("\n(terminal)" if k == K else "") for k in range(1, K + 1)])
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{model} · {mode} · K={K}", color=INK, fontsize=10, loc="left")
            ax.set_xlabel("corrupted hop k", color=MUTED)
            if c == 0:
                ax.set_ylabel("recovery rate", color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=2)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_forest(tab: pd.DataFrame, value: str, path: Path, title: str, label_cols: list[str], xlabel: str):
    import matplotlib.pyplot as plt
    if tab.empty:
        return
    t = tab.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(t) + 1.2))
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.8)
    y = np.arange(len(t))[::-1]
    for yi, (_, r) in zip(y, t.iterrows()):
        col = COLORS.get(r.get("error_type"), INK)
        ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(r[value], yi, marker=MARKERS.get(r.get("error_type"), "o"), color=col, ms=8)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, [" · ".join(str(r[c]) for c in label_cols) for _, r in t.iterrows()], fontsize=8)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


MODE_COLORS = {"nothink": "#2a78d6", "think": "#eb6834", "think_trace": "#1baf7a"}   # validated slots 1-3
MODE_MARKERS = {"nothink": "o", "think": "s", "think_trace": "^"}


def fig_mode_compare(rec: pd.DataFrame, path: Path, title: str):
    """Recovery by k, nothink vs think, one panel per K (rec = recovery_table with error types pooled)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, K in zip(axes, [3, 4]):
        _style(ax)
        modes = [m for m in MODE_ORDER if m in set(rec["mode"])]
        for i, mode in enumerate(modes):
            s = rec[(rec["K"] == K) & (rec["mode"] == mode)].sort_values("k")
            if s.empty:
                continue
            inter, term = s[~s["is_terminal"]], s[s["is_terminal"]]
            off = 0.08 * (i - (len(modes) - 1) / 2)
            ax.errorbar(inter["k"] + off, inter["RR"], yerr=[inter["RR"] - inter["ci_lo"], inter["ci_hi"] - inter["RR"]],
                        color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], ms=7, lw=2, capsize=0, elinewidth=1.2,
                        label=mode)
            # terminal hop: hollow marker, not connected to the intermediate line
            ax.errorbar(term["k"] + off, term["RR"], yerr=[term["RR"] - term["ci_lo"], term["ci_hi"] - term["RR"]],
                        color=MODE_COLORS[mode], marker=MODE_MARKERS[mode], ms=7, mfc="white", mew=2, lw=0,
                        elinewidth=1.2)
        ax.axvline(K - 0.5, color="#c9c8c3", lw=1, ls=":")
        ax.set_xticks(range(1, K + 1), [f"{k}" + ("\n(terminal)" if k == K else "") for k in range(1, K + 1)])
        ax.set_ylim(0, 1.02)
        n = rec[rec["K"] == K].groupby("mode")["n"].sum()
        ax.set_title(f"K={K} · cells: " + ", ".join(f"{m} {v}" for m, v in n.items()), color=INK, fontsize=10,
                     loc="left")
        ax.set_xlabel("corrupted hop k", color=MUTED)
    axes[0].set_ylabel("recovery rate", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", ncol=3)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "analysis"))
    ap.add_argument("--reps", type=int, default=2000, help="bootstrap reps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")

    df = load_runs(args.runs)
    out, tabs, figs = Path(args.out), Path(args.out) / "tables", Path(args.out) / "figures"
    tabs.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    banner = []
    if df.get("eligibility_relaxed", pd.Series([False])).any():
        banner.append("!! eligibility_relaxed rows present: DEBUG RUN, numbers are not results")
    if (df.get("chain_source", pd.Series(["model"])) == "gold_debug").any():
        banner.append("!! gold_debug stand-in chains present: DEBUG RUN, numbers are not results")
    for b in banner:
        print(b)

    report = list(banner)
    report.append(f"rows: {len(df)}; runs: {', '.join(sorted(df['run_id'].unique()))}")
    for name, d in subsets(df).items():
        rec = recovery_table(d, args.reps, rng)
        rec.to_csv(tabs / f"recovery_{name}.csv", index=False)
        psi = psi_table(d, args.reps, rng)
        psi.to_csv(tabs / f"psi_{name}.csv", index=False)
        if name in ("linear", "all"):
            matched_remaining_table(d, args.reps, rng).to_csv(tabs / f"matched_remaining_{name}.csv", index=False)
            terminal_table(d, args.reps, rng).to_csv(tabs / f"terminal_{name}.csv", index=False)
            mcn = mcnemar_table(d)
            mcn.to_csv(tabs / f"mcnemar_{name}.csv", index=False)
            clean_twin_table(d).to_csv(tabs / f"clean_twin_{name}.csv", index=False)
            text, coefs = logistic(d)
            coefs.to_csv(tabs / f"logit_{name}.csv", index=False)
            (out / f"logit_{name}.txt").write_text("\n".join(banner + [text]), encoding="utf-8")
            if not rec.empty:
                fig_recovery(rec, figs / f"recovery_{name}.png",
                             f"Recovery by corrupted hop ({name} items; terminal hop hollow)")
            fig_forest(psi, "PSI", figs / f"psi_{name}.png", f"PSI = RR_late − RR_early ({name} items)",
                       GROUP + ["error_type", "K"], "PSI (95% item-bootstrap CI)")
            pooled_t = psi_pooled_terminal_table(d, args.reps, rng)
            pooled_t.to_csv(tabs / f"psi_pooled_terminal_{name}.csv", index=False)
            report += [f"\n== {name} ==", "PSI:", psi.round(3).to_string(index=False) if len(psi) else "(none)",
                       "McNemar:", mcn.round(4).to_string(index=False) if len(mcn) else "(none)",
                       text.split("\n\n")[0],
                       "PSI if the terminal hop were pooled with late (H3):",
                       pooled_t.round(3).to_string(index=False) if len(pooled_t) else "(none)"]
            if name == "all":   # RQ2 with branching items, where position and remaining come apart
                fixed = fixed_remaining_table(d, args.reps, rng)
                fixed.to_csv(tabs / "fixed_remaining_branching.csv", index=False)
                fits = position_vs_remaining(d)
                fits.to_csv(tabs / "position_vs_remaining_fit.csv", index=False)
                report += ["\nPosition at fixed remaining (branching items, within item, error types pooled):",
                           fixed.round(3).to_string(index=False) if len(fixed) else "(none)",
                           "Position vs remaining, model fit (all items, intermediate):",
                           fits.round(4).to_string(index=False)]
    truncation_table(df).to_csv(tabs / "truncation.csv", index=False)

    # E1 (exploratory): explicit vs silent recovery, rule-based labels
    if "continuation" in df:
        for name in ("linear", "all"):
            lab = label_sc(subsets(df)[name])
            sc_table(lab, GROUP + ["K", "k", "error_type"]).to_csv(tabs / f"silent_explicit_{name}.csv", index=False)
            summ = sc_table(lab, GROUP + ["is_terminal"])
            summ.to_csv(tabs / f"silent_explicit_summary_{name}.csv", index=False)
            report += [f"\n== explicit vs silent recovery ({name} items, rule-based) ==",
                       summ.round(3).to_string(index=False)]
            if name == "all":
                sc_examples(lab, out / "silent_explicit_examples.txt")
        # mechanism (exploratory): at which later hop non-thinking models get back on track
        cells, locus = recovery_locus(df)
        if not locus.empty:
            cells.to_csv(tabs / "recovery_locus_cells_linear.csv", index=False)
            locus.to_csv(tabs / "recovery_locus_linear.csv", index=False)
            report += ["\n== where recovery happens (non-thinking, linear, intermediate) ==",
                       locus.round(3).to_string(index=False)]

    # robustness: compare modes on the items eligible in both
    inter = intersection(df)
    if not inter.empty:
        n_tab = intersection_n_table(df, inter)
        n_tab.to_csv(tabs / "n_items_intersection.csv", index=False)
        report += ["\n== intersection: items eligible in every mode ==", n_tab.to_string(index=False)]
        for name in ("linear", "all"):
            d = subsets(inter)[name]
            recovery_table(d, args.reps, rng).to_csv(tabs / f"recovery_{name}_intersection.csv", index=False)
            psi = psi_table(d, args.reps, rng)
            psi.to_csv(tabs / f"psi_{name}_intersection.csv", index=False)
            term = terminal_table(d, args.reps, rng)
            term.to_csv(tabs / f"terminal_{name}_intersection.csv", index=False)
            md = mode_diff_table(d, args.reps, rng)
            md.to_csv(tabs / f"mode_diff_{name}_intersection.csv", index=False)
            text, coefs = logistic(d)
            coefs.to_csv(tabs / f"logit_{name}_intersection.csv", index=False)
            (out / f"logit_{name}_intersection.txt").write_text("\n".join(banner + [text]), encoding="utf-8")
            pooled = recovery_table(d.assign(error_type="both"), args.reps, rng)
            pooled.to_csv(tabs / f"recovery_pooled_{name}_intersection.csv", index=False)
            if not pooled.empty:
                fig_mode_compare(pooled, figs / f"mode_compare_{name}_intersection.png",
                                 f"Shared items: recovery by corrupted hop ({name}; error types pooled)")
            report += [f"\n== {name}_intersection ==", "PSI:", psi.round(3).to_string(index=False) if len(psi) else "(none)",
                       "terminal vs intermediate:", term.round(3).to_string(index=False) if len(term) else "(none)",
                       "mode difference (paired cells):", md.round(3).to_string(index=False) if len(md) else "(none)",
                       text.split("\n\n")[0]]
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[len(banner):]))
    print(f"\ntables -> {tabs}\nfigures -> {figs}")


if __name__ == "__main__":
    main()
