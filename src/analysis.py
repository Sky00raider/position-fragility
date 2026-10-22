"""Analysis: recovery tables, PSI, matched-remaining contrasts, McNemar, logistic regression, figures.

    python -m src.analysis --runs runs/qwen3_8b_nothink runs/qwen3_8b_think [--out runs/analysis]

Definitions (CLAUDE.md):
  RR(cell)  = P(inject correct | clean twin at the same item and k correct)
  PSI       = RR_late - RR_early on intermediate hops (K=3: k2-k1; K=4: k3-k1). Terminal never pooled.
  Subsets   : "linear" (primary), "all" (robustness), each also "_notrunc" (truncated runs dropped).
CIs are item-clustered bootstrap percentiles (resample items with replacement).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT
from .store import read_jsonl

GROUP = ["model", "mode"]
ERROR_TYPES = ["in_context", "off_context"]
EARLY_LATE = {3: (1, 2), 4: (1, 3)}            # intermediate early / late hop per K
MATCHED = [((3, 2), (4, 3), 1), ((3, 1), (4, 2), 2)]   # (K,k) pairs with equal remaining steps


# ---------------------------------------------------------------- loading

def load_runs(run_dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in run_dirs:
        rows += read_jsonl(Path(d) / "scored.jsonl")
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
            report += [f"\n== {name} ==", "PSI:", psi.round(3).to_string(index=False) if len(psi) else "(none)",
                       "McNemar:", mcn.round(4).to_string(index=False) if len(mcn) else "(none)",
                       text.split("\n\n")[0]]
    truncation_table(df).to_csv(tabs / "truncation.csv", index=False)
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[len(banner):]))
    print(f"\ntables -> {tabs}\nfigures -> {figs}")


if __name__ == "__main__":
    main()
