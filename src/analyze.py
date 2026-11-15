"""Pre-registered analyses (docs/analysis_plan.md §2, §5).

Usage:
    python src/analyze.py --top 8000
Outputs:
    results/results_top<N>.json   all numbers reported in the paper
    results/km_rq2_top<N>.png      Kaplan-Meier figure
    results/cap_timeline_top<N>.png
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import median_survival_times
from scipy.stats import binomtest, fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
EVENT = pd.Timestamp(datetime(2024, 6, 16, tzinfo=timezone.utc))
OUT = ROOT / "results"


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def rq1(pk: pd.DataFrame) -> dict:
    res = {
        "n_numpy_dependent": int(len(pk)),
        "n_compiled": int(pk.compiled.sum()),
        "n_ever_capped": int(pk.ever_capped.sum()),
        "cap_origin": {k: int(v) for k, v in pk.cap_timing.value_counts().items()},
    }

    def fisher(df: pd.DataFrame) -> dict:
        tab = pd.crosstab(df.compiled, df.ever_capped).reindex(
            index=[True, False], columns=[True, False], fill_value=0)
        _, p = fisher_exact(tab.values, alternative="two-sided")
        t = Table2x2(tab.values.astype(float) + (0.5 if (tab.values == 0).any() else 0))
        lo, hi = t.oddsratio_confint()
        return {"table_[compiled T/F]x[capped T/F]": tab.values.tolist(),
                "odds_ratio": float(t.oddsratio), "or_95ci": [float(lo), float(hi)],
                "p": float(p),
                "cap_rate_compiled": float(df[df.compiled].ever_capped.mean()),
                "cap_rate_pure": float(df[~df.compiled].ever_capped.mean())}

    res["H1_fisher"] = fisher(pk)
    # Sensitivity: legacy caps are not responses to NumPy 2 (deviation D1).
    res["H1_fisher_excluding_legacy"] = fisher(pk[pk.cap_timing != "legacy"])
    return res


def rq2(pk: pd.DataFrame, collected_at: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    capped = pk[pk.ever_capped]
    before = capped[capped.adaptation == "before-release"]
    res = {
        "adaptation_counts": {k: int(v) for k, v in capped.adaptation.value_counts().items()},
        "before_release_days_median": (float(before.days_event_to_uncap.median())
                                       if len(before) else None),
    }
    # Survival cohort: capped packages not adapted by release day (D2).
    s = capped[capped.adaptation.isin(["after-release", "still-capped"])].copy()
    horizon = (collected_at - EVENT).days
    s["duration"] = s.days_event_to_uncap.fillna(horizon).clip(lower=0.5)
    s["observed"] = s.adaptation.eq("after-release").astype(int)
    s["log_downloads"] = np.log10(s.downloads_30d.astype(float))
    s["compiled_i"] = s.compiled.astype(int)
    res["survival_n"] = int(len(s))
    res["survival_events"] = int(s.observed.sum())

    kmf = KaplanMeierFitter().fit(s.duration, s.observed)
    ci = median_survival_times(kmf.confidence_interval_)
    res["km_median_days"] = float(kmf.median_survival_time_)
    res["km_median_95ci"] = [float(ci.iloc[0, 0]), float(ci.iloc[0, 1])]
    res["km_share_adapted_at"] = {
        f"{d}d": float(1 - kmf.survival_function_at_times(d).iloc[0]) for d in (30, 90, 180, 365)}

    covs = ["compiled_i", "log_downloads", "releases_2023", "age_years_at_event"]
    d = s[["duration", "observed"] + covs].dropna()
    res["cox_n"] = int(len(d))
    cph = CoxPHFitter().fit(d, "duration", "observed")
    sm = cph.summary
    holm = multipletests(sm["p"].values, method="holm")[1]
    res["H2_cox"] = {
        c: {"HR": float(sm.loc[c, "exp(coef)"]),
            "HR_95ci": [float(sm.loc[c, "exp(coef) lower 95%"]), float(sm.loc[c, "exp(coef) upper 95%"])],
            "p": float(sm.loc[c, "p"]), "p_holm": float(h)}
        for c, h in zip(sm.index, holm)}
    res["cox_concordance"] = float(cph.concordance_index_)
    ph = proportional_hazard_test(cph, d, time_transform="rank")
    res["ph_test_p"] = {c: float(p) for c, p in zip(ph.summary.index, ph.summary["p"])}

    # Sensitivity (D3): dropping the dependency counted as censored, not adapted.
    s2 = s.copy()
    s2.loc[s2.uncap_kind == "dropped-dependency", "observed"] = 0
    k2 = KaplanMeierFitter().fit(s2.duration, s2.observed)
    res["sensitivity_D3_km_median_days"] = float(k2.median_survival_time_)

    # Supplementary effect size: days to adapt, compiled vs pure (observed events only).
    ev = s[s.observed == 1]
    res["cliffs_delta_days_compiled_vs_pure"] = cliffs_delta(
        ev[ev.compiled].duration, ev[~ev.compiled].duration)

    # Figure: KM by compiled.
    fig, ax = plt.subplots(figsize=(6, 4))
    for flag, label in [(True, "Compiled extension"), (False, "Pure Python")]:
        g = s[s.compiled == flag]
        if len(g):
            KaplanMeierFitter().fit(g.duration, g.observed, label=f"{label} (n={len(g)})"
                                    ).plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel("Days since NumPy 2.0.0 release")
    ax.set_ylabel("Share still capped")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT / f"km_rq2_top{ARGS.top}{ARGS.variant}.png", dpi=200)
    plt.close(fig)
    return res, s


def rq3(pk: pd.DataFrame) -> dict:
    capped = pk[pk.ever_capped].set_index("package")
    rows = []
    for down, r in capped.iterrows():
        for up in str(r.capped_deps).split(";"):
            if up in capped.index and up != down:
                rows.append({"downstream": down, "upstream": up,
                             "down_uncap": r.uncap, "up_uncap": capped.loc[up, "uncap"]})
    pairs = pd.DataFrame(rows)
    res = {"n_pairs": int(len(pairs))}
    if pairs.empty:
        return res
    both = pairs.dropna(subset=["down_uncap", "up_uncap"]).copy()
    both["lag_days"] = (both.down_uncap - both.up_uncap).dt.total_seconds() / 86400
    nontie = both[both.lag_days.abs() >= 1]
    k = int((nontie.lag_days > 0).sum())
    n = int(len(nontie))
    res.update({
        "n_pairs_both_uncapped": int(len(both)),
        "n_ties_same_day": int(len(both) - n),
        "n_downstream_after_upstream": k,
        "share_after": k / n if n else None,
        "sign_test_p_one_sided": float(binomtest(k, n, 0.5, alternative="greater").pvalue) if n else None,
        "lag_days_median": float(both.lag_days.median()),
        "lag_days_iqr": [float(both.lag_days.quantile(.25)), float(both.lag_days.quantile(.75))],
        "pairs": both.assign(down_uncap=both.down_uncap.astype(str), up_uncap=both.up_uncap.astype(str)
                             ).to_dict("records"),
    })
    return res


def timeline_figure(pk: pd.DataFrame) -> None:
    c = pk[pk.ever_capped & pk.uncap.notna()].copy()
    c["start"] = c.first_cap.where(c.first_cap > EVENT - pd.Timedelta(days=365),
                                   EVENT - pd.Timedelta(days=365))
    c = c.sort_values("uncap")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.12 * len(c))))
    colors = {"legacy": "#8c8c8c", "pre-emptive": "#2b6cb0", "reactive": "#c05621"}
    for i, r in enumerate(c.itertuples()):
        ax.plot([r.start.to_pydatetime(), r.uncap.to_pydatetime()], [i, i],
                color=colors.get(r.cap_timing, "k"), lw=1.5)
    ax.axvline(EVENT.to_pydatetime(), color="k", ls="--", lw=1)
    ax.set_yticks([])
    ax.set_ylabel(f"Capped packages (n={len(c)}), sorted by lift date", fontsize=12)
    ax.set_xlabel("Capped period (dashed: NumPy 2.0.0 release)", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    for k, v in colors.items():
        ax.plot([], [], color=v, lw=3, label=k)
    ax.legend(loc="upper left", fontsize=13, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / f"cap_timeline_top{ARGS.top}{ARGS.variant}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pk = pd.read_csv(ROOT / "data" / f"packages_top{ARGS.top}{ARGS.variant}.csv",
                     parse_dates=["first_cap", "uncap", "first_release", "latest_release"])
    meta = json.loads((ROOT / "data" / f"meta_top{ARGS.top}.json").read_text(encoding="utf-8"))
    collected = pd.Timestamp(meta["collected_at"])
    results = {"collected_at": meta["collected_at"], "top_snapshot": meta["top_snapshot"],
               "RQ1": rq1(pk)}
    results["RQ2"], _ = rq2(pk, collected)
    results["RQ3"] = rq3(pk)
    timeline_figure(pk)
    path = OUT / f"results_top{ARGS.top}{ARGS.variant}.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    summary = {k: v for k, v in results.items() if k != "RQ3"}
    summary["RQ3"] = {k: v for k, v in results["RQ3"].items() if k != "pairs"}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8000)
    ap.add_argument("--variant", default="", help='"" (pre-registered) or "_mainline" (D8)')
    ARGS = ap.parse_args()
    main()
