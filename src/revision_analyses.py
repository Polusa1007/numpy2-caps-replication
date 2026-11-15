"""Post-hoc analyses (all EXPLORATORY; mainline data by default).

R1  Early adaptation by code type: Fisher test on "lifted before release" among
    capped packages; Cox model with time origin at NumPy 2.0.0rc1 (2024-03-30).
R2  Delayed entry: release-origin survival with left truncation at the cap date;
    cap-duration (cap date -> lift) for caps whose start is observed;
    ecosystem-wide prevalence of capped latest releases over time (all 746).
R3  Equivalence-style reporting: 90% CIs vs. margin [0.5, 2]; RMST difference
    and log-rank instead of Cliff's delta; releases scaled per 10.
R4  Objective layers: in-sample reverse-dependency in-degree (core = >=5
    NumPy-dependent dependents) and download quartiles.
R5  Build-tool caps: share of Poetry-built first capped wheels; sensitivity.
R6  RQ3 clustering: downstream-level test; stratified permutation (upstream
    layer x download quartile); time-varying Cox with "upstream still capped".
R7  Broad cap definition: constraints that admit some 1.x but no 2.x
    (e.g. numpy<1.25), which the pre-registered rule labels "other".

Usage:
    python src/revision_analyses.py --top 8000 [--variant _mainline]
Outputs:
    results/revision_top<N><variant>.json, results/prevalence_top<N><variant>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times, restricted_mean_survival_time
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from scipy.stats import binomtest, fisher_exact, wilcoxon
from statsmodels.stats.contingency_tables import Table2x2

ROOT = Path(__file__).resolve().parent.parent
EVENT = pd.Timestamp("2024-06-16", tz="UTC")
RC1 = pd.Timestamp("2024-03-30", tz="UTC")       # GitHub release v2.0.0rc1
CORE_MIN_INDEGREE = 5                             # fixed before inspecting layer results
SEED, DRAWS = 0, 5000


def days(ts, origin):
    return (ts - origin).dt.total_seconds() / 86400


def km_summary(dur, ev, entry=None) -> dict:
    k = KaplanMeierFitter().fit(dur, ev, entry=entry)
    ci = median_survival_times(k.confidence_interval_)
    return {"n": int(len(dur)), "events": int(np.sum(ev)),
            "median": float(k.median_survival_time_),
            "median_95ci": [float(ci.iloc[0, 0]), float(ci.iloc[0, 1])],
            "adapted_at": {f"{d}d": float(1 - k.survival_function_at_times(d).iloc[0])
                           for d in (30, 90, 180, 365)}}


def cox(df: pd.DataFrame, covs: list[str], entry: str | None = None) -> dict:
    d = df[["duration", "observed"] + covs + ([entry] if entry else [])].dropna()
    m = CoxPHFitter().fit(d, "duration", "observed", entry_col=entry)
    s = m.summary
    ci90 = m.confidence_intervals_.copy()
    z = 1.6448536269514722
    return {"n": int(len(d)), "events": int(d.observed.sum()), "concordance": float(m.concordance_index_),
            "terms": {c: {"HR": float(s.loc[c, "exp(coef)"]),
                          "HR_95ci": [float(s.loc[c, "exp(coef) lower 95%"]), float(s.loc[c, "exp(coef) upper 95%"])],
                          "HR_90ci": [float(np.exp(s.loc[c, "coef"] - z * s.loc[c, "se(coef)"])),
                                      float(np.exp(s.loc[c, "coef"] + z * s.loc[c, "se(coef)"]))],
                          "p": float(s.loc[c, "p"])} for c in s.index}}


def admits_some(spec: str, versions) -> bool:
    try:
        ss = SpecifierSet(spec)
    except Exception:
        return False
    return any(ss.contains(v, prereleases=True) for v in versions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8000)
    ap.add_argument("--variant", default="_mainline")
    a = ap.parse_args()
    tag = f"top{a.top}{a.variant}"
    meta = json.loads((ROOT / "data" / f"meta_top{a.top}.json").read_text(encoding="utf-8"))
    horizon_ts = pd.Timestamp(meta["collected_at"])
    pk = pd.read_csv(ROOT / "data" / f"packages_{tag}.csv")
    for c in ("first_cap", "uncap", "first_release"):
        pk[c] = pd.to_datetime(pk[c], utc=True, format="mixed")
    rel = pd.read_csv(ROOT / "data" / f"releases_top{a.top}.csv")
    rel["uploaded"] = pd.to_datetime(rel.uploaded, utc=True, format="mixed")
    use = rel[~rel.prerelease & ~rel.yanked & ~rel.missing_requires_dist]
    if a.variant == "_mainline":
        use = use[use.frontier]
    gen = pd.read_csv(ROOT / "data" / f"generator_{tag}.csv")[["package", "tool"]]
    pk = pk.merge(gen, on="package", how="left")

    pk["log_downloads"] = np.log10(pk.downloads_30d.astype(float))
    pk["releases_2023_per10"] = pk.releases_2023 / 10
    pk["compiled_i"] = pk.compiled.astype(int)
    pk["dl_quartile"] = pd.qcut(pk.downloads_30d.rank(method="first"), 4, labels=["Q1 (least)", "Q2", "Q3", "Q4 (most)"])

    # ---- R4: layers from in-sample reverse dependencies -------------------
    dep_set = set(pk.package)
    indeg: dict[str, set] = {p: set() for p in dep_set}
    for p, g in use[use.package.isin(dep_set)].groupby("package"):
        for s in g.deps.dropna():
            for d in str(s).split(";"):
                if d in dep_set and d != p:
                    indeg[d].add(p)
    pk["indegree"] = pk.package.map(lambda p: len(indeg[p]))
    pk["layer"] = np.where(pk.indegree >= CORE_MIN_INDEGREE, "core", "periphery")
    covs = ["compiled_i", "log_downloads", "releases_2023_per10", "age_years_at_event"]
    out: dict = {"variant": a.variant or "pre-registered", "core_min_indegree": CORE_MIN_INDEGREE}

    capped = pk[pk.ever_capped].copy()
    capped["early"] = capped.adaptation.eq("before-release")

    # ---- R1: early adaptation by code type ---------------------------------
    tab = pd.crosstab(capped.compiled, capped.early).reindex(index=[True, False], columns=[True, False], fill_value=0)
    t2 = Table2x2(tab.values.astype(float))
    out["R1_early_by_compiled"] = {
        "table_[compiled T/F]x[early T/F]": tab.values.tolist(),
        "early_rate_compiled": float(capped[capped.compiled].early.mean()),
        "early_rate_pure": float(capped[~capped.compiled].early.mean()),
        "OR": float(t2.oddsratio), "OR_95ci": [float(x) for x in t2.oddsratio_confint()],
        "fisher_p": float(fisher_exact(tab.values)[1]),
        "early_packages": sorted(capped[capped.early].package),
    }
    # Cox with origin at RC1; caps lifted before RC1 get duration 0.5 d.
    s = capped.copy()
    horizon_rc = (horizon_ts - RC1).days
    s["duration"] = days(s.uncap, RC1).fillna(horizon_rc).clip(lower=0.5)
    s["observed"] = s.uncap.notna().astype(int)
    s["entry"] = days(s.first_cap, RC1).clip(lower=0)
    s.loc[s.entry >= s.duration, "entry"] = 0          # cap and lift both before RC1
    out["R1_rc1_origin"] = {"n_lifted_before_rc1": int((days(s.uncap, RC1) <= 0).sum()),
                            "km": km_summary(s.duration, s.observed, s.entry),
                            "cox": cox(s, covs, "entry")}

    # ---- R2: delayed entry from release day; cap duration; prevalence ------
    r = capped[capped.adaptation.isin(["after-release", "still-capped"])].copy()
    horizon = (horizon_ts - EVENT).days
    r["duration"] = days(r.uncap, EVENT).fillna(horizon).clip(lower=0.5)
    r["observed"] = r.uncap.notna().astype(int)
    r["entry"] = days(r.first_cap, EVENT).clip(lower=0)
    r["entry"] = np.minimum(r.entry, r.duration - 0.25)
    out["R2_release_origin_no_entry"] = km_summary(r.duration, r.observed)
    out["R2_release_origin_delayed_entry"] = {"km": km_summary(r.duration, r.observed, r.entry),
                                              "cox": cox(r, covs, "entry")}
    fig, ax = plt.subplots(figsize=(6, 4))
    for flag, label in [(True, "Compiled extension"), (False, "Pure Python")]:
        g = r[r.compiled == flag]
        KaplanMeierFitter().fit(g.duration, g.observed, entry=g.entry, label=f"{label} (n={len(g)})"
                                ).plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel("Days since NumPy 2.0.0 release")
    ax.set_ylabel("Share of packages at risk still capped")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"km_delayed_entry_{tag}.png", dpi=200)
    plt.close(fig)
    c = capped[capped.cap_timing.isin(["pre-emptive", "reactive"])].copy()
    c["duration"] = days(c.uncap.fillna(horizon_ts), c.first_cap).clip(lower=0.5)
    c["observed"] = c.uncap.notna().astype(int)
    out["R2_cap_duration"] = {k: km_summary(g.duration, g.observed) for k, g in c.groupby("cap_timing")}
    out["R2_cap_duration"]["all_observed_start"] = km_summary(c.duration, c.observed)

    # prevalence: share of NumPy-dependent packages whose latest release is capped
    grid = np.arange(-365, horizon + 1, 7)
    seqs = {p: g.sort_values("uploaded")[["uploaded", "numpy_status"]].values for p, g in use[use.package.isin(dep_set)].groupby("package")}
    prev_rows = []
    for d in grid:
        t = EVENT + pd.Timedelta(days=int(d))
        st = []
        for p, arr in seqs.items():
            past = [x for x in arr if x[0] <= t]
            if past:
                st.append(past[-1][1])
        st = pd.Series(st)
        dep_now = st.isin(["capped", "compatible", "other"])
        prev_rows.append({"day": int(d), "n_dependent": int(dep_now.sum()),
                          "share_capped": float((st[dep_now] == "capped").mean())})
    prev = pd.DataFrame(prev_rows)
    out["R2_prevalence"] = {f"day_{d}": prev.loc[(prev.day - d).abs().idxmin()].to_dict()
                            for d in (-90, 0, 30, 90, 180, 365, 730)}
    after = prev[prev.day >= -180]
    out["R2_prevalence"]["peak_from_day_-180"] = after.loc[after.share_capped.idxmax()].to_dict()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    shown = prev[prev.n_dependent >= 200]   # early window: few packages have an observed release yet
    ax.plot(shown.day, shown.share_capped * 100, color="#2b6cb0")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline((RC1 - EVENT).days, color="grey", ls=":", lw=1)
    ax.set_xlabel("Days since NumPy 2.0.0 release (dotted: rc1)")
    ax.set_ylabel("% of NumPy-dependent packages\nwhose latest release is capped")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / f"prevalence_{tag}.png", dpi=200)
    plt.close(fig)

    # ---- R3: effect sizes for code type -------------------------------------
    rm = {}
    for flag in (True, False):
        g = r[r.compiled == flag]
        rm[flag] = restricted_mean_survival_time(KaplanMeierFitter().fit(g.duration, g.observed), t=365)
    lr = logrank_test(r[r.compiled].duration, r[~r.compiled].duration, r[r.compiled].observed, r[~r.compiled].observed)
    h1 = pd.crosstab(pk.compiled, pk.ever_capped).reindex(index=[True, False], columns=[True, False], fill_value=0)
    t1 = Table2x2(h1.values.astype(float))
    out["R3_code_type"] = {
        "H1_OR": float(t1.oddsratio), "H1_OR_95ci": [float(x) for x in t1.oddsratio_confint(alpha=0.05)],
        "H1_OR_90ci": [float(x) for x in t1.oddsratio_confint(alpha=0.10)],
        "equivalence_margin": [0.5, 2.0],
        "RMST365_compiled": float(rm[True]), "RMST365_pure": float(rm[False]),
        "RMST365_diff_compiled_minus_pure": float(rm[True] - rm[False]),
        "logrank_p": float(lr.p_value),
    }

    # ---- R4: layer and popularity strata ------------------------------------
    def strata_summary(col: str) -> dict:
        res = {}
        for k, g in pk.groupby(col, observed=True):
            cg = g[g.ever_capped]
            rr = r[r.package.isin(g.package)]
            res[str(k)] = {
                "n_dependent": int(len(g)), "n_capped": int(len(cg)),
                "cap_rate": float(len(cg) / len(g)),
                "share_of_caps_reactive": float((cg.cap_timing == "reactive").mean()) if len(cg) else None,
                "share_of_caps_lifted_before_release": float((cg.adaptation == "before-release").mean()) if len(cg) else None,
                "km_delayed_entry": km_summary(rr.duration, rr.observed, rr.entry) if len(rr) >= 5 else None,
            }
        return res
    out["R4_layers"] = {"definition": f"core = >= {CORE_MIN_INDEGREE} NumPy-dependent packages in the sample depend on it",
                        "core_packages": sorted(pk[pk.layer == "core"].package),
                        "by_layer": strata_summary("layer"),
                        "by_download_quartile": strata_summary("dl_quartile")}
    r["core_i"] = (r.layer == "core").astype(int)
    out["R4_cox_with_layer"] = cox(r, covs + ["core_i"], "entry")

    # ---- R5: build tool -----------------------------------------------------
    out["R5_build_tool"] = {
        "tool_counts": capped.tool.value_counts().to_dict(),
        "poetry_by_cap_timing": {k: int(v) for k, v in capped[capped.tool == "poetry"].cap_timing.value_counts().items()},
        "cap_origin_without_poetry": {k: int(v) for k, v in capped[capped.tool != "poetry"].cap_timing.value_counts().items()},
    }
    np_ = r[r.tool != "poetry"]
    out["R5_build_tool"]["delayed_entry_without_poetry"] = {
        "km": km_summary(np_.duration, np_.observed, np_.entry), "cox": cox(np_, covs, "entry")}

    # ---- R6: RQ3 clustering, stratified permutation, time-varying Cox -------
    capset = capped.set_index("package")
    pairs = []
    for down, row in capset.iterrows():
        for up in str(row.capped_deps).split(";"):
            if up in capset.index and up != down:
                pairs.append({"down": down, "up": up, "d": row.uncap, "u": capset.loc[up, "uncap"]})
    P = pd.DataFrame(pairs).dropna(subset=["d", "u"])
    P["lag"] = (P.d - P.u).dt.total_seconds() / 86400
    P = P[P.lag.abs() >= 1]
    per_down = P.groupby("down").lag.apply(lambda x: (x > 0).mean())
    maj = per_down[per_down != 0.5]
    out["R6_rq3_clustered"] = {
        "n_pairs": int(len(P)), "n_downstreams": int(len(per_down)),
        "pairs_per_downstream_median": float(P.groupby("down").size().median()),
        "downstreams_all_followed": int((per_down == 1).sum()),
        "downstreams_majority_followed": int((maj > 0.5).sum()), "downstreams_majority_not": int((maj < 0.5).sum()),
        "sign_test_downstream_level_p": float(binomtest(int((maj > 0.5).sum()), int(len(maj)), 0.5, alternative="greater").pvalue),
        "wilcoxon_share_minus_half_p": float(wilcoxon(per_down - 0.5, alternative="greater").pvalue),
        "mean_downstream_share": float(per_down.mean()),
    }
    lifted = capped[capped.uncap.notna()].copy()
    lifted["stratum"] = lifted.layer + "|" + lifted.dl_quartile.astype(str)
    strat = lifted.set_index("package").stratum
    pool = {k: g.uncap.values for k, g in lifted.groupby("stratum")}
    P["stratum"] = P.up.map(strat)
    rng = np.random.default_rng(SEED)
    dvals = P.d.dt.tz_convert(None).values.astype("datetime64[ns]")
    null = np.empty(DRAWS)
    for i in range(DRAWS):
        u = np.array([rng.choice(pool[s]) for s in P.stratum]).astype("datetime64[ns]")
        lag = (dvals - u) / np.timedelta64(1, "D")
        m = np.abs(lag) >= 1
        tmp = pd.DataFrame({"down": P.down.values[m], "after": lag[m] > 0})
        null[i] = tmp.groupby("down").after.mean().mean()
    out["R6_rq3_stratified_permutation"] = {
        "statistic": "mean over downstreams of share of upstreams followed",
        "strata": "upstream layer x download quartile",
        "observed": float(per_down.mean()), "null_mean": float(null.mean()),
        "null_95_range": [float(np.quantile(null, .025)), float(np.quantile(null, .975))],
        "p_one_sided": float(((null >= per_down.mean()).sum() + 1) / (DRAWS + 1)), "draws": DRAWS}
    # time-varying Cox: blocked(t) = some capped upstream not yet lifted at t
    rows = []
    for x in r.itertuples():
        ups = [u for u in str(x.capped_deps).split(";") if u in capset.index and u != x.package]
        u_times = [capset.loc[u, "uncap"] for u in ups]
        clear = (0.0 if not ups else
                 (np.inf if any(pd.isna(t) for t in u_times) else max(0.0, max((t - EVENT).total_seconds() / 86400 for t in u_times))))
        base = {"id": x.package, "compiled_i": x.compiled_i, "log_downloads": x.log_downloads,
                "releases_2023_per10": x.releases_2023_per10, "age_years_at_event": x.age_years_at_event}
        start, stop = float(x.entry), float(x.duration)
        if clear <= start:
            rows.append({**base, "start": start, "stop": stop, "blocked": 0, "event": x.observed})
        elif clear >= stop:
            rows.append({**base, "start": start, "stop": stop, "blocked": 1, "event": x.observed})
        else:
            rows.append({**base, "start": start, "stop": clear, "blocked": 1, "event": 0})
            rows.append({**base, "start": clear, "stop": stop, "blocked": 0, "event": x.observed})
    tv = pd.DataFrame(rows).dropna()
    ctv = CoxTimeVaryingFitter().fit(tv, id_col="id", start_col="start", stop_col="stop", event_col="event")
    sm = ctv.summary
    out["R6_time_varying_cox"] = {
        "n_packages": int(tv.id.nunique()), "n_with_capped_upstream": int(tv.groupby("id").blocked.max().sum()),
        "terms": {c: {"HR": float(sm.loc[c, "exp(coef)"]),
                      "HR_95ci": [float(sm.loc[c, "exp(coef) lower 95%"]), float(sm.loc[c, "exp(coef) upper 95%"])],
                      "p": float(sm.loc[c, "p"])} for c in sm.index}}

    # ---- R7: broad cap definition ------------------------------------------
    v1s = [Version(f"1.{m}.0") for m in range(0, 27)] + [Version("1.26.4")]
    v2s = [Version(f"2.{m}.0") for m in range(0, 6)]
    oth = use[use.numpy_status == "other"].copy()
    oth["broad_capped"] = [admits_some(s, v1s) and not admits_some(s, v2s) for s in oth.numpy_spec.fillna("")]
    newly = sorted(set(oth[oth.broad_capped].package) - set(capped.package))
    out["R7_broad_cap"] = {"releases_other": int(len(oth)), "releases_broad_capped": int(oth.broad_capped.sum()),
                           "packages_newly_capped": newly, "n_newly_capped": len(newly)}

    path = ROOT / "results" / f"revision_{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
