"""Supplementary analyses, set 6 (all exploratory; main frame, mainline).

  * cluster-robust Cox and logistic models (clusters = GitHub repository / organisation)
  * RQ3 restricted to upstreams that lifted after the release; lag from the last
    upstream lift to the downstream lift against a stratified permutation baseline
  * pre-emptive caps split at the NumPy 2.0.0 beta (2024-03-12)
  * releases classified "other" and a broad cap definition
  * packages that re-introduced a cap, and sensitivity excluding them
  * lift dates counting pre-releases / yanked releases
  * exposure measured from extension binaries (audit_wheel_capi.py), with bounds for
    packages of unknown build status

Usage:
    python src/review6.py
Output:
    results/review6.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from scipy.stats import binomtest, fisher_exact
from statsmodels.stats.contingency_tables import Table2x2

from revision_analyses import days, km_summary

ROOT = Path(__file__).resolve().parent.parent
EVENT = pd.Timestamp("2024-06-16", tz="UTC")
BETA = pd.Timestamp("2024-03-12", tz="UTC")
SEED, DRAWS = 0, 5000


def prep():
    meta = json.loads((ROOT / "data" / "meta_pre2024.json").read_text(encoding="utf-8"))
    horizon_ts = pd.Timestamp(meta["collected_at"])
    pk = pd.read_csv(ROOT / "data" / "packages_pre2024_mainline.csv")
    for c in ("first_cap", "uncap", "first_release"):
        pk[c] = pd.to_datetime(pk[c], utc=True, format="mixed")
    rel = pd.read_csv(ROOT / "data" / "releases_pre2024.csv")
    rel["uploaded"] = pd.to_datetime(rel.uploaded, utc=True, format="mixed")
    capi = pd.read_csv(ROOT / "data" / "capi_pre2024.csv")[["package", "capi"]]
    gen = pd.read_csv(ROOT / "data" / "generator_pre2024_mainline.csv")[["package", "tool"]]
    gh = pd.read_csv(ROOT / "data" / "github_signatures_pre2024.csv")[["package", "repo"]]
    pk = pk.merge(capi, on="package", how="left").merge(gen, on="package", how="left").merge(gh, on="package", how="left")
    pk["cluster_repo"] = pk.repo.fillna(pk.package)
    pk["cluster_org"] = pk.repo.str.split("/").str[0].fillna(pk.package)
    pk["log_downloads"] = np.log10(pk.downloads_30d.astype(float))
    pk["releases_2023_per10"] = pk.releases_2023 / 10
    pk["capi_i"] = pk.capi.eq("build-numpy").astype(int)
    pk["capi_unknown_i"] = pk.capi.isin(["no-sdist", "unknown"]).astype(int)
    pk["poetry_i"] = pk.tool.eq("poetry").astype(int)
    use = rel[~rel.prerelease & ~rel.yanked & ~rel.missing_requires_dist & rel.frontier]
    dep = set(pk.package)
    indeg = {p: set() for p in dep}
    for p, g in use[use.package.isin(dep) & (use.uploaded < EVENT)].groupby("package"):
        for s in g.deps.dropna():
            for d in str(s).split(";"):
                if d in dep and d != p:
                    indeg[d].add(p)
    pk["core_i"] = pk.package.map(lambda p: int(len(indeg[p]) >= 5))
    pk["dl_quartile"] = pd.qcut(pk.downloads_30d.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return pk, rel, use, horizon_ts


def cohort(pk, horizon_ts, uncap_col="uncap"):
    r = pk[pk.ever_capped & pk.adaptation.isin(["after-release", "still-capped"])].copy()
    horizon = (horizon_ts - EVENT).days
    r["duration"] = days(r[uncap_col], EVENT).fillna(horizon).clip(lower=0.5)
    r["observed"] = r[uncap_col].notna().astype(int)
    r["entry"] = np.minimum(days(r.first_cap, EVENT).clip(lower=0), r.duration - 0.25)
    return r


COVS = ["capi_i", "capi_unknown_i", "poetry_i", "log_downloads", "core_i", "releases_2023_per10", "age_years_at_event"]


def cox_terms(m) -> dict:
    s = m.summary
    return {c: {"HR": float(s.loc[c, "exp(coef)"]),
                "HR_95ci": [float(s.loc[c, "exp(coef) lower 95%"]), float(s.loc[c, "exp(coef) upper 95%"])],
                "p": float(s.loc[c, "p"])} for c in s.index}


def main() -> None:
    pk, rel, use, horizon_ts = prep()
    out: dict = {}
    capped = pk[pk.ever_capped].copy()
    capped["early"] = capped.adaptation.eq("before-release").astype(int)
    r = cohort(pk, horizon_ts)

    # ---- cluster-robust models -------------------------------------------------
    out["clusters"] = {"repos": int(pk.cluster_repo.nunique()), "orgs": int(pk.cluster_org.nunique()),
                       "cohort_repos": int(r.cluster_repo.nunique()), "cohort_orgs": int(r.cluster_org.nunique())}
    for cl in ("cluster_repo", "cluster_org"):
        d = r[["duration", "observed", "entry", cl] + COVS].dropna()
        m = CoxPHFitter().fit(d, "duration", "observed", entry_col="entry", cluster_col=cl)
        out[f"cox_{cl}"] = cox_terms(m)
        k = capped[capped.capi_unknown_i == 0]
        X = sm.add_constant(k[["capi_i", "core_i", "log_downloads"]].astype(float))
        lg = sm.Logit(k.early, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(k[cl])[0]})
        ci = np.exp(lg.conf_int())
        out[f"logit_{cl}"] = {c: {"OR": float(np.exp(lg.params[c])), "OR_95ci": [float(ci.loc[c, 0]), float(ci.loc[c, 1])],
                                  "p": float(lg.pvalues[c])} for c in ["capi_i", "core_i", "log_downloads"]}

    # ---- RQ3 restricted and lag analysis -----------------------------------------
    cs = capped.set_index("package")
    rows = []
    for dn, row in cs.iterrows():
        for up in str(row.capped_deps).split(";"):
            if up in cs.index and up != dn:
                rows.append({"down": dn, "up": up, "d": row.uncap, "u": cs.loc[up, "uncap"]})
    P = pd.DataFrame(rows).dropna(subset=["d", "u"])
    P["lag"] = (P.d - P.u).dt.total_seconds() / 86400
    Q = P[(P.u >= EVENT) & (P.lag.abs() >= 1)]
    per = Q.groupby("down").lag.apply(lambda x: (x > 0).mean())
    maj = per[per != 0.5]
    out["rq3_upstream_after_release"] = {
        "pairs": int(len(Q)), "pairs_after": int((Q.lag > 0).sum()), "downstreams": int(len(per)),
        "downstream_majority": int((maj > 0.5).sum()), "downstream_minority": int((maj < 0.5).sum()),
        "sign_p": float(binomtest(int((maj > 0.5).sum()), int(len(maj)), 0.5, alternative="greater").pvalue) if len(maj) else None,
        "lag_median": float(Q.lag.median()) if len(Q) else None}
    # lag from the last upstream lift to the downstream lift
    lastup = P.groupby("down").u.max()
    dd = cs.loc[lastup.index, "uncap"]
    lag = (dd - lastup).dt.total_seconds() / 86400
    lifted = capped[capped.uncap.notna()].copy()
    lifted["stratum"] = np.where(lifted.core_i == 1, "core", "periphery") + "|" + lifted.dl_quartile.astype(str)
    pool = {k: g.uncap.values for k, g in lifted.groupby("stratum")}
    strat = lifted.set_index("package").stratum
    rng = np.random.default_rng(SEED)
    lu = lastup.dt.tz_convert(None).values.astype("datetime64[ns]")
    null_med, null_30 = np.empty(DRAWS), np.empty(DRAWS)
    for i in range(DRAWS):
        dr = np.array([rng.choice(pool[strat[x]]) for x in lastup.index]).astype("datetime64[ns]")
        lg_ = (dr - lu) / np.timedelta64(1, "D")
        null_med[i] = np.median(np.abs(lg_))
        null_30[i] = np.mean((lg_ >= 0) & (lg_ <= 30))
    obs_30 = float(((lag >= 0) & (lag <= 30)).mean())
    out["rq3_last_upstream_lag"] = {
        "downstreams": int(len(lag)), "lag_median": float(lag.median()), "lag_iqr": [float(lag.quantile(.25)), float(lag.quantile(.75))],
        "share_within_30d_after": obs_30, "null_share_within_30d": float(null_30.mean()),
        "null_share_95": [float(np.quantile(null_30, .025)), float(np.quantile(null_30, .975))],
        "p_one_sided": float(((null_30 >= obs_30).sum() + 1) / (DRAWS + 1)),
        "abs_lag_median": float(np.median(np.abs(lag))), "null_abs_lag_median": float(null_med.mean()),
        "null_abs_lag_95": [float(np.quantile(null_med, .025)), float(np.quantile(null_med, .975))],
        "p_abs_lag_smaller": float(((null_med <= np.median(np.abs(lag))).sum() + 1) / (DRAWS + 1))}

    # ---- pre-emptive caps split at the beta ---------------------------------------
    pre = capped[capped.cap_timing == "pre-emptive"].copy()
    pre["after_beta"] = pre.first_cap >= BETA
    hz = horizon_ts
    split = {}
    for flag, g in pre.groupby("after_beta"):
        dur = days(g.uncap.fillna(hz), g.first_cap).clip(lower=0.5)
        split["after_beta" if flag else "before_beta"] = {
            "n": int(len(g)), "poetry": int((g.tool == "poetry").sum()),
            "major_cap": int((g.cap_style == "major-cap").sum()), "early_adapters": int((g.adaptation == "before-release").sum()),
            "cap_duration": km_summary(dur, g.uncap.notna().astype(int))}
    out["preemptive_split"] = split

    # ---- "other" releases and the broad cap definition ---------------------------
    u2 = use[use.package.isin(set(pk.package))]
    v1s = [Version(f"1.{m}.0") for m in range(27)] + [Version("1.26.4")]
    v2s = [Version(f"2.{m}.0") for m in range(6)]

    def admits(spec, vs):
        try:
            ss = SpecifierSet(spec)
        except Exception:
            return False
        return any(ss.contains(v, prereleases=True) for v in vs)
    oth = u2[u2.numpy_status == "other"].copy()
    oth["broad_capped"] = [admits(s, v1s) and not admits(s, v2s) for s in oth.numpy_spec.fillna("")]
    out["other"] = {"releases_other": int(len(oth)), "packages_with_other": int(oth.package.nunique()),
                    "releases_broad_capped": int(oth.broad_capped.sum()),
                    "packages_newly_capped_broad": sorted(set(oth[oth.broad_capped].package) - set(capped.package)),
                    "capped_packages_with_other_release": int(oth.package.isin(capped.package).sum() and
                                                              oth[oth.package.isin(capped.package)].package.nunique())}

    # ---- re-capped packages ----------------------------------------------------------
    recap = []
    for p, g in use[use.package.isin(capped.package)].groupby("package"):
        un = cs.loc[p, "uncap"]
        if pd.notna(un) and (g[(g.uploaded > un)].numpy_status == "capped").any():
            recap.append(p)
    rr = r[~r.package.isin(recap)]
    out["recapped"] = {"n": len(recap), "packages": recap,
                       "km_excluding": km_summary(rr.duration, rr.observed, rr.entry)}

    # ---- lifts counting pre-releases / yanked releases -----------------------------
    for label, mask in (("prereleases", ~rel.yanked & ~rel.missing_requires_dist),
                        ("yanked", ~rel.prerelease & ~rel.missing_requires_dist)):
        alt = rel[mask & rel.package.isin(capped.package)]
        new_uncap = {}
        for p, g in alt.groupby("package"):
            fc = cs.loc[p, "first_cap"]
            later = g[(g.uploaded > fc) & g.numpy_status.isin(["compatible", "none"])].sort_values("uploaded")
            cand = later.uploaded.iloc[0] if len(later) else pd.NaT
            orig = cs.loc[p, "uncap"]
            new_uncap[p] = min(x for x in (cand, orig) if pd.notna(x)) if (pd.notna(cand) or pd.notna(orig)) else pd.NaT
        pk2 = pk.copy()
        pk2["uncap_alt"] = pk2.package.map(new_uncap)
        pk2.loc[pk2.uncap_alt.isna(), "uncap_alt"] = pk2.uncap
        pk2["uncap_alt"] = pd.to_datetime(pk2.uncap_alt, utc=True)
        ra = cohort(pk2, horizon_ts, "uncap_alt")
        out[f"lift_with_{label}"] = {"km": km_summary(ra.duration, ra.observed, ra.entry),
                                     "packages_earlier_lift": int((ra.uncap_alt < ra.uncap).sum())}

    # ---- binary-level exposure --------------------------------------------------------
    wf = ROOT / "data" / "wheel_capi_pre2024.csv"
    if wf.exists():
        w = pd.read_csv(wf)[["package", "binary_capi"]]
        pkb = pk.merge(w, on="package", how="left")
        pkb["binary_capi"] = pkb.binary_capi.fillna("pure")
        known = pkb[pkb.binary_capi.isin(["uses-capi", "no-capi", "pure"])].copy()
        known["bin_i"] = known.binary_capi.eq("uses-capi")
        res = {"counts": pkb.binary_capi.value_counts().to_dict()}
        # agreement with the build-requirement rule (compiled packages with both classifications)
        both = pkb[pkb.compiled & pkb.capi.isin(["build-numpy", "no-build-numpy"]) & pkb.binary_capi.isin(["uses-capi", "no-capi"])]
        ct = pd.crosstab(both.capi.eq("build-numpy"), both.binary_capi.eq("uses-capi"))
        tp = int(((both.capi == "build-numpy") & (both.binary_capi == "uses-capi")).sum())
        fp = int(((both.capi == "build-numpy") & (both.binary_capi == "no-capi")).sum())
        fn = int(((both.capi == "no-build-numpy") & (both.binary_capi == "uses-capi")).sum())
        tn = int(((both.capi == "no-build-numpy") & (both.binary_capi == "no-capi")).sum())
        res["agreement"] = {"n": int(len(both)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}
        unk = pkb[pkb.capi.isin(["no-sdist", "unknown"])]
        res["unknown_group"] = unk.binary_capi.value_counts().to_dict()

        def fisher(df, col, outcome):
            tab = pd.crosstab(df[col], df[outcome]).reindex(index=[True, False], columns=[True, False], fill_value=0)
            t = Table2x2(tab.values.astype(float) + (0.5 if (tab.values == 0).any() else 0))
            return {"table": tab.values.tolist(), "OR": float(t.oddsratio), "OR_95ci": [float(x) for x in t.oddsratio_confint()],
                    "p": float(fisher_exact(tab.values)[1])}
        res["cap_rate_by_binary"] = fisher(known, "bin_i", "ever_capped")
        kc = known[known.ever_capped].copy()
        kc["early"] = kc.adaptation.eq("before-release")
        res["early_by_binary"] = fisher(kc, "bin_i", "early")
        # bounds for unknown build status in the build-requirement analysis
        for label, val in (("unknown_as_build", True), ("unknown_as_not", False)):
            b = pkb.copy()
            b["bn"] = b.capi.eq("build-numpy") | (b.capi.isin(["no-sdist", "unknown"]) & val)
            res[f"cap_rate_{label}"] = fisher(b, "bn", "ever_capped")
        rb = cohort(pkb.assign(bin_i=pkb.binary_capi.eq("uses-capi").astype(int),
                               bin_unk_i=(~pkb.binary_capi.isin(["uses-capi", "no-capi", "pure"])).astype(int)), horizon_ts)
        covs_b = ["bin_i", "bin_unk_i", "poetry_i", "log_downloads", "core_i", "releases_2023_per10", "age_years_at_event"]
        d = rb[["duration", "observed", "entry"] + covs_b].dropna()
        d = d.loc[:, (d != d.iloc[0]).any() | d.columns.isin(["duration", "observed", "entry"])]
        res["cox_binary"] = cox_terms(CoxPHFitter().fit(d, "duration", "observed", entry_col="entry"))
        res["km_n_binary_cohort"] = {"uses_capi": int(rb.bin_i.sum()), "other_known": int(((rb.bin_i == 0) & (rb.bin_unk_i == 0)).sum())}
        out["binary_exposure"] = res

    (ROOT / "results" / "review6.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
