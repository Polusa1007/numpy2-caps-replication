"""Main analysis (revised design, v2).

Design changes relative to the original plan (reported in the paper; the planned
analysis on the 2026 frame is kept in an appendix):
  * frame defined before the event (top 8,000 on 2024-06-01) -> no survivorship
    selection on post-event popularity; popularity = pre-event 30-day downloads;
  * no latest-release pre-filter (probe screening, see collect_v2.py);
  * mainline release definition (D8);
  * delayed entry: a package is at risk from its first capped release;
  * NumPy C-API indicator from build requirements (audit_capi.py);
  * objective network layer (in-sample in-degree >= 5).
All inferential results are reported with 95% CIs; p-values are unadjusted
except the Holm-adjusted Cox covariates of the main model.

Usage:
    python src/analysis_v2.py
Outputs:
    results/main_pre2024.json, results/*_pre2024_mainline.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from scipy.stats import binomtest, fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.multitest import multipletests

from revision_analyses import cox, days, km_summary

ROOT = Path(__file__).resolve().parent.parent
FRAME = "pre2024"
TAG = f"{FRAME}_mainline"
EVENT = pd.Timestamp("2024-06-16", tz="UTC")
RC1 = pd.Timestamp("2024-03-30", tz="UTC")
BETA = pd.Timestamp("2024-03-12", tz="UTC")
RC2 = pd.Timestamp("2024-05-12", tz="UTC")
CORE_MIN_INDEGREE = 5
SEED, DRAWS = 0, 5000
OUT = ROOT / "results"


def fisher2(df: pd.DataFrame, row: str, col: str) -> dict:
    tab = pd.crosstab(df[row], df[col]).reindex(index=[True, False], columns=[True, False], fill_value=0)
    t = Table2x2(tab.values.astype(float) + (0.5 if (tab.values == 0).any() else 0))
    return {"table": tab.values.tolist(),
            "rate_true": float(df[df[row]][col].mean()), "rate_false": float(df[~df[row]][col].mean()),
            "OR": float(t.oddsratio), "OR_95ci": [float(x) for x in t.oddsratio_confint()],
            "OR_90ci": [float(x) for x in t.oddsratio_confint(alpha=0.10)],
            "p": float(fisher_exact(tab.values)[1])}


def logit(df: pd.DataFrame, y: str, xs: list[str]) -> dict:
    m = sm.Logit(df[y].astype(int), sm.add_constant(df[xs].astype(float))).fit(disp=0, method="bfgs", maxiter=1000)
    ci = np.exp(m.conf_int())
    return {"n": int(len(df)), "events": int(df[y].sum()),
            "terms": {x: {"OR": float(np.exp(m.params[x])), "OR_95ci": [float(ci.loc[x, 0]), float(ci.loc[x, 1])],
                          "p": float(m.pvalues[x])} for x in xs}}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="py312-linux")
    env = ap.parse_args().env
    envsfx = "" if env == "py312-linux" else f"_{env}"
    meta = json.loads((ROOT / "data" / f"meta_{FRAME}.json").read_text(encoding="utf-8"))
    horizon_ts = pd.Timestamp(meta["collected_at"])
    horizon = (horizon_ts - EVENT).days
    pk = pd.read_csv(ROOT / "data" / f"packages_{TAG}{envsfx}.csv")
    for c in ("first_cap", "uncap", "first_release"):
        pk[c] = pd.to_datetime(pk[c], utc=True, format="mixed")
    rel = pd.read_csv(ROOT / "data" / f"releases_{FRAME}{envsfx}.csv")
    rel["uploaded"] = pd.to_datetime(rel.uploaded, utc=True, format="mixed")
    use = rel[~rel.prerelease & ~rel.yanked & ~rel.missing_requires_dist & rel.frontier]
    gen = pd.read_csv(ROOT / "data" / f"generator_{TAG}.csv")[["package", "tool"]]
    capi = pd.read_csv(ROOT / "data" / f"capi_{FRAME}.csv")[["package", "capi"]]
    pk = pk.merge(gen, on="package", how="left").merge(capi, on="package", how="left")

    # covariates (pre-event)
    pk["log_downloads"] = np.log10(pk.downloads_30d.astype(float))
    pk["releases_2023_per10"] = pk.releases_2023 / 10
    pk["capi_i"] = pk.capi.eq("build-numpy").astype(int)
    pk["capi_unknown_i"] = pk.capi.isin(["no-sdist", "unknown"]).astype(int)
    pk["compiled_i"] = pk.compiled.astype(int)
    pk["build_numpy"] = pk.capi.eq("build-numpy")
    dep_set = set(pk.package)
    indeg = {p: set() for p in dep_set}
    for p, g in use[use.package.isin(dep_set) & (use.uploaded < EVENT)].groupby("package"):
        for s in g.deps.dropna():
            for d in str(s).split(";"):
                if d in dep_set and d != p:
                    indeg[d].add(p)
    pk["indegree"] = pk.package.map(lambda p: len(indeg[p]))
    pk["core"] = pk.indegree >= CORE_MIN_INDEGREE
    pk["core_i"] = pk.core.astype(int)
    pk["poetry_i"] = pk.tool.eq("poetry").astype(int)
    fi = pd.read_csv(ROOT / "data" / "frame_indegree_pre2024.csv")
    pk = pk.merge(fi, on="package", how="left")
    pk["frame_indegree"] = pk.frame_indegree.fillna(0)
    pk["log_frame_indeg"] = np.log10(1 + pk.frame_indegree)
    pk["dl_quartile"] = pd.qcut(pk.downloads_30d.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])

    out: dict = {"env": env, "frame": meta["frame_source"], "snapshot": meta["top_snapshot"],
                 "collected_at": meta["collected_at"], "horizon_days": horizon}

    # ---- funnel --------------------------------------------------------------
    in_win = rel
    out["funnel"] = {
        "frame": meta["top_n"], "fetched": len(meta["fetched"]), "screened_in": len(meta["candidates"]),
        "releases_in_window": int(len(in_win)),
        "excluded_prerelease": int(in_win.prerelease.sum()),
        "excluded_yanked": int((~in_win.prerelease & in_win.yanked).sum()),
        "excluded_no_metadata": int((~in_win.prerelease & ~in_win.yanked & in_win.missing_requires_dist).sum()),
        "usable_releases": int((~in_win.prerelease & ~in_win.yanked & ~in_win.missing_requires_dist).sum()),
        "mainline_releases": int(len(use)),
        "dependent": int(len(pk)), "compiled": int(pk.compiled.sum()), "pure": int((~pk.compiled).sum()),
        "capi": {k: int(v) for k, v in pk.capi.value_counts().items()},
        "core": int(pk.core.sum()), "capped": int(pk.ever_capped.sum()),
    }

    # ---- RQ1 ---------------------------------------------------------------
    capped = pk[pk.ever_capped].copy()
    first_cap_days = days(capped.first_cap, EVENT)
    react = capped[capped.cap_timing == "reactive"]
    rd = days(react.first_cap, EVENT)
    out["RQ1"] = {
        "cap_rate": float(pk.ever_capped.mean()),
        "cap_origin": {k: int(v) for k, v in capped.cap_timing.value_counts().items()},
        "reactive_within_30d": int((rd <= 30).sum()), "reactive_within_90d": int((rd <= 90).sum()),
        "reactive_after_365d": int((rd > 365).sum()),
        "cap_style": {k: int(v) for k, v in capped.cap_style.value_counts().items()},
        "poetry": {"n": int((capped.tool == "poetry").sum()),
                   "by_origin": {k: int(v) for k, v in capped[capped.tool == "poetry"].cap_timing.value_counts().items()}},
        "by_capi_vs_rest": fisher2(pk[pk.capi != "no-sdist"].assign(build_numpy=lambda d: d.capi.eq("build-numpy")), "build_numpy", "ever_capped"),
        "by_compiled": fisher2(pk, "compiled", "ever_capped"),
        "by_core": fisher2(pk, "core", "ever_capped"),
        "by_capi_category": {k: {"n": int(len(g)), "capped": int(g.ever_capped.sum()), "rate": float(g.ever_capped.mean())}
                             for k, g in pk.groupby("capi")},
    }

    # prevalence over time (all NumPy-dependent packages)
    grid = np.arange(-365, horizon + 1, 7)
    seqs = {p: g.sort_values("uploaded")[["uploaded", "numpy_status"]].values
            for p, g in use[use.package.isin(dep_set)].groupby("package")}
    # Backfill: status of the last stable release before the window, so that packages
    # count from the start of the window rather than from their first in-window release.
    from classify import RAW as _RAW, pre_window_status, safe as _safe
    backfilled = 0
    for p_ in list(seqs):
        rels = json.loads((_RAW / "pkg" / f"{_safe(p_)}.json").read_text(encoding="utf-8"))["releases"]
        st0 = pre_window_status(p_, rels)
        if st0 != "unknown":
            t0 = pd.Timestamp("2023-06-15", tz="UTC")
            seqs[p_] = np.vstack([np.array([[t0, st0]], dtype=object), seqs[p_]])
            backfilled += 1
    # Fixed cohort (sensitivity): packages NumPy-dependent at day FIXED_DAY, followed with a
    # constant denominator; a package that later drops NumPy counts as not capped.
    FIXED_DAY = -180
    t_fix = EVENT + pd.Timedelta(days=FIXED_DAY)
    fixed = {p for p, arr in seqs.items()
             if (arr[:, 0] <= t_fix).any() and arr[arr[:, 0] <= t_fix][-1, 1] in ("capped", "compatible", "other")}
    rows = []
    for d in grid:
        t = EVENT + pd.Timedelta(days=int(d))
        lat = {p: arr[arr[:, 0] <= t][-1, 1] for p, arr in seqs.items() if (arr[:, 0] <= t).any()}
        st = pd.Series(list(lat.values()))
        dep_now = st.isin(["capped", "compatible", "other"])
        row = {"day": int(d), "n": int(dep_now.sum()), "share": float((st[dep_now] == "capped").mean())}
        if d >= FIXED_DAY:
            row["fixed_share"] = float(np.mean([lat[p] == "capped" for p in fixed]))
        rows.append(row)
    prev = pd.DataFrame(rows)
    pick = lambda d: prev.loc[(prev.day - d).abs().idxmin()].to_dict()  # noqa: E731
    after = prev[prev.day >= -180]
    out["prevalence"] = {**{f"day_{d}": pick(d) for d in (-180, -90, 0, 30, 90, 180, 365, 730)},
                         "peak": after.loc[after.share.idxmax()].to_dict(),
                         "fixed_cohort_day": FIXED_DAY, "fixed_cohort_n": len(fixed), "backfilled": backfilled,
                         "fixed_peak": after.loc[after.fixed_share.idxmax()].to_dict()}
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    shown = prev
    ax.plot(shown.day, shown.share * 100, color="#2b6cb0", label="All packages with a NumPy dependency at the time")
    fx = prev[prev.day >= FIXED_DAY]
    ax.plot(fx.day, fx.fixed_share * 100, color="#c05621", ls="-.",
            label=f"Fixed cohort: NumPy-dependent at day {FIXED_DAY} (n={len(fixed)})")
    ax2.plot(shown.day, shown.n, color="#2b6cb0")
    ax2.axhline(len(fixed), color="#c05621", ls="-.")
    ax2.set_ylabel("Denominator")
    for a_ in (ax, ax2):
        a_.axvline(0, color="k", ls="--", lw=1)
        a_.axvline((BETA - EVENT).days, color="#6b8e23", ls=(0, (1, 3)), lw=1.2)
        a_.axvline((RC1 - EVENT).days, color="grey", ls=":", lw=1.2)
        a_.axvline((RC2 - EVENT).days, color="purple", ls=(0, (3, 2, 1, 2)), lw=1.0)
    ax.plot([], [], color="k", ls="--", lw=1, label="NumPy 2.0.0 (2024-06-16)")
    ax.plot([], [], color="#6b8e23", ls=(0, (1, 3)), lw=1.2, label="2.0.0b1 (2024-03-12)")
    ax.plot([], [], color="grey", ls=":", lw=1.2, label="2.0.0rc1 (2024-03-30)")
    ax.plot([], [], color="purple", ls=(0, (3, 2, 1, 2)), lw=1.0, label="2.0.0rc2 (2024-05-12)")
    ax.set_ylim(bottom=0)
    ax2.set_xlabel("Days since NumPy 2.0.0 release")
    ax.set_ylabel("% whose latest release\nis capped")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / f"prevalence_{TAG}{envsfx}.png", dpi=200)
    plt.close(fig)

    # ---- RQ2: delayed-entry main model ---------------------------------------
    capped["early"] = capped.adaptation.eq("before-release")
    r = capped[capped.adaptation.isin(["after-release", "still-capped"])].copy()
    r["duration"] = days(r.uncap, EVENT).fillna(horizon).clip(lower=0.5)
    r["observed"] = r.uncap.notna().astype(int)
    r["entry"] = np.minimum(days(r.first_cap, EVENT).clip(lower=0), r.duration - 0.25)
    covs = ["capi_i", "capi_unknown_i", "log_downloads", "core_i", "releases_2023_per10", "age_years_at_event",
            "poetry_i"]
    main_cox = cox(r, covs, "entry")
    # PH check for a left-truncated model: split follow-up at CUT days and test
    # covariate x late-period interactions in a counting-process Cox model.
    main_cox["ph_test_p_by_cut"] = {}
    for CUT in (90.0, 365.0, 150.0):
      ep = []
      for x in r.dropna(subset=covs).itertuples():
          base = {"id": x.package, **{k: getattr(x, k) for k in covs}}
          a0, b0 = float(x.entry), float(x.duration)
          if a0 < CUT < b0:
              ep.append({**base, "start": a0, "stop": CUT, "late": 0, "event": 0})
              ep.append({**base, "start": CUT, "stop": b0, "late": 1, "event": x.observed})
          else:
              ep.append({**base, "start": a0, "stop": b0, "late": int(a0 >= CUT), "event": x.observed})
      ep = pd.DataFrame(ep)
      for k in covs:
          ep[f"{k}_x_late"] = ep[k] * ep.late
      for pen in (0.01, 0.05, 0.2):
          try:
              ph = CoxTimeVaryingFitter(penalizer=pen).fit(ep.drop(columns="late"), id_col="id", start_col="start",
                                                           stop_col="stop", event_col="event").summary
              break
          except Exception:  # noqa: BLE001 - sparse late-period cells
              continue
      main_cox.setdefault("ph_penalizer", {})[str(int(CUT))] = pen
      main_cox["ph_test_p_by_cut"][str(int(CUT))] = {k: float(ph.loc[f"{k}_x_late", "p"]) for k in covs}
      main_cox["ph_test_p"] = main_cox["ph_test_p_by_cut"][str(int(CUT))]
    main_cox["ph_test_note"] = "piecewise test, cuts at 90/150/365 days (ph_test_p = 150), ridge penalty 0.01"
    ps = [main_cox["terms"][c]["p"] for c in covs]
    for c, h in zip(covs, multipletests(ps, method="holm")[1]):
        main_cox["terms"][c]["p_holm"] = float(h)
    out["RQ2"] = {
        "n_early": int(capped.early.sum()),
        "early_median_days_before": float(-days(capped[capped.early].uncap, EVENT).median()) if capped.early.any() else None,
        "km_delayed_entry": km_summary(r.duration, r.observed, r.entry),
        "km_release_origin_no_entry": km_summary(r.duration, r.observed),
        "cox_delayed_entry": main_cox,
        "still_capped": int((r.observed == 0).sum()),
        "dropped_dependency_lifts": int((r.uncap_kind == "dropped-dependency").sum()),
    }
    rr = r.copy()
    rr.loc[rr.uncap_kind == "dropped-dependency", "observed"] = 0
    out["RQ2"]["sens_dropped_as_censored_median"] = float(
        KaplanMeierFitter().fit(rr.duration, rr.observed, entry=rr.entry).median_survival_time_)
    c = capped[capped.cap_timing.isin(["pre-emptive", "reactive"])].copy()
    c["duration"] = days(c.uncap.fillna(horizon_ts), c.first_cap).clip(lower=0.5)
    c["observed"] = c.uncap.notna().astype(int)
    out["RQ2"]["cap_duration"] = {k: km_summary(g.duration, g.observed) for k, g in c.groupby("cap_timing")}
    np_ = r[r.tool != "poetry"]
    out["RQ2"]["sens_no_poetry"] = {"km": km_summary(np_.duration, np_.observed, np_.entry),
                                    "cox": cox(np_, [c for c in covs if c != "poetry_i"], "entry")}
    fr_covs = [("log_frame_indeg" if c == "core_i" else c) for c in covs]
    out["RQ2"]["sens_frame_indegree"] = {"cox": cox(r, fr_covs, "entry"),
                                         "n_core_frame10": int((pk.frame_indegree >= 10).sum())}
    mj = r[r.cap_style == "major-cap"]
    out["RQ2"]["sens_major_caps"] = {"km": km_summary(mj.duration, mj.observed, mj.entry), "cox": cox(mj, covs, "entry")}
    kn = r[r.capi_unknown_i == 0]
    out["RQ2"]["sens_known_capi"] = {"cox": cox(kn, [x for x in covs if x != "capi_unknown_i"], "entry")}
    out["RQ2"]["by_quartile_median"] = {str(k): km_summary(g.duration, g.observed, g.entry)["median"]
                                        for k, g in r.groupby("dl_quartile", observed=True) if len(g) >= 5}
    out["RQ2"]["by_quartile_km"] = {str(k): km_summary(g.duration, g.observed, g.entry)
                                    for k, g in r.groupby("dl_quartile", observed=True) if len(g) >= 5}
    out["RQ2"]["km_group_n"] = {"build_numpy": int(((r.capi_i == 1) & (r.capi_unknown_i == 0)).sum()),
                                "other_known": int(((r.capi_i == 0) & (r.capi_unknown_i == 0)).sum()),
                                "unknown": int(r.capi_unknown_i.sum())}

    # early adaptation
    out["RQ2"]["early"] = {
        "by_capi": fisher2(capped[capped.capi != "no-sdist"], "build_numpy", "early"),
        "by_compiled": fisher2(capped, "compiled", "early"),
        "by_core": fisher2(capped, "core", "early"),
        "logit": logit(capped[capped.capi_unknown_i == 0], "early", ["capi_i", "core_i", "log_downloads"]),
        "logit_frame_indegree": logit(capped[capped.capi_unknown_i == 0], "early",
                                      ["capi_i", "log_frame_indeg", "log_downloads"]),
        "n_early_known_capi": int(capped[capped.capi_unknown_i == 0].early.sum()),
        "n_capped_known_capi": int((capped.capi_unknown_i == 0).sum()),
        "packages": sorted(capped[capped.early].package),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    for flag, label in [(1, "Builds against NumPy"), (0, "Does not / pure Python")]:
        g = r[(r.capi_i == flag) & (r.capi_unknown_i == 0)]
        if len(g) >= 3:
            KaplanMeierFitter().fit(g.duration, g.observed, entry=g.entry, label=f"{label} (n={len(g)})"
                                    ).plot_survival_function(ax=ax, ci_show=True)
    ax.set_xlabel("Days since NumPy 2.0.0 release")
    ax.set_ylabel("Share of packages at risk still capped")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(OUT / f"km_{TAG}{envsfx}.png", dpi=200)
    plt.close(fig)

    # ---- RQ3 ---------------------------------------------------------------
    cs = capped.set_index("package")
    pairs = [{"down": dn, "up": up, "d": row.uncap, "u": cs.loc[up, "uncap"]}
             for dn, row in cs.iterrows() for up in str(row.capped_deps).split(";")
             if up in cs.index and up != dn]
    P = pd.DataFrame(pairs)
    allp = len(P)
    P = P.dropna(subset=["d", "u"])
    P["lag"] = (P.d - P.u).dt.total_seconds() / 86400
    ties = int((P.lag.abs() < 1).sum())
    P = P[P.lag.abs() >= 1]
    per = P.groupby("down").lag.apply(lambda x: (x > 0).mean())
    maj = per[per != 0.5]
    lifted = capped[capped.uncap.notna()].copy()
    lifted["stratum"] = np.where(lifted.core, "core", "periphery") + "|" + lifted.dl_quartile.astype(str)
    pool = {k: g.uncap.values for k, g in lifted.groupby("stratum")}
    P["stratum"] = P.up.map(lifted.set_index("package").stratum)
    rng = np.random.default_rng(SEED)
    dv = P.d.dt.tz_convert(None).values.astype("datetime64[ns]")
    null = np.empty(DRAWS)
    for i in range(DRAWS):
        u = np.array([rng.choice(pool[s]) for s in P.stratum]).astype("datetime64[ns]")
        lag = (dv - u) / np.timedelta64(1, "D")
        m = np.abs(lag) >= 1
        null[i] = pd.DataFrame({"g": P.down.values[m], "a": lag[m] > 0}).groupby("g").a.mean().mean()
    # alternative baseline: keep upstream dates fixed, reassign downstream lift dates
    lifted_d = lifted.set_index("package")
    downs = P.down.unique()
    dstrat = {d: lifted_d.loc[d, "stratum"] for d in downs}
    null2 = np.empty(DRAWS)
    uv = P.u.dt.tz_convert(None).values.astype("datetime64[ns]")
    for i in range(DRAWS):
        draw = {d: rng.choice(pool[dstrat[d]]) for d in downs}
        dd = np.array([draw[d] for d in P.down]).astype("datetime64[ns]")
        lag = (dd - uv) / np.timedelta64(1, "D")
        m = np.abs(lag) >= 1
        null2[i] = pd.DataFrame({"g": P.down.values[m], "a": lag[m] > 0}).groupby("g").a.mean().mean()
    tv_rows = []
    for x in r.itertuples():
        ups = [u for u in str(x.capped_deps).split(";") if u in cs.index and u != x.package]
        ut = [cs.loc[u, "uncap"] for u in ups]
        clear = 0.0 if not ups else (np.inf if any(pd.isna(t) for t in ut)
                                     else max(0.0, max((t - EVENT).total_seconds() / 86400 for t in ut)))
        base = {"id": x.package, **{k: getattr(x, k) for k in covs}}
        a0, b0 = float(x.entry), float(x.duration)
        if clear <= a0 or clear >= b0:
            tv_rows.append({**base, "start": a0, "stop": b0, "blocked": int(clear > a0), "event": x.observed})
        else:
            tv_rows.append({**base, "start": a0, "stop": clear, "blocked": 1, "event": 0})
            tv_rows.append({**base, "start": clear, "stop": b0, "blocked": 0, "event": x.observed})
    tv = pd.DataFrame(tv_rows).dropna()
    ctv = CoxTimeVaryingFitter().fit(tv, id_col="id", start_col="start", stop_col="stop", event_col="event")
    s = ctv.summary
    out["RQ3"] = {
        "n_pairs_all": int(allp), "n_pairs_both_lifted": int(len(P) + ties), "ties": ties,
        "n_pairs": int(len(P)), "pair_share_after": float((P.lag > 0).mean()),
        "lag_median": float(P.lag.median()), "lag_iqr": [float(P.lag.quantile(.25)), float(P.lag.quantile(.75))],
        "top_upstreams": P.up.value_counts().head(5).to_dict(),
        "n_downstreams": int(len(per)), "downstreams_all_followed": int((per == 1).sum()),
        "downstreams_majority": int((maj > 0.5).sum()), "downstreams_minority": int((maj < 0.5).sum()),
        "downstream_share_mean": float(per.mean()),
        "downstream_sign_p": float(binomtest(int((maj > 0.5).sum()), int(len(maj)), 0.5, alternative="greater").pvalue),
        "perm_null_mean": float(null.mean()), "perm_null_95": [float(np.quantile(null, .025)), float(np.quantile(null, .975))],
        "perm_p": float(((null >= per.mean()).sum() + 1) / (DRAWS + 1)),
        "perm2_null_mean": float(null2.mean()),
        "perm2_null_95": [float(np.quantile(null2, .025)), float(np.quantile(null2, .975))],
        "perm2_p": float(((null2 >= per.mean()).sum() + 1) / (DRAWS + 1)),
        "tv_n_with_capped_upstream": int(tv.groupby("id").blocked.max().sum()),
        "tv_events_total": int(tv.event.sum()),
        "tv_events_in_packages_with_upstream": int(tv[tv.id.isin(tv[tv.blocked == 1].id)].event.sum()),
        "tv_events_while_blocked": int(tv[(tv.blocked == 1)].event.sum()),
        "tv_n_covariates": len(covs) + 1,
        "tv_blocked": {"HR": float(s.loc["blocked", "exp(coef)"]),
                       "HR_95ci": [float(s.loc["blocked", "exp(coef) lower 95%"]), float(s.loc["blocked", "exp(coef) upper 95%"])],
                       "p": float(s.loc["blocked", "p"])},
    }

    # timeline figure (all capped packages with an observed lift)
    tl = capped[capped.uncap.notna()].copy()
    floor = EVENT - pd.Timedelta(days=365)
    tl["start"] = tl.first_cap.where(tl.first_cap > floor, floor)
    tl = tl.sort_values("uncap")
    colors = {"legacy": "#8c8c8c", "pre-emptive": "#2b6cb0", "reactive": "#c05621"}
    fig, ax = plt.subplots(figsize=(7, max(4, 0.1 * len(tl))))
    for i, x in enumerate(tl.itertuples()):
        ax.plot([x.start.to_pydatetime(), x.uncap.to_pydatetime()], [i, i], color=colors[x.cap_timing], lw=1.4)
    ax.axvline(EVENT.to_pydatetime(), color="k", ls="--", lw=1)
    ax.set_yticks([])
    ax.set_ylabel(f"Capped packages with an observed lift (n={len(tl)}), sorted by lift date", fontsize=11)
    ax.set_xlabel("Capped period (dashed: NumPy 2.0.0 release)", fontsize=11)
    ax.tick_params(axis="x", labelsize=10, labelrotation=30)
    for k, v in colors.items():
        ax.plot([], [], color=v, lw=3, label=k)
    ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"timeline_{TAG}{envsfx}.png", dpi=200)
    plt.close(fig)
    out["timeline_n"] = int(len(tl))

    (OUT / f"main_{FRAME}{envsfx}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str)[:12000])


if __name__ == "__main__":
    main()
