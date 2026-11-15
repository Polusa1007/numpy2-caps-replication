"""Supplementary analyses, set 4 (all exploratory).

  * sample bookkeeping (capped with unknown build status, compiled early adapters)
  * early adapters split at NumPy 2.0.0rc1 (2024-03-30)
  * delayed-entry medians for core and periphery separately
  * core-threshold sensitivity (in-degree >= 3, 5, 10)
  * Firth-penalised logistic regression for early adaptation
  * collinearity of Cox covariates (correlations, VIF)
  * CI for the GitHub cross-check difference in proportions
  * stepwise design comparison: change one design element at a time

Usage:
    python src/review4.py
Output:
    results/review4.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from packaging.utils import canonicalize_name
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import confint_proportions_2indep

from revision_analyses import cox, days, km_summary

ROOT = Path(__file__).resolve().parent.parent
EVENT = pd.Timestamp("2024-06-16", tz="UTC")
RC1 = pd.Timestamp("2024-03-30", tz="UTC")
PLANNED = ["compiled_i", "log_downloads", "releases_2023_per10", "age_years_at_event"]


def load(frame: str, tag: str, meta_name: str) -> tuple[pd.DataFrame, int]:
    meta = json.loads((ROOT / "data" / f"meta_{meta_name}.json").read_text(encoding="utf-8"))
    horizon = (pd.Timestamp(meta["collected_at"]) - EVENT).days
    pk = pd.read_csv(ROOT / "data" / f"packages_{tag}.csv")
    for c in ("first_cap", "uncap", "first_release"):
        pk[c] = pd.to_datetime(pk[c], utc=True, format="mixed")
    pk["log_downloads"] = np.log10(pk.downloads_30d.astype(float))
    pk["releases_2023_per10"] = pk.releases_2023 / 10
    pk["compiled_i"] = pk.compiled.astype(int)
    return pk, horizon


def cohort(pk: pd.DataFrame, horizon: int, delayed: bool) -> pd.DataFrame:
    r = pk[pk.ever_capped & pk.adaptation.isin(["after-release", "still-capped"])].copy()
    r["duration"] = days(r.uncap, EVENT).fillna(horizon).clip(lower=0.5)
    r["observed"] = r.uncap.notna().astype(int)
    r["entry"] = np.minimum(days(r.first_cap, EVENT).clip(lower=0), r.duration - 0.25) if delayed else 0.0
    return r


def firth_logit(y: np.ndarray, X: np.ndarray, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Firth (Jeffreys-prior) penalised logistic regression; returns beta and Wald SE."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        w = p * (1 - p)
        info = X.T @ (X * w[:, None])
        inv = np.linalg.inv(info)
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(w)[:, None], inv, X * np.sqrt(w)[:, None])
        u = X.T @ (y - p + h * (0.5 - p))
        step = inv @ u
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ b))
    inv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    return b, np.sqrt(np.diag(inv))


def main() -> None:
    out: dict = {}
    pk, horizon = load("pre2024", "pre2024_mainline", "pre2024")
    capi = pd.read_csv(ROOT / "data" / "capi_pre2024.csv")[["package", "capi"]]
    gen = pd.read_csv(ROOT / "data" / "generator_pre2024_mainline.csv")[["package", "tool"]]
    fi = pd.read_csv(ROOT / "data" / "frame_indegree_pre2024.csv")
    pk = pk.merge(capi, on="package", how="left").merge(gen, on="package", how="left").merge(fi, on="package", how="left")
    pk["capi_i"] = pk.capi.eq("build-numpy").astype(int)
    pk["capi_unknown_i"] = pk.capi.isin(["no-sdist", "unknown"]).astype(int)
    pk["poetry_i"] = pk.tool.eq("poetry").astype(int)
    # in-sample in-degree (same rule as analysis_v2)
    rel = pd.read_csv(ROOT / "data" / "releases_pre2024.csv")
    use = rel[~rel.prerelease & ~rel.yanked & ~rel.missing_requires_dist & rel.frontier]
    dep = set(pk.package)
    indeg = {p: set() for p in dep}
    use = use.assign(uploaded=pd.to_datetime(use.uploaded, utc=True, format="mixed"))
    for p, g in use[use.package.isin(dep) & (use.uploaded < EVENT)].groupby("package"):
        for s in g.deps.dropna():
            for d in str(s).split(";"):
                if d in dep and d != p:
                    indeg[d].add(p)
    pk["indegree"] = pk.package.map(lambda p: len(indeg[p]))
    pk["core_i"] = (pk.indegree >= 5).astype(int)
    capped = pk[pk.ever_capped].copy()
    capped["early"] = capped.adaptation.eq("before-release")

    # bookkeeping
    out["bookkeeping"] = {
        "capped": int(len(capped)), "capped_unknown_build": int(capped.capi_unknown_i.sum()),
        "capped_known_build": int((capped.capi_unknown_i == 0).sum()),
        "still_capped": int((capped.adaptation == "still-capped").sum()),
        "early": int(capped.early.sum()), "early_compiled": int((capped.early & capped.compiled).sum()),
        "early_build_numpy": int((capped.early & (capped.capi_i == 1)).sum()),
        "early_unknown_build": int((capped.early & (capped.capi_unknown_i == 1)).sum()),
        "early_core": int((capped.early & (capped.core_i == 1)).sum()),
        "capped_core": int(capped.core_i.sum()),
    }
    # early split at rc1
    e = capped[capped.early]
    out["early_by_rc1"] = {"before_rc1": int((e.uncap < RC1).sum()),
                           "rc1_to_release": int(((e.uncap >= RC1) & (e.uncap < EVENT)).sum()),
                           "core_before_rc1": int(((e.uncap < RC1) & (e.core_i == 1)).sum()),
                           "core_rc1_to_release": int(((e.uncap >= RC1) & (e.core_i == 1)).sum())}
    # layer-specific KM
    r = cohort(pk, horizon, True)
    out["km_periphery"] = km_summary(r[r.core_i == 0].duration, r[r.core_i == 0].observed, r[r.core_i == 0].entry)
    out["km_core"] = km_summary(r[r.core_i == 1].duration, r[r.core_i == 1].observed, r[r.core_i == 1].entry)
    # core thresholds
    covs = ["capi_i", "capi_unknown_i", "poetry_i", "log_downloads", "core_i", "releases_2023_per10", "age_years_at_event"]
    out["core_threshold"] = {}
    for k in (3, 5, 10):
        rk = r.copy()
        rk["core_i"] = (rk.indegree >= k).astype(int)
        ck = capped.copy()
        ck["core_i"] = (ck.indegree >= k).astype(int)
        t = cox(rk, covs, "entry")["terms"]["core_i"]
        a = int((ck.early & (ck.core_i == 1)).sum()); b = int((ck.core_i == 1).sum())
        c_ = int((ck.early & (ck.core_i == 0)).sum()); d_ = int((ck.core_i == 0).sum())
        out["core_threshold"][str(k)] = {"n_core": int((pk.indegree >= k).sum()), "cox_HR": t["HR"], "cox_HR_95ci": t["HR_95ci"],
                                         "early_core": f"{a} of {b}", "early_periphery": f"{c_} of {d_}"}
    # Firth logistic (known build status)
    k = capped[capped.capi_unknown_i == 0]
    X = np.column_stack([np.ones(len(k)), k.capi_i, k.core_i, k.log_downloads]).astype(float)
    bta, se = firth_logit(k.early.astype(float).values, X)
    names = ["const", "capi_i", "core_i", "log_downloads"]
    out["firth"] = {"n": int(len(k)), "events": int(k.early.sum()),
                    "terms": {nm: {"OR": float(np.exp(bta[i])),
                                   "OR_95ci": [float(np.exp(bta[i] - 1.96 * se[i])), float(np.exp(bta[i] + 1.96 * se[i]))]}
                              for i, nm in enumerate(names) if nm != "const"}}
    # same Firth model with the frame-wide dependent count in place of the core indicator
    k = k.assign(log_frame_indeg=np.log10(1 + k.frame_indegree.fillna(0)))
    X = np.column_stack([np.ones(len(k)), k.capi_i, k.log_frame_indeg, k.log_downloads]).astype(float)
    bta, se = firth_logit(k.early.astype(float).values, X)
    names = ["const", "capi_i", "log_frame_indeg", "log_downloads"]
    out["firth_frame"] = {"terms": {nm: {"OR": float(np.exp(bta[i])),
                                         "OR_95ci": [float(np.exp(bta[i] - 1.96 * se[i])), float(np.exp(bta[i] + 1.96 * se[i]))]}
                                    for i, nm in enumerate(names) if nm != "const"}}
    # collinearity
    Z = r[covs].dropna().astype(float)
    Zc = np.column_stack([np.ones(len(Z)), Z.values])
    out["collinearity"] = {"corr_downloads_core": float(np.corrcoef(Z.log_downloads, Z.core_i)[0, 1]),
                           "vif": {c: float(variance_inflation_factor(Zc, i + 1)) for i, c in enumerate(covs)}}
    # GitHub difference in proportions
    g = json.loads((ROOT / "results" / "github_issues_pre2024.json").read_text(encoding="utf-8"))
    lo, hi = confint_proportions_2indep(g["capped_genuine"], g["capped_repo"], g["never_genuine"], g["never_repo"],
                                        method="newcomb")
    out["github_diff"] = {"diff": g["capped_genuine"] / g["capped_repo"] - g["never_genuine"] / g["never_repo"],
                          "ci95": [float(lo), float(hi)]}

    # ---- stepwise design comparison (planned covariates throughout) --------
    steps = []

    def add(label, frame, entry, popularity, pkx, hz, delayed):
        rr = cohort(pkx, hz, delayed)
        km = km_summary(rr.duration, rr.observed, rr.entry if delayed else None)
        t = cox(rr, PLANNED + (["below_2026_list"] if "below_2026_list" in rr and rr.below_2026_list.nunique() > 1 else []),
                "entry" if delayed else None)["terms"]["log_downloads"]
        steps.append({"step": label, "frame": frame, "events": entry, "popularity": popularity,
                      "capped": int(pkx.ever_capped.sum()), "cohort": int(len(rr)), "still_capped": int((rr.observed == 0).sum()),
                      "median": km["median"], "median_95ci": km["median_95ci"],
                      "HR_downloads": t["HR"], "HR_downloads_95ci": t["HR_95ci"]})

    p26, h26 = load("top8000", "top8000", "top8000")
    add("A planned", "2026", "all releases, release-day origin", "2026", p26, h26, False)
    p26m, _ = load("top8000", "top8000_mainline", "top8000")
    add("B + mainline", "2026", "mainline, release-day origin", "2026", p26m, h26, False)
    add("C + delayed entry", "2026", "mainline, delayed entry", "2026", p26m, h26, True)
    top26 = json.loads((ROOT / "data" / "raw" / "top-pypi-packages.json").read_text(encoding="utf-8"))
    dl26 = {canonicalize_name(r_["project"]): r_["download_count"] for r_ in top26["rows"]}
    floor = min(dl26.values())
    p24 = pk.copy()
    p24["below_2026_list"] = (~p24.package.isin(dl26)).astype(int)
    p24["log_downloads"] = np.log10(p24.package.map(dl26).fillna(floor).astype(float))
    add("D + pre-event frame", "2024", "mainline, delayed entry", "2026 (+indicator if below top 15,000)", p24, horizon, True)
    add("E + pre-event popularity", "2024", "mainline, delayed entry", "2024", pk, horizon, True)
    out["stepwise"] = steps
    out["below_2026_list_capped"] = int(p24[p24.ever_capped].below_2026_list.sum())

    (ROOT / "results" / "review4.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
