"""Supplementary analyses, set 7 (all exploratory; main frame, mainline).

  * maintenance status of the packages still capped at collection (last release,
    archived repository) and a sensitivity analysis excluding inactive ones
  * GitHub error reports vs. caps, adjusted for popularity and compiled code
  * early adaptation counting only lifts after the NumPy 2.0 beta
  * RQ3 lag analysis restricted to downstream packages all of whose capped upstreams lifted
  * bookkeeping for the binary-level C-API indicator

Usage:
    python src/review7.py
Output:
    results/review7.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import Table2x2

from classify import RAW, safe, upload_time
from review6 import BETA, EVENT, cohort, prep
from revision_analyses import km_summary

ROOT = Path(__file__).resolve().parent.parent
GH = requests.Session()
GH.headers.update({"User-Agent": "numpy2-ecosystem-study/0.1 (academic research)", "Accept": "application/vnd.github+json"})
SEED, DRAWS = 0, 5000


def archived(repo: str) -> bool | None:
    f = RAW / "github" / "repo_meta" / (repo.replace("/", "__") + ".json")
    if not f.exists():
        r = GH.get(f"https://api.github.com/repos/{repo}", timeout=60)
        data = r.json() if r.status_code == 200 else None
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(2)
    d = json.loads(f.read_text(encoding="utf-8"))
    return None if d is None else bool(d.get("archived"))


def fisher(df, col, outcome):
    tab = pd.crosstab(df[col], df[outcome]).reindex(index=[True, False], columns=[True, False], fill_value=0)
    t = Table2x2(tab.values.astype(float) + (0.5 if (tab.values == 0).any() else 0))
    return {"table": tab.values.tolist(), "OR": float(t.oddsratio), "OR_95ci": [float(x) for x in t.oddsratio_confint()],
            "p": float(fisher_exact(tab.values)[1])}


def main() -> None:
    pk, rel, use, horizon_ts = prep()
    meta = json.loads((ROOT / "data" / "meta_pre2024.json").read_text(encoding="utf-8"))
    out: dict = {}
    capped = pk[pk.ever_capped].copy()

    # ---- maintenance status of still-capped packages ------------------------------
    still = capped[capped.adaptation == "still-capped"].copy()
    rows = []
    for p in still.itertuples():
        rels = json.loads((RAW / "pkg" / f"{safe(p.package)}.json").read_text(encoding="utf-8"))["releases"]
        last = max((t for f in rels.values() if (t := upload_time(f))), default=None)
        arch = archived(p.repo) if isinstance(p.repo, str) else None
        rows.append({"package": p.package, "last_release": last, "archived": arch,
                     "days_since_last_release": (horizon_ts - pd.Timestamp(last)).days if last else None})
    st = pd.DataFrame(rows)
    st["inactive"] = (st.days_since_last_release > 365) | st.archived.fillna(False).astype(bool)
    out["still_capped"] = {"n": int(len(st)), "inactive": int(st.inactive.sum()),
                           "archived": int(st.archived.fillna(False).astype(bool).sum()),
                           "no_release_12m": int((st.days_since_last_release > 365).sum()),
                           "released_last_12m": int((st.days_since_last_release <= 365).sum()),
                           "packages": st.assign(last_release=st.last_release.astype(str)).to_dict("records")}
    r = cohort(pk, horizon_ts)
    rr = r[~r.package.isin(st[st.inactive].package)]
    out["km_excluding_inactive_still_capped"] = km_summary(rr.duration, rr.observed, rr.entry)

    # ---- GitHub error reports adjusted for popularity and compiled code ---------
    sig = pd.read_csv(ROOT / "data" / "github_signatures_pre2024.csv")
    s = sig[sig.searchable].merge(pk[["package", "log_downloads"]], on="package", how="left")
    s["capped_i"] = s.capped.astype(int)
    s["compiled_i"] = s.compiled.astype(int)
    s["reported_i"] = s.reported.astype(int)
    X = sm.add_constant(s[["capped_i", "log_downloads", "compiled_i"]].astype(float))
    m = sm.Logit(s.reported_i, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(s.repo)[0]})
    ci = np.exp(m.conf_int())
    out["github_adjusted"] = {c: {"OR": float(np.exp(m.params[c])), "OR_95ci": [float(ci.loc[c, 0]), float(ci.loc[c, 1])],
                                  "p": float(m.pvalues[c])} for c in ["capped_i", "log_downloads", "compiled_i"]}
    out["github_adjusted"]["n"] = int(len(s))

    # ---- early adaptation counting only lifts after the beta -------------------------
    c2 = capped.copy()
    c2["early_any"] = c2.adaptation.eq("before-release")
    c2["early_post_beta"] = c2.early_any & (c2.uncap >= BETA)
    c2["core"] = c2.core_i.eq(1)
    out["early_post_beta"] = {"n_early_any": int(c2.early_any.sum()), "n_early_post_beta": int(c2.early_post_beta.sum()),
                              "by_core": fisher(c2, "core", "early_post_beta"),
                              "by_compiled": fisher(c2.assign(comp=c2.compiled.astype(bool)), "comp", "early_post_beta")}

    # ---- RQ3 lag restricted to downstreams whose capped upstreams all lifted ------
    cs = capped.set_index("package")
    allp = []
    for dn, row in cs.iterrows():
        for up in str(row.capped_deps).split(";"):
            if up in cs.index and up != dn:
                allp.append({"down": dn, "up": up, "d": row.uncap, "u": cs.loc[up, "uncap"]})
    A = pd.DataFrame(allp)
    full = A.groupby("down").apply(lambda g: g.u.notna().all() and pd.notna(g.d.iloc[0]), include_groups=False)
    eligible = full[full].index
    lastup = A[A.down.isin(eligible)].groupby("down").u.max()
    lag = (cs.loc[lastup.index, "uncap"] - lastup).dt.total_seconds() / 86400
    lifted = capped[capped.uncap.notna()].copy()
    lifted["stratum"] = np.where(lifted.core_i == 1, "core", "periphery") + "|" + lifted.dl_quartile.astype(str)
    pool = {k: g.uncap.values for k, g in lifted.groupby("stratum")}
    strat = lifted.set_index("package").stratum
    rng = np.random.default_rng(SEED)
    lu = lastup.dt.tz_convert(None).values.astype("datetime64[ns]")
    null30 = np.empty(DRAWS)
    for i in range(DRAWS):
        dr = np.array([rng.choice(pool[strat[x]]) for x in lastup.index]).astype("datetime64[ns]")
        lg = (dr - lu) / np.timedelta64(1, "D")
        null30[i] = np.mean((lg >= 0) & (lg <= 30))
    obs30 = float(((lag >= 0) & (lag <= 30)).mean())
    out["rq3_lag_all_upstreams_lifted"] = {
        "downstreams_with_capped_upstream": int(A.down.nunique()), "eligible": int(len(eligible)),
        "share_within_30d": obs30, "null_mean": float(null30.mean()),
        "null_95": [float(np.quantile(null30, .025)), float(np.quantile(null30, .975))],
        "p": float(((null30 >= obs30).sum() + 1) / (DRAWS + 1)), "lag_median": float(lag.median())}

    # ---- binary indicator bookkeeping -----------------------------------------------
    w = pd.read_csv(ROOT / "data" / "wheel_capi_pre2024.csv")
    miss = w[~w.binary_capi.isin(["uses-capi", "no-capi"])]
    out["binary_missing"] = {"n": int(len(miss)), "by_reason": miss.binary_capi.value_counts().to_dict(),
                             "capped_among_missing": int(miss.package.isin(capped.package).sum())}

    (ROOT / "results" / "review7.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "still_capped"}, indent=1, default=str))
    print(json.dumps({k: v for k, v in out["still_capped"].items() if k != "packages"}, indent=1))


if __name__ == "__main__":
    main()
