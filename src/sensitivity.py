"""Exploratory sensitivity analyses (NOT pre-registered; label as such in the paper).

S1: drop "late" caps first added > 365 days after NumPy 2.0.0.
S2: drop packages first released after NumPy 2.0.0 (born into the NumPy 2 era).
S3: S1 + S2 combined.
Each re-runs the RQ2 Kaplan-Meier median and the Cox model.

Usage:
    python src/sensitivity.py --top 8000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import median_survival_times
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
EVENT = pd.Timestamp("2024-06-16", tz="UTC")
COVS = ["compiled_i", "log_downloads", "releases_2023", "age_years_at_event"]


def cohort(pk: pd.DataFrame, horizon: int) -> pd.DataFrame:
    s = pk[pk.ever_capped & pk.adaptation.isin(["after-release", "still-capped"])].copy()
    s["duration"] = s.days_event_to_uncap.fillna(horizon).clip(lower=0.5)
    s["observed"] = s.adaptation.eq("after-release").astype(int)
    s["log_downloads"] = np.log10(s.downloads_30d.astype(float))
    s["compiled_i"] = s.compiled.astype(int)
    return s


def fit(s: pd.DataFrame) -> dict:
    kmf = KaplanMeierFitter().fit(s.duration, s.observed)
    ci = median_survival_times(kmf.confidence_interval_)
    d = s[["duration", "observed"] + COVS].dropna()
    cph = CoxPHFitter().fit(d, "duration", "observed")
    sm = cph.summary
    holm = multipletests(sm["p"].values, method="holm")[1]
    return {
        "n": int(len(s)), "events": int(s.observed.sum()),
        "km_median_days": float(kmf.median_survival_time_),
        "km_median_95ci": [float(ci.iloc[0, 0]), float(ci.iloc[0, 1])],
        "cox": {c: {"HR": round(float(sm.loc[c, "exp(coef)"]), 3),
                    "HR_95ci": [round(float(sm.loc[c, "exp(coef) lower 95%"]), 3),
                                round(float(sm.loc[c, "exp(coef) upper 95%"]), 3)],
                    "p_holm": float(h)} for c, h in zip(sm.index, holm)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8000)
    ap.add_argument("--variant", default="")
    args = ap.parse_args()
    pk = pd.read_csv(ROOT / "data" / f"packages_top{args.top}{args.variant}.csv")
    for c in ("first_cap", "first_release"):
        pk[c] = pd.to_datetime(pk[c], utc=True, format="mixed")
    meta = json.loads((ROOT / "data" / f"meta_top{args.top}.json").read_text(encoding="utf-8"))
    horizon = (pd.Timestamp(meta["collected_at"]) - EVENT).days

    late = (pk.cap_timing == "reactive") & ((pk.first_cap - EVENT).dt.days > 365)
    born = pk.first_release > EVENT
    variants = {
        "primary": pk,
        "S1_drop_late_caps": pk[~late],
        "S2_drop_born_after_event": pk[~born],
        "S3_drop_both": pk[~late & ~born],
        # D9: only caps written as an explicit major-version bound (<2, ==1.*).
        "S4_major_cap_style_only": pk[~pk.ever_capped | (pk.cap_style == "major-cap")],
    }
    out = {name: fit(cohort(df, horizon)) for name, df in variants.items()}
    out["_excluded"] = {"late_caps": sorted(pk[late].package),
                        "born_after_event_and_capped": sorted(pk[born & pk.ever_capped].package)}
    path = ROOT / "results" / f"sensitivity_top{args.top}{args.variant}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for name in variants:
        r = out[name]
        dl = r["cox"]["log_downloads"]
        print(f"{name:26s} n={r['n']:3d} ev={r['events']:3d}  median={r['km_median_days']:.0f}d "
              f"[{r['km_median_95ci'][0]:.0f},{r['km_median_95ci'][1]:.0f}]  "
              f"HR_downloads={dl['HR']} {dl['HR_95ci']} p_holm={dl['p_holm']:.2g}  "
              + "  ".join(f"{c}:p={v['p_holm']:.2g}" for c, v in r["cox"].items() if c != "log_downloads"))


if __name__ == "__main__":
    main()
