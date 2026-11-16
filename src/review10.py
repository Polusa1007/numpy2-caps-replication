"""Supplementary numbers, set 10 (main frame, mainline).

  * share of the survival cohort that had declared compatibility by the end of the
    GitHub search window (31 December 2025), from the delayed-entry Kaplan-Meier
    estimate of the main analysis (reproduces its one-year value as a check)
  * length of follow-up from the release to the collection date, in years

Usage:
    python src/review10.py
Output:
    results/review10.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from review6 import prep

ROOT = Path(__file__).resolve().parent.parent
EVENT = pd.Timestamp("2024-06-16", tz="UTC")
GH_END = pd.Timestamp("2025-12-31", tz="UTC")     # last day of the GitHub search window


def main() -> None:
    pk, _, _, horizon_ts = prep()
    horizon = (horizon_ts - EVENT).days
    r = pk[pk.ever_capped & pk.adaptation.isin(["after-release", "still-capped"])].copy()
    r["duration"] = ((r.uncap - EVENT).dt.total_seconds() / 86400).fillna(horizon).clip(lower=0.5)
    r["observed"] = r.uncap.notna().astype(int)
    first = ((r.first_cap - EVENT).dt.total_seconds() / 86400).clip(lower=0)
    r["entry"] = np.minimum(first, r.duration - 0.25)
    km = KaplanMeierFitter().fit(r.duration, r.observed, entry=r.entry)
    main = json.loads((ROOT / "results" / "main_pre2024.json").read_text(encoding="utf-8"))
    ref = main["RQ2"]["km_delayed_entry"]
    one_year = 1 - float(km.survival_function_at_times(365).iloc[0])
    assert abs(one_year - ref["adapted_at"]["365d"]) < 1e-6, (one_year, ref["adapted_at"]["365d"])
    gh_day = (GH_END - EVENT).days
    out = {"n": int(len(r)),
           "adapted_at_365d_check": one_year,
           "github_window_end": GH_END.date().isoformat(),
           "github_window_end_day": gh_day,
           "adapted_at_github_window_end": 1 - float(km.survival_function_at_times(gh_day).iloc[0]),
           "followup_years": (horizon_ts - EVENT).days / 365.25}
    (ROOT / "results" / "review10.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
