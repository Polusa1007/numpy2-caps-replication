"""Exploratory robustness checks for RQ3 (NOT pre-registered).

R1: restrict to pairs whose upstream lifted its cap on/after NumPy 2.0.0.
R2: permutation baseline - shuffle upstream lift dates among all lifted
    packages (5,000 draws) to estimate the "downstream after upstream" share
    expected from the time trend alone.

Usage:
    python src/rq3_robustness.py --top 8000 [--variant _mainline]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
EVENT = pd.Timestamp("2024-06-16", tz="UTC")
DRAWS, SEED = 5000, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8000)
    ap.add_argument("--variant", default="")
    a = ap.parse_args()
    tag = f"top{a.top}{a.variant}"
    r = json.loads((ROOT / "results" / f"results_{tag}.json").read_text(encoding="utf-8"))
    p = pd.DataFrame(r["RQ3"]["pairs"])
    for c in ("up_uncap", "down_uncap"):
        p[c] = pd.to_datetime(p[c], utc=True, format="mixed")

    q = p[(p.up_uncap >= EVENT) & (p.lag_days.abs() >= 1)]
    k, n = int((q.lag_days > 0).sum()), int(len(q))
    r1 = {"n_pairs": n, "n_after": k, "share_after": k / n,
          "sign_test_p_one_sided": float(binomtest(k, n, 0.5, alternative="greater").pvalue),
          "lag_days_median": float(q.lag_days.median())}

    pk = pd.read_csv(ROOT / "data" / f"packages_{tag}.csv")
    pk["uncap"] = pd.to_datetime(pk.uncap, utc=True, format="mixed")
    times = pk[pk.ever_capped & pk.uncap.notna()].uncap.values.astype("datetime64[ns]")
    down = p.down_uncap.dt.tz_convert(None).values.astype("datetime64[ns]")
    rng = np.random.default_rng(SEED)
    null = np.empty(DRAWS)
    for i in range(DRAWS):
        lag = (down - rng.choice(times, len(down))) / np.timedelta64(1, "D")
        m = np.abs(lag) >= 1
        null[i] = (lag[m] > 0).mean()
    obs = r["RQ3"]["share_after"]
    r2 = {"observed_share": obs, "null_mean": float(null.mean()),
          "null_95_range": [float(np.quantile(null, .025)), float(np.quantile(null, .975))],
          "p_one_sided": float(((null >= obs).sum() + 1) / (DRAWS + 1)),
          "draws": DRAWS, "seed": SEED}
    top_up = p.upstream.value_counts().head(5).to_dict()

    out = {"R1_upstream_lifted_after_release": r1, "R2_permutation": r2, "top_upstreams": top_up}
    (ROOT / "results" / f"rq3_robustness_{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
