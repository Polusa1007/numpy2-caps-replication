"""Supplementary analyses, set 9 (all exploratory; main frame, mainline).

  * early adaptation counting only lifts between NumPy 2.0.0rc1 (30 March 2024) and the
    release, by core layer and compiled code (matches the rc1 statements in the text)
  * runtime import test (Linux): 95% CI of the difference in failure rates and in own-code
    failure rates between capped and compatible packages (Newcombe), to state which
    differences the test cannot exclude

Usage:
    python src/review9.py
Output:
    results/review9.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import confint_proportions_2indep

from review6 import prep
from review7 import fisher

ROOT = Path(__file__).resolve().parent.parent
RC1 = pd.Timestamp("2024-03-30", tz="UTC")       # GitHub release tag v2.0.0rc1


def diff_ci(a, n1, b, n2):
    lo, hi = confint_proportions_2indep(a, n1, b, n2, method="newcomb")
    return {"a": a, "n1": n1, "b": b, "n2": n2, "diff": a / n1 - b / n2, "diff_95ci": [float(lo), float(hi)]}


def main() -> None:
    pk, *_ = prep()
    c = pk[pk.ever_capped].copy()
    c["early_any"] = c.adaptation.eq("before-release")
    c["early_post_rc1"] = c.early_any & (c.uncap >= RC1)
    c["core"] = c.core_i.eq(1)
    out = {"early_post_rc1": {"n_early_any": int(c.early_any.sum()), "n_early_post_rc1": int(c.early_post_rc1.sum()),
                              "by_core": fisher(c, "core", "early_post_rc1"),
                              "by_compiled": fisher(c.assign(comp=c.compiled.astype(bool)), "comp", "early_post_rc1")}}
    r8 = json.loads((ROOT / "results" / "review8.json").read_text(encoding="utf-8"))["runtime_origin_linux"]
    rt = pd.read_csv(ROOT / "data" / "runtime_test_linux.csv")
    t = rt[rt.outcome.isin(["runtime-compatible", "runtime-incompatible"])]
    fail = t.outcome.eq("runtime-incompatible").groupby(t.declared).agg(["sum", "count"])
    out["runtime_linux_fail_diff"] = diff_ci(int(fail.loc["capped", "sum"]), int(fail.loc["capped", "count"]),
                                             int(fail.loc["compatible", "sum"]), int(fail.loc["compatible", "count"]))
    out["runtime_linux_own_diff"] = diff_ci(r8["capped"]["own_failures"], r8["capped"]["tested"],
                                            r8["compatible"]["own_failures"], r8["compatible"]["tested"])
    (ROOT / "results" / "review9.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
