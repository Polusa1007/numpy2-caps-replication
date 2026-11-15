"""Supplementary analyses, set 8 (all exploratory; main frame, mainline).

  * GitHub error reports restricted to NumPy-2-specific strings: the ABI messages
    "numpy.dtype size changed" and "numpy.core.multiarray failed to import" also occur
    for older ABI mismatches, so only "cannot be run in NumPy 2", "_ARRAY_API not found"
    and "was removed in the NumPy 2.0 release" are kept (re-uses the cached search results)
  * runtime import test: origin of each failure (the package's own code vs. a dependency),
    decided without executing anything:
      - pure-Python release (only py3-none-any wheels / sdist) failing with an ABI error or a
        dependency's version check -> the failure originates in a dependency
      - API-removal error -> own code if the removed name occurs in the release's own
        source distribution (static text search, nothing is installed or run)
      - release with own compiled extensions failing with an ABI error -> own code
    plus attrition (not-testable) by declared status
  * Holm adjustment within the RQ1 cap-rate tests and within the early-adaptation tests

Usage:
    python src/review8.py
Output:
    results/review8.json
"""
from __future__ import annotations

import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from classify import RAW, safe

ROOT = Path(__file__).resolve().parent.parent
SIGCACHE = RAW / "github" / "signatures"
STRICT = {"cannot be run in NumPy 2": "ABI", "_ARRAY_API not found": "ABI",
          "was removed in the NumPy 2.0 release": "API"}
ORIGIN_CACHE = RAW / "runtime_origin.json"
REMOVED = re.compile(r"`?np\.(\w+)`? was removed in the NumPy 2\.0 release")


def or_table(tab):
    t = Table2x2([[x + 0.5 for x in r] for r in tab]) if 0 in sum(tab, []) else Table2x2(tab)
    return float(t.oddsratio), [float(x) for x in t.oddsratio_confint()]


def github_strict(pk: pd.DataFrame) -> dict:
    sig = pd.read_csv(ROOT / "data" / "github_signatures_pre2024.csv")
    s = sig[sig.searchable].copy()
    hits = []
    for repo in s.repo:
        items = json.loads((SIGCACHE / (repo.replace("/", "__") + ".json")).read_text(encoding="utf-8")) or []
        m = [x for x in items if any(k in STRICT for k in x["matched"])]
        hits.append((len(m), sorted({STRICT[k] for x in m for k in x["matched"] if k in STRICT})))
    s["n_strict"] = [h[0] for h in hits]
    s["kinds_strict"] = [";".join(h[1]) for h in hits]
    s["reported_strict"] = s.n_strict > 0
    c, nc = s[s.capped], s[~s.capped]
    tab = [[int(c.reported_strict.sum()), int((~c.reported_strict).sum())],
           [int(nc.reported_strict.sum()), int((~nc.reported_strict).sum())]]
    lo, hi = confint_proportions_2indep(tab[0][0], len(c), tab[1][0], len(nc), method="newcomb")
    orr, orci = or_table(tab)
    s = s.merge(pk[["package", "log_downloads"]], on="package", how="left")
    X = sm.add_constant(pd.DataFrame({"capped": s.capped.astype(float), "log_downloads": s.log_downloads,
                                      "compiled": s.compiled.astype(float)}))
    m = sm.Logit(s.reported_strict.astype(int), X).fit(disp=0, cov_type="cluster",
                                                       cov_kwds={"groups": pd.factorize(s.repo)[0]})
    nr = nc[nc.reported_strict]
    return {
        "signatures": STRICT, "searchable": int(len(s)), "capped_n": int(len(c)), "never_n": int(len(nc)),
        "capped_reported": tab[0][0], "never_reported": tab[1][0],
        "share_capped": tab[0][0] / len(c), "share_never": tab[1][0] / len(nc),
        "diff": tab[0][0] / len(c) - tab[1][0] / len(nc), "diff_95ci": [float(lo), float(hi)],
        "OR": orr, "OR_95ci": orci, "fisher_p": float(fisher_exact(tab)[1]),
        "adj_OR": float(np.exp(m.params["capped"])),
        "adj_OR_95ci": [float(x) for x in np.exp(m.conf_int().loc["capped"])],
        "never_reported_abi": int(nr.kinds_strict.str.contains("ABI").sum()),
        "never_reported_api": int(nr.kinds_strict.str.contains("API").sum()),
        "dropped_by_strict": int((s.reported & ~s.reported_strict).sum()),
        "dropped_by_strict_never": int((s.reported & ~s.reported_strict & ~s.capped).sum()),
    }


def sdist_contains(pkg: str, ver: str, name: str) -> bool | None:
    d = requests.get(f"https://pypi.org/pypi/{pkg}/{ver}/json", timeout=60).json()
    sd = [u for u in d["urls"] if u["packagetype"] == "sdist"]
    if not sd:
        return None
    data = requests.get(sd[0]["url"], timeout=300).content       # read in memory, never extracted to disk
    pat = re.compile(rf"\bnp\.{re.escape(name)}\b|numpy\.{re.escape(name)}\b|\b{re.escape(name)}\b")
    texts = []
    if sd[0]["filename"].endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(data))
        texts = [z.read(n).decode("utf-8", "replace") for n in z.namelist()
                 if n.endswith(".py") and "/test" not in n]
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".py") and "/test" not in m.name:
                    texts.append(tf.extractfile(m).read().decode("utf-8", "replace"))
    return any(pat.search(t) for t in texts)


def runtime_origin(suffix: str = "") -> dict:
    rt = pd.read_csv(ROOT / "data" / f"runtime_test{suffix}.csv")
    cache = json.loads(ORIGIN_CACHE.read_text(encoding="utf-8")) if ORIGIN_CACHE.exists() else {}
    rows = []
    for r in rt[rt.outcome == "runtime-incompatible"].itertuples():
        _x = eval(r.np2)                                   # noqa: S307 - our own CSV, written by runtime_test.py
        err = " ".join((_x.get("failed") or {}).values()) or str(_x.get("detail", ""))
        files = json.loads((RAW / "pkg" / f"{safe(r.package)}.json").read_text(encoding="utf-8"))["releases"][r.version]
        names = [f["filename"] for f in files]
        pure = all(n.endswith(("none-any.whl", ".tar.gz", ".zip")) for n in names)
        m = REMOVED.search(err)
        if m:
            kind = "API"
            key = f"{r.package}=={r.version}:{m.group(1)}"
            if key not in cache:
                cache[key] = sdist_contains(r.package, r.version, m.group(1))
            origin = "own" if cache[key] else ("dependency" if cache[key] is False else "unknown")
        elif "dtype size changed" in err or "multiarray failed to import" in err or "_ARRAY_API" in err \
                or "cannot be run in NumPy 2" in err:
            kind = "ABI"
            origin = "dependency" if pure else "own"
        else:
            kind = "other"
            origin = "dependency" if pure else "unknown"
        rows.append({"package": r.package, "version": r.version, "declared": r.declared, "kind": kind,
                     "pure_python": pure, "origin": origin})
    ORIGIN_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    o = pd.DataFrame(rows)
    out = {"failures": rows}
    for d, g in rt.groupby("declared"):
        tested = int(g.outcome.isin(["runtime-compatible", "runtime-incompatible"]).sum())
        own = int(((o.declared == d) & (o.origin == "own")).sum())
        dep = int(((o.declared == d) & (o.origin == "dependency")).sum())
        ci = proportion_confint(own, tested, method="wilson")
        out[d] = {"sampled": int(len(g)), "not_testable": int((g.outcome == "not-testable").sum()),
                  "tested": tested, "own_failures": own, "dependency_failures": dep,
                  "own_share": own / tested, "own_95ci": [float(x) for x in ci]}
    nt = [[out["capped"]["not_testable"], out["capped"]["sampled"] - out["capped"]["not_testable"]],
          [out["compatible"]["not_testable"], out["compatible"]["sampled"] - out["compatible"]["not_testable"]]]
    out["attrition_fisher_p"] = float(fisher_exact(nt)[1])
    own = [[out["capped"]["own_failures"], out["capped"]["tested"] - out["capped"]["own_failures"]],
           [out["compatible"]["own_failures"], out["compatible"]["tested"] - out["compatible"]["own_failures"]]]
    out["own_fisher_p"] = float(fisher_exact(own)[1])
    out["kind_by_origin"] = o.groupby(["kind", "origin"]).size().rename("n").reset_index().to_dict("records")
    return out


def holm_families() -> dict:
    m = json.loads((ROOT / "results" / "main_pre2024.json").read_text(encoding="utf-8"))
    b = json.loads((ROOT / "results" / "review6.json").read_text(encoding="utf-8"))["binary_exposure"]
    fam = {
        "rq1_cap_rate": {"build": m["RQ1"]["by_capi_vs_rest"]["p"], "compiled": m["RQ1"]["by_compiled"]["p"],
                         "core": m["RQ1"]["by_core"]["p"], "binary": b["cap_rate_by_binary"]["p"]},
        "early_adaptation": {"build": m["RQ2"]["early"]["by_capi"]["p"], "compiled": m["RQ2"]["early"]["by_compiled"]["p"],
                             "core": m["RQ2"]["early"]["by_core"]["p"], "binary": b["early_by_binary"]["p"]},
    }
    out = {}
    for f, ps in fam.items():
        adj = multipletests(list(ps.values()), method="holm")[1]
        out[f] = {k: {"p": float(p), "p_holm": float(a)} for (k, p), a in zip(ps.items(), adj)}
    return out


def main() -> None:
    pk = pd.read_csv(ROOT / "data" / "packages_pre2024_mainline.csv")
    pk["log_downloads"] = np.log10(pk.downloads_30d.astype(float))
    out = {"github_strict": github_strict(pk), "runtime_origin": runtime_origin(), "holm": holm_families()}
    if (ROOT / "data" / "runtime_test_linux.csv").exists():
        out["runtime_origin_linux"] = runtime_origin("_linux")
        w = pd.read_csv(ROOT / "data" / "runtime_test.csv").set_index("package").outcome
        lx = pd.read_csv(ROOT / "data" / "runtime_test_linux.csv").set_index("package").outcome
        both = w.index.intersection(lx.index)
        cmp = pd.crosstab(w[both], lx[both])
        tested_both = [x for x in both if w[x] != "not-testable" and lx[x] != "not-testable"]
        out["windows_vs_linux"] = {"overlap": int(len(both)), "tested_both": len(tested_both),
                                   "agree": int(sum(w[x] == lx[x] for x in tested_both)),
                                   "crosstab": {i: {c: int(v) for c, v in r.items()} for i, r in cmp.iterrows()},
                                   "disagree": [{"package": x, "windows": w[x], "linux": lx[x]}
                                                for x in tested_both if w[x] != lx[x]]}
    # sampling population of the runtime test: latest pre-event mainline release in the window
    rel = pd.read_csv(ROOT / "data" / "releases_pre2024.csv")
    rel["uploaded"] = pd.to_datetime(rel.uploaded, utc=True, format="mixed")
    use = rel[~rel.prerelease & ~rel.yanked & ~rel.missing_requires_dist & rel.frontier
              & (rel.uploaded < pd.Timestamp("2024-06-16", tz="UTC")) & rel.package.isin(pk.package)]
    last = use.sort_values("uploaded").groupby("package").tail(1)
    out["runtime_population"] = {"packages": int(len(pk)), "with_pre_event_release": int(len(last)),
                                 **{k: int(v) for k, v in last.numpy_status.value_counts().items()}}
    (ROOT / "results" / "review8.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
