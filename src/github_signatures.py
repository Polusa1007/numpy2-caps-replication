"""Objective cross-check of declared vs. reported NumPy 2 problems via error signatures.

For every NumPy-dependent package of the main frame with a GitHub repository, we
search its issues created between 2024-03-01 and 2025-12-31 for error messages
that NumPy 2 itself produces:
  ABI  "cannot be run in NumPy 2"                 (A module that was compiled using NumPy 1.x ...)
  ABI  "_ARRAY_API not found"
  ABI  "numpy.core.multiarray failed to import"
  ABI  "numpy.dtype size changed"                  (binary incompatibility)
  API  "was removed in the NumPy 2.0 release"      (e.g. np.float_, np.NaN)
No manual judgement is involved. Unauthenticated GitHub search (<= 10 requests/min).

Usage:
    python src/github_signatures.py
Outputs:
    data/github_signatures_pre2024.csv, results/github_signatures_pre2024.json
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.proportion import confint_proportions_2indep


ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "github" / "signatures"
TAG = "pre2024_mainline"
LO, HI = "2024-03-01", "2025-12-31"
SIGS = {
    "cannot be run in NumPy 2": "ABI",
    "_ARRAY_API not found": "ABI",
    "numpy.core.multiarray failed to import": "ABI",
    "numpy.dtype size changed": "ABI",
    "was removed in the NumPy 2.0 release": "API",
}
PYPI_CACHE = ROOT / "data" / "raw" / "github" / "pypi"
S = requests.Session()
S.headers.update({"User-Agent": "numpy2-ecosystem-study/0.1 (academic research)",
                  "Accept": "application/vnd.github+json"})


def repo_of(pkg: str) -> str | None:
    f = PYPI_CACHE / f"{pkg}.json"
    if not f.exists():
        r = S.get(f"https://pypi.org/pypi/{pkg}/json", timeout=60)
        r.raise_for_status()
        info = r.json()["info"]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"project_urls": info.get("project_urls"), "home_page": info.get("home_page")}),
                     encoding="utf-8")
    d = json.loads(f.read_text(encoding="utf-8"))
    urls = list((d.get("project_urls") or {}).values()) + [d.get("home_page") or ""]
    cands = []
    for u in urls:
        m = re.search(r"github\.com/([\w.-]+)/([\w.-]+)", u or "")
        if m and m.group(1).lower() not in ("sponsors", "orgs"):
            cands.append(f"{m.group(1)}/{m.group(2).removesuffix('.git')}")
    if not cands:
        return None
    # prefer the URL whose repo name resembles the package name
    key = re.sub(r"[-_.]", "", pkg.lower())
    cands.sort(key=lambda c: re.sub(r"[-_.]", "", c.split("/")[1].lower()) != key)
    return cands[0]


def search(repo: str) -> list[dict] | None:
    f = CACHE / (repo.replace("/", "__") + ".json")
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    phrases = " OR ".join(f'"{s}"' for s in SIGS)
    q = f"repo:{repo} is:issue created:{LO}..{HI} ({phrases})"
    items: list[dict] | None = None
    for attempt in range(12):
        try:
            r = S.get("https://api.github.com/search/issues", params={"q": q, "per_page": 100}, timeout=60)
        except requests.RequestException:     # network outage: wait and retry
            time.sleep(60)
            continue
        if r.status_code in (403, 429):
            time.sleep(30 * min(attempt + 1, 4))
            continue
        if r.status_code == 422:          # repository missing or not searchable
            items = None
            break
        r.raise_for_status()
        items = [{"number": i["number"], "title": i["title"], "created_at": i["created_at"],
                  "matched": [s for s in SIGS if s.lower() in ((i["title"] or "") + " " + (i.get("body") or "")).lower()]}
                 for i in r.json()["items"]]
        break
    else:
        raise RuntimeError(f"GitHub search failed repeatedly for {repo}; not cached, rerun to resume")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(items), encoding="utf-8")
    time.sleep(7)
    return items


def main() -> None:
    pk = pd.read_csv(ROOT / "data" / f"packages_{TAG}.csv")
    for c in ("first_cap", "uncap"):
        pk[c] = pd.to_datetime(pk[c], utc=True, format="mixed")
    rows = []
    for i, p in enumerate(pk.itertuples(), 1):
        try:
            repo = repo_of(p.package)
        except Exception:  # noqa: BLE001
            repo = None
        items = search(repo) if repo else None
        # the search engine also matches on tokens; keep only issues whose text contains a phrase
        hits = [x for x in (items or []) if x["matched"]]
        first = min((x["created_at"] for x in hits), default=None)
        kinds = sorted({SIGS[s] for x in hits for s in x["matched"]})
        rows.append({"package": p.package, "capped": bool(p.ever_capped), "cap_timing": p.cap_timing,
                     "first_cap": p.first_cap, "uncap": p.uncap, "compiled": bool(p.compiled),
                     "repo": repo, "searchable": items is not None, "n_reports": len(hits),
                     "first_report": first, "kinds": ";".join(kinds)})
        if i % 50 == 0:
            print(f"{i}/{len(pk)}", flush=True)
    df = pd.DataFrame(rows)
    df["first_report"] = pd.to_datetime(df.first_report, utc=True)
    df["reported"] = df.n_reports > 0
    df.to_csv(ROOT / "data" / "github_signatures_pre2024.csv", index=False)

    s = df[df.searchable]
    c, nc = s[s.capped], s[~s.capped]
    tab = [[int(c.reported.sum()), int((~c.reported).sum())], [int(nc.reported.sum()), int((~nc.reported).sum())]]
    t2 = Table2x2([[x + 0.5 for x in r] for r in tab]) if 0 in sum(tab, []) else Table2x2(tab)
    lo, hi = confint_proportions_2indep(tab[0][0], len(c), tab[1][0], len(nc), method="newcomb")
    react = c[(c.cap_timing == "reactive") & c.reported]
    out = {
        "packages": int(len(df)), "with_repo": int(df.repo.notna().sum()), "searchable": int(len(s)),
        "capped_searchable": int(len(c)), "never_capped_searchable": int(len(nc)),
        "capped_reported": tab[0][0], "never_capped_reported": tab[1][0],
        "share_capped": tab[0][0] / len(c), "share_never": tab[1][0] / len(nc),
        "diff": tab[0][0] / len(c) - tab[1][0] / len(nc), "diff_95ci": [float(lo), float(hi)],
        "OR": float(t2.oddsratio), "OR_95ci": [float(x) for x in t2.oddsratio_confint()],
        "fisher_p": float(fisher_exact(tab)[1]),
        "kinds_never_capped": nc[nc.reported].kinds.value_counts().to_dict(),
        "kinds_capped": c[c.reported].kinds.value_counts().to_dict(),
        "never_capped_reported_compiled": int((nc.reported & nc.compiled).sum()),
        "reactive_reported": int(len(react)),
        "reactive_report_before_cap": int((react.first_report <= react.first_cap).sum()),
        "window": [LO, HI], "signatures": SIGS,
    }
    (ROOT / "results" / "github_signatures_pre2024.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
