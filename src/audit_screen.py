"""Recall audit of the five-probe screening rule (main frame).

Samples packages of the pre-event frame that were NOT screened in, retrieves the
metadata of every release in the observation window, and counts packages that
were in fact NumPy-dependent (non-optional, reference environment) at some point,
and how many of those capped NumPy.

Usage:
    python src/audit_screen.py --n 300
Output:
    results/screen_audit_pre2024.json
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from classify import RAW, WINDOW_START, active_reqs, numpy_status, safe, upload_time
from collect import get_json
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
SEED = 23


def audit(name: str) -> dict:
    pkg = get_json(f"https://pypi.org/pypi/{name}/json", RAW / "pkg" / f"{safe(name)}.json")
    if not pkg:
        return {"package": name, "statuses": []}
    seq = []
    for ver, files in pkg["releases"].items():
        t = upload_time(files)
        if not t or t < WINDOW_START or any(f.get("yanked") for f in files):
            continue
        d = get_json(f"https://pypi.org/pypi/{name}/{ver}/json", RAW / "rel" / safe(name) / f"{safe(ver)}.json")
        rd = d["info"].get("requires_dist") if d else None
        if rd is not None:
            seq.append((t.isoformat(), numpy_status(active_reqs(rd))))
    seq.sort()
    return {"package": name, "statuses": [s for _, s in seq]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    meta = json.loads((ROOT / "data" / "meta_pre2024.json").read_text(encoding="utf-8"))
    cands = set(meta["candidates"])
    pool = [p for p in meta["fetched"] if p not in cands]
    sample = random.Random(SEED).sample(pool, a.n)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(audit, sample))
    dep = [r for r in res if any(s in ("capped", "compatible", "other") for s in r["statuses"])]
    capped = [r for r in dep if "capped" in r["statuses"]]

    def ci(k):
        c = binomtest(k, a.n).proportion_ci(method="exact")
        return [round(c.low * len(pool)), round(c.high * len(pool))]

    out = {"pool": len(pool), "sampled": a.n, "seed": SEED,
           "dependent": len(dep), "capped": len(capped),
           "dependent_packages": [r["package"] for r in dep], "capped_packages": [r["package"] for r in capped],
           "est_missed_dependent": round(len(dep) / a.n * len(pool)), "est_missed_dependent_95": ci(len(dep)),
           "est_missed_capped": round(len(capped) / a.n * len(pool)), "est_missed_capped_95": ci(len(capped))}
    (ROOT / "results" / "screen_audit_pre2024.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
