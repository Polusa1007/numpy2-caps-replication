"""Audit the D5 pre-filter (revision R-minor).

Packages entered stage 2 only if their *latest* release mentions numpy. This
audit samples packages that FAILED the pre-filter, fetches every in-window
release's metadata, and counts how many were in fact NumPy-dependent, capped,
or ended a cap by dropping NumPy.

Usage:
    python src/audit_prefilter.py --top 8000 --n 300
Output:
    results/prefilter_audit_top<N>.json
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from classify import RAW, WINDOW_START, active_reqs, numpy_status, safe, upload_time
from collect import get_json

ROOT = Path(__file__).resolve().parent.parent
SEED = 7


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
            seq.append((t.isoformat(), ver, numpy_status(active_reqs(rd))))
    seq.sort()
    return {"package": name, "statuses": [s for _, _, s in seq], "versions": [v for _, v, _ in seq]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8000)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    meta = json.loads((ROOT / "data" / f"meta_top{a.top}.json").read_text(encoding="utf-8"))
    cands = set(meta["candidates"])
    pool = [p for p in meta["frame"] if p not in cands]
    sample = random.Random(SEED).sample(pool, a.n)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(audit, sample))
    dep = [r for r in res if any(s in ("capped", "compatible", "other") for s in r["statuses"])]
    capped = [r for r in dep if "capped" in r["statuses"]]
    dropped = [r for r in capped
               if r["statuses"][-1] == "none"]
    out = {
        "pool_size": len(pool), "sampled": len(sample), "seed": SEED,
        "numpy_dependent_in_window": len(dep),
        "capped_in_window": len(capped),
        "capped_then_dropped_numpy": len(dropped),
        "dependent_packages": [r["package"] for r in dep],
        "capped_packages": [r["package"] for r in capped],
        "share_dependent": len(dep) / len(sample),
        "estimated_missed_dependents_in_pool": round(len(dep) / len(sample) * len(pool)),
        "estimated_missed_capped_in_pool": round(len(capped) / len(sample) * len(pool)),
    }
    (ROOT / "results" / f"prefilter_audit_top{a.top}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
