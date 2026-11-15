"""Frame-wide reverse-dependency counts (robustness for the core-layer definition).

For every package in the pre-event frame (top 8,000 on 2024-06-01), take its last
stable, non-yanked release uploaded before 2024-06-16 and its non-optional
dependencies (reference environment). The in-degree of a package is the number
of frame packages that depend on it. Unlike the main core definition, this does
not restrict dependents to NumPy-dependent packages.

Usage:
    python src/frame_indegree.py
Output:
    data/frame_indegree_pre2024.csv
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from classify import active_reqs, upload_time
from collect import RAW, get_json, safe

ROOT = Path(__file__).resolve().parent.parent
EVENT = datetime(2024, 6, 16, tzinfo=timezone.utc)


def deps_before_event(name: str) -> tuple[str, list[str] | None]:
    d = get_json(f"https://pypi.org/pypi/{name}/json", RAW / "pkg" / f"{safe(name)}.json")
    if not d:
        return name, None
    best = None
    for ver, files in d["releases"].items():
        t = upload_time(files)
        if t is None or t >= EVENT or any(f.get("yanked") for f in files):
            continue
        try:
            if Version(ver).is_prerelease:
                continue
        except InvalidVersion:
            continue
        if best is None or t > best[0]:
            best = (t, ver)
    if best is None:
        return name, None
    r = get_json(f"https://pypi.org/pypi/{name}/{best[1]}/json", RAW / "rel" / safe(name) / f"{safe(best[1])}.json")
    rd = r["info"].get("requires_dist") if r else None
    if rd is None:
        return name, None
    return name, sorted({canonicalize_name(q.name) for q in active_reqs(rd)})


def main() -> None:
    meta = json.loads((ROOT / "data" / "meta_pre2024.json").read_text(encoding="utf-8"))
    frame = [canonicalize_name(n) for n in meta["frame"]]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(deps_before_event, meta["frame"]))
    fs = set(frame)
    indeg = {p: 0 for p in frame}
    known = 0
    for name, deps in res:
        if deps is None:
            continue
        known += 1
        for d in deps:
            if d in fs and d != canonicalize_name(name):
                indeg[d] += 1
    df = pd.DataFrame({"package": list(indeg), "frame_indegree": list(indeg.values())})
    df.to_csv(ROOT / "data" / "frame_indegree_pre2024.csv", index=False)
    print(f"frame packages with pre-event dependency metadata: {known}/{len(frame)}")
    print(df.sort_values("frame_indegree", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
