"""Collection for the main (v2) analysis: pre-event frame, no latest-release pre-filter.

Frame: top 8,000 packages by 30-day downloads in the hugovk/top-pypi-packages
snapshot of 2024-06-01 (commit 344411331e24), i.e. defined BEFORE NumPy 2.0.0.
Downloads in that snapshot are pre-event popularity.

Screening replaces deviation D5. A package is crawled in full if ANY of these
probe releases mentions numpy in requires_dist:
  latest release; first in-window release; last release before 2024-06-16;
  first release on/after 2024-06-16; last release before 2025-06-16.

Usage:
    python src/collect_v2.py
Writes data/meta_pre2024.json (plus the shared raw cache).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from collect import RAW, WINDOW_START, get_json, mentions_numpy, release_upload_time, safe

ROOT = Path(__file__).resolve().parent.parent
FRAME = RAW / "frames" / "2024-06-01_top-pypi-packages-30-days.min.json"
EVENT = datetime(2024, 6, 16, tzinfo=timezone.utc)
YEAR1 = datetime(2025, 6, 16, tzinfo=timezone.utc)
TAG = "pre2024"
WORKERS = 8


def pkg_json(name: str):
    return get_json(f"https://pypi.org/pypi/{name}/json", RAW / "pkg" / f"{safe(name)}.json")


def rel_json(name: str, ver: str):
    return get_json(f"https://pypi.org/pypi/{name}/{ver}/json", RAW / "rel" / safe(name) / f"{safe(ver)}.json")


def probes(releases: dict) -> list[str]:
    times = sorted((t, v) for v, f in releases.items() if (t := release_upload_time(f)))
    win = [(t, v) for t, v in times if t >= WINDOW_START]
    pre = [(t, v) for t, v in times if WINDOW_START <= t < EVENT]
    post = [(t, v) for t, v in times if t >= EVENT]
    y1 = [(t, v) for t, v in times if WINDOW_START <= t < YEAR1]
    pick = []
    if win:
        pick.append(win[0][1])
    if pre:
        pick.append(pre[-1][1])
    if post:
        pick.append(post[0][1])
    if y1:
        pick.append(y1[-1][1])
    return list(dict.fromkeys(pick))


def screen(name: str) -> tuple[str, bool]:
    d = pkg_json(name)
    if not d:
        return name, False
    if mentions_numpy(d["info"].get("requires_dist")):
        return name, True
    for v in probes(d["releases"]):
        r = rel_json(name, v)
        if r and mentions_numpy(r["info"].get("requires_dist")):
            return name, True
    return name, False


def main() -> None:
    frame = json.loads(FRAME.read_text(encoding="utf-8"))
    rows = frame["rows"][:8000]
    names = [r["project"] for r in rows]
    print(f"Frame snapshot {frame['last_update']}: {len(names)} packages")

    with ThreadPoolExecutor(WORKERS) as ex:
        res = list(ex.map(screen, names))
    fetched = [n for n in names if pkg_json(n)]
    cands = [n for n, ok in res if ok]
    print(f"metadata retrieved: {len(fetched)}; screened in (numpy in any probe): {len(cands)}")

    jobs = []
    for n in cands:
        for v, files in pkg_json(n)["releases"].items():
            t = release_upload_time(files)
            if t and t >= WINDOW_START:
                jobs.append((n, v))
    print(f"per-release requests: {len(jobs)}")
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, _ in enumerate(ex.map(lambda nv: rel_json(*nv), jobs), 1):
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    meta = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "top_snapshot": frame["last_update"],
        "frame_source": "hugovk/top-pypi-packages@344411331e24 top-pypi-packages-30-days.min.json",
        "top_n": len(names),
        "frame": names,
        "fetched": fetched,
        "downloads": {r["project"]: r["download_count"] for r in rows},
        "candidates": cands,
        "screen": "latest + first-in-window + last-pre-event + first-post-event + last-before-2025-06-16",
    }
    (ROOT / "data" / f"meta_{TAG}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
