"""Collect PyPI metadata for the NumPy 2.0 downstream-adaptation study.

Usage:
    python src/collect.py --top 1500          # pilot
    python src/collect.py --top 8000          # full study

All HTTP responses are cached under data/raw/, so reruns are cheap and the
snapshot can be archived for replication.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TOP_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"
WINDOW_START = datetime(2023, 6, 16, tzinfo=timezone.utc)
HEADERS = {"User-Agent": "numpy2-ecosystem-study/0.1 (academic research; polite crawler)"}
WORKERS = 8

_session = requests.Session()
_session.headers.update(HEADERS)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def slim(data: dict) -> dict:
    """Keep only the fields the study uses (cuts cache size ~10-20x)."""
    out = {"info": {"requires_dist": data.get("info", {}).get("requires_dist")}}
    if "releases" in data:
        out["releases"] = {
            ver: [{k: f.get(k) for k in ("filename", "upload_time_iso_8601", "yanked")}
                  for f in files]
            for ver, files in data["releases"].items()
        }
    return out


def get_json(url: str, cache: Path, keep_full: bool = False) -> dict | None:
    """GET with on-disk cache and retry. Returns None on 404."""
    if cache.exists():
        txt = cache.read_text(encoding="utf-8")
        if txt == "null":
            return None
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            cache.unlink()  # partial write from an interrupted run; refetch
    tmp = cache.with_suffix(".tmp")
    for attempt in range(5):
        try:
            r = _session.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("null", encoding="utf-8")
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        data = r.json() if keep_full else slim(r.json())
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(cache)  # atomic: a cache file is either complete or absent
        return data
    raise RuntimeError(f"failed after retries: {url}")


def mentions_numpy(requires_dist: list[str] | None) -> bool:
    if not requires_dist:
        return False
    return any(re.match(r"^\s*numpy\b", r, re.I) for r in requires_dist)


def release_upload_time(files: list[dict]) -> datetime | None:
    times = [f.get("upload_time_iso_8601") for f in files if f.get("upload_time_iso_8601")]
    if not times:
        return None
    return min(datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=1500)
    args = ap.parse_args()

    top = get_json(TOP_URL, RAW / "top-pypi-packages.json", keep_full=True)
    rows = top["rows"][: args.top]
    print(f"Top list snapshot: {top.get('last_update')}  ({len(rows)} packages)")

    # Stage 1: package-level JSON for every package in the frame.
    pkgs: dict[str, dict] = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {
            ex.submit(get_json, f"https://pypi.org/pypi/{r['project']}/json",
                      RAW / "pkg" / f"{safe(r['project'])}.json"): r["project"]
            for r in rows
        }
        for i, f in enumerate(as_completed(futs), 1):
            name = futs[f]
            data = f.result()
            if data:
                pkgs[name] = data
            if i % 250 == 0:
                print(f"  stage1 {i}/{len(rows)}")

    # Pre-filter: latest release mentions numpy (deviation risk logged in plan).
    cands = [n for n, d in pkgs.items() if mentions_numpy(d["info"].get("requires_dist"))]
    print(f"Packages fetched: {len(pkgs)}; latest release mentions numpy: {len(cands)}")

    # Stage 2: per-release JSON for releases inside the observation window.
    jobs = []
    for name in cands:
        for ver, files in pkgs[name]["releases"].items():
            t = release_upload_time(files)
            if t and t >= WINDOW_START:
                jobs.append((name, ver))
    print(f"Per-release requests needed: {len(jobs)}")

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [
            ex.submit(get_json, f"https://pypi.org/pypi/{n}/{v}/json",
                      RAW / "rel" / safe(n) / f"{safe(v)}.json")
            for n, v in jobs
        ]
        for i, f in enumerate(as_completed(futs), 1):
            f.result()
            if i % 1000 == 0:
                print(f"  stage2 {i}/{len(jobs)}")

    meta = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "top_snapshot": top.get("last_update"),
        "top_n": args.top,
        "frame": [r["project"] for r in rows],
        "downloads": {r["project"]: r["download_count"] for r in rows},
        "candidates": cands,
    }
    (ROOT / "data" / f"meta_top{args.top}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
