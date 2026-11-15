"""Second, binary-level indicator of exposure to the NumPy ABI break.

For every NumPy-dependent package with platform-specific wheels, take the last
release before NumPy 2.0.0 (else the first after), pick its smallest
CPython x86-64 Linux wheel (else any platform wheel), and scan the compiled
extension modules (.so / .pyd) for strings that only appear when an extension
uses the NumPy C API at run time:
    _ARRAY_API, PyArray_API, numpy.core.multiarray, numpy.core._multiarray_umath
Members are streamed with HTTP range requests; scanning stops at the first hit or
after BUDGET bytes per package (then: "unscanned").

This indicator also covers packages that use the C API without NumPy as a build
requirement (e.g. via pybind11), so it measures a broader construct than the
build-requirement rule; we report agreement, not accuracy against ground truth.

Usage:
    python src/audit_wheel_capi.py
Output:
    data/wheel_capi_pre2024.csv
"""
from __future__ import annotations

import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from audit_generator import HttpRangeFile
from classify import safe
from validate import simple_files

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "validation" / "wheel_capi"
EVENT = pd.Timestamp("2024-06-16", tz="UTC")
PATTERNS = [b"_ARRAY_API", b"PyArray_API", b"numpy.core.multiarray", b"numpy.core._multiarray_umath"]
BUDGET = 400 * 1024 * 1024
CHUNK = 1 << 20


def pick_wheel(files: list[dict], version: str) -> dict | None:
    whl = [f for f in files if f["filename"].endswith(".whl") and f["filename"].split("-")[1].lower() == version.lower()
           and not f["filename"].endswith("-none-any.whl")]
    if not whl:
        return None

    def rank(f):
        fn = f["filename"]
        return ("linux" not in fn or "x86_64" not in fn, "cp312" not in fn, f.get("size") or 0)
    return sorted(whl, key=rank)[0]


def scan(pkg: str, versions: list[tuple[pd.Timestamp, str]]) -> dict:
    cache = CACHE / f"{safe(pkg)}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    pre = [v for t, v in versions if t < EVENT]
    order = ([pre[-1]] if pre else []) + [v for t, v in versions if t >= EVENT][:1]
    out = {"package": pkg, "binary_capi": "no-wheel", "wheel": None, "hit": None}
    try:
        files = simple_files(pkg)
        for ver in order:
            f = pick_wheel(files, ver)
            if not f:
                continue
            out["wheel"] = f["filename"]
            zf = zipfile.ZipFile(HttpRangeFile(f["url"].split("#")[0], f["size"]))
            exts = [i for i in zf.infolist() if re.search(r"\.(so(\.\d+)*|pyd)$", i.filename)]
            exts.sort(key=lambda i: i.file_size)
            read, hit = 0, None
            for info in exts:
                tail = b""
                with zf.open(info) as fh:
                    while True:
                        chunk = fh.read(CHUNK)
                        if not chunk:
                            break
                        buf = tail + chunk
                        hit = next((p.decode() for p in PATTERNS if p in buf), None)
                        if hit:
                            break
                        tail = buf[-64:]
                        read += len(chunk)
                        if read > BUDGET:
                            break
                if hit or read > BUDGET:
                    break
            out["binary_capi"] = "uses-capi" if hit else ("unscanned" if read > BUDGET else "no-capi")
            out["hit"] = hit
            out["n_extensions"] = len(exts)
            break
    except Exception as e:  # noqa: BLE001
        out["binary_capi"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def main() -> None:
    pk = pd.read_csv(ROOT / "data" / "packages_pre2024_mainline.csv")
    rel = pd.read_csv(ROOT / "data" / "releases_pre2024.csv")
    rel["uploaded"] = pd.to_datetime(rel.uploaded, utc=True, format="mixed")
    rel = rel[~rel.prerelease & ~rel.yanked].sort_values("uploaded")
    vers = {p: list(zip(g.uploaded, g.version)) for p, g in rel.groupby("package")}
    comp = pk[pk.compiled]
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(lambda p: scan(p, vers.get(p, [])), comp.package))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "wheel_capi_pre2024.csv", index=False)
    print(df.binary_capi.value_counts().to_dict())


if __name__ == "__main__":
    main()
