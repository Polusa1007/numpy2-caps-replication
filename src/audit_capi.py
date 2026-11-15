"""Classify whether a package builds against NumPy (i.e. likely uses the NumPy C API).

For each NumPy-dependent package that ships platform-specific wheels, take the
last stable release before NumPy 2.0.0 (else the first after) and inspect its
sdist:
  * pyproject.toml [build-system].requires contains numpy / oldest-supported-numpy
    -> "build-numpy"
  * else setup.py references numpy.get_include / numpy.distutils / setup_requires numpy
    -> "build-numpy"
  * else -> "no-build-numpy"
Packages without an sdist are "unknown". Pure-Python packages are "pure".

Tar sdists are streamed and decompression stops as soon as the top-level files
are found (or after MAX_BYTES); zip sdists are read with HTTP range requests.

Usage:
    python src/audit_capi.py --tag pre2024 [--tag top8000 ...]
Output:
    data/capi_<tag>.csv
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
import tomllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from audit_generator import HttpRangeFile
from classify import safe
from validate import simple_files

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "validation" / "capi"
EVENT = pd.Timestamp("2024-06-16", tz="UTC")
MAX_BYTES = 200 * 1024 * 1024
S = requests.Session()
S.headers["User-Agent"] = "numpy2-ecosystem-study/0.1 (academic research; polite crawler)"
NP_RE = re.compile(r"^\s*(numpy|oldest-supported-numpy)\b", re.I)
SETUP_RE = re.compile(r"numpy\.get_include|get_numpy_include|numpy\.distutils|setup_requires[^\n]*numpy", re.I)


class _Counting:
    def __init__(self, raw):
        self.raw, self.n = raw, 0

    def read(self, k=-1):
        if self.n > MAX_BYTES:
            return b""
        b = self.raw.read(k)
        self.n += len(b)
        return b


def top_level(name: str) -> str | None:
    parts = name.strip("./").split("/")
    return parts[1] if len(parts) == 2 else None


def inspect_files(get) -> dict:
    """get(fname) -> text or None, for 'pyproject.toml' / 'setup.py'."""
    py, st = get("pyproject.toml"), get("setup.py")
    reqs = []
    if py:
        try:
            reqs = tomllib.loads(py).get("build-system", {}).get("requires", []) or []
        except Exception:
            reqs = re.findall(r'"([^"]+)"', py.split("[build-system]", 1)[-1].split("\n[", 1)[0])
    np_build = any(NP_RE.match(r) for r in reqs)
    setup_np = bool(st and SETUP_RE.search(st))
    return {"has_pyproject": py is not None, "has_setup_py": st is not None,
            "build_requires": ";".join(reqs), "build_numpy": np_build or setup_np,
            "evidence": "pyproject" if np_build else ("setup.py" if setup_np else "")}


def read_sdist(f: dict) -> dict:
    url = f["url"].split("#")[0]
    fn = f["filename"].lower()
    if fn.endswith(".zip"):
        zf = zipfile.ZipFile(HttpRangeFile(url, f["size"]))
        names = {top_level(n): n for n in zf.namelist() if top_level(n)}
        return inspect_files(lambda k: zf.read(names[k]).decode("utf-8", "replace") if k in names else None)
    found: dict[str, str] = {}
    with S.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        with tarfile.open(fileobj=_Counting(r.raw), mode="r|*") as tf:
            for m in tf:
                k = top_level(m.name)
                if k in ("pyproject.toml", "setup.py") and m.isfile():
                    found[k] = tf.extractfile(m).read().decode("utf-8", "replace")
                    if len(found) == 2:
                        break
    return inspect_files(found.get)


def classify(pkg: str, compiled: bool, versions: list[tuple[pd.Timestamp, str]]) -> dict:
    if not compiled:
        return {"package": pkg, "capi": "pure"}
    cache = CACHE / f"{safe(pkg)}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    pre = [v for t, v in versions if t < EVENT]
    order = ([pre[-1]] if pre else []) + [v for t, v in versions if t >= EVENT][:1]
    out = {"package": pkg, "capi": "unknown", "version": None}
    try:
        files = simple_files(pkg)
        for ver in order:
            sd = [f for f in files if not f["filename"].endswith(".whl")
                  and f["filename"].lower().rsplit("-", 1)[-1].startswith(ver.lower() + ".")]
            if not sd:
                continue
            info = read_sdist(sd[0])
            out = {"package": pkg, "version": ver, "sdist": sd[0]["filename"], **info,
                   "capi": "build-numpy" if info["build_numpy"] else
                           ("no-build-numpy" if info["has_pyproject"] or info["has_setup_py"] else "unknown")}
            break
        else:
            out["capi"] = "no-sdist"
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    a = ap.parse_args()
    for tag in a.tag:
        pk = pd.read_csv(ROOT / "data" / f"packages_{tag}_mainline.csv")
        rel = pd.read_csv(ROOT / "data" / f"releases_{tag}.csv")
        rel["uploaded"] = pd.to_datetime(rel.uploaded, utc=True, format="mixed")
        rel = rel[~rel.prerelease & ~rel.yanked].sort_values("uploaded")
        vers = {p: list(zip(g.uploaded, g.version)) for p, g in rel.groupby("package")}
        with ThreadPoolExecutor(6) as ex:
            rows = list(ex.map(lambda r: classify(r.package, bool(r.compiled), vers.get(r.package, [])),
                               pk.itertuples()))
        df = pd.DataFrame(rows)
        df.to_csv(ROOT / "data" / f"capi_{tag}.csv", index=False)
        print(tag, df.capi.value_counts().to_dict())


if __name__ == "__main__":
    main()
