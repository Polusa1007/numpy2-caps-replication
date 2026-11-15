"""Runtime (import-level) compatibility test with NumPy 1.26.4 vs. NumPy 2.0.0.

Sample: packages whose latest stable mainline release before 2024-06-16 was either
capped or compatible (25+25 per stratum by default x2 = --n per group), fixed seed.
For each package we install that release into a fresh CPython 3.12 virtual
environment with uv, resolving all other dependencies as of 2024-07-01
(--exclude-newer), once forcing numpy==1.26.4 and once forcing numpy==2.0.0
(uv --override, which also overrides the package's own cap), and then import
every top-level module the distribution provides.

    runtime-incompatible  = imports with 1.26.4 but not with 2.0.0
    runtime-compatible    = imports with both
    not-testable          = install fails, too large, or fails already with 1.26.4

Platform: the machine it runs on (Windows x86-64, or Linux x86-64, e.g. WSL Ubuntu);
results are kept apart by platform. The 1.26.4 control run removes failures unrelated
to NumPy 2.

Usage:
    python src/runtime_test.py --n 50                       # Windows (paper: n=50)
    TMPDIR=~/rt_tmp python src/runtime_test.py --n 100      # Linux / WSL (paper: n=100)
On WSL, /tmp is a RAM-backed tmpfs that large environments (e.g. PyTorch) overflow,
so TMPDIR must point to a directory on disk.
Outputs:
    data/runtime_test.csv, results/runtime_test.json                (Windows)
    data/runtime_test_linux.csv, results/runtime_test_linux.json    (Linux)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from validate import simple_files

ROOT = Path(__file__).resolve().parent.parent
LINUX = sys.platform.startswith("linux")
SUFFIX = "_linux" if LINUX else ""
CACHE = ROOT / "data" / "raw" / f"runtime{SUFFIX}"
EVENT = pd.Timestamp("2024-06-16", tz="UTC")
EXCLUDE_NEWER = "2024-07-01T00:00:00Z"
MAX_WHEEL = 150 * 1024 * 1024
SEED = 31
UV = [sys.executable, "-m", "uv"]

IMPORT_SCRIPT = r'''
import importlib, json, sys, traceback
from importlib.metadata import packages_distributions
from packaging.utils import canonicalize_name
dist = canonicalize_name(sys.argv[1])
mods = sorted(m for m, ds in packages_distributions().items()
              if any(canonicalize_name(d) == dist for d in ds)
              and not m.startswith("_") and m not in ("tests", "test", "testing", "docs", "examples", "benchmarks"))
res = {}
for m in mods:
    try:
        importlib.import_module(m)
        res[m] = "ok"
    except BaseException as e:
        res[m] = type(e).__name__ + ": " + str(e)[:300]
print("RESULT" + json.dumps(res))
'''


def run(cmd, timeout):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def try_numpy(pkg: str, ver: str, np_ver: str, work: Path) -> dict:
    env = work / f"v{np_ver.replace('.', '')}"
    rc, _, err = run(UV + ["venv", "--python", "3.12", "--quiet", str(env)], 300)
    if rc != 0:
        return {"stage": "venv", "ok": False, "detail": err[-300:]}
    py = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    over = work / f"override_{np_ver}.txt"
    over.write_text(f"numpy=={np_ver}\n", encoding="utf-8")
    rc, _, err = run(UV + ["pip", "install", "--python", str(py), "--quiet", f"{pkg}=={ver}", "packaging",
                           "--override", str(over), "--exclude-newer", EXCLUDE_NEWER], 1500)
    if rc != 0:
        return {"stage": "install", "ok": False, "detail": err[-400:]}
    script = work / "imp.py"
    script.write_text(IMPORT_SCRIPT, encoding="utf-8")
    rc, out, err = run([str(py), str(script), pkg], 600)
    line = next((l for l in out.splitlines() if l.startswith("RESULT")), None)
    if line is None:
        return {"stage": "import-crash", "ok": False, "detail": (err or out)[-400:]}
    res = json.loads(line[6:])
    if not res:
        return {"stage": "no-modules", "ok": False, "detail": ""}
    bad = {m: e for m, e in res.items() if e != "ok"}
    return {"stage": "import", "ok": not bad, "modules": len(res), "failed": bad}


def test(row: dict) -> dict:
    pkg, ver = row["package"], row["version"]
    cache = CACHE / f"{pkg}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out = {**row}
    files = [f for f in simple_files(pkg) if f["filename"].lower().split("-")[1:2] == [ver.lower()]
             and f["filename"].endswith(".whl")]
    sizes = [f.get("size") or 0 for f in files if ((("manylinux" in f["filename"] and "x86_64" in f["filename"]) if LINUX else "win_amd64" in f["filename"])
                                     or "none-any" in f["filename"])]
    if sizes and min(sizes) > MAX_WHEEL:
        out["outcome"] = "not-testable"
        out["reason"] = "wheel too large"
    else:
        work = Path(tempfile.mkdtemp(prefix="rt_"))
        try:
            a = try_numpy(pkg, ver, "1.26.4", work)
            b = try_numpy(pkg, ver, "2.0.0", work) if a["ok"] else {"stage": "skipped", "ok": False}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        out["np1"], out["np2"] = a, b
        if not a["ok"]:
            out["outcome"], out["reason"] = "not-testable", f"fails with 1.26.4 ({a['stage']})"
        else:
            out["outcome"] = "runtime-compatible" if b["ok"] else "runtime-incompatible"
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="packages per declared status")
    ap.add_argument("--only", nargs="*", help="test only these packages (debugging)")
    a = ap.parse_args()
    rel = pd.read_csv(ROOT / "data" / "releases_pre2024.csv")
    rel["uploaded"] = pd.to_datetime(rel.uploaded, utc=True, format="mixed")
    use = rel[~rel.prerelease & ~rel.yanked & ~rel.missing_requires_dist & rel.frontier & (rel.uploaded < EVENT)]
    pk = pd.read_csv(ROOT / "data" / "packages_pre2024_mainline.csv")
    last = use[use.package.isin(pk.package)].sort_values("uploaded").groupby("package").tail(1)
    last = last[last.numpy_status.isin(["capped", "compatible"])][["package", "version", "numpy_status"]]
    last = last.rename(columns={"numpy_status": "declared"})
    if a.only:
        sample = last[last.package.isin(a.only)]
    else:
        sample = pd.concat([g.sample(min(a.n, len(g)), random_state=SEED) for _, g in last.groupby("declared")])
    with ThreadPoolExecutor(4 if LINUX else 3) as ex:
        rows = list(ex.map(test, sample.astype(str).to_dict("records")))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / f"runtime_test{SUFFIX}.csv", index=False)
    tab = pd.crosstab(df.declared, df.outcome)
    out = {"population_at_release": last.declared.value_counts().to_dict(), "sampled": int(len(df)),
           "crosstab": {d: {k: int(v) for k, v in r.items()} for d, r in tab.iterrows()},
           "exclude_newer": EXCLUDE_NEWER, "platform": sys.platform, "seed": SEED}
    if not a.only:
        (ROOT / "results" / f"runtime_test{SUFFIX}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
