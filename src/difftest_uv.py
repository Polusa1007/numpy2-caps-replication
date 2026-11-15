"""Differential test of the NumPy-status classification against uv (independent Rust
implementation of PEP 440 / PEP 508 and of dependency resolution).

For every distinct set of NumPy requirement lines among the usable releases of the
main frame, we build a probe wheel whose metadata contains exactly those lines and
ask `uv pip compile` (CPython 3.12, x86_64 Linux, no extras, offline) to resolve it
twice: once against a local index offering only numpy 1.26.4, once against an index
offering only numpy 2.0.0-2.5.0 (empty stub wheels). From the two outcomes:
    numpy not required in either run         -> none
    resolves with a 2.x                       -> compatible
    resolves with 1.26.4 only                 -> capped
    resolves with neither                     -> other
and compare with src/classify.py.

Usage:
    python src/difftest_uv.py
Outputs:
    results/difftest_uv.json, data/difftest_uv.csv
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from classify import RAW, active_reqs, numpy_status, safe

ROOT = Path(__file__).resolve().parent.parent
NUMPY_LINE = re.compile(r"^\s*numpy\s*(?:[(<>=!~;\[ ]|$)", re.I)
V1 = ["1.26.4"]
V2 = [f"2.{m}.0" for m in range(6)]


def stub_wheel(folder: Path, name: str, version: str, requires: list[str]) -> None:
    dist = f"{name}-{version}.dist-info"
    meta = [f"Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"] + [f"Requires-Dist: {r}" for r in requires]
    wheel = "Wheel-Version: 1.0\nGenerator: difftest\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    with zipfile.ZipFile(folder / f"{name}-{version}-py3-none-any.whl", "w") as z:
        z.writestr(f"{dist}/METADATA", "\n".join(meta) + "\n")
        z.writestr(f"{dist}/WHEEL", wheel)
        z.writestr(f"{dist}/RECORD", f"{dist}/METADATA,,\n{dist}/WHEEL,,\n{dist}/RECORD,,\n")


def resolve(probe_dir: Path, name: str, numpy_dir: Path, cache: Path) -> str:
    """'numpy' if resolved with numpy, 'no-numpy' if resolved without, 'fail' otherwise."""
    cmd = [sys.executable, "-m", "uv", "pip", "compile", "-", "--no-index",
           "--find-links", str(probe_dir), "--find-links", str(numpy_dir),
           "--python-version", "3.12", "--python-platform", "x86_64-unknown-linux-gnu",
           "--no-header", "--no-annotate", "--quiet", "--cache-dir", str(cache)]
    r = subprocess.run(cmd, input=f"{name}==1.0\n", capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return "fail"
    return "numpy" if re.search(r"^numpy==", r.stdout, re.M) else "no-numpy"


def uv_status(a: str, b: str) -> str:
    if a == "no-numpy" and b == "no-numpy":
        return "none"
    if b == "numpy":
        return "compatible"
    if a == "numpy":
        return "capped"
    return "other"


def main() -> None:
    rel = pd.read_csv(ROOT / "data" / "releases_pre2024.csv")
    use = rel[~rel.prerelease & ~rel.yanked & ~rel.missing_requires_dist]
    sets: Counter = Counter()
    for p, v in zip(use.package, use.version):
        d = json.loads((RAW / "rel" / safe(p) / f"{safe(str(v))}.json").read_text(encoding="utf-8"))
        lines = tuple(sorted(r for r in (d["info"].get("requires_dist") or []) if NUMPY_LINE.match(r)))
        sets[lines] += 1
    work = Path(tempfile.mkdtemp(prefix="difftest_uv_"))
    try:
        d1, d2, cache = work / "np1", work / "np2", work / "cache"
        for d_, vs in ((d1, V1), (d2, V2)):
            d_.mkdir()
            for v in vs:
                stub_wheel(d_, "numpy", v, [])
        items = list(sets.items())

        def run(i_item):
            i, (lines, n) = i_item
            pdir = work / f"p{i}"
            pdir.mkdir()
            name = f"probe{i}"
            stub_wheel(pdir, name, "1.0", list(lines))
            a, b = resolve(pdir, name, d1, cache), resolve(pdir, name, d2, cache)
            ours = numpy_status(active_reqs(list(lines)))
            return {"lines": " || ".join(lines), "releases": n, "uv_1x": a, "uv_2x": b,
                    "uv_status": uv_status(a, b), "our_status": ours}

        with ThreadPoolExecutor(8) as ex:
            rows = list(ex.map(run, enumerate(items)))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    df = pd.DataFrame(rows)
    df["agree"] = df.uv_status == df.our_status
    df.to_csv(ROOT / "data" / "difftest_uv.csv", index=False)
    out = {
        "uv_version": subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True, text=True).stdout.strip(),
        "distinct_requirement_sets": int(len(df)), "releases": int(df.releases.sum()),
        "agree_sets": int(df.agree.sum()), "agree_releases": int(df[df.agree].releases.sum()),
        "by_status": {s: int(n) for s, n in df.groupby("our_status").releases.sum().items()},
        "disagreements": df[~df.agree].drop(columns="agree").to_dict("records"),
    }
    (ROOT / "results" / "difftest_uv.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=2))
    print(f"disagreeing sets: {len(out['disagreements'])}")
    for d_ in out["disagreements"][:20]:
        print(" ", d_)


if __name__ == "__main__":
    main()
