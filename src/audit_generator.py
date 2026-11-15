"""Identify the build tool behind each package's first capped release (revision R5).

Poetry's caret syntax (numpy = "^1.24") is emitted as ">=1.24,<2.0", so a cap may be
a tool default rather than a decision about NumPy 2. The build tool is recorded in
the `Generator:` line of the wheel's `*.dist-info/WHEEL` file. We read only that
file using HTTP range requests (the wheel's zip central directory + one entry),
never downloading whole wheels.

Usage:
    python src/audit_generator.py --top 8000 [--variant _mainline]
Output:
    data/generator_top<N><variant>.csv
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

from classify import safe
from validate import simple_files

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "validation" / "wheelfile"
S = requests.Session()
S.headers["User-Agent"] = "numpy2-ecosystem-study/0.1 (academic research; polite crawler)"


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, fetched in cached 64 KiB blocks."""
    BLOCK = 1 << 16

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos, self.blocks = url, size, 0, {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.blocks:
            lo, hi = i * self.BLOCK, min(self.size, (i + 1) * self.BLOCK) - 1
            r = S.get(self.url, headers={"Range": f"bytes={lo}-{hi}"}, timeout=60)
            r.raise_for_status()
            self.blocks[i] = r.content
        return self.blocks[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = bytearray()
        while n > 0 and self.pos < self.size:
            b = self._block(self.pos // self.BLOCK)
            off = self.pos % self.BLOCK
            chunk = b[off: off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def wheel_generator(pkg: str, version: str) -> tuple[str, str]:
    cache = CACHE / safe(pkg) / f"{safe(version)}.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        return d["generator"], d["file"]
    wheels = [f for f in simple_files(pkg)
              if f["filename"].endswith(".whl") and f["filename"].split("-")[1].lower() == version.lower()]
    if not wheels:
        gen, fn = "no-wheel", ""
    else:
        wheels.sort(key=lambda f: ("none-any" not in f["filename"], f.get("size") or 0))
        f = wheels[0]
        fn = f["filename"]
        try:
            zf = zipfile.ZipFile(HttpRangeFile(f["url"].split("#")[0], f["size"]))
            name = next(n for n in zf.namelist() if n.endswith(".dist-info/WHEEL"))
            text = zf.read(name).decode("utf-8", "replace")
            gen = next((l.split(":", 1)[1].strip() for l in text.splitlines()
                        if l.lower().startswith("generator:")), "unknown")
        except Exception as e:  # noqa: BLE001 - record and continue
            gen = f"error: {type(e).__name__}"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"generator": gen, "file": fn}), encoding="utf-8")
    return gen, fn


def tool_of(gen: str) -> str:
    g = gen.lower()
    for k in ("poetry", "hatchling", "setuptools", "bdist_wheel", "flit", "maturin",
              "scikit-build", "meson", "pdm", "uv", "bazel"):
        if k in g:
            return {"bdist_wheel": "setuptools"}.get(k, k)
    return "no-wheel" if g == "no-wheel" else "other/unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8000)
    ap.add_argument("--variant", default="_mainline")
    ap.add_argument("--frame", default=None, help="frame tag, e.g. pre2024 (default: top<N>)")
    a = ap.parse_args()
    frame = a.frame or f"top{a.top}"
    tag = f"{frame}{a.variant}"
    pk = pd.read_csv(ROOT / "data" / f"packages_{tag}.csv")
    rel = pd.read_csv(ROOT / "data" / f"releases_{frame}.csv")
    rel["uploaded"] = pd.to_datetime(rel.uploaded, utc=True, format="mixed")
    use = rel[~rel.prerelease & ~rel.yanked & ~rel.missing_requires_dist]
    if a.variant == "_mainline":
        use = use[use.frontier]
    rows = []
    for p in pk[pk.ever_capped].itertuples():
        g = use[(use.package == p.package) & (use.numpy_status == "capped")].sort_values("uploaded")
        r = g.iloc[0]
        gen, fn = wheel_generator(p.package, r.version)
        rows.append({"package": p.package, "version": r.version, "numpy_spec": r.numpy_spec,
                     "cap_timing": p.cap_timing, "cap_style": p.cap_style,
                     "generator": gen, "tool": tool_of(gen), "wheel": fn})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / f"generator_{tag}.csv", index=False)
    print(df.tool.value_counts().to_string())
    print(pd.crosstab(df.cap_timing, df.tool == "poetry", margins=True).to_string())


if __name__ == "__main__":
    main()
