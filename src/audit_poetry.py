"""Cross-check of the Poetry (tool-generated cap) indicator against source metadata.

For each capped package, read pyproject.toml from the sdist of its first capped
mainline release and record (a) whether the build backend is Poetry and (b)
whether NumPy is declared with Poetry's caret/tilde syntax ("^1.x", "~1.x"),
which Poetry translates into an upper bound. Compare with the wheel-Generator
indicator used in the paper.

Usage:
    python src/audit_poetry.py
Output:
    data/poetry_check_pre2024.csv, results/poetry_check_pre2024.json
"""
from __future__ import annotations

import json
import tarfile
import tomllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from audit_capi import S, _Counting, top_level
from audit_generator import HttpRangeFile
from classify import safe
from validate import simple_files

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "validation" / "poetry"


def pyproject(f: dict) -> str | None:
    url = f["url"].split("#")[0]
    if f["filename"].lower().endswith(".zip"):
        zf = zipfile.ZipFile(HttpRangeFile(url, f["size"]))
        names = {top_level(n): n for n in zf.namelist() if top_level(n)}
        return zf.read(names["pyproject.toml"]).decode("utf-8", "replace") if "pyproject.toml" in names else None
    with S.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        with tarfile.open(fileobj=_Counting(r.raw), mode="r|*") as tf:
            for m in tf:
                if top_level(m.name) == "pyproject.toml" and m.isfile():
                    return tf.extractfile(m).read().decode("utf-8", "replace")
    return None


def check(pkg: str, version: str) -> dict:
    cache = CACHE / f"{safe(pkg)}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    out = {"package": pkg, "version": version, "source": "no-sdist"}
    try:
        sd = [f for f in simple_files(pkg) if not f["filename"].endswith(".whl")
              and f["filename"].lower().rsplit("-", 1)[-1].startswith(version.lower() + ".")]
        if sd:
            text = pyproject(sd[0])
            if text is None:
                out["source"] = "no-pyproject"
            else:
                t = tomllib.loads(text)
                backend = t.get("build-system", {}).get("build-backend", "") or ""
                deps = t.get("tool", {}).get("poetry", {}).get("dependencies", {}) or {}
                spec = deps.get("numpy")
                if isinstance(spec, dict):
                    spec = spec.get("version")
                if isinstance(spec, list):
                    spec = ";".join(str(x.get("version", x)) if isinstance(x, dict) else str(x) for x in spec)
                out.update({"source": "pyproject", "poetry_backend": "poetry" in backend,
                            "poetry_numpy_spec": spec,
                            "caret_or_tilde": bool(spec and any(c in str(spec) for c in "^~") and "~=" not in str(spec))})
    except Exception as e:  # noqa: BLE001
        out["source"] = "error"
        out["error"] = f"{type(e).__name__}: {e}"[:200]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def main() -> None:
    gen = pd.read_csv(ROOT / "data" / "generator_pre2024_mainline.csv")
    with ThreadPoolExecutor(4) as ex:
        rows = list(ex.map(lambda r: check(r.package, str(r.version)), gen.itertuples()))
    df = gen[["package", "tool"]].merge(pd.DataFrame(rows), on="package")
    df.to_csv(ROOT / "data" / "poetry_check_pre2024.csv", index=False)
    k = df[df.source == "pyproject"].copy()
    k["wheel_poetry"] = k.tool.eq("poetry")
    k["src_poetry"] = k.poetry_backend.fillna(False).astype(bool)
    tab = pd.crosstab(k.wheel_poetry, k.src_poetry)
    out = {"capped": int(len(df)), "with_pyproject": int(len(k)),
           "source_counts": df.source.value_counts().to_dict(),
           "agree": int((k.wheel_poetry == k.src_poetry).sum()),
           "wheel_yes_src_yes": int((k.wheel_poetry & k.src_poetry).sum()),
           "wheel_yes_src_no": int((k.wheel_poetry & ~k.src_poetry).sum()),
           "wheel_no_src_yes": int((~k.wheel_poetry & k.src_poetry).sum()),
           "poetry_caret_numpy": int((k.src_poetry & k.caret_or_tilde.fillna(False).astype(bool)).sum()),
           "crosstab": {str(i): {str(c): int(v) for c, v in r.items()} for i, r in tab.iterrows()}}
    (ROOT / "results" / "poetry_check_pre2024.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
