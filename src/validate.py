"""Validate the classification against an independent metadata source.

The pipeline reads `requires_dist` from the PyPI JSON API. This script instead
reads the core METADATA file shipped inside the distribution itself, served
by the PyPI Simple API (PEP 658/691), and re-derives the NumPy status of the
releases that drive each package's classification.

Sample (fixed seed): 20 capped packages stratified by cap origin + 10 never-capped.
Checks per capped package:
  first_cap        -> must be "capped"
  before_first_cap -> must NOT be "capped" (pre-emptive / reactive only)
  last_capped      -> last release before the lift, must be "capped"
  uncap            -> must be "compatible" (lifted) or "none" (dropped)
Checks per never-capped package: latest + one random in-window release -> not "capped".

Usage:
    python src/validate.py --top 8000
Outputs:
    data/validation_top<N>.csv  one row per check, with the raw numpy lines
"""
from __future__ import annotations

import argparse
import json
import random
from email.parser import HeaderParser
from pathlib import Path

import pandas as pd
import requests

from classify import active_reqs, numpy_status, safe

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "raw" / "validation"
S = requests.Session()
S.headers["User-Agent"] = "numpy2-ecosystem-study/0.1 (academic research; polite crawler)"
SEED = 42


def simple_files(pkg: str) -> list[dict]:
    path = CACHE / "simple" / f"{safe(pkg)}.json"
    if not path.exists():
        r = S.get(f"https://pypi.org/simple/{pkg}/",
                  headers={"Accept": "application/vnd.pypi.simple.v1+json"}, timeout=60)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))["files"]


def dist_metadata(pkg: str, version: str) -> tuple[list[str] | None, str]:
    """Requires-Dist lines from the distribution's own METADATA file."""
    files = simple_files(pkg)
    norm = safe(pkg).replace("-", "_").lower()
    cands = []
    for f in files:
        fn = f["filename"]
        if not f.get("core-metadata") and not f.get("dist-info-metadata"):
            continue
        low = fn.lower()
        if low.endswith(".whl") and low.split("-")[1] == version.lower():
            cands.append((0, f))
        elif (low.endswith((".tar.gz", ".zip")) and low.startswith(norm + "-" + version.lower() + ".")):
            cands.append((1, f))  # sdist with PEP 643 metadata
    if not cands:
        return None, "no-metadata-file"
    cands.sort(key=lambda x: (x[0], "none-any" not in x[1]["filename"]))  # prefer pure wheel
    f = cands[0][1]
    path = CACHE / "meta" / safe(pkg) / (safe(f["filename"]) + ".metadata")
    if not path.exists():
        r = S.get(f["url"].split("#")[0] + ".metadata", timeout=60)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    msg = HeaderParser().parsestr(path.read_text(encoding="utf-8", errors="replace"))
    return msg.get_all("Requires-Dist") or [], f["filename"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8000)
    ap.add_argument("--frame", default=None, help="frame tag, e.g. pre2024 (default: top<N>)")
    ap.add_argument("--mainline", action="store_true")
    args = ap.parse_args()
    frame = args.frame or f"top{args.top}"
    tag = frame + ("_mainline" if args.mainline else "")
    pk = pd.read_csv(ROOT / "data" / f"packages_{tag}.csv")
    rel = pd.read_csv(ROOT / "data" / f"releases_{frame}.csv")
    rel["uploaded"] = pd.to_datetime(rel.uploaded, utc=True, format="mixed")
    use = rel[~rel.prerelease & ~rel.yanked & ~rel.missing_requires_dist].sort_values("uploaded")
    if args.mainline:
        use = use[use.frontier]

    capped = pk[pk.ever_capped]
    # 20 capped packages allocated to cap origins proportionally (largest remainder)
    share = capped.cap_timing.value_counts(normalize=True) * 20
    quota = {k: int(v) for k, v in share.items()}
    for k in (share - share.astype(int)).sort_values(ascending=False).index[: 20 - sum(quota.values())]:
        quota[k] += 1
    sample = pd.concat([capped[capped.cap_timing == k].sample(n, random_state=SEED)
                        for k, n in quota.items()]
                       + [pk[~pk.ever_capped].sample(10, random_state=SEED)])
    rng = random.Random(SEED)

    checks = []
    for p in sample.itertuples():
        g = use[use.package == p.package].reset_index(drop=True)
        if p.ever_capped:
            i_cap = g.index[g.numpy_status == "capped"][0]
            checks.append((p, "first_cap", g.loc[i_cap], {"capped"}))
            if p.cap_timing != "legacy" and i_cap > 0:
                checks.append((p, "before_first_cap", g.loc[i_cap - 1], {"compatible", "none", "other"}))
            if isinstance(p.uncap_kind, str):
                i_un = g.index[(g.index > i_cap) & g.numpy_status.isin(["compatible", "none"])][0]
                checks.append((p, "last_capped", g.loc[i_un - 1], {"capped"}))
                checks.append((p, "uncap", g.loc[i_un], {"compatible"} if p.uncap_kind == "lifted" else {"none"}))
        else:
            checks.append((p, "latest", g.iloc[-1], {"compatible", "none", "other"}))
            checks.append((p, "random", g.iloc[rng.randrange(len(g))], {"compatible", "none", "other"}))

    rows = []
    for p, kind, r, expect in checks:
        lines, src = dist_metadata(p.package, r.version)
        indep = numpy_status(active_reqs(lines)) if lines is not None else "unavailable"
        rows.append({
            "package": p.package, "cap_timing": p.cap_timing, "check": kind,
            "version": r.version, "pipeline_status": r.numpy_status,
            "independent_status": indep, "expected": "|".join(sorted(expect)),
            "source_file": src,
            "numpy_lines": " || ".join(l for l in (lines or []) if l.lower().lstrip().startswith("numpy")),
        })
    df = pd.DataFrame(rows)
    df["agree_sources"] = df.pipeline_status == df.independent_status
    df["meets_definition"] = [s in e.split("|") for s, e in zip(df.independent_status, df.expected)]
    df.to_csv(ROOT / "data" / f"validation_{tag}.csv", index=False)
    sample[["package", "ever_capped", "cap_timing", "first_cap", "uncap", "uncap_kind"]].to_csv(
        ROOT / "data" / f"validation_sample_{tag}.csv", index=False)

    av = df[df.independent_status != "unavailable"]
    pk_ok = av.groupby("package").apply(lambda x: bool(x.meets_definition.all()), include_groups=False)
    print(f"checks: {len(df)}  with independent metadata: {len(av)}  unavailable: {len(df) - len(av)}")
    print(f"source agreement (JSON API vs METADATA): {av.agree_sources.mean():.1%} ({av.agree_sources.sum()}/{len(av)})")
    print(f"check-level correctness: {av.meets_definition.mean():.1%} ({av.meets_definition.sum()}/{len(av)})")
    print(f"package-level correctness: {pk_ok.mean():.1%} ({pk_ok.sum()}/{len(pk_ok)})")
    bad = df[~df.meets_definition | ~df.agree_sources]
    if len(bad):
        print("\nFor manual review:")
        print(bad[["package", "check", "version", "pipeline_status", "independent_status",
                   "expected", "numpy_lines"]].to_string(index=False))


if __name__ == "__main__":
    main()
