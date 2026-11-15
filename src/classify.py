"""Turn cached PyPI JSON into release-level and package-level tables.

Implements the operational definitions in docs/analysis_plan.md §4.

Usage:
    python src/classify.py --top 1500
Outputs:
    data/releases_top<N>.csv   one row per (package, release) in window
    data/packages_top<N>.csv   one row per NumPy-dependent package
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EVENT = datetime(2024, 6, 16, tzinfo=timezone.utc)
WINDOW_START = datetime(2023, 6, 16, tzinfo=timezone.utc)
V1 = Version("1.26.4")
V2S = [Version(f"2.{m}.0") for m in range(0, 6)]

# Reference environment: CPython 3.12, Linux x86_64, no extras.
ENV = default_environment()
ENV.update({
    "python_version": "3.12", "python_full_version": "3.12.0",
    "implementation_name": "cpython", "platform_python_implementation": "CPython",
    "sys_platform": "linux", "platform_system": "Linux", "os_name": "posix",
    "platform_machine": "x86_64", "extra": "",
})
# Alternative reference environments for sensitivity analysis (--env).
ENV_VARIANTS = {
    "py312-linux": {},
    "py39-linux": {"python_version": "3.9", "python_full_version": "3.9.0"},
    "py312-windows": {"sys_platform": "win32", "platform_system": "Windows", "os_name": "nt",
                      "platform_machine": "AMD64"},
}


_BASE_ENV = dict(ENV)


def set_env(name: str) -> None:
    """Switch the reference environment (always starting from the CPython 3.12 / Linux base)."""
    ENV.clear()
    ENV.update(_BASE_ENV)
    ENV.update(ENV_VARIANTS[name])
    _parse_active.cache_clear()


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


@lru_cache(maxsize=None)
def _parse_active(s: str) -> Requirement | None:
    """Parse one requirement string; None if invalid or inactive. Cached because
    the same strings repeat across hundreds of releases of one package."""
    if "extra ==" in s or "extra==" in s:
        # Optional-extra requirements are inactive under the no-extras environment.
        # Skipping them before parsing avoids parsing thousands of extras.
        if " or " not in s:
            return None
    try:
        req = Requirement(s)
    except InvalidRequirement:
        return None
    if req.marker is None or req.marker.evaluate(ENV):
        return req
    return None


def active_reqs(requires_dist: list[str] | None) -> list[Requirement]:
    """Non-optional requirements under the reference environment."""
    return [r for r in map(_parse_active, requires_dist or []) if r is not None]


def numpy_status(reqs: list[Requirement]) -> str:
    """'none' | 'capped' | 'compatible' | 'other' (e.g. rejects 1.26.4 too)."""
    np_reqs = [r for r in reqs if canonicalize_name(r.name) == "numpy"]
    if not np_reqs:
        return "none"
    ok1 = all(r.specifier.contains(V1, prereleases=True) for r in np_reqs)
    # Compatible = admits some NumPy 2.x (e.g. "numpy>=2.1" rejects 2.0.0 but is 2-compatible).
    ok2 = any(all(r.specifier.contains(v, prereleases=True) for r in np_reqs) for v in V2S)
    if ok2:
        return "compatible"
    if ok1:
        return "capped"
    return "other"


def upload_time(files: list[dict]) -> datetime | None:
    ts = [f["upload_time_iso_8601"] for f in files if f.get("upload_time_iso_8601")]
    return min(datetime.fromisoformat(t.replace("Z", "+00:00")) for t in ts) if ts else None


def is_compiled(files: list[dict]) -> bool:
    return any(f["filename"].endswith(".whl") and not f["filename"].endswith("-none-any.whl")
               for f in files)


def pre_window_status(pkg: str, releases: dict) -> str:
    """NumPy status of the last stable, non-yanked release before the window.
    Fetched on demand (one request per capped package) and cached."""
    from collect import get_json  # same directory; network only on cache miss
    best = None
    for ver, files in releases.items():
        t = upload_time(files)
        if t is None or t >= WINDOW_START or any(f.get("yanked") for f in files):
            continue
        try:
            if Version(ver).is_prerelease:
                continue
        except InvalidVersion:
            continue
        if best is None or t > best[0]:
            best = (t, ver)
    if best is None:
        return "unknown"
    data = get_json(f"https://pypi.org/pypi/{pkg}/{best[1]}/json",
                    RAW / "rel" / safe(pkg) / f"{safe(best[1])}.json")
    rd = data["info"].get("requires_dist") if data else None
    return numpy_status(active_reqs(rd)) if rd is not None else "unknown"


def cap_style(spec: str) -> str:
    """How a capped specifier blocks NumPy 2 (deviation D9):
    'major-cap' (<2, <2.0.0a0, ==1.*), 'minor-cap' (e.g. <1.27), 'pin' (==1.26.4, ~=1.26.0)."""
    from packaging.specifiers import SpecifierSet
    try:
        ss = SpecifierSet(spec)
    except Exception:
        return "unknown"
    styles = set()
    for s in ss:
        if s.operator in ("==", "===") and not s.version.endswith(".*"):
            styles.add("pin")
        elif s.operator == "~=":
            styles.add("pin")
        elif s.operator == "==" and s.version.endswith(".*"):
            styles.add("major-cap" if s.version.count(".") == 1 else "minor-cap")
        elif s.operator in ("<", "<="):
            v = Version(s.version)
            styles.add("major-cap" if v >= Version("2.0.0a0") or v.release[:1] == (2,) else "minor-cap")
    for k in ("pin", "minor-cap", "major-cap"):
        if k in styles:
            return k
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=1500)
    ap.add_argument("--tag", default=None, help="frame tag, e.g. pre2024 (default: top<N>)")
    ap.add_argument("--env", default="py312-linux", choices=list(ENV_VARIANTS),
                    help="reference environment for marker evaluation (sensitivity)")
    ap.add_argument("--mainline", action="store_true",
                    help="use only frontier (mainline) releases; deviation D8")
    args = ap.parse_args()
    tag = args.tag or f"top{args.top}"
    set_env(args.env)
    envsfx = "" if args.env == "py312-linux" else f"_{args.env}"
    meta = json.loads((ROOT / "data" / f"meta_{tag}.json").read_text(encoding="utf-8"))
    rank = {canonicalize_name(n): i + 1 for i, n in enumerate(meta["frame"])}

    rel_rows = []
    pkg_releases: dict[str, dict] = {}
    for name in meta["candidates"]:
        pkg = json.loads((RAW / "pkg" / f"{safe(name)}.json").read_text(encoding="utf-8"))
        pkg_releases[canonicalize_name(name)] = pkg["releases"]
        for ver, files in pkg["releases"].items():
            t = upload_time(files)
            if not t or t < WINDOW_START:
                continue
            row = {"package": canonicalize_name(name), "version": ver, "uploaded": t,
                   "yanked": any(f.get("yanked") for f in files),
                   "compiled_wheel": is_compiled(files)}
            try:
                row["prerelease"] = Version(ver).is_prerelease
            except InvalidVersion:
                row["prerelease"] = True
            cache = RAW / "rel" / safe(name) / f"{safe(ver)}.json"
            rd = None
            if cache.exists() and cache.read_text(encoding="utf-8") != "null":
                rd = json.loads(cache.read_text(encoding="utf-8"))["info"].get("requires_dist")
            row["missing_requires_dist"] = rd is None
            reqs = active_reqs(rd)
            row["numpy_status"] = numpy_status(reqs) if rd is not None else "unknown"
            row["deps"] = ";".join(sorted({canonicalize_name(r.name) for r in reqs}))
            row["numpy_spec"] = ",".join(str(r.specifier) for r in reqs
                                         if canonicalize_name(r.name) == "numpy")
            rel_rows.append(row)

    rel = pd.DataFrame(rel_rows).sort_values(["package", "uploaded"])
    usable = ~rel.prerelease & ~rel.yanked & ~rel.missing_requires_dist
    # Mainline (deviation D8): a usable release is "frontier" if its version is higher
    # than every usable release uploaded before it; backports to old branches are not.
    rel["frontier"] = False
    for _, g in rel[usable].groupby("package"):
        top_v = None
        for idx, ver in zip(g.index, g.version):
            v = Version(ver)
            if top_v is None or v > top_v:
                rel.loc[idx, "frontier"] = True
                top_v = v
    rel.to_csv(ROOT / "data" / f"releases_{tag}{envsfx}.csv", index=False)

    # Package-level summary using only usable releases.
    use = rel[usable]
    if args.mainline:
        use = use[use.frontier]
    pk_rows = []
    for pkg, g in use.groupby("package"):
        g = g.sort_values("uploaded")
        dep = g[g.numpy_status.isin(["capped", "compatible", "other"])]
        if dep.empty:
            continue
        capped = g[g.numpy_status == "capped"]
        first_cap = capped.uploaded.min() if not capped.empty else pd.NaT
        uncap, uncap_kind = pd.NaT, None
        if not capped.empty:
            # Deviation D3: dropping the NumPy dependency also ends the cap.
            after = g[(g.uploaded > first_cap) & g.numpy_status.isin(["compatible", "none"])]
            if not after.empty:
                uncap = after.uploaded.iloc[0]
                uncap_kind = "lifted" if after.numpy_status.iloc[0] == "compatible" else "dropped-dependency"
        # Cap origin (deviation D1): a cap already present in the first in-window
        # release of a package with earlier history is "legacy", not a response.
        all_times = [t for f in pkg_releases[pkg].values() if (t := upload_time(f)) is not None]
        first_release = min(all_times) if all_times else pd.NaT
        had_history = any(t < WINDOW_START for t in all_times)
        # Union of dependencies across capped releases (used for RQ3 pairs).
        capped_deps = sorted({d for s in capped.deps.dropna() for d in str(s).split(";") if d})
        if capped.empty:
            cap_timing = None
        elif had_history and pre_window_status(pkg, pkg_releases[pkg]) == "capped":
            # Deviation D6: legacy = the last release before the window was already capped.
            cap_timing = "legacy"
        elif first_cap < EVENT:
            cap_timing = "pre-emptive"
        else:
            cap_timing = "reactive"
        # Adaptation timing (deviation D2): relative to the NumPy 2.0.0 release.
        if capped.empty:
            adaptation = "never-capped"
        elif pd.isna(uncap):
            adaptation = "still-capped"
        elif uncap < EVENT:
            adaptation = "before-release"
        else:
            adaptation = "after-release"
        pk_rows.append({
            "package": pkg,
            "download_rank": rank.get(pkg),
            "downloads_30d": meta["downloads"].get(pkg) or next(
                (v for k, v in meta["downloads"].items() if canonicalize_name(k) == pkg), None),
            "compiled": bool(rel.loc[rel.package == pkg, "compiled_wheel"].any()),
            "n_releases_window": len(g),
            "releases_2023": int((g.uploaded.dt.year == 2023).sum()),
            "ever_capped": not capped.empty,
            "first_cap": first_cap,
            "cap_timing": cap_timing,
            "cap_style": cap_style(capped.numpy_spec.iloc[0]) if not capped.empty else None,
            "uncap": uncap,
            "adaptation": adaptation,
            "uncap_kind": uncap_kind,
            "days_event_to_uncap": (uncap - EVENT).days if pd.notna(uncap) else None,
            "latest_status": g.numpy_status.iloc[-1],
            "latest_release": g.uploaded.iloc[-1],
            "latest_deps": g.deps.iloc[-1],
            "capped_deps": ";".join(capped_deps),
            "first_release": first_release,
            "age_years_at_event": ((EVENT - first_release).days / 365.25
                                   if pd.notna(first_release) else None),
        })
    pk = pd.DataFrame(pk_rows)
    suffix = "_mainline" if args.mainline else ""
    pk.to_csv(ROOT / "data" / f"packages_{tag}{suffix}{envsfx}.csv", index=False)

    n_all = len(rel)
    print(f"Releases in window: {n_all}")
    print(f"  pre-release: {rel.prerelease.sum()}  yanked: {rel.yanked.sum()}  "
          f"missing requires_dist: {rel.missing_requires_dist.sum()} "
          f"({rel.missing_requires_dist.mean():.1%})")
    print(f"NumPy-dependent packages: {len(pk)}")
    print(f"  compiled: {pk.compiled.sum()}  pure-python: {(~pk.compiled).sum()}")
    print(f"  ever capped: {pk.ever_capped.sum()}")
    print("  cap origin:", pk.cap_timing.value_counts().to_dict())
    print("  adaptation:", pk.adaptation.value_counts().to_dict())


if __name__ == "__main__":
    main()
