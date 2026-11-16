"""Generate paper/numbers.tex: every number quoted in the manuscript, as LaTeX macros.

Main analysis:     results/main_pre2024.json        (macros without prefix)
Planned analysis:  results/results_top8000.json,    (macros prefixed with P)
                   results/sensitivity_top8000.json
Validation:        data/validation_pre2024_mainline.csv

Usage:
    python src/make_numbers.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LINES: list[str] = []


class PStr(str):
    """A formatted p-value; mac() also emits <name>Rel ('< .001' or '= .xx') for running text."""


# Quantities computed in loops shared with reported ones but not reported in the paper;
# mac() skips them so that numbers.tex lists only macros the manuscript uses.
NOT_REPORTED = {
    "BoundLoA",
    "BoundLoB",
    "BoundUpA",
    "BoundUpB",
    "CapCapiP",
    "CapCompCIninety",
    "CapCompP",
    "CapCoreCIninety",
    "CapCoreP",
    "ClOrgAgeCI",
    "ClOrgDlCI",
    "ClRepoAgeCI",
    "ClRepoDlCI",
    "CoreThrFiveCI",
    "CoreThrFiveEarlyA",
    "CoreThrFiveEarlyB",
    "CoreThrFiveEarlyCI",
    "CoreThrFiveEarlyOR",
    "CoreThrFiveHR",
    "CoreThrFiveN",
    "CoreThrTenN",
    "CoreThrThreeN",
    "EarlyCapiRateA",
    "EarlyCapiRateB",
    "EarlyCompRateA",
    "EarlyCompRateB",
    "EarlyCoreRateA",
    "EarlyCoreRateB",
    "EnvPyCore",
    "EnvPyTV",
    "EnvPyTVCI",
    "EnvWinCore",
    "EnvWinTV",
    "EnvWinTVCI",
    "HRAgeCIninety",
    "HRCapiCIninety",
    "HRCapiUnkCIninety",
    "HRCoreCIninety",
    "HRCoreMajor",
    "HRCoreNoPoetry",
    "HRDlCIninety",
    "HRDlMajor",
    "HRDlNoPoetry",
    "HRPoetryCIninety",
    "HRRelCIninety",
    "HoCapBin",
    "HoCapBuild",
    "HoCapComp",
    "HoEarlyBin",
    "HoEarlyComp",
    "HoEarlyCore",
    "LogitCapi",
    "LogitCapiCI",
    "LogitCore",
    "LogitCoreCI",
    "LogitDl",
    "LogitDlCI",
    "LxCapDep",
    "LxCompDep",
    "LxCompOk",
    "LxCompOkPct",
    "NCompiled",
    "PreAEarly",
    "PreAMajor",
    "PreAPoetry",
    "PreBEarly",
    "RtCapDep",
    "RtCapFailCI",
    "RtCapFailPct",
    "RtCapNT",
    "RtCapOk",
    "RtCapOkPct",
    "RtCapOwn",
    "RtCapOwnCI",
    "RtCapOwnPct",
    "RtCompDep",
    "RtCompFailCI",
    "RtCompFailPct",
    "RtCompNT",
    "RtCompOk",
    "RtCompOkPct",
    "RtCompOwn",
    "RtCompOwnCI",
    "RtCompOwnPct",
    "StAStillN",
    "StBStillN",
    "StEStillN",
}
# p-values built in loops whose running-text form (<name>Rel) is used, with or without the plain form
REL_NAMES = set()
REL_ONLY = {"HoCapCore"}


def mac(name: str, value, rel: bool | str = False) -> None:
    """Emit \\name; for a p-value, rel=True also emits \\nameRel and rel="only" emits just \\nameRel."""
    assert name.isalpha(), name
    if name in NOT_REPORTED:
        return
    rel = "only" if name in REL_ONLY else (True if name in REL_NAMES else rel)
    if rel != "only":
        LINES.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    if isinstance(value, PStr) and rel:
        rel = "{<}\\,.001" if value.startswith("{<}") else "=" + str(value)
        LINES.append(f"\\newcommand{{\\{name}Rel}}{{{rel}}}")


def n(x) -> str:
    x = int(round(float(x)))
    return f"{x:,}".replace(",", "{,}")


def pct(x, d=1) -> str:
    return f"{100 * float(x):.{d}f}"


def f2(x) -> str:
    return f"{float(x):.2f}"


def ci(lo_hi, fmt=f2) -> str:
    return f"[{fmt(lo_hi[0])}, {fmt(lo_hi[1])}]"


def days(x) -> str:
    return n(x)


def pval(p) -> str:
    p = float(p)
    if p < 0.001:
        return PStr("{<}.001")
    return PStr(f"{p:.3f}".lstrip("0") if p < 0.1 else f"{p:.2f}".lstrip("0"))


def main() -> None:
    m = json.loads((ROOT / "results" / "main_pre2024.json").read_text(encoding="utf-8"))
    F, R1, R2, R3, PV = m["funnel"], m["RQ1"], m["RQ2"], m["RQ3"], m["prevalence"]

    # funnel
    for k, v in {"FrameN": F["frame"], "Fetched": F["fetched"], "ScreenedIn": F["screened_in"],
                 "RelWindow": F["releases_in_window"], "RelPre": F["excluded_prerelease"],
                 "RelYanked": F["excluded_yanked"], "RelNoMeta": F["excluded_no_metadata"],
                 "RelUsable": F["usable_releases"], "RelMainline": F["mainline_releases"],
                 "NDep": F["dependent"], "NCompiled": F["compiled"], "NPure": F["pure"],
                 "NCapiBuild": F["capi"].get("build-numpy", 0), "NCapiNoBuild": F["capi"].get("no-build-numpy", 0),
                 "NCapiNoSdist": F["capi"].get("no-sdist", 0) + F["capi"].get("unknown", 0),
                 "NCore": F["core"], "NCapped": F["capped"]}.items():
        mac(k, n(v))
    mac("CapRate", pct(R1["cap_rate"]))

    # RQ1
    co = R1["cap_origin"]
    mac("NReactive", n(co["reactive"])); mac("NPreemptive", n(co["pre-emptive"])); mac("NLegacy", n(co["legacy"]))
    for k, v in co.items():
        mac({"reactive": "ShReactive", "pre-emptive": "ShPreemptive", "legacy": "ShLegacy"}[k], pct(v / F["capped"], 0))
    mac("ReactThirty", n(R1["reactive_within_30d"])); mac("ReactNinety", n(R1["reactive_within_90d"]))
    mac("ReactYear", n(R1["reactive_after_365d"]))
    cs = R1["cap_style"]
    mac("NMajorCap", n(cs.get("major-cap", 0))); mac("NMinorCap", n(cs.get("minor-cap", 0))); mac("NPin", n(cs.get("pin", 0)))
    po = R1["poetry"]
    mac("NPoetry", n(po["n"])); mac("NPoetryPre", n(po["by_origin"].get("pre-emptive", 0)))
    mac("NPoetryReact", n(po["by_origin"].get("reactive", 0))); mac("NPoetryLegacy", n(po["by_origin"].get("legacy", 0)))
    for key, pre in (("by_capi_vs_rest", "CapCapi"), ("by_compiled", "CapComp"), ("by_core", "CapCore")):
        d = R1[key]
        mac(pre + "RateA", pct(d["rate_true"])); mac(pre + "RateB", pct(d["rate_false"]))
        mac(pre + "OR", f2(d["OR"])); mac(pre + "CI", ci(d["OR_95ci"])); mac(pre + "CIninety", ci(d["OR_90ci"]))
        mac(pre + "P", pval(d["p"]))

    # prevalence
    mac("PrevZero", pct(PV["day_0"]["share"])); mac("PrevZeroN", n(PV["day_0"]["n"]))
    mac("PrevPeak", pct(PV["peak"]["share"]))
    mac("PrevPeakWeeks", n(PV["peak"]["day"] / 7))
    mac("FixedN", n(PV["fixed_cohort_n"])); mac("FixedDay", n(-PV["fixed_cohort_day"]))
    mac("FixedPeak", pct(PV["fixed_peak"]["fixed_share"])); mac("FixedPeakDay", n(PV["fixed_peak"]["day"]))
    mac("FixedYear", pct(PV["day_365"]["fixed_share"])); mac("FixedZero", pct(PV["day_0"]["fixed_share"]))
    for d, nm in ((180, "PrevHalf"), (365, "PrevYear"), (730, "PrevTwo")):
        mac(nm, pct(PV[f"day_{d}"]["share"]))

    # RQ2
    km = R2["km_delayed_entry"]
    mac("NEarly", n(R2["n_early"])); mac("EarlyDaysBefore", days(R2["early_median_days_before"]))
    mac("SurvN", n(km["n"])); mac("SurvEvents", n(km["events"])); mac("StillCapped", n(R2["still_capped"]))
    mac("DroppedLifts", n(R2["dropped_dependency_lifts"]))
    mac("MedDE", days(km["median"])); mac("MedDECI", ci(km["median_95ci"], days))
    for d, nm in ((30, "AdThirty"), (90, "AdNinety"), (180, "AdHalf"), (365, "AdYear")):
        mac(nm, pct(km["adapted_at"][f"{d}d"], 0))
    ro = R2["km_release_origin_no_entry"]
    mac("MedRO", days(ro["median"])); mac("MedROCI", ci(ro["median_95ci"], days))
    mac("MedDroppedCens", days(R2["sens_dropped_as_censored_median"]))
    cd = R2["cap_duration"]
    mac("DurReact", days(cd["reactive"]["median"])); mac("DurReactCI", ci(cd["reactive"]["median_95ci"], days))
    mac("DurPre", days(cd["pre-emptive"]["median"])); mac("DurPreCI", ci(cd["pre-emptive"]["median_95ci"], days))
    mac("MedNoPoetry", days(R2["sens_no_poetry"]["km"]["median"]))
    mac("MedMajor", days(R2["sens_major_caps"]["km"]["median"]))
    q = R2["by_quartile_median"]
    for k, nm in (("Q1", "QOne"), ("Q2", "QTwo"), ("Q3", "QThree"), ("Q4", "QFour")):
        mac(nm, days(q[k]))
    cx = R2["cox_delayed_entry"]
    mac("CoxN", n(cx["n"])); mac("CoxEvents", n(cx["events"])); mac("CoxC", f2(cx["concordance"]))
    mac("PHmin", f2(min(min(v.values()) for v in cx["ph_test_p_by_cut"].values())))
    mac("NCov", n(len(cx["terms"])))
    for k, nm in (("capi_i", "Capi"), ("capi_unknown_i", "CapiUnk"), ("log_downloads", "Dl"), ("core_i", "Core"),
                  ("releases_2023_per10", "Rel"), ("age_years_at_event", "Age"), ("poetry_i", "Poetry")):
        t = cx["terms"][k]
        mac(f"HR{nm}", f2(t["HR"])); mac(f"HR{nm}CI", ci(t["HR_95ci"])); mac(f"HR{nm}CIninety", ci(t["HR_90ci"]))
        mac(f"HR{nm}P", pval(t["p"])); mac(f"HR{nm}PH", pval(t["p_holm"]))
    for key, nm in (("sens_no_poetry", "NoPoetry"), ("sens_major_caps", "Major")):
        t = R2[key]["cox"]["terms"]
        mac(f"HRCore{nm}", f2(t["core_i"]["HR"])); mac(f"HRDl{nm}", f2(t["log_downloads"]["HR"]))
    t = R2["sens_known_capi"]["cox"]["terms"]
    mac("HRCapiKnown", f2(t["capi_i"]["HR"])); mac("HRCapiKnownCI", ci(t["capi_i"]["HR_95ci"]))
    e = R2["early"]
    for key, nm in (("by_capi", "EarlyCapi"), ("by_compiled", "EarlyComp"), ("by_core", "EarlyCore")):
        d = e[key]
        (a, b), (c, dd) = d["table"]
        mac(nm + "A", f"{a} of {a + b}"); mac(nm + "B", f"{c} of {c + dd}")
        mac(nm + "RateA", pct(d["rate_true"])); mac(nm + "RateB", pct(d["rate_false"]))
        mac(nm + "OR", f2(d["OR"])); mac(nm + "CI", ci(d["OR_95ci"]))
    mac("NEarlyKnown", n(e["n_early_known_capi"])); mac("NCappedKnown", n(e["n_capped_known_capi"]))
    kg = R2["km_group_n"]
    mac("KMnBuild", n(kg["build_numpy"])); mac("KMnOther", n(kg["other_known"])); mac("KMnUnk", n(kg["unknown"]))
    for k, nm in (("Q1", "QOne"), ("Q2", "QTwo"), ("Q3", "QThree"), ("Q4", "QFour")):
        mac(nm + "CI", ci(R2["by_quartile_km"][k]["median_95ci"], days))
    t = R2["sens_frame_indegree"]["cox"]["terms"]["log_frame_indeg"]
    mac("HRFrame", f2(t["HR"])); mac("HRFrameCI", ci(t["HR_95ci"]))
    t = e["logit_frame_indegree"]["terms"]["log_frame_indeg"]
    lt = e["logit"]
    for k, nm in (("capi_i", "Capi"), ("core_i", "Core"), ("log_downloads", "Dl")):
        mac(f"Logit{nm}", f2(lt["terms"][k]["OR"])); mac(f"Logit{nm}CI", ci(lt["terms"][k]["OR_95ci"]))

    # RQ3
    mac("PairsAll", n(R3["n_pairs_all"])); mac("PairsBoth", n(R3["n_pairs_both_lifted"])); mac("PairsTies", n(R3["ties"]))
    mac("PairsN", n(R3["n_pairs"])); mac("PairsAfter", n(round(R3["pair_share_after"] * R3["n_pairs"])))
    tu = R3["top_upstreams"]
    mac("UpstreamList", ", ".join(f"{k} ({v})" for k, v in tu.items()))
    mac("NDown", n(R3["n_downstreams"])); mac("DownAll", n(R3["downstreams_all_followed"]))
    mac("DownMin", n(R3["downstreams_minority"]))
    mac("DownShare", pct(R3["downstream_share_mean"])); mac("PermNull", pct(R3["perm_null_mean"]))
    mac("PermNullRange", f"{pct(R3['perm_null_95'][0])}--{pct(R3['perm_null_95'][1])}")
    mac("PermTwoNull", pct(R3["perm2_null_mean"]))
    mac("PermTwoRange", f"{pct(R3['perm2_null_95'][0])}--{pct(R3['perm2_null_95'][1])}")
    mac("PermTwoP", pval(R3["perm2_p"]), rel="only")
    mac("PermTwoExcess", f"{float(pct(R3['downstream_share_mean'])) - float(pct(R3['perm2_null_mean'])):.1f}")
    mac("PermOneP", pval(R3["perm_p"]), rel="only")
    mac("TVEventsUp", n(R3["tv_events_in_packages_with_upstream"]))
    mac("TVEventsBlocked", n(R3["tv_events_while_blocked"]))
    mac("TVN", n(R3["tv_n_with_capped_upstream"])); mac("TVHR", f2(R3["tv_blocked"]["HR"]))
    mac("TVCI", ci(R3["tv_blocked"]["HR_95ci"]))

    # environment sensitivity
    for env, nm in (("py39-linux", "EnvPy"), ("py312-windows", "EnvWin")):
        x = json.loads((ROOT / "results" / f"main_pre2024_{env}.json").read_text(encoding="utf-8"))
        mac(nm + "Capped", n(x["funnel"]["capped"])); mac(nm + "Med", days(x["RQ2"]["km_delayed_entry"]["median"]))
        mac(nm + "Core", f2(x["RQ2"]["cox_delayed_entry"]["terms"]["core_i"]["HR"]))
        mac(nm + "TV", f2(x["RQ3"]["tv_blocked"]["HR"])); mac(nm + "TVCI", ci(x["RQ3"]["tv_blocked"]["HR_95ci"]))
    # GitHub issue cross-check
    gfile = ROOT / "results" / "github_issues_pre2024.json"
    if gfile.exists():
        g = json.loads(gfile.read_text(encoding="utf-8"))
        # manual coding (data/github_issues_manual_coding.csv)
    # capped counts by build status (absolute)
    (a, b), (c, dd) = R1["by_capi_vs_rest"]["table"]
    mac("CapCapiA", f"{a} of {a + b}"); mac("CapCapiB", f"{c} of {c + dd}")

    # golden tests and differential test against uv
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_classify", ROOT / "tests" / "test_classify.py")
    tmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tmod)
    mac("GoldN", n(len(tmod.CASES) + 1))   # parametrised cases + environment-switch test
    dt = json.loads((ROOT / "results" / "difftest_uv.json").read_text(encoding="utf-8"))
    mac("DiffSets", n(dt["distinct_requirement_sets"])); mac("DiffReleases", n(dt["releases"]))
    mac("DiffAgreeSets", n(dt["agree_sets"])); mac("DiffAgreeReleases", n(dt["agree_releases"]))
    mac("DiffUv", dt["uv_version"].split()[1])
    # GitHub error-signature cross-check
    sfile = ROOT / "results" / "github_signatures_pre2024.json"
    if sfile.exists():
        g = json.loads(sfile.read_text(encoding="utf-8"))
        mac("SigRepo", n(g["with_repo"])); mac("SigSearch", n(g["searchable"]))
        mac("SigCapN", n(g["capped_searchable"])); mac("SigNevN", n(g["never_capped_searchable"]))
        mac("SigCapPct", pct(g["share_capped"])); mac("SigNevPct", pct(g["share_never"]))
        mac("SigOR", f2(g["OR"])); mac("SigORCI", ci(g["OR_95ci"]))
        mac("SigReact", n(g["reactive_reported"])); mac("SigReactBefore", n(g["reactive_report_before_cap"]))
        sd = pd.read_csv(ROOT / "data" / "github_signatures_pre2024.csv")
        mac("SigRepoDistinct", n(sd[sd.searchable].repo.nunique()))
        mac("SigReportedDistinctRepos", n(sd[sd.n_reports > 0].repo.nunique()))
        mac("SigReportedPkgs", n(int((sd.n_reports > 0).sum())))

    # validation (main frame)
    v = pd.read_csv(ROOT / "data" / "validation_pre2024_mainline.csv")
    mac("ValChecks", n(len(v)))

    # planned analysis (2026 frame)
    p = json.loads((ROOT / "results" / "results_top8000.json").read_text(encoding="utf-8"))
    P1, P2, P3 = p["RQ1"], p["RQ2"], p["RQ3"]
    mac("PNDep", n(P1["n_numpy_dependent"])); mac("PNCapped", n(P1["n_ever_capped"]))
    mac("PCapRate", pct(P1["n_ever_capped"] / P1["n_numpy_dependent"]))
    for k, nm in (("reactive", "PNReactive"), ("pre-emptive", "PNPreemptive"), ("legacy", "PNLegacy")):
        mac(nm, n(P1["cap_origin"][k]))
    h = P1["H1_fisher"]
    mac("PHOneOR", f2(h["odds_ratio"])); mac("PHOneCI", ci(h["or_95ci"])); mac("PHOneP", pval(h["p"]), rel="only")
    mac("PHOneRateA", pct(h["cap_rate_compiled"])); mac("PHOneRateB", pct(h["cap_rate_pure"]))
    mac("PNEarly", n(P2["adaptation_counts"]["before-release"]))
    mac("PSurvN", n(P2["survival_n"])); mac("PSurvEvents", n(P2["survival_events"]))
    mac("PMed", days(P2["km_median_days"])); mac("PMedCI", ci(P2["km_median_95ci"], days))
    for k, nm in (("compiled_i", "Comp"), ("log_downloads", "Dl"), ("releases_2023", "Rel"), ("age_years_at_event", "Age")):
        t = P2["H2_cox"][k]
        hr, lo, hi = t["HR"], t["HR_95ci"][0], t["HR_95ci"][1]
        if k == "releases_2023":
            hr, lo, hi = hr ** 10, lo ** 10, hi ** 10
        mac(f"PHR{nm}", f2(hr)); mac(f"PHR{nm}CI", ci([lo, hi])); mac(f"PHR{nm}P", pval(t["p"])); mac(f"PHR{nm}PH", pval(t["p_holm"]))
    mac("PCoxC", f2(P2["cox_concordance"])); mac("PPHmin", f2(min(P2["ph_test_p"].values())))
    mac("PPairs", n(P3["n_pairs_both_uncapped"] - P3["n_ties_same_day"])); mac("PPairsAfter", n(P3["n_downstream_after_upstream"]))
    mac("PPairShare", pct(P3["share_after"]))
    s = json.loads((ROOT / "results" / "sensitivity_top8000.json").read_text(encoding="utf-8"))
    meds = [v["km_median_days"] for k, v in s.items() if not k.startswith("_")]
    mac("PMedMin", days(min(meds))); mac("PMedMax", days(max(meds)))
    pa = json.loads((ROOT / "results" / "prefilter_audit_top8000.json").read_text(encoding="utf-8"))
    mac("PAuditDep", n(pa["numpy_dependent_in_window"])); mac("PAuditCapped", n(pa["capped_in_window"]))

    mac("TimelineN", n(m["timeline_n"]))
    # original 2026 frame, mainline + revision analyses (for the frame comparison table)
    o = json.loads((ROOT / "results" / "revision_top8000_mainline.json").read_text(encoding="utf-8"))
    ok = o["R2_release_origin_delayed_entry"]
    t = ok["cox"]["terms"]["log_downloads"]

    # supplementary set 4 (results/review4.json)
    r4 = json.loads((ROOT / "results" / "review4.json").read_text(encoding="utf-8"))
    bk = r4["bookkeeping"]
    mac("NEarlyCompiled", n(bk["early_compiled"]))
    mac("NEarlyBuild", n(bk["early_build_numpy"]))
    mac("NEarlyCore", n(bk["early_core"])); mac("NCappedCore", n(bk["capped_core"]))
    mac("ShReactiveExact", f"{co['reactive']} of {F['capped']}")
    er = r4["early_by_rc1"]
    mac("EarlyBeforeRC", n(er["before_rc1"])); mac("EarlyRCRel", n(er["rc1_to_release"]))
    mac("EarlyCoreRCRel", n(er["core_rc1_to_release"]))
    kp, kc = r4["km_periphery"], r4["km_core"]
    mac("MedPeri", days(kp["median"])); mac("MedPeriCI", ci(kp["median_95ci"], days)); mac("PeriN", n(kp["n"]))
    mac("MedCoreL", days(kc["median"])); mac("CoreLN", n(kc["n"]))
    for k, nm in (("3", "Three"), ("5", "Five"), ("10", "Ten")):
        t = r4["core_threshold"][k]
        mac(f"CoreThr{nm}N", n(t["n_core"])); mac(f"CoreThr{nm}HR", f2(t["cox_HR"])); mac(f"CoreThr{nm}CI", ci(t["cox_HR_95ci"]))
        mac(f"CoreThr{nm}EarlyA", t["early_core"]); mac(f"CoreThr{nm}EarlyB", t["early_periphery"])
        _a, _na = map(int, t["early_core"].split(" of ")); _c, _nc = map(int, t["early_periphery"].split(" of "))
        from statsmodels.stats.contingency_tables import Table2x2 as _T
        _tab = [[_a, _na - _a], [_c, _nc - _c]]
        _t = _T([[x + 0.5 for x in r] for r in _tab]) if 0 in sum(_tab, []) else _T(_tab)
        mac(f"CoreThr{nm}EarlyOR", f2(_t.oddsratio)); mac(f"CoreThr{nm}EarlyCI", ci(_t.oddsratio_confint()))
    fr = r4["firth"]["terms"]
    for k, nm in (("capi_i", "Capi"), ("core_i", "Core"), ("log_downloads", "Dl")):
        mac(f"Firth{nm}", f2(fr[k]["OR"])); mac(f"Firth{nm}CI", ci(fr[k]["OR_95ci"]))
    ff = r4["firth_frame"]["terms"]
    for k, nm in (("capi_i", "Capi"), ("log_frame_indeg", "Frame"), ("log_downloads", "Dl")):
        mac(f"FirthF{nm}", f2(ff[k]["OR"])); mac(f"FirthF{nm}CI", ci(ff[k]["OR_95ci"]))
    co_ = r4["collinearity"]
    mac("CorrDlCore", f2(co_["corr_downloads_core"])); mac("VIFmax", f2(max(co_["vif"].values())))
    mac("BelowListCapped", n(r4["below_2026_list_capped"]))
    for st, nm in zip(r4["stepwise"], ("A", "B", "C", "D", "E")):
        mac(f"St{nm}Capped", n(st["capped"])); mac(f"St{nm}Still", f"{st['still_capped']} of {st['cohort']}")
        mac(f"St{nm}StillN", n(st["still_capped"]))
        if nm == "A":   # identical to the planned analysis (Appendix A), reported with its own rounding
            st = {**st, "median": P2["km_median_days"], "median_95ci": P2["km_median_95ci"]}
        mac(f"St{nm}Med", days(st["median"])); mac(f"St{nm}MedCI", ci(st["median_95ci"], days))
        mac(f"St{nm}HR", f2(st["HR_downloads"])); mac(f"St{nm}HRCI", ci(st["HR_downloads_95ci"]))

    # supplementary set 6 (results/review6.json and audits)
    r6 = json.loads((ROOT / "results" / "review6.json").read_text(encoding="utf-8"))
    cl = r6["clusters"]
    mac("ClRepos", n(cl["cohort_repos"])); mac("ClOrgs", n(cl["cohort_orgs"]))
    for key, nm in (("cox_cluster_repo", "Repo"), ("cox_cluster_org", "Org")):
        t = r6[key]["core_i"]; mac(f"Cl{nm}CoreCI", ci(t["HR_95ci"]))
        t = r6[key]["age_years_at_event"]; mac(f"Cl{nm}AgeCI", ci(t["HR_95ci"]))
        t = r6[key]["log_downloads"]; mac(f"Cl{nm}DlCI", ci(t["HR_95ci"]))
    t = r6["logit_cluster_org"]
    q = r6["rq3_upstream_after_release"]
    mac("RqAPairs", n(q["pairs"])); mac("RqAPairsAfter", n(q["pairs_after"])); mac("RqADown", n(q["downstreams"]))
    mac("RqADownMaj", n(q["downstream_majority"])); mac("RqADownMin", n(q["downstream_minority"]))
    mac("RqAP", pval(q["sign_p"]), rel="only"); mac("RqALag", days(q["lag_median"]))
    q = r6["rq3_last_upstream_lag"]
    mac("RqLDown", n(q["downstreams"]))
    mac("RqLThirty", pct(q["share_within_30d_after"])); mac("RqLNull", pct(q["null_share_within_30d"]))
    mac("RqLNullRange", f"{pct(q['null_share_95'][0])}--{pct(q['null_share_95'][1])}"); mac("RqLP", pval(q["p_one_sided"]), rel="only")
    mac("RqLAbs", days(q["abs_lag_median"])); mac("RqLAbsNull", days(q["null_abs_lag_median"]))
    sp = r6["preemptive_split"]
    for k, nm in (("before_beta", "PreB"), ("after_beta", "PreA")):
        x = sp[k]
        mac(f"{nm}N", n(x["n"])); mac(f"{nm}Poetry", n(x["poetry"])); mac(f"{nm}Early", n(x["early_adapters"]))
        mac(f"{nm}Major", n(x["major_cap"])); mac(f"{nm}Med", days(x["cap_duration"]["median"]))
        mac(f"{nm}MedCI", ci(x["cap_duration"]["median_95ci"], days))
    o = r6["other"]
    mac("OthRel", n(o["releases_other"])); mac("OthPkg", n(o["packages_with_other"]))
    mac("OthBroadNew", n(len(o["packages_newly_capped_broad"])))
    rc = r6["recapped"]
    mac("Recap", n(rc["n"])); mac("RecapMed", days(rc["km_excluding"]["median"]))
    x = r6["lift_with_prereleases"]
    mac("PreRelMed", days(x["km"]["median"])); mac("PreRelMedCI", ci(x["km"]["median_95ci"], days))
    mac("PreRelEarlier", n(x["packages_earlier_lift"]))
    x = r6["lift_with_yanked"]
    mac("YankMed", days(x["km"]["median"])); mac("YankEarlier", n(x["packages_earlier_lift"]))
    if "binary_exposure" in r6:
        b = r6["binary_exposure"]
        a = b["agreement"]
        mac("BinN", n(a["n"])); mac("BinTP", n(a["tp"])); mac("BinFP", n(a["fp"])); mac("BinFN", n(a["fn"])); mac("BinTN", n(a["tn"]))
        mac("BinPrec", pct(a["precision"], 0)); mac("BinRec", pct(a["recall"], 0))
        uk = b["unknown_group"]
        mac("BinUnkUses", n(uk.get("uses-capi", 0))); mac("BinUnkNo", n(uk.get("no-capi", 0)))
        for key, nm in (("cap_rate_by_binary", "BinCap"), ("early_by_binary", "BinEarly"),
                        ("cap_rate_unknown_as_build", "BoundUp"), ("cap_rate_unknown_as_not", "BoundLo")):
            x = b[key]; (a1, b1), (c1, d1) = x["table"]
            mac(f"{nm}A", f"{a1} of {a1 + b1}"); mac(f"{nm}B", f"{c1} of {c1 + d1}")
            mac(f"{nm}OR", f2(x["OR"])); mac(f"{nm}CI", ci(x["OR_95ci"]))
        t = b["cox_binary"]["bin_i"]; mac("BinHR", f2(t["HR"])); mac("BinHRCI", ci(t["HR_95ci"]))
    pc = json.loads((ROOT / "results" / "poetry_check_pre2024.json").read_text(encoding="utf-8"))
    mac("PoChk", n(pc["with_pyproject"])); mac("PoAgree", n(pc["agree"])); mac("PoBoth", n(pc["wheel_yes_src_yes"]))
    mac("PoCaret", n(pc["poetry_caret_numpy"]))
    sa = json.loads((ROOT / "results" / "screen_audit_pre2024.json").read_text(encoding="utf-8"))
    mac("ScrPool", n(sa["pool"])); mac("ScrN", n(sa["sampled"])); mac("ScrDep", n(sa["dependent"]))
    mac("ScrUpper", n(sa["est_missed_dependent_95"][1]))
    PVb = m["prevalence"]
    mac("PrevStart", pct(PVb["day_-180"]["share"]))
    mac("PrevMinusNinety", pct(PVb["day_-90"]["share"]))

    # supplementary set 7 (results/review7.json)
    r7 = json.loads((ROOT / "results" / "review7.json").read_text(encoding="utf-8"))
    sc = r7["still_capped"]
    mac("StillInactive", n(sc["inactive"])); mac("StillArchived", n(sc["archived"])); mac("StillActive", n(sc["released_last_12m"]))
    k = r7["km_excluding_inactive_still_capped"]; mac("MedActive", days(k["median"])); mac("MedActiveCI", ci(k["median_95ci"], days))
    g = r7["github_adjusted"]
    mac("GhAdjOR", f2(g["capped_i"]["OR"])); mac("GhAdjCI", ci(g["capped_i"]["OR_95ci"]))
    e = r7["early_post_beta"]
    mac("EarlyPostBeta", n(e["n_early_post_beta"]))
    (a1, b1), (c1, d1) = e["by_core"]["table"]
    mac("EpbCoreA", f"{a1} of {a1 + b1}"); mac("EpbCoreB", f"{c1} of {c1 + d1}")
    mac("EpbCoreOR", f2(e["by_core"]["OR"])); mac("EpbCoreCI", ci(e["by_core"]["OR_95ci"]))
    q = r7["rq3_lag_all_upstreams_lifted"]
    bm = r7["binary_missing"]
    mac("BinMissing", n(bm["n"])); mac("BinNoWheel", n(bm["by_reason"].get("no-wheel", 0)))
    mac("BinUnscan", n(bm["by_reason"].get("unscanned", 0))); mac("BinMissCapped", n(bm["capped_among_missing"]))
    r6x = json.loads((ROOT / "results" / "review6.json").read_text(encoding="utf-8"))
    mac("RqATies", n(r6x["rq3_upstream_after_release"]["downstreams"] - r6x["rq3_upstream_after_release"]["downstream_majority"]
                     - r6x["rq3_upstream_after_release"]["downstream_minority"]))
    mac("NDownTies", n(q["eligible"] - m["RQ3"]["n_downstreams"]))

    # runtime import test (src/runtime_test.py)
    rt = ROOT / "data" / "runtime_test.csv"
    if rt.exists():
        import ast
        from scipy.stats import fisher_exact as _fx
        from statsmodels.stats.proportion import proportion_confint
        d = pd.read_csv(rt)
        t = d[d.outcome != "not-testable"]
        for dec, nm in (("capped", "Cap"), ("compatible", "Comp")):
            g = t[t.declared == dec]
            k = int((g.outcome == "runtime-incompatible").sum())
            lo, hi = proportion_confint(k, len(g), method="wilson")
            mac(f"Rt{nm}Tested", n(len(g))); mac(f"Rt{nm}Fail", n(k)); mac(f"Rt{nm}Ok", n(len(g) - k))
            mac(f"Rt{nm}FailPct", pct(k / len(g), 0)); mac(f"Rt{nm}OkPct", pct(1 - k / len(g), 0))
            mac(f"Rt{nm}FailCI", f"[{pct(lo, 0)}, {pct(hi, 0)}]")
            mac(f"Rt{nm}Sampled", n((d.declared == dec).sum()))
        tab = [[int(((t.declared == "capped") & (t.outcome == "runtime-incompatible")).sum()),
                int(((t.declared == "capped") & (t.outcome == "runtime-compatible")).sum())],
               [int(((t.declared == "compatible") & (t.outcome == "runtime-incompatible")).sum()),
                int(((t.declared == "compatible") & (t.outcome == "runtime-compatible")).sum())]]
        mac("RtTested", n(len(t)))
        fails = t[t.outcome == "runtime-incompatible"]
        def kind(x):
            e = " ".join(ast.literal_eval(x).get("failed", {}).values()) if isinstance(x, str) else ""
            if any(k in e for k in ("dtype size changed", "multiarray failed to import", "_ARRAY_API", "compiled using NumPy 1.x")):
                return "ABI"
            if "removed in the NumPy 2.0 release" in e:
                return "API"
            return "other"
        kinds = fails.np2.map(kind).value_counts()
        rj = json.loads((ROOT / "results" / "runtime_test.json").read_text(encoding="utf-8"))
        pop = rj["population_at_release"]
        mac("RtPopCap", n(pop["capped"])); mac("RtPopComp", n(pop["compatible"]))

    # supplementary set 8 (src/review8.py)
    r8f = ROOT / "results" / "review8.json"
    if r8f.exists():
        r8 = json.loads(r8f.read_text(encoding="utf-8"))
        g = r8["github_strict"]
        mac("SsCapRep", n(g["capped_reported"])); mac("SsNevRep", n(g["never_reported"]))
        mac("SsCapPct", pct(g["share_capped"])); mac("SsNevPct", pct(g["share_never"]))
        mac("SsDiff", pct(g["diff"])); mac("SsDiffCI", f"[{pct(g['diff_95ci'][0])}, {pct(g['diff_95ci'][1])}]")
        mac("SsOR", f2(g["OR"])); mac("SsORCI", ci(g["OR_95ci"]))
        mac("SsAdjOR", f2(g["adj_OR"])); mac("SsAdjCI", ci(g["adj_OR_95ci"]))
        mac("SsNevABI", n(g["never_reported_abi"])); mac("SsNevAPI", n(g["never_reported_api"]))
        o = r8["runtime_origin"]
        for dec, nm in (("capped", "Cap"), ("compatible", "Comp")):
            mac(f"Rt{nm}Own", n(o[dec]["own_failures"])); mac(f"Rt{nm}Dep", n(o[dec]["dependency_failures"]))
            mac(f"Rt{nm}OwnPct", pct(o[dec]["own_share"], 0))
            mac(f"Rt{nm}OwnCI", f"[{pct(o[dec]['own_95ci'][0], 0)}, {pct(o[dec]['own_95ci'][1], 0)}]")
            mac(f"Rt{nm}NT", n(o[dec]["not_testable"]))
        rp = r8["runtime_population"]
        mac("RtPopPre", n(rp["with_pre_event_release"])); mac("RtPopNone", n(rp.get("none", 0)))
        mac("RtPopOther", n(rp.get("other", 0)))
        h = r8["holm"]
        for fam, pre in (("rq1_cap_rate", "HoCap"), ("early_adaptation", "HoEarly")):
            for k, nm in (("build", "Build"), ("compiled", "Comp"), ("core", "Core"), ("binary", "Bin")):
                mac(f"{pre}{nm}", pval(h[fam][k]["p_holm"]))

    # Linux import test (src/runtime_test.py run under WSL Ubuntu) -- prefix Lx
    lxf = ROOT / "data" / "runtime_test_linux.csv"
    if lxf.exists():
        import ast
        from scipy.stats import fisher_exact as _fx
        from statsmodels.stats.proportion import proportion_confint
        d = pd.read_csv(lxf)
        t = d[d.outcome != "not-testable"]
        for dec, nm in (("capped", "Cap"), ("compatible", "Comp")):
            g = t[t.declared == dec]
            k = int((g.outcome == "runtime-incompatible").sum())
            lo, hi = proportion_confint(k, len(g), method="wilson")
            mac(f"Lx{nm}Sampled", n((d.declared == dec).sum())); mac(f"Lx{nm}NT", n(((d.declared == dec) & (d.outcome == "not-testable")).sum()))
            mac(f"Lx{nm}Tested", n(len(g))); mac(f"Lx{nm}Fail", n(k)); mac(f"Lx{nm}Ok", n(len(g) - k))
            mac(f"Lx{nm}FailPct", pct(k / len(g), 0)); mac(f"Lx{nm}OkPct", pct(1 - k / len(g), 0))
            mac(f"Lx{nm}FailCI", f"[{pct(lo, 0)}, {pct(hi, 0)}]")
        tab = [[int(((t.declared == x) & (t.outcome == "runtime-incompatible")).sum()),
                int(((t.declared == x) & (t.outcome == "runtime-compatible")).sum())] for x in ("capped", "compatible")]
        mac("LxFisherP", pval(_fx(tab)[1]), rel="only")
        nt = [[int(((d.declared == x) & (d.outcome == "not-testable")).sum()),
               int(((d.declared == x) & (d.outcome != "not-testable")).sum())] for x in ("capped", "compatible")]
        mac("LxAttrP", pval(_fx(nt)[1]), rel="only")
        mac("LxTested", n(len(t))); mac("LxNotTestable", n(len(d) - len(t)))
        def _kind(x):
            e = " ".join(ast.literal_eval(x).get("failed", {}).values()) if isinstance(x, str) else ""
            if any(k in e for k in ("dtype size changed", "multiarray failed to import", "_ARRAY_API", "compiled using NumPy 1.x")):
                return "ABI"
            if "removed in the NumPy 2.0 release" in e:
                return "API"
            return "other"
        kinds = t[t.outcome == "runtime-incompatible"].np2.map(_kind).value_counts()
        mac("LxABI", n(kinds.get("ABI", 0))); mac("LxAPI", n(kinds.get("API", 0))); mac("LxOther", n(kinds.get("other", 0)))
        r8 = json.loads((ROOT / "results" / "review8.json").read_text(encoding="utf-8"))
        if "runtime_origin_linux" in r8:
            o = r8["runtime_origin_linux"]
            for dec, nm in (("capped", "Cap"), ("compatible", "Comp")):
                mac(f"Lx{nm}Own", n(o[dec]["own_failures"])); mac(f"Lx{nm}Dep", n(o[dec]["dependency_failures"]))
                mac(f"Lx{nm}OwnPct", pct(o[dec]["own_share"], 0))
                mac(f"Lx{nm}OwnCI", f"[{pct(o[dec]['own_95ci'][0], 0)}, {pct(o[dec]['own_95ci'][1], 0)}]")
            unk = sum(1 for f in o["failures"] if f["origin"] == "unknown")
            mac("LxOwnN", n(o["capped"]["own_failures"] + o["compatible"]["own_failures"]))
            mac("LxDepN", n(o["capped"]["dependency_failures"] + o["compatible"]["dependency_failures"]))
            mac("LxUnkN", n(unk))
            wl = r8["windows_vs_linux"]
            mac("WlOverlap", n(wl["tested_both"])); mac("WlAgree", n(wl["agree"]))

    # supplementary set 9 (src/review9.py)
    r9f = ROOT / "results" / "review9.json"
    if r9f.exists():
        r9 = json.loads(r9f.read_text(encoding="utf-8"))
        e = r9["early_post_rc1"]
        mac("EarlyPostRC", n(e["n_early_post_rc1"]))
        for key, nm in (("by_core", "Core"), ("by_compiled", "Comp")):
            (a1, b1), (c1, d1) = e[key]["table"]
            mac(f"Erc{nm}A", f"{a1} of {a1 + b1}"); mac(f"Erc{nm}B", f"{c1} of {c1 + d1}")
            mac(f"Erc{nm}OR", f2(e[key]["OR"])); mac(f"Erc{nm}CI", ci(e[key]["OR_95ci"]))
        for key, nm in (("runtime_linux_fail_diff", "LxFailDiff"), ("runtime_linux_own_diff", "LxOwnDiff")):
            d = r9[key]
            mac(nm, f"{100 * d['diff']:.1f}".replace("-", "$-$"))
            mac(f"{nm}CI", f"[{100 * d['diff_95ci'][0]:.1f}, {100 * d['diff_95ci'][1]:.1f}]".replace("-", "$-$"))
            mac(f"{nm}Hi", f"{100 * d['diff_95ci'][1]:.0f}")

    # review 10: end of the GitHub window, follow-up length, derived ratios
    r10f = ROOT / "results" / "review10.json"
    if r10f.exists():
        r10 = json.loads(r10f.read_text(encoding="utf-8"))
        mac("AdGhEnd", pct(r10["adapted_at_github_window_end"], 0))
        mac("FollowYears", f"{r10['followup_years']:.1f}")
    med_main, med_major = float(km["median"]), float(R2["sens_major_caps"]["km"]["median"])
    mac("MajorShorter", f"{100 * (med_main - med_major) / med_main:.0f}")
    core_ns = [int(r4["core_threshold"][k]["early_core"].split(" of ")[1]) for k in ("3", "5", "10")]
    mac("CoreThrMinN", n(min(core_ns))); mac("CoreThrMaxN", n(max(core_ns)))
    # verbal descriptions used in the text must stay true for the data
    cap_rate, react_share = float(R1["cap_rate"]), co["reactive"] / F["capped"]
    assert 0.25 < cap_rate < 1 / 3, "'more than one in four' no longer holds"
    assert 0.40 <= react_share < 0.50, "'nearly half' no longer holds"
    assert 4.5 <= med_main / 30.44 < 5.5, "'about five months' no longer holds"

    # range of the sensitivity-analysis medians (reference environment, cap definitions,
    # pre-releases, yanked releases, re-introduced caps)
    import re as _re
    vals = {m.group(1): m.group(2) for m in (_re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}$", x) for x in LINES) if m}
    sens = [int(vals[k]) for k in ("EnvPyMed", "EnvWinMed", "MedNoPoetry", "MedMajor", "MedDroppedCens",
                                   "PreRelMed", "YankMed", "RecapMed") if k in vals and vals[k].isdigit()]
    if sens:
        mac("SensMin", n(min(sens))); mac("SensMax", n(max(sens)))

    out = ROOT / "paper" / "numbers.tex"
    out.parent.mkdir(exist_ok=True)
    out.write_text("% AUTO-GENERATED by src/make_numbers.py -- do not edit by hand.\n" + "\n".join(LINES) + "\n",
                   encoding="utf-8")
    print(f"{len(LINES)} macros -> {out}")


if __name__ == "__main__":
    main()
