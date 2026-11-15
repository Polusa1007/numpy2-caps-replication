# Pre-registered Analysis Plan

**Working title:** Breaking the Ecosystem: An Empirical Study of Downstream Adaptation to the NumPy 2.0 Major Release

**Version:** 1.0 (written 2026-11-08, before any hypothesis-testing analysis on the full dataset)

**Target venues:** Empirical Software Engineering → Information and Software Technology → JSS → PeerJ CS

**Reporting standard:** ACM SIGSOFT Empirical Standards — "Repository Mining" standard

---

## 1. Background

NumPy 2.0.0 was released on **2024-06-16**. It broke the C ABI (extensions built against 1.x fail at import under 2.x), changed type-promotion rules (NEP 50), and removed/relocated many Python-level APIs. Many downstream packages responded by declaring an upper bound (`numpy<2`) in their published metadata, then later lifted it.

Because the release date is fixed and exogenous to downstream projects, the event is a natural experiment for studying how an ecosystem absorbs a major breaking change in a foundational library.

## 2. Research Questions and Hypotheses

**RQ1 — Prevalence and timing of defensive caps.**
How many NumPy-dependent packages declared an upper bound excluding NumPy 2, and was the cap *pre-emptive* (first appearing before 2024-06-16) or *reactive* (first appearing on/after it)?

- H1: Packages shipping compiled extensions are more likely to have capped than pure-Python packages. Test: Fisher's exact test (two-sided); effect size: odds ratio with 95% CI.

**RQ2 — Time to adaptation.**
For capped packages, how long until a release allows NumPy 2?

- Time origin: 2024-06-16 (NumPy 2.0.0 release); see deviation D2.
- Event: first subsequent release whose NumPy constraint admits `2.0.0`.
- Censoring: packages still capped at their latest release (as of data-collection date).
- Description: Kaplan–Meier curve, median time with 95% CI.
- H2 (two-sided): Time to adaptation differs by (a) compiled extension, (b) popularity (log downloads), (c) pre-event release cadence (releases in 2023), (d) project age. Model: Cox proportional hazards; report hazard ratios with 95% CI; PH assumption checked with Schoenfeld residuals. Holm correction across the four covariate tests.

**RQ3 — Propagation of blocking.**
When a capped package depends (non-optional) on another capped package in the sample, does it lift its cap only after its upstream does?

- Report the proportion of dependent pairs where downstream uncapping follows upstream uncapping, and the distribution of the lag (days).
- Test: one-sided sign test that downstream uncaps after upstream more often than 50%.

## 3. Data

| Item | Source |
|---|---|
| Sampling frame | Top 8,000 PyPI packages by 30-day downloads (hugovk/top-pypi-packages snapshot; date recorded) |
| Release list, upload times, wheel tags | PyPI JSON API `/pypi/<pkg>/json` |
| Per-release dependency metadata | PyPI JSON API `/pypi/<pkg>/<version>/json` → `info.requires_dist` |

**Observation window:** releases uploaded 2023-06-16 (one year before the event) through the collection date.

## 4. Operational Definitions

- **NumPy-dependent:** at least one release in the window has a non-optional `numpy` requirement.
- **Non-optional requirement:** a `requires_dist` entry for `numpy` whose environment marker evaluates true under a reference environment (CPython 3.12, Linux x86_64, no extras). Entries gated by `extra == ...` are excluded.
- **Capped release:** the evaluated specifier admits `1.26.4` but rejects `2.0.0`.
- **NumPy-2-compatible release:** the evaluated specifier admits `2.0.0`.
- **Compiled extension:** any wheel in the package's releases within the window carries a platform-specific tag (not `none-any`).
- Pre-releases, yanked releases, and releases with missing `requires_dist` are excluded from event detection; their counts are reported.

## 5. Statistical Conventions

- α = 0.05, two-sided unless stated.
- Effect sizes always reported (odds ratio, hazard ratio, Cliff's δ for any two-group continuous comparison).
- Holm–Bonferroni correction within each RQ.
- All scripts and the dataset snapshot will be archived (Zenodo) for replication.

## 6. Pilot Policy (anti p-hacking)

A pilot on the top 1,500 packages is used **only** to check feasibility: cohort sizes, data completeness, and correctness of the classification code. No hypothesis tests or models are run on pilot data. Any change to definitions after the pilot is logged in §8 with its reason.

## 7. Threats to Validity (planned discussion)

- **Construct:** declared metadata ≠ actual runtime compatibility; a package may be compatible without lifting a cap, or broken without a cap.
- **Construct:** resolvers may pick older, uncapped releases, so caps do not fully protect users.
- **External:** the most-downloaded packages are not representative of the whole ecosystem.
- **Internal:** the reference environment (Python 3.12) may misclassify packages with Python-version-specific constraints.

## 8. Deviation Log

All deviations below were made after the feasibility pilot (top 1,500) and **before** any hypothesis test or model was run. They change definitions only, never tests or thresholds.

| ID | Date | Change | Reason |
|---|---|---|---|
| D1 | 2026-11-08 | RQ1 cap origin gets a third category, **legacy**: the first in-window release is already capped and the package has releases before the window. | Pilot showed long-standing `numpy<2` caps (e.g. streamlit, scipy) that predate the NumPy 2 announcement; counting them as "pre-emptive" responses would be a construct error. |
| D2 | 2026-11-08 | RQ2 time origin is fixed at the NumPy 2.0.0 release (2024-06-16). Packages that lifted their cap before it are reported as **adapted-before-release** (descriptive, with negative day counts) and excluded from the survival model. The survival cohort is every capped package that had not adapted by release day, including reactive caps added after it; all durations are measured from release day. | Pilot showed 9/27 capped packages (core scientific stack) adapted during the NumPy 2 RC period; `max(event, first_cap)` produced negative durations. |
| D3 | 2026-11-08 | The cap also ends when a later release **drops the NumPy dependency**; recorded as `uncap_kind = dropped-dependency`. Sensitivity analysis will re-run RQ2 treating these as censored. | Pilot found a package that removed NumPy entirely; otherwise its end-of-cap was detected ~1.5 years late. |
| D4 | 2026-11-08 | "Compatible" means the specifier admits *any* of NumPy 2.0–2.5, not only 2.0.0. | Specifiers like `numpy>=2.1` are NumPy-2 compatible but reject 2.0.0. |
| D5 | 2026-11-08 | Sampling pre-filter: a package enters stage 2 only if its *latest* release mentions numpy. | Cuts per-release requests ~10x. Packages that dropped NumPy before the latest release are missed; reported as a limitation. |
| D6 | 2026-11-08 | Replaces D1's rule. **Legacy** = the last stable, non-yanked release *before* the window was already capped (its metadata is fetched directly). | Full-data sanity check (before any test was run) found D1 misclassified packages with no releases between 2023-06 and 2024-06: their first in-window release came after NumPy 2.0 (e.g. bayesian-optimization), so a reactive cap was labelled legacy. Legacy count fell 39 → 19. |
| D7 | 2026-11-08 | Report, descriptively, packages that re-introduce a cap after first lifting it (15 found). The RQ2 event remains the *first* lift. | Mostly maintenance releases on older branches uploaded after newer ones; not a change of adaptation status. |

**Post-analysis deviations.** D8–D9 were made *after* the pre-registered tests had been run, in response to a measurement-validation exercise (`src/validate.py`), not to the test results. The paper reports both the pre-registered and the corrected variants side by side.

| ID | Date | Change | Reason |
|---|---|---|---|
| D8 | 2026-11-08 | **Mainline variant** (`classify.py --mainline`): cap and lift events are detected only on *frontier* releases, i.e. releases whose version exceeds every earlier-uploaded usable release. | Validation of 30 sampled packages against wheel METADATA (PEP 658) showed 100% source agreement but 2/20 capped packages with a spurious lift caused by old-branch maintenance releases (bpy 4.2.13 LTS after 4.5.1; qiskit 0.46.1 after 1.0.2). Effect on results: 175 → 173 capped; all conclusions unchanged. |
| D9 | 2026-11-08 | Record **cap style** of the first capped release: major-cap (`<2`, `==1.*`), minor-cap (e.g. numba's routine `<1.27`), pin (`==1.26.4`, `~=1.26.0`). Sensitivity S4 keeps only major-cap packages. | Validation showed some "caps" are routine version policies, not NumPy-2-specific decisions. Distribution: major-cap 142, pin 17, minor-cap 14. |

**Exploratory analyses added after the planned analysis (2026-11-08)** — `src/revision_analyses.py`, `src/audit_generator.py`, `src/audit_prefilter.py`. None replaces a planned test; all are labelled exploratory in the paper. Mainline data (D8) unless noted.

- R1 early adaptation by code type (Fisher, logistic with layer + popularity; Cox with origin at NumPy 2.0.0rc1, 2024-03-30)
- R2 delayed entry (left truncation at first capped release); cap duration; weekly ecosystem prevalence of capped latest releases
- R3 90% CIs vs. equivalence margin [0.5, 2]; RMST(365) and log-rank instead of Cliff's δ; releases per 10
- R4 layers: core = in-sample non-optional in-degree ≥ 5 (threshold fixed before inspecting layer results); download quartiles
- R5 build tool from wheel `Generator` (Poetry caret caps)
- R6 RQ3 downstream-level tests, stratified permutation (upstream layer × download quartile), time-varying Cox
- R7 broad cap definition (admits some 1.x, no 2.x)
- Pre-filter audit: 300 sampled excluded packages

**Wording note:** this plan was written on the same day as data collection, after the feasibility pilot, and has no external timestamp. The paper therefore calls it a "time-stamped analysis plan fixed before testing", not a pre-registration.
