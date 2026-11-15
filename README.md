# NumPy 2.0 Downstream Adaptation Study

Empirical study of how popular PyPI packages used version caps (`numpy<2`) around the NumPy 2.0 release (2024-06-16).
Replication package for the manuscript (under review). The manuscript sources are not included; `paper/numbers.tex` lists every data-derived number reported in it.

**Cached raw metadata:** `data/raw.tar.gz` (98 MB) is too large for the Git repository. On GitHub it is attached to the [v1.0 release](https://github.com/Polusa1007/numpy2-caps-replication/releases/tag/v1.0); the [Zenodo archive](https://doi.org/10.5281/zenodo.24249008) (DOI: 10.5281/zenodo.24249008) includes it. Download it into `data/` and extract it from the repository root with `tar -xzf data/raw.tar.gz` (creates `data/raw/`) before running `reproduce_all.sh`.

**One command:** `bash reproduce_all.sh` re-runs every analysis from the cached data and regenerates `paper/numbers.tex`.

## Layout

| Path | Content |
|---|---|
| `paper/numbers.tex` | Every data-derived number in the manuscript (generated; do not edit) |
| `docs/analysis_plan.md` | Stage-1 analysis plan, deviation log, list of revisions |
| `tests/test_classify.py` | Golden tests for the classification (`python -m pytest tests -q`) |
| `data/` | Cached raw JSON and derived CSVs (generated; not hand-edited) |
| `results/` | Results JSON and figures |

### Scripts — main analysis (pre-event frame, paper §4–6)

| Script | Purpose |
|---|---|
| `src/collect_v2.py` | Frame = top 8,000 on 2024-06-01; 5-probe screening; full release crawl |
| `src/classify.py` | Release- and package-level tables (`--tag`, `--mainline`, `--env`) |
| `src/audit_capi.py` | Builds-against-NumPy indicator from sdist build requirements |
| `src/audit_generator.py` | Build tool (Poetry) from the wheel `Generator` field |
| `src/frame_indegree.py` | Frame-wide pre-event reverse-dependency counts |
| `src/validate.py` | Validation against PEP 658 metadata |
| `src/analysis_v2.py` | Main analyses (prevalence, delayed-entry Cox, early adaptation, RQ3) |
| `src/difftest_uv.py` | Differential test of the classification against uv on all requirement sets |
| `src/github_signatures.py` | GitHub cross-check by NumPy 2 error signatures (no manual coding) |
| `src/audit_screen.py` | Recall audit of the five-probe screening rule |
| `src/audit_wheel_capi.py` | Binary-level NumPy C-API indicator from wheel extension modules |
| `src/audit_poetry.py` | Poetry indicator checked against sdist `pyproject.toml` |
| `src/review6.py` | Cluster-robust models, RQ3 lag analysis, beta split, other/recap/pre-release sensitivity |
| `src/review7.py` | Maintenance status of still-capped packages, adjusted GitHub association, post-beta early adaptation |
| `src/review8.py` | NumPy 2-specific GitHub error strings, origin of runtime failures (static), Holm adjustment for RQ1 and early adaptation |
| `src/review9.py` | Early adaptation counting only lifts after rc1; 95% CIs of the runtime failure-rate differences |
| `src/runtime_test.py` | Import test under NumPy 1.26.4 vs 2.0.0: Linux/WSL run (all 74 capped + 100 compatible releases, main result; outputs `*_linux`) and Windows run (50 + 50, replication). Installs and imports third-party code; run only in an environment you are willing to use for that, and on WSL set `TMPDIR` to a directory on disk |
| `src/review4.py` | Supplementary analyses (thresholds, Firth, VIF, stepwise design comparison) |
| `src/make_numbers.py` | Generates `paper/numbers.tex` |

### Scripts — planned analysis (2026 sample, paper Appendix A)

`src/collect.py`, `src/analyze.py`, `src/sensitivity.py`, `src/rq3_robustness.py`, `src/revision_analyses.py`, `src/audit_prefilter.py`.

## Reproduce

```bash
pip install requests pandas packaging statsmodels lifelines matplotlib scipy pymupdf uv pytest openpyxl
```

Main analysis:

```bash
python src/collect_v2.py
python src/classify.py --tag pre2024 --top 8000 --mainline
python src/audit_capi.py --tag pre2024
python src/audit_generator.py --frame pre2024 --variant _mainline
python src/validate.py --frame pre2024 --mainline
python src/frame_indegree.py
python src/analysis_v2.py
python src/classify.py --tag pre2024 --top 8000 --mainline --env py39-linux
python src/classify.py --tag pre2024 --top 8000 --mainline --env py312-windows
python src/analysis_v2.py --env py39-linux
python src/analysis_v2.py --env py312-windows
python src/difftest_uv.py
python src/github_signatures.py
python -m pytest tests -q
python src/make_numbers.py
```

Planned analysis (Appendix A):

```bash
python src/collect.py --top 8000
python src/classify.py --top 8000
python src/classify.py --top 8000 --mainline
python src/analyze.py --top 8000
python src/analyze.py --top 8000 --variant _mainline
python src/sensitivity.py --top 8000
python src/revision_analyses.py --top 8000 --variant _mainline
```


## License

Code: MIT (`LICENSE`). Data, results, and documentation: CC BY 4.0 (`LICENSE-DATA.md`).
