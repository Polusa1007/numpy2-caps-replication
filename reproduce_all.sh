#!/usr/bin/env bash
# Reproduce every result, figure, and number in the paper.
#
#   bash reproduce_all.sh            # offline: re-run all analyses from the cached raw data
#   bash reproduce_all.sh --collect  # also re-crawl PyPI/GitHub (cache misses only; hours)
#
# Requires Python >= 3.11 and: requests pandas packaging statsmodels lifelines matplotlib scipy pymupdf uv pytest
# Paper compilation additionally needs a LaTeX distribution (pdflatex + bibtex).
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--collect" ]]; then
  python src/collect.py --top 8000          # 2026 sample (planned analysis)
  python src/collect_v2.py                  # pre-event frame (main analysis)
fi

echo "== Planned analysis (2026 sample, Appendix A)"
python src/classify.py --top 8000
python src/classify.py --top 8000 --mainline
python src/analyze.py --top 8000
python src/analyze.py --top 8000 --variant _mainline
python src/sensitivity.py --top 8000
python src/sensitivity.py --top 8000 --variant _mainline
python src/rq3_robustness.py --top 8000
python src/audit_generator.py --top 8000 --variant _mainline
python src/revision_analyses.py --top 8000 --variant _mainline

echo "== Main analysis (pre-event frame)"
python src/classify.py --tag pre2024 --top 8000 --mainline
python src/audit_capi.py --tag pre2024
python src/audit_generator.py --frame pre2024 --variant _mainline
python src/validate.py --frame pre2024 --mainline
python src/frame_indegree.py
python src/analysis_v2.py
for env in py39-linux py312-windows; do
  python src/classify.py --tag pre2024 --top 8000 --mainline --env "$env"
  python src/analysis_v2.py --env "$env"
done
python src/review4.py
python src/audit_screen.py --n 300         # recall audit of the screening rule
python src/audit_wheel_capi.py            # binary-level NumPy C-API indicator
python src/audit_poetry.py                # Poetry indicator vs. sdist pyproject.toml
python src/difftest_uv.py                 # differential test against uv
python src/github_signatures.py           # uses cached GitHub search results when present
python src/review6.py
python src/review7.py
python src/review8.py   # uses the shipped runtime_test*.csv; nothing is installed or executed
python src/review9.py
# python src/runtime_test.py --n 50                    # Windows; installs and imports third-party packages (cached in data/raw/runtime)
# TMPDIR=~/rt_tmp python src/runtime_test.py --n 100  # Linux/WSL, main result (cached in data/raw/runtime_linux)
python -m pytest tests -q                 # golden classification tests

echo "== Numbers and paper"
python src/make_numbers.py
if [ -f paper/main.tex ] && command -v pdflatex >/dev/null 2>&1; then
  (cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
     && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
  echo "paper/main.pdf rebuilt"
else
  echo "paper sources or pdflatex not found: skipped paper compilation"
fi
