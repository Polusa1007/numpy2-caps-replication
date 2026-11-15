"""Golden tests for the NumPy-status classification (paper §4.3).

Reference environment: CPython 3.12, Linux x86-64, no extras.
Run:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from classify import active_reqs, numpy_status, set_env  # noqa: E402

set_env("py312-linux")

CASES = [
    # --- plain specifiers -------------------------------------------------------
    (["numpy"], "compatible"),
    (["numpy>=1.21"], "compatible"),
    (["numpy>=2"], "compatible"),
    (["numpy>=2.1"], "compatible"),
    (["numpy<3"], "compatible"),
    (["numpy<2.1"], "compatible"),
    (["numpy<2.0.1"], "compatible"),
    (["numpy<=2.0"], "compatible"),
    (["numpy==2.0.0"], "compatible"),
    (["numpy==2.*"], "compatible"),
    (["numpy>=1.26,<2.3"], "compatible"),
    (["numpy!=2.0.0"], "compatible"),
    (["numpy<2"], "capped"),
    (["numpy<2.0"], "capped"),
    (["numpy<2.0.0"], "capped"),
    (["numpy<2.0.0a0"], "capped"),
    (["numpy<2,>=1.22"], "capped"),
    (["numpy>=1.22,<2"], "capped"),
    (["numpy<=1.26.4"], "capped"),
    (["numpy<1.27"], "capped"),
    (["numpy==1.*"], "capped"),
    (["numpy==1.26.*"], "capped"),
    (["numpy==1.26.4"], "capped"),
    (["numpy~=1.26.0"], "capped"),
    (["numpy~=1.26"], "capped"),
    (["numpy>=1.2,!=2.*"], "capped"),
    (["numpy<1.25"], "other"),
    (["numpy==1.24.4"], "other"),
    (["numpy>=1.27,<2"], "other"),
    ([], "none"),
    # --- legacy parenthesised form -------------------------------------------
    (["numpy (>=1.18,<2.0)"], "capped"),
    (["numpy (>=1.17)"], "compatible"),
    # --- extras are never active --------------------------------------------
    (['numpy>=1.20; extra == "all"'], "none"),
    (['numpy<2; extra == "test"', "numpy>=1.20"], "compatible"),
    (['numpy>=1.20', 'numpy<2; extra == "gpu"'], "compatible"),
    # --- similarly named packages are not numpy ---------------------------------
    (['numpydoc; extra == "docs"', "numpy>=1.20"], "compatible"),
    (["numpy-stl<4", "numpy>=1.20"], "compatible"),
    (["numpy-financial<2"], "none"),
    # --- Python-version markers (3.12 applies) ----------------------------------
    (['numpy<2; python_version < "3.12"', 'numpy>=1.26; python_version >= "3.12"'], "compatible"),
    (['numpy>=1.20; python_version < "3.12"', 'numpy<2,>=1.26; python_version >= "3.12"'], "capped"),
    (['numpy<2; python_version <= "3.11"'], "none"),
    (['numpy~=1.25.0; python_version < "3.12"', 'numpy~=1.26.0; python_version >= "3.12"'], "capped"),
    # --- platform markers (Linux applies) ------------------------------------
    (['numpy<2; platform_system == "Windows"', 'numpy>=1.22; platform_system != "Windows"'], "compatible"),
    (['numpy<2; sys_platform == "linux"', 'numpy>=1.22; sys_platform != "linux"'], "capped"),
    (['numpy>=1.22; platform_machine == "x86_64"'], "compatible"),
    # --- several active lines are intersected --------------------------------------
    (["numpy>=1.22", "numpy<2"], "capped"),
    (["numpy>=2", "numpy<2"], "other"),
]


@pytest.mark.parametrize("lines,expected", CASES)
def test_numpy_status(lines, expected):
    assert numpy_status(active_reqs(lines)) == expected


def test_environment_switch():
    lines = ['numpy<2; python_version < "3.10"', 'numpy>=1.22; python_version >= "3.10"']
    set_env("py39-linux")
    try:
        assert numpy_status(active_reqs(lines)) == "capped"
    finally:
        set_env("py312-linux")
    assert numpy_status(active_reqs(lines)) == "compatible"
