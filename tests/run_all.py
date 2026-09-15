"""Run every check in the repo.

    python3 tests/run_all.py

Exit code is what CI reads, so nothing here prints a summary it has not earned.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.runner import run_checks  # noqa: E402

MODULES = [
    "tests.test_config",
    "tests.test_seed",
    "tests.test_data",
    "tests.test_model",
    "tests.test_artifact",
    "tests.test_train",
    "tests.test_gate",
    "tests.test_canary",
    "tests.test_deps",
    "tests.test_workflow",
]


if __name__ == "__main__":
    sys.exit(run_checks(MODULES, verbose="-v" in sys.argv))
