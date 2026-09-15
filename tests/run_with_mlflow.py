"""Every check in run_all.py, plus the ones that need a real MLflow store.

    python3 tests/run_with_mlflow.py

It fails rather than skips when mlflow is absent. A suite that skips its way to green is
the reason the tracking layer would sit broken for a week without anybody noticing, and
this runner exists precisely because tests/run_all.py cannot import the tracking module.

    pip install -r requirements-tracking.txt
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.run_all import MODULES  # noqa: E402
from tests.runner import run_checks  # noqa: E402

EXTRA = ["tests.test_tracking", "tests.test_registry", "tests.test_deploy"]


def main(argv) -> int:
    try:
        import mlflow  # noqa: F401
    except ImportError as exc:
        print("FAIL: mlflow is not importable: {}".format(exc))
        print("      pip install -r requirements-tracking.txt")
        return 1

    print("mlflow {}".format(mlflow.__version__))

    # Guard against the shape where this runner and run_all.py drift apart and this one
    # silently stops covering the base suite.
    modules = list(MODULES) + EXTRA
    if len(set(modules)) != len(modules):
        print("FAIL: a module is listed twice: {}".format(modules))
        return 1

    return run_checks(modules, verbose="-v" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
