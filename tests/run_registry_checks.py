"""Only the checks that exercise the registry and the deployment state.

    python3 tests/run_registry_checks.py

Measured on this machine: 15.4 seconds and 81 checks, against 28.6 seconds and 299 for
tests/run_with_mlflow.py. Both figures come from the runs recorded in docs/adr-0006 and
either can be reproduced by timing the two commands.

There are two reasons for it and the second is the one that made it necessary.

It is the inner loop while changing mcr/registry.py or mcr/deploy.py. The tracking checks
are the single most expensive thing in the full run and no mutation of either module can
reach them.

And a mutation pass needs an oracle that fits inside one call. A pass runs its oracle once
per mutation site and mcr/registry.py has 47 of them, so the full runner is twenty two
minutes of suite and the shell that drives it is killed at about 178 seconds. At 15.4
seconds a slice of eight sites plus its two controls fits with room to spare.

A narrower oracle is only honest if nothing outside it can kill a mutant in the modules it
covers. That is asserted rather than assumed, by tests/test_deps.py, which fails when a
module importing mcr.registry or mcr.deploy is missing from the list below.

CI runs the full tests/run_with_mlflow.py. This one is a development and measurement tool
and it is not a substitute for that.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.runner import run_checks  # noqa: E402

# The modules that import mcr.registry or mcr.deploy. Written out rather than derived,
# because a list derived from the same import scan the check below performs would agree
# with it whatever either one did.
COVERS = ("mcr/registry.py", "mcr/deploy.py")
MODULES = ["tests.test_registry", "tests.test_deploy"]


def main(argv) -> int:
    try:
        import mlflow  # noqa: F401
    except ImportError as exc:
        print("FAIL: mlflow is not importable: {}".format(exc))
        print("      pip install -r requirements-tracking.txt")
        return 1

    return run_checks(list(MODULES), verbose="-v" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
