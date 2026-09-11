"""The check collector.

It lives in its own module rather than inside `run_all.py` because a second runner will
eventually stand in for the first, and a runner that imports `run_all` to get the loop
would import itself once it has been copied over the top of it. That happened on an earlier
project of mine and cost an afternoon.

A check is a module level function whose name starts with `check_`. No decorators and no
class hierarchy. The whole thing is forty lines and it never needs to be more.
"""

from __future__ import annotations

import importlib
import traceback
from typing import Callable, List, Tuple


def collect(module_names: List[str]) -> List[Tuple[str, Callable]]:
    found = []
    for name in module_names:
        mod = importlib.import_module(name)
        for attr in sorted(dir(mod)):
            if attr.startswith("check_"):
                found.append(("{}.{}".format(name, attr), getattr(mod, attr)))
    return found


def run_checks(module_names: List[str], verbose: bool = False) -> int:
    checks = collect(module_names)

    # A collector that finds nothing prints "0 failed" and exits 0, which reads as a pass.
    # Refuse instead. This exact shape has shipped green on an earlier project of mine.
    if not checks:
        print("FAIL: collected zero checks from {}".format(module_names))
        return 1

    failures = []
    for name, fn in checks:
        try:
            fn()
            if verbose:
                print("  ok   {}".format(name))
        except Exception:
            failures.append((name, traceback.format_exc()))
            print("  FAIL {}".format(name))

    for name, tb in failures:
        print("\n--- {} ---\n{}".format(name, tb))

    print("{} passed, {} failed, {} checks".format(
        len(checks) - len(failures), len(failures), len(checks)
    ))
    return 1 if failures else 0
