"""Load a module out of `scripts/` by path.

`scripts/` is not a package and does not want to become one. Adding an `__init__.py` so two
check modules can import one file would put a package marker in a directory whose whole job
is to hold command line entry points.
"""

from __future__ import annotations

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_script(name):
    """`load_script("ci_local")` gives back the module object."""
    path = os.path.join(ROOT, "scripts", name + ".py")
    spec = importlib.util.spec_from_file_location("script_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
