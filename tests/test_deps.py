"""Does the repo declare what it imports, and does it declare nothing else.

An earlier project of mine shipped its first day with no requirements file at all,
including the database driver every command in its README needed, and nothing caught it
for a day. A later one declared one package and imported five, which stayed invisible
because the build machine happened to have the other four.

So this walks the tree with ast and compares both directions. It is the cheapest check in
the repo and it is the one that decides whether a stranger can run anything.
"""

from __future__ import annotations

import ast
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Distribution name on the left, import name on the right, where they differ.
DIST_TO_IMPORT = {"PyYAML": "yaml"}

LOCAL = {"mcr", "tests", "scripts"}

# requirements.txt is what a clone needs to train a model. Anything in a requirements-*.txt
# is needed by one optional path and the runner for that path checks it imports. Splitting
# them means tests/run_all.py stays runnable on a machine with numpy and nothing else.
CORE = "requirements.txt"


def _source_files():
    found = []
    for folder in ("mcr", "tests", "scripts"):
        base = os.path.join(ROOT, folder)
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            if entry.endswith(".py"):
                found.append(os.path.join(base, entry))
    return found


def _top_level_imports(path):
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, which is always local by definition.
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


def _requirements_files():
    """Every requirements file, found rather than listed.

    A hard coded list is how a new optional requirements file ends up outside every check
    in this module while looking covered.
    """
    return sorted(glob.glob(os.path.join(ROOT, "requirements*.txt")))


def _declared_in(path):
    out = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            dist = line.split("==")[0].split(">=")[0].split("[")[0].strip()
            out.add(DIST_TO_IMPORT.get(dist, dist.lower().replace("-", "_")))
    return out


def _declared():
    """Everything declared anywhere. The right side of the is-it-declared question."""
    out = set()
    for path in _requirements_files():
        out |= _declared_in(path)
    return out


def _third_party():
    stdlib = set(sys.stdlib_module_names)
    found = set()
    for path in _source_files():
        for name in _top_level_imports(path):
            if name not in stdlib and name not in LOCAL:
                found.add(name)
    return found


def check_the_walk_actually_reads_something():
    """A check that can pass on zero inputs will eventually be pointed at zero inputs.

    If the folder names above ever go stale, every other check in this file passes by
    looking at nothing. This is the one that notices.
    """
    files = _source_files()
    assert len(files) >= 10, "only found {} source files".format(len(files))
    assert _third_party(), "found no third party imports at all, which cannot be right"

    found = _requirements_files()
    assert os.path.join(ROOT, CORE) in found, "requirements.txt is not in {}".format(found)
    for path in found:
        assert _declared_in(path), "{} declares nothing".format(os.path.basename(path))


def check_every_import_is_declared():
    missing = sorted(_third_party() - _declared())
    assert not missing, "imported and declared in no requirements file: {}".format(missing)


def check_nothing_on_the_core_training_path_needs_an_optional_package():
    """The split is only worth having if it is true.

    A clone that installs requirements.txt has to be able to train a model. If anything
    under mcr/ other than the tracking layer reached for mlflow, requirements.txt would be
    a lie and the second runner would be the only way to run anything.
    """
    optional = _declared() - _declared_in(os.path.join(ROOT, CORE))
    assert optional, "no optional packages, so this check is looking at nothing"

    stdlib = set(sys.stdlib_module_names)
    allowed = {"mcr/tracking.py", "tests/test_tracking.py", "tests/run_with_mlflow.py"}
    used_the_exemption = set()

    for path in _source_files():
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        imported = {
            n for n in _top_level_imports(path) if n not in stdlib and n not in LOCAL
        }
        leaked = sorted(imported & optional)
        if rel in allowed:
            if leaked:
                used_the_exemption.add(rel)
            continue
        assert not leaked, "{} imports the optional {}".format(rel, leaked)

    # An exemption nobody needs is an exemption that will one day cover a real leak. Every
    # name on the list has to be there for a reason the source can show.
    stale = sorted(allowed - used_the_exemption)
    assert not stale, "exempted from the optional rule and importing nothing optional: {}".format(stale)


def check_every_declared_package_is_imported():
    """The other direction. A requirements file listing something nobody imports makes a
    clone install weight it does not need, and it is usually the fossil of a dependency
    that was removed from the code and left in the file."""
    unused = sorted(_declared() - _third_party())
    assert not unused, "in requirements.txt and imported nowhere: {}".format(unused)


def check_the_core_requirements_are_importable():
    """Declaring it is not the same as it resolving. A version pin that cannot be
    satisfied together with another one fails here rather than on a stranger's machine.

    Only the core file. The optional packages are checked by the runner that needs them,
    tests/run_with_mlflow.py, which refuses to run rather than skipping. Importing them
    here would make tests/run_all.py fail on a machine that deliberately installed less.
    """
    import importlib

    for name in sorted(_declared_in(os.path.join(ROOT, CORE))):
        importlib.import_module(name)
