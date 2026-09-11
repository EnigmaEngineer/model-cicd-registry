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
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Distribution name on the left, import name on the right, where they differ.
DIST_TO_IMPORT = {"PyYAML": "yaml"}

LOCAL = {"mcr", "tests", "scripts"}


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


def _declared():
    path = os.path.join(ROOT, "requirements.txt")
    out = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            dist = line.split("==")[0].split(">=")[0].split("[")[0].strip()
            out.add(DIST_TO_IMPORT.get(dist, dist.lower().replace("-", "_")))
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


def check_every_import_is_declared():
    missing = sorted(_third_party() - _declared())
    assert not missing, "imported and not in requirements.txt: {}".format(missing)


def check_every_declared_package_is_imported():
    """The other direction. A requirements file listing something nobody imports makes a
    clone install weight it does not need, and it is usually the fossil of a dependency
    that was removed from the code and left in the file."""
    unused = sorted(_declared() - _third_party())
    assert not unused, "in requirements.txt and imported nowhere: {}".format(unused)


def check_everything_declared_is_importable():
    """Declaring it is not the same as it resolving. A version pin that cannot be
    satisfied together with another one fails here rather than on a stranger's machine."""
    import importlib

    for name in sorted(_declared()):
        importlib.import_module(name)
