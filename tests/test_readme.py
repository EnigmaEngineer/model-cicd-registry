"""Does the README still describe this repo, for the numbers the repo already knows.

Two counts and one ratio in the README are facts about the files next to it. They were
typed, they went stale, and nothing noticed for long enough that both were wrong by more
than thirty. A number a document states about the repo it lives in is the one kind of stale
figure that is free to prevent, so it gets derived here and the message carries the right
answer.

The counts are taken by reading the files rather than by importing them. The full suite
needs a store and this check has to run on a machine that has none, and a skip would be
this module reporting green for doing nothing. Reading the files is also a second reader,
which is the point. `check_the_counter_agrees_with_the_real_collector` ties it back to the
collector the runners actually use, so the two cannot drift apart quietly.
"""

from __future__ import annotations

import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ROOT)

from tests.run_all import MODULES as BASE  # noqa: E402
from tests.run_with_mlflow import EXTRA  # noqa: E402


def count_checks(module: str) -> int:
    """Module level functions named check_, read off the file.

    Module level only, matching the collector, which reads `dir(mod)` and would not see a
    function nested inside another one either.
    """
    path = os.path.join(ROOT, module.replace(".", os.sep) + ".py")
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    return sum(
        1
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("check_")
    )


def _readme() -> str:
    with open(os.path.join(ROOT, "README.md"), "r", encoding="utf-8") as fh:
        return fh.read()


def check_the_readme_check_counts_are_current():
    without = sum(count_checks(m) for m in BASE)
    with_store = without + sum(count_checks(m) for m in EXTRA)

    text = _readme()
    found = re.search(
        r"carrying ([\d,]+) checks without\s+MLflow and ([\d,]+) with it", text
    )
    assert found, "the README no longer states its check counts in the expected sentence"

    said_without = int(found.group(1).replace(",", ""))
    said_with = int(found.group(2).replace(",", ""))
    assert (said_without, said_with) == (without, with_store), (
        "the README says {} without MLflow and {} with it. It is {} and {}".format(
            said_without, said_with, without, with_store
        )
    )


def check_the_counter_agrees_with_the_real_collector():
    """A control on the reader, over modules that need no store.

    Without this the count above could be derived by something that disagrees with what
    the runners collect, and both sides would be internally consistent and wrong.
    """
    from tests.runner import collect

    sample = ["tests.test_config", "tests.test_deps", "tests.test_attest"]
    for module in sample:
        got = len(collect([module]))
        read = count_checks(module)
        assert got == read, (module, got, read)


def check_every_module_the_readme_counts_is_a_real_file():
    """The counter raises on a missing file, so a typo in a runner list would read as an
    import error rather than as a wrong count. This names it instead."""
    for module in list(BASE) + list(EXTRA):
        path = os.path.join(ROOT, module.replace(".", os.sep) + ".py")
        assert os.path.isfile(path), path
