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


def _module_level_imports(path):
    """Only the imports that run when the module is imported.

    `_top_level_imports` walks the whole tree, which is what the declared-against-imported
    checks want. It is the wrong question for the optional package rule. An import sitting
    inside `if args.track:` does not run on a clone that never passes that flag, so a
    script carrying one is not on the core path and does not need an exemption. Exempting
    the whole file instead would hide a later module level import in the same file, which
    is the failure the stale exemption check below exists to prevent.
    """
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)

    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
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
    # Only files that reach for an optional package at import time. tests/test_tracking.py
    # and tests/run_with_mlflow.py were on this list and came off it, because neither
    # actually imports mlflow at module level and the sharper reading above says so.
    allowed = {
        "mcr/tracking.py",
        "mcr/registry.py",
        "mcr/deploy.py",
        "tests/test_registry.py",
        "tests/test_deploy.py",
        "scripts/registry_probe.py",
        "scripts/drill.py",
    }
    used_the_exemption = set()
    guarded = 0

    for path in _source_files():
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        imported = {
            n for n in _module_level_imports(path) if n not in stdlib and n not in LOCAL
        }
        if (_top_level_imports(path) & optional) - imported:
            guarded += 1
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

    # The other way this check can quietly stop working. If the module level and whole
    # tree readings ever agree everywhere, the distinction above is doing nothing and the
    # scripts have either lost their guards or gained an exemption.
    assert guarded, "no file imports an optional package behind a guard, so the module "\
                    "level reading is not being tested against anything"


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


def _binds_and_uses(path, covered_modules):
    """Does this file import one of `covered_modules` and then actually use the name.

    This was a substring grep, looking for `registry.` in the file body. It
    was wrong in both directions. It matched an assert message that merely named
    `registry.promote`, which is how it first fired on a file with no such import, and it
    would have missed `from mcr import registry as reg` entirely, because the alias is
    never spelled the way the grep spells it.

    An import that is never used does not execute the module, so binding alone is not
    enough and the use has to be found too.
    """
    import ast

    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    leaves = {mod.split(".")[-1] for mod in covered_modules}
    aliases = set()
    direct = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "mcr":
                for alias in node.names:
                    if alias.name in leaves:
                        aliases.add(alias.asname or alias.name)
            elif node.module and node.module.split(".")[-1] in leaves:
                # from mcr.registry import promote. The symbol is the module's own code.
                if node.module.startswith("mcr."):
                    direct = True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("mcr.") and alias.name.split(".")[-1] in leaves:
                    # import mcr.registry binds `mcr`, so the use shows up as an attribute.
                    aliases.add(alias.asname or "mcr")

    if direct:
        return True
    if not aliases:
        return False
    return any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in aliases
        for node in ast.walk(tree)
    )


def check_the_narrow_registry_runner_covers_everything_that_could_kill_a_mutant():
    """A mutation oracle that misses a module reports survivors that are not survivors.

    tests/run_registry_checks.py exists because the full store runner is too slow to use as
    an oracle inside one shell call. Narrowing an oracle is only sound while no check
    outside it executes the module under test, so that is checked here rather than trusted.

    The list in that runner is hand written and this derives the answer from the imports,
    so the two have different sources and can therefore disagree.
    """
    sys.path.insert(0, ROOT)
    from tests.run_registry_checks import COVERS, MODULES

    covered_modules = {c[: -len(".py")].replace("/", ".") for c in COVERS}

    importers = set()
    for path in _source_files():
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        if not rel.startswith("tests/") or not rel.startswith("tests/test_"):
            continue
        names = _top_level_imports(path)
        if "mcr" not in names:
            continue
        if _binds_and_uses(path, covered_modules):
            importers.add(rel[: -len(".py")].replace("/", "."))

    assert importers, "no check module touches {}, so this check is reading nothing".format(
        COVERS
    )
    missing = sorted(importers - set(MODULES))
    assert not missing, (
        "tests/run_registry_checks.py is used as a mutation oracle for {} and these check "
        "modules touch those and are not in its list: {}".format(COVERS, missing)
    )


def check_the_oracle_coverage_reader_separates_a_real_import_from_a_mention():
    """A control on `_binds_and_uses`, because it replaced a grep that fired wrongly.

    Four cases with a known answer. Two of them are the exact failures of the substring
    version it replaced: a prose mention that matched, and an aliased import that did not.
    Without this, loosening the reader until nothing fails would look like a pass.
    """
    import tempfile
    import textwrap

    cases = [
        (False, "only names it in a message", '''
            from mcr import gate
            def check_x():
                assert gate.PROMOTE, "registry.promote moves the stage"
        '''),
        (True, "imports and uses it", '''
            from mcr import registry
            def check_x():
                assert registry.STAGES
        '''),
        (True, "imports it under another name", '''
            from mcr import registry as reg
            def check_x():
                assert reg.STAGES
        '''),
        (False, "imports it and never uses it", '''
            from mcr import registry
            def check_x():
                assert True
        '''),
        (True, "imports a symbol straight out of it", '''
            from mcr.deploy import Deployment
            def check_x():
                assert Deployment
        '''),
    ]

    covered = {"mcr.registry", "mcr.deploy"}
    with tempfile.TemporaryDirectory() as tmp:
        for want, label, body in cases:
            path = os.path.join(tmp, "case.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(textwrap.dedent(body))
            got = _binds_and_uses(path, covered)
            assert got is want, "{}: wanted {} and got {}".format(label, want, got)


def check_every_check_module_is_in_the_runner_ci_uses():
    """A check module nobody runs is worth less than no check module at all.

    tests/test_attest.py shipped into this repo listed in the narrow development runner and
    in neither of the two that matter, so twenty six checks passed on demand and would
    never have run again. The narrow runner is a measurement tool. The full one is what CI
    reads, so that is the list this compares against.

    Derived from the directory rather than from a second list, because a list compared
    against a list of the same provenance agrees with itself.
    """
    sys.path.insert(0, ROOT)
    from tests.run_all import MODULES as BASE
    from tests.run_with_mlflow import EXTRA

    on_disk = {
        "tests." + entry[: -len(".py")]
        for entry in sorted(os.listdir(os.path.join(ROOT, "tests")))
        if entry.startswith("test_") and entry.endswith(".py")
    }
    assert on_disk, "no check modules found, so this check is reading nothing"

    missing = sorted(on_disk - set(BASE) - set(EXTRA))
    assert not missing, (
        "these check modules exist and no runner CI uses lists them, so they never "
        "run: {}".format(missing)
    )


def _calls_create_version(path):
    """Every call to registry.create_version in one file, as line numbers.

    This reads the whole tree rather than `tree.body`. The import walk above deliberately
    reads only `tree.body`, because an import behind a branch does not run at import time
    and the question there is what executes on import. A call site is a different question
    and a call nested inside a function is exactly the one worth finding, so the two
    readers disagree on purpose.
    """
    import ast

    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)

    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "create_version":
            hits.append(node.lineno)
        elif isinstance(fn, ast.Name) and fn.id == "create_version":
            hits.append(node.lineno)
    return hits


def check_nothing_in_the_library_walks_round_the_admission_door():
    """`register` checks an artefact. `create_version` does not. Only one is the door.

    Fifteen of sixteen call sites moved to the unguarded twin when the signature changed,
    which left the guard on one production path and nothing enforcing the split. A comment
    recorded the decision and a comment is not a check.

    The rule is about library code. Inside `mcr/` only the registry itself may reach the
    primitive. Checks, probes and the drill all build stores full of fabricated hashes and
    none of them is claiming those bytes are a trained model, which is the case the
    primitive exists for, so they are not restricted here.

    Scoping it to `mcr/` rather than to a list of exempt filenames is deliberate. An
    exemption list grades what its author remembered on the day, and this repo has found
    that failure in several places already.
    """
    offenders = {}
    for path in _source_files():
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        if not rel.startswith("mcr/") or rel == "mcr/registry.py":
            continue
        hits = _calls_create_version(path)
        if hits:
            offenders[rel] = hits

    assert not offenders, (
        "these library modules call registry.create_version, which skips the admission "
        "rule that register applies: {}".format(offenders)
    )


def check_the_call_site_reader_finds_a_call_it_should_and_ignores_a_mention():
    """A control, because a reader that finds nothing would pass the check above forever.

    Four cases with a known answer. The nested call is the one `tree.body` cannot see, and
    the docstring mention is what a grep got wrong on this repo before.
    """
    import tempfile
    import textwrap

    cases = (
        ("registry.create_version(cli)", 1),
        ("def f():\n    return registry.create_version(cli)\n", 1),
        ('"""a docstring naming registry.create_version"""\n', 0),
        ("x = create_version\n", 0),
    )
    folder = tempfile.mkdtemp(prefix="callsite-")
    for i, (body, want) in enumerate(cases):
        path = os.path.join(folder, "case{}.py".format(i))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(textwrap.dedent(body))
        got = len(_calls_create_version(path))
        assert got == want, (body, got, want)
