"""Does the CI workflow name things this repo actually has.

The workflow has never run. There is no GitHub runner here, so the file is a description of
intended wiring and the usual evidence for one of these is that it went green somewhere.
That evidence is not available, which leaves two choices. Ship it unchecked and let it rot,
or check the part that does not need a runner.

Three things rot without a runner and all three are readable from here. A step can name a
script that was renamed or never written. A step can pass arguments the script's parser
refuses. And a step can branch on an exit code the script cannot return. The first fails on
the first merge after the rename and the second fails on every run. The third never fails at
all. It quietly takes the wrong branch forever.

The argument check was added after the first two passed on a workflow whose opening deploy
step could not have run. Name checks and exit code checks are checks on the two things the
author thought of. Handing the real argv to the real parser is a check on the thing itself.

Parsed with a small reader rather than PyYAML. The workflow is the only YAML in the repo and
adding a parser to requirements.txt so the test suite can read one file is a dependency the
training pipeline would then carry for nothing. The reader handles the shape this file has
and raises on anything it does not understand, which is the important half.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shlex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "ci.yml")


def _text():
    with open(WORKFLOW, "r", encoding="utf-8") as fh:
        return fh.read()


def _join_continuations(lines):
    """Fold `\\` continuations into one command.

    Written after the exit code check below fired on a line that was the tail of a wrapped
    `python3` call. The check was right about the rule and wrong about what a line is.
    """
    out = []
    pending = ""
    for line in lines:
        if line.endswith("\\"):
            pending += line[:-1].rstrip() + " "
            continue
        out.append((pending + line).strip())
        pending = ""
    if pending:
        out.append(pending.strip())
    return out


def _run_commands(text):
    """Every shell command under a `run:` key, block scalars included."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("run:"):
            rest = stripped[len("run:"):].strip()
            if rest and rest != "|":
                out.append(rest)
                i += 1
                continue
            indent = len(line) - len(line.lstrip())
            body = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                    break
                if nxt.strip():
                    body.append(nxt.strip())
                i += 1
            out.extend(_join_continuations(body))
            continue
        i += 1
    return out


def check_the_workflow_file_is_there():
    assert os.path.isfile(WORKFLOW), "no workflow at {}".format(WORKFLOW)


def check_every_script_the_workflow_runs_exists():
    missing = []
    for command in _run_commands(_text()):
        for token in re.findall(r"(?:scripts|tests)/[A-Za-z0-9_./-]+\.py", command):
            if not os.path.isfile(os.path.join(ROOT, token)):
                missing.append((token, command))
    assert not missing, "workflow runs scripts that do not exist: {}".format(missing)


def check_every_config_the_workflow_names_exists():
    missing = []
    for command in _run_commands(_text()):
        for token in re.findall(r"configs/[A-Za-z0-9_.-]+\.yml", command):
            if not os.path.isfile(os.path.join(ROOT, token)):
                missing.append((token, command))
    assert not missing, "workflow names configs that do not exist: {}".format(missing)


def check_every_requirements_file_the_workflow_installs_exists():
    missing = []
    for command in _run_commands(_text()):
        for token in re.findall(r"requirements[A-Za-z0-9_.-]*\.txt", command):
            if not os.path.isfile(os.path.join(ROOT, token)):
                missing.append(token)
    assert not missing, "workflow installs requirements files that do not exist: {}".format(
        missing
    )


def _python_invocations():
    """Every `python3 <something>.py ...` in the workflow, as an argv list.

    Shell redirects and anything after them are dropped, because they are the shell's
    business and not the parser's.
    """
    out = []
    for command in _run_commands(_text()):
        try:
            tokens = shlex.split(command)
        except ValueError:
            continue
        if not tokens or tokens[0] != "python3":
            continue
        clean = []
        for token in tokens[1:]:
            if token in (">>", ">", "|", "&&", ";"):
                break
            clean.append(token)
        if clean:
            out.append(clean)
    return out


def check_every_command_the_workflow_runs_actually_parses():
    """Run each workflow command through the parser of the script it calls.

    The two checks above this one ask whether the script exists and whether the exit codes
    line up. Both passed on a workflow whose very first deploy step called train.py with
    `--store` against a script that takes `--track`, and with a bare `--register` against a
    flag that needs a value. Neither would have run, and the only way to find out was a
    runner nobody has here.

    Checking names and codes is checking the two things I happened to think of. Handing the
    real argv to the real parser checks the thing itself.
    """
    parsers = {}
    checked = 0
    for argv in _python_invocations():
        script = argv[0]
        if not script.startswith("scripts/"):
            # tests/run_all.py and friends take no arguments and have no parser.
            assert len(argv) == 1, "{} is called with arguments it does not take".format(
                script
            )
            continue

        if len(argv) == 1:
            # Called bare. There are no arguments to read, so there is nothing to grade
            # and demanding a parser would be demanding one for its own sake.
            continue

        name = os.path.basename(script)[: -len(".py")]
        if name not in parsers:
            spec = importlib.util.spec_from_file_location(
                "wf_" + name, os.path.join(ROOT, script)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            parsers[name] = getattr(module, "build_parser", None)

        # The first version of this check skipped a script with no build_parser, and three
        # of them built their parser inside main. So the workflow's very first deploy step
        # was skipped by the check written to catch exactly that step, and a control run
        # against the known bad invocation came back green. Refusing is the fix. A script
        # the workflow passes arguments to has to expose the parser that reads them.
        factory = parsers[name]
        assert factory is not None, (
            "the workflow runs `{}` with arguments and {} has no build_parser, so nothing "
            "can check them".format(" ".join(argv), script)
        )
        parser = factory()

        try:
            parser.parse_args(argv[1:])
        except SystemExit as exc:
            raise AssertionError(
                "the workflow runs `{}` and {} refuses it with exit {}".format(
                    " ".join(argv), script, exc.code
                )
            )
        checked += 1

    assert checked >= 5, "only parsed {} commands, so this check is reading almost "\
        "nothing".format(checked)


def _exit_constants(script):
    """The values of the EXIT_* constants a script assigns at module level."""
    path = os.path.join(ROOT, "scripts", script)
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("EXIT_"):
                    found[target.id] = node.value.value
    return found


def check_the_workflow_branches_on_codes_the_scripts_can_return():
    """A branch on an exit code nothing returns is a branch that never runs.

    This is the check worth having. A step reading `verdict == '0'` against a script whose
    success code moved to something else takes the wrong path on every build and nothing
    fails, so there is no run to look at afterwards.
    """
    text = _text()
    for step_id, script in (("gate", "gate.py"), ("canary", "canary.py")):
        codes = set(str(v) for v in _exit_constants(script).values())
        assert codes, "no EXIT_ constants found in scripts/{}".format(script)
        branches = re.findall(
            r"steps\.{}\.outputs\.verdict\s*==\s*'(\d+)'".format(step_id), text
        )
        assert branches, "nothing in the workflow branches on the {} verdict".format(step_id)
        unreachable = [b for b in branches if b not in codes]
        assert not unreachable, (
            "the workflow branches on {} verdict {} and scripts/{} can only return "
            "{}".format(step_id, unreachable, script, sorted(codes))
        )


def check_a_step_that_reads_an_exit_code_captured_one():
    """`$?` has to be read before anything else overwrites it.

    A `run` block that calls a script, echoes something, then reads `$?` gets the echo's
    status. It is always zero, so the verdict is always success. Costs nothing to check and
    it is the kind of thing that looks right in a diff.
    """
    text = _text()
    blocks = re.findall(r"run: \|\n(.*?)(?=\n      - name|\n  \w|\Z)", text, re.DOTALL)
    seen = 0
    for block in blocks:
        lines = _join_continuations(
            [ln.strip() for ln in block.splitlines() if ln.strip()]
        )
        for i, line in enumerate(lines):
            if "$?" not in line:
                continue
            seen += 1
            assert i > 0, "a block reads $? on its first line"
            previous = lines[i - 1]
            assert previous.startswith("python3"), (
                "'{}' reads $? and the command before it is '{}', which sets its own "
                "status".format(line, previous)
            )
    # Without this the check passes on a workflow that captures no exit codes at all, and
    # the whole gate-then-deploy shape rests on it capturing two.
    assert seen == 2, "expected two exit code captures, found {}".format(seen)


def check_the_deploy_job_waits_for_the_checks():
    """Deploying without the suite passing is the failure this whole job order exists for."""
    text = _text()
    match = re.search(r"\n  deploy:\n(.*?)(?=\n  \w|\Z)", text, re.DOTALL)
    assert match, "no deploy job in the workflow"
    needs = re.search(r"needs:\s*\[([^\]]*)\]", match.group(1))
    assert needs, "the deploy job declares no needs"
    named = {n.strip() for n in needs.group(1).split(",")}
    assert "checks" in named, "deploy does not wait for checks, it waits for {}".format(named)


def check_the_drills_run_in_ci():
    """The drills are the only thing covering the crash paths, so a build that skips them
    is a build with no evidence about rollback at all."""
    assert any(
        "scripts/drill.py" in command for command in _run_commands(_text())
    ), "nothing in the workflow runs the failure drills"


def check_the_reader_refuses_a_shape_it_does_not_understand():
    """The parser above is hand written, so it needs a case where it must not stay quiet.

    A reader that returns an empty list on an unfamiliar file makes every check above pass
    having read nothing, which is the shape this repo keeps finding. So the commands list is
    asserted non empty against the real file rather than only against a fixture.
    """
    commands = _run_commands(_text())
    assert len(commands) >= 10, "only found {} run commands, the reader is not reading the "\
        "file".format(len(commands))

    sample = "\n".join([
        "    steps:",
        "      - name: one",
        "        run: python3 scripts/drill.py",
        "      - name: two",
        "        run: |",
        "          echo hello",
        "          python3 tests/run_all.py",
        "      - name: three",
        "        uses: actions/checkout@v4",
    ])
    assert _run_commands(sample) == [
        "python3 scripts/drill.py",
        "echo hello",
        "python3 tests/run_all.py",
    ], _run_commands(sample)
