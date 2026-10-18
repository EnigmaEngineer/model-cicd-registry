"""Does the CI workflow name things this repo actually has.

There is still no GitHub runner here. What changed is that the file gets executed anyway,
by `scripts/ci_local.py`, which runs a job in local shells with the step outputs captured.
That covers the sequence. It does not cover the runner image or the python version, so the
checks below are still worth having and the division of labour is now deliberate rather
than forced.

What belongs here is whatever is cheap and reads the file. What belongs in an execution is
whatever depends on the state one command leaves for the next.

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
import os
import re
import shlex

from tests.loader import ROOT, load_script

# One reader for this file. The checks below used to scrape `run:` lines with a second pass
# that knew nothing about jobs or steps, which is how a comment between two steps ended up
# inside the block above it. `scripts/ci_local.py` has to parse the structure anyway in
# order to execute a job, so the structure is what everything reads now.
ci = load_script("ci_local")

WORKFLOW = ci.WORKFLOW


def _text():
    return ci.workflow_text()


def _join_continuations(lines):
    return ci.join_continuations(lines)


def _run_commands(text):
    return ci.run_commands(text)


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
            parsers[name] = getattr(load_script(name), "build_parser", None)

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
    """The reader is hand written, so it needs a case where it must not stay quiet.

    A reader that returns an empty list on an unfamiliar file makes every check above pass
    having read nothing. The shapes it refuses are graded in tests/test_ci_local.py, next
    to the reader itself. What is asserted here is the half that matters to these checks,
    which is that the real file is being read rather than skipped.
    """
    commands = _run_commands(_text())
    assert len(commands) >= 10, "only found {} run commands, the reader is not reading the "\
        "file".format(len(commands))

    try:
        ci.read_workflow("steps:\n  - name: no job around this\n")
    except ci.WorkflowError:
        pass
    else:
        raise AssertionError("the reader accepted a fragment with no jobs in it")


def check_a_canary_step_runs_only_when_a_canary_was_opened():
    """Found by running the deploy job rather than by reading it.

    `read the canary` carried no condition, so a rejected candidate reached it. The gate
    exits 1, `open the canary` is correctly skipped, and the canary read then runs against
    a store where nothing is aliased `canary`. Measured: canary.py refuses with exit 2 and
    the block's `exit 0` reported the step green.

    Every static check passed on that version. The names were right, the flags parsed, and
    both branch codes were reachable. What none of them read was that one step had a guard
    and the next one needed the same guard.
    """
    doc = ci.read_workflow(_text())
    steps = doc["jobs"]["deploy"]["steps"]

    opener = None
    for step in steps:
        if "run" in step and "deploy.py open" in step["run"]:
            opener = step
            break
    assert opener is not None, "nothing in the deploy job opens a canary"
    guard = opener.get("if")
    assert guard, "`deploy.py open` has no condition, so it runs on a rejected candidate"

    reached = False
    ungated = []
    for step in steps:
        if step is opener:
            reached = True
            continue
        if not reached or "run" not in step:
            continue
        if "canary" not in step["run"]:
            continue
        own = step.get("if")
        if own in (guard, "always()"):
            continue
        ungated.append((step.get("name"), own))
    assert not ungated, (
        "these steps touch the canary after it is opened and are not gated on {!r}: "
        "{}".format(guard, ungated)
    )


def check_neither_capture_step_calls_a_refusal_a_verdict():
    """A refusal means the script could not answer, which is a broken build.

    The gate step branched on its refusal code from the start. The canary step did not, so
    exit 2 was written into the output as though it were a verdict and then matched none of
    the branches reading it. The job went green having decided nothing.

    The code is read out of each script rather than typed here, so moving a constant fails
    this check instead of silently making it meaningless.
    """
    text = _text()
    for step_id, script in (("gate", "gate.py"), ("canary", "canary.py")):
        refuse = _exit_constants(script).get("EXIT_REFUSE")
        assert refuse is not None, "scripts/{} has no EXIT_REFUSE".format(script)

        block = text[text.index("id: {}".format(step_id)):]
        block = block[: block.index("$GITHUB_OUTPUT")] + block[block.index("$GITHUB_OUTPUT"):]
        block = block[: block.index("- name:", block.index("$GITHUB_OUTPUT"))]
        assert 'if [ "$code" = "{}" ]'.format(refuse) in block, (
            "the {} step writes exit {} into its output and never branches on it, so a "
            "refusal reaches the next step as a verdict".format(step_id, refuse)
        )


def check_every_adr_is_reachable_from_the_readme():
    """An ADR nobody links is an ADR nobody reads.

    `docs/adr-0006` sat unreferenced for a day while the README carried a whole section on
    the finding it documents. Every other ADR was linked, so the gap was invisible by
    inspection and only showed up when the two lists were diffed.

    This is cheap and it only proves the link exists, not that it points at the right
    paragraph.
    """
    import re

    docs = os.path.join(ROOT, "docs")
    adrs = sorted(f for f in os.listdir(docs) if f.startswith("adr-") and f.endswith(".md"))
    assert len(adrs) >= 6, "found {} adrs, which cannot be right".format(len(adrs))

    with open(os.path.join(ROOT, "README.md"), "r", encoding="utf-8") as fh:
        readme = fh.read()

    missing = []
    for name in adrs:
        stem = re.match(r"(adr-\d+)", name).group(1)
        if stem not in readme:
            missing.append(name)
    assert not missing, "these adrs are not linked from the README: {}".format(missing)


def check_every_adr_the_readme_names_actually_exists():
    """The other direction. A link to an ADR that was renamed is worse than no link."""
    import re

    docs = os.path.join(ROOT, "docs")
    present = {
        re.match(r"(adr-\d+)", f).group(1)
        for f in os.listdir(docs)
        if f.startswith("adr-") and f.endswith(".md")
    }

    named = set()
    for path in ("README.md",):
        with open(os.path.join(ROOT, path), "r", encoding="utf-8") as fh:
            named |= set(re.findall(r"adr-\d+", fh.read()))

    assert named, "the README names no adr at all, so this check reads nothing"
    dangling = sorted(named - present)
    assert not dangling, "the README points at adrs that do not exist: {}".format(dangling)


def check_the_canary_has_an_incumbent_to_compare_against():
    """The defect that took the first real execution of this workflow.

    `deploy.py open` refuses twice over, and both refusals are correct. With nothing in
    production there is no control arm. With the candidate already promoted into production
    there is only a model against itself. The runner starts with an empty workspace, so the
    store is new on every run and neither condition fixes itself.

    Every static check passed on the broken version, because each command was spelled
    correctly and every exit code was reachable. What none of them read was the state the
    commands leave for each other. This one reads exactly that and nothing else.

    It is deliberately narrow. It does not simulate the job. It asserts the two things that
    have to be true before an `open` can succeed, and the second one is an absence.
    """
    text = _text()

    deploy = text[text.index("\n  deploy:"):]

    # Comments in this job quote the very commands being checked for, because they explain
    # the failure that produced the check. Reading them as commands would mean a deleted
    # step still passes while its explanation survives. Same defect as the substring oracle
    # guard in tests/test_deps.py, found the same day, so it does not get shipped twice.
    ran = "\n".join(
        line for line in deploy.splitlines() if not line.lstrip().startswith("#")
    )

    assert "deploy.py open" in ran, "the deploy job no longer opens a canary"
    deploy = ran

    # One. Something must reach production before the canary opens.
    seed = re.search(r"promote\s+\S+\s+production", deploy)
    assert seed, (
        "nothing in the deploy job puts a version into production before "
        "`deploy.py open`, so the canary has no control arm on a fresh store"
    )
    assert deploy.index(seed.group(0)) < deploy.index("deploy.py open"), (
        "production is seeded after the canary opens, which is too late"
    )

    # Two. The gate must not promote, or the canary gets the version it just promoted.
    gate_step = deploy[deploy.index("scripts/gate.py"):]
    gate_cmd = gate_step[: gate_step.index("code=$?")]
    assert "--promote" not in gate_cmd, (
        "the deploy job's gate passes --promote, which moves production before the canary "
        "runs and leaves `deploy.py open` canarying the version it just promoted"
    )

    # And the thing that does move production has to still be there.
    assert "deploy.py land" in deploy, (
        "nothing lands the canary, so a promoted candidate never reaches production"
    )
