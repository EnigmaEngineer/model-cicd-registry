"""Grade the local workflow runner.

The runner exists because the static checks on `ci.yml` could all pass while the file
described a sequence the runner cannot produce. That means this module is checking the
thing that checks the thing, so the controls matter more than usual. A reader that quietly
returns nothing would make the execution report green on an empty job.

Nothing here trains a model. The execution costs minutes and belongs in a command somebody
runs on purpose. What is graded here is the reading, the substitution and the conditions,
which are the parts that decide which commands run at all.
"""

from __future__ import annotations

import os

from tests.loader import ROOT, load_script

ci = load_script("ci_local")

# A full workflow in the shape the reader has to handle. Small enough to reason about and
# it carries the two things that broke today, which are a comment between steps and a
# block scalar.
SAMPLE = "\n".join([
    "name: sample",
    "env:",
    "  STORE: sqlite:///${{ github.workspace }}/x.db",
    "jobs:",
    "  one:",
    "    runs-on: ubuntu-latest",
    "    steps:",
    "      - uses: actions/checkout@v4",
    "      - name: first",
    "        run: |",
    "          echo hello",
    "          echo world",
    "",
    "      # A note about the step below, sitting at the step indent.",
    "      - name: second",
    "        id: two",
    "        if: github.event_name == 'push'",
    "        run: python3 scripts/drill.py",
])


def check_the_real_workflow_reads():
    doc = ci.read_workflow(ci.workflow_text())
    assert set(doc["jobs"]) == {"checks", "probes", "deploy"}, sorted(doc["jobs"])
    assert doc["env"]["MCR_MODEL"] == "fraud-scorer", doc["env"]


def check_the_reader_keeps_a_comment_between_steps_out_of_the_block_above_it():
    """The defect the first local run surfaced.

    A `#` line at the step indent is a note about the next step. The first version of the
    block scalar collector treated it as block content, so `install` came back carrying the
    eight comment lines that explain the step after it. Those would have gone to a shell.
    """
    doc = ci.read_workflow(SAMPLE)
    first = doc["jobs"]["one"]["steps"][1]
    assert first["run"] == "echo hello\necho world", repr(first["run"])
    assert "#" not in first["run"], repr(first["run"])

    # And the real file, which is where it was found.
    real = ci.read_workflow(ci.workflow_text())
    install = [s for s in real["jobs"]["deploy"]["steps"] if s.get("name") == "install"][0]
    assert install["run"].splitlines() == [
        "pip install -r requirements.txt",
        "pip install -r requirements-tracking.txt",
    ], install["run"]


def check_a_comment_indented_into_a_block_is_kept():
    """The other side of the rule above.

    A `#` line indented deeper than the `run:` key really is part of the script. Deleting
    it would be the fix overshooting, and a shell comment inside a block is a normal thing
    to write.
    """
    text = "\n".join([
        "jobs:",
        "  one:",
        "    steps:",
        "      - name: first",
        "        run: |",
        "          # why this is here",
        "          echo hello",
    ])
    step = ci.read_workflow(text)["jobs"]["one"]["steps"][0]
    assert step["run"] == "# why this is here\necho hello", repr(step["run"])


def check_the_reader_refuses_what_it_cannot_read():
    """Five shapes it must raise on rather than return something empty for."""
    cases = {
        "no jobs": "name: x\non:\n  push:\n    branches: [main]\n",
        "job with no steps": "jobs:\n  one:\n    runs-on: ubuntu-latest\n",
        "steps not a list": "jobs:\n  one:\n    steps:\n      name: nope\n",
        "folded scalar": "jobs:\n  one:\n    steps:\n      - name: a\n        run: >\n          echo hi\n",
        "key with no value": "jobs:\n  one:\n    steps:\n      - name: a\n        run:\n",
    }
    for label, text in cases.items():
        try:
            ci.read_workflow(text)
        except ci.WorkflowError:
            continue
        raise AssertionError("the reader accepted {!r}".format(label))


def check_the_reader_is_actually_reading_the_file():
    """A reader that returns an empty list makes every check above pass having read nothing.

    So the count is asserted against the real file rather than only against the fixture.
    """
    commands = ci.run_commands(ci.workflow_text())
    assert len(commands) >= 20, "only found {} commands".format(len(commands))
    assert any("scripts/drill.py" in c for c in commands), commands


def check_run_commands_comes_out_of_the_reader():
    """One reader for the file.

    The command inventory used to be scraped by a second pass over the lines. Two readers
    of one file drift and the one that gets fixed is never both.
    """
    assert ci.run_commands(SAMPLE) == [
        "echo hello",
        "echo world",
        "python3 scripts/drill.py",
    ], ci.run_commands(SAMPLE)


def check_a_wrapped_command_is_one_command():
    joined = ci.join_continuations(["python3 x.py \\", "--flag value", "echo done"])
    assert joined == ["python3 x.py --flag value", "echo done"], joined


def check_substitution_refuses_an_expression_it_does_not_know():
    """Leaving an unresolved expression in place sends it to a shell as a literal.

    A step running against `${{ secrets.TOKEN }}` as text is a command pointed at the wrong
    thing, and it would not look wrong in the output.
    """
    context = {"github.workspace": "/w"}
    assert ci.resolve("sqlite:///${{ github.workspace }}/r.db", context) == "sqlite:////w/r.db"
    try:
        ci.resolve("${{ secrets.TOKEN }}", context)
    except ci.WorkflowError:
        return
    raise AssertionError("an unknown expression was substituted rather than refused")


def check_a_condition_is_answered_or_refused():
    context = {"steps.gate.outputs.verdict": "0", "github.event_name": "push"}
    assert ci.evaluate_if("always()", context) is True
    assert ci.evaluate_if("steps.gate.outputs.verdict == '0'", context) is True
    assert ci.evaluate_if("steps.gate.outputs.verdict == '1'", context) is False
    assert ci.evaluate_if("steps.gate.outputs.verdict != '1'", context) is True
    assert ci.evaluate_if("github.event_name == 'pull_request'", context) is False

    # A step that was skipped leaves no output, and a runner reads that as the empty string.
    assert ci.evaluate_if("steps.canary.outputs.verdict == '0'", context) is False

    for bad in ("success() && github.ref == 'refs/heads/main'", "!cancelled()", "1 == 1"):
        try:
            ci.evaluate_if(bad, context)
        except ci.WorkflowError:
            continue
        raise AssertionError("the evaluator answered {!r} rather than refusing".format(bad))


def check_the_plan_refuses_a_job_that_is_not_there():
    doc = ci.read_workflow(ci.workflow_text())
    try:
        ci.plan(doc, "nonexistent")
    except ci.WorkflowError:
        return
    raise AssertionError("planned a job the workflow does not have")


def check_the_plan_carries_the_raw_block_and_not_only_the_joined_lines():
    """Execution runs the raw block.

    The joined list is for the inventory the static checks read. Running it instead loses
    the shape of a shell `if`, and the gate step is a shell `if`.
    """
    shape = ci.plan(ci.read_workflow(ci.workflow_text()), "deploy")
    gate = [s for s in shape["steps"] if s["name"] == "promotion gate"][0]
    assert gate["script"].splitlines()[0] == "set +e", gate["script"]
    assert "          " not in gate["script"], "the block kept its outer indentation"
    assert len(gate["commands"]) >= 5, gate["commands"]


def check_the_plan_keeps_the_two_actions_that_cannot_run_here():
    """`actions/checkout` and `actions/setup-python` need a runner.

    The plan has to carry them so the report can name them. Dropping them would make the
    output read as a full execution of the job, and the python version in particular is not
    modelled at all.
    """
    doc = ci.read_workflow(ci.workflow_text())
    absent = [s["uses"] for s in ci.plan(doc, "deploy")["steps"] if s["uses"]]
    assert absent == ["actions/checkout@v4", "actions/setup-python@v5"], absent


def check_the_checkout_stand_in_lands_exactly_the_committed_tree():
    """The empty workspace is the whole reason the first real execution failed.

    Copying the working directory instead would hide it, because my working directory holds
    a store and a cache from the last thing I ran. The oracle is `git ls-tree`, which is
    what `git archive HEAD` is defined to produce.
    """
    import shutil
    import subprocess
    import tempfile

    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    committed = set(listing)
    assert len(committed) > 30, "the oracle found {} files, which cannot be right".format(
        len(committed)
    )

    destination = os.path.join(tempfile.mkdtemp(prefix="ci-checkout-"), "ws")
    try:
        note = ci.checkout(destination)
        landed = set()
        for base, dirs, files in os.walk(destination):
            if ".git" in dirs:
                dirs.remove(".git")
            for name in files:
                landed.add(os.path.relpath(os.path.join(base, name), destination))
        assert landed == committed, sorted(committed ^ landed)
        assert note.startswith("cloned the committed tree at"), note

        # The archive version of this left no repository behind and the checks job holds a
        # check that shells out to git. A workspace with no `.git` is not a checkout.
        assert os.path.isdir(os.path.join(destination, ".git")), "no repository in the workspace"
    finally:
        shutil.rmtree(os.path.dirname(destination), ignore_errors=True)
