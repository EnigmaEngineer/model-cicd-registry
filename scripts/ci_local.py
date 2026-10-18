"""Run a workflow job here, because a runner has run one of them once and the other two never.

`.github/workflows/ci.yml` describes a merge going through the suite, then the gate, then a
canary. It executed once, on a real runner, and the deploy job died in 63 seconds. Ten
static checks had passed on it for six days before that. Every one of them was correct. They
read script names and flag spellings and exit code constants, which is what an author gets
wrong from memory. None of them read the state each command leaves for the next one, and
that is what was broken.

So this runs the job instead of reading it. Each `run` block goes into its own shell the way
a runner does it, `$GITHUB_OUTPUT` is a real file that the next step's `if` is evaluated
against, and the workspace is a fresh copy of the committed tree rather than the directory I
happen to be sitting in. That last part is the whole reason the first execution failed. The
runner checks out into an empty workspace, so the store is new on every run and there is
never an incumbent for the canary to compare against.

What this is not. There is no runner here, so `actions/checkout` and `actions/setup-python`
cannot be performed. Checkout is modelled by the workspace copy. The python version is not
modelled at all and the report says which interpreter actually ran. A green result from this
script is evidence that the sequence of commands works. It is not a green badge and it does
not prove the file works on `ubuntu-latest`.

    python3 scripts/ci_local.py --job deploy
    python3 scripts/ci_local.py --job deploy --plan-only
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "ci.yml")

# What a runner puts in front of a `run` block on ubuntu. Worth being exact about, because
# the gate step relies on `set +e` turning off the `-e` this brings with it.
SHELL = ["bash", "--noprofile", "--norc", "-eo", "pipefail"]


class WorkflowError(Exception):
    """The file is not the shape this reader understands.

    Raised rather than returning something empty. A reader that quietly returns nothing
    makes every check above it pass having read nothing.
    """


def workflow_text(path=WORKFLOW):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _blank(line):
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


def _indent(line):
    return len(line) - len(line.lstrip())


def _scalar(text):
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_scalar(part) for part in inner.split(",")]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _dedent(raw):
    kept = [ln for ln in raw if ln.strip()]
    if not kept:
        return ""
    pad = min(_indent(ln) for ln in kept)
    return "\n".join(ln[pad:].rstrip() for ln in raw).strip("\n")


def _parse_map(lines, i, indent):
    out = {}
    while i < len(lines):
        if _blank(lines[i]):
            i += 1
            continue
        here = _indent(lines[i])
        if here < indent:
            break
        stripped = lines[i].strip()
        if stripped.startswith("- ") or here > indent:
            break
        if ":" not in stripped:
            raise WorkflowError("line {} is not a mapping entry: {!r}".format(i + 1, stripped))
        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()

        if rest == ">":
            raise WorkflowError("folded scalars are not handled, line {}".format(i + 1))
        if rest == "|":
            body = []
            i += 1
            # `_blank` counts a comment as blank, which is right for skipping and wrong
            # here. A `#` line at an outer indent is a note about the next step and ending
            # the block on it is the fix. One indented deeper is really block content.
            while i < len(lines) and (not lines[i].strip() or _indent(lines[i]) > indent):
                body.append(lines[i])
                i += 1
            while body and not body[-1].strip():
                body.pop()
            out[key] = _dedent(body)
            continue
        if rest:
            out[key] = _scalar(rest)
            i += 1
            continue

        nxt = i + 1
        while nxt < len(lines) and _blank(lines[nxt]):
            nxt += 1
        if nxt >= len(lines) or _indent(lines[nxt]) <= indent:
            raise WorkflowError("key {!r} on line {} has no value".format(key, i + 1))
        child = _indent(lines[nxt])
        if lines[nxt].strip().startswith("- "):
            out[key], i = _parse_seq(lines, nxt, child)
        else:
            out[key], i = _parse_map(lines, nxt, child)
    return out, i


def _parse_seq(lines, i, indent):
    out = []
    while i < len(lines):
        if _blank(lines[i]):
            i += 1
            continue
        here = _indent(lines[i])
        if here < indent:
            break
        if here != indent or not lines[i].strip().startswith("- "):
            break

        # Re-indent the dash line so the item reads as a plain mapping, then hand the whole
        # slice to the mapping parser. Keeps one parser for `uses: x` and for `run: |`.
        chunk = [" " * (indent + 2) + lines[i][indent + 2:]]
        j = i + 1
        while j < len(lines):
            if _blank(lines[j]):
                chunk.append(lines[j])
                j += 1
                continue
            if _indent(lines[j]) <= indent:
                break
            chunk.append(lines[j])
            j += 1
        item, _ = _parse_map(chunk, 0, indent + 2)
        if not item:
            raise WorkflowError("empty sequence item at line {}".format(i + 1))
        out.append(item)
        i = j
    return out, i


def read_workflow(text):
    """The whole file as nested dicts and lists."""
    lines = text.splitlines()
    first = 0
    while first < len(lines) and _blank(lines[first]):
        first += 1
    doc, _ = _parse_map(lines, first, 0)
    if "jobs" not in doc or not isinstance(doc["jobs"], dict):
        raise WorkflowError("no jobs mapping in the workflow")
    for name, job in doc["jobs"].items():
        if not isinstance(job, dict) or "steps" not in job:
            raise WorkflowError("job {!r} has no steps".format(name))
        if not isinstance(job["steps"], list):
            raise WorkflowError("job {!r} steps is not a list".format(name))
    return doc


def join_continuations(lines):
    """Fold `\\` continuations into one command.

    A wrapped `python3` call arrives here as two lines and the tail of it is not a command.
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


def run_commands(text):
    """Every shell command under a `run` key, across every job.

    Derived from the reader above rather than scraped separately. Two readers of one file
    drift, and the one that gets fixed is never both.
    """
    out = []
    for job in read_workflow(text)["jobs"].values():
        for step in job["steps"]:
            if "run" not in step:
                continue
            body = [ln.strip() for ln in str(step["run"]).splitlines() if ln.strip()]
            out.extend(join_continuations(body))
    return out


EXPR = re.compile(r"\$\{\{(.*?)\}\}")


def resolve(value, context):
    """Substitute every `${{ ... }}` in a string.

    Refuses an expression it does not know rather than leaving it in place. A literal
    `${{ secrets.TOKEN }}` reaching a shell is a command running against the wrong thing.
    """
    def one(match):
        ref = match.group(1).strip()
        if ref in context:
            return str(context[ref])
        raise WorkflowError("nothing here can resolve {!r}".format(ref))

    return EXPR.sub(one, value)


CONDITION = re.compile(r"^(?P<left>[\w.\-]+)\s*(?P<op>==|!=)\s*'(?P<right>[^']*)'$")


def evaluate_if(expression, context):
    """Answer a step `if`, or refuse.

    Two shapes appear in this file. A comparison against a literal, and `always()`. Anything
    else raises. Defaulting an unreadable condition to true runs a step that should not run,
    and defaulting it to false hides a step that should, so neither default is safe.
    """
    expression = expression.strip()
    if expression == "always()":
        return True
    match = CONDITION.match(expression)
    if not match:
        raise WorkflowError("cannot evaluate the condition {!r}".format(expression))
    left = match.group("left")
    if left not in context:
        # An unset step output is the empty string on a runner, which is what a skipped
        # earlier step leaves behind.
        actual = ""
    else:
        actual = str(context[left])
    if match.group("op") == "==":
        return actual == match.group("right")
    return actual != match.group("right")


def checkout(destination, from_worktree=False):
    """Stand in for `actions/checkout`.

    The committed tree by default. A runner checks out a commit and not my working
    directory, and that difference is the point of running this at all.

    This clones rather than unpacking an archive, which was the first version. An archive
    gives the right files and no `.git`, and the checks job turned out to hold a check that
    shells out to git. It died with a CalledProcessError in the workspace and passed on the
    mount, so the archive was modelling checkout badly. The real action leaves a repository
    behind and anything in the suite is entitled to assume one.
    """
    os.makedirs(destination, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", "--local", ROOT, destination],
        check=True, capture_output=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=destination,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not from_worktree:
        return "cloned the committed tree at {}".format(head)

    for name in os.listdir(ROOT):
        if name == ".git":
            continue
        src = os.path.join(ROOT, name)
        dst = os.path.join(destination, name)
        if os.path.isdir(dst):
            shutil.rmtree(dst, ignore_errors=True)
        elif os.path.exists(dst):
            os.remove(dst)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)
    return "cloned at {} and copied the working tree over it".format(head)


def plan(doc, job_name, event="push"):
    """What the job would do, without doing any of it.

    Separate from execution because the execution trains models and takes minutes, and the
    resolution and the conditions are worth grading on every suite run.
    """
    if job_name not in doc["jobs"]:
        raise WorkflowError("no job named {!r}, there is {}".format(
            job_name, sorted(doc["jobs"])
        ))
    job = doc["jobs"][job_name]
    context = {"github.workspace": "<workspace>", "github.event_name": event}
    env = {}
    for key, raw in (doc.get("env") or {}).items():
        env[key] = resolve(str(raw), context)
        context["env." + key] = env[key]

    gate = job.get("if")
    planned = []
    for index, step in enumerate(job["steps"]):
        entry = {
            "index": index,
            "name": step.get("name") or step.get("uses") or "step {}".format(index),
            "id": step.get("id"),
            "if": step.get("if"),
            "uses": step.get("uses"),
            "commands": [],
            "script": None,
        }
        if "run" in step:
            entry["script"] = str(step["run"])
            entry["commands"] = join_continuations(
                [ln.strip() for ln in str(step["run"]).splitlines() if ln.strip()]
            )
        planned.append(entry)
    return {"job": job_name, "if": gate, "env": env, "steps": planned}


def execute(doc, job_name, workspace, event="push", skip=(), from_worktree=False,
            pythonpath=None, stream=sys.stdout):
    job = doc["jobs"][job_name]
    note = checkout(workspace, from_worktree=from_worktree)
    print("workspace  {}".format(workspace), file=stream)
    print("checkout   {}".format(note), file=stream)
    print("python     {}".format(sys.version.split()[0]), file=stream)
    wanted = job.get("runs-on")
    print("runs-on    {} is declared and is not what this is".format(wanted), file=stream)
    print("", file=stream)

    context = {"github.workspace": workspace, "github.event_name": event}
    base = dict(os.environ)
    for key, raw in (doc.get("env") or {}).items():
        value = resolve(str(raw), context)
        context["env." + key] = value
        base[key] = value
    if pythonpath:
        base["PYTHONPATH"] = pythonpath

    results = []
    outputs_dir = os.path.join(workspace, "_outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    failed = False
    for step in plan(doc, job_name, event=event)["steps"]:
        label = step["name"]
        condition = None if step["if"] is None else str(step["if"]).strip()

        # Once a step has failed, a runner skips everything that does not say `always()`,
        # because the implicit condition on a step is `success()`.
        if failed and condition != "always()":
            print("skip   {:<28} an earlier step failed".format(label), file=stream)
            results.append({"step": label, "state": "skipped-after-failure", "code": None})
            continue

        if step["if"] is not None and not evaluate_if(str(step["if"]), context):
            print("skip   {:<28} if {} is false".format(label, step["if"]), file=stream)
            results.append({"step": label, "state": "skipped", "code": None})
            continue
        if step["uses"]:
            why = "no runner here"
            if "checkout" in step["uses"]:
                why = "the workspace copy stands in for it"
            print("absent {:<28} {} ({})".format(label, step["uses"], why), file=stream)
            results.append({"step": label, "state": "not-executable", "code": None})
            continue
        if label in skip:
            print("skip   {:<28} asked to skip it".format(label), file=stream)
            results.append({"step": label, "state": "skipped-by-request", "code": None})
            continue

        script = resolve(step["script"], context)
        out_file = os.path.join(outputs_dir, "{}.txt".format(step["index"]))
        open(out_file, "w").close()
        env = dict(base)
        env["GITHUB_OUTPUT"] = out_file

        proc = subprocess.run(
            SHELL + ["-c", script], cwd=workspace, env=env,
            capture_output=True, text=True,
        )
        print("run    {:<28} exit {}".format(label, proc.returncode), file=stream)
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        for line in tail[-6:]:
            print("       | {}".format(line), file=stream)

        with open(out_file, "r", encoding="utf-8") as fh:
            for line in fh:
                if "=" not in line:
                    continue
                key, _, value = line.strip().partition("=")
                if step["id"]:
                    context["steps.{}.outputs.{}".format(step["id"], key)] = value
        results.append({
            "step": label, "state": "ran", "code": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr,
        })
        if proc.returncode != 0 and not failed:
            failed = True
            print("       the job has failed, only always() steps run from here",
                  file=stream)

    return results


def build_parser():
    p = argparse.ArgumentParser(description="run a workflow job locally")
    p.add_argument("--job", default="deploy")
    p.add_argument("--event", default="push")
    p.add_argument("--workspace", default=None, help="defaults to a fresh temporary dir")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--from-worktree", action="store_true",
                   help="copy the working tree instead of the committed one")
    p.add_argument("--skip", action="append", default=[],
                   help="a step name to record as skipped, repeatable")
    p.add_argument("--pythonpath", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    doc = read_workflow(workflow_text())

    if args.plan_only:
        shape = plan(doc, args.job, event=args.event)
        print("job {}".format(shape["job"]))
        for key, value in sorted(shape["env"].items()):
            print("env {} = {}".format(key, value))
        for step in shape["steps"]:
            mark = "uses" if step["uses"] else "run "
            print("{} {:<30} if={}".format(mark, step["name"], step["if"]))
            for command in step["commands"]:
                print("       {}".format(command))
        return 0

    workspace = args.workspace or tempfile.mkdtemp(prefix="ciws-")
    results = execute(
        doc, args.job, workspace, event=args.event, skip=set(args.skip),
        from_worktree=args.from_worktree, pythonpath=args.pythonpath,
    )
    ran = [r for r in results if r["state"] == "ran"]
    bad = [r for r in ran if r["code"] != 0]
    print("")
    print("{} steps ran, {} failed, {} of {} were not executable or skipped".format(
        len(ran), len(bad), len(results) - len(ran), len(results),
    ))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
