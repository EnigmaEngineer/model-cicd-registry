# ADR 0011. The workflow gets executed, not only read

Status: accepted. 2026-10-18.

## The problem

`.github/workflows/ci.yml` had ten static checks on it and they all passed. They read the
script names, handed every command to its script's own argument parser, and asserted that
each branch tested an exit code the script can actually return. All correct. The file still
described something a runner could not produce, and the one real execution of it died in 63
seconds on the deploy job.

The reason is structural. A check on a file it cannot execute reads names and constants,
because names and constants are what an author gets wrong from memory. None of them reads
the state each command leaves for the next one. `adr-0006` records that in general terms.
What it did not do is fix it.

## The decision

`scripts/ci_local.py` executes a job here. Each `run` block goes into its own
`bash --noprofile --norc -eo pipefail`, which is what a runner puts in front of one.
`$GITHUB_OUTPUT` is a real file per step and the next step's `if` is evaluated against what
the previous one wrote into it. The workspace is a clone rather than the directory I am
sitting in.

The workspace is the part that matters. The deploy job failed on a real runner because the
runner checks out into an empty workspace, so the store is new on every run and there is
never an incumbent for the canary to compare against. Running the job in my own directory
would have hidden that, because my directory holds a store from the last thing I ran.

The reader is shared. `tests/test_workflow.py` used to scrape `run:` lines with a second
pass of its own. Two readers of one file drift and the one that gets fixed is never both,
so the structured reader the executor needs is what the static checks read now.

## What it found

Three things, and the first is a defect in the reader rather than in the workflow.

**A comment between two steps was being read as part of the step above it.** The block
scalar collector treated any line at a deeper indent as content, and it counted a `#` line
as blank for the purpose of continuing. So the `install` step came back carrying the eight
comment lines that explain the step after it. Those would have gone to a shell. A comment
indented into a block really is content, so the fix had to keep that case and it does.

**`read the canary` ran on a rejected candidate.** The step carried no condition. Measured:
the gate exits 1, `open the canary` is correctly skipped, and the canary read then runs
against a store where nothing is aliased `canary`. `canary.py` refuses with exit 2. The
block ends in `exit 0`, so the step reported green. A build could reject a candidate and
still claim it had read a canary.

**The two capture steps disagreed about what a refusal is.** The gate step branched on its
own `EXIT_REFUSE` and failed the build, on the stated grounds that a refusal is not a
verdict. The canary step wrote the same code into its output and never looked at it, so
exit 2 arrived at the two steps reading that output as a code neither of them matches. Both
were skipped and the job went green having decided nothing.

Both are fixed. `read the canary` is gated on the same verdict that opens the canary, and it
fails the build on a refusal. Two checks encode it, and both were run against the unfixed
file first and both failed there.

## What this does not prove

A green result here is not a green badge. `actions/checkout` is modelled by the clone.
`actions/setup-python` is not modelled at all, so the declared `3.11` is not what ran. The
report prints the interpreter it used, which was 3.10.12, and prints the declared
`runs-on` next to a line saying that is not what this is.

The install step was skipped by name on the run recorded here, because this machine has no
writable site-packages. The report names a skipped step and its reason rather than passing
over it.

One execution exercises one path. The promote path ran in seven steps. The reject path was
driven by hand to find the defect above. `take it off traffic` has still never executed,
because reaching it needs the gate to promote and the canary to roll back, and no choice of
config drives the job into that disagreement. That one is still only read, not run.

## What was rejected

**Putting the execution in the suite.** The deploy job trains two models. That is minutes,
and `tests/run_all.py` is the thing that has to stay fast enough to run on every edit. The
reading and the substitution and the conditions are graded in the suite, because those
decide which commands run at all. The execution is a command somebody runs on purpose.

**Keeping the old line scraper alongside the new reader.** It would have meant the static
checks and the execution disagreeing about what the file says, which is the defect the
shared reader exists to prevent.

**Writing the precondition for each job by hand.** `adr-0006` proposed it and it does not
generalise. The canary check in `tests/test_workflow.py` is one job's preconditions written
out by hand, and it took a real execution to find the one it was missing.
