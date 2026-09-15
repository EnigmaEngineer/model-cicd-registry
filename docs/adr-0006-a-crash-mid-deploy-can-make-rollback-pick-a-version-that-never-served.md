# ADR 0006: a crash mid deploy can make rollback pick a version that never served

## Context

`mcr/registry.py` writes its transition log before it moves the alias. That ordering was
chosen deliberately and `adr-0003` gives the reason: the two writes are not in a
transaction, so one of them will eventually happen without the other, and a recorded move
that did not happen was judged better than a silent move that did. A recorded lie is
visible. A silent move is the thing a rollback cannot survive.

The first half of that is true. The second half turned out to be backwards, and it took a
drill to find out.

## What a crash actually leaves behind

Two orders, both measured on sqlite backed MLflow 3.16.0 through `scripts/drill.py`.

**The alias moves and the log write dies.** The store is on version 4 and the log's last
entry moved to 3.

```
crash after the alias moved         damaged: refused    control: 2
alias moved outside promote         damaged: refused    control: 2
```

`rollback_target` refuses. That is correct and it is not recoverable. The version that
served is not in the log at all, and `kind` is the field a rollback walk depends on, so
there is no way to tell afterwards whether the move was a promote or a rollback.

**The log write lands and the alias move dies.** This is the order the code chose, and it
is the one that produces a wrong deploy rather than a refusal.

Promote 1, then 2, then 3. Crash while writing the entry for version 4. Work out that
version 4 was the thing that broke the deploy, and ship version 5 instead.

```
crash then ship a different version   raw log: 4   settled: 3   control: 3
```

The raw log reads `[1, 2, 3, 4, 5]`. Version 4 is in it and version 4 never served a single
request. The check that exists to catch this compares the log's last entry against the
alias. It passes cleanly here. The retry wrote a correct last entry on top of the lie, so
the store looks healthy and a rollback deploys a model never seen in production.

Before the fix, `registry.rollback` on that store moved production to 4 and reported
success.

## Where the defect stops

A same version retry is harmless, and knowing that is what makes the above precise rather
than alarming.

```
crash then retry same version, answer   3 (wanted 3)
raw log agrees here, so no damage       3 (wanted 3)
and settle still dropped the phantom    5 raw, 4 kept, 1 dropped
```

Crashing on version 4 and then successfully shipping version 4 leaves a duplicate entry and
the right answer, because the version the phantom names is the version that went on to
serve. The crash alone is not the problem. The crash plus a retry that lands somewhere else
is, and that pair is the ordinary sequence, because the reason you ship something else is
usually that the first thing broke.

## Decision

Drop an entry whose move the store cannot confirm, and derive the confirmation from the log
rather than from anything new.

A promote that completed leaves the alias on its `to_version`, so the next entry written
reads that value as its own `from_version`. Every entry but the last therefore already has
a witness inside the log. The last entry's witness is the alias itself.

`registry.settle` applies that rule and returns the entries it kept alongside the ones it
dropped, each with a sentence saying why. `rollback_target` reads the settled log.
`registry.walk_back` is the walk itself, split out so the drill can run it over the raw log
and the settled one and compare the two, which is what makes the drill a measurement rather
than an assertion.

Nothing new is written at deploy time and there is no transaction. The information needed to
tell a completed move from an abandoned one was already in the log and nobody was reading it.

It does not recover the other crash order and it is not supposed to. That one still refuses.
The refusal now names the raw last entry alongside the confirmed one and the number of
entries dropped, because the entry an operator is looking for is the one that got dropped.

## The canary is a stage now, which is what gives a rollback verdict something to do

`adr-0005` ended with a gap. The canary's traffic share was a command line flag, so nothing
in the store knew a canary was running or at what fraction, and a rollback verdict had
nothing to undo. The exit code was the whole action.

That is also what made the wrong action look right. The first version of `scripts/canary.py`
grew a `--rollback` flag that called `registry.rollback`. It ran end to end and moved
production off a model the verdict said nothing about.

A deployment is now three facts the registry holds. `production` and `canary` are aliases,
and `canary.fraction` is a registered model tag. `mcr/deploy.py` has the three operations
over them.

- `open_canary` puts a version on trial at a share.
- `abort_canary` takes it off traffic. Production does not move. This is what a rollback
  verdict means.
- `land_canary` makes the canary production and ends the trial.

The distinction the deleted flag got wrong is now two functions with two names.
`registry.rollback` moves production to an earlier version and is a statement about
production. `deploy.abort_canary` ends a trial and is a statement about the canary.

```
aborting a canary leaves production   damaged: 1   control: 2
```

The damaged arm is the old flag's behaviour, which is `registry.rollback`. The control is
`abort_canary`. Production ends on 1 under the old flag and stays on 2 under the new one.

`scripts/canary.py` still takes no action. It reads a verdict and prints which command to
run. A script that reads a verdict and a script that changes what is deployed are two jobs,
and the exit code is what joins them.

The share is no longer duplicated either. `--fraction` now defaults to whatever the registry
recorded when the canary was opened, and a `--fraction` that disagrees with it is refused.
Measuring a split at a fraction nobody deployed is a measurement of an imaginary deployment.

## The two crash windows the deployment operations have

Both are the same shape as the registry's and both were drilled.

`open_canary` writes the share before the alias, so a crash between them leaves a share with
no canary. `land_canary` moves production before removing the canary alias, so a crash
between them leaves both aliases on one version.

```
canary open crashed, then repaired   consistent (wanted consistent)
and it was really broken first       broken (wanted broken)
canary land crashed, then repaired   consistent (wanted consistent)
and it was really broken first       broken (wanted broken)
that repair left production alone    3 (wanted 3)
```

`check_consistency` names each state in a sentence and `abort_canary` is the repair for
both. It clears a share the alias never caught up with, and it takes a canary alias off a
version that is already production without moving production. An operator reaching for
abort after a failed open is reaching for the right thing, which is why abort clears a stray
share rather than refusing on the grounds that there is no canary.

The second pair of rows in that block is the part worth keeping. A repair drill that only
asserts the end state passes whether or not anything was ever broken.

## CI, and a check on a workflow that has never run

What this needs is GitHub Actions taking a merge through the suite and the gate and into
a canary. There is no runner in the environment this was built in, so `.github/workflows/ci.yml` is a description of
intended wiring and it has never gone green anywhere. Saying otherwise would be the
fabrication this repo spends most of its checks preventing.

What can be checked without a runner is narrower and `tests/test_workflow.py` does it. Three
things rot in a workflow and all three are readable from the file. A step can name a script
that was renamed. A step can pass arguments the script refuses. A step can branch on an exit
code the script cannot return.

The first draft checked names and exit codes and passed. The deploy job's opening step then
turned out to call `scripts/train.py --store` against a script that takes `--track`, with a
bare `--register` against a flag that needs a value. Neither would have run.

So the check now hands the real argv to the real parser. Every `python3 scripts/X.py ...`
in the workflow is parsed by `X.build_parser()`. Four controls confirm it refuses a renamed
flag. They also cover a wrong arity and a swapped subcommand order and a parser that moved
inside `main`.

That last control is the one that matters. The first version of the argument check skipped
any script with no `build_parser`, and three of them built theirs inside `main`, including
`train.py`. So the check written to catch that exact step skipped that exact step, and a
control run against the known bad workflow came back green. A check that quietly covers
nothing is the failure this repo has now found in seven different places.

## What was rejected

**Reversing the write order.** It converts the wrong deploy into a refusal, which is safer,
and it loses `kind` permanently on any crash. The settled reading keeps the good ordering and
removes its cost, so there is nothing left to buy.

**A transaction over the two writes.** There is no API for one. The store is reachable only
through the MLflow client and a registered model tag write and an alias write are two calls.

**Writing a confirmation record after the alias moves.** A third write with a third crash
window, and it answers a question the existing two writes already answer.

**Repairing an unconfirmed entry rather than dropping it.** The missing fact is whether the
alias moved. Neither the log nor the store holds it, so a repair would be a guess wearing a
record's clothes.

**Keeping `--rollback` on the canary script and pointing it at `abort_canary`.** Closer to
right and still wrong. It puts an action on a script whose job is to read a measurement, and
the symmetry with the gate's `--promote` is exactly what made the first version look correct.

## What is not solved

**The drills simulate a crash, they do not cause one.** Each damaged arm reproduces half of
`promote` by hand. That is a faithful model of the state a crash leaves and it is not the
same as killing a process mid write.

**Nothing drains traffic or restarts anything.** There is no serving process in this project,
so the operational half of a rollback is absent rather than implemented badly. The obvious
way to test a rollback is under load, and load is not a thing that can be measured here.
What was drilled is the state machine, which is the half that exists.

**The workflow has still never executed.** Everything above is a check on the file.

**`_next_seq` survives a mutation.** Moving `max(used) + 1` to `+ 2` leaves gaps in the
sequence and changes no outcome, because the keys are zero padded to a fixed width and only
their order matters. It halves the headroom before that width overflows, from ten thousand
transitions to five thousand, which is not a number anything here rests on.
