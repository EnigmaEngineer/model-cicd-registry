# model-cicd-registry

A model training pipeline where the same config always produces the same bytes, and where
the thing that identifies a model is kept separate from the thing that decides whether it
is any good. The end state is a merge that takes a model all the way to a canary deploy.
Rollback is one command.

What is here is the config system, the seed control and the training pipeline. On top of
those sit the MLflow tracking layer and the registry, which carries stage transitions and a
one command rollback. In front of the registry is the promotion gate, which decides whether
a candidate is allowed to move production. Behind it is the canary, which routes a slice of
traffic, and the deployment state that says which version is on trial and at what share.

The rollback path is drilled rather than assumed. `scripts/drill.py` puts the store into
states a crash really produces and checks what the recovery does, and one of those drills is
why the rollback walk no longer trusts the raw transition log.

## Run it

```
pip install -r requirements.txt
python3 scripts/train.py configs/baseline.yml --out artifacts/baseline.json
```

```
name         baseline
config       70a46feb1141
artifact     8b83b8d77ac9e226e996859a28e5725e80d6ba657e6a5eda7b513f46dda003b3
seed         20260911

holdout_accuracy         0.816000
holdout_log_loss         0.449239
holdout_n_rows           5000.000000
holdout_positive_rate    0.185200
holdout_roc_auc          0.675518

wrote artifacts/baseline.json
```

Then check the claim the rest of it rests on.

```
python3 scripts/repro_probe.py configs/baseline.yml
python3 tests/run_all.py
```

Tracking is a separate install, because training a model does not need it.

```
pip install -r requirements-tracking.txt
python3 scripts/train.py configs/baseline.yml --track sqlite:///mlflow.db
python3 scripts/track_probe.py
python3 tests/run_with_mlflow.py
```

The registry sits on the same store.

```
python3 scripts/train.py configs/baseline.yml --track sqlite:///mlflow.db --register mcr-fraud
python3 scripts/registry.py --store sqlite:///mlflow.db list
python3 scripts/registry.py --store sqlite:///mlflow.db promote 1 production
python3 scripts/registry.py --store sqlite:///mlflow.db promote 1 production --require-gate
python3 scripts/registry.py --store sqlite:///mlflow.db rollback production
python3 scripts/registry_probe.py
```

`promote` moves an alias and asks nothing about how the version got there. `--require-gate`
refuses unless the gate cleared that version, for that stage, **against whoever holds the
stage right now**. `deploy.py open` takes the same flag, because the canary is the other way
in. Both are opt in and the section below says what that does and does not buy.

And the gate sits in front of it.

```
python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2
python3 scripts/registry.py --store sqlite:///mlflow.db report 2
python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2 --promote
python3 scripts/gate_probe.py
```

Every comparison writes its verdict onto the candidate's run, whether or not anybody asked
to promote it. `report` reads it back. `--no-report` is the way to compare without writing.

And the canary sits behind it. Opening one is a change to the registry, not a flag on a
command, so the share survives the process that set it.

```
python3 scripts/deploy.py open   --store sqlite:///mlflow.db --canary 4 --fraction 0.05
python3 scripts/deploy.py open   --store sqlite:///mlflow.db --canary 4 --fraction 0.05 --require-gate
python3 scripts/deploy.py status --store sqlite:///mlflow.db
python3 scripts/canary.py --store sqlite:///mlflow.db --canary canary
python3 scripts/deploy.py abort  --store sqlite:///mlflow.db
python3 scripts/canary_probe.py
```

`canary.py` reads the share off the registry. Passing a `--fraction` that disagrees with
what was deployed is refused rather than believed.

The failure drills are the evidence for the rollback path and they run in seconds.

```
python3 scripts/drill.py
```

```
ok     crash then retry same version, answer  3 (wanted 3)
ok     raw log agrees here, so no damage      3 (wanted 3)
ok     and settle still dropped the phantom   5 raw, 4 kept, 1 dropped
ok     crash then ship a different version    raw log: 4  settled: 3  control: 3
ok     crash after the alias moved            damaged: refused    control: 2
ok     alias moved outside promote            damaged: refused    control: 2
ok     two rollbacks reach the oldest         damaged: 1          control: 3

ok     canary open crashed, then repaired     consistent (wanted consistent)
ok     and it was really broken first         broken (wanted broken)
ok     canary land crashed, then repaired     consistent (wanted consistent)
ok     and it was really broken first         broken (wanted broken)
ok     that repair left production alone      3 (wanted 3)
ok     aborting a canary leaves production    damaged: 1          control: 2

13 drills, 0 bad
```

`gate_probe.py` and `canary_probe.py` need no MLflow at all.

## What is here

```
mcr/config.py     frozen dataclass, YAML on disk, refuses unknown keys, content fingerprint
mcr/seed.py       named random streams derived from one run seed
mcr/data.py       synthetic corpus, deterministic given the seed
mcr/model.py      logistic regression by gradient descent, numpy only
mcr/artifact.py   deterministic serialisation and a content hash
mcr/train.py      the pipeline. config in, artefact out, nothing else
mcr/tracking.py   MLflow. write a run, read it back, rebuild the config from what came back
mcr/registry.py   which model is in production, which one was there before, and rollback
mcr/gate.py       whether a candidate is allowed to replace the incumbent
mcr/canary.py     routing a slice of traffic, and the two comparisons that gives you
mcr/deploy.py     what is deployed. production, the canary on trial, and the share it takes
```

Eleven entry points in `scripts/`. Three runners in `tests/`, carrying 215 checks without
MLflow and 319 with it. The third runner covers the registry and deployment modules only,
at 15.4 seconds against 28.6 for the full store suite, which is what a mutation pass over
those two modules needs to fit inside one shell invocation.

## Tracking, and the question it is built to answer

The point of the tracking layer is not that a number reaches a dashboard. It is that the
store holds enough to rebuild the run. The promotion gate will never hold the config
that trained the incumbent. It will hold a row in a database.

So `tracking.params_for` and `tracking.config_from_params` are inverses, and
`scripts/track_probe.py` retrains from what the store gives back and compares the bytes.

```
config rebuilds from stored params       OK   fingerprint 70a46feb1141
retrain from the store matches bytes     OK   8b83b8d77ac9 vs tagged 8b83b8d77ac9
metrics round trip exactly               OK   10 metrics, 0 inexact
every param crosses as a string          OK   value types ['str']
control: a missing param is refused      OK   model.epochs
control: a changed param changes the fit OK   epochs 400 -> 401
control: one recipe, two run ids         OK   2 runs tagged 70a46feb1141
control: junk param value is refused     OK   data.noise is not a number: 'nine'
```

The recipe goes into params and the mutable state goes into tags, and that split was
measured rather than copied from a tutorial. On mlflow 3.16.0, `log_param` on a key holding
a different value raises and `set_tag` overwrites silently. So params are immutable for
free, which is what a recipe should be. The stage does **not** live in a tag, and
`docs/adr-0003` reverses that half of the argument on a later measurement.

Two more measurements from the same pass. A param value over 6,000 characters and a tag
value over 8,000 come back **truncated**, with a warning on stderr and no exception, so a
content hash long enough to be shortened would become a registry key pointing at nothing.
And the file store is gone in this version: mlflow 3.16.0 raises on a `file:` tracking URI
and tells you to use a database, so every command above uses sqlite.

`docs/adr-0002` has the rest, including the defect this layer uncovered in the config
fingerprint.

## The registry, and the two things MLflow will not do

MLflow has a registry with stages in it. This project does not use them, and the reason is
measured rather than stylistic. Everything below is `python3 scripts/registry_probe.py` on
mlflow 3.16.0 today.

```
stage transition, nothing passed it does not require   versions in Production: [1, 2]
the same call with archive_existing_versions=True      versions in Production: [1]
two versions both tagged stage=production              allowed, search returns [1, 2]
alias moved from v1 to v2                              v1 aliases [], v2 aliases ['production']
v1 after losing the alias                              last_updated_timestamp == creation_timestamp
```

`transition_model_version_stage` is deprecated with a removal notice, and its default leaves
two versions in Production at once. A deploy reading "the production model" out of that
store gets whichever row comes back first, and `search_model_versions` does not sort. It
returned `[3, 1]` and `[2, 1]` in one session.

A version tag has the same hole, which is why the stage is not in one.

An alias does not. Moving `production` to a new version takes it off the old one in the same
call, so exclusivity is the store's behaviour rather than a rule this code has to remember.

**The cost is that an alias move leaves no trace at all.** The row of the version that lost
it is not even touched. So the registry can say what is in production and it cannot say what
was, and a rollback is a question about what was. The transition log is written here, one
tag per transition on the registered model.

The last measurement is the one worth arguing about. A rollback target derived from the
store alone has to guess, and the obvious guess is the highest version that is not current.
That is wrong as soon as a candidate is registered and rejected without ever being promoted,
which is the normal case once the gate exists.

```
log says 1, store-only guess says 4
```

## The reference that names two models

MLflow refuses an alias containing a slash. It accepts an alias that is a decimal number.
So this is allowed while version 2 exists:

```python
cli.set_registered_model_alias(name, "2", "4")
```

and the string `"2"` now names two different models, trained from two different runs.

```
get_model_version(name, "2")             run 6b8c4048
get_model_version_by_alias(name, "2")    run 6fd44793
```

`registry.resolve` refuses rather than picking one. Which model you get otherwise depends on
which function the caller reached for, and the caller has no way to know that is a question.

## Rollback is not "whatever it held before"

That was the first rule written and a check caught it the same hour.

Promote version 1. Then 2. Then 3. Now roll back. Production sits on 2 and the version it
held immediately before 2 really was 3. So a second rollback goes back to 3, which is the
model that was just rolled away from. Rollback would oscillate between the last two versions,
never reach version 1, and report success every time.

A version a rollback moved away from is abandoned now, and the walk skips it. The direction
of each move is stored on the log entry rather than worked out from the version numbers,
because a deliberate redeploy of an older model has exactly the same shape in the log.

## A crash mid deploy made rollback pick a version that never served

The transition log is written before the alias moves. The two are not in a transaction, so
one of them will eventually happen without the other, and this order was chosen on the
argument that a recorded move which did not happen is visible while a silent move is not.

The visible one turned out to be the dangerous one.

Promote 1, then 2, then 3. Crash while writing the entry for version 4. Work out that
version 4 is what broke the deploy and ship version 5 instead.

```
raw log          [1, 2, 3, 4, 5]
alias            5
rollback_target  4
```

Version 4 never served a request. It exists only as a log entry a crash left behind. The
check meant to catch this compares the log's last entry against the alias, and it passes,
because the retry wrote a correct last entry on top of the lie. So the store looks healthy
and one command deploys a model that has never been in production.

The fix needed no new writes, because the log already knew. A promote that completed leaves
the alias on its `to_version`, so the next entry written reads that value as its own
`from_version`. Every entry but the last has a witness inside the log, and the last one's
witness is the alias. `registry.settle` drops any entry without one and says why.

```
crash then ship a different version   raw log: 4   settled: 3   control: 3
```

The control is the same sequence with no crash in it. Settled and control agree, and the
raw reading disagrees with both, which is what makes that row a measurement rather than a
green tick.

The other crash order still refuses and that is the right answer. If the alias moved and
nothing was written, the version that served is not in the log at all and the direction of
the move is unrecoverable.

This came out of `scripts/drill.py`, which is thirteen drills over states a crash really
produces. Each one carries a control that has to come back healthy, because a drill whose
control also reports the failure has proved nothing about the damage.

`docs/adr-0006` is the full argument, including the orders that stay unrecoverable and why
the witness rule is a recovery mechanism rather than a defence against a forged entry.

## Opening a canary is a change to the registry

The canary's share used to be a command line flag. Nothing in the store knew a canary was
running or at what fraction, so a rollback verdict had nothing to undo and the exit code
was the whole action.

That is also what made the wrong action look right. An earlier version of the canary script
had a `--rollback` flag which called `registry.rollback`. It ran end to end and moved
production off a model the verdict said nothing about.

A deployment is three facts the registry holds. Two aliases and a tag.

```
production        an alias, the version serving the control share
canary            an alias, the version on trial, or nothing
canary.fraction   a tag, the share the canary takes
```

`open_canary` puts a version on trial. `abort_canary` takes it off and leaves production
where it is, which is what a rollback verdict means. `land_canary` makes the canary
production. The distinction the deleted flag got wrong is now two functions with two names.

```
aborting a canary leaves production   damaged: 1   control: 2
```

The damaged arm is what the old flag did. The control is `abort_canary`.

Both operations have a crash window of their own and both are drilled. `open_canary` writes
the share before the alias, so a crash leaves a share with no canary. `land_canary` moves
production before removing the canary alias, so a crash leaves both aliases on one version.
`check_consistency` names each state in a sentence and `abort_canary` is the repair for
both, without moving production in either case.

## CI, and what three static checks could not see

`.github/workflows/ci.yml` takes a merge through the suite and the drills and the probes.
What survives that reaches the gate and then the canary.

There was no GitHub runner in the environment this was built in, so for six days the file
was checked rather than executed. A step can name a script that was renamed, pass arguments
the script refuses, or branch on an exit code the script cannot return. `tests/test_workflow.py`
reads the file and checks all three on every commit.

The argument check exists because the first two were not enough. They passed on a workflow
whose opening deploy step called `scripts/train.py --store` against a script that takes
`--track`, with a bare `--register` against a flag that needs a value. Neither would have
run. So every `python3 scripts/X.py ...` in the workflow is now handed to `X.build_parser()`
and really parsed.

**Then it ran, and the deploy job failed.** All three checks still passed on the file that
failed, and they were right to. Every script existed, every command parsed, every branch was
reachable. What none of them read is the state each command leaves for the next one.

The runner starts with an empty workspace, so the store is a new database on every run and
there is never an incumbent. The job trained one model into that empty store, let the gate
promote it, then tried to canary it. `deploy.py open` refused, because the version it was
asked to canary was the version now in production, which compares a model against itself.
Removing the promote does not help either. With nothing in production the canary has no
control arm and the same command refuses for the opposite reason.

Both refusals are correct. The code was right and the pipeline was impossible.

```
before   train -> gate --promote -> open canary    refused, model against itself
after    seed an incumbent -> train -> gate -> open canary -> read -> land
```

The incumbent is now seeded on purpose and labelled as seeded, an underfit model put into
production so the candidate has something real to beat, and the gate no longer passes
`--promote` because `deploy.py land` is what moves production after the canary reads.

The check added for it asserts the two conditions an `open` needs, and one of them is an
absence. Its control is the exact file that failed on the runner.

The honest summary is that a check on an artefact it cannot execute reads names and
constants, because those are what an author gets wrong from memory. A sequence of
individually correct commands that cannot happen in order is a different kind of wrong, and
it took a real runner to find it.

## What the rollback changed, and what it did not

Two models are registered. Rolling production from version 2 back to version 1 swaps one
artefact for a different one, and the two are indistinguishable.

```
artefact hashes                         8b83b8d77ac9 and 827ec647b844, different
max |p_a - p_b| over 5,000 holdout rows                       2.014e-09
holdout rows whose 0.5 decision differs                       0 of 5000
holdout_roc_auc gap                                           0.000e+00
holdout_log_loss gap                                          1.806e-11
```

The rollback is real. The registry moved, the alias moved, the log recorded it. Nothing a
user could observe changed, because `baseline` and `candidate-lr` both converge on this
corpus and adr-0001 already said so.

That is a fact about the corpus rather than about the registry, and it is the reason
`configs/candidate-underfit.yml` exists. One epoch instead of four hundred.

```
                      baseline    candidate-lr   candidate-underfit
holdout_roc_auc       0.675518    0.675518       0.675312
holdout_log_loss      0.449239    0.449239       0.641871
holdout_accuracy      0.816000    0.816000       0.814800
```

**A gate reading AUC would pass it.** Across seven configs measured today, from one epoch
to four hundred and from a learning rate of 0.0001 up to 1.5, holdout AUC spans 0.675312 to
0.675518. That is a range of 2.06e-04 on a rank based metric, because the direction of the
weight vector settles almost immediately and AUC only reads the ranking. Log loss over the
same seven spans 0.449239 to 0.688811, and that is the metric with something to say.

## The gate, and why it does not read the numbers in the store

Both runs are in the tracking store and both recorded a holdout log loss, so the obvious
gate is one line comparing them. That line is wrong twice and the second one is the
interesting one.

Every run generates its own corpus from its own config, so its holdout is a slice of that
corpus. Two runs share a holdout only when their data section and their seed both match.
Nothing was checking that, and the seed is inside the config fingerprint, so two configs
differing only in it are two perfectly legitimate candidates.

Hold the model config completely still and move only the seed.

```
12 seeds, each on its own holdout: 0.341545 to 0.449239, span 0.107693
the gap the gate exists to catch:  0.192633
so the corpus draw is 55.9% of that gap
```

So the gate builds one holdout, rebuilds both models out of their artefact bytes, and
scores them on the same rows. The numbers the runs recorded still appear in the report, in
a column saying that is what they are. When the two columns disagree, the disagreement is
the point.

```
                           shared holdout  its own holdout      roc auc
baseline                         0.449239         0.449239     0.675518
candidate-lr                     0.449239         0.449239     0.675518
candidate-underfit               0.641871         0.641871     0.675312
candidate-inverted              19.701270        19.701270     0.377109
candidate-diverged                    nan              nan     0.515824
baseline-other-corpus            0.811947         0.341545     0.351162
```

The last row is the whole argument. It recorded a better number than the incumbent and it
is nearly twice as bad on the rows they are both judged on. A gate reading the store
promotes it.

Some of that gap is this generator drawing a fresh set of true weights per seed, so a model
from another corpus has no reason to transfer at all. The 0.107693 span above is the honest
figure and it is measured with nothing transferring anywhere.

## Rejecting a model

Three models registered, `baseline` in production, and `candidate-underfit` put up against
it.

```
$ python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2
candidate    version 2 of mcr-fraud
incumbent    version 1

holdout      56ab7ecf5a6b
metric       log_loss, lower is better

                                  on this holdout    roc auc   as the run ran
incumbent baseline                       0.449239   0.675518         0.449239
candidate candidate-underfit             0.641871   0.675312         0.641871

paired diff  +1.926328e-01  interval [+1.780119e-01, +2.072537e-01]
verdict      reject  (worse)
             candidate is worse by 0.192633 on log_loss, interval [0.178012, 0.207254]
exit=1
```

Exit codes carry the verdict. 0 promote, 1 reject, 2 refuse. The last two are different on
purpose and the next section is why.

### The rejection has to land somewhere

An exit code lives as long as the shell that read it. The comparison above is the thing
somebody asks about six weeks later, usually in the form "why is this model not in
production yet", so it is written back to the run that produced the candidate.

```
$ python3 scripts/registry.py --store sqlite:///mlflow.db report 2
version 2 has never been gated

$ python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2
... the comparison above, exit 1 ...

$ python3 scripts/registry.py --store sqlite:///mlflow.db report 2
gate.candidate_score = 0.64187146762992
gate.holdout = 56ab7ecf5a6b
gate.incumbent_score = 0.44923865033232196
gate.interval = 0.17801190162342817,0.20725373297176775
gate.mean_diff = 0.19263281729759796
gate.metric = log_loss
gate.reason = worse
gate.stage = production
gate.verdict = reject
```

Nine tags, and the command that wrote them passed no flags. `report` is the read side and
it says "never been gated" rather than printing nothing, because an empty block and a
missing feature look identical. That is a recent change and the thing it replaced is the
more interesting half.

Until today the write sat inside the `--promote` branch, under a comment reading "a
rejection that leaves no trace on the run is a rejection nobody can audit later". The
comment was right and it was describing the wrong branch. `--promote` means move the stage
on a win, so it is the flag you pass when you expect to succeed. Every rejection that came
from somebody merely checking left nothing behind. Measured on a clean store: nine tags
with the flag and zero without.

The effect is a store whose audit trail is biased toward wins, which is the opposite of
what an audit trail is for. Full scores are boring. The rejections are the record.

So the two concerns are separate now. Writing the report is what a comparison does.
`--promote` moves the stage and nothing else. `--no-report` compares without touching the
store, for a pipeline that must not write.

Tags rather than params, because a run is written before it is ever gated and a param
refuses a rewrite. A candidate can be gated more than once against different incumbents,
and the latest verdict is the one worth reading off the run. The cost is that the earlier
ones are gone, which is in the limitations.

### Nothing stopped you promoting a model the gate had just refused

A reader asked whether promoting straight past the gate was intentional. It was not
prevented and it was not documented. Worth showing rather than describing.

```
$ python3 scripts/registry.py --store sqlite:///mlflow.db report 2
gate.reason = candidate_not_finite
gate.verdict = refuse

$ python3 scripts/registry.py --store sqlite:///mlflow.db promote 2 production
production: 1 -> 2
```

A model scoring NaN, in production, in one command. The refusal was already written on that
version's run and `promote` never looked. `history production` then shows an ordinary
transition, so nothing downstream can tell the difference.

**The first fix was wrong in two ways and both are worth keeping.**

*It guarded one door out of three.* `registry.promote` has three callers that reach
production. Adding a flag to the registry command left `deploy.py open` followed by
`deploy.py land` as a two command path onto the same NaN model, because `land` promotes
whatever is on trial and nothing asked how the trial started. Measured, not reasoned about.
The check now sits on `open` as well, which is where a trial should be refused. It is
deliberately not on `land`, because by then the canary has its own evidence.

*It checked that a verdict existed rather than that it still applied.* A comparison is
against something. Clearing a candidate while a weak model held production says nothing
about the strong model holding it now, and the tag is overwritten on every gate run so a
stale verdict is indistinguishable from a fresh one. The gate now records `gate.incumbent`
and the check refuses a verdict earned against anything else.

That second one was not hypothetical. A candidate that passed against the weak incumbent
came back `reject (not_separated)` when re-gated against the one actually in production.

```
$ registry.py promote 2 production --require-gate
refused: version 2: it was gated when version 1 held production and version 3 holds it now
```

**Where the check lives.** `mcr.gate.refusal_for` is a pure function over a run's tags, the
stage, and whoever holds that stage now. The library can answer whether the gate cleared a
version. It never asks the question itself, and `mcr.registry.promote` still knows nothing
about gates. Deciding to ask is policy and lives in the two commands that reach production.

**`rollback` never asks,** and under the staleness rule it could not. The version you return
to was cleared against an incumbent that is no longer there, so every recovery would refuse.
Recovery has to work when the world has moved on. Pinned as an absence in the checks.

**It is not a security control.** The verdict lives in an MLflow tag and a tag overwrites
silently, so anybody who can promote can write `gate.verdict=promote` first. This stops a
pipeline promoting something the gate refused. It does not stop a writer who means to, and
the transition log has the same property, which is why `settle` recovers a crash and does
not authenticate a writer.

## The comparison that is False in both directions

`configs/candidate-diverged.yml` trains to a NaN. `l2: 100` overflows and training does not
raise. The artefact serialises. The content hash is computed over a payload holding a NaN
and the registry takes it.

A comparison against a NaN is False whichever way round it is written.

```
lower is better, candidate is NaN:  nan < good   ->  False
lower is better, incumbent is NaN:  good < nan   ->  False
```

So a one line gate rejects a NaN candidate, which looks correct. And it rejects every
candidate forever once a NaN model holds the stage, with a message blaming the candidate.
One silent False, two completely different outcomes.

```
$ python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 1
candidate    version 1 of mcr-fraud
incumbent    version 3

holdout      56ab7ecf5a6b
metric       log_loss, lower is better

                                  on this holdout    roc auc   as the run ran
incumbent candidate-diverged                  nan   0.515824              nan
candidate baseline                       0.449239   0.675518         0.449239

verdict      refuse  (incumbent_not_finite)
             the incumbent candidate-diverged scores nan on log_loss, so nothing can be shown to beat it. Fix or unpoint the incumbent.
exit=2
```

`configs/candidate-inverted.yml` is the harder one. `l2: 10` does not diverge. Holdout AUC
comes back at 0.377109, which is worse than a coin. Log loss is 19.701270. Every metric is
finite and nothing about their shape is wrong. It puts one `RuntimeWarning` on stderr,
`overflow encountered in matmul`, and exits 0. Nothing reads exit code zero and then goes
hunting through stderr.

This paragraph used to give the size of that warning in bytes. The completion pass could
not reproduce the number and the reason is worth keeping. A warning carries the absolute
path of the file that raised it, so the byte count is the length of wherever you cloned
the repo plus 139. It read like a fact about the model and it was a fact about my
filesystem. See `docs/adr-0008`.

## What the gate scores, and what doing nothing scores

Nine cases with a known right answer, and the floor printed underneath, because a headline
of nine out of nine means nothing until somebody says what a coin gets.

```
incumbent              candidate              want      got       reason
-                      baseline               promote   promote   no_incumbent
baseline               candidate-lr           reject    reject    not_separated
baseline               candidate-underfit     reject    reject    worse
candidate-underfit     baseline               promote   promote   better
baseline               candidate-inverted     reject    reject    worse
baseline               candidate-diverged     refuse    refuse    candidate_not_finite
candidate-diverged     baseline               refuse    refuse    incumbent_not_finite
baseline               baseline               refuse    refuse    same_artifact
baseline               other-corpus           reject    reject    worse

  promote everything   2 of 9
  reject everything    4 of 9
  refuse everything    3 of 9
  the naive rule       4 of 9
  this gate            9 of 9
```

The naive rule is the one line this replaced, kept in the probe so the cases have something
to be compared against. It gets four, which is what rejecting everything gets.

The nine cases were written by me, so the honest thing to say is what is missing from them.
There is no case where a candidate is better by a small but real margin. The resolution
table below is what covers that ground instead.

## What the gate can actually see

```
epochs   gap vs the incumbent   relative   verdict
20                 8.2012e-03    1.8256%   worse
40                 1.1332e-03    0.2523%   worse
45                 7.6109e-04    0.1694%   not separated
100                2.8463e-05    0.0063%   not separated
400                0.0000e+00    0.0000%   not separated
```

Smallest difference it called, 1.1332e-03. Largest it declined, 7.6109e-04. So on 5,000
rows its resolution is about a fifth of one percent of the loss.

Put that beside the corpus span and the design argument is one line. **The number a gate
reading the store would have compared moves by 95 times the smallest real difference this
gate can detect.**

The interval is a two sided 95 percent interval on the mean paired difference, computed in
closed form. It started out as a percentile bootstrap and `docs/adr-0004` has the
measurement that took it off the decision path. A mutation pass moved the bootstrap's
default seed from 0 to 1 and survived, and on a candidate near the resolution above the
verdict really does flip between `worse` and `not_separated` across thirty two seeds.

A vote would not work either. `candidate-lr` beats `baseline` on 3,846 of 5,000 rows with a
mean difference of 1.8e-11, because the last bits of a float carry a consistent sign.

## The measurement

Reproducibility is a claim, so it gets a probe rather than a sentence. Five arms, measured
today on this machine.

```
same config, one process           OK    3 runs, 1 distinct hash
same config, fresh processes       OK    3 runs, 1 distinct hash, equal to the above
control: seed + 1 differs          OK
control: epochs + 1 differs        OK
control: init changes the fit      OK
```

The three controls are the point. A pipeline that ignored every input it was given would
pass the first two arms perfectly. The separate process arm matters on its own, because two
runs inside one interpreter can agree for reasons that do not survive a restart.

The suite is also checked by mutation, one AST node changed at a time, with the suite as the
oracle. Control clean either side of every pass.

```
mcr/config.py     36 of 36 killed      re-measured after the coercion change
mcr/gate.py       52 of 57
mcr/registry.py   31 of 32             oracle is the registry checks alone
mcr/tracking.py   10 of 10             oracle is tests/run_with_mlflow.py
mcr/seed.py        7 of 7
mcr/model.py      81 of 89
mcr/artifact.py    6 of 7
mcr/data.py       18 of 26
```

The gate pass is the one that changed the code rather than the tests. It went 24 of 40
first, and the sixteen survivors were classified one at a time by unparsing each mutant and
reading the diff. Two of them were the comparisons that decide a promotion, `hi < 0` and
`lo > 0`, alive because every fixture sat well away from both boundaries. One loosened the
clip in the loss, which is the only thing between a fully confident model and an infinite
loss. And one moved the bootstrap's default seed, which is what put the resampler on trial
and eventually off the decision path.

The five that are left in `gate.py` are `RESAMPLES`, which only the second interval reads,
and four inside the bisection that finds a normal quantile. That loop breaks on a tolerance
after about fifty rounds against a cap of two hundred, so its cap is unreachable, and the
comparisons around it are equalities on floats. The answer is pinned against four known
quantiles to nine decimal places instead.

The one survivor in `registry.py` steps the log sequence number by two instead of one.
Ordering comes from a lexical sort over the keys and nothing reads the number itself, so
gaps are invisible by construction. It reaches the four digit ceiling twice as fast, which
is the only difference and is ten thousand transitions away.

`tracking.py` needed its own oracle. Graded against `tests/run_all.py`, which cannot import
it, every mutant survives and the pass reads as a coverage disaster rather than as a pass
pointed at the wrong suite.

The eight survivors in `model.py` and the eight in `data.py` all leave the baseline artefact
byte identical, because a pinned hash check is in the suite. So none of them changes the
shipped model. What they could still do is change behaviour on an input nothing here
reaches, and closing that is the next job on the tests.

## What the first pass found

Two configs that differ only in learning rate, 0.5 against 1.5, produce different artefact
hashes and identical behaviour.

```
holdout_roc_auc      0.675518300215   vs 0.675518300215    gap 0.000e+00
holdout_accuracy     0.816000000000   vs 0.816000000000    gap 0.000e+00
holdout_log_loss     0.449238650332   vs 0.449238650314    gap 1.806e-11
max weight gap                                             2.695e-09
holdout rows whose 0.5 decision differs                    0 of 5000
```

Both runs converged to the same optimum, because the objective is convex and 400 epochs is
enough at either rate. So the content hash separates two models that nothing can tell apart.

That is why a model here carries two identities. The hash says which bytes are deployed. The
holdout metrics say which model is better. `docs/adr-0001` has the argument and the numbers,
including the problem this leaves for the promotion gate.

## Seeds

One seed per run, in the config. Every consumer gets its own generator derived from it by
name, and nothing reads a global.

```python
feat_rng, weight_rng, noise_rng, split_rng = streams(
    seed, "data.features", "data.weights", "data.noise", "data.split"
)
```

The alternative is `np.random.seed(n)` once at the top. That works until somebody adds a
draw or reorders two calls, and then every draw after the change moves. The run stays
deterministic and stops being the same run, which is the bad version because the artefact
changes and the config does not.

Derivation is a hash of the seed and the stream name rather than the seed plus an offset.
Under addition, seed 1 with the second stream and seed 2 with the first are the same number,
so two runs one seed apart would share a stream. `tests/test_seed.py` checks 160 pairs for
collisions.

## The canary, and the measurement it cannot make

The gate scores both models on one holdout. Every row is scored twice, so the comparison is
paired, and that is where its resolution comes from.

A canary routes a slice of traffic to the new model. Every request goes to one arm, so no
request is ever scored by both and the comparison becomes two independent samples. The
first draft of `mcr/canary.py` treated that as a detail.

Five thousand rows, one incumbent, candidates at the epoch counts below. Measured by
`scripts/canary_probe.py`, which takes no MLflow.

```
  epochs      true diff       corr  paired width split 5% width     ratio
       1    +1.9263e-01   0.976561    2.9242e-02    3.6825e-02       1.3
       5    +8.1657e-02   0.987130    1.9754e-02    6.6804e-02       3.4
      20    +8.2012e-03   0.996622    6.5510e-03    1.2007e-01      18.3
      40    +1.1332e-03   0.999390    2.1588e-03    1.3692e-01      63.4
      45    +7.6109e-04   0.999611    1.6742e-03    1.3863e-01      82.8
     100    +2.8463e-05   0.999998    1.1736e-04    1.4376e-01    1225.0
```

The cost of splitting is not a constant. It runs from 1.3 to 1225 and the correlation
column is why. Two models that disagree everywhere give pairing little to exploit. Two
models that are close give it a lot, and the split throws all of it away.

Which matters because **a canary only ever sees models that are close.** Anything obviously
bad has already been stopped by the gate. So the split is cheapest on the models a canary
will never be shown and most expensive on every model it will.

Here is that as a run. `configs/candidate-close.yml` is worse by 8.2012e-03 and the gate
rejects it deterministically.

```
$ python3 scripts/canary.py --store sqlite:///mlflow.db --canary 4 --fraction 0.25

                               requests    mean log_loss  positive rate
control baseline                   3696         0.452341         0.1864
canary candidate-close             1304         0.451203         0.1817

split   (served)   -1.137506e-03  interval [-3.289643e-02, +3.062142e-02]
shadow  (paired)   +8.201181e-03  interval [+4.925695e-03, +1.147667e-02]
the split interval is 9.7 times wider than the paired one
to reach the paired resolution the split needs about 605,539 requests at 25.0%

verdict      hold  (not_separated)
             the paired comparison on the same models says rollback
```

A quarter of all traffic, and the served numbers have the canary *ahead* by 1.1e-03 on a
model that is really behind by 8.2e-03.

The split comparison is not miscalibrated. Over two hundred draws at a five percent share,
two models that genuinely are alike come back promote about one draw in twenty, which is
the nominal rate of a two sided ninety five percent interval. It is correct and
underpowered, and the underpowering was chosen rather than forced.

So the module separates the two things a canary was doing at once. **Routing decides what
is served** and bounds the blast radius, which is worth having on its own. **Scoring
decides what is measured**, and `observe` scores both models on every request, which makes
the comparison paired again for one extra forward pass. The verdict still comes off the
served traffic, because that is the comparison a real canary has, and the shadow figures
sit beside it in the report. When they disagree the report says so.

That only works for a metric computable without knowing what was served. Log loss is. A
click on the recommendation that was shown is not, and for those the split is the only
comparison there is and the bill above is real. `docs/adr-0005` carries both halves.

## Known limitations

**The corpus is synthetic and nothing here says otherwise.** The generator is in
`mcr/data.py` and its shape was chosen by hand. No accuracy number produced by this repo
means anything about the world. What it can measure is whether the promotion machinery
behaves, which is the point of the project.

**The generator and the model do not share a functional form, deliberately.** Labels come
from log odds that are linear in the logs of the features, and noise is added before the
draw. If the generator were itself a fitted logistic model a good fit would be a tautology.

**There is no serving process anywhere in this project.** The registry holds which version
is in production and which is on trial, and nothing is answering requests. Traffic is a
replay of a generated holdout. So the operational half of a rollback, draining connections
and restarting something, is absent rather than implemented badly. The obvious way to test
a rollback is under load, and load is not a thing that can be measured here. What is
drilled is the state machine, which is the half that exists.

**The drills simulate a crash, they do not cause one.** Each damaged arm reproduces half of
`promote` by hand rather than killing a process mid write. That is a faithful model of the
state a crash leaves behind and it is not the same thing.

**The CI workflow's deploy job proves the wiring and not the deployment.** It ran for the
first time on 2026-09-16 and failed, and the fix is above. It passes now, but what it
exercises is a store seeded inside the same job and torn down with the runner. Nothing here
has ever deployed anything that outlived a build. A real pipeline's incumbent is whatever
last shipped, and this one's incumbent is a model trained ninety seconds earlier for the
purpose.

**Ten checks read that workflow and an eleventh kind of thing broke it.** All ten are worth
having and not one could see a sequence of correct commands that cannot happen in order.
Assume the next kind exists too.

**A crash between the alias move and the log write is not recoverable.** `settle` recovers
the other order by dropping an entry the store cannot confirm. This one leaves the version
that served absent from the log entirely, and the direction of the move with it, so
`rollback_target` refuses. The refusal names what it found and there is no repair.

**`--require-gate` is off by default on both commands, so the unguarded promotion is still
one flag away.** An operator who never passes it has exactly the behaviour a reader
flagged. Turning it on by default is a real option and the reason it has not been taken is
that it changes the meaning of every existing call rather than that it would be wrong.

**The gate requirement reads a mutable tag.** Anybody who can promote can write
`gate.verdict=promote` first. It is a guard against automation, not against intent.

**Only two of the three callers that reach production ask.** `scripts/canary.py` promotes on
a canary verdict and does not consult the gate, on the argument that the canary is its own
evidence. That argument is weaker than it sounds, because the canary can only have started
if somebody opened it, and `open` is only guarded when asked.

**The gate records its latest verdict on a run and not its earlier ones.** A candidate
gated three times against three incumbents keeps the third. Tags were chosen over params
because a run is written before it is ever gated and a param refuses a rewrite, so the
choice was between the latest verdict and none. Keeping the history means a child run per
comparison or a second table, and `docs/adr-0007` says why neither was built yet.

**`--no-report` has never met a store that refuses writes.** It is checked by parsing its
flag and by the report write being outside the promote branch, not by being pointed at a
read only backend, because there is no read only backend here.

**Nothing checks that a published figure is portable.** A pass over every published
figure found a byte count in this
README that was the length of my checkout path plus 139, so it changed on every machine and
reproduced on none. `docs/adr-0008` has the measurement. A figure that depends on a path, a
hostname, a timing or a temporary directory would fail the same way and nothing here would
catch it.

**`settle` recovers a crash and does not authenticate a writer.** The witness rule works
because `promote` and `retire` read the live alias into `from_version` immediately before
writing. Anything else with write access to the store can add a tag that satisfies the rule
and says whatever it likes. Defending against that needs a signature over each entry, which
is a different problem from the one this solves.

**`_next_seq` survives a mutation and the reason is that nothing rests on it.** Moving
`max(used) + 1` to `+ 2` leaves gaps in the sequence and changes no outcome, because the
keys are zero padded and only their order matters. It halves the headroom in the limitation
two entries below this one.

**A config that trains to a NaN loss still reaches the registry.** `l2: 100` overflows in
`mcr/model.py`, training returns a model, the artefact gets a hash and `registry.register`
takes it. The gate refuses to compare it and the gate is not on the registration path, so a
broken model can still be registered. It just cannot be promoted. Moving the check earlier
would mean `register` computing a metric, which means `register` knowing about holdouts, and
that is a worse shape than the gap.

**The holdout is not frozen in the usual sense.** `--holdout` defaults to
`configs/baseline.yml`, so the rows come from regenerating the incumbent recipe's own
corpus. Freezing properly means writing the rows to disk and hashing the file. The
fingerprint already identifies the rows, so that is a small change rather than a large one,
and it has not been made.

**The gate checks no latency and no fairness.** A serious promotion gate usually checks
both. Latency is measurable here and is not measured. Fairness has no subject at all,
because the corpus is synthetic and carries no attribute anybody would protect, and
inventing one would be a metric wearing a credible name.

**The interval is on the mean.** A candidate that is better on most rows and catastrophic on
a few can still win. A quantile of the paired difference would say something the mean does
not, and there is nothing here that looks at the shape of the difference at all.

**The nine gate cases were written by the same person who wrote the gate.** They cover no
case where a candidate is genuinely better by a small margin, which is the case a real
pipeline sees most often. The resolution table covers that ground by walking a candidate
toward the incumbent, and that is a different thing from a case with a known right answer.

**The transition log holds ten thousand entries per stage.** The sequence number is zero
padded to four digits and ordering is a lexical sort over the tag keys, so entry 10000
sorts into the wrong place. Not worth engineering around and worth knowing.

**`promote` cannot survive two writers.** It picks a sequence number from a read and there
is no compare and set in this API. It reads the key back and refuses if it does not hold
what it wrote, which detects a lost update rather than preventing one. Single writer here,
so it has never fired.

**`scripts/track_probe.py` cannot see a typing defect and I know it.** MLflow returns every
value as a string, so the rebuild only ever exercises one branch of `config.coerce`. The
float rule was reverted on purpose and all the probe's arms stayed green while the check in
`tests/test_config.py` failed. The probe's own docstring says this. The suite owns that half
and the probe owns the store half.

**Nothing grades `scripts/track_probe.py` itself.** Its comparisons live in the script
rather than in a module the suite imports, so a defect in an arm is invisible to mutation.
The same is true of `scripts/repro_probe.py`. Moving the arithmetic into `mcr/` is the fix
and it is not done. `scripts/registry_probe.py` is the same shape and answers it differently
for now: every arm about this repo's code runs a control against a stand in carrying the
defect the arm exists to catch, and the probe fails if a control passes. That caught a bad
arm on the first run, where the naive rollback guess and the right answer happened to agree.

**The holdout AUC of 0.676 is modest and that is the corpus, not the optimiser.** The noise
term is 0.9 in log odds and nine of the twelve features are damped to near zero weight.
Turning that number up would be a matter of editing the generator, which is the reason it is
not a number worth reporting as an achievement.

**The MLflow behaviour above was measured on one version on one machine.**
`requirements-tracking.txt` is deliberately unpinned. Every number in the tracking section
came off mlflow 3.16.0 here, and pinning a version I have only ever run in one place would
invent provenance I do not have.

## What I would do differently

The first version of the label generator put the intercept at the `1 - positive_rate`
quantile of the log odds. That sets the point where the probability crosses 0.5 and not the
mean of the probabilities, so a config asking for a 20 percent positive rate got 29.

The code was wrong and the comment above it described a root find that had never been
written. It was caught by a check asserting the realised rate, which I nearly did not write
because the line looked obviously correct. Shape checks would have passed. Every metric
downstream would have been computed on a corpus nobody asked for.

`mcr/data.py` now solves for the intercept by bisection, and `tests/test_data.py` pins both
the fix and the defect, so the shortcut cannot come back quietly.

The second one is worse and it was found by building on top of it. The config fingerprint
was a hash of the JSON of the config object, and the config object held whatever
`yaml.safe_load` produced, so the type was never checked against what the dataclass
declared. `noise: 1` and `noise: 1.0` are the same training run. They fingerprinted
differently, and because the fingerprint sits inside the artefact payload, the artefact hash
differed too.

```
                      noise: 1          noise: 1.0
config fingerprint    6299a382bfec      003ec18d7c82
artifact hash         a556e05066f1...   0284b85ef746...
model section         845 bytes         845 bytes, BYTE IDENTICAL
payload keys differing                  ['config_fingerprint']
```

The registry key moved on a YAML typing detail while the model stayed identical to the byte.
A check asking whether production runs the approved artefact would have said no.

`mcr/config.py` now coerces every value to its declared type and refuses any cast that
loses something, so `epochs: 5.5` is an error rather than 5. Getting the types at all needed
`get_type_hints`, because `dataclasses.fields(cls)[i].type` is the string `"int"` in a
module with postponed annotations and the obvious `f.type(value)` calls a string.

The check that ran first was the one confirming the two shipped configs still fingerprint to
`70a46feb1141` and `80a2bc7f7405`. A fix to a key derivation that quietly moved every
published key would have been worse than the defect.
