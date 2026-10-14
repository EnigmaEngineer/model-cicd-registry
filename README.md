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

The failure drills are the evidence for the rollback path and they run in seconds.

```
python3 scripts/drill.py
```

Thirteen drills over states a crash really produces, each with a control that has to come
back healthy. Tracking and the registry have their own commands, and so do the gate and
the canary. All of them are in [docs/running-the-checks.md](docs/running-the-checks.md).

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

## Results

**Same config, same bytes, across fresh processes.**

```
same config, one process           OK    3 runs, 1 distinct hash
same config, fresh processes       OK    3 runs, 1 distinct hash, equal to the above
control: seed + 1 differs          OK
control: epochs + 1 differs        OK
control: init changes the fit      OK
```

**The promotion gate gets nine of nine cases right, and doing nothing gets four.** Nine
cases with a known answer, and the floor printed under them. Rejecting every candidate
scores four of nine, which is exactly what the one line rule this replaced scored.

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

On 5,000 holdout rows the gate's resolution is about a fifth of one percent of the loss.
The number a gate reading the store would have compared moves by 95 times that.

**A crash mid deploy made rollback pick a version that never served.** The transition log
is written before the alias moves and the two are not in a transaction. A crash while
writing version 4, followed by shipping version 5, left a log saying `[1, 2, 3, 4, 5]` and a
rollback target of 4. The check meant to catch that passed. `registry.settle` now drops any
log entry nothing in the log or the alias can witness, and the drill reads 3, matching its
crash free control.

**Three static checks passed on a CI workflow that could not run.** For six days there was
no runner, so the file was read rather than executed. The first real run failed the deploy
job, because the canary had no incumbent to compare against. The workflow test now hands
every script call in the file to that script's own argument parser.

## Known limitations

- The corpus is synthetic. No accuracy number here means anything about the world.
- There is no serving process anywhere in this project. Traffic is a replay of a generated
  holdout.
- The drills simulate a crash. They do not cause one.
- `--require-gate` is off by default, so the unguarded promotion is still the default path.
- The gate requirement reads a mutable MLflow tag. It is not a security control.
- A config that trains to a NaN loss still reaches the registry.
- `promote` cannot survive two writers.
- The nine gate cases were written by the same person who wrote the gate.

The full list is in [docs/limitations.md](docs/limitations.md), with what I would do
differently.

## More detail

**[Design notes](docs/design-notes.md)**

- [Tracking, and the question it is built to answer](docs/design-notes.md#tracking-and-the-question-it-is-built-to-answer)
- [The registry, and the two things MLflow will not do](docs/design-notes.md#the-registry-and-the-two-things-mlflow-will-not-do)
- [The reference that names two models](docs/design-notes.md#the-reference-that-names-two-models)
- [Rollback is not "whatever it held before"](docs/design-notes.md#rollback-is-not-whatever-it-held-before)
- [A crash mid deploy made rollback pick a version that never served](docs/design-notes.md#a-crash-mid-deploy-made-rollback-pick-a-version-that-never-served)
- [Opening a canary is a change to the registry](docs/design-notes.md#opening-a-canary-is-a-change-to-the-registry)
- [CI, and what three static checks could not see](docs/design-notes.md#ci-and-what-three-static-checks-could-not-see)
- [What the rollback changed, and what it did not](docs/design-notes.md#what-the-rollback-changed-and-what-it-did-not)
- [The gate, and why it does not read the numbers in the store](docs/design-notes.md#the-gate-and-why-it-does-not-read-the-numbers-in-the-store)
- [Rejecting a model](docs/design-notes.md#rejecting-a-model)
- [The comparison that is False in both directions](docs/design-notes.md#the-comparison-that-is-false-in-both-directions)
- [What the gate scores, and what doing nothing scores](docs/design-notes.md#what-the-gate-scores-and-what-doing-nothing-scores)
- [What the gate can actually see](docs/design-notes.md#what-the-gate-can-actually-see)
- [The measurement](docs/design-notes.md#the-measurement)
- [What the first pass found](docs/design-notes.md#what-the-first-pass-found)
- [Seeds](docs/design-notes.md#seeds)
- [The canary, and the measurement it cannot make](docs/design-notes.md#the-canary-and-the-measurement-it-cannot-make)

**[Running the checks](docs/running-the-checks.md)**

**Other docs**

- [adr-0001-two-identities-for-a-model](docs/adr-0001-two-identities-for-a-model.md)
- [adr-0002-the-fingerprint-was-a-hash-of-the-yaml](docs/adr-0002-the-fingerprint-was-a-hash-of-the-yaml.md)
- [adr-0003-the-stage-is-an-alias](docs/adr-0003-the-stage-is-an-alias.md)
- [adr-0004-the-gate-scores-both-models-itself](docs/adr-0004-the-gate-scores-both-models-itself.md)
- [adr-0005-a-canary-cannot-pair-and-that-is-the-cost](docs/adr-0005-a-canary-cannot-pair-and-that-is-the-cost.md)
- [adr-0006-a-crash-mid-deploy-can-make-rollback-pick-a-version-that-never-served](docs/adr-0006-a-crash-mid-deploy-can-make-rollback-pick-a-version-that-never-served.md)
- [adr-0007-the-audit-trail-was-biased-toward-wins](docs/adr-0007-the-audit-trail-was-biased-toward-wins.md)
- [adr-0008-a-published-byte-count-was-measuring-my-filesystem](docs/adr-0008-a-published-byte-count-was-measuring-my-filesystem.md)
