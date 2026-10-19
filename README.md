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

Twenty three drills over states a crash really produces, each with a control that has to come
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
mcr/attest.py     a signature over one transition log entry, and the three readings of it
mcr/gate.py       whether a candidate is allowed to replace the incumbent
mcr/canary.py     routing a slice of traffic, and the two comparisons that gives you
mcr/deploy.py     what is deployed. production, the canary on trial, and the share it takes
```

Twelve entry points in `scripts/`. Three runners in `tests/`, carrying 307 checks without
MLflow and 433 with it. Those two counts are derived by `tests/test_readme.py`, which fails
with the right answer, because both of them were typed once and then went stale by more
than thirty. The third runner covers the registry and deployment modules only and comes in
at under half the full store suite, which is what a mutation pass over those modules needs
to fit inside one shell invocation.

## Results

**Same config, same bytes, across fresh processes.**

```
same config, one process           OK       3 runs, 1 distinct, 8b83b8d77ac9
same config, fresh processes       OK       3 runs, 1 distinct, 8b83b8d77ac9
control: seed + 1 differs          OK       2455cc6f00a6
control: epochs + 1 differs        OK       1b33f9807369
control: init changes the fit      OK       zeros 8b83b8d7 normal 5dae2066
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

**A forged transition log entry moved the rollback target, and the integrity check has to
run before the recovery one.** Entries now carry a MAC over the body and the slot they sit
in. Run recovery first and it drops the forgery as an unwitnessed crash artefact, takes the
genuine entry beside it, and what is left reads as a healthy log.

```
ok     forged log entry refused               damaged: refused                        control: 2
ok     recovery first answers instead of refusing 1 (wanted 1)
ok     because settle dropped the evidence first 0 rejected after settle (wanted 0 rejected after settle)
```

**A CI workflow can go green having decided nothing.** The file is executed here now
rather than only read. `scripts/ci_local.py` runs a job the way a runner does, with each
block in its own shell and every `if` evaluated against a real `$GITHUB_OUTPUT`. Driving
the gate to a reject found a step that ran on a rejected candidate and swallowed the
refusal. Ten static checks passed on it. `docs/adr-0011`.

**The registry refuses an artefact that never trained, and one of the two refusals was not
enough on its own.** `register` takes the artefact rather than its hash and applies an
admission rule before the store is touched. Measured end to end against a sqlite store.

No command prints the next table. Each row is `mcr.admission.inspect` over the artefact that
config trains, and the last column is the verdict's `reasons` tuple rather than its message.

| config | logit scale | admitted | reasons |
|---|---|---|---|
| baseline | 2.23082 | yes | |
| candidate-close | 1.69692 | yes | |
| candidate-lr | 2.23082 | yes | |
| candidate-underfit | 0.20277 | yes | |
| candidate-inverted | 6.03803e+238 | no | `non_finite_parameter` `saturated_parameters` |
| candidate-diverged | nan | no | `non_finite_metric` `non_finite_parameter` |

Saturation starts at logit 36.7368005696771, where a float64 sigmoid returns exactly 0 or 1.

`candidate-inverted` is the case worth the attention. At `l2` 10 it diverges with every
metric finite. Holdout log loss is 19.70 and AUC is 0.377109, both real numbers, against a
baseline AUC of 0.6755. A rule reading only the recorded metrics admits it. It is caught on
`model.final_loss`, which carries the l2 penalty and overflows, and caught again on the
parameter scale. The threshold for that second rule is `log(2 / eps)`, or 36.736801, which
is the logit at which a float64 sigmoid stops being able to return anything below 1.

## Known limitations

- The corpus is synthetic. No accuracy number here means anything about the world.
- There is no serving process anywhere in this project. Traffic is a replay of a generated
  holdout.
- The drills simulate a crash. They do not cause one.
- `--require-gate` is off by default, so the unguarded promotion is still the default path.
- The gate requirement reads a mutable MLflow tag. It is not a security control.
- Admission asks whether a model trained, never whether it is any good. `candidate-underfit` passes it and is meant to.
- `create_version` is the unguarded store primitive. A check keeps it out of `mcr/`, and checks and probes still reach it deliberately.
- `promote` cannot survive two writers.
- The transition log signature authenticates whoever holds one key. There is no rotation and no per pipeline identity.
- A transition log with no signed entry in it is undefended by default, because the rule refusing an unsigned entry needs a signed one to anchor on. `require_attested` is the only setting that covers it.
- The nine gate cases were written by the same person who wrote the gate.

The full list is in [docs/limitations.md](docs/limitations.md), with what I would do
differently.

## More detail

**[Design notes](docs/design-notes.md)**

- [The registry, and the two things MLflow will not do](docs/design-notes.md#the-registry-and-the-two-things-mlflow-will-not-do)
- [Rollback is not "whatever it held before"](docs/design-notes.md#rollback-is-not-whatever-it-held-before)
- [A crash mid deploy made rollback pick a version that never served](docs/design-notes.md#a-crash-mid-deploy-made-rollback-pick-a-version-that-never-served)
- [The gate, and why it does not read the numbers in the store](docs/design-notes.md#the-gate-and-why-it-does-not-read-the-numbers-in-the-store)
- [CI, and what three static checks could not see](docs/design-notes.md#ci-and-what-three-static-checks-could-not-see)

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
- [adr-0009-admission-is-a-separate-question-from-promotion](docs/adr-0009-admission-is-a-separate-question-from-promotion.md)
- [adr-0010-recovery-and-authorisation-are-two-questions](docs/adr-0010-recovery-and-authorisation-are-two-questions.md)
