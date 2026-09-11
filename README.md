# model-cicd-registry

A model training pipeline where the same config always produces the same bytes, and where
the thing that identifies a model is kept separate from the thing that decides whether it
is any good. The end state is a merge that takes a model all the way to a canary deploy.
Rollback is one command.

What is here so far is the config system, the seed control and the training pipeline. The
registry, the promotion gate and the deploy are not built yet.

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

## What is here

```
mcr/config.py     frozen dataclass, YAML on disk, refuses unknown keys, content fingerprint
mcr/seed.py       named random streams derived from one run seed
mcr/data.py       synthetic corpus, deterministic given the seed
mcr/model.py      logistic regression by gradient descent, numpy only
mcr/artifact.py   deterministic serialisation and a content hash
mcr/train.py      the pipeline. config in, artefact out, nothing else
```

Two entry points in `scripts/` and 78 checks in `tests/`.

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

The suite is also checked by mutation, one AST node changed at a time against
`tests/run_all.py` as the oracle. Counts measured today, control clean either side of every
pass.

```
mcr/config.py     30 of 30 killed
mcr/seed.py        7 of 7
mcr/model.py      81 of 89
mcr/artifact.py    6 of 7
mcr/data.py       18 of 26
```

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

## Known limitations

**The corpus is synthetic and nothing here says otherwise.** The generator is in
`mcr/data.py` and its shape was chosen by hand. No accuracy number produced by this repo
means anything about the world. What it can measure is whether the promotion machinery
behaves, which is the point of the project.

**The generator and the model do not share a functional form, deliberately.** Labels come
from log odds that are linear in the logs of the features, and noise is added before the
draw. If the generator were itself a fitted logistic model a good fit would be a tautology.

**Nothing has been registered, gated or deployed yet.** Tracking and a registry come
first, then a promotion gate against a frozen holdout, then a canary deploy with a
rollback.

**The holdout AUC of 0.676 is modest and that is the corpus, not the optimiser.** The noise
term is 0.9 in log odds and nine of the twelve features are damped to near zero weight.
Turning that number up would be a matter of editing the generator, which is the reason it is
not a number worth reporting as an achievement.

**No MLflow yet.** Tracking and the registry come next. The artefact format here is
deliberately plain JSON so that whatever stores it later has nothing to unpick.

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
