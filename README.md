# model-cicd-registry

A model training pipeline where the same config always produces the same bytes, and where
the thing that identifies a model is kept separate from the thing that decides whether it
is any good. The end state is a merge that takes a model all the way to a canary deploy.
Rollback is one command.

What is here so far is the config system, the seed control, the training pipeline and the
MLflow tracking layer. The registry, the promotion gate and the deploy are not built yet.

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

## What is here

```
mcr/config.py     frozen dataclass, YAML on disk, refuses unknown keys, content fingerprint
mcr/seed.py       named random streams derived from one run seed
mcr/data.py       synthetic corpus, deterministic given the seed
mcr/model.py      logistic regression by gradient descent, numpy only
mcr/artifact.py   deterministic serialisation and a content hash
mcr/train.py      the pipeline. config in, artefact out, nothing else
mcr/tracking.py   MLflow. write a run, read it back, rebuild the config from what came back
```

Three entry points in `scripts/`. Two runners in `tests/`, carrying 85 checks without
MLflow and 104 with it.

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
free, which is what a recipe should be, and tags are where the registry's stage
transitions belong.

Two more measurements from the same pass. A param value over 6,000 characters and a tag
value over 8,000 come back **truncated**, with a warning on stderr and no exception, so a
content hash long enough to be shortened would become a registry key pointing at nothing.
And the file store is gone in this version: mlflow 3.16.0 raises on a `file:` tracking URI
and tells you to use a database, so every command above uses sqlite.

`docs/adr-0002` has the rest, including the defect this layer uncovered in the config
fingerprint.

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
mcr/tracking.py   10 of 10             oracle is tests/run_with_mlflow.py
mcr/seed.py        7 of 7
mcr/model.py      81 of 89
mcr/artifact.py    6 of 7
mcr/data.py       18 of 26
```

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

## Known limitations

**The corpus is synthetic and nothing here says otherwise.** The generator is in
`mcr/data.py` and its shape was chosen by hand. No accuracy number produced by this repo
means anything about the world. What it can measure is whether the promotion machinery
behaves, which is the point of the project.

**The generator and the model do not share a functional form, deliberately.** Labels come
from log odds that are linear in the logs of the features, and noise is added before the
draw. If the generator were itself a fitted logistic model a good fit would be a tautology.

**Nothing has been registered, gated or deployed yet.** Tracking is in. The registry with
stage transitions comes next, then a promotion gate against a frozen holdout, then a canary
deploy with a rollback.

**`scripts/track_probe.py` cannot see a typing defect and I know it.** MLflow returns every
value as a string, so the rebuild only ever exercises one branch of `config.coerce`. The
float rule was reverted on purpose and all the probe's arms stayed green while the check in
`tests/test_config.py` failed. The probe's own docstring says this. The suite owns that half
and the probe owns the store half.

**Nothing grades `scripts/track_probe.py` itself.** Its comparisons live in the script
rather than in a module the suite imports, so a defect in an arm is invisible to mutation.
The same is true of `scripts/repro_probe.py`. Moving the arithmetic into `mcr/` is the fix
and it is not done.

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
