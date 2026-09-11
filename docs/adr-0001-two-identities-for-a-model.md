# ADR 0001: a model needs two identities and they answer different questions

Status: accepted.

## Context

The registry keys on something. The promotion gate compares on something. It looked
obvious that both should be the artefact content hash, because it is exact, cheap and already
there.

Measuring it on the first two configs this repo ships says otherwise.

## What was measured

Both runs below are on this machine, today, at 20,000 rows over 12 features with a 5,000 row
holdout. `configs/baseline.yml` and `configs/candidate-lr.yml` are identical except for the
learning rate, which is 0.5 against 1.5.

```
config fingerprint     70a46feb1141          80a2bc7f7405
artifact hash          8b83b8d77ac9...       827ec647b844...
holdout_roc_auc        0.675518300215        0.675518300215      gap 0.000e+00
holdout_accuracy       0.816000000000        0.816000000000      gap 0.000e+00
holdout_log_loss       0.449238650332        0.449238650314      gap 1.806e-11
max weight gap                                                   2.695e-09
holdout rows whose 0.5 decision differs                          0 of 5000
max per row probability gap                                      2.014e-09
```

The same shape appears from the other direction. Holding the config fixed and moving only
`init` from `zeros` to `normal` gives two different artefact hashes, a max weight gap of
4.003e-12, and a holdout AUC identical to every digit printed.

## The point

A tripled learning rate is not a small change to a training run. It produces a different
artefact hash, a different config fingerprint and a model that is indistinguishable from the
first one on every measurement the gate is going to make. Zero of five thousand holdout rows
change their decision.

Both runs converged. The objective is convex and 400 epochs is enough at either rate, so both
arrived at the same optimum and the remaining difference is float noise from a different path
to the same place.

So the two identities disagree, and each is right about its own question.

- The **artefact hash** answers "are these the same bytes". That is a deployment question. It
  is what a rollback target has to be, and what tells you whether the thing running in
  production is the thing that was approved.
- The **holdout metrics** answer "is this a better model". That is the promotion question.

The hash is strictly finer than behaviour. It separates models nothing can tell apart. Using
it to decide promotion would reject a rebuild of the incumbent. Using metrics alone to decide
deployment identity would let two different binaries share one slot.

## Decision

Store both, and never let one stand in for the other.

1. `config.fingerprint()` identifies the recipe. Two runs with the same fingerprint should
   produce the same model, so it is what a cache or a skip-rebuild check keys on.
2. `artifact.content_hash()` identifies the bytes. It is the registry primary key and the
   rollback target.
3. Holdout metrics decide promotion. The gate reads these and nothing else.

## Consequences, including the uncomfortable one

The promotion gate has a problem waiting for it and it is visible from here. If two configs this
different score identically, the gate has almost no signal to work with on this corpus. A
promotion rule of "candidate beats incumbent on holdout AUC" would be deciding on a gap of
0.000e+00, which is a coin flip wearing a threshold.

That is a fact about the corpus rather than about the gate. The generator produces a problem
a logistic model solves to the optimum from any reasonable starting point, so there is nothing
for two logistic models to disagree about. The gate needs candidates that are genuinely worse,
not merely different, and `configs/candidate-lr.yml` is not one.

Recorded now rather than discovered later. The honest version of the gate demo is a
deliberately crippled candidate, which was the intention in the first place.

## What was rejected

**Hashing the weights to a tolerance.** Rounding the floats before hashing would make the two
runs above collide and would make the hash stable under float noise. It also makes the hash
non transitive, because two models can each be within tolerance of a third and not of each
other, and a registry key that is not an equivalence relation is not a key.

**Dropping the config fingerprint and keying only on the artefact.** It works until a run
fails halfway. Then there is no artefact and no way to say what was being attempted.
