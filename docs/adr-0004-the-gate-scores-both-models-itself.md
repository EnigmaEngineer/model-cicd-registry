# ADR 0004: the gate scores both models itself, and it reads log loss

Status: accepted.

## Context

The gate has to answer one question. Does this candidate beat what is in production.

The obvious implementation is one line. Both runs are in the tracking store, both recorded
a `holdout_roc_auc` and a `holdout_log_loss`, so compare the two numbers. That is what the
first draft did and it is wrong twice over. Both are measured below.

## Measurement one: AUC cannot separate anything here

Ten model configs on one corpus, from one epoch to four hundred and from a learning rate of
0.0001 up to 1.5. Reproduce with `python3 scripts/gate_probe.py`.

```
epochs=1         auc 0.675312072  logloss 0.641871  acc 0.814800
epochs=2         auc 0.675330627  logloss 0.602641  acc 0.814800
epochs=5         auc 0.675346532  logloss 0.530895  acc 0.814800
epochs=20        auc 0.675396366  logloss 0.457440  acc 0.815000
epochs=100       auc 0.675513529  logloss 0.449267  acc 0.816000
epochs=400       auc 0.675518300  logloss 0.449239  acc 0.816000
lr=0.0001        auc 0.675316579  logloss 0.688811  acc 0.814800
lr=0.01          auc 0.675366413  logloss 0.499915  acc 0.814800
lr=0.1           auc 0.675501070  logloss 0.449329  acc 0.816200
lr=1.5           auc 0.675518300  logloss 0.449239  acc 0.816000

auc      span 2.062280e-04
accuracy span 1.400000e-03
logloss  span 2.395728e-01
```

AUC reads the ranking and nothing else, and a logistic model settles the direction of its
weight vector in the first few epochs. Everything after that is calibration, which AUC is
blind to by construction. A gate reading it passes `configs/candidate-underfit.yml`, which
trains for exactly one pass. Accuracy is barely better, because at a 0.185 positive rate
almost every threshold decision is the same one.

So the metric is log loss, lower is better. `tests/test_gate.py` pins this from the other
side too, with a model whose weights are scaled by three. The ranking is identical to the
bit so AUC is equal, and the loss is worse.

## Measurement two: the recorded number is partly a fact about the corpus draw

Every run generates its own corpus from its own config, so its holdout is a slice of that
corpus. Two runs share a holdout only when their data section and their seed both match,
and nothing was checking that.

Hold the model config completely still and move only the seed.

```
12 seeds, each on its own holdout: 0.341545 to 0.449239, span 0.107693
the gap the gate exists to catch:  0.192633
so the corpus draw is 55.9% of that gap
```

The seed is inside the config fingerprint, so two configs differing only in it are two
legitimate candidates as far as the rest of this project is concerned.

The consequence is sharper than the percentage makes it sound. A model trained on another
corpus records 0.341545 against the incumbent's 0.449239. Scored on the incumbent's
holdout it comes back at 0.811947. The one line rule promotes it.

Part of that is this generator drawing a fresh set of true weights per seed, so a model
from another corpus has no reason to transfer at all. The 0.107693 span is the honest
number and it is measured with nothing transferring anywhere. Both are in the probe.

## Decision

The gate does not read the metrics either run recorded. It takes a `HoldoutSpec` and
builds one holdout from it. Then it rebuilds both models out of their artefact bytes and
scores them on the same rows.

The holdout is its own object rather than "whatever the candidate's config says". A
candidate that supplies its own holdout is a candidate choosing its own exam.

The recorded numbers still go in the comparison report, in a column labelled as what each
run said about its own holdout. When the two disagree, the disagreement is the finding.

`HoldoutSpec.fingerprint` hashes the rows rather than the spec fields. Hashing the fields
would be cheaper and would hold still through a change to the generator that moved every
row underneath it.

## Why an interval rather than a threshold

Two models scored on the same rows can be compared row by row, so the comparison is paired
and there is no sampling noise between two scorings of it. What is left is whether the
difference is bigger than the holdout it was measured on can support.

A vote does not answer that. On the shipped configs `candidate-lr` is better than
`baseline` on 3,846 of 5,000 rows, and the mean difference is 1.8e-11, because the last
bits of a float carry a consistent sign. Anything counting rows promotes a model that is
identical to eleven decimal places. A strict `<` on the means does the same thing.

So the gate builds a two sided 95 percent interval on the mean paired difference. It
promotes when the whole interval is below zero and rejects otherwise.

## The interval is closed form, and the reason is a defect this repo shipped

The first version was a percentile bootstrap. A bootstrap needs a resample count and a
seed, and neither is a number anybody chose.

On a candidate sitting near the gate's resolution, four hundred epochs against forty, the
verdict came back `worse` on some bootstrap seeds and `not_separated` on others across
thirty two of them. A mutation pass found it, by moving the default seed from 0 to 1 and
surviving. Pinning the seed harder would only have buried the instability under a constant.

The mean of five thousand paired differences is normal enough for the closed form, and the
two agree on every case in the probe, so the resampler was buying nothing and charging an
arbitrary constant for it.

`bootstrap_interval` is still in the module and nothing on the decision path calls it. It
is there so the interval the gate does use has a second implementation to be graded
against, and `tests/test_gate.py` asserts they agree at 5,000 rows and records that they
can disagree at 500.

## What the gate can see

```
epochs   gap vs the incumbent   relative   verdict
20                 8.2012e-03    1.8256%   worse
40                 1.1332e-03    0.2523%   worse
45                 7.6109e-04    0.1694%   not separated
100                2.8463e-05    0.0063%   not separated
400                0.0000e+00    0.0000%   not separated
```

The smallest difference it called is 1.1332e-03 and the largest it declined is 7.6109e-04,
so its resolution on 5,000 rows is about a fifth of one percent of the loss.

Put that next to the corpus span and the design argument becomes one line. The number the
one line rule was reading moves by 95 times the smallest real difference this gate can
detect.

## Three verdicts, not two

`promote`, `reject` and `refuse`. A reject is a fact about the candidate. A refuse is a
fact about the gate's inputs, and the two need different handling by whatever called it.
The command line returns 0, 1 and 2 for exactly this reason. A pipeline treating the last
two the same retries a broken incumbent forever.

## Non finite metrics are refused, from both sides

A config can train to a NaN and still produce a registerable artefact. `l2: 100` overflows
and the holdout log loss comes back NaN. Training does not raise and the artefact
serialises. The content hash is computed over a payload holding a NaN and the registry
takes it.

A comparison against a NaN is False whichever way round it is written.

```
lower is better, candidate is NaN:  nan < good   ->  False
lower is better, incumbent is NaN:  good < nan   ->  False
```

So `candidate < incumbent` rejects a NaN candidate, which looks correct and is correct for
the wrong reason, and rejects every candidate forever when the incumbent is the NaN. Same
silent False. The second case is a production outage carrying a message that blames the
candidate.

Both are refusals here, with different reason codes, and the incumbent message names the
incumbent. An operator reading the wrong name goes and fixes the wrong model.

`configs/candidate-inverted.yml` is the harder case. `l2: 10` does not diverge. Holdout AUC
comes back at 0.377109, worse than a coin, with log loss at 19.701270 and every metric
finite. It puts 204 bytes of RuntimeWarning on stderr and exits 0. Nothing reads exit code
zero and then goes looking through stderr, so the gate is the only thing standing in front
of it, and the gate catches it on the metric like any other bad model.

## What was rejected

**Comparing the stored metrics.** Measured above. It reads the corpus.

**Retraining both models from their recovered configs and scoring those.** The rebuild path
exists, because the registry stores an artefact hash and a run id and not the artefact, and
`source` on a version points at a path nothing in this project writes. But scoring a
retrained model is scoring something the registry does not point at unless something checks
the bytes. `gate.rebuild` does check, and refuses when the hash moved. That refusal is the
load bearing half.

**A fixed threshold on the difference.** A threshold is a number chosen before seeing how
much the holdout can support, and the honest version of it is the interval.

**Letting the candidate supply the holdout.** Named above. It is the exam problem.

**Refusing a candidate whose training corpus differs from the incumbent's.** This would be
defensible and it is stricter than necessary. Two models trained on different corpora are
still two models, and scoring both on one holdout is a question with an answer. Refusing
would also make the gate unusable the first time the training data is legitimately
refreshed, which is the normal case in any real pipeline.

## What is not solved

**One holdout, chosen by whoever runs the gate.** `--holdout` defaults to
`configs/baseline.yml`, so the rows are the incumbent recipe's own holdout. That is a
defensible default and it is not a frozen holdout in the sense the phrase usually means,
which is a set fixed once and never regenerated. Freezing it properly means writing the
rows to disk and hashing the file, and the fingerprint is what makes that a small change
rather than a large one.

**The gate has no opinion about latency or fairness.** A serious promotion gate usually
checks both and this one checks neither. Latency is measurable here and is simply not
measured yet. A fairness check has no subject at all, because the corpus is synthetic and
carries no attribute anybody would protect. Adding one would be a metric wearing a
credible name, which is worth less than an admitted gap.

**The interval is on the mean and the mean is not the only thing that matters.** A
candidate that is much better on most rows and catastrophic on a few can win here. A
quantile of the paired difference would say something the mean does not.
