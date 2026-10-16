# ADR 0009. Admission is a separate question from promotion

Status: accepted

## The defect

A config that overshoots produces a weight vector of NaNs and a registerable artefact.
Measured on `configs/candidate-diverged.yml` at `l2` 100 with no changes to the code.
Training emits a hundred overflow warnings and `holdout_log_loss` comes back `nan`. Then
`artifact.build` serialises the payload and the content hash is computed over bytes
containing a NaN. Then `registry.register` hands back a version number. Every step did
exactly what it says it does. Nothing on the path from a config to a registered version asked whether the
numbers mean anything.

The promotion gate already refuses a non finite candidate. That refusal is real and it is
not enough, because it only fires when somebody proposes a promotion. An artefact sitting in
the registry has a version number, and the deploy path, the rollback walk and every audit
query treat a version number as a model.

## The decision

Three parts, and the second is the one that took the longest to settle.

**Detection is a pure function of the payload.** `mcr/admission.py` reads an artefact
payload and returns a verdict. It computes nothing, loads no data and trains nothing. The
objection that kept this check out of the registry for a cycle was that the registry would
have to compute a metric to apply it. It does not. The payload already carries every metric
and every parameter, so the only thing the registry was missing was the payload itself.

**Refusal happens at `register`, and `register` now takes the artefact.** It derives the
content hash rather than being handed one. Being handed a hash means being handed a name for
bytes the function cannot read, which is the whole reason the check had nowhere to live. The
store primitive it used to be is still there as `create_version`, unguarded and documented
as such, because the store's own behaviour has to stay testable without a trained model.

**`train.run` and `artifact.build` stay permissive on purpose.** Both are further upstream
and either one could refuse earlier. Refusing there would stop the gate and the canary ever
seeing a non finite candidate, and both have refusals written for exactly that case with
checks behind them. A guard that makes another guard untestable is a bad trade. So training
still returns a diverged fit, the artefact still serialises, and the door is at the point
where an artefact becomes a thing the rest of the system will refer to.

## The two conditions, and why one was not enough

Finiteness is the obvious one. Every float in the payload has to be finite. The rule walks
the payload rather than reading a list of fields, so a number added later is covered without
an edit.

Saturation is the one that came out of a measurement. `configs/candidate-inverted.yml` at
`l2` 10 diverges without producing a single NaN metric. Holdout log loss is 19.70, AUC is
0.377109 and accuracy is 0.4296 on a corpus where the baseline reaches 0.6755. All three are
real numbers. The largest weight is 4.867e+238. Those metrics are finite only because
`model.log_loss` clips probabilities to 1e-15 before taking the log, which it does for a
good reason and which also lets a diverged model report a loss.

So a finiteness rule reading `metrics` alone admits this artefact. That is asserted rather
than described, in `check_a_metrics_only_reading_would_have_admitted_it`.

The rule here catches it twice. Reading the whole payload catches it on `model.final_loss`,
which carries the `0.5 * l2 * (w @ w)` penalty and overflows to an infinity. The parameter
scale catches it independently. Two unrelated reasons to refuse one artefact is a good place
to be, and it is also why neither condition gets dropped on the evidence of this one config.

## Where the threshold comes from

`1 / (1 + exp(-z))` returns exactly 1.0 in float64 once `exp(-z)` falls below half an
epsilon. That happens at `log(2 / eps)`, which is 36.736801. Past it the sigmoid has
returned a bit rather than a probability and the loss is whatever the clip says.

The table measuring where that starts to bite is in
[design-notes](design-notes.md#admission-and-the-metric-that-was-finite-anyway) rather than
repeated here. The short version is that a weight scale of 100 already returns exactly 0 or
1 on 36.3 percent of the baseline holdout.

The statistic compared against it is the weight norm plus the absolute bias, which bounds
the logit at a one sigma standardised input. It understates what the model can reach, by
roughly the square root of the feature count, because a real input carries every feature at
once. Understating is the safe direction for a door that must not refuse a model somebody
trained deliberately.

The threshold is not a close call on anything here. Across the six shipped configs the five
that train score 0.20277 to 2.23082 and the one that diverges scores 6.038e+238. There is a
factor of sixteen below and two hundred and thirty seven orders above. Where the line sits
decides nothing in this repo, and that is the honest defence of it rather than the
derivation. A derived constant can still be the wrong constant.

## What this costs

Every caller that registered a fabricated hash moved to `create_version`. That is eight
checks in `tests/test_registry.py`, one in `tests/test_deploy.py`, five in
`scripts/registry_probe.py` and one in `scripts/drill.py`. All of them are asking the store
about itself and none of them had a model, so the move is honest rather than a workaround.
Four check names changed to say `create_version` instead of `register`, because a name
claiming to test the door while testing the primitive is worse than no name.

`create_version` is reachable, so the door can be walked around by calling it. That is true
of any layering and it is not defended here beyond a check that records the decision.

## What it does not catch

A model can train, stay finite, stay far from saturation and still be useless.
`configs/candidate-underfit.yml` is admissible and is meant to be. Deciding whether a model
is any good is the promotion gate's job and this rule deliberately does not encroach on it.
See `docs/limitations.md`.

A refused registration still leaves a tracking run behind, because `scripts/train.py` logs
the run before it registers. Verified on a sqlite store: three configs produced five runs
and one registered version. That is the intended split rather than a leak. The run happened
and a diverged run is worth having a record of.
