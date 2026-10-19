# Limitations

Moved out of the README so the front page stays short. Back to the [README](../README.md).

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

**Admission asks whether a model trained, not whether it is worth anything.** The rule in
`mcr/admission.py` refuses a non finite artefact and one whose parameters have saturated the
sigmoid. `configs/candidate-underfit.yml` passes it, scores 0.6419 holdout log loss against
the baseline's 0.4492, and is admitted on purpose. Judging quality is the promotion gate's
job. A rule doing both would be two rules sharing a name and the weaker one would win
arguments it should lose.

**The saturation threshold is derived and still a judgement at the margin.** `log(2 / eps)`
is where a float64 sigmoid stops returning anything below 1, which is a fact. That this
makes a good admission boundary is an argument. Nothing shipped here lands near it, with the
five training configs between 0.20277 and 2.23082 and the diverged one at 6.038e+238, so the
placement is untested in the range where it would matter. A model legitimately needing a
logit scale above 36 would be refused and the message would not help much.

**`create_version` is reachable and unguarded.** The store primitive `register` is built on
still exists, because the store's own behaviour has to be testable without training a model
first. Anything calling it goes round the admission rule. The checks record that this is a
decision rather than an oversight and nothing enforces it.

**A refused registration still leaves a tracking run behind.** `scripts/train.py` logs the
run before it registers, so a diverged config produces a tracking record and no registered
version. Verified at three configs to five runs and one version. That split is intended, and
it does mean the tracking store and the registry disagree about how many models exist.

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

**The facts the probes decide on are graded now, and the probes themselves are not.** The
comparisons used to live in the scripts, where nothing in `tests/` could import them. They
are in `mcr/roundtrip.py` and `mcr/canary.py`, which score 9 of 9 and 122 of 125 under a
mutation pass. What remains ungraded is the driving and the printing, which is the right
half to leave. `scripts/repro_probe.py` has not been moved and its arithmetic is still in
the script. `scripts/registry_probe.py` is that shape and answers it differently: every arm about this repo's code runs a control against a stand in carrying the
defect the arm exists to catch, and the probe fails if a control passes. That caught a bad
arm on the first run, where the naive rollback guess and the right answer happened to agree.

**`main` in `scripts/ci_local.py` is graded past its plan branch by nothing.** Seven
mutants survive in it, all in the arithmetic that counts what ran and turns that into an
exit code. Reaching them means executing the real job, which trains two models, so the fast
suite cannot carry it. The plan branch is driven and the executor underneath it scores 25 of
34. The gap is the wrapper, and it is the thinnest code in the file.

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
