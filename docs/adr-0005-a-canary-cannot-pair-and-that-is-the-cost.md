# ADR 0005: a canary cannot pair, and the bill for that comes due exactly where a canary is used

Status: accepted.

## Context

The promotion gate in `adr-0004` scores a candidate and an incumbent on one holdout, every row
scored twice, and compares them row by row. That pairing is what gives it a resolution of
about a fifth of one percent of the loss on five thousand rows.

A canary routes a slice of live traffic to the new model. Every request goes to one arm.
No request is ever scored by both, so the comparison stops being paired and becomes two
independent samples.

The first draft of this module treated that as a detail and reused the gate's interval on
two arm means. It is not a detail.

## Measurement one: what the split costs, and what it depends on

Reproduce with `python3 scripts/canary_probe.py`. Five thousand holdout rows, one
incumbent at four hundred epochs, candidates at the epoch counts below. Paired interval
against the unpaired interval at a five percent canary share, on the same rows.

```
  epochs      true diff       corr  paired width split 5% width     ratio
       1    +1.9263e-01   0.976561    2.9242e-02    3.6825e-02       1.3
       5    +8.1657e-02   0.987130    1.9754e-02    6.6804e-02       3.4
      20    +8.2012e-03   0.996622    6.5510e-03    1.2007e-01      18.3
      40    +1.1332e-03   0.999390    2.1588e-03    1.3692e-01      63.4
      45    +7.6109e-04   0.999611    1.6742e-03    1.3863e-01      82.8
     100    +2.8463e-05   0.999998    1.1736e-04    1.4376e-01    1225.0
```

The ratio is not a constant. It runs from 1.3 to 1225.

The correlation column is why. Two models that disagree everywhere produce row losses that
barely track each other, so pairing has little to exploit and throwing it away costs
little. Two models that are close produce row losses that track almost exactly, the paired
difference is tiny compared to the loss itself, and splitting discards all of it. On this
corpus the row loss has a standard deviation of 5.880232e-01 and the paired difference at
forty epochs has 3.894286e-02, fifteen times smaller.

The consequence is the part worth writing down. **A canary only ever sees models that are
close.** Anything obviously bad has already been stopped by the gate before a single
request moves. So the traffic split is cheapest on the models a canary will never be shown
and most expensive on every model it will.

## Measurement two: the split is not merely noisy, it is wrong about real differences

Two hundred draws of the split itself, at a five percent share, on the same fixed models.

```
  epochs      true diff   paired verdict                split at 5%
       1    +1.9263e-01         rollback          0 promote  200 rollback    0 hold
       5    +8.1657e-02         rollback          0 promote  200 rollback    0 hold
      20    +8.2012e-03         rollback          6 promote    6 rollback  188 hold
      40    +1.1332e-03         rollback         10 promote    3 rollback  187 hold
      45    +7.6109e-04             hold         10 promote    3 rollback  187 hold
     100    +2.8463e-05             hold         11 promote    3 rollback  186 hold
```

The first two rows are the control and they matter. The split catches an obviously bad
model on every single draw, so the comparison is not broken and the module is not rigged
against it.

The twenty epoch row is the finding. That model is worse by 8.2e-03, the gate rejects it
deterministically on every run, and the split promoted it on six draws out of two hundred
and could not separate it on a hundred and eighty eight.

The hundred epoch row is the other control. Those two models really are alike, and its
promote count is the nominal error rate of a two sided ninety five percent interval. So
the split comparison is behaving correctly as a statistical test. It is not miscalibrated.
It is underpowered, and the underpowering was chosen rather than forced.

## Measurement three: the traffic it would take

`required_rows` inverts the Welch standard error. With a canary share `f` the arms hold
`f*n` and `(1-f)*n` rows, so the sizing carries a `1/(f*(1-f))` term, which is 4 at an even
split and 21 at a twentieth.

To reach the paired half width of 1.0794e-03, in total requests:

```
  at 50.0% to the canary             4,559,983
  at 10.0% to the canary            12,666,620
  at  5.0% to the canary            23,999,912
  against 5,000 rows for the paired comparison.
```

Twenty four million requests against five thousand rows. That is the price of the routing
decision, and no amount of patience at a five percent share buys back what pairing had for
free, because the small canary arm is the binding term.

## Decision

Three parts, and the first two are separable on purpose.

**Routing decides what is served. It does not decide what is measured.** The router exists
to bound the blast radius. A model that is wrong should be wrong for five percent of
traffic rather than all of it, and that is worth having on its own.

**Score both models on every request whenever the metric allows it.** `observe` runs both
forward passes and returns the pair. The comparison is then paired again and gets the
gate's resolution back at the cost of one extra forward pass per request. `shadow_interval`
delegates to `gate.paired_interval` so there is one definition of the paired comparison in
this repo rather than two.

**The verdict still comes off the served traffic.** `decide` reads the split interval, not
the shadow one, and `check_decide_takes_its_verdict_from_the_split_and_not_the_shadow`
pins that by handing it an inverted pair and asserting the verdict does not move. The
shadow figures go in the report beside the split ones, and when the two disagree the
report says so in a line. That is the same shape as the gate's report carrying each run's
own recorded number next to the gate's.

## Which metrics can be shadowed and which cannot

This is the limit of the above and it is not a small one.

A metric is shadowable when it can be computed from the request and its label without
knowing which answer was served. Log loss is. So is anything scored against a label that
arrives independently of the served prediction.

A metric is not shadowable when its value depends on what was served. A click on the
recommendation that was shown. A conversion after a price the model chose. A downstream
cost incurred because the model said yes. For those the counterfactual does not exist, the
split is the only comparison available, and the bill in measurement three is real.

So this module ships both and names which is which, rather than shipping the paired one and
letting a reader assume it always applies.

## Four verdicts, not three

The four are `promote` and `rollback` and `hold` and `refuse`. Their exit codes run 0 to 3
in that order.

The gate's negative answer is `reject`, a fact about a candidate that was never serving.
A canary is already serving, so its negative answer is `rollback`, which means stop
routing traffic to the canary. It is not an instruction to move the stage, and the flag
that tried to make it one is in the rejected list below.

`hold` is the one that earns its place. Given the widths above, the honest answer on most
canary runs is that the traffic so far does not separate the arms. A pipeline that maps
that onto rollback tears down a canary that had not finished answering. One that maps it
onto promote ships on noise, and given a promote rate of about one draw in twenty at the
nominal level, it ships on noise regularly.

## The router is sticky, and that costs something too

Assignment is `sha256(salt|key)` against the fraction. Sticky, so a user does not see a
different model on consecutive requests. Deterministic across processes, which `hash()`
would not be, because Python salts that per process and a restart would reassign every key.

Monotone in the fraction, which is checked. Ramping from five percent to twenty five keeps
every key already in the slice, so the arms stay comparable across a ramp.

The cost is composition. A sticky slice is a fixed set of keys, so whatever imbalance it
has does not average out as the canary runs longer. Measured on the labels at a five
percent share:

```
  sticky slice   positive rate 0.2101 against 0.1838, gap +0.0263
  coin flip      gap over 200 redraws: mean +0.0004, spread 0.0264
```

The sticky slice is not more imbalanced than a coin flip. Its gap is about one standard
deviation of the coin flip's distribution. The difference is that the coin flip's gap
averages away and the sticky one does not, so time does not fix it. Only a change to the
key population does. `slice_imbalance` reports it and the CLI prints it under every run.

## What was rejected

**Reusing the gate's paired interval on two arm means.** The first draft. It reports an
interval about ten to a thousand times tighter than the data supports, and every number in
it looks reasonable.

**A pooled variance.** The arms are different sizes by design and there is no reason to
assume their spreads match. Welch costs nothing and drops the assumption.

**A bootstrap.** Same reason `adr-0004` dropped one. It needs a resample count and a seed,
neither of which is a number anybody chose, and on this project that produced a verdict
that moved with a default argument.

**Ramping the fraction until the comparison separates.** This is the obvious escape from
measurement three and it is testing until the answer is significant. The interval is
already two sided at ninety five percent and looking repeatedly at it is not accounted for
anywhere here.

**Refusing to canary at all and relying on the gate.** Defensible on the numbers above and
wrong on purpose. The gate measures on a generated holdout. A canary measures on traffic.
Those answer different questions and the right response to a weak measurement is to say it
is weak, not to delete it.

**A `--rollback` flag on the canary script.** This one was written and run end to end and
then removed. It is worth recording because it looked obviously right.

The gate has `--promote`, so the canary appeared to want a `--rollback` for symmetry. It
called `registry.rollback`, which moves the stage back to the previous version. Run
against a bad canary it did exactly that and reported success.

It is the wrong action. The canary is a version on trial. The control arm is whatever
holds the stage. So a rollback verdict is a statement about the canary and the flag
responded by moving production off a model the verdict says nothing about, onto whatever
came before it. The one component known to be fine was the only one it touched.

What a rollback verdict means is stop routing traffic to the canary. The canary's share is
a command line flag rather than registry state, so there is nothing in the registry to
undo and the exit code is the whole action. The script prints that in a line and
`check_the_cli_has_no_rollback_flag` pins the absence, by AST rather than by substring,
because the first version of that check failed on the comment explaining the absence.

## What is not solved

**The traffic is a replay of a generated holdout.** There is no serving path in this
project and no users. The arithmetic of splitting a sample is real and reproduces. The loss
values are a fact about a generator, stated at the point of use in the report, in the probe
and in the CLI docstring rather than only here.

**One request per key.** The replay gives every row its own key, so stickiness has nothing
to grip in the shipped run and is exercised by fixtures that build their own repeats. A
real stream has heavy repeat keys, and with repeats the effective sample size of an arm is
the number of keys rather than the number of requests, which makes measurement three worse
rather than better.

**No sequential test.** The canary is scored once over a whole replay. A real one is read
continuously, which is a different statistical problem and is not attempted here.

**No automatic ramp.** `--fraction` is a flag a human sets. A ramp schedule would need the
sequential test above to be safe.

**The canary's traffic share is not registry state.** It is a flag, so nothing persists
about which version is being canaried or at what fraction, and nothing can be undone. The
`canary.*` tags on the run are a record after the fact rather than state anybody could act
on. Making the share into a stage the registry holds is the change that would give a
rollback verdict something to do, and it is not made here.

**Nothing drains traffic or restarts a process.** There is no serving process, so the whole
operational half of a rollback is absent rather than implemented badly. It is named here
so the next document that needs it can find it.
