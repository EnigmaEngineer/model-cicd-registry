# ADR 0003: the stage is an alias, and the registry cannot remember

Status: accepted. Reverses a decision recorded in adr-0002.

## Context

The registry has to answer three questions. Which model is in production. Which one was
there before it. Which model is this reference talking about.

MLflow has a model registry with stages built in, so the first question looked answered.
adr-0002 had already decided where a stage should live, on the evidence that a param
refuses to be overwritten and a tag overwrites silently. The conclusion there was that the
mutable stage belongs in a tag.

That conclusion was about run tags in the tracking store, and it does not survive contact
with the registry. It is reversed below, on a measurement.

## What was measured

mlflow 3.16.0, sqlite backend, on this machine today. Everything here is reproducible with
`python3 scripts/registry_probe.py`.

```
stage transition, nothing passed that it does not require   versions in Production: [1, 2]
the same call with archive_existing_versions=True           versions in Production: [1]
two versions both tagged stage=production                   allowed, search returns [1, 2]
alias moved from v1 to v2                                   v1 aliases [], v2 aliases ['production']
v1 after losing the alias                                   last_updated_timestamp == creation_timestamp
```

Three things fall out of that.

`transition_model_version_stage` emits a FutureWarning saying stages will be removed, and
it is deprecated as of 2.9.0. Building on it means building on something with a removal
notice attached.

Its default leaves two versions in Production at the same time. You get the property the
word "production" implies only by remembering `archive_existing_versions=True`. A deploy
reading "the production model" out of a store holding two of them gets whichever row the
search returns first, and `search_model_versions` does not sort. It came back `[3, 1]` and
`[2, 1]` in the same session.

A version tag has the same hole. Two versions can carry `stage=production` at once and
nothing objects, so the adr-0002 plan would have shipped the same defect by a different
route. The measurement that decided adr-0002 was about which store refuses a rewrite. The
question here is a different one, which is whether the store enforces that a stage points
at one thing.

An alias does. Pointing `production` at a new version takes it off the old one in the same
call. Exclusivity comes free and is the store's own behaviour rather than a rule this code
has to remember.

## Decision

The stage is an alias. `STAGES` is `("staging", "production")`, lowercase and with no null
member. A version that no alias points at is already in no stage and does not need a name
for it.

`transition_model_version_stage` is not used anywhere in this project.

## The cost, which is most of this module

An alias move leaves no trace. After `production` moves from version 1 to version 3,
version 1 does not record that it ever held it. Its `last_updated_timestamp` does not change
either, so the row was not touched at all. `get_registered_model(name).aliases` returns the
current mapping and nothing else. There is no history API on the client.

So the registry can say what is in production and it cannot say what was. A rollback is a
question about what was.

The transition log is therefore written here, as one tag per transition on the registered
model, keyed `transition.<stage>.<seq>` with the sequence zero padded to four digits. One
key per entry, so nothing overwrites anything, and the ordering is a lexical sort over the
keys. Two consequences are worth stating rather than discovering later.

The width is a compatibility surface. A build writing four digits and a build reading five
would sort one store's entries into the other's wrongly, and within a single store any
consistent width passes every check. `check_the_log_key_is_padded_to_a_fixed_width` pins it
for that reason and for no other.

Ten thousand transitions per stage is the ceiling. Past that the key gains a digit and the
sort breaks. That is not a limit worth engineering around and it is worth writing down.

## Rollback is not "whatever it held before"

That was the first rule written here and a check caught it.

Promote version 1. Then 2. Then 3. Now roll back. Production sits on 2 and the version it
held immediately before 2 really was 3. So a second rollback returns to 3, which is the
model that was just rolled away from. Rollback would oscillate between the last two versions and
could never reach version 1, and it would report success every time.

So a version that a rollback moved away from is abandoned and the walk skips it. That is
why the direction of a move is stored on the entry as `kind` rather than worked out from
the version numbers. A deliberate redeploy of an older model has exactly the same shape in
the log as a rollback, and inferring the difference means guessing.

## An ambiguous reference is refused

MLflow rejects an alias containing a slash and accepts an alias that is a decimal number.
So `set_registered_model_alias(name, "2", "4")` is allowed while version 2 exists, and the
string `"2"` then names two different models trained from two different runs.

```
get_model_version(name, "2")             run 6b8c4048
get_model_version_by_alias(name, "2")    run 6fd44793
```

Picking one is the failure, because which one you get depends on the function the caller
reached for and the caller has no way to know that is even a question. `resolve` refuses.

## What is not solved

**Two writers.** `promote` chooses a sequence number from a read, and there is no compare
and set anywhere in this API. It reads the key back and refuses if it does not hold what it
wrote, which detects a lost update rather than preventing one. Single writer here, so it
has never fired.

**Two calls, no transaction.** The log entry and the alias move are separate calls. The log
is written first, so a crash between them leaves a recorded transition that did not happen
rather than a silent one that did. A recorded lie is visible to the disagreement check. A
silent move is the thing rollback cannot survive.

**Anything that moves the alias outside `promote`.** The MLflow UI will do it, and so will
one line of script. The alias stays correct and this module's account of how it got there
stops being true. `rollback_target` compares the two and refuses rather than deploying some
old version and reporting success.

## What was rejected

**Using MLflow's stages anyway, with `archive_existing_versions=True` everywhere.** It
would work today. It is deprecated with a removal notice, it needs a keyword argument
remembered at every call site to be correct, and it still has no history.

**Deriving the rollback target from the store instead of a log.** The obvious derivation is
the highest version that is not current, and it is wrong the moment a candidate is
registered and rejected without ever being promoted, which is the normal case once the
promotion gate exists. Measured in the probe: the log says 1 and that guess says 4.

**Putting the transition log in a table of its own.** In a real system it belongs in one,
and a reviewer is right to say so. The registry is the only store this project has, and
standing up a second database to hold four fields per row would be more machinery than the
thing it records. The tag approach has two costs and both are named above, which is the
trade being made rather than a claim that there is no trade.

**Keeping both stages and aliases in step.** They are two independent stores. Measured: v1
was Production by stage while the `prod` alias pointed at v3, and nothing objected. Two
sources of truth for one question is worse than the weaker of the two.
