# ADR 0010: recovery and authorisation are two questions, and the order they run in matters

## Context

`registry.settle` reads the transition log and drops any entry that nothing can witness.
`adr-0006` explains why. A promote writes its log entry before it moves the alias, so a
crash between the two leaves an entry for a move that never happened, and a rollback
reading the raw log deploys a version that never served a request.

The witness rule works because a completed promote leaves the alias on its `to_version`,
and the next entry written reads that value as its own `from_version`. Every entry but the
last is witnessed by the one after it. The last is witnessed by the alias.

That rule recovers a crash. It was also doing something it was never able to do, which is
vouch for who wrote the log.

The transition log is a set of tags on a registered model. MLflow has no conditional write
and no per tag owner, so anything holding store credentials can add a tag or overwrite one.
A forger picks `from_version` and can therefore satisfy the witness rule on purpose. The
`settle` docstring said as much and the README named it as a limitation, which is honest
and is not a defence.

## The measurement that decided the shape

Three versions promoted in order, so production is on 3 and the rollback target is 2. Then
slot `0001` is overwritten with an entry claiming a move to version 9, which was never
promoted. That breaks the witness rule for the genuine entry it replaced.

```
ok     forged log entry refused               damaged: refused                        control: 2
ok     recovery first answers instead of refusing 1 (wanted 1)
ok     because settle dropped the evidence first 0 rejected after settle (wanted 0 rejected after settle)
```

The middle row is the result worth having. Run recovery first and the forged entry is
dropped as an unwitnessed crash artefact, which also drops the genuine entry beside it. What
is left reads as a healthy two entry log, nothing refuses, and the rollback target has moved
from 2 to 1. The third row is why. By the time an integrity check runs it has nothing to
object to, because the recovery pass already discarded the only evidence of tampering.

So the two filters are not commutative and the integrity check has to run first. This is the
same shape `adr-0006` found in the write ordering. A failure being visible is not the same as
a failure being safe, and the question is what the visible failure lets the next operation do.

## Decision

Two functions, two inputs, composed in one order.

`settle` is unchanged and keeps taking the log and the live alias. Both of those are already
in the store, which is exactly why it cannot answer a question about a secret.

`authentic` is new and answers who wrote an entry. `mcr/attest.py` carries a MAC over the
entry body and the slot it sits in. `rollback_target` runs `authentic` and then `settle`, and
refuses outright when anything fails the first one.

An unconfirmed entry is dropped and the walk continues, because a crash leaves a log that is
still the store's own account of itself. A rejected entry is a refusal instead. It says
something else has been writing to the history, which is a fact about the store rather than
about one move, so every remaining entry is a question and computing a deploy from the
remainder would be answering anyway.

The slot is signed material. Without it a genuine entry can be lifted into a different
sequence key and still verify, and the log is ordered by that key, so moving one entry
backwards changes which version the walk reaches without altering a byte.

## The unsigned entry, which is the part the first rule got wrong

The first version of `authentic` accepted an unsigned entry unconditionally, on the grounds
that every log written before this existed is unsigned and refusing them all would refuse
every store this code has been pointed at.

That rule defends against nothing. An attacker who can overwrite a tag can leave the
signature off, and that is cheaper than faking one.

```
ok     unsigned entry in a signed log refused damaged: refused                        control: 2
ok     and the wrong order still answers      1 (wanted 1)
```

So the log states its own policy. A log holding at least one entry that verified is a log
whose writer holds the key, and an unsigned entry sitting in it is either a downgrade or a
second writer. Both refuse. A log holding no verified entry at all is read as the legacy it
almost certainly is.

```
ok     an unsigned log answers normally       2 (wanted 2)
ok     and it really was unsigned             unattested (wanted unattested)
```

Nothing has to be configured for that and there is no flag to set wrongly, which matters,
because a flag like this gets set once and read by nobody afterwards. `require_attested`
only tightens it, for a caller that will not read a legacy log at all.

Three readings rather than two, for the same reason the drills take three. `attested` and
`forged` and `unattested` are distinct, and `unattested` is not a weaker `forged`. It means
the question was not asked, either because the entry predates the signature or because no
key was supplied. A verifier handed a signed entry and no key returns `unattested`, because
returning `attested` would accept a signature on its own word and returning `forged` would
be a guess.

## What this is worth, stated narrowly

It authenticates the holder of a key. It is not a principal, there is no rotation, and
anybody with the key can write whatever they like and this accepts all of it. The key comes
from an environment variable, which means the threat it covers is a second writer on the
store rather than an attacker on the machine running the deploy.

A store whose log is entirely unsigned gets no protection at all, by design, and that is
every store this code has touched so far.

## What was rejected

**Verifying inside `settle`.** This was the first plan and the obvious one. It fuses the two
questions back together and it makes a pure function over a log take a key. The measurement
above is the argument against it. A single fused function has one output, so the composition
order stops being a thing anybody can see, and the order is where the result turned out to
be.

**Refusing every unsigned entry by default.** Correct and unusable. It refuses every existing
history, so it would be switched off and stay off.

**Accepting every unsigned entry.** What the first draft did. It makes the signature
decorative, because omitting it is easier than forging it.

**A confirmation record written after the alias moves.** Rejected in `adr-0006` for crash
recovery and rejected again here. It is a third write with a third crash window and it does
not answer a question about authorship at all.

**Storing the body as nested JSON inside the envelope.** Verifying would mean re-serialising
it, and two serialisers disagreeing on key order or spacing would read as a forgery. The
signed bytes are stored as the exact string that was signed.

## What is not solved

**A log with no signed entry is undefended, and that is the sharp one.** The no downgrade
rule needs an entry it verified before it can call an unsigned one a downgrade. A model the
real pipeline has never promoted offers no such entry, so a log written end to end by
somebody else reads as legacy and is accepted.

```
ok     a log nobody signed answers anyway     2 (wanted 2)
ok     and only require_attested refuses it   3 of 3 rejected (wanted 3 of 3 rejected)
```

`require_attested` is the only setting that covers it, which is why it exists. It cannot be
the default, because a store full of genuine legacy logs is indistinguishable from an
attacked one by construction. Anchoring on a marker recorded against the registered model
would move the hole rather than close it, since that marker is another tag anything can
delete. Closing it needs a store with conditional writes.

**A writer holding the key is unconstrained.** That is the definition of what a MAC buys and
it is the reason this says authorisation rather than authorisation and audit.

**There is no key rotation and no key identity.** One key, one environment variable. A second
deploy pipeline would need the same key, and then neither entry can be attributed to either
pipeline.

**Nothing protects the alias.** The alias is still a plain registry alias and anything with
store access can move it. `rollback_target` already refuses when the log disagrees with the
alias, which turns that into a visible failure rather than a defended one.

**A deleted entry is invisible.** The MAC covers the slot, so an entry cannot be moved. It
does not cover the set of slots, so deleting the most recent entry leaves a shorter log that
verifies cleanly. The witness rule then compares the new last entry against the alias and
refuses. That catches it by accident rather than by design.
