# ADR 0002: the config fingerprint was a function of the file, not of the run

Status: accepted. Supersedes nothing in adr-0001 and weakens one claim in it.

## Context

adr-0001 set up two identities. The config fingerprint identifies the recipe and the
artefact content hash identifies the bytes. It also claimed that two runs with the same
fingerprint produce the same model, which is true and is only half the property that was
needed.

Building the tracking layer forced the other half. MLflow stores a param as a string, so
recovering a config out of the store means casting every value back to its declared type,
and that raised the question of what the loader had been doing with types up to now. The
answer is nothing. It stored whatever `yaml.safe_load` handed it.

## What was measured

On this machine today, against `configs/baseline.yml` with one field rewritten.

```
                      noise: 1          noise: 1.0
config fingerprint    6299a382bfec      003ec18d7c82
artifact hash         a556e05066f1...   0284b85ef746...
model section         845 bytes         845 bytes, BYTE IDENTICAL
metrics section       byte identical
holdout_roc_auc       0.6719322619296195 on both sides
payload keys differing                  ['config_fingerprint']
```

`noise: 1` and `noise: 1.0` are the same training run. numpy does not care which one it
gets. The weights come out bit for bit identical and so do all ten metrics. What moved was
`json.dumps`. It writes `1` for an int and `1.0` for a float. So the fingerprint moved. The
fingerprint sits inside the artefact payload, so the artefact hash moved with it.

## The point

adr-0001 said the artefact hash is strictly finer than behaviour and defended that as
correct, because two different binaries should not share one deployment slot. That defence
assumed the extra fineness was information about the model.

Here it is information about the typography of a YAML file. The registry primary key and the
rollback target both moved while the model stayed byte identical. A check asking whether
production is running the approved artefact would have said no. Nothing about the model
differed.

So the fingerprint was not an identity for a recipe. It was a hash of one serialisation of
one file that happened to describe a recipe.

## Decision

The loader coerces every value to the type its dataclass declares, and refuses any cast
that would lose something.

1. `config.declared_types` resolves the real types with `get_type_hints`. This is not
   optional. `dataclasses.fields(cls)[i].type` is the **string** `"int"` in `mcr/config.py`,
   because of the `from __future__ import annotations` at the top of it. The obvious
   `f.type(value)` calls a string and raises.
2. `config.coerce` is the one rule, used by the loader and by
   `tracking.config_from_params`. An int field takes an int, an integral float or a string
   that parses to one. A float field takes any number or a string that parses. A str field
   takes only a str.
3. A lossy cast raises. `epochs: 5.5` is refused rather than truncated to 5. That matters
   more than the fix it accompanies, because trading a visible fingerprint problem for an
   invisible training one is a worse deal than leaving the bug alone.
4. `bool` is refused everywhere. `isinstance(True, int)` is `True` in Python, so `epochs:
   yes` reaches an int field unless something stops it, and YAML reads `yes` as a bool.

## What this did not fix

Coercion closes the int against float case and the string case. It does not make the
fingerprint invariant under every way of writing the same run, and I have not proved a
bound on what is left. A `name` differing only in case is a different fingerprint and
arguably a different run, so that one is fine. The honest statement is that the fingerprint
is now a function of the typed config rather than of the file, and that the typed config is
one canonical form among the ones YAML can express.

The two shipped configs were checked before and after. `configs/baseline.yml` fingerprints
to `70a46feb1141` and `configs/candidate-lr.yml` to `80a2bc7f7405`, both unchanged, so the
table in adr-0001 still reads correctly. That check ran first. A fix to a key derivation
that silently moved every published key would have been worse than the defect.

## Params against tags, decided by measurement rather than by convention

The same build needed to know where the recipe lives in MLflow. Measured on mlflow 3.16.0.

```
log_param on a key holding a different value    raises MlflowException
log_param on a key holding the same value       allowed
set_tag on an existing key                      overwrites silently
param value over 6,000 characters               TRUNCATED, warning on stderr, no exception
tag value over 8,000 characters                 TRUNCATED, warning on stderr, no exception
two runs of one config                          two run ids, one fingerprint tag
```

So the split comes free. The recipe goes in params, where MLflow itself refuses to let it be
rewritten after the run finished. The mutable stage the registry needs goes in tags, which
overwrite. Nothing keys on a run id, because two runs of one recipe produce two of them.

The truncation row is the one worth keeping. A value long enough to be shortened comes back
shortened and the call succeeds, so a 64 character content hash silently becoming 60
characters is a registry key pointing at nothing. Nothing here is near the limit and
`tracking._short_enough` refuses rather than resting on the headroom.

## What was rejected

**Dropping `config_fingerprint` from the artefact payload.** It would have made the artefact
hash stable under this defect without fixing the fingerprint, and adr-0001 put the
fingerprint in the payload so the artefact can say which recipe produced it. That is worth
keeping.

**Canonicalising at fingerprint time instead of at load time.** Rounding or normalising
inside `fingerprint()` would fix the hash and leave the config object holding an int where
a float was declared, so everything else reading the config would still see the wrong type.
The type is the thing that was wrong.
