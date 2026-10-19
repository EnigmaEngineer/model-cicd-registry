"""What a recipe has to survive to come back out of a store.

The tracking probe asks whether a run written into MLflow can be read back and retrained to
the same bytes. Every arm of that question used to be an expression inside the probe script,
and nothing in the suite imports a script, so none of it could be falsified by a mutant. The
arms graded the store and nothing graded the arms.

These are the facts the probe decides on. They take dictionaries of strings and floats rather
than a config or a client, so this module holds no dependency on MLflow and the fast suite
can grade it. That split is the point. A store is a thing you need a store to test. Whether
two metric dictionaries agree is not.

`bend` and `without` build the probe's controls. A control that the probe constructs inline
is a control nothing checks, and both of these have a rule in them worth pinning: `without`
has to return a dictionary missing exactly one key, and `bend` has to return one whose value
really moved.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple


class RoundTripError(RuntimeError):
    """Raised when a round trip cannot be judged at all."""


def value_types(params: Mapping[str, str]) -> List[str]:
    """The distinct Python type names of a param mapping's values, sorted.

    MLflow hands every value back as a string, so the rebuild only ever reaches the string
    branch of the config loader's coercion. If a value ever arrived already typed, the
    rebuild would take a different branch and the fingerprint would start depending on
    which one. The probe asserts this comes back as exactly `["str"]`, and that assertion is
    worth nothing unless something grades the function producing it.
    """
    if not params:
        raise RoundTripError("cannot report the value types of an empty mapping")
    return sorted({type(v).__name__ for v in params.values()})


def inexact_metrics(
    sent: Mapping[str, float], got: Mapping[str, float]
) -> Dict[str, Tuple[float, float]]:
    """Metrics that did not come back out of the store exactly as they went in.

    Returns the ones that moved, each with what was sent and what came back. Exact rather
    than close on purpose. The two configs this repo ships differ on holdout log loss by
    1.806e-11, so a store that rounded on the way in would leave the promotion gate
    comparing two numbers it thinks are equal.

    A metric that is missing on the way back counts as inexact rather than as absent,
    because the caller is asking whether the round trip held and a dropped key is the
    loudest way for it not to have.
    """
    if not sent:
        raise RoundTripError("cannot judge a round trip over no metrics")
    out = {}
    for key, value in sent.items():
        if key not in got:
            out[key] = (float(value), float("nan"))
        elif got[key] != value:
            out[key] = (float(value), float(got[key]))
    return out


def without(params: Mapping[str, str], key: str) -> Dict[str, str]:
    """A copy of the params with one key removed, or a refusal.

    The probe's first control. A store that dropped a param would pass every round trip arm
    as long as the rebuild filled the gap from a default, so the control has to hand the
    rebuild a mapping that really is short by one. Refusing an absent key is what stops the
    control from passing on a mapping it never changed.
    """
    if key not in params:
        raise RoundTripError("{} is not in the params, so removing it proves nothing".format(key))
    out = dict(params)
    del out[key]
    return out


def bend(params: Mapping[str, str], key: str, value: str) -> Dict[str, str]:
    """A copy of the params with one value moved, or a refusal.

    The probe's second control. The params have to carry the numbers rather than the shape,
    so moving one value must change the rebuilt fingerprint. Handing back a mapping whose
    value did not move would make that control pass against a fingerprint that ignores the
    field entirely.
    """
    if key not in params:
        raise RoundTripError("{} is not in the params, so bending it proves nothing".format(key))
    if params[key] == value:
        raise RoundTripError(
            "{} already holds {}, so this control would compare a value to itself".format(
                key, value
            )
        )
    out = dict(params)
    out[key] = value
    return out


def bumped(value: str) -> str:
    """The next integer up, as a string, for bending an integer valued param.

    Here rather than in the probe because `str(int(v) + 1)` is three operations and a
    mutant can break any of them. A reader who sees this named is also told what the control
    is doing, which an inline expression does not say.
    """
    try:
        return str(int(value) + 1)
    except (TypeError, ValueError) as exc:
        raise RoundTripError("{!r} is not an integer valued param".format(value)) from exc
