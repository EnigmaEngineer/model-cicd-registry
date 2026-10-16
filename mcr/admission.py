"""Whether an artefact is admissible to the registry at all.

This is a different question from the one the promotion gate asks, and keeping them apart
is the point of the module. The gate asks whether a candidate beats an incumbent. This asks
whether the candidate is a model. A config that overshoots produces a weight vector of
NaNs, and every step after that works exactly as written: the fit returns, the payload
serialises, the content hash is computed over bytes containing a NaN, and the store hands
back a version number. Nothing on that path ever asked whether the numbers mean anything.

Two conditions, and the second exists because the first is not enough.

Finiteness is the easy one. Every float in the payload has to be finite.

Saturation is the second one, and the case that motivates it is worth stating carefully. At
`l2` 10 on the shipped baseline data every *metric* comes back finite. Holdout log loss is
19.70, AUC is 0.377109 and accuracy is 0.4296, which is worse than a coin on a corpus where
the baseline reaches 0.6755. The largest weight is 4.867e+238. Those metrics are finite only
because `model.log_loss` clips probabilities to 1e-15 before taking the log, so the clip,
which is there for a good reason, is also what lets a diverged model report a number. A
finiteness rule reading only the metrics admits this artefact.

The rule here reads the whole payload, which catches it on `model.final_loss`. That carries
the training objective including the `0.5 * l2 * (w @ w)` penalty, and squaring 4.867e+238
overflows, so the field is an infinity. The saturation rule below then catches it a second
time and independently. Two unrelated reasons to refuse the same artefact is a good place to
be, and it is also why neither rule can be dropped on the evidence of this one config.

What separates the two cases is the scale of the parameters, and the threshold for it is
derived rather than chosen. `1 / (1 + exp(-z))` rounds to exactly 1.0 in float64 once
`exp(-z)` falls below half an epsilon, which happens at `z` of about 36.74. Past that the
sigmoid returns a bit, every row gets probability 0 or 1, and the loss is whatever the clip
says it is. Measured on the shipped baseline holdout: at a weight scale of 100 the model
returns exactly 0 or 1 on 36.3 percent of rows, at 1e4 on 96.9 percent, and at 1e10 on all
of them with two distinct probabilities across 5,000 rows.

The threshold is not a close call on anything this repo ships. Measured across the six
configs in `configs/`, the five that train score a logit scale of 0.20277 to 2.23082 against
a threshold of 36.736801, so the nearest one has a factor of sixteen in hand. The one that
diverges scores 6.038e+238, which is two hundred and thirty seven orders the other way.
Where the threshold sits decides nothing here, and that is the honest defence of it. A
derived constant can still be the wrong constant. What makes this one safe is the gap either
side rather than the derivation, and a config landing anywhere near 36 would need a look.

The verdict is returned and never raised. Deciding is this module's job and refusing is the
caller's, so the gate and the canary can still be handed a non finite candidate on purpose.
They have their own refusals for it and those refusals are tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

# The logit at which float64 stops being able to represent a probability below 1. Derived,
# not picked: `exp(-z) < eps / 2` is where `1 + exp(-z)` rounds to 1 and the division
# returns exactly 1.0. Comes out at 36.736801.
SATURATION_LOGIT = float(np.log(2.0 / np.finfo(np.float64).eps))

NON_FINITE_METRIC = "non_finite_metric"
NON_FINITE_PARAMETER = "non_finite_parameter"
NON_FINITE_FIELD = "non_finite_field"
SATURATED_PARAMETERS = "saturated_parameters"


@dataclass(frozen=True)
class Admission:
    """The verdict. `ok` is the answer and `reasons` is why, in stable tokens.

    `reasons` is sorted and deduplicated so two payloads failing the same way compare
    equal. `detail` is for a human reading a refusal message and its wording is not
    something anything should branch on.
    """

    ok: bool
    reasons: Tuple[str, ...]
    detail: Tuple[str, ...]
    logit_scale: float

    def why(self) -> str:
        if self.ok:
            return "admissible"
        return "; ".join(self.detail)


def _floats(value: Any, path: str) -> Iterator[Tuple[str, float]]:
    """Every float leaf in the payload, with a dotted path to it.

    Walked rather than listed. A field list would have to be edited every time the payload
    grows a number, and the edit that gets forgotten is the one that matters. Integers are
    skipped because no integer is NaN or infinite, and `bool` is checked before `int`
    because in Python a bool is one.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, (float, np.floating)):
        yield path, float(value)
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, dict):
        for key in sorted(value):
            child = "{}.{}".format(path, key) if path else str(key)
            yield from _floats(value[key], child)
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        for i, item in enumerate(value):
            yield from _floats(item, "{}[{}]".format(path, i))


def parameter_norm(weights: Any) -> float:
    """Euclidean norm that does not overflow on the input it exists to measure.

    `np.linalg.norm` on the diverged weight vector returns `inf`, because squaring 4.867e+238
    does not fit in a float64. The verdict would still be a refusal, by the finiteness rule
    and for the wrong reason, and a number that overflows on the way to a comparison is not
    a number worth comparing. Factoring the largest element out first keeps the sum of
    squares below the feature count.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        return 0.0
    biggest = float(np.max(np.abs(w)))
    # Both cases are already the answer. An all zero vector has norm zero, and a vector
    # carrying a NaN or an infinity has that as its norm and is refused on finiteness.
    if biggest == 0.0 or not math.isfinite(biggest):
        return biggest
    scaled = w / biggest
    return biggest * math.sqrt(float(scaled @ scaled))


def logit_scale(payload: Dict[str, Any]) -> float:
    """An upper bound on the logit this model produces at a one sigma standardised input.

    `predict_proba` standardises with the mean and scale in the payload, so a feature
    arriving at one standard deviation enters the dot product at magnitude 1 and the bound
    is the weight norm plus the bias. That understates what the model can actually reach,
    because a real input has all the features at once and its norm goes as the square root
    of the feature count. Understating is the safe direction for a gate that must not refuse
    a model somebody trained on purpose.
    """
    model = payload.get("model")
    if not isinstance(model, dict):
        return 0.0
    norm = parameter_norm(model.get("weights", []))
    bias = model.get("bias", 0.0)
    try:
        bias = abs(float(bias))
    except (TypeError, ValueError):
        bias = 0.0
    return norm + bias


def inspect(payload: Dict[str, Any]) -> Admission:
    """Decide whether this payload is a model. Returns a verdict, never raises."""
    reasons: List[str] = []
    detail: List[str] = []

    offenders = [(p, v) for p, v in _floats(payload, "") if not math.isfinite(v)]
    for path, value in offenders:
        if path.startswith("metrics."):
            reason = NON_FINITE_METRIC
        elif path.startswith("model."):
            reason = NON_FINITE_PARAMETER
        else:
            reason = NON_FINITE_FIELD
        reasons.append(reason)
        detail.append("{} is {}".format(path, value))

    scale = logit_scale(payload)

    # Only meaningful when it is a number. A non finite scale means the weights are already
    # refused above and saying so twice tells a reader nothing.
    if math.isfinite(scale) and scale > SATURATION_LOGIT:
        reasons.append(SATURATED_PARAMETERS)
        detail.append(
            "logit scale {:.6g} exceeds {:.6g}, so the model returns 0 or 1 and its "
            "loss is the clip rather than a fit".format(scale, SATURATION_LOGIT)
        )

    # Capped because a diverged twelve feature vector produces twelve identical lines and a
    # refusal nobody reads to the end is a refusal nobody acts on. The count stays honest.
    shown = detail[:4]
    if len(detail) > len(shown):
        shown.append("and {} more".format(len(detail) - len(shown)))

    return Admission(
        ok=not reasons,
        reasons=tuple(sorted(set(reasons))),
        detail=tuple(shown),
        logit_scale=scale,
    )
