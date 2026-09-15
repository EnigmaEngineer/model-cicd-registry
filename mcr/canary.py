"""Canary routing, and the comparison you can actually make once traffic is split.

A canary sends a slice of traffic to a new model and watches what happens. The routing
half is easy and this module keeps it small. The measurement half is where the project
was about to lose something, so most of what follows is about that.

The promotion gate scores both models on one holdout. Every row is scored twice, so the
comparison is paired and the interval is on the mean of a per row difference. A canary
cannot do that by default. Each request goes to one arm, so no request is ever scored by
both, and the comparison becomes two independent samples.

That is not a small change. On this corpus the two arms' row losses correlate at 0.999390,
so the paired difference has a standard deviation about fifteen times smaller than the
row loss itself. Splitting the traffic throws all of that away.

`split_interval` is the honest unpaired comparison. `shadow_interval` is the paired one,
available whenever the metric can be computed without knowing which answer was served.
`required_rows` says what the first costs in traffic. The numbers are in docs/adr-0005.

Every number this module produces here is measured on a replay of a generated holdout,
not on production traffic, because this project has no production traffic. That is stated
again wherever a figure is printed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import gate

CANARY = "canary"
CONTROL = "control"

# The verdicts. Deliberately not the gate's three.
#
# The gate answers a question about a candidate and returns promote, reject or refuse. A
# canary is already serving traffic, so its negative answer is an action on production
# rather than a fact about a model, and `hold` is a fourth thing again: keep the canary up
# and keep collecting, which is neither a promotion nor a teardown. A pipeline that maps
# hold onto reject tears down a canary that had not finished answering.
PROMOTE = "promote"
ROLLBACK = "rollback"
HOLD = "hold"
REFUSE = "refuse"

# Two sided, matching the gate. A canary that used a different confidence from the gate
# that let the candidate out would be two thresholds for one decision.
CONFIDENCE = gate.CONFIDENCE

METRIC = gate.METRIC


class CanaryError(RuntimeError):
    """Raised when the canary cannot be handed what it needs to measure anything."""


@dataclass(frozen=True)
class Router:
    """Sticky assignment of a request key to an arm.

    Sticky rather than a per request coin flip, because a user who sees the new model on
    one request and the old one on the next has been given an inconsistent product. That
    is the normal reason to hash, and it has a consequence nobody states: the slice is a
    fixed set of keys, so its composition does not average out as the canary runs longer.
    A coin flip's does. `slice_imbalance` measures this and the probe shows both.

    The salt is part of the assignment. Two canaries running at once with the same salt
    route the same keys to the treatment arm, which makes their effects inseparable.
    """

    fraction: float
    salt: str = "canary"

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction < 1.0:
            raise CanaryError(
                "fraction must be strictly between 0 and 1, got {}".format(self.fraction)
            )
        if not self.salt:
            raise CanaryError("salt must not be empty")

    def _position(self, key: str) -> float:
        """Where this key falls in [0, 1).

        sha256 over the salt and the key. The first eight bytes are plenty and the
        division is exact, because 2**64 is a float and the numerator is an integer below
        it. A cheaper hash would be fine here and `hash()` would not, because Python
        salts it per process, so the same key would move arms on a restart.
        """
        digest = hashlib.sha256("{}|{}".format(self.salt, key).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def arm(self, key: str) -> str:
        return CANARY if self._position(key) < self.fraction else CONTROL

    def assign(self, keys: Sequence[str]) -> np.ndarray:
        """Boolean mask, True where the key routes to the canary arm."""
        return np.array([self.arm(k) == CANARY for k in keys], dtype=bool)


def replay_keys(n: int, prefix: str = "req") -> List[str]:
    """Request keys for a replay of n rows.

    One key per row. A real stream has repeat keys and this does not, which is why
    `slice_imbalance` takes the keys rather than assuming them, and why the stickiness
    check in the probe builds its own repeats instead of using these.
    """
    return ["{}-{:06d}".format(prefix, i) for i in range(n)]


@dataclass(frozen=True)
class ArmObservations:
    """What one arm saw and what it scored on it."""

    name: str
    model_name: str
    artifact_hash: str
    row_losses: np.ndarray
    labels: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.row_losses))

    @property
    def mean_loss(self) -> float:
        return float(np.mean(self.row_losses)) if self.n else float("nan")

    @property
    def positive_rate(self) -> float:
        return float(np.mean(self.labels)) if self.n else float("nan")

    def finite(self) -> bool:
        return bool(self.n > 0 and np.isfinite(self.row_losses).all())


def split_interval(
    canary: np.ndarray, control: np.ndarray, confidence: float = CONFIDENCE
) -> Tuple[float, float, float]:
    """Unpaired interval on the difference of two arm means. Welch, closed form.

    Returns the mean difference and the two ends. Positive means the canary is worse,
    matching the gate's sign convention on a metric where lower is better.

    Welch rather than a pooled variance because the arms are different sizes by design
    and there is no reason to assume their spreads match. Closed form for the same reason
    the gate's interval is: a resampler needs a seed nobody chose, and on this project
    that produced a verdict that moved with a default argument.
    """
    if len(canary) < 2 or len(control) < 2:
        raise CanaryError(
            "cannot build an interval on arms of {} and {} rows".format(
                len(canary), len(control)
            )
        )
    z = gate._z_for(confidence)
    mean = float(np.mean(canary)) - float(np.mean(control))
    se = math.sqrt(
        float(np.var(canary, ddof=1)) / len(canary)
        + float(np.var(control, ddof=1)) / len(control)
    )
    return mean, mean - z * se, mean + z * se


def shadow_interval(
    canary_losses: np.ndarray, control_losses: np.ndarray, confidence: float = CONFIDENCE
) -> Tuple[float, float, float]:
    """Paired interval, for when both models scored every request.

    This is the gate's interval and it delegates to it rather than reimplementing it, so
    there is one definition of the paired comparison in this repo. The canary only gets
    to use it when the metric can be computed without knowing which answer was served.
    """
    if len(canary_losses) != len(control_losses):
        raise CanaryError(
            "a paired comparison needs the same rows on both sides, got {} and {}".format(
                len(canary_losses), len(control_losses)
            )
        )
    diff = np.asarray(canary_losses, dtype=np.float64) - np.asarray(
        control_losses, dtype=np.float64
    )
    lo, hi = gate.paired_interval(diff, confidence=confidence)
    return float(np.mean(diff)), lo, hi


def required_rows(
    sd: float, half_width: float, fraction: float, confidence: float = CONFIDENCE
) -> float:
    """Total requests a split comparison needs to reach a given half width.

    Inverts the Welch standard error with one spread on both arms, which is the right
    assumption when the arms differ only in which model answered. With a canary share f,
    the two arms hold f*n and (1-f)*n rows, so

        se = sd * sqrt(1/(f*n) + 1/((1-f)*n))

    and setting z*se to the target half width gives n directly. The 1/(f*(1-f)) term is
    why a small canary is expensive: at one twentieth of traffic the factor is 21, against
    4 for an even split.

    Returned as a float and deliberately not rounded, because the caller printing it is
    the thing that should decide how to present a number this large.
    """
    if sd <= 0.0:
        raise CanaryError("cannot size a comparison on a spread of {}".format(sd))
    if half_width <= 0.0:
        raise CanaryError("cannot size a comparison to a half width of {}".format(half_width))
    if not 0.0 < fraction < 1.0:
        raise CanaryError("fraction must be strictly between 0 and 1")
    z = gate._z_for(confidence)
    return (z * sd / half_width) ** 2 * (1.0 / (fraction * (1.0 - fraction)))


def slice_imbalance(
    router: Router, keys: Sequence[str], values: np.ndarray
) -> Tuple[float, float, float]:
    """How far the routed slice's mean sits from the other arm's, on some row value.

    Returns the canary mean, the control mean and the difference. Handed the labels this
    says whether the two arms are answering the same question at all.

    The point of measuring it is what a sticky router does with it. A fixed slice carries
    a fixed composition, so running the canary for another hour does not shrink this. It
    only moves when the key population does.
    """
    mask = router.assign(keys)
    values = np.asarray(values, dtype=np.float64)
    if mask.sum() == 0 or (~mask).sum() == 0:
        raise CanaryError("one arm is empty at fraction {}".format(router.fraction))
    can = float(np.mean(values[mask]))
    con = float(np.mean(values[~mask]))
    return can, con, can - con


@dataclass(frozen=True)
class CanaryResult:
    verdict: str
    reason: str
    detail: str
    holdout: str
    fraction: float
    canary: Optional[ArmObservations] = None
    control: Optional[ArmObservations] = None
    split: Optional[Tuple[float, float, float]] = None
    shadow: Optional[Tuple[float, float, float]] = None
    required: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)

    # There was a `promoted` property here, mirroring the one on gate.Decision. It had no
    # caller. The gate's version is read by scripts/gate.py and by three checks, and this
    # one existed because the two classes looked like they should match. A mutation pass
    # flipped its comparison and nothing noticed, which is how it was found. The CLI reads
    # `result.verdict == canary_mod.PROMOTE` directly, which is one fewer thing to keep
    # in step with the four verdicts above.


def _verdict_from(mean: float, lo: float, hi: float) -> Tuple[str, str]:
    """Read an interval on a lower-is-better difference.

    Canary minus control. Below zero the whole way means the canary is better.
    """
    if hi < 0.0:
        return PROMOTE, "better"
    if lo > 0.0:
        return ROLLBACK, "worse"
    return HOLD, "not_separated"


def observe(
    router: Router,
    keys: Sequence[str],
    x: np.ndarray,
    y: np.ndarray,
    canary_payload: Dict[str, object],
    control_payload: Dict[str, object],
    canary_name: str,
    control_name: str,
    canary_hash: str,
    control_hash: str,
    shadow: bool = True,
) -> Tuple[ArmObservations, ArmObservations, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Run the replay and record what each arm scored.

    Returns the two arms as they were served, plus the shadow pair when asked for. The
    shadow pair is both models' losses on every request in the same row order, which is
    what makes a paired comparison possible. Computing it costs a second forward pass per
    request and nothing else here, and that cost is the whole argument for doing it.
    """
    if len(keys) != len(y) or len(x) != len(y):
        raise CanaryError(
            "replay needs one key and one row per label, got {}, {} and {}".format(
                len(keys), len(x), len(y)
            )
        )
    mask = router.assign(keys)
    if mask.sum() == 0 or (~mask).sum() == 0:
        raise CanaryError(
            "routing at fraction {} put every request in one arm over {} requests".format(
                router.fraction, len(keys)
            )
        )

    can_model = gate.model_from_artifact(canary_payload)
    con_model = gate.model_from_artifact(control_payload)

    can_all = gate._row_losses(y, can_model.predict_proba(x))
    con_all = gate._row_losses(y, con_model.predict_proba(x))

    served_canary = ArmObservations(
        name=CANARY,
        model_name=canary_name,
        artifact_hash=canary_hash,
        row_losses=can_all[mask],
        labels=y[mask],
    )
    served_control = ArmObservations(
        name=CONTROL,
        model_name=control_name,
        artifact_hash=control_hash,
        row_losses=con_all[~mask],
        labels=y[~mask],
    )
    pair = (can_all, con_all) if shadow else None
    return served_canary, served_control, pair


def decide(
    canary: ArmObservations,
    control: ArmObservations,
    holdout: str,
    fraction: float,
    pair: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> CanaryResult:
    """Decide on the served traffic, and report what a shadow would have said.

    The verdict comes off the split comparison, because that is the comparison a canary
    really has when the metric depends on what was served. The shadow figures go in the
    report beside it. On this project they disagree constantly and the disagreement is
    the finding, which is the same shape as the gate's report carrying each run's own
    recorded number beside the gate's.
    """
    if canary.artifact_hash == control.artifact_hash:
        return CanaryResult(
            verdict=REFUSE,
            reason="same_artifact",
            detail="both arms are the same bytes, {}".format(canary.artifact_hash[:12]),
            holdout=holdout,
            fraction=fraction,
            canary=canary,
            control=control,
        )

    for arm in (canary, control):
        if not arm.finite():
            return CanaryResult(
                verdict=REFUSE,
                reason="{}_not_finite".format(arm.name),
                detail=(
                    "the {} arm {} scored {} on {} over {} requests, and a comparison "
                    "against a non finite value is False in both directions".format(
                        arm.name, arm.model_name, arm.mean_loss, METRIC, arm.n
                    )
                ),
                holdout=holdout,
                fraction=fraction,
                canary=canary,
                control=control,
            )

    try:
        split = split_interval(canary.row_losses, control.row_losses)
    except CanaryError as exc:
        # Every other way this function can fail to answer comes back as a refusal with a
        # reason code, and this one used to come back as an exception through the caller.
        # A canary at the start of a ramp really can have one request in an arm, so the
        # first honest run of a real ramp would have ended in a traceback rather than in
        # a verdict. Found by a check written for a mutant on the emptiness guard.
        return CanaryResult(
            verdict=REFUSE,
            reason="too_few_requests",
            detail=str(exc),
            holdout=holdout,
            fraction=fraction,
            canary=canary,
            control=control,
        )
    verdict, reason = _verdict_from(*split)

    shadow = None
    if pair is not None:
        shadow = shadow_interval(pair[0], pair[1])

    half = 0.5 * (split[2] - split[1])
    sd = float(np.std(np.concatenate([canary.row_losses, control.row_losses]), ddof=1))
    required = None
    if shadow is not None:
        shadow_half = 0.5 * (shadow[2] - shadow[1])
        if shadow_half > 0.0:
            required = required_rows(sd, shadow_half, fraction)

    detail = (
        "canary minus control {:+.6e} on {}, interval [{:+.6e}, {:+.6e}] over "
        "{} and {} requests".format(
            split[0], METRIC, split[1], split[2], canary.n, control.n
        )
    )

    extra = {"split_half_width": half, "row_loss_sd": sd}
    if shadow is not None:
        extra["shadow_half_width"] = 0.5 * (shadow[2] - shadow[1])
        if extra["shadow_half_width"] > 0.0:
            extra["width_ratio"] = half / extra["shadow_half_width"]

    return CanaryResult(
        verdict=verdict,
        reason=reason,
        detail=detail,
        holdout=holdout,
        fraction=fraction,
        canary=canary,
        control=control,
        split=split,
        shadow=shadow,
        required=required,
        extra=extra,
    )


def _num(value: Optional[float], fmt: str = "{:.6f}") -> str:
    if value is None:
        return "-"
    if not np.isfinite(value):
        return "nan"
    return fmt.format(value)


def report_lines(result: CanaryResult) -> List[str]:
    """The canary report.

    It prints the split comparison, which is the one the verdict came from, and the
    shadow comparison underneath it. Two numbers about one pair of models, and the second
    one exists to say how much the first threw away.
    """
    out = ["holdout      {}  (replay of a generated holdout, not live traffic)".format(
        result.holdout)]
    out.append("metric       {}, lower is better".format(METRIC))
    out.append("routing      {:.1%} to the canary arm, sticky by request key".format(
        result.fraction))
    out.append("")

    out.append("{:<28} {:>10} {:>16} {:>14}".format(
        "", "requests", "mean {}".format(METRIC), "positive rate"))
    for arm in (result.control, result.canary):
        if arm is None:
            continue
        out.append("{:<28} {:>10} {:>16} {:>14}".format(
            "{} {}".format(arm.name, arm.model_name)[:28],
            arm.n,
            _num(arm.mean_loss),
            _num(arm.positive_rate, "{:.4f}"),
        ))

    out.append("")
    if result.split is not None:
        m, lo, hi = result.split
        out.append("split   (served)   {:+.6e}  interval [{:+.6e}, {:+.6e}]".format(m, lo, hi))
    if result.shadow is not None:
        m, lo, hi = result.shadow
        out.append("shadow  (paired)   {:+.6e}  interval [{:+.6e}, {:+.6e}]".format(m, lo, hi))
    if "width_ratio" in result.extra:
        out.append("the split interval is {:.1f} times wider than the paired one".format(
            result.extra["width_ratio"]))
    if result.required is not None:
        out.append(
            "to reach the paired resolution the split needs about {:,.0f} requests at "
            "{:.1%}".format(result.required, result.fraction)
        )

    out.append("")
    out.append("verdict      {}  ({})".format(result.verdict, result.reason))
    out.append("             {}".format(result.detail))
    if result.shadow is not None and result.split is not None:
        sv, _ = _verdict_from(*result.shadow)
        if sv != result.verdict:
            out.append(
                "             the paired comparison on the same models says {}".format(sv)
            )
    return out
