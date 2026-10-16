"""Checks for the admission rule.

No MLflow here on purpose. Admission is a pure function of a payload and every question
about it can be asked without a store, which keeps these checks in `tests/run_all.py` where
a mutation pass can reach them cheaply.

Two of the fixtures train a real model rather than hand building a payload. That is the
expensive choice and it is the right one for these two, because the whole claim is about
what the shipped training code produces on a shipped config. A payload I typed with a NaN in
it would be checking my idea of divergence.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from mcr import admission, config, train

_TRAINED = {}


def trained(name: str):
    """One run per config for the whole module. Training is the cost here, not the checks."""
    if name not in _TRAINED:
        cfg = config.load("configs/{}.yml".format(name))
        with warnings.catch_warnings():
            # A diverged config prints overflow warnings by design. The warnings are not
            # the subject and letting them through makes a passing run look broken.
            warnings.simplefilter("ignore")
            _TRAINED[name] = train.run(cfg)
    return _TRAINED[name]


def payload_of(name: str):
    return trained(name).artifact.payload


def check_the_saturation_threshold_is_where_float64_says_it_is():
    """Asserted as the literal it was derived at, not as its own name.

    A check spelling the bound as `admission.SATURATION_LOGIT` on both sides agrees with
    whatever the constant becomes, including a mutant that moved it. The number is
    log(2 / eps) for float64 and it is 36.736801 on any IEEE double.
    """
    assert abs(admission.SATURATION_LOGIT - 36.736801) < 1e-6, admission.SATURATION_LOGIT

    # And the property it is supposed to mark. One epsilon the other side of the bound the
    # sigmoid still returns something below 1, and above it the bit is gone.
    below = 1.0 / (1.0 + math.exp(-(admission.SATURATION_LOGIT - 0.5)))
    above = 1.0 / (1.0 + math.exp(-(admission.SATURATION_LOGIT + 0.5)))
    assert below < 1.0, below
    assert above == 1.0, above


def check_a_trained_baseline_is_admissible():
    v = admission.inspect(payload_of("baseline"))
    assert v.ok, v.why()
    assert v.reasons == (), v.reasons
    assert v.why() == "admissible"


def check_every_config_that_trains_is_admissible():
    """The false positive question. A gate nobody can pass is not a gate.

    All four of the other configs that reach a fit have to come back admissible, including
    `candidate-underfit`, which is a deliberately bad model and still a model.
    """
    for name in ("candidate-close", "candidate-lr", "candidate-underfit"):
        v = admission.inspect(payload_of(name))
        assert v.ok, "{} was refused: {}".format(name, v.why())


def check_the_nan_config_is_refused_on_both_a_metric_and_a_parameter():
    """`candidate-diverged` at l2 100. The case the whole rule exists for."""
    v = admission.inspect(payload_of("candidate-diverged"))
    assert not v.ok
    assert admission.NON_FINITE_METRIC in v.reasons, v.reasons
    assert admission.NON_FINITE_PARAMETER in v.reasons, v.reasons
    assert "nan" in v.why().lower(), v.why()


def check_the_diverged_config_with_finite_metrics_is_still_refused():
    """`candidate-inverted` at l2 10, and the reason it is here twice.

    Every metric this config records is finite. Holdout log loss is 19.70 and AUC is
    0.377109, both real numbers, so a rule reading `metrics` alone admits it. It is refused
    on `model.final_loss`, which carries the l2 penalty and overflows, and refused again on
    the parameter scale. Both reasons are asserted, because either one alone would let a
    later edit drop the other without a check noticing.
    """
    p = payload_of("candidate-inverted")

    for key in ("holdout_log_loss", "holdout_roc_auc", "holdout_accuracy"):
        assert math.isfinite(p["metrics"][key]), key

    v = admission.inspect(p)
    assert not v.ok
    assert admission.SATURATED_PARAMETERS in v.reasons, v.reasons
    assert admission.NON_FINITE_PARAMETER in v.reasons, v.reasons


def check_a_metrics_only_reading_would_have_admitted_it():
    """States the gap rather than describing it, so the claim above cannot go stale.

    If a later change makes `candidate-inverted` fail a metrics only reading, this check
    fails and the docstring that calls it the hard case has to be rewritten.
    """
    p = payload_of("candidate-inverted")
    metrics_only = {"metrics": p["metrics"]}
    assert admission.inspect(metrics_only).ok, "the metrics alone are no longer finite"


def check_the_parameter_norm_does_not_overflow_on_the_vector_it_is_for():
    """`np.linalg.norm` returns inf here and that is the reason the helper exists."""
    w = np.asarray(payload_of("candidate-inverted")["model"]["weights"])
    with np.errstate(over="ignore"):
        naive = float(np.linalg.norm(w))
    assert not math.isfinite(naive), naive

    careful = admission.parameter_norm(w)
    assert math.isfinite(careful), careful
    assert careful > 1e238, careful

    # And it agrees with the naive version wherever the naive version works.
    small = np.array([3.0, 4.0])
    assert abs(admission.parameter_norm(small) - 5.0) < 1e-12


def check_the_norm_of_an_empty_or_zero_vector_is_zero():
    assert admission.parameter_norm([]) == 0.0
    assert admission.parameter_norm([0.0, 0.0, 0.0]) == 0.0
    assert admission.logit_scale({"model": {"weights": [], "bias": 0.0}}) == 0.0


def check_a_non_finite_weight_survives_the_norm_as_non_finite():
    """It must not come back as a large finite number, which would read as merely diverged."""
    assert math.isnan(admission.parameter_norm([1.0, float("nan")]))
    assert math.isinf(admission.parameter_norm([1.0, float("inf")]))


def check_the_walk_finds_a_float_nobody_declared():
    """The rule reads every float, so a field added later is covered without an edit.

    A hand maintained field list is the failure mode this avoids. The payload here carries
    a NaN at a path no shipped payload has.
    """
    v = admission.inspect({"something_new": {"nested": [1.0, float("nan")]}})
    assert not v.ok
    assert admission.NON_FINITE_FIELD in v.reasons, v.reasons
    assert "something_new.nested[1]" in v.why(), v.why()


def check_an_integer_field_is_not_mistaken_for_a_float():
    """`artifact_version`, `seed` and `epochs_run` are ints and no int is NaN.

    A bool is an int in Python and the walk checks for it first, so a flag does not arrive
    at the finiteness test as 0.0 or 1.0.
    """
    v = admission.inspect({"artifact_version": 1, "seed": 20260911,
                           "model": {"epochs_run": 400, "flag": True}})
    assert v.ok, v.why()


def check_the_bias_counts_toward_the_logit_scale():
    """A model is saturated by a large intercept alone, with every weight at zero."""
    big = {"model": {"weights": [0.0, 0.0], "bias": 500.0}}
    v = admission.inspect(big)
    assert not v.ok
    assert v.reasons == (admission.SATURATED_PARAMETERS,), v.reasons
    assert v.logit_scale == 500.0


def check_a_payload_just_inside_the_threshold_is_admitted():
    """The boundary from both sides, so a mutant flipping the comparison is killed."""
    inside = {"model": {"weights": [admission.SATURATION_LOGIT - 0.01], "bias": 0.0}}
    outside = {"model": {"weights": [admission.SATURATION_LOGIT + 0.01], "bias": 0.0}}
    assert admission.inspect(inside).ok
    assert not admission.inspect(outside).ok


def check_the_detail_is_capped_and_says_how_many_it_dropped():
    """A twelve feature divergence produces more lines than anybody reads."""
    v = admission.inspect({"metrics": {"m{}".format(i): float("nan") for i in range(9)}})
    assert not v.ok
    assert len(v.detail) == 5, v.detail
    assert v.detail[-1] == "and 5 more", v.detail[-1]
    assert v.reasons == (admission.NON_FINITE_METRIC,), v.reasons


def check_the_verdict_compares_equal_for_two_payloads_failing_the_same_way():
    """`reasons` is sorted and deduplicated, so it is usable as a key."""
    a = admission.inspect({"metrics": {"x": float("nan"), "y": float("nan")}})
    b = admission.inspect({"metrics": {"q": float("nan")}})
    assert a.reasons == b.reasons == (admission.NON_FINITE_METRIC,)


def check_inspect_returns_rather_than_raises_on_the_worst_payload():
    """The gate and the canary are handed non finite candidates on purpose.

    Admission decides and the caller refuses. If this function started raising, the gate
    could no longer be shown the case its own refusal exists for.
    """
    v = admission.inspect(payload_of("candidate-diverged"))
    assert isinstance(v, admission.Admission)
    assert v.ok is False


def check_an_empty_payload_is_admissible_and_that_is_deliberate():
    """Admission answers one question and absence is not its question.

    An empty dict has no non finite float and no parameters, so it passes. Whether a
    payload has the fields it should is `mcr/artifact.py`'s round trip rule and the store's
    own schema. A rule that also policed shape would be two rules sharing a name.
    """
    assert admission.inspect({}).ok
