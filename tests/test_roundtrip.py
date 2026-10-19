"""Checks on the facts the tracking probe decides on.

The probe's arms used to be expressions in the script and nothing could reach them. These
checks exist so a mutant breaking one of those rules fails something.

Two fixture rules are applied deliberately. Every predicate is checked on a case where it
says yes and a case where it says no, because a predicate checked one way round passes
against its own inversion. And the metric fixtures differ by less than any rounding a store
would plausibly do, because a fixture whose values are far apart cannot test an exactness
rule at all.
"""

from __future__ import annotations

from mcr.roundtrip import (
    RoundTripError,
    bend,
    bumped,
    inexact_metrics,
    value_types,
    without,
)

# 1.806e-11 is the real gap between the two shipped configs on holdout log loss. A fixture
# built on a larger gap would pass under a store that rounded to six places.
TINY = 1.806e-11

PARAMS = {
    "name": "baseline",
    "seed": "7",
    "model.epochs": "40",
    "data.noise": "1.0",
}


def check_value_types_reports_the_string_only_case():
    assert value_types(PARAMS) == ["str"]


def check_value_types_sees_a_value_that_arrived_already_typed():
    mixed = dict(PARAMS)
    mixed["model.epochs"] = 40
    got = value_types(mixed)
    assert got == ["int", "str"], got
    # The inversion. If this came back as ["str"] the probe's arm would pass on a store
    # that had started handing typed values back, which is the branch nothing exercises.
    assert got != ["str"]


def check_value_types_refuses_an_empty_mapping():
    try:
        value_types({})
    except RoundTripError as exc:
        assert "empty" in str(exc), str(exc)
    else:
        raise AssertionError("an empty mapping has no value types and was not refused")


def check_a_metric_round_trip_that_held_reports_nothing():
    sent = {"holdout_log_loss": 0.5, "holdout_roc_auc": 0.671932}
    assert inexact_metrics(sent, dict(sent)) == {}


def check_a_metric_off_by_less_than_any_rounding_is_still_inexact():
    sent = {"holdout_log_loss": 0.5}
    got = {"holdout_log_loss": 0.5 + TINY}
    moved = inexact_metrics(sent, got)
    assert list(moved) == ["holdout_log_loss"], moved
    assert moved["holdout_log_loss"] == (0.5, 0.5 + TINY)


def check_a_metric_missing_on_the_way_back_counts_as_inexact():
    sent = {"holdout_log_loss": 0.5, "holdout_roc_auc": 0.6}
    moved = inexact_metrics(sent, {"holdout_log_loss": 0.5})
    assert list(moved) == ["holdout_roc_auc"], moved
    # Reported as nan rather than dropped, so a caller printing the pair cannot mistake a
    # missing metric for one that came back correct.
    assert moved["holdout_roc_auc"][1] != moved["holdout_roc_auc"][1]


def check_an_extra_metric_on_the_way_back_is_not_a_round_trip_failure():
    sent = {"holdout_log_loss": 0.5}
    got = {"holdout_log_loss": 0.5, "something_else": 9.0}
    assert inexact_metrics(sent, got) == {}


def check_inexact_metrics_refuses_an_empty_sent_side():
    try:
        inexact_metrics({}, {"a": 1.0})
    except RoundTripError as exc:
        assert "no metrics" in str(exc), str(exc)
    else:
        raise AssertionError("a round trip over no metrics was judged rather than refused")


def check_without_removes_exactly_one_key():
    short = without(PARAMS, "model.epochs")
    assert set(PARAMS) - set(short) == {"model.epochs"}
    assert len(short) == len(PARAMS) - 1
    # The original is not touched. A control that mutated the mapping it was handed would
    # leave every arm after it running against a short config.
    assert "model.epochs" in PARAMS


def check_without_refuses_a_key_that_is_not_there():
    try:
        without(PARAMS, "model.nonexistent")
    except RoundTripError as exc:
        assert "proves nothing" in str(exc), str(exc)
    else:
        raise AssertionError("removing an absent key returned a mapping nothing had changed")


def check_bend_moves_the_value_it_names_and_nothing_else():
    moved = bend(PARAMS, "model.epochs", "41")
    assert moved["model.epochs"] == "41"
    assert {k: v for k, v in moved.items() if k != "model.epochs"} == {
        k: v for k, v in PARAMS.items() if k != "model.epochs"
    }
    assert PARAMS["model.epochs"] == "40"


def check_bend_refuses_a_value_equal_to_the_one_already_there():
    try:
        bend(PARAMS, "model.epochs", "40")
    except RoundTripError as exc:
        assert "itself" in str(exc), str(exc)
    else:
        raise AssertionError("a control comparing a value to itself was allowed")


def check_bend_refuses_a_key_that_is_not_there():
    try:
        bend(PARAMS, "model.absent", "1")
    except RoundTripError as exc:
        assert "proves nothing" in str(exc), str(exc)
    else:
        raise AssertionError("bending an absent key was allowed")


def check_bumped_is_the_next_integer_up():
    assert bumped("40") == "41"
    assert bumped("0") == "1"
    assert bumped("-1") == "0"


def check_bumped_refuses_a_value_that_is_not_an_integer():
    for bad in ("1.5", "nine", ""):
        try:
            bumped(bad)
        except RoundTripError as exc:
            assert "integer valued" in str(exc), str(exc)
        else:
            raise AssertionError("{!r} was bumped rather than refused".format(bad))
