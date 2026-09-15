"""Canary routing and comparison checks.

Three things here are load bearing.

The first is that the split interval must come out wider than the paired one on the same
rows and the same models. That is the day's whole finding and a mutant that swapped the
unpaired standard error for the paired one would otherwise be invisible.

The second is the fixture rule this project keeps relearning. A fixture with one row per
group cannot test a rule about choosing between rows. Here the equivalent is a fixture
where both arms happen to be the same size, which makes Welch's two variance terms
interchangeable and hides half of the expression. The arms are deliberately lopsided.

The third is that `_verdict_from` is read three ways, by the CLI's exit code, by the
report and by the probe's answer key. Three readings of one rule want a check that the
rule is one rule.
"""

from __future__ import annotations

import math
import os

import numpy as np

from mcr import canary, gate
from mcr import train as train_mod
from mcr.config import from_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(name="fixture", seed=11, epochs=80, n_rows=2000):
    return from_dict(
        {
            "name": name,
            "seed": seed,
            "data": {
                "n_rows": n_rows,
                "n_features": 6,
                "positive_rate": 0.2,
                "noise": 1.0,
                "holdout_frac": 0.25,
            },
            "model": {
                "learning_rate": 0.5,
                "epochs": epochs,
                "l2": 0.001,
                "init": "zeros",
            },
        }
    )


def _two_models(cand_epochs=1):
    """An incumbent and a worse candidate, scored on one holdout.

    `cand_epochs` decides how far apart they are and that turns out to matter more than
    anything else in this file. At one epoch the candidate is obviously bad and pairing
    buys almost nothing, because two models that disagree everywhere have row losses that
    barely correlate. At twenty the two are close, the correlation is 0.9972 on this
    fixture, and pairing buys a factor of seventeen.

    A canary lives at the second end. Anything far apart has already been stopped by the
    promotion gate, so the traffic split is at its most expensive exactly where a canary is
    used. The checks below that are about the split's cost use twenty.
    """
    base = _cfg("base", epochs=80)
    worse = _cfg("worse", epochs=cand_epochs)
    spec = gate.spec_from_config(base)
    x, y = spec.rows()
    rb = train_mod.run(base)
    rw = train_mod.run(worse)
    mb = gate.model_from_artifact(rb.artifact.payload)
    mw = gate.model_from_artifact(rw.artifact.payload)
    return (
        spec,
        x,
        y,
        gate._row_losses(y, mb.predict_proba(x)),
        gate._row_losses(y, mw.predict_proba(x)),
        rb,
        rw,
    )


# --- the router -------------------------------------------------------------------


def check_router_refuses_a_fraction_at_or_outside_the_open_unit_interval():
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            canary.Router(fraction=bad)
        except canary.CanaryError:
            continue
        raise AssertionError("Router accepted fraction {}".format(bad))


def check_router_refuses_an_empty_salt():
    try:
        canary.Router(fraction=0.1, salt="")
    except canary.CanaryError:
        return
    raise AssertionError("Router accepted an empty salt")


def check_router_is_sticky_across_objects():
    keys = canary.replay_keys(300)
    a = [canary.Router(fraction=0.2).arm(k) for k in keys]
    b = [canary.Router(fraction=0.2).arm(k) for k in keys]
    assert a == b, "a fresh Router moved keys between arms"


def check_the_salt_changes_the_assignment():
    keys = canary.replay_keys(300)
    a = [canary.Router(fraction=0.2, salt="one").arm(k) for k in keys]
    b = [canary.Router(fraction=0.2, salt="two").arm(k) for k in keys]
    assert a != b, "two salts produced the same slice, so two canaries cannot be separated"


def check_the_router_hits_its_target_within_three_standard_errors():
    keys = canary.replay_keys(20000)
    for f in (0.05, 0.25, 0.5):
        got = float(canary.Router(fraction=f).assign(keys).mean())
        se = math.sqrt(f * (1.0 - f) / len(keys))
        assert abs(got - f) <= 3.0 * se, "target {} achieved {}".format(f, got)


def check_raising_the_fraction_only_ever_adds_keys_to_the_slice():
    """A threshold on a fixed position is monotone, and the point is what that buys.

    Ramping a canary from 5 percent to 25 percent keeps every key already in it, so the
    users who have seen the new model keep seeing it. A router that rehashed on each step
    would reshuffle the population at every ramp and make the arms incomparable across
    steps. Nothing else in this repo would notice.
    """
    keys = canary.replay_keys(2000)
    small = set(k for k in keys if canary.Router(fraction=0.05).arm(k) == canary.CANARY)
    large = set(k for k in keys if canary.Router(fraction=0.25).arm(k) == canary.CANARY)
    assert small, "the 5 percent slice was empty"
    assert small <= large, "{} keys left the slice when the fraction went up".format(
        len(small - large)
    )


def check_the_position_does_not_depend_on_python_hash_randomisation():
    """Pinned literal, so a switch to `hash()` is caught rather than merely suspected.

    `hash()` on a str is salted per process, so a router built on it would reassign every
    key on a restart while every check that compares two Routers in one process passed.

    The literal below was read off a run. The first draft of this check carried a number
    I had written from nothing and it failed on the first execution, which is the point of
    running a new check before committing it rather than after.
    """
    got = canary.Router(fraction=0.5)._position("req-000000")
    assert abs(got - 0.2041058573655147) < 1e-12, "position moved to {!r}".format(got)


# --- the two intervals ------------------------------------------------------------


def check_split_interval_is_wider_than_the_paired_one_on_the_same_rows():
    """The day's finding, as a check.

    Same rows, same two models. The only difference is that the split throws away the
    correlation between the arms. Twenty epochs against eighty, so the two models are
    close, which is the regime a canary actually runs in.
    """
    spec, x, y, base_losses, worse_losses, _rb, _rw = _two_models(cand_epochs=20)
    _, plo, phi = canary.shadow_interval(worse_losses, base_losses)
    mask = canary.Router(fraction=0.1).assign(canary.replay_keys(len(y)))
    _, slo, shi = canary.split_interval(worse_losses[mask], base_losses[~mask])
    assert (shi - slo) > 5.0 * (phi - plo), (
        "split width {:.4e} against paired {:.4e}".format(shi - slo, phi - plo)
    )


def check_split_interval_uses_both_arms_variances():
    """Lopsided arms on purpose.

    With equal sizes the two terms of Welch's standard error are interchangeable, so a
    mutant reading one arm's variance twice survives. Here one arm is spread and small
    and the other is tight and large, which makes the two terms different numbers.
    """
    rng = np.random.default_rng(3)
    small_spread = rng.normal(0.0, 4.0, size=60)
    large_tight = rng.normal(0.0, 0.1, size=3000)
    _, lo, hi = canary.split_interval(small_spread, large_tight)
    half = 0.5 * (hi - lo)
    z = gate._z_for(canary.CONFIDENCE)
    want = z * math.sqrt(
        float(np.var(small_spread, ddof=1)) / 60 + float(np.var(large_tight, ddof=1)) / 3000
    )
    assert abs(half - want) < 1e-12, "{!r} against {!r}".format(half, want)
    # And the term that dominates is the small arm's, which is the asymmetry a mutant
    # reading the large arm twice would flatten.
    only_large = z * math.sqrt(2.0 * float(np.var(large_tight, ddof=1)) / 3000)
    assert half > 20.0 * only_large, "the small arm is not dominating, so the fixture is weak"


def check_split_interval_refuses_an_arm_of_fewer_than_two_rows():
    a = np.array([1.0])
    b = np.arange(50, dtype=float)
    for pair in ((a, b), (b, a)):
        try:
            canary.split_interval(*pair)
        except canary.CanaryError:
            continue
        raise AssertionError("split_interval accepted a one row arm")


def check_split_interval_accepts_an_arm_of_exactly_two_rows():
    """The other side of the same limit, which is the side a mutant walks through.

    A check that only ever hands the guard a one row arm passes against `< 3` and against
    `<= 2` as well as against `< 2`, because one row is refused by all three. Two rows is
    the smallest sample a variance exists for and it has to be allowed.
    """
    two = np.array([1.0, 3.0])
    other = np.array([0.0, 1.0])
    mean, lo, hi = canary.split_interval(two, other)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert abs(mean - 1.5) < 1e-12, mean
    # And both ways round, so a mutant moving only the control arm's limit is caught too.
    mean2, _, _ = canary.split_interval(other, two)
    assert abs(mean2 + 1.5) < 1e-12, mean2


def check_shadow_interval_refuses_mismatched_lengths():
    try:
        canary.shadow_interval(np.zeros(10), np.zeros(11))
    except canary.CanaryError:
        return
    raise AssertionError("shadow_interval accepted arms of different lengths")


def check_shadow_interval_is_the_gates_interval():
    """One definition of the paired comparison in this repo, not two."""
    rng = np.random.default_rng(5)
    a = rng.normal(1.0, 0.3, size=400)
    b = rng.normal(0.0, 0.3, size=400)
    mean, lo, hi = canary.shadow_interval(a, b)
    glo, ghi = gate.paired_interval(a - b)
    assert (lo, hi) == (glo, ghi), "the canary's paired interval is not the gate's"
    assert abs(mean - float(np.mean(a - b))) < 1e-15


def check_the_sign_convention_matches_the_gate():
    """Positive means the canary is worse, on a metric where lower is better.

    A flipped sign here turns a rollback into a promotion, and both intervals would still
    look entirely reasonable in the report.
    """
    worse = np.full(500, 2.0)
    better = np.full(500, 1.0)
    mean, _, _ = canary.split_interval(worse, better + np.linspace(-0.01, 0.01, 500))
    assert mean > 0.0, "a worse canary produced a negative difference"


# --- sizing -----------------------------------------------------------------------


def check_required_rows_falls_as_the_square_of_the_half_width():
    a = canary.required_rows(sd=0.5, half_width=1e-3, fraction=0.5)
    b = canary.required_rows(sd=0.5, half_width=2e-3, fraction=0.5)
    assert abs(a / b - 4.0) < 1e-9, "ratio {!r}".format(a / b)


def check_required_rows_is_symmetric_in_the_fraction_and_worst_at_a_half():
    """The 1/(f*(1-f)) term, checked on both of its properties.

    Its minimum is at an even split. That is the whole reason a small canary is expensive.
    It is also symmetric, which a mutant writing 1/f alone breaks in both directions.
    """
    even = canary.required_rows(sd=0.5, half_width=1e-3, fraction=0.5)
    for f in (0.05, 0.2, 0.4):
        assert canary.required_rows(sd=0.5, half_width=1e-3, fraction=f) > even
        assert abs(
            canary.required_rows(sd=0.5, half_width=1e-3, fraction=f)
            - canary.required_rows(sd=0.5, half_width=1e-3, fraction=1.0 - f)
        ) < 1e-6, "not symmetric at {}".format(f)


def check_required_rows_agrees_with_a_measured_interval():
    """Sized against a real interval rather than only against itself.

    Draw two arms at a known spread, ask `required_rows` for the n that reaches the
    observed half width, and check it returns about the n actually used. A formula graded
    only by its own algebra can be internally consistent and wrong.
    """
    rng = np.random.default_rng(11)
    n, f = 8000, 0.25
    k = int(n * f)
    a = rng.normal(0.0, 0.6, size=k)
    b = rng.normal(0.0, 0.6, size=n - k)
    _, lo, hi = canary.split_interval(a, b)
    half = 0.5 * (hi - lo)
    sd = float(np.std(np.concatenate([a, b]), ddof=1))
    got = canary.required_rows(sd, half, f)
    assert 0.8 * n < got < 1.25 * n, "asked for {!r} rows against {} used".format(got, n)


def check_required_rows_refuses_degenerate_inputs():
    for kwargs in (
        {"sd": 0.0, "half_width": 1e-3, "fraction": 0.5},
        {"sd": -1.0, "half_width": 1e-3, "fraction": 0.5},
        {"sd": 0.5, "half_width": 0.0, "fraction": 0.5},
        {"sd": 0.5, "half_width": 1e-3, "fraction": 0.0},
        {"sd": 0.5, "half_width": 1e-3, "fraction": 1.0},
    ):
        try:
            canary.required_rows(**kwargs)
        except canary.CanaryError:
            continue
        raise AssertionError("required_rows accepted {}".format(kwargs))


# --- verdicts ---------------------------------------------------------------------


def check_verdict_reads_the_interval_and_not_the_mean():
    assert canary._verdict_from(-1.0, -2.0, -0.5)[0] == canary.PROMOTE
    assert canary._verdict_from(1.0, 0.5, 2.0)[0] == canary.ROLLBACK
    # A large mean with an interval covering zero is a hold, not a promotion.
    assert canary._verdict_from(-5.0, -11.0, 1.0)[0] == canary.HOLD
    assert canary._verdict_from(5.0, -1.0, 11.0)[0] == canary.HOLD


def check_an_interval_touching_zero_is_a_hold():
    """The boundary, on both sides. A `<=` in either comparison flips these."""
    assert canary._verdict_from(-1.0, -2.0, 0.0)[0] == canary.HOLD
    assert canary._verdict_from(1.0, 0.0, 2.0)[0] == canary.HOLD


def check_an_empty_arm_is_not_finite():
    """`finite()` guards on `n > 0` and every fixture had rows, so the limit was untested.

    An empty arm has a NaN mean and `np.isfinite(...).all()` on an empty array is True, so
    without the count term an arm that served nothing reports itself healthy and reaches
    the interval. Same shape as a checker reporting clean on zero files.
    """
    empty = canary.ArmObservations(
        canary.CANARY, "none", "aaa", np.array([]), np.array([])
    )
    assert not empty.finite()
    assert not np.isfinite(empty.mean_loss)
    one = canary.ArmObservations(
        canary.CANARY, "one", "aaa", np.array([0.5]), np.array([1])
    )
    assert one.finite(), "an arm with a single finite row should be finite"


def check_the_arm_dataclasses_are_frozen():
    """Three frozen dataclasses here and nothing asserted any of them.

    A result object that can be edited after the verdict is one a caller can talk itself
    into adjusting, and the report would then describe something the decision did not.
    """
    import dataclasses

    for cls in (canary.Router, canary.ArmObservations, canary.CanaryResult):
        assert dataclasses.fields(cls) is not None
        assert cls.__dataclass_params__.frozen, "{} is not frozen".format(cls.__name__)

    arm = canary.ArmObservations(canary.CANARY, "n", "h", np.array([1.0]), np.array([1]))
    try:
        arm.name = canary.CONTROL
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("an ArmObservations accepted a write to name")


def check_a_position_exactly_on_the_fraction_routes_to_control():
    """The router's boundary, which no real key is likely to land on.

    `_position` returns k over 2**64, so a position exactly equal to the fraction is
    reachable and vanishingly rare, which means a `<` quietly becoming a `<=` would never
    show up in traffic and would still be a changed rule. Forced here through a subclass
    rather than hunting for a key that hashes to it.
    """

    class Fixed(canary.Router):
        def _position(self, key):
            return {"on": 0.25, "below": 0.25 - 1e-12, "above": 0.25 + 1e-12}[key]

    r = Fixed(fraction=0.25)
    assert r.arm("below") == canary.CANARY
    assert r.arm("on") == canary.CONTROL, "a position equal to the fraction joined the slice"
    assert r.arm("above") == canary.CONTROL


def check_the_four_verdicts_are_distinct_strings():
    verdicts = (canary.PROMOTE, canary.ROLLBACK, canary.HOLD, canary.REFUSE)
    assert len(set(verdicts)) == 4, "two verdicts collide: {}".format(verdicts)
    # And rollback is not the gate's reject. The CLI maps them to different exit codes and
    # a pipeline reads those, so an alias here would be a silent contract change.
    assert canary.ROLLBACK != gate.REJECT


# --- observe and decide -----------------------------------------------------------


def _observed(fraction=0.2, shadow=True, cand_epochs=1):
    spec, x, y, base_losses, worse_losses, rb, rw = _two_models(cand_epochs)
    keys = canary.replay_keys(len(y))
    return canary.observe(
        router=canary.Router(fraction=fraction),
        keys=keys,
        x=x,
        y=y,
        canary_payload=rw.artifact.payload,
        control_payload=rb.artifact.payload,
        canary_name="worse",
        control_name="base",
        canary_hash=rw.content_hash,
        control_hash=rb.content_hash,
        shadow=shadow,
    ) + (spec, y)


def check_observe_splits_every_request_into_exactly_one_arm():
    arm_can, arm_con, pair, spec, y = _observed()
    assert arm_can.n + arm_con.n == len(y), "{} + {} against {}".format(
        arm_can.n, arm_con.n, len(y)
    )
    assert arm_can.n > 0 and arm_con.n > 0


def check_observe_scores_each_arm_with_its_own_model():
    """A mutant handing both arms the same model would leave every count intact.

    The two arms are scored by models whose losses differ, so the served means have to
    differ too. Checked against the shadow pair, where the right answer is known per row.
    """
    arm_can, arm_con, pair, spec, y = _observed()
    can_all, con_all = pair
    mask = canary.Router(fraction=0.2).assign(canary.replay_keys(len(y)))
    assert np.allclose(arm_can.row_losses, can_all[mask])
    assert np.allclose(arm_con.row_losses, con_all[~mask])
    assert not np.allclose(can_all, con_all), "the fixture's two models score alike"


def check_observe_returns_no_pair_when_shadow_is_off():
    _c, _n, pair, _s, _y = _observed(shadow=False)
    assert pair is None


def check_observe_shadows_by_default():
    """The default is the argument nothing passes, so nothing was checking it.

    Every caller in this repo passes `shadow=` explicitly. A mutant flipping the default
    to False turns the paired comparison off for anyone who does not, which is the case
    this module most wants to be on.
    """
    import inspect

    sig = inspect.signature(canary.observe)
    assert sig.parameters["shadow"].default is True, sig.parameters["shadow"].default


def check_observe_refuses_a_ragged_replay():
    spec = gate.spec_from_config(_cfg())
    x, y = spec.rows()
    r = train_mod.run(_cfg())
    try:
        canary.observe(
            router=canary.Router(fraction=0.2),
            keys=canary.replay_keys(len(y) - 1),
            x=x,
            y=y,
            canary_payload=r.artifact.payload,
            control_payload=r.artifact.payload,
            canary_name="a",
            control_name="b",
            canary_hash="a",
            control_hash="b",
        )
    except canary.CanaryError:
        return
    raise AssertionError("observe accepted one fewer key than labels")


def check_observe_refuses_when_the_fraction_empties_an_arm():
    """Small n and a small fraction, which is how a real canary starts.

    A routing that puts every request in one arm is not a canary with no signal, it is a
    canary that never happened, and letting it through produces an interval on an empty
    array further down.
    """
    spec = gate.spec_from_config(_cfg(n_rows=200))
    x, y = spec.rows()
    r = train_mod.run(_cfg(n_rows=200))
    try:
        canary.observe(
            router=canary.Router(fraction=1e-6),
            keys=canary.replay_keys(len(y)),
            x=x,
            y=y,
            canary_payload=r.artifact.payload,
            control_payload=r.artifact.payload,
            canary_name="a",
            control_name="b",
            canary_hash="a",
            control_hash="b",
        )
    except canary.CanaryError:
        return
    raise AssertionError("observe routed every request to one arm and did not say so")


def check_observe_allows_an_arm_of_exactly_one_request():
    """The other side of the emptiness guard.

    The guard refuses an arm of zero. A mutant moving it to one refuses an arm of one
    instead, and every fixture has arms of hundreds so nothing noticed. One request is a
    real canary at the start of a ramp. It is refused later by `split_interval`, which is
    where a sample size rule belongs, and the message there says what is wrong.

    Run both ways round. The guard has a term per arm and a fixture that only ever puts
    the single request in the canary arm leaves the control arm's term untested.
    """

    class One(canary.Router):
        """All but one request to the canary, or all but one to the control."""

        def __init__(self, fraction, lonely):
            object.__setattr__(self, "lonely", lonely)
            super().__init__(fraction=fraction)

        def assign(self, keys):
            mask = np.zeros(len(keys), dtype=bool)
            if self.lonely == canary.CANARY:
                mask[0] = True
            else:
                mask[:] = True
                mask[0] = False
            return mask

    spec = gate.spec_from_config(_cfg(n_rows=400))
    x, y = spec.rows()
    ra = train_mod.run(_cfg("a", n_rows=400, epochs=40))
    rb = train_mod.run(_cfg("b", n_rows=400, epochs=4))

    for lonely in (canary.CANARY, canary.CONTROL):
        arm_can, arm_con, _pair = canary.observe(
            router=One(0.5, lonely),
            keys=canary.replay_keys(len(y)),
            x=x,
            y=y,
            canary_payload=rb.artifact.payload,
            control_payload=ra.artifact.payload,
            canary_name="b",
            control_name="a",
            canary_hash=rb.content_hash,
            control_hash=ra.content_hash,
        )
        small, large = (arm_can, arm_con) if lonely == canary.CANARY else (arm_con, arm_can)
        assert small.n == 1, "{} arm has {} requests".format(lonely, small.n)
        assert large.n == len(y) - 1

        # And `decide` refuses rather than raising. It used to raise, which this check
        # found. Every other failure here is a refusal with a reason code and a caller
        # reads the verdict, so one path leaving by exception is one the CLI cannot see.
        out = canary.decide(arm_can, arm_con, "fp", 0.5, None)
        assert out.verdict == canary.REFUSE, out.verdict
        assert out.reason == "too_few_requests", out.reason
        assert "cannot build an interval" in out.detail, out.detail


def check_decide_refuses_two_arms_running_the_same_bytes():
    arm_can, arm_con, pair, spec, y = _observed()
    same = canary.ArmObservations(
        name=canary.CANARY,
        model_name="base",
        artifact_hash=arm_con.artifact_hash,
        row_losses=arm_can.row_losses,
        labels=arm_can.labels,
    )
    out = canary.decide(same, arm_con, "fp", 0.2, pair)
    assert out.verdict == canary.REFUSE, out.verdict
    assert out.reason == "same_artifact", out.reason


def check_decide_refuses_a_non_finite_arm_from_either_side():
    """Two refusals with different reasons, the same shape as the gate's NaN pair.

    A canary arm scoring NaN and a control arm scoring NaN are different operational
    facts. The second means production is already broken, and a message blaming the
    canary sends somebody to the wrong model.
    """
    arm_can, arm_con, pair, spec, y = _observed()
    bad_can = canary.ArmObservations(
        canary.CANARY, "worse", "aaa",
        np.concatenate([arm_can.row_losses[:-1], [np.nan]]), arm_can.labels)
    bad_con = canary.ArmObservations(
        canary.CONTROL, "base", "bbb",
        np.concatenate([arm_con.row_losses[:-1], [np.nan]]), arm_con.labels)

    first = canary.decide(bad_can, arm_con, "fp", 0.2, None)
    assert first.verdict == canary.REFUSE
    assert first.reason == "canary_not_finite", first.reason
    assert "worse" in first.detail

    second = canary.decide(arm_can, bad_con, "fp", 0.2, None)
    assert second.verdict == canary.REFUSE
    assert second.reason == "control_not_finite", second.reason
    assert "base" in second.detail, "the refusal does not name the control model"
    assert first.reason != second.reason


def check_decide_reports_the_shadow_comparison_beside_the_split_one():
    arm_can, arm_con, pair, spec, y = _observed()
    out = canary.decide(arm_can, arm_con, "fp", 0.2, pair)
    assert out.split is not None and out.shadow is not None
    split_w = out.split[2] - out.split[1]
    shadow_w = out.shadow[2] - out.shadow[1]
    assert split_w > shadow_w, "{:.4e} against {:.4e}".format(split_w, shadow_w)
    assert abs(out.extra["width_ratio"] - split_w / shadow_w) < 1e-9


def check_decide_takes_its_verdict_from_the_split_and_not_the_shadow():
    """The verdict is a fact about the traffic that was actually served.

    Handing `decide` a shadow pair that says the opposite of the served arms must not
    move the verdict. If it did, the report's two columns would not be two columns.
    """
    arm_can, arm_con, pair, spec, y = _observed()
    honest = canary.decide(arm_can, arm_con, "fp", 0.2, pair)
    n = len(pair[0])
    inverted = (np.zeros(n), np.ones(n))
    other = canary.decide(arm_can, arm_con, "fp", 0.2, inverted)
    assert other.verdict == honest.verdict, "{} moved to {}".format(
        honest.verdict, other.verdict
    )
    assert other.shadow[0] < 0.0, "the inverted pair did not reach the report"


def check_decide_sizes_the_split_against_the_shadow_it_was_given():
    arm_can, arm_con, pair, spec, y = _observed()
    out = canary.decide(arm_can, arm_con, "fp", 0.2, pair)
    assert out.required is not None
    assert out.required > len(y), (
        "sizing came back at {!r} rows, no more than the {} already used".format(
            out.required, len(y)
        )
    )
    without = canary.decide(arm_can, arm_con, "fp", 0.2, None)
    assert without.required is None, "sized a split with no paired target to size against"


def check_decide_with_a_clearly_worse_canary_says_rollback_at_a_large_fraction():
    """Half the traffic and a model trained for one epoch. If the split cannot see this
    it cannot see anything, and the rest of the module's claims would be about a broken
    comparison rather than about splitting.
    """
    arm_can, arm_con, pair, spec, y = _observed(fraction=0.5)
    out = canary.decide(arm_can, arm_con, "fp", 0.5, pair)
    assert out.verdict == canary.ROLLBACK, "{} {}".format(out.verdict, out.detail)


def check_slice_imbalance_reports_both_arms_and_their_gap():
    spec = gate.spec_from_config(_cfg())
    _x, y = spec.rows()
    keys = canary.replay_keys(len(y))
    can, con, gap = canary.slice_imbalance(canary.Router(fraction=0.2), keys, y)
    assert abs((can - con) - gap) < 1e-12
    mask = canary.Router(fraction=0.2).assign(keys)
    assert abs(can - float(y[mask].mean())) < 1e-12


def check_slice_imbalance_refuses_an_empty_arm_from_either_side():
    """Both terms of the guard.

    A fixture that only ever empties the canary arm leaves the control arm's term alone,
    and a mutant on it survives. The second case is the one that happens in practice, when
    somebody ramps a canary to everything and the control arm quietly disappears.
    """
    keys = canary.replay_keys(200)
    values = np.arange(200, dtype=float)
    for fraction in (1e-9, 1.0 - 1e-9):
        try:
            canary.slice_imbalance(canary.Router(fraction=fraction), keys, values)
        except canary.CanaryError:
            continue
        raise AssertionError(
            "slice_imbalance measured a gap against an empty arm at {}".format(fraction)
        )


# --- the report -------------------------------------------------------------------


def check_report_names_the_traffic_as_a_replay():
    """The 07-31 rule, at the point of use.

    A reader meeting these numbers in a report should not have to find a limitations
    section to learn there is no production traffic behind them.
    """
    arm_can, arm_con, pair, spec, y = _observed()
    text = "\n".join(canary.report_lines(canary.decide(arm_can, arm_con, "fp", 0.2, pair)))
    assert "replay" in text and "not live traffic" in text, text[:200]


def check_report_flags_the_two_comparisons_disagreeing():
    """A split that says hold and a shadow that says rollback is the finding, so the
    report has to say it rather than print two intervals and leave it to the reader.

    Twenty epochs against eighty. The paired comparison rejects it. The split cannot see
    it at five percent. It cannot see it at half the traffic either.
    """
    arm_can, arm_con, pair, spec, y = _observed(fraction=0.05, cand_epochs=20)
    out = canary.decide(arm_can, arm_con, "fp", 0.05, pair)
    split_v = canary._verdict_from(*out.split)[0]
    shadow_v = canary._verdict_from(*out.shadow)[0]
    text = "\n".join(canary.report_lines(out))
    if split_v == shadow_v:
        raise AssertionError(
            "the fixture no longer produces a disagreement at 5 percent, so this check "
            "asserts nothing: {} and {}".format(split_v, shadow_v)
        )
    assert "the paired comparison on the same models says" in text
    assert shadow_v in text


def check_report_prints_a_nan_arm_as_nan_rather_than_a_dash():
    """A number the arm never had and a number it computed as NaN are different facts."""
    arm_can, arm_con, pair, spec, y = _observed()
    bad = canary.ArmObservations(
        canary.CANARY, "worse", "aaa",
        np.concatenate([arm_can.row_losses[:-1], [np.nan]]), arm_can.labels)
    text = "\n".join(canary.report_lines(canary.decide(bad, arm_con, "fp", 0.2, None)))
    assert "nan" in text
    assert canary.REFUSE in text


def check_report_prints_a_table_row_for_each_arm():
    """Asserted against the table rows, not against the whole report.

    The first version of this check looked for each arm's request count anywhere in the
    text. A mutant that skipped every arm and printed no table at all survived it, because
    the detail line further down also carries both counts. The check passed on a line it
    was not about.
    """
    arm_can, arm_con, pair, spec, y = _observed()
    lines = canary.report_lines(canary.decide(arm_can, arm_con, "fp", 0.2, pair))
    rows = [ln for ln in lines if ln.startswith(("canary ", "control "))]
    assert len(rows) == 2, "expected one table row per arm, got {}: {}".format(
        len(rows), rows
    )
    assert str(arm_con.n) in rows[0] and arm_con.model_name in rows[0], rows[0]
    assert str(arm_can.n) in rows[1] and arm_can.model_name in rows[1], rows[1]


def check_the_detail_line_quotes_the_mean_the_verdict_came_from():
    """A detail reading an end of the interval instead of the middle still looks fine."""
    arm_can, arm_con, pair, spec, y = _observed()
    out = canary.decide(arm_can, arm_con, "fp", 0.2, pair)
    assert "{:+.6e}".format(out.split[0]) in out.detail, out.detail
    assert "{:+.6e}".format(out.split[1]) in out.detail
    assert "{:+.6e}".format(out.split[2]) in out.detail
    # The three are distinct, so a detail quoting one of them twice is a real difference
    # rather than something the fixture hides.
    assert len({out.split[0], out.split[1], out.split[2]}) == 3


def check_report_works_when_there_is_no_shadow_to_report():
    """The `--no-shadow` path, which has a split and no pair.

    The footer comparing the two verdicts guards on both being present. A mutant turning
    that `and` into an `or` reaches for `result.shadow` when it is None, and no check hit
    that combination until this one, because every refusal fixture had neither.
    """
    arm_can, arm_con, _pair, spec, y = _observed(shadow=False)
    out = canary.decide(arm_can, arm_con, "fp", 0.2, None)
    assert out.split is not None and out.shadow is None
    text = "\n".join(canary.report_lines(out))
    assert "split   (served)" in text
    assert "shadow" not in text
    assert "times wider" not in text, "reported a ratio with nothing to compare against"


def check_num_prints_a_missing_value_and_a_nan_differently():
    """Directly, because both branches are one line and the report reaches neither."""
    assert canary._num(None) == "-"
    assert canary._num(float("nan")) == "nan"
    assert canary._num(1.5) == "1.500000"


# --- the CLI contract -------------------------------------------------------------


def check_the_cli_maps_every_verdict_to_its_own_exit_code():
    """Four verdicts, four codes, no collisions.

    A pipeline reads these. Two verdicts sharing a code is the failure mode the gate's
    reject and refuse split was written to avoid, and this module added a fourth.
    """
    import importlib.util

    path = os.path.join(ROOT, "scripts", "canary.py")
    spec_ = importlib.util.spec_from_file_location("canary_cli", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)

    codes = {
        canary.PROMOTE: mod.EXIT_PROMOTE,
        canary.ROLLBACK: mod.EXIT_ROLLBACK,
        canary.REFUSE: mod.EXIT_REFUSE,
        canary.HOLD: mod.EXIT_HOLD,
    }
    assert len(set(codes.values())) == 4, "exit codes collide: {}".format(codes)
    assert codes[canary.PROMOTE] == 0, "a promotion must be the zero exit"
    assert mod.EXIT_HOLD != mod.EXIT_ROLLBACK, "hold and rollback share an exit code"


def check_the_cli_help_works_without_mlflow():
    """`--help` must not need the optional dependency.

    The registry import sits inside main for this reason and it is easy to undo by
    tidying the imports to the top of the file.
    """
    import importlib.util

    path = os.path.join(ROOT, "scripts", "canary.py")
    spec_ = importlib.util.spec_from_file_location("canary_cli_help", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    text = mod.build_parser().format_help()
    for flag in ("--fraction", "--salt", "--no-shadow", "--promote"):
        assert flag in text, "{} is missing from the help".format(flag)


def check_the_cli_has_no_rollback_flag():
    """A rollback verdict must not move the stage, and the first draft's flag did.

    The canary is a version on trial and the control arm is whatever holds the stage, so
    rolling the stage back in response to a bad canary moves production off a model the
    canary result says nothing about. It reads as the obvious safety action and it is a
    change to the one thing that was working.

    Pinned as an absence because the flag is easy to add back by anyone reading the four
    verdicts and reaching for symmetry.
    """
    import importlib.util

    path = os.path.join(ROOT, "scripts", "canary.py")
    spec_ = importlib.util.spec_from_file_location("canary_cli_norollback", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    assert "--rollback" not in mod.build_parser().format_help()

    # By AST rather than by substring. The first version of this check grepped the source
    # and failed on the comment that explains why the call is absent, which would have
    # left the only two ways out as deleting the explanation or weakening the check.
    import ast

    tree = ast.parse(open(path).read())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "rollback"
    ]
    assert not calls, (
        "the canary script calls a rollback at line {}, which moves the stage the control "
        "arm is serving from".format(calls[0].lineno)
    )


def check_the_cli_report_tags_carry_both_intervals():
    """Recording only the interval the verdict came from throws away the finding."""
    import importlib.util

    path = os.path.join(ROOT, "scripts", "canary.py")
    spec_ = importlib.util.spec_from_file_location("canary_cli_tags", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)

    arm_can, arm_con, pair, _spec, _y = _observed(cand_epochs=20)
    result = canary.decide(arm_can, arm_con, "fp", 0.2, pair)
    tags = mod.report_tags(result, "production", canary)
    assert "canary.split" in tags and "canary.shadow" in tags, sorted(tags)
    assert tags["canary.verdict"] == result.verdict
    assert tags["canary.fraction"] == repr(0.2)
    # Round trip, so a tag holding a truncated float is caught here rather than by a
    # reader six weeks later.
    assert eval(tags["canary.split"]) == result.split
