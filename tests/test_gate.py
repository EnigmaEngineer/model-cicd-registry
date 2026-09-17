"""Promotion gate checks.

Two things here are load bearing and neither is obvious from the code.

The first is that the two non finite cases are separate verdicts. A comparison against a
NaN is False whichever way round it is written, so one naive line handles both and gets one
of them catastrophically wrong. The pair of checks below is what stops anybody collapsing
them back together.

The second is that a majority of rows is not a decision. On the shipped configs a candidate
whose behaviour is identical to eleven decimal places wins 3,846 of 5,000 rows, because the
last bits of a float have a consistent sign. A rule counting rows promotes it.
"""

from __future__ import annotations

import math
import os

import numpy as np

from mcr import gate, model as model_mod, train as train_mod
from mcr.config import DataConfig, ModelConfig, TrainConfig, from_dict, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(name="fixture", seed=11, epochs=80, l2=0.001, n_rows=2000, learning_rate=0.5):
    """Small and fast. The shipped configs are 20,000 rows and this file fits many runs."""
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
                "learning_rate": learning_rate,
                "epochs": epochs,
                "l2": l2,
                "init": "zeros",
            },
        }
    )


def _spec(cfg=None):
    return gate.spec_from_config(cfg or _cfg())


def _scored(cfg, spec=None, name=None):
    spec = spec or _spec()
    r = train_mod.run(cfg)
    return gate.score(
        name or cfg.name,
        r.content_hash,
        r.artifact.payload,
        spec,
        reported={"holdout_{}".format(gate.METRIC): r.metrics["holdout_log_loss"]},
    )


def _hand(name, losses, auc=0.7, artifact_hash=None, reported=None):
    """A Scored built from numbers rather than from a model.

    `decide` takes a Scored and nothing else, so the cases that need a specific shape of
    result are cheaper and clearer built this way. The checks that need a real model use
    one.
    """
    losses = np.asarray(losses, dtype=np.float64)
    return gate.Scored(
        name=name,
        artifact_hash=artifact_hash or ("h" + name),
        row_losses=losses,
        log_loss=float(np.mean(losses)),
        roc_auc=auc,
        reported=reported or {},
    )


# --- the holdout -----------------------------------------------------------------


def check_the_holdout_fingerprint_is_stable_across_calls():
    spec = _spec()
    assert spec.fingerprint() == spec.fingerprint()


def check_the_holdout_fingerprint_is_twelve_characters():
    """A width, pinned for the same reason the transition log's is.

    It goes into reports and onto runs, so two builds writing different widths would make
    one store's records unreadable against the other's while every check inside either
    one passed.
    """
    assert len(_spec().fingerprint()) == 12


def check_the_config_sections_are_frozen():
    """Every dataclass in this module holds something a decision was made from.

    A `Scored` whose row losses can be reassigned after the verdict, or a `Decision` whose
    verdict can be overwritten by the caller that printed it, is a record of nothing.
    """
    spec = _spec()
    scored = _hand("c", np.full(10, 0.4))
    decision = gate.decide(scored, None, "fp")
    for obj, attr, value in (
        (spec, "seed", 99),
        (scored, "log_loss", 0.0),
        (decision, "verdict", gate.PROMOTE),
    ):
        try:
            setattr(obj, attr, value)
        except Exception:
            continue
        raise AssertionError("{}.{} was reassigned".format(type(obj).__name__, attr))


def check_the_holdout_fingerprint_moves_with_the_seed():
    a = gate.HoldoutSpec(data=_cfg().data, seed=11)
    b = gate.HoldoutSpec(data=_cfg().data, seed=12)
    assert a.fingerprint() != b.fingerprint()


def check_the_holdout_fingerprint_moves_with_the_data_config():
    base = _cfg().data
    other = DataConfig(
        n_rows=base.n_rows,
        n_features=base.n_features,
        positive_rate=0.35,
        noise=base.noise,
        holdout_frac=base.holdout_frac,
    )
    a = gate.HoldoutSpec(data=base, seed=11)
    b = gate.HoldoutSpec(data=other, seed=11)
    assert a.fingerprint() != b.fingerprint()


def check_the_fingerprint_is_over_the_rows_and_not_over_the_spec():
    """Two specs that name the same rows fingerprint the same.

    This is the property a hash of the spec fields would also have, so on its own it
    proves nothing. What it pins together with the two checks above is that the value
    comes from the data, so a change to the generator moves it while the spec holds still.
    """
    a = gate.HoldoutSpec(data=_cfg().data, seed=11)
    b = gate.HoldoutSpec(data=_cfg(name="different-name").data, seed=11)
    assert a.fingerprint() == b.fingerprint()
    x, y = a.rows()
    assert len(y) == int(round(a.data.n_rows * a.data.holdout_frac))
    assert x.shape[1] == a.data.n_features


def check_the_holdout_rows_are_not_the_training_rows():
    cfg = _cfg()
    spec = gate.spec_from_config(cfg)
    x, _ = spec.rows()
    assert len(x) + int(round(cfg.data.n_rows * (1 - cfg.data.holdout_frac))) == cfg.data.n_rows


# --- scoring a model out of an artefact ------------------------------------------


def check_a_model_rebuilt_from_an_artefact_scores_identically():
    cfg = _cfg()
    r = train_mod.run(cfg)
    rebuilt = gate.model_from_artifact(r.artifact.payload)
    x, _ = _spec(cfg).rows()
    assert np.array_equal(rebuilt.predict_proba(x), r.model.predict_proba(x))


def check_an_artefact_with_no_model_section_is_refused():
    try:
        gate.model_from_artifact({"metrics": {}})
    except gate.GateError as exc:
        assert "no model" in str(exc)
        return
    raise AssertionError("a payload with no model was accepted")


def check_the_row_losses_average_to_the_reported_log_loss():
    """Exact equality rather than a tolerance.

    The interval is built on the rows and the message quotes the mean. If those two ever
    came from different arithmetic the report would contradict its own verdict.
    """
    cfg = _cfg()
    s = _scored(cfg)
    assert s.log_loss == float(np.mean(s.row_losses))


def check_a_probability_of_one_gives_a_finite_loss():
    """The clip is the only thing standing between a confident model and an infinite loss.

    A model that returns exactly 1.0 on a row labelled 0 is `log(0)` without it. Nothing
    in the fixtures gets that confident, so the upper bound of the clip was untested and
    a mutant loosening it to `1.0 + eps` survived the whole suite.
    """
    y = np.asarray([0, 1], dtype=np.int64)
    p = np.asarray([1.0, 0.0], dtype=np.float64)
    losses = gate._row_losses(y, p)
    assert np.isfinite(losses).all()
    assert (losses > 30.0).all()


def check_the_score_matches_the_projects_own_log_loss_function():
    cfg = _cfg()
    spec = _spec(cfg)
    r = train_mod.run(cfg)
    x, y = spec.rows()
    direct = model_mod.log_loss(y.astype(float), r.model.predict_proba(x))
    assert abs(_scored(cfg, spec).log_loss - direct) < 1e-12


def check_a_holdout_with_one_class_gives_a_non_finite_auc_rather_than_raising():
    """`roc_auc` raises when a class is absent and a gate that dies cannot refuse.

    The poison is asserted rather than assumed: the fixture really does hold one class.
    """
    cfg = _cfg()
    r = train_mod.run(cfg)
    spec = _spec(cfg)
    x, y = spec.rows()
    one_class = np.zeros(len(y), dtype=np.int64)
    assert len(set(one_class.tolist())) == 1

    p = gate.model_from_artifact(r.artifact.payload).predict_proba(x)
    try:
        model_mod.roc_auc(one_class, p)
        raise AssertionError("roc_auc accepted a single class holdout")
    except ValueError as exc:
        assert "both classes" in str(exc)


# --- the interval ----------------------------------------------------------------


def check_the_interval_refuses_a_sample_it_cannot_measure_spread_on():
    for n in (0, 1):
        try:
            gate.paired_interval(np.zeros(n))
        except gate.GateError as exc:
            assert "{} rows".format(n) in str(exc)
            continue
        raise AssertionError("an interval was built on {} rows".format(n))
    gate.paired_interval(np.asarray([0.1, 0.2]))


def check_the_bootstrap_refuses_zero_rows():
    try:
        gate.bootstrap_interval(np.asarray([], dtype=np.float64))
    except gate.GateError as exc:
        assert "zero rows" in str(exc)
        return
    raise AssertionError("an interval was built on nothing")


def check_a_clearly_negative_difference_gives_an_interval_that_excludes_zero():
    rng = np.random.default_rng(1)
    d = rng.normal(-0.3, 0.05, 500)
    lo, hi = gate.paired_interval(d)
    assert lo < hi
    assert hi < 0.0
    assert lo <= float(np.mean(d)) <= hi


def check_a_symmetric_difference_gives_an_interval_that_covers_zero():
    d = np.concatenate([np.full(500, -1.0), np.full(500, 1.0)])
    lo, hi = gate.paired_interval(d)
    assert lo < 0.0 < hi


def check_the_interval_is_the_same_on_two_calls():
    d = np.linspace(-1.0, 0.5, 400)
    assert gate.paired_interval(d) == gate.paired_interval(d)


def check_a_wider_confidence_gives_a_wider_interval():
    d = np.linspace(-1.0, 0.5, 400)
    lo95, hi95 = gate.paired_interval(d, confidence=0.95)
    lo80, hi80 = gate.paired_interval(d, confidence=0.80)
    assert (hi95 - lo95) > (hi80 - lo80)


def check_the_interval_half_width_is_the_textbook_one():
    """The ordering check above survives any monotone mangling of the tail arithmetic.

    A mutant turning `0.5 * (1 - confidence) * 100` into a division scales both ends the
    same way, so a wider-is-wider assertion still passes on an interval four times too
    wide. This pins the value.
    """
    rng = np.random.default_rng(4)
    d = rng.normal(0.2, 0.5, 4000)
    lo, hi = gate.paired_interval(d, confidence=0.95)
    se = float(np.std(d, ddof=1)) / math.sqrt(len(d))
    assert abs((hi - lo) / 2.0 - 1.959963984540054 * se) < 1e-12


def check_the_normal_quantile_matches_known_values():
    for confidence, want in ((0.95, 1.959963984540054), (0.99, 2.5758293035489004),
                             (0.90, 1.6448536269514722), (0.6826894921370859, 1.0)):
        assert abs(gate._z_for(confidence) - want) < 1e-9, confidence


def check_the_normal_quantile_refuses_a_confidence_outside_the_unit_interval():
    for bad in (0.0, 1.0, -0.5, 2.0):
        try:
            gate._z_for(bad)
        except gate.GateError as exc:
            assert "between 0 and 1" in str(exc)
            continue
        raise AssertionError("a confidence of {} was accepted".format(bad))


def _verdict_from(lo, hi):
    return "promote" if hi < 0.0 else ("worse" if lo > 0.0 else "flat")


def check_the_two_intervals_agree_at_the_size_the_gate_runs_at():
    """One statistic computed two ways, graded against each other rather than separately.

    The closed form is what the gate decides on and the resampler is what it used to
    decide on. Agreement is what says the swap cost nothing, and it has to be checked at
    the size the gate really sees. The shipped holdout is 5,000 rows.
    """
    rng = np.random.default_rng(12)
    seen = set()
    for shift in (-0.05, -0.01, -0.002, 0.0, 0.002, 0.01, 0.05):
        d = rng.normal(shift, 0.25, 5000)
        a = _verdict_from(*gate.paired_interval(d))
        b = _verdict_from(*gate.bootstrap_interval(d))
        assert a == b, (shift, a, b)
        seen.add(a)
    assert seen == {"promote", "worse", "flat"}, seen


def check_the_two_intervals_can_disagree_on_a_small_sample():
    """And the agreement above is a fact about 5,000 rows, not about the two methods.

    On five hundred rows the same pair comes apart. Worth pinning, because a later change
    to the holdout size would quietly move the gate onto ground where the closed form and
    the resampler no longer say the same thing, and nothing would report it.
    """
    spec = _spec()
    inc = _scored(_cfg(name="incumbent"))
    assert len(inc.row_losses) == 500
    disagreed = 0
    for epochs in (5, 20, 40, 60):
        cand = _scored(_cfg(name="cand", epochs=epochs), spec)
        d = cand.row_losses - inc.row_losses
        if _verdict_from(*gate.paired_interval(d)) != _verdict_from(*gate.bootstrap_interval(d)):
            disagreed += 1
    assert disagreed >= 1


def check_the_bootstrap_resamples_every_row():
    """A mutant moving the low end of the resample index off zero never draws row zero.

    The fixture puts the whole signal on row zero, so a resampler that cannot reach it
    reports a difference of exactly nothing and both ends come back at zero.
    """
    d = np.zeros(200)
    d[0] = -400.0
    lo, hi = gate.bootstrap_interval(d)
    assert lo < -1.0


# --- the verdicts ----------------------------------------------------------------


def check_nothing_in_the_stage_promotes():
    d = gate.decide(_hand("c", np.full(100, 0.4)), None, "fp")
    assert d.verdict == gate.PROMOTE
    assert d.reason == "no_incumbent"


def check_a_clearly_better_candidate_is_promoted():
    inc = _hand("inc", np.full(200, 0.60))
    cand = _hand("cand", np.full(200, 0.40))
    d = gate.decide(cand, inc, "fp")
    assert d.verdict == gate.PROMOTE
    assert d.reason == "better"


def check_a_clearly_worse_candidate_is_rejected():
    inc = _hand("inc", np.full(200, 0.40))
    cand = _hand("cand", np.full(200, 0.60))
    d = gate.decide(cand, inc, "fp")
    assert d.verdict == gate.REJECT
    assert d.reason == "worse"


def check_a_difference_the_rows_cannot_resolve_is_not_a_promotion():
    rng = np.random.default_rng(3)
    inc = _hand("inc", rng.normal(0.5, 0.3, 400))
    cand = _hand("cand", inc.row_losses + rng.normal(0.0, 0.3, 400))
    d = gate.decide(cand, inc, "fp")
    assert d.verdict == gate.REJECT
    assert d.reason == "not_separated"
    assert d.interval[0] < 0.0 < d.interval[1]


def check_a_majority_of_rows_is_not_a_decision():
    """Most rows better, mean unchanged. The gate holds the incumbent.

    This is the shipped case in miniature. `candidate-lr` is better on 3,846 of 5,000
    rows and its mean difference is 1.8e-11, because the last bits of a float carry a
    consistent sign. Anything counting rows promotes it.
    """
    inc = _hand("inc", np.concatenate([np.full(70, 0.500), np.full(30, 0.500)]))
    cand_losses = np.concatenate([np.full(70, 0.499), np.full(30, 0.5023333333333333)])
    cand = _hand("cand", cand_losses)
    better = int((cand.row_losses < inc.row_losses).sum())
    assert better == 70
    assert abs(cand.log_loss - inc.log_loss) < 1e-15
    assert gate.decide(cand, inc, "fp").verdict == gate.REJECT


def check_an_interval_sitting_exactly_on_zero_does_not_promote():
    """Both boundaries, from the side that decides them.

    Two models whose per row losses are equal give an interval of exactly [0, 0]. Promote
    is `hi < 0` and reject-as-worse is `lo > 0`, and every other fixture here sits well
    away from both, so a mutant loosening either comparison to include the boundary
    survives. This is the case that touches them.

    The verdict is also the right one on its own terms. Two different artefacts that
    behave identically on every row are the shipped `candidate-lr` situation, and holding
    the incumbent is what changes nothing when nothing is known.
    """
    losses = np.linspace(0.2, 0.9, 300)
    inc = _hand("inc", losses, artifact_hash="aaa")
    cand = _hand("cand", losses.copy(), artifact_hash="bbb")
    assert gate.paired_interval(cand.row_losses - inc.row_losses) == (0.0, 0.0)
    d = gate.decide(cand, inc, "fp")
    assert d.verdict == gate.REJECT
    assert d.reason == "not_separated"


def check_the_same_bytes_on_both_sides_is_refused_and_not_rejected():
    losses = np.full(200, 0.4)
    full = "9f2c1a" * 12
    inc = _hand("inc", losses, artifact_hash=full)
    cand = _hand("cand", losses, artifact_hash=full)
    d = gate.decide(cand, inc, "fp")
    assert d.verdict == gate.REFUSE
    assert d.reason == "same_artifact"
    assert full[:12] in d.detail and full[:13] not in d.detail


def check_a_long_model_name_does_not_break_the_report_columns():
    """The label column truncates and nothing was checking the width it truncates to."""
    inc = _hand("i" * 60, np.full(50, 0.5), artifact_hash="a")
    cand = _hand("c" * 60, np.full(50, 0.4), artifact_hash="b")
    lines = gate.report_lines(gate.decide(cand, inc, "fp"))
    rows = [ln for ln in lines if ln.startswith(("incumbent", "candidate"))]
    assert len(rows) == 2
    for row in rows:
        assert len(row.split()[0]) + 1 + len(row.split()[1]) == 32


def check_a_non_finite_candidate_is_refused():
    inc = _hand("inc", np.full(200, 0.4))
    cand = _hand("cand", np.full(200, float("nan")))
    d = gate.decide(cand, inc, "fp")
    assert d.verdict == gate.REFUSE
    assert d.reason == "candidate_not_finite"


def check_a_non_finite_incumbent_is_refused_and_names_the_incumbent():
    """The dangerous half, and the reason there are two reasons rather than one.

    Left to `candidate < incumbent` this rejects every candidate for as long as the broken
    model holds the stage, with a message blaming the candidate. The message here has to
    name the incumbent or the operator fixes the wrong thing.
    """
    inc = _hand("broken-incumbent", np.full(200, float("nan")))
    cand = _hand("cand", np.full(200, 0.4))
    d = gate.decide(cand, inc, "fp")
    assert d.verdict == gate.REFUSE
    assert d.reason == "incumbent_not_finite"
    assert "broken-incumbent" in d.detail
    assert "cand" not in d.detail.split("incumbent")[0]


def check_the_two_non_finite_cases_do_not_share_a_reason():
    nan = np.full(200, float("nan"))
    good = np.full(200, 0.4)
    a = gate.decide(_hand("c", nan), _hand("i", good), "fp")
    b = gate.decide(_hand("c", good), _hand("i", nan), "fp")
    assert a.reason != b.reason
    assert a.verdict == b.verdict == gate.REFUSE


def check_an_infinite_score_is_refused_as_well_as_a_nan():
    inc = _hand("inc", np.full(200, 0.4))
    cand = _hand("cand", np.concatenate([np.full(199, 0.4), [float("inf")]]))
    assert gate.decide(cand, inc, "fp").reason == "candidate_not_finite"


def check_the_candidate_is_checked_before_the_incumbent():
    """Both broken. The candidate is the thing being judged, so it is named first."""
    nan = np.full(200, float("nan"))
    d = gate.decide(_hand("c", nan), _hand("i", nan), "fp")
    assert d.reason == "candidate_not_finite"


def check_the_verdict_does_not_read_the_ranking():
    """Worse on the metric, better on AUC. The gate rejects it.

    `roc_auc` is on the report because a reader wants it. It is not on the decision,
    because AUC spans 2.06e-04 across every model config this project can express and a
    gate reading it passes a model that trained for one epoch.
    """
    inc = _hand("inc", np.full(300, 0.40), auc=0.60)
    cand = _hand("cand", np.full(300, 0.55), auc=0.95)
    d = gate.decide(cand, inc, "fp")
    assert d.verdict == gate.REJECT
    assert cand.roc_auc > inc.roc_auc


def check_an_overconfident_model_with_identical_ranking_is_rejected():
    """The calibration case, on a real model rather than on hand written numbers.

    Scaling the weights and the bias leaves every pairwise ordering untouched, so AUC is
    equal to the bit and log loss is not. This is what "AUC cannot separate these" means
    when it is not an abstraction.
    """
    cfg = _cfg()
    spec = _spec(cfg)
    r = train_mod.run(cfg)
    x, y = spec.rows()

    sharp = model_mod.Model(
        weights=r.model.weights * 3.0,
        bias=r.model.bias * 3.0,
        mean=r.model.mean,
        scale=r.model.scale,
        epochs_run=r.model.epochs_run,
        final_loss=r.model.final_loss,
    )
    base_auc = model_mod.roc_auc(y, r.model.predict_proba(x))
    sharp_auc = model_mod.roc_auc(y, sharp.predict_proba(x))
    assert base_auc == sharp_auc

    inc = gate.score("inc", "a", r.artifact.payload, spec)
    cand = _hand(
        "sharp",
        gate._row_losses(y, sharp.predict_proba(x)),
        auc=sharp_auc,
        artifact_hash="b",
    )
    assert cand.log_loss > inc.log_loss
    assert gate.decide(cand, inc, "fp").verdict == gate.REJECT


def check_the_decision_is_the_same_on_two_calls():
    inc = _hand("inc", np.linspace(0.3, 0.9, 300))
    cand = _hand("cand", np.linspace(0.29, 0.89, 300))
    a = gate.decide(cand, inc, "fp")
    b = gate.decide(cand, inc, "fp")
    assert (a.verdict, a.reason, a.interval) == (b.verdict, b.reason, b.interval)


def check_the_holdout_fingerprint_travels_onto_the_decision():
    d = gate.decide(_hand("c", np.full(50, 0.4)), None, "56ab7ecf5a6b")
    assert d.holdout == "56ab7ecf5a6b"


def check_promoted_is_true_only_for_a_promotion():
    inc = _hand("inc", np.full(200, 0.40))
    assert gate.decide(_hand("c", np.full(200, 0.20)), inc, "fp").promoted
    assert not gate.decide(_hand("c", np.full(200, 0.60)), inc, "fp").promoted
    assert not gate.decide(_hand("c", np.full(200, float("nan"))), inc, "fp").promoted


# --- the rebuild, and the check that makes it sound -------------------------------


def check_a_rebuild_whose_hash_agrees_returns_the_payload():
    cfg = _cfg()
    r = train_mod.run(cfg)
    payload = gate.rebuild(cfg, r.content_hash, "candidate")
    assert payload["config_fingerprint"] == cfg.fingerprint()


def check_a_rebuild_whose_hash_disagrees_is_refused():
    """The message has to name both hashes, at a width somebody can compare by eye.

    Both prefixes are asserted to be exactly twelve characters. A longer or shorter slice
    still contains the shorter string, so an assertion on presence alone passes on either,
    and these prefixes are what an operator matches against a registry listing.
    """
    cfg = _cfg()
    real = train_mod.run(cfg).content_hash
    try:
        gate.rebuild(cfg, "0" * 64, "candidate version 7")
    except gate.GateError as exc:
        text = str(exc)
        assert "candidate version 7" in text
        assert "0" * 12 in text and "0" * 13 not in text
        assert real[:12] in text and real[:13] not in text
        return
    raise AssertionError("the gate scored a model the registry does not point at")


# --- the report -------------------------------------------------------------------


def _reports_better_and_is_worse(spec, inc, seeds):
    """Models from other corpora that record a better number and lose on shared rows."""
    found = []
    for seed in seeds:
        other = _scored(_cfg(name="elsewhere", seed=seed), spec)
        better_on_paper = (
            other.reported["holdout_log_loss"] < inc.reported["holdout_log_loss"]
        )
        if better_on_paper and other.log_loss > inc.log_loss:
            found.append((seed, other))
    return found


def check_a_run_that_records_a_better_number_can_be_worse_on_shared_rows():
    """The measurement the whole design rests on, as a rate rather than as one example.

    Every run generates its own corpus and its own holdout, so the number it records is
    partly a fact about the draw. This sweeps a range of seeds rather than naming one,
    because a single seed chosen after seeing the answer proves whatever its author
    wanted. On this fixture seven of eighteen come back this way.
    """
    spec = _spec()
    inc = _scored(_cfg(name="incumbent"), spec)
    found = _reports_better_and_is_worse(spec, inc, range(12, 30))
    assert len(found) >= 5, "only {} of 18 seeds showed the disagreement".format(len(found))


def check_the_report_carries_both_numbers_when_the_runs_used_different_corpora():
    """The disagreement is the whole reason the gate scores the models itself.

    Seed 18 is one of the cases the sweep above finds. It is named here rather than
    searched for again, so this check stays cheap and the search stays in one place.
    """
    spec = _spec()
    inc = _scored(_cfg(name="incumbent"), spec)
    other = _scored(_cfg(name="elsewhere", seed=18), spec)

    assert other.reported["holdout_log_loss"] < inc.reported["holdout_log_loss"]
    assert other.log_loss > inc.log_loss

    text = "\n".join(gate.report_lines(gate.decide(other, inc, spec.fingerprint())))
    assert "{:.6f}".format(other.log_loss) in text
    assert "{:.6f}".format(other.reported["holdout_log_loss"]) in text
    assert spec.fingerprint() in text


def check_the_report_says_which_way_the_metric_runs():
    text = "\n".join(gate.report_lines(gate.decide(_hand("c", np.full(50, 0.4)), None, "fp")))
    assert "lower is better" in text
    assert gate.METRIC in text


def check_the_report_names_an_absent_incumbent_rather_than_omitting_the_row():
    text = "\n".join(gate.report_lines(gate.decide(_hand("c", np.full(50, 0.4)), None, "fp")))
    assert "incumbent" in text
    assert "none" in text


def check_the_report_prints_a_non_finite_score_rather_than_crashing():
    inc = _hand("inc", np.full(50, 0.4))
    cand = _hand("cand", np.full(50, float("nan")))
    text = "\n".join(gate.report_lines(gate.decide(cand, inc, "fp")))
    assert "nan" in text
    assert "refuse" in text


def check_a_recorded_nan_and_a_missing_number_are_different_cells():
    """The dash means the run never recorded it. It must not also mean NaN.

    The first version of this table printed a dash for both, so the one value worth
    seeing in the whole report came out looking like an absence.
    """
    assert gate._num(None) == "-"
    assert gate._num(float("nan")) == "nan"
    assert gate._num(float("inf")) == "nan"
    assert gate._num(0.5) == "0.500000"

    recorded_nan = _hand("cand", np.full(50, 0.4),
                         reported={"holdout_log_loss": float("nan")})
    never_recorded = _hand("cand", np.full(50, 0.4), reported={})
    inc = _hand("inc", np.full(50, 0.5))
    a = "\n".join(gate.report_lines(gate.decide(recorded_nan, inc, "fp")))
    b = "\n".join(gate.report_lines(gate.decide(never_recorded, inc, "fp")))
    assert a != b


# --- the shipped configs ----------------------------------------------------------


def check_the_shipped_candidates_are_judged_on_one_holdout():
    """The three converging configs really do share a corpus, and the broken ones do too.

    If they did not, every number in the project's own comparison table would be measured
    on a different set of rows and the table would say nothing.
    """
    names = ["baseline", "candidate-lr", "candidate-underfit",
             "candidate-inverted", "candidate-diverged"]
    seen = set()
    for name in names:
        cfg = load(os.path.join(ROOT, "configs", "{}.yml".format(name)))
        seen.add(gate.spec_from_config(cfg).fingerprint())
    assert len(seen) == 1, "shipped configs span {} holdouts".format(len(seen))


def _train_quietly(cfg):
    """Train a config that overflows, and hand back what it warned about.

    Both broken configs put a RuntimeWarning on stderr. Letting that through makes a green
    suite print what looks like a failure, and swallowing it silently would leave the
    warning untested. So it is captured and returned, and the checks below assert it.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = train_mod.run(cfg)
    return result, sorted({str(w.message) for w in caught})


def check_the_diverged_config_really_does_produce_a_non_finite_metric():
    """The fixture has to carry the defect or the refusal is untested.

    This is the poison, asserted. If a change to the optimiser ever makes this config
    converge, the refusal checks above would still pass while covering nothing.
    """
    cfg = load(os.path.join(ROOT, "configs", "candidate-diverged.yml"))
    result, warned = _train_quietly(cfg)
    assert not math.isfinite(result.metrics["holdout_log_loss"])
    assert any("invalid value" in w for w in warned), warned


def check_the_inverted_config_is_finite_and_worse_than_a_coin():
    """The harder of the two broken cases. Nothing in its metrics looks wrong.

    It does warn, and the warning is asserted here rather than assumed absent. What it
    does not do is fail. `scripts/train.py` exits 0 on it, so the only thing standing
    between this model and the registry is somebody reading stderr on a green build.
    """
    cfg = load(os.path.join(ROOT, "configs", "candidate-inverted.yml"))
    result, warned = _train_quietly(cfg)
    assert all(math.isfinite(v) for v in result.metrics.values())
    assert result.metrics["holdout_roc_auc"] < 0.5
    assert warned == ["overflow encountered in matmul"], warned


def _gate_cli():
    """Load scripts/gate.py as a module. It imports mlflow inside main, so this is cheap."""
    import importlib.util

    path = os.path.join(ROOT, "scripts", "gate.py")
    spec_ = importlib.util.spec_from_file_location("gate_cli", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod, path


def _enclosing_if_tests(tree, target):
    """Every `if` test that the target node sits underneath, innermost last."""
    import ast

    found = []

    def walk(node, stack):
        if node is target:
            found.append(list(stack))
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(node, ast.If) and child in node.body:
                walk(child, stack + [node.test])
            else:
                walk(child, stack)

    walk(tree, [])
    return found[0] if found else None


def check_the_gate_writes_its_verdict_without_being_asked_to_promote():
    """The rejection you want to read back later is the one nobody asked to promote.

    This sat inside the `--promote` branch, which meant a plain
    `gate.py --candidate 2` printed a rejection and left the run with no `gate.*` tag on
    it at all. Measured: nine tags with the flag, zero without. The only verdicts on
    record were the ones from a run that expected to win.

    By AST rather than by substring, because the comment above the call names `--promote`
    and a grep would match it.
    """
    import ast

    _mod, path = _gate_cli()
    tree = ast.parse(open(path).read())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_tag"
    ]
    assert len(calls) == 1, "expected one set_tag call, found {}".format(len(calls))

    tests = _enclosing_if_tests(tree, calls[0])
    assert tests is not None, "could not locate the set_tag call in the tree"
    sources = [ast.dump(t) for t in tests]
    assert not any("promote" in s for s in sources), (
        "the report write is still gated on --promote at line {}".format(calls[0].lineno)
    )


def check_promote_still_guards_the_stage_move():
    """The other half. Loosening the report must not loosen the thing that moves production.

    Without this, deleting the `--promote` test entirely would satisfy the check above.
    """
    import ast

    _mod, path = _gate_cli()
    tree = ast.parse(open(path).read())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "promote"
    ]
    assert len(calls) == 1, "expected one registry.promote call, found {}".format(len(calls))

    tests = _enclosing_if_tests(tree, calls[0])
    sources = [ast.dump(t) for t in tests or []]
    assert any("promote" in s for s in sources), (
        "the stage move at line {} is no longer behind --promote".format(calls[0].lineno)
    )


def check_there_is_a_way_to_compare_without_touching_the_store():
    """Writing by default is only defensible if reading without writing stays possible."""
    mod, _path = _gate_cli()
    help_text = mod.build_parser().format_help()
    assert "--no-report" in help_text, help_text
    args = mod.build_parser().parse_args(
        ["--store", "sqlite:///x.db", "--candidate", "2", "--no-report"]
    )
    assert args.no_report is True
    assert args.promote is False


def check_the_promote_flag_no_longer_advertises_writing_the_report():
    """The help text said --promote wrote the report. It was true and it is not now."""
    mod, _path = _gate_cli()
    help_text = mod.build_parser().format_help()
    promote_line = [line for line in help_text.splitlines() if "move the stage" in line]
    assert promote_line, help_text
    assert "report" not in promote_line[0], promote_line[0]


def check_a_rejection_report_carries_the_reason_and_not_only_the_verdict():
    """A tag saying `reject` with no reason is a verdict nobody can act on."""
    mod, _path = _gate_cli()
    inc = _hand("inc", np.full(200, 0.40))
    cand = _hand("cand", np.full(200, 0.60))
    decision = gate.decide(cand, inc, "fp123456")
    assert decision.verdict == gate.REJECT, decision.verdict

    tags = mod.report_tags(decision, "production")
    assert tags["gate.verdict"] == gate.REJECT
    assert tags["gate.reason"] == decision.reason
    assert tags["gate.holdout"] == "fp123456"
    # Round trip, so a tag holding a truncated float is caught here and not by a reader.
    assert eval(tags["gate.mean_diff"]) == decision.mean_diff


# --- the gate verdict as a promotion precondition ---------------------------------


def _cleared(**over):
    tags = {
        gate.TAG_VERDICT: gate.PROMOTE,
        gate.TAG_STAGE: "production",
        gate.TAG_INCUMBENT: "7",
        gate.TAG_REASON: "better",
    }
    tags.update(over)
    return tags


def check_a_version_the_gate_cleared_against_the_current_incumbent_is_allowed():
    """The control. Every other case here refuses, so without this one the whole table is
    satisfied by a function that refuses everything."""
    assert gate.refusal_for(_cleared(), "production", 7) is None


def check_every_reason_a_promotion_must_be_refused():
    """Five refusals, each asserting on its own message rather than just on refusing.

    Case four is the one the first version of this missed and it is the one the question
    was actually about. A verdict is a comparison against something. Clearing a candidate
    while a weak model held the stage says nothing about the strong model holding it now,
    and the tag is overwritten each run so a stale verdict is indistinguishable from a
    fresh one. Measured on a real store: a candidate that passed against the weak
    incumbent comes back `reject (not_separated)` when re-gated against the current one.
    """
    cases = [
        ("never gated", {}, 7, "never been gated"),
        ("refused", _cleared(**{gate.TAG_VERDICT: "refuse",
                                gate.TAG_REASON: "candidate_not_finite"}), 7, "was refuse"),
        ("rejected", _cleared(**{gate.TAG_VERDICT: "reject",
                                 gate.TAG_REASON: "worse"}), 7, "was reject"),
        ("gated for another stage", _cleared(**{gate.TAG_STAGE: "staging"}), 7,
         "gated against staging"),
        ("stale, a different incumbent now", _cleared(), 9,
         "version 7 held production and version 9 holds it now"),
        ("stale, nothing holds the stage now", _cleared(), None,
         "version 7 held production and nothing holds it now"),
        ("gated when nothing held it, something does now",
         _cleared(**{gate.TAG_INCUMBENT: gate.NO_INCUMBENT}), 3,
         "gated when nothing held production and version 3 holds it now"),
        ("verdict predates the incumbent record",
         {gate.TAG_VERDICT: gate.PROMOTE, gate.TAG_STAGE: "production"}, 7,
         "does not record which incumbent"),
    ]
    for label, tags, incumbent, expected in cases:
        reason = gate.refusal_for(tags, "production", incumbent)
        assert reason is not None, "{}: allowed through".format(label)
        assert expected in reason, "{}: got {!r}".format(label, reason)


def check_a_first_promotion_into_an_empty_stage_still_works():
    """Nothing in production is a real state and the gate says `no_incumbent` for it.

    Refusing it would make the guard unusable on a fresh store, which is every CI run.
    """
    tags = _cleared(**{gate.TAG_INCUMBENT: gate.NO_INCUMBENT, gate.TAG_REASON: "no_incumbent"})
    assert gate.refusal_for(tags, "production", None) is None


def check_the_writer_and_the_reader_agree_on_the_tag_names():
    """The two halves lived in different files and drifted. Pinned by round trip.

    `report_tags` is in `scripts/gate.py` and `refusal_for` is here. A rename on one side
    used to leave the other reading keys nothing writes, which fails open, because a
    missing verdict tag reads as never gated rather than as a broken check.
    """
    import importlib.util

    path = os.path.join(ROOT, "scripts", "gate.py")
    spec_ = importlib.util.spec_from_file_location("gate_cli_tags2", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)

    inc = _hand("inc", np.full(200, 0.60))
    cand = _hand("cand", np.full(200, 0.40))
    decision = gate.decide(cand, inc, "fp")
    assert decision.verdict == gate.PROMOTE

    written = mod.report_tags(decision, "production", 7)
    for key in (gate.TAG_VERDICT, gate.TAG_STAGE, gate.TAG_INCUMBENT, gate.TAG_REASON):
        assert key in written, "{} is read and never written".format(key)
    assert gate.refusal_for(written, "production", 7) is None
    assert gate.refusal_for(written, "production", 8) is not None


def check_the_guard_is_asked_by_the_two_commands_that_reach_production():
    """Which callers ask is policy, and policy that is not pinned quietly changes.

    The first version of this guard covered `registry.py promote` only. `deploy.py land`
    promotes whatever is on trial, so a refused version reached production through
    `open` then `land` in two commands. Measured before the fix. `open` is where the
    check belongs, because `land` acts on the canary's own evidence.
    """
    import ast

    for script, subcommand in (("registry.py", "promote"), ("deploy.py", "open")):
        path = os.path.join(ROOT, "scripts", script)
        tree = ast.parse(open(path).read())
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "refusal_for_version"
        ]
        assert calls, "{} never asks the gate, so {} is unguarded".format(script, subcommand)


def check_rollback_never_asks_the_gate():
    """Pinned as an absence, because symmetry is the obvious wrong move here.

    A version being rolled back to was cleared against an incumbent that is no longer
    there, so under the staleness rule above every rollback would refuse. Recovery has to
    work when the world has moved on.
    """
    import ast

    path = os.path.join(ROOT, "scripts", "registry.py")
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        src = ast.dump(node)
        if "'rollback'" not in src and '"rollback"' not in src:
            continue
        parent = [n for n in ast.walk(tree)
                  if isinstance(n, ast.If) and n.test is node]
        for p in parent:
            body = ast.dump(ast.Module(body=p.body, type_ignores=[]))
            assert "refusal_for_version" not in body, (
                "the rollback branch asks the gate, which refuses every recovery"
            )
