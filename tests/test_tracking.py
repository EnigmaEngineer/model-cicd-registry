"""Checks for the MLflow layer.

Every check here talks to a real sqlite tracking store in a temporary directory. None of
them mock the client. A mocked store would agree with whatever this module believes about
MLflow, and on an earlier project of mine the thing that went wrong was exactly that belief.

Importing this module needs mlflow, so it is collected by tests/run_with_mlflow.py rather
than by tests/run_all.py.
"""

from __future__ import annotations

import os
import shutil
import tempfile

from mcr import tracking
from mcr import train as train_mod
from mcr.config import ConfigError, load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "configs", "baseline.yml")


class _Store:
    """A throwaway sqlite tracking store, removed on exit."""

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="mcr-test-")
        self.cli = tracking.client("sqlite:///{}".format(os.path.join(self.dir, "t.db")))
        return self.cli

    def __exit__(self, *exc):
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def check_param_keys_are_derived_and_do_not_collide():
    keys = tracking.param_keys()
    assert len(set(keys)) == len(keys)
    # Dotted, so a field name declared by two sections cannot overwrite itself. n_features
    # is the one that would be at risk if ModelConfig ever grew it.
    assert "data.n_features" in keys
    assert all(k.count(".") <= 1 for k in keys)


def check_params_for_and_config_from_params_are_inverses():
    cfg = load(BASELINE)
    back = tracking.config_from_params(tracking.params_for(cfg))
    assert back == cfg, "{} != {}".format(back, cfg)
    assert back.fingerprint() == cfg.fingerprint()


def check_the_round_trip_restores_types_and_not_only_values():
    """A param comes back as a string and the fingerprint is type sensitive.

    This is the check that would have caught docs/adr-0002 before it shipped. The
    assertion is on the type rather than on equality, because `1 == 1.0` in Python and an
    equality check passes while the fingerprint moves.
    """
    cfg = load(BASELINE)
    back = tracking.config_from_params(tracking.params_for(cfg))
    assert isinstance(back.data.noise, float), type(back.data.noise)
    assert isinstance(back.model.epochs, int), type(back.model.epochs)
    assert isinstance(back.seed, int), type(back.seed)
    assert not isinstance(back.model.epochs, bool)


def check_a_missing_param_is_refused_rather_than_defaulted():
    cfg = load(BASELINE)
    for dropped in ("model.epochs", "data.n_rows", "seed", "name"):
        thin = dict(tracking.params_for(cfg))
        thin.pop(dropped)
        try:
            tracking.config_from_params(thin)
        except tracking.TrackingError as exc:
            assert dropped in str(exc), str(exc)
        else:
            raise AssertionError("rebuilt a config with {} missing".format(dropped))


def check_a_junk_param_value_is_refused():
    cfg = load(BASELINE)
    bad = dict(tracking.params_for(cfg))
    bad["data.noise"] = "nine"
    try:
        tracking.config_from_params(bad)
    except ConfigError as exc:
        assert "noise" in str(exc)
    else:
        raise AssertionError("accepted a non numeric noise")


def check_an_extra_param_does_not_stop_the_rebuild():
    """MLflow runs carry params this module did not write, and the registry will add
    more. Refusing unknown params would make the recovery path break every time anything
    else logs anything."""
    cfg = load(BASELINE)
    extra = dict(tracking.params_for(cfg))
    extra["something.else"] = "7"
    assert tracking.config_from_params(extra) == cfg


def check_the_truncation_limits_are_the_measured_numbers():
    """The limits are golden values and they have to be asserted as literals.

    Every other check in this file spells the limit as `tracking.PARAM_VALUE_LIMIT`, so a
    mutant moving the constant moves both sides of the comparison and survives. It did.
    These two numbers were measured against mlflow 3.16.0 by logging values of increasing
    length and watching where the warning appeared, and they are the point at which the
    value is silently shortened rather than refused.
    """
    assert tracking.PARAM_VALUE_LIMIT == 6000, tracking.PARAM_VALUE_LIMIT
    assert tracking.TAG_VALUE_LIMIT == 8000, tracking.TAG_VALUE_LIMIT


def check_a_value_long_enough_to_be_truncated_is_refused():
    """A truncated content hash is a registry key pointing at nothing, so it has to raise
    here rather than resting on the headroom a 64 character hash happens to have."""
    try:
        tracking._short_enough("k", "x" * (tracking.PARAM_VALUE_LIMIT + 1),
                               tracking.PARAM_VALUE_LIMIT)
    except tracking.TrackingError as exc:
        assert "truncates" in str(exc)
    else:
        raise AssertionError("accepted a value past the truncation limit")

    # And the boundary from the other side, because a limit tested only from outside says
    # nothing about whether it is off by one.
    at_limit = "x" * tracking.PARAM_VALUE_LIMIT
    assert tracking._short_enough("k", at_limit, tracking.PARAM_VALUE_LIMIT) == at_limit


def check_params_for_writes_exactly_the_keys_param_keys_names():
    """Asserted from outside the module rather than guarded inside it.

    The guard that used to live in `params_for` compared these two against each other, and
    both derive from the same walk over the same dict, so it could not fail. Here the
    expected list is written out, which is a different source, so the two can disagree.
    """
    expected = [
        "data.holdout_frac", "data.n_features", "data.n_rows", "data.noise",
        "data.positive_rate",
        "model.epochs", "model.init", "model.l2", "model.learning_rate",
        "name", "seed",
    ]
    assert sorted(tracking.param_keys()) == expected, sorted(tracking.param_keys())
    assert sorted(tracking.params_for(load(BASELINE))) == expected


def check_the_dot_is_what_keeps_two_sections_apart():
    """Two sections declaring one field name must not overwrite each other.

    `n_features` lives on DataConfig today. If ModelConfig ever grew it, a flat namespace
    would keep one value and drop the other silently. Forced here by pointing both sections
    at one dataclass, which is the situation a flat namespace cannot survive.
    """
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Same:
        n_rows: int

    original = dict(tracking._SECTIONS)
    try:
        tracking._SECTIONS.clear()
        tracking._SECTIONS.update({"data": Same, "model": Same})
        keys = tracking.param_keys()
        assert keys == ["name", "seed", "data.n_rows", "model.n_rows"], keys
        assert len(set(keys)) == len(keys)
    finally:
        tracking._SECTIONS.clear()
        tracking._SECTIONS.update(original)

    assert tracking.param_keys()[:2] == ["name", "seed"]


def check_the_default_run_name_is_the_config_and_the_hash():
    """`run_name or derived` had `or` flipped to `and` by a mutant and nothing noticed.

    With `and` an explicit run name is thrown away and the default becomes None, so the
    whole parameter was untested. The name is cosmetic and nothing keys on it, which is
    exactly why it needed a check rather than trust.
    """
    cfg = load(BASELINE)
    result = train_mod.run(cfg)
    with _Store() as cli:
        default = tracking.log_training_run(
            cli, cfg, result.content_hash, result.metrics, experiment="t"
        )
        asked = tracking.log_training_run(
            cli, cfg, result.content_hash, result.metrics, experiment="t",
            run_name="chosen-by-hand",
        )
        names = {
            r: cli.get_run(r).data.tags["mlflow.runName"] for r in (default, asked)
        }
    assert names[default] == "baseline-{}".format(result.content_hash[:8]), names
    assert names[asked] == "chosen-by-hand", names


def check_recovered_is_frozen():
    """It is handed to the promotion gate, and a gate that edits its own input is the
    kind of thing nobody looks for until a number is wrong."""
    cfg = load(BASELINE)
    got = tracking.Recovered(
        run_id="r", config=cfg, artifact_hash="h",
        config_fingerprint=cfg.fingerprint(), metrics={},
    )
    try:
        got.artifact_hash = "something else"
    except Exception:
        return
    raise AssertionError("Recovered let a field be reassigned")


def check_a_logged_run_recovers_to_the_config_that_wrote_it():
    cfg = load(BASELINE)
    result = train_mod.run(cfg)
    with _Store() as cli:
        rid = tracking.log_training_run(
            cli, cfg, result.content_hash, result.metrics, experiment="t"
        )
        got = tracking.recover(cli, rid)
    assert got.config == cfg
    assert got.artifact_hash == result.content_hash
    assert got.config_fingerprint == cfg.fingerprint()
    assert got.run_id == rid


def check_metrics_survive_the_store_exactly():
    """The two shipped configs differ on holdout log loss by 1.806e-11. A store that
    rounded anywhere would hand the promotion gate two equal numbers and a false tie."""
    cfg = load(BASELINE)
    result = train_mod.run(cfg)
    with _Store() as cli:
        rid = tracking.log_training_run(
            cli, cfg, result.content_hash, result.metrics, experiment="t"
        )
        got = tracking.recover(cli, rid)
    assert set(got.metrics) == set(result.metrics)
    for key, value in result.metrics.items():
        assert got.metrics[key] == value, "{} {!r} != {!r}".format(
            key, got.metrics[key], value
        )


def check_recover_refuses_a_run_whose_tag_and_params_disagree():
    """The one failure mode that would poison everything downstream quietly.

    If the fingerprint tag and the params ever describe different runs, a gate reading the
    store compares an incumbent that never existed. Built by overwriting the tag, which
    MLflow allows, which is itself why the recipe lives in params instead.
    """
    cfg = load(BASELINE)
    result = train_mod.run(cfg)
    with _Store() as cli:
        rid = tracking.log_training_run(
            cli, cfg, result.content_hash, result.metrics, experiment="t"
        )
        cli.set_tag(rid, tracking.TAG_CONFIG_FINGERPRINT, "000000000000")
        try:
            tracking.recover(cli, rid)
        except tracking.TrackingError as exc:
            assert "000000000000" in str(exc), str(exc)
        else:
            raise AssertionError("recovered a run whose tag contradicts its params")


def check_recover_refuses_a_run_with_no_tags_at_all():
    """A run created by anything other than log_training_run. The store is shared and
    nothing stops another process writing into the same experiment."""
    with _Store() as cli:
        run = cli.create_run(experiment_id=tracking.experiment_id(cli, "t"))
        try:
            tracking.recover(cli, run.info.run_id)
        except tracking.TrackingError as exc:
            assert tracking.TAG_CONFIG_FINGERPRINT in str(exc)
        else:
            raise AssertionError("recovered a run with no fingerprint tag")


def check_two_runs_of_one_recipe_are_two_runs_under_one_fingerprint():
    cfg = load(BASELINE)
    result = train_mod.run(cfg)
    with _Store() as cli:
        ids = [
            tracking.log_training_run(
                cli, cfg, result.content_hash, result.metrics, experiment="t"
            )
            for _ in range(2)
        ]
        found = tracking.runs_for_fingerprint(cli, cfg.fingerprint(), experiment="t")
        other = tracking.runs_for_fingerprint(cli, "cafecafecafe", experiment="t")
    assert len(set(ids)) == 2, "two runs shared a run id"
    assert sorted(found) == sorted(ids), "{} != {}".format(found, ids)
    assert other == [], "a fingerprint nobody logged returned {}".format(other)


def check_a_param_cannot_be_rewritten_but_a_tag_can():
    """The measurement the params and tags split rests on.

    If MLflow ever allowed a param to be overwritten, the recipe would stop being immutable
    and the whole reason for putting it there instead of in tags would be gone. Worth
    pinning, because it is a property of a dependency rather than of this repo.
    """
    from mlflow.exceptions import MlflowException

    with _Store() as cli:
        run = cli.create_run(experiment_id=tracking.experiment_id(cli, "t"))
        rid = run.info.run_id
        cli.log_param(rid, "lr", "0.5")
        try:
            cli.log_param(rid, "lr", "1.5")
        except MlflowException:
            pass
        else:
            raise AssertionError("MLflow allowed a param to be rewritten")

        # Same value again is not a rewrite and must stay allowed, because logging is
        # retried on a flaky store and an idempotent write cannot be an error.
        cli.log_param(rid, "lr", "0.5")

        cli.set_tag(rid, "stage", "staging")
        cli.set_tag(rid, "stage", "production")
        assert cli.get_run(rid).data.tags["stage"] == "production"


def check_the_experiment_is_created_once_and_reused():
    with _Store() as cli:
        first = tracking.experiment_id(cli, "t")
        second = tracking.experiment_id(cli, "t")
        assert first == second
        assert tracking.experiment_id(cli, "other") != first
