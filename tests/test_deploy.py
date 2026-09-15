"""Checks for the deployment state, against a real sqlite backed MLflow store.

Same fixture shape as tests/test_registry.py and for the same two reasons. One store for
the module keeps a mutation pass inside the time a call has, and a store holding several
models is the only fixture that can tell a correct name filter from a missing one.

The fixtures here put production and canary on different versions wherever it is possible
to do so. A deployment whose two aliases hold the same number cannot tell an operation that
reads the right one from an operation that reads either, and the land path moves one alias
and removes the other, which is exactly the shape that mistake hides in.
"""

from __future__ import annotations

import atexit
import itertools
import shutil
import tempfile

import mlflow

from mcr import deploy, registry

_SHARED = {}
_COUNTER = itertools.count()


class Store:
    def __enter__(self):
        if "cli" not in _SHARED:
            _SHARED["dir"] = tempfile.mkdtemp(prefix="mcr-deploy-")
            uri = "sqlite:///{}/deploy.db".format(_SHARED["dir"])
            _SHARED["cli"] = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        return _SHARED["cli"], "deploy-model-{}".format(next(_COUNTER))

    def __exit__(self, *exc):
        return False


def _drop_the_store():
    if "dir" in _SHARED:
        shutil.rmtree(_SHARED["dir"], ignore_errors=True)
        _SHARED.clear()


atexit.register(_drop_the_store)


def a_run(cli) -> str:
    found = cli.get_experiment_by_name("deploy-checks")
    exp = found.experiment_id if found else cli.create_experiment("deploy-checks")
    run = cli.create_run(experiment_id=exp)
    cli.set_terminated(run.info.run_id, status="FINISHED")
    return run.info.run_id


def ready(cli, model, versions=4, production="1"):
    """Versions registered and production pointed somewhere, which most checks need."""
    for i in range(versions):
        registry.register(
            cli, model, a_run(cli), "{:02d}{}".format(i, "d" * 62), "fp{}".format(i)
        )
    if production is not None:
        registry.promote(cli, model, production, deploy.PRODUCTION)
    return cli, model


def check_opening_a_canary_records_the_version_and_the_share():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        state = deploy.open_canary(cli, MODEL, "3", 0.25)
        assert state.production == 1, state
        assert state.canary == 3, state
        assert state.fraction == 0.25, state
        assert state.is_canarying()


def check_the_share_survives_a_read_through_the_store():
    """The tag comes back as a string and a caller comparing it to a float would be wrong
    on every reading."""
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        deploy.open_canary(cli, MODEL, "2", 0.05)
        again = deploy.deployment(cli, MODEL)
        assert isinstance(again.fraction, float), type(again.fraction)
        assert again.fraction == 0.05, again.fraction


def check_a_fraction_that_does_not_parse_is_refused():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        cli.set_registered_model_tag(MODEL, deploy.FRACTION_TAG, "a quarter")
        try:
            deploy.deployment(cli, MODEL)
        except deploy.DeployError as exc:
            assert "not a number" in str(exc), exc
            return
        raise AssertionError("a non numeric share was accepted")


def check_the_smallest_allowed_share_is_accepted():
    """The largest rejected value is checked below. A validator tested only well away from
    its own limit does not pin the limit, which cost eleven mutants on an earlier day."""
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        state = deploy.open_canary(cli, MODEL, "2", deploy.MIN_FRACTION)
        assert state.fraction == deploy.MIN_FRACTION


def check_the_largest_allowed_share_is_accepted():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        state = deploy.open_canary(cli, MODEL, "2", deploy.MAX_FRACTION)
        assert state.fraction == deploy.MAX_FRACTION


def check_a_share_just_under_the_floor_is_refused():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        try:
            deploy.open_canary(cli, MODEL, "2", deploy.MIN_FRACTION / 2)
        except deploy.DeployError as exc:
            assert "outside" in str(exc), exc
            return
        raise AssertionError("a share below the floor was accepted")


def check_a_share_just_over_the_ceiling_is_refused():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        try:
            deploy.open_canary(cli, MODEL, "2", deploy.MAX_FRACTION + 0.01)
        except deploy.DeployError as exc:
            assert "outside" in str(exc), exc
            return
        raise AssertionError("a share above the ceiling was accepted")


def check_canarying_the_production_version_is_refused():
    """Both arms would be the same bytes, which mcr/canary.py already refuses at scoring
    time. Refusing it here means the store never enters the state at all."""
    with Store() as (cli, MODEL):
        ready(cli, MODEL, production="2")
        try:
            deploy.open_canary(cli, MODEL, "2", 0.1)
        except deploy.DeployError as exc:
            assert "already in production" in str(exc), exc
            return
        raise AssertionError("canarying the production version was allowed")


def check_a_second_canary_is_refused():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        deploy.open_canary(cli, MODEL, "2", 0.1)
        try:
            deploy.open_canary(cli, MODEL, "3", 0.2)
        except deploy.DeployError as exc:
            assert "already canarying" in str(exc), exc
            return
        raise AssertionError("two canaries at once were allowed")


def check_opening_a_canary_with_nothing_in_production_is_refused():
    with Store() as (cli, MODEL):
        ready(cli, MODEL, production=None)
        try:
            deploy.open_canary(cli, MODEL, "2", 0.1)
        except deploy.DeployError as exc:
            assert "no control arm" in str(exc), exc
            return
        raise AssertionError("a canary was opened with no control")


def check_aborting_leaves_production_exactly_where_it_was():
    """The deleted --rollback flag moved production in response to a verdict about the
    canary. This is the check that the replacement does not."""
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        registry.promote(cli, MODEL, "2", deploy.PRODUCTION)
        deploy.open_canary(cli, MODEL, "4", 0.25)
        state = deploy.abort_canary(cli, MODEL)
        assert state.production == 2, state
        assert state.canary is None, state
        assert state.fraction is None, state


def check_aborting_writes_a_retirement_to_the_log():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        deploy.open_canary(cli, MODEL, "3", 0.1)
        deploy.abort_canary(cli, MODEL)
        log = registry.history(cli, MODEL, deploy.CANARY)
        assert [(e.to_version, e.kind) for e in log] == [
            (3, registry.PROMOTE),
            (None, registry.RETIRE),
        ], log


def check_aborting_nothing_is_refused():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        try:
            deploy.abort_canary(cli, MODEL)
        except deploy.DeployError as exc:
            assert "not canarying" in str(exc), exc
            return
        raise AssertionError("aborting with no canary was allowed")


def check_aborting_clears_a_share_left_by_a_half_finished_open():
    """open_canary writes the share before the alias, so a crash between them leaves a
    share with no canary. Abort is the repair for that as well as the normal ending."""
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        cli.set_registered_model_tag(MODEL, deploy.FRACTION_TAG, "0.25")
        assert deploy.check_consistency(cli, MODEL) is not None
        state = deploy.abort_canary(cli, MODEL)
        assert state.fraction is None, state
        assert deploy.check_consistency(cli, MODEL) is None


def check_landing_moves_production_and_ends_the_trial():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        deploy.open_canary(cli, MODEL, "3", 0.25)
        state = deploy.land_canary(cli, MODEL)
        assert state.production == 3, state
        assert state.canary is None, state
        assert state.fraction is None, state


def check_landing_leaves_a_rollback_target_behind_it():
    """A landed canary is a promote like any other, so production can still walk back."""
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        registry.promote(cli, MODEL, "2", deploy.PRODUCTION)
        deploy.open_canary(cli, MODEL, "4", 0.1)
        deploy.land_canary(cli, MODEL)
        assert registry.rollback_target(cli, MODEL, deploy.PRODUCTION) == 2


def check_landing_nothing_is_refused():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        try:
            deploy.land_canary(cli, MODEL)
        except deploy.DeployError as exc:
            assert "not canarying" in str(exc), exc
            return
        raise AssertionError("landing with no canary was allowed")


def check_a_clean_deployment_is_consistent():
    """The control for every check below. Without it they all pass on a function that
    returns a complaint about everything."""
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        assert deploy.check_consistency(cli, MODEL) is None
        deploy.open_canary(cli, MODEL, "2", 0.1)
        assert deploy.check_consistency(cli, MODEL) is None


def check_a_canary_with_no_share_is_named():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        deploy.open_canary(cli, MODEL, "2", 0.1)
        cli.delete_registered_model_tag(MODEL, deploy.FRACTION_TAG)
        problem = deploy.check_consistency(cli, MODEL)
        assert problem is not None and "no fraction" in problem, problem


def check_both_aliases_on_one_version_is_named():
    """What a crash between the two halves of land_canary leaves behind."""
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        deploy.open_canary(cli, MODEL, "3", 0.1)
        registry.promote(cli, MODEL, "3", deploy.PRODUCTION)
        problem = deploy.check_consistency(cli, MODEL)
        assert problem is not None and "both production and canary" in problem, problem


def check_repairing_that_state_does_not_move_production():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        deploy.open_canary(cli, MODEL, "3", 0.1)
        registry.promote(cli, MODEL, "3", deploy.PRODUCTION)
        deploy.abort_canary(cli, MODEL)
        assert registry.current(cli, MODEL, deploy.PRODUCTION) == 3
        assert deploy.check_consistency(cli, MODEL) is None


def check_describe_says_what_is_deployed():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        assert deploy.deployment(cli, MODEL).describe() == (
            "production on version 1, no canary"
        )
        deploy.open_canary(cli, MODEL, "2", 0.25)
        assert deploy.deployment(cli, MODEL).describe() == (
            "production on version 1, canary on version 2 at 0.2500"
        )


def check_describe_handles_an_empty_deployment():
    with Store() as (cli, MODEL):
        ready(cli, MODEL, production=None)
        assert deploy.deployment(cli, MODEL).describe() == "nothing in production"


def check_is_canarying_is_false_with_no_canary():
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        assert not deploy.deployment(cli, MODEL).is_canarying()


def check_a_deployment_cannot_be_edited_after_it_is_read():
    """A Deployment is a reading taken at a moment. Letting a caller adjust one means a
    stale reading can be made to look current, and then check_consistency is answering
    about something that was never in the store.

    A mutant flipping `frozen=True` off has now survived on three days running, on three
    different frozen dataclasses. Two lines is cheaper than accepting it a fourth time.
    """
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        state = deploy.deployment(cli, MODEL)
        try:
            state.production = 99
        except Exception:
            return
        raise AssertionError("a Deployment accepted an edit")


def check_an_empty_deployment_is_consistent():
    """Nothing deployed is a legal state, not a broken one.

    This is the check a mutant found missing. With no canary and nothing in production,
    flipping the canary test in check_consistency from `is not None` to `is None` makes an
    empty store report a canary on version None, and every other check passed anyway
    because none of them ever looked at an empty store.
    """
    with Store() as (cli, MODEL):
        ready(cli, MODEL, production=None)
        state = deploy.deployment(cli, MODEL)
        assert state.production is None and state.canary is None, state
        assert deploy.check_consistency(cli, MODEL) is None


def check_a_canary_with_nothing_in_production_is_named():
    """open_canary refuses to create this state, so it is built by hand.

    A branch that the shipped operations cannot reach is still worth checking when a repair
    tool reads it. Somebody pointing the canary alias by hand, or a retirement of production
    while a canary is up, both land here.
    """
    with Store() as (cli, MODEL):
        ready(cli, MODEL)
        deploy.open_canary(cli, MODEL, "2", 0.1)
        registry.retire(cli, MODEL, deploy.PRODUCTION)
        problem = deploy.check_consistency(cli, MODEL)
        assert problem is not None, "a canary with no control arm was called consistent"
        assert "nothing in production" in problem, problem
