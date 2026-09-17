"""Checks for the registry, against a real sqlite backed MLflow store.

Nothing is mocked. The whole subject of this module is what the store does when you ask it
something awkward, so a fake store would be checking my idea of MLflow rather than MLflow.

Note where the fixtures are deliberately awkward. A history with two entries cannot test a
rule about which entry to read, and a set of versions where the newest is also the last
promoted cannot test a rollback that has to ignore the newest. Both of those were arms in
the probe that passed for the wrong reason before the fixture was fixed.
"""

from __future__ import annotations

import atexit
import dataclasses
import itertools
import json
import os
import shutil
import tempfile
import time

import mlflow

from mcr import registry
from mcr.tracking import TAG_ARTIFACT_HASH, TAG_CONFIG_FINGERPRINT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SHARED = {}
_COUNTER = itertools.count()


class Store:
    """One sqlite store for the whole module, and a fresh model name per check.

    The first version built a store per check. It was honest and it cost 41 seconds for 35
    checks, because MLflow creates and migrates the schema on every new sqlite file, and a
    mutation pass over this module at that price does not fit in the time available.

    Sharing the store is not only cheaper, it is a better fixture. Every function here
    takes a model name and filters on it, and a store holding exactly one model cannot
    tell a correct filter from a missing one. Every check writes under its own name into
    one database, so a `search_model_versions` that dropped its name clause fails at once.

    There is no module level model name on purpose. A check that reached for a shared one
    would see whatever an earlier check left behind, and the failure would depend on the
    order the collector happened to walk.
    """

    def __enter__(self):
        if "cli" not in _SHARED:
            _SHARED["dir"] = tempfile.mkdtemp(prefix="mcr-registry-")
            uri = "sqlite:///{}/registry.db".format(_SHARED["dir"])
            _SHARED["cli"] = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        return _SHARED["cli"], "check-model-{}".format(next(_COUNTER))

    def __exit__(self, *exc):
        return False


def _drop_the_store():
    if "dir" in _SHARED:
        shutil.rmtree(_SHARED["dir"], ignore_errors=True)
        _SHARED.clear()


# Registered rather than called by the runner, because the runner does not know this
# module exists and a cleanup nobody calls is a cleanup that does not happen.
atexit.register(_drop_the_store)


def a_run(cli) -> str:
    found = cli.get_experiment_by_name("checks")
    exp = found.experiment_id if found else cli.create_experiment("checks")
    run = cli.create_run(experiment_id=exp)
    cli.set_terminated(run.info.run_id, status="FINISHED")
    return run.info.run_id


def some_versions(cli, model: str, n: int):
    return [
        registry.register(
            cli, model, a_run(cli), "{:02d}{}".format(i, "f" * 62), "fp{}".format(i)
        )
        for i in range(n)
    ]


def check_register_returns_increasing_versions():
    with Store() as (cli, MODEL):
        got = some_versions(cli, MODEL, 3)
        assert got == [1, 2, 3], got


def check_register_is_idempotent_on_the_artefact_hash():
    with Store() as (cli, MODEL):
        digest = "a" * 64
        first = registry.register(cli, MODEL, a_run(cli), digest, "fp")
        second = registry.register(cli, MODEL, a_run(cli), digest, "fp")
        assert first == second, (first, second)
        assert len(registry.versions(cli, MODEL)) == 1


def check_register_tags_both_identities():
    with Store() as (cli, MODEL):
        registry.register(cli, MODEL, a_run(cli), "b" * 64, "fp0")
        tags = cli.get_model_version(MODEL, "1").tags
        assert tags[TAG_ARTIFACT_HASH] == "b" * 64
        assert tags[TAG_CONFIG_FINGERPRINT] == "fp0"


def check_two_runs_of_one_recipe_are_one_version():
    """The same bytes from two occasions. The registry holds models, not occasions."""
    with Store() as (cli, MODEL):
        digest = "c" * 64
        a = registry.register(cli, MODEL, a_run(cli), digest, "fp")
        b = registry.register(cli, MODEL, a_run(cli), digest, "fp")
        assert a == b
        assert len(registry.versions(cli, MODEL)) == 1


def check_versions_come_back_sorted():
    """search_model_versions does not sort. It returned [3, 1] in one session."""
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 5)
        got = [int(v.version) for v in registry.versions(cli, MODEL)]
        assert got == sorted(got), got
        assert got == [1, 2, 3, 4, 5], got


def check_resolve_takes_a_version_number():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        assert registry.resolve(cli, MODEL, "2") == 2


def check_resolve_takes_a_stage():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        registry.promote(cli, MODEL, "2", "production")
        assert registry.resolve(cli, MODEL, "production") == 2


def check_resolve_refuses_a_reference_that_names_two_models():
    """A decimal alias colliding with a version number. MLflow allows it."""
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        cli.set_registered_model_alias(MODEL, "2", "3")
        by_version = cli.get_model_version(MODEL, "2").run_id
        by_alias = cli.get_model_version_by_alias(MODEL, "2").run_id
        assert by_version != by_alias, "the fixture did not build the collision"
        try:
            registry.resolve(cli, MODEL, "2")
        except registry.RegistryError as exc:
            assert "two models" in str(exc), str(exc)
        else:
            raise AssertionError("resolve returned a model for an ambiguous reference")


def check_resolve_refuses_a_version_that_is_not_there():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        try:
            registry.resolve(cli, MODEL, "9")
        except registry.RegistryError as exc:
            assert "no version 9" in str(exc), str(exc)
        else:
            raise AssertionError("resolved version 9 of a model with two versions")


def check_resolve_refuses_an_unknown_stage_name():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        try:
            registry.resolve(cli, MODEL, "canary")
        except registry.RegistryError as exc:
            assert "nothing is aliased" in str(exc), str(exc)
        else:
            raise AssertionError("resolved a stage nothing points at")


def check_promote_refuses_a_stage_this_project_does_not_have():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        try:
            registry.promote(cli, MODEL, "1", "Production")
        except registry.RegistryError as exc:
            assert "unknown stage" in str(exc), str(exc)
        else:
            raise AssertionError("promoted to a stage that is not in STAGES")


def check_promote_moves_the_alias_and_takes_it_off_the_old_version():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        registry.promote(cli, MODEL, "1", "production")
        registry.promote(cli, MODEL, "2", "production")
        assert list(cli.get_model_version(MODEL, "1").aliases) == []
        assert list(cli.get_model_version(MODEL, "2").aliases) == ["production"]
        assert registry.current(cli, MODEL, "production") == 2


def check_current_is_none_before_anything_is_promoted():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        assert registry.current(cli, MODEL, "production") is None


def check_staging_and_production_do_not_share_a_log():
    """Two stages, interleaved, and each one's history has to hold only its own."""
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 4)
        registry.promote(cli, MODEL, "1", "production")
        registry.promote(cli, MODEL, "2", "staging")
        registry.promote(cli, MODEL, "3", "production")
        registry.promote(cli, MODEL, "4", "staging")

        prod = [e.to_version for e in registry.history(cli, MODEL, "production")]
        stag = [e.to_version for e in registry.history(cli, MODEL, "staging")]
        assert prod == [1, 3], prod
        assert stag == [2, 4], stag
        assert registry.current(cli, MODEL, "production") == 3
        assert registry.current(cli, MODEL, "staging") == 4


def check_the_log_records_where_it_came_from():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        registry.promote(cli, MODEL, "1", "production")
        registry.promote(cli, MODEL, "3", "production")
        log = registry.history(cli, MODEL, "production")
        assert [(e.from_version, e.to_version) for e in log] == [(None, 1), (1, 3)]


def check_the_log_stays_in_order_past_ten_entries():
    """The sequence number is a zero padded string and the tags sort lexically.

    Without the padding, entry 10 sorts before entry 9 and the history comes back in an
    order that is right for the first nine transitions and wrong after that. A fixture
    with fewer than ten entries cannot see it.
    """
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        expected = []
        for i in range(11):
            want = (i % 2) + 1
            registry.promote(cli, MODEL, str(want), "production")
            expected.append(want)
        got = [e.to_version for e in registry.history(cli, MODEL, "production")]
        assert got == expected, (got, expected)


def check_promoting_where_it_already_points_writes_nothing():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        registry.promote(cli, MODEL, "1", "production")
        again = registry.promote(cli, MODEL, "1", "production")
        assert again.logged is False
        assert len(registry.history(cli, MODEL, "production")) == 1


def check_rollback_target_is_none_on_a_first_promotion():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        registry.promote(cli, MODEL, "2", "production")
        assert registry.rollback_target(cli, MODEL, "production") is None


def check_rollback_target_is_none_when_nothing_is_deployed():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        assert registry.rollback_target(cli, MODEL, "production") is None


def check_rollback_target_ignores_a_newer_version_that_was_never_promoted():
    """The candidate a gate rejected is a high version number and not a rollback target."""
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 4)
        registry.promote(cli, MODEL, "1", "production")
        registry.promote(cli, MODEL, "2", "production")
        assert registry.rollback_target(cli, MODEL, "production") == 1


def check_rollback_moves_production_back():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 4)
        registry.promote(cli, MODEL, "1", "production")
        registry.promote(cli, MODEL, "3", "production")
        entry = registry.rollback(cli, MODEL, "production")
        assert entry.to_version == 1
        assert registry.current(cli, MODEL, "production") == 1
        assert [e.to_version for e in registry.history(cli, MODEL, "production")] == [1, 3, 1]


def check_rollback_twice_walks_back_twice():
    """Three promotions then two rollbacks. The second must not undo the first.

    This is the check that caught the real defect in this module. The obvious rule is
    "whatever the stage held immediately before". Under it the second rollback returns to
    version 3, because after rolling 3 down to 2 the thing production held before 2 really
    was 3. So rollback would put the version you just fled straight back into production.
    """
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        for v in ("1", "2", "3"):
            registry.promote(cli, MODEL, v, "production")
        registry.rollback(cli, MODEL, "production")
        assert registry.current(cli, MODEL, "production") == 2
        registry.rollback(cli, MODEL, "production")
        assert registry.current(cli, MODEL, "production") == 1


def check_rollback_runs_out_rather_than_looping():
    """Past the oldest version there is nothing left, and that is not version 2 again."""
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        for v in ("1", "2", "3"):
            registry.promote(cli, MODEL, v, "production")
        registry.rollback(cli, MODEL, "production")
        registry.rollback(cli, MODEL, "production")
        assert registry.rollback_target(cli, MODEL, "production") is None


def check_a_rollback_is_logged_as_a_rollback():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        registry.promote(cli, MODEL, "1", "production")
        registry.promote(cli, MODEL, "2", "production")
        entry = registry.rollback(cli, MODEL, "production")
        assert entry.kind == registry.ROLLBACK
        kinds = [e.kind for e in registry.history(cli, MODEL, "production")]
        assert kinds == [registry.PROMOTE, registry.PROMOTE, registry.ROLLBACK], kinds


def check_a_promotion_after_a_rollback_still_has_somewhere_to_go_back_to():
    """Deploy 3, roll back to 2, deploy 4. Rolling back from 4 goes to 2, never to 3."""
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 4)
        for v in ("1", "2", "3"):
            registry.promote(cli, MODEL, v, "production")
        registry.rollback(cli, MODEL, "production")
        registry.promote(cli, MODEL, "4", "production")
        assert registry.rollback_target(cli, MODEL, "production") == 2


def check_a_deliberate_redeploy_of_an_older_version_is_not_a_rollback():
    """Same shape as a rollback in the log and a different meaning, which is why kind is
    stored rather than worked out from the version numbers."""
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        registry.promote(cli, MODEL, "1", "production")
        registry.promote(cli, MODEL, "3", "production")
        registry.promote(cli, MODEL, "2", "production")
        assert registry.rollback_target(cli, MODEL, "production") == 3


def check_rollback_refuses_when_there_is_nothing_behind_it():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        registry.promote(cli, MODEL, "1", "production")
        try:
            registry.rollback(cli, MODEL, "production")
        except registry.RegistryError as exc:
            assert "no previous version" in str(exc), str(exc)
        else:
            raise AssertionError("rolled back a stage that has only ever held one version")


def check_rollback_refuses_when_the_alias_moved_outside_promote():
    """What the MLflow UI does to this module's account of the store."""
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        registry.promote(cli, MODEL, "1", "production")
        registry.promote(cli, MODEL, "2", "production")
        cli.set_registered_model_alias(MODEL, "production", "3")
        try:
            registry.rollback_target(cli, MODEL, "production")
        except registry.RegistryError as exc:
            assert "does not describe this store" in str(exc), str(exc)
        else:
            raise AssertionError("computed a rollback target off a log that had drifted")


def check_rollback_refuses_when_the_log_is_missing_entirely():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        cli.set_registered_model_alias(MODEL, "production", "1")
        try:
            registry.rollback_target(cli, MODEL, "production")
        except registry.RegistryError as exc:
            assert "log is empty" in str(exc), str(exc)
        else:
            raise AssertionError("computed a rollback target with no log at all")


def check_ensure_model_is_safe_to_call_twice():
    with Store() as (cli, MODEL):
        registry.ensure_model(cli, MODEL)
        registry.ensure_model(cli, MODEL)
        assert cli.get_registered_model(MODEL).name == MODEL


def check_transition_round_trips_through_its_stored_form():
    entry = registry.Transition(
        stage="production", to_version=7, from_version=3, at_ms=1757000000000
    )
    body = json.loads(entry.as_value())
    assert body == {
        "stage": "production",
        "to": 7,
        "from": 3,
        "at_ms": 1757000000000,
        "kind": "promote",
    }


def check_a_first_transition_stores_a_null_from():
    entry = registry.Transition(
        stage="staging", to_version=1, from_version=None, at_ms=1
    )
    assert json.loads(entry.as_value())["from"] is None


def check_a_real_promotion_is_marked_as_logged():
    """The other side of the no-op check, and the one a mutation pass wanted.

    Only the no-op case asserted anything about `logged`, so flipping the field's default
    to False left every check green while the command line reported a real promotion as
    "already points at version N".
    """
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        first = registry.promote(cli, MODEL, "1", "production")
        second = registry.promote(cli, MODEL, "2", "production")
        assert first.logged is True
        assert second.logged is True


def check_a_transition_cannot_be_edited_after_it_is_made():
    entry = registry.Transition(
        stage="production", to_version=1, from_version=None, at_ms=1
    )
    try:
        entry.to_version = 2
    except dataclasses.FrozenInstanceError as exc:
        # The message rather than only the type. A dataclass that is not frozen raises
        # nothing at all, so the else branch is what really separates the two, and the
        # field name is what says the refusal was about the thing being assigned.
        assert "to_version" in str(exc), str(exc)
    else:
        raise AssertionError("a Transition let its version be rewritten")


def check_the_stored_form_is_byte_stable():
    """Pinned as an exact string, not as a dict that came back through json.loads.

    Reading it back with the same parser that wrote it cannot see a key order change, and
    the stored form is the compatibility surface between a store written by one build and
    read by the next.
    """
    entry = registry.Transition(
        stage="production", to_version=7, from_version=3, at_ms=1757000000000
    )
    assert entry.as_value() == (
        '{"at_ms":1757000000000,"from":3,"kind":"promote","stage":"production","to":7}'
    ), entry.as_value()


def check_the_log_key_is_padded_to_a_fixed_width():
    """The width is the other half of that compatibility surface.

    Ordering comes from a lexical sort over the tag keys, so a build that wrote four
    digits and a build that reads five would sort one store's entries into the other's
    wrongly. Nothing else pins the width, because within a single store any consistent
    width sorts correctly and the whole suite passes on five.
    """
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        registry.promote(cli, MODEL, "1", "production")
        keys = [
            k
            for k in cli.get_registered_model(MODEL).tags
            if k.startswith(registry.HISTORY_PREFIX)
        ]
        assert keys == ["transition.production.0000"], keys


def check_the_transition_time_is_the_clock_in_milliseconds():
    """at_ms is never read by any decision here, so nothing else constrains it.

    Four separate mutants of the two places it is built survived without this, including
    one dividing by a thousand instead of multiplying. The number is shown to whoever asks
    when a deploy happened, and a timestamp twenty days out is worse than no timestamp.
    """
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        before = int(time.time() * 1000)
        entry = registry.promote(cli, MODEL, "1", "production")
        no_op = registry.promote(cli, MODEL, "1", "production")
        # retire builds the same expression a third time and had no check on it, so two
        # more mutants of it survived. A rule covering two of the three places it is
        # written is a rule the next one will be written outside of.
        registry.promote(cli, MODEL, "2", "canary")
        retired = registry.retire(cli, MODEL, "canary")
        after = int(time.time() * 1000)
        for label, value in (
            ("logged", entry.at_ms),
            ("no-op", no_op.at_ms),
            ("retire", retired.at_ms),
        ):
            assert before <= value <= after, (label, value, before, after)


def check_register_points_the_version_at_the_run_that_made_it():
    with Store() as (cli, MODEL):
        run = a_run(cli)
        registry.register(cli, MODEL, run, "e" * 64, "fp")
        assert cli.get_model_version(MODEL, "1").source == "runs:/{}/model".format(run)


def check_an_explicit_source_is_used_instead():
    with Store() as (cli, MODEL):
        registry.register(
            cli, MODEL, a_run(cli), "f" * 64, "fp", source="s3://bucket/model"
        )
        assert cli.get_model_version(MODEL, "1").source == "s3://bucket/model"


def check_the_disagreement_message_names_the_version_the_log_last_moved_to():
    """The message carries the two numbers somebody needs to work out what happened.

    A mutant reading the second to last log entry instead of the last one survived the
    first pass, because the check on this path only looked for a phrase in the message and
    never read the numbers in it.
    """
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 4)
        registry.promote(cli, MODEL, "1", "production")
        registry.promote(cli, MODEL, "2", "production")
        registry.promote(cli, MODEL, "3", "production")
        cli.set_registered_model_alias(MODEL, "production", "4")
        try:
            registry.rollback_target(cli, MODEL, "production")
        except registry.RegistryError as exc:
            assert "at version 4" in str(exc), str(exc)
            assert "moved to 3" in str(exc), str(exc)
        else:
            raise AssertionError("no refusal at all")


def check_the_stage_list_has_no_null_member():
    """MLflow's vocabulary includes the string "None" as a stage. This one does not."""
    assert "None" not in registry.STAGES
    assert all(s == s.lower() for s in registry.STAGES)


def check_history_of_a_model_that_does_not_exist_is_empty():
    with Store() as (cli, MODEL):
        assert registry.history(cli, MODEL, "production") == []


def check_version_for_artifact_returns_none_when_it_is_not_there():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        assert registry.version_for_artifact(cli, MODEL, "z" * 64) is None


def _t(to, frm, kind=registry.PROMOTE, stage="production"):
    return registry.Transition(stage=stage, to_version=to, from_version=frm, at_ms=1, kind=kind)


def check_settle_keeps_every_entry_of_a_clean_log():
    """The control. Without it every check below passes on a settle that drops the lot."""
    log = [_t(1, None), _t(2, 1), _t(3, 2)]
    kept, dropped = registry.settle(log, 3)
    assert kept == log, kept
    assert dropped == [], dropped


def check_settle_drops_an_entry_the_next_one_contradicts():
    """A promote that crashed before moving the alias. The entry after it left the version
    the crashed one claimed to have replaced, which is the witness."""
    log = [_t(1, None), _t(2, 1), _t(4, 2), _t(3, 2)]
    kept, dropped = registry.settle(log, 3)
    assert [e.to_version for e in kept] == [1, 2, 3], kept
    assert len(dropped) == 1 and dropped[0].entry.to_version == 4, dropped
    assert dropped[0].position == 2, dropped[0]


def check_settle_drops_a_last_entry_the_alias_contradicts():
    log = [_t(1, None), _t(2, 1), _t(3, 2)]
    kept, dropped = registry.settle(log, 2)
    assert [e.to_version for e in kept] == [1, 2], kept
    assert len(dropped) == 1 and dropped[0].entry.to_version == 3, dropped


def check_settle_reports_why_it_dropped_each_entry():
    """The reason is what an operator reads. A drop with no sentence behind it is a store
    that silently disagrees with its own log."""
    kept, dropped = registry.settle([_t(1, None), _t(4, 1), _t(2, 1)], 2)
    assert len(dropped) == 1, dropped
    assert "never reached the alias" in dropped[0].reason, dropped[0].reason


def check_settle_handles_an_empty_log():
    assert registry.settle([], None) == ([], [])


def check_settle_keeps_a_retirement_the_next_entry_confirms():
    """A retirement leaves the stage pointing at nothing, so the next entry reads None as
    its own from_version and the same comparison does the work."""
    log = [_t(3, None), _t(None, 3, registry.RETIRE), _t(4, None)]
    kept, dropped = registry.settle(log, 4)
    assert kept == log, kept
    assert dropped == [], dropped


def check_settle_keeps_a_retirement_that_is_last_and_agrees():
    log = [_t(3, None), _t(None, 3, registry.RETIRE)]
    kept, dropped = registry.settle(log, None)
    assert kept == log, kept
    assert dropped == [], dropped


def check_rollback_target_ignores_a_version_that_never_served():
    """The whole point of settle, end to end through the store.

    Promote 1, 2 and 3. Crash while writing the entry for 4. Decide 4 was the problem and
    ship 5 instead. Version 4 never served a request and the raw log cannot tell.
    """
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 5)
        for ref in ("1", "2", "3"):
            registry.promote(cli, MODEL, ref, "production")

        tags = dict(cli.get_registered_model(MODEL).tags)
        phantom = registry.Transition(
            stage="production", to_version=4, from_version=3, at_ms=1
        )
        key = "{}.{}.{:0{}d}".format(
            registry.HISTORY_PREFIX, "production",
            registry._next_seq(tags, "production"), registry.SEQ_WIDTH,
        )
        cli.set_registered_model_tag(MODEL, key, phantom.as_value())

        registry.promote(cli, MODEL, "5", "production")

        raw = registry.history(cli, MODEL, "production")
        assert [e.to_version for e in raw] == [1, 2, 3, 4, 5], raw
        # The walk over the raw log picks the phantom. This is the defect, pinned, so a
        # later edit that drops settle fails here rather than going quiet.
        assert registry.walk_back(raw, 5) == 4
        assert registry.rollback_target(cli, MODEL, "production") == 3


def check_retire_removes_the_alias_and_logs_it():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        registry.promote(cli, MODEL, "2", "canary")
        entry = registry.retire(cli, MODEL, "canary")
        assert entry.to_version is None and entry.from_version == 2, entry
        assert entry.kind == registry.RETIRE
        assert registry.current(cli, MODEL, "canary") is None


def check_retiring_a_stage_that_holds_nothing_is_refused():
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 2)
        try:
            registry.retire(cli, MODEL, "canary")
        except registry.RegistryError as exc:
            assert "not pointing at anything" in str(exc), exc
            return
        raise AssertionError("retiring an empty stage was allowed")


def check_retire_refuses_a_stage_that_is_not_one():
    with Store() as (cli, MODEL):
        try:
            registry.retire(cli, MODEL, "shadow")
        except registry.RegistryError as exc:
            assert "unknown stage" in str(exc), exc
            return
        raise AssertionError("retired a stage that does not exist")


def check_walk_back_skips_a_retirement():
    """Rolling back to "nothing was deployed" is not a rollback. The version before the
    gap is the one that last served."""
    log = [_t(1, None), _t(2, 1), _t(None, 2, registry.RETIRE), _t(3, None)]
    assert registry.walk_back(log, 3) == 2


def check_walk_back_returns_none_with_nothing_behind_it():
    assert registry.walk_back([_t(1, None)], 1) is None


def check_canary_is_a_stage():
    """It was a command line flag, and that was why a rollback verdict had nothing to
    undo."""
    assert "canary" in registry.STAGES


def check_an_unconfirmed_entry_cannot_be_edited():
    """An Unconfirmed is evidence about a log, and evidence a caller can rewrite is not
    evidence. Same frozen mutant as the one on Deployment, on a third dataclass."""
    kept, dropped = registry.settle([_t(1, None), _t(4, 1), _t(2, 1)], 2)
    assert len(dropped) == 1, dropped
    try:
        dropped[0].reason = "something else"
    except Exception:
        return
    raise AssertionError("an Unconfirmed accepted an edit")


def _registry_cli():
    """Load scripts/registry.py. It imports mlflow inside main, so this stays cheap."""
    import importlib.util

    path = os.path.join(ROOT, "scripts", "registry.py")
    spec_ = importlib.util.spec_from_file_location("registry_cli", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


class _FakeCli:
    """Enough MLflow to answer gate_refusal, and nothing else.

    The real client needs a store and a trained model. What this function reads is one
    run's tags, so a stand in makes the case table cheap enough to cover every branch.
    """

    def __init__(self, tags):
        self._tags = tags

    def get_model_version(self, name, version):
        return type("V", (), {"run_id": "run-{}".format(version)})()

    def get_run(self, run_id):
        data = type("D", (), {"tags": self._tags})()
        return type("R", (), {"data": data})()


class _FakeRegistry:
    @staticmethod
    def resolve(cli, model, ref):
        return int(ref)


def check_require_gate_refuses_every_verdict_that_is_not_a_promotion():
    """The guard a reader asked for on 2026-09-17, with the cases that must refuse.

    Demonstrated on a real store before this was written: a config training to a NaN is
    refused by the gate and promoted by hand in the very next command, and the transition
    log records an ordinary promotion. The gate had already written the refusal onto the
    version's run and nothing read it.

    Every row here is a refusal. The one that passes is below, so this cannot be satisfied
    by refusing everything.
    """
    mod = _registry_cli()

    cases = [
        ("never gated", {}, "never been gated"),
        (
            "gate refused it",
            {"gate.verdict": "refuse", "gate.reason": "candidate_not_finite",
             "gate.stage": "production"},
            "was refuse",
        ),
        (
            "gate rejected it",
            {"gate.verdict": "reject", "gate.reason": "worse", "gate.stage": "production"},
            "was reject",
        ),
        (
            "passed for a different stage",
            {"gate.verdict": "promote", "gate.stage": "staging"},
            "gated against staging",
        ),
        (
            "verdict with no stage recorded",
            {"gate.verdict": "promote"},
            "unrecorded stage",
        ),
    ]
    for label, tags, expected in cases:
        refusal = mod.gate_refusal(_FakeCli(tags), _FakeRegistry, "m", "2", "production")
        assert refusal is not None, "{}: was allowed through".format(label)
        assert expected in refusal, "{}: got {!r}".format(label, refusal)


def check_require_gate_allows_a_version_the_gate_promoted_for_that_stage():
    """The control. Without it the check above passes with the guard refusing everything."""
    mod = _registry_cli()
    tags = {"gate.verdict": "promote", "gate.reason": "better", "gate.stage": "production"}
    assert mod.gate_refusal(_FakeCli(tags), _FakeRegistry, "m", "2", "production") is None


def check_the_gate_requirement_is_opt_in_and_off_by_default():
    """Default off is a decision, so it is pinned rather than left to whoever edits next.

    `rollback` calls `promote`, and a version being rolled back to was gated against a
    different incumbent. Enforcing this by default makes the recovery path depend on a
    stale verdict.
    """
    mod = _registry_cli()
    args = mod.build_parser().parse_args(
        ["--store", "sqlite:///x.db", "promote", "2", "production"]
    )
    assert args.require_gate is False
    args = mod.build_parser().parse_args(
        ["--store", "sqlite:///x.db", "promote", "2", "production", "--require-gate"]
    )
    assert args.require_gate is True


def check_the_library_promote_still_takes_no_gate_argument():
    """Pinned as an absence. The guard belongs at the CLI and not in the library.

    If this ever moves into `mcr.registry.promote`, `rollback` inherits it, because
    rollback is implemented as a promote. Anybody reaching for symmetry will try.
    """
    import inspect

    names = list(inspect.signature(registry.promote).parameters)
    assert names == ["cli", "name", "ref", "stage", "kind"], names
