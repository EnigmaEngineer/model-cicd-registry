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

from mcr import admission, artifact as artifact_mod, attest, registry
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
        registry.create_version(
            cli, model, a_run(cli), "{:02d}{}".format(i, "f" * 62), "fp{}".format(i)
        )
        for i in range(n)
    ]


def check_create_version_returns_increasing_versions():
    with Store() as (cli, MODEL):
        got = some_versions(cli, MODEL, 3)
        assert got == [1, 2, 3], got


def check_create_version_is_idempotent_on_the_artefact_hash():
    with Store() as (cli, MODEL):
        digest = "a" * 64
        first = registry.create_version(cli, MODEL, a_run(cli), digest, "fp")
        second = registry.create_version(cli, MODEL, a_run(cli), digest, "fp")
        assert first == second, (first, second)
        assert len(registry.versions(cli, MODEL)) == 1


def check_create_version_tags_both_identities():
    with Store() as (cli, MODEL):
        registry.create_version(cli, MODEL, a_run(cli), "b" * 64, "fp0")
        tags = cli.get_model_version(MODEL, "1").tags
        assert tags[TAG_ARTIFACT_HASH] == "b" * 64
        assert tags[TAG_CONFIG_FINGERPRINT] == "fp0"


def check_two_runs_of_one_recipe_are_one_version():
    """The same bytes from two occasions. The registry holds models, not occasions."""
    with Store() as (cli, MODEL):
        digest = "c" * 64
        a = registry.create_version(cli, MODEL, a_run(cli), digest, "fp")
        b = registry.create_version(cli, MODEL, a_run(cli), digest, "fp")
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


def check_create_version_points_the_version_at_the_run_that_made_it():
    with Store() as (cli, MODEL):
        run = a_run(cli)
        registry.create_version(cli, MODEL, run, "e" * 64, "fp")
        assert cli.get_model_version(MODEL, "1").source == "runs:/{}/model".format(run)


def check_an_explicit_source_is_used_instead():
    with Store() as (cli, MODEL):
        registry.create_version(
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


# These check that `register` is a door and `create_version` is not. Which configs map to
# which verdict is tests/test_admission.py's subject and is not re-asked here, so the
# payloads below are the smallest thing that is admissible or is not. That keeps this
# module's runtime where a mutation pass can afford it, since it is the oracle for
# mcr/registry.py.


def an_artifact(admissible: bool, tag: str = "x") -> artifact_mod.Artifact:
    """A payload on one side of the rule or the other, and otherwise uninteresting.

    `tag` moves the bytes, which moves the content hash. Two checks needing two distinct
    admissible artefacts cannot share one, because `register` is idempotent on the hash and
    would hand the second one the first one's version.
    """
    weights = [0.25, -0.5] if admissible else [float("nan"), 0.1]
    return artifact_mod.Artifact(payload={
        "artifact_version": 1,
        "name": "fixture-{}".format(tag),
        "seed": 1,
        "model": {"weights": weights, "bias": 0.1, "epochs_run": 10},
        "metrics": {"holdout_log_loss": 0.5},
    })


def check_register_accepts_an_admissible_artefact():
    with Store() as (cli, MODEL):
        v = registry.register(cli, MODEL, a_run(cli), an_artifact(True), "fp")
        assert v == 1, v


def check_register_refuses_an_artefact_that_did_not_train():
    """The defect this door was built for. A NaN artefact used to come back with a version."""
    with Store() as (cli, MODEL):
        art = an_artifact(False)
        try:
            registry.register(cli, MODEL, a_run(cli), art, "fp")
        except registry.RegistryError as exc:
            assert "model.weights[0] is nan" in str(exc), str(exc)
            assert MODEL in str(exc), str(exc)
        else:
            raise AssertionError("register accepted a non finite artefact")


def check_a_refused_artefact_leaves_no_version_behind():
    """A refusal that has already written is not a refusal.

    `ensure_model` creates the registered model and the old `register` called it first, so
    the ordering here is the thing being checked rather than a detail. The refusal has to
    come before anything touches the store.
    """
    with Store() as (cli, MODEL):
        try:
            registry.register(cli, MODEL, a_run(cli), an_artifact(False), "fp")
        except registry.RegistryError:
            pass
        assert registry.versions(cli, MODEL) == [], registry.versions(cli, MODEL)


def check_the_refusal_names_every_reason_it_found():
    """A diverged model fails two ways and an operator reading one of them fixes half.

    The weights here are huge and finite, which is the shape `candidate-inverted` actually
    has. The first version of this fixture used an infinite weight and failed, because a non
    finite parameter scale suppresses the saturation line on purpose. Saying a number is not
    a number and then comparing it to a threshold tells a reader nothing. So an artefact
    reaches two reasons by being saturated in its parameters and non finite somewhere else,
    which is exactly what the l2 10 config does through `model.final_loss`.
    """
    art = artifact_mod.Artifact(payload={
        "name": "both",
        "model": {"weights": [1e300, -1e300], "bias": 0.0},
        "metrics": {"holdout_log_loss": float("nan")},
    })
    verdict = admission.inspect(art.payload)
    assert len(verdict.reasons) >= 2, verdict.reasons

    with Store() as (cli, MODEL):
        try:
            registry.register(cli, MODEL, a_run(cli), art, "fp")
        except registry.RegistryError as exc:
            message = str(exc)
        else:
            raise AssertionError("register accepted it")

    assert "nan" in message, message
    assert "logit scale" in message, message


def check_register_derives_the_hash_from_the_artefact():
    """It is handed the artefact, so a caller can no longer pass a hash that disagrees."""
    with Store() as (cli, MODEL):
        art = an_artifact(True, tag="derive")
        registry.register(cli, MODEL, a_run(cli), art, "fp0")
        tags = cli.get_model_version(MODEL, "1").tags
        assert tags[TAG_ARTIFACT_HASH] == art.content_hash(), tags[TAG_ARTIFACT_HASH]
        assert len(tags[TAG_ARTIFACT_HASH]) == 64


def check_register_is_still_idempotent_on_the_artefact():
    """The property create_version has, through the door rather than round it."""
    with Store() as (cli, MODEL):
        art = an_artifact(True, tag="idem")
        first = registry.register(cli, MODEL, a_run(cli), art, "fp")
        second = registry.register(cli, MODEL, a_run(cli), art, "fp")
        assert first == second, (first, second)
        assert len(registry.versions(cli, MODEL)) == 1


def check_create_version_is_the_unguarded_path_and_says_so():
    """Asserted rather than trusted to a docstring.

    `create_version` exists so the store's own behaviour stays testable without a trained
    model. That means it does not check, and a future edit moving the admission rule into
    it would break every fixture in this module that registers a fabricated hash. If that
    is ever the right change, this check is where the decision gets made again.
    """
    with Store() as (cli, MODEL):
        v = registry.create_version(cli, MODEL, a_run(cli), "f" * 64, "fp")
        assert v == 1, v


def _attested(to, frm, kind=registry.PROMOTE):
    return dataclasses.replace(_t(to, frm, kind), attestation=attest.ATTESTED)


def _forged(to, frm, kind=registry.PROMOTE):
    return dataclasses.replace(_t(to, frm, kind), attestation=attest.FORGED)


def check_authentic_keeps_a_clean_signed_log():
    """The control, and the rest of this group means nothing without it."""
    log = [_attested(1, None), _attested(2, 1), _attested(3, 2)]
    kept, rejected = registry.authentic(log)
    assert kept == log, kept
    assert rejected == []


def check_authentic_keeps_a_log_that_predates_the_signature():
    """A history written before any of this still has to answer.

    A rule that refused an unsigned log would refuse every store this code has ever been
    pointed at, which is the shape where a safety check gets switched off and stays off.
    """
    log = [_t(1, None), _t(2, 1), _t(3, 2)]
    kept, rejected = registry.authentic(log)
    assert kept == log, kept
    assert rejected == []


def check_authentic_rejects_a_forged_entry():
    kept, rejected = registry.authentic([_attested(1, None), _forged(9, 1), _attested(3, 2)])
    assert len(kept) == 2, kept
    assert len(rejected) == 1, rejected
    assert rejected[0].position == 1
    assert "signature" in rejected[0].reason


def check_authentic_rejects_an_unsigned_entry_inside_a_signed_log():
    """The cheapest forgery is to leave the signature off, so this is the one that counts.

    The first version of this rule accepted an unsigned entry unconditionally. That made
    the signature worth nothing, because anybody who can overwrite a tag can write one
    without a signature, and that is easier than faking one.
    """
    kept, rejected = registry.authentic([_attested(1, None), _t(9, 1), _attested(3, 2)])
    assert len(rejected) == 1, rejected
    assert rejected[0].position == 1
    assert "the rest of this log is signed" in rejected[0].reason


def check_require_attested_refuses_a_log_that_predates_the_signature():
    """The knob only tightens. It is for a caller that will not read a legacy log at all."""
    log = [_t(1, None), _t(2, 1)]
    kept, rejected = registry.authentic(log, require_attested=True)
    assert kept == []
    assert len(rejected) == 2, rejected
    assert "one was required" in rejected[0].reason


def check_settle_does_not_look_at_who_wrote_an_entry():
    """The separation, asserted rather than left to the two docstrings.

    A forged entry whose `from_version` chains correctly is indistinguishable to the
    recovery rule, because the forger picks that field. If a later edit folded the
    signature check into `settle` this check is where that decision gets made again.
    """
    log = [_attested(1, None), _forged(2, 1), _attested(3, 2)]
    kept, dropped = registry.settle(log, 3)
    assert kept == log, kept
    assert dropped == []


def check_a_forged_entry_is_refused_rather_than_dropped():
    """Through the store, and the refusal has to be the specific one.

    An unconfirmed entry is dropped and the walk continues. This is not that, and a caller
    cannot tell the two apart from a bare RegistryError.
    """
    key = b"checks-key"
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        for v in ("1", "2", "3"):
            registry.promote(cli, MODEL, v, "production", key=key)
        cli.set_registered_model_tag(MODEL, registry._slot("production", 1), json.dumps(
            {"body": _t(9, 1).as_value(), "mac": "00" * 32},
            sort_keys=True, separators=(",", ":"),
        ))
        try:
            registry.rollback_target(cli, MODEL, "production", key=key)
        except registry.UnauthenticEntry as exc:
            assert "cannot be attributed" in str(exc), str(exc)
        else:
            raise AssertionError("a forged transition log entry was accepted")


def check_the_unauthentic_refusal_is_catchable_as_a_registry_error():
    """Additive by construction, so every existing handler still works."""
    assert issubclass(registry.UnauthenticEntry, registry.RegistryError)


def check_promote_writes_a_signed_entry_when_a_key_is_present():
    """And reads it back as attested, which is the end to end of the writing half."""
    key = b"checks-key"
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        registry.promote(cli, MODEL, "1", "production", key=key)
        log = registry.history(cli, MODEL, "production", key=key)
        assert [e.attestation for e in log] == [attest.ATTESTED], log


def check_promote_writes_an_unsigned_entry_when_there_is_no_key():
    """The default, and it has to stay readable rather than becoming an error."""
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        registry.promote(cli, MODEL, "1", "production", key=None)
        log = registry.history(cli, MODEL, "production", key=None)
        assert [e.attestation for e in log] == [attest.UNATTESTED], log


def check_attestation_is_never_part_of_the_signed_body():
    """It describes the tag, not the transition. Signing over it would make the signature
    depend on the reading of itself."""
    body = _attested(2, 1).as_value()
    assert "attestation" not in body, body
    assert body == _t(2, 1).as_value()


def check_history_refuses_a_slot_that_does_not_parse():
    """A corrupt tag under the history prefix is named rather than skipped.

    Skipping it would leave a gap, and the witness rule reads a gap as a crash, so a
    corrupt entry would be laundered into a recovered one.
    """
    with Store() as (cli, MODEL):
        some_versions(cli, MODEL, 3)
        registry.promote(cli, MODEL, "1", "production")
        cli.set_registered_model_tag(MODEL, registry._slot("production", 1), "{broken")
        try:
            registry.history(cli, MODEL, "production")
        except registry.RegistryError as exc:
            assert "does not hold a readable entry" in str(exc), str(exc)
        else:
            raise AssertionError("an unreadable transition log slot was skipped")


def check_an_unattributed_entry_cannot_be_edited():
    """Same reason as Unconfirmed. Evidence a caller can rewrite is not evidence."""
    _, rejected = registry.authentic([_attested(1, None), _forged(9, 1)])
    try:
        rejected[0].reason = "fine actually"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("an Unattributed accepted an edit")


def check_a_log_with_no_signed_entry_cannot_be_defended_by_default():
    """The hole in the policy, asserted so nobody finds it in an interview instead.

    The no downgrade rule needs at least one entry it verified, to know that the writer of
    this log holds the key. A model the real pipeline has never promoted has no such entry,
    so a log written start to finish by somebody else reads as a legacy log and is
    accepted. The rollback target is then whatever they chose.

    This cannot be fixed in the default, because a store full of genuine legacy logs is
    indistinguishable from this by construction. `require_attested` is the answer and the
    check below is the pair to this one.
    """
    forged = [_t(1, None), _t(2, 1), _t(3, 2)]
    kept, rejected = registry.authentic(forged)
    assert rejected == [], rejected
    assert registry.walk_back(registry.settle(kept, 3)[0], 3) == 2


def check_require_attested_closes_the_wholly_unsigned_log():
    """The pair to the check above, and the reason that parameter is not dead code."""
    forged = [_t(1, None), _t(2, 1), _t(3, 2)]
    kept, rejected = registry.authentic(forged, require_attested=True)
    assert kept == []
    assert len(rejected) == 3, rejected
