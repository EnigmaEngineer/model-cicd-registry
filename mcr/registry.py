"""The registry. Which model is in production, and which one was there before it.

MLflow has a model registry with stages built in, and this module does not use them. The
reason is measured rather than stylistic and it is in docs/adr-0003.

Two facts decided the shape of everything here.

`transition_model_version_stage` emits a FutureWarning saying stages will be removed, and
its default puts two versions in Production at the same time. You get the one-at-a-time
property only by remembering `archive_existing_versions=True`, and a deploy that reads
"the production model" from a store holding two of them gets whichever one the search
returns first.

An alias has that property for free. Pointing `production` at a new version removes it from
the old one in the same call. So the stage is an alias.

The cost is the thing this module spends most of its code on. An alias move leaves no
trace. After `production` moves from version 1 to version 3, version 1 does not record that
it ever held it and its `last_updated_timestamp` does not even change. The registry can say
what is in production. It cannot say what was, and a rollback is a question about what was.
So the transition log is written here, on the registered model, one tag per transition.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import mlflow

from .tracking import TAG_ARTIFACT_HASH, TAG_CONFIG_FINGERPRINT

# The stages this project moves a model through. Deliberately not MLflow's vocabulary,
# which is capitalised and includes "None" as a string. Lowercase, and no null stage,
# because "not pointed at by any alias" already means that and does not need a name.
STAGES = ("staging", "production")

# Prefix for the transition log. One registered model tag per transition, sequence number
# in the key so nothing overwrites anything. See _next_seq for what this does not protect.
HISTORY_PREFIX = "transition"
SEQ_WIDTH = 4


class RegistryError(RuntimeError):
    """Raised when the registry cannot answer which model something refers to."""


PROMOTE = "promote"
ROLLBACK = "rollback"


@dataclass(frozen=True)
class Transition:
    stage: str
    to_version: int
    from_version: Optional[int]
    at_ms: int
    # Which way this move went. It is stored rather than inferred from the shape of the
    # log, and that is the whole reason a second rollback works. See rollback_target.
    kind: str = PROMOTE
    # False when promote was asked for a move the store was already in. The alias did not
    # move and nothing went into the log. A bool rather than a sentinel timestamp, because
    # a caller checking `at_ms == 0` is reading a magic number to answer a real question.
    logged: bool = True

    def as_value(self) -> str:
        return json.dumps(
            {
                "stage": self.stage,
                "to": self.to_version,
                "from": self.from_version,
                "at_ms": self.at_ms,
                "kind": self.kind,
            },
            sort_keys=True,
            separators=(",", ":"),
        )


def _check_stage(stage: str) -> str:
    if stage not in STAGES:
        raise RegistryError("unknown stage '{}', expected one of {}".format(stage, STAGES))
    return stage


def ensure_model(cli: mlflow.MlflowClient, name: str) -> str:
    """Create the registered model if it is not there. Returns the name.

    `create_registered_model` raises when the name is taken, which is the behaviour that
    makes this safe to call on every run rather than once at setup.
    """
    try:
        cli.get_registered_model(name)
    except mlflow.exceptions.MlflowException:
        cli.create_registered_model(name)
    return name


def versions(cli: mlflow.MlflowClient, name: str) -> List:
    """Every live version of one model, oldest first.

    Sorted here because the store does not sort. `search_model_versions` came back
    [3, 1] in one session and [2, 1] in the same one. A caller taking the first result as
    "the newest" would be reading an arbitrary row.
    """
    found = cli.search_model_versions("name='{}'".format(name))
    return sorted(found, key=lambda v: int(v.version))


def version_for_artifact(
    cli: mlflow.MlflowClient, name: str, artifact_hash: str
) -> Optional[int]:
    for v in versions(cli, name):
        if v.tags.get(TAG_ARTIFACT_HASH) == artifact_hash:
            return int(v.version)
    return None


def register(
    cli: mlflow.MlflowClient,
    name: str,
    run_id: str,
    artifact_hash: str,
    config_fingerprint: str,
    source: Optional[str] = None,
) -> int:
    """Put one artefact in the registry. Returns the version number.

    Idempotent on the artefact hash, and that is the whole point of it. The store is happy
    to create two versions from one run, and two versions holding identical bytes make
    every later question ambiguous. Rolling back to "the previous production model" would
    land on a version whose bytes are the ones you are rolling away from.

    Nothing keys on the run id. Two runs of one recipe get two run ids, so a run is the
    occasion and the artefact hash is the model.
    """
    ensure_model(cli, name)

    existing = version_for_artifact(cli, name, artifact_hash)
    if existing is not None:
        return existing

    mv = cli.create_model_version(
        name,
        source=source or "runs:/{}/model".format(run_id),
        run_id=run_id,
        tags={
            TAG_ARTIFACT_HASH: artifact_hash,
            TAG_CONFIG_FINGERPRINT: config_fingerprint,
        },
    )
    return int(mv.version)


def _alias_exists(cli: mlflow.MlflowClient, name: str, alias: str) -> bool:
    try:
        cli.get_model_version_by_alias(name, alias)
        return True
    except mlflow.exceptions.MlflowException:
        return False


def resolve(cli: mlflow.MlflowClient, name: str, ref: str) -> int:
    """Turn a reference into a version number, or refuse.

    A reference is a version number or a stage. The refusal exists because MLflow lets
    those two namespaces overlap. It rejects an alias containing a slash and accepts an
    alias that is a decimal number, so `set_registered_model_alias(name, "2", "4")` is
    allowed while version 2 exists. Measured: `get_model_version(name, "2")` and
    `get_model_version_by_alias(name, "2")` then return two different models, trained from
    two different runs.

    Picking one of them is the failure. A store carrying that collision is a store whose
    answer to "what is version 2" depends on which function the caller reached for, and
    the caller cannot know. So an ambiguous reference is an error.
    """
    if ref.isdigit():
        if _alias_exists(cli, name, ref):
            raise RegistryError(
                "'{}' is both a version number and an alias on {}, so it names two "
                "models".format(ref, name)
            )
        try:
            return int(cli.get_model_version(name, ref).version)
        except mlflow.exceptions.MlflowException as exc:
            raise RegistryError("no version {} of {}: {}".format(ref, name, exc))

    try:
        return int(cli.get_model_version_by_alias(name, ref).version)
    except mlflow.exceptions.MlflowException as exc:
        raise RegistryError("nothing is aliased '{}' on {}: {}".format(ref, name, exc))


def current(cli: mlflow.MlflowClient, name: str, stage: str) -> Optional[int]:
    _check_stage(stage)
    if not _alias_exists(cli, name, stage):
        return None
    return int(cli.get_model_version_by_alias(name, stage).version)


def _next_seq(tags: Dict[str, str], stage: str) -> int:
    prefix = "{}.{}.".format(HISTORY_PREFIX, stage)
    used = [int(k[len(prefix):]) for k in tags if k.startswith(prefix)]
    return max(used) + 1 if used else 0


def history(cli: mlflow.MlflowClient, name: str, stage: str) -> List[Transition]:
    """Every transition of one stage, oldest first.

    Read out of the registered model's tags rather than off the versions, because the
    versions do not carry it. An alias move does not touch the row of the version it was
    taken from, so there is nothing there to read.
    """
    _check_stage(stage)
    try:
        tags = dict(cli.get_registered_model(name).tags)
    except mlflow.exceptions.MlflowException:
        return []

    prefix = "{}.{}.".format(HISTORY_PREFIX, stage)
    out = []
    for key in sorted(k for k in tags if k.startswith(prefix)):
        body = json.loads(tags[key])
        out.append(
            Transition(
                stage=body["stage"],
                to_version=body["to"],
                from_version=body["from"],
                at_ms=body["at_ms"],
                kind=body["kind"],
            )
        )
    return out


def promote(
    cli: mlflow.MlflowClient, name: str, ref: str, stage: str, kind: str = PROMOTE
) -> Transition:
    """Point a stage at a version and write down that it moved.

    The alias move and the log entry are two calls and there is no transaction over them.
    If the process dies between them the alias has moved and nothing records it, which is
    the same state the registry is in without this module at all. The log is written
    first for that reason, so the failure mode is a recorded transition that did not
    happen rather than a silent one that did. A recorded lie is visible. A silent move is
    the thing rollback cannot survive.
    """
    _check_stage(stage)
    version = resolve(cli, name, ref)
    before = current(cli, name, stage)

    if before == version:
        # Setting an alias to where it already points is allowed by the store and is a
        # no-op. Logging it would put an entry in the history with from equal to to, and
        # a rollback walking backwards would then land on the version it started from.
        return Transition(
            stage=stage,
            to_version=version,
            from_version=before,
            at_ms=int(time.time() * 1000),
            kind=kind,
            logged=False,
        )

    entry = Transition(
        stage=stage,
        to_version=version,
        from_version=before,
        at_ms=int(time.time() * 1000),
        kind=kind,
    )

    tags = dict(cli.get_registered_model(name).tags)
    key = "{}.{}.{:0{}d}".format(HISTORY_PREFIX, stage, _next_seq(tags, stage), SEQ_WIDTH)
    cli.set_registered_model_tag(name, key, entry.as_value())

    # A registered model tag overwrites silently, so the sequence number above is chosen
    # from a read that another writer could have invalidated. There is no compare and set
    # in this API. This reads the key back and refuses if it does not hold what we wrote,
    # which detects a lost update rather than preventing one. Single writer here, so it
    # has never fired. It would fire the moment two pipelines promoted at once.
    written = dict(cli.get_registered_model(name).tags).get(key)
    if written != entry.as_value():
        raise RegistryError(
            "transition log key {} was overwritten between write and read".format(key)
        )

    cli.set_registered_model_alias(name, stage, str(version))
    return entry


def rollback_target(cli: mlflow.MlflowClient, name: str, stage: str) -> Optional[int]:
    """The most recent version this stage held that has not already been rolled away from.

    The obvious rule is "whatever it held immediately before", and that rule is wrong. It
    was the first thing written here and a test caught it. Promote version 1. Then 2. Then
    3. Now roll back. Production is on 2 and the version it held immediately before 2 is 3,
    which is the one that was just rolled away from. So a second rollback would put the known bad
    model straight back into production and report success. Rollback would oscillate
    between the last two versions and could never reach version 1.

    So a version that a rollback moved away from is abandoned, and the walk skips it. That
    is why `kind` is stored on the entry rather than inferred. Inferring it from the shape
    of the log means guessing that a move to an older version was a rollback, and a
    deliberate redeploy of an older model is the same shape.

    Returns None when there is nothing left behind the current version, which is a real
    answer rather than an error.

    The disagreement check is the other half. The alias is the truth about what is deployed
    and the log is this module's account of how it got there. Anything moving the alias
    without going through `promote`, including the MLflow UI and a one line script, breaks
    the account without breaking the alias. A rollback computed off a log that no longer
    describes the store would deploy some old version and report success, so it refuses.
    """
    _check_stage(stage)
    now = current(cli, name, stage)
    if now is None:
        return None

    log = history(cli, name, stage)
    if not log:
        raise RegistryError(
            "{} on {} is at version {} and the transition log is empty".format(
                stage, name, now
            )
        )

    if log[-1].to_version != now:
        raise RegistryError(
            "{} is at version {} and the log's last entry moved to {}, so the log does "
            "not describe this store".format(stage, now, log[-1].to_version)
        )

    abandoned = {e.from_version for e in log if e.kind == ROLLBACK}
    held = [e.to_version for e in log]

    for version in reversed(held[:-1]):
        if version != now and version not in abandoned:
            return version
    return None


def rollback(cli: mlflow.MlflowClient, name: str, stage: str) -> Transition:
    target = rollback_target(cli, name, stage)
    if target is None:
        raise RegistryError(
            "{} on {} has no previous version to roll back to".format(stage, name)
        )
    return promote(cli, name, str(target), stage, kind=ROLLBACK)
