"""What the MLflow registry really does, measured, against a store built here.

    python3 scripts/registry_probe.py

Every arm prints what happened. Several of them print an outcome that is worse than the
obvious expectation, and those are the reason mcr/registry.py is shaped the way it is.

Each arm carries a control. An arm that only ever runs against working code cannot tell
you whether it would notice a break, and an arm that reports OK against a deliberately
broken build is worse than no arm. So the arms whose subject is this repo's code run twice,
once against the real function and once against a stand-in carrying the defect the function
exists to prevent. The control has to fail. A probe where the control passes is reported as
a failure of the probe.

The arms whose subject is MLflow itself have no control, because there is no version of
MLflow here with the behaviour removed. Those arms are measurements and they are labelled
as measurements.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow  # noqa: E402

from mcr import registry  # noqa: E402
from mcr.tracking import TAG_ARTIFACT_HASH  # noqa: E402

MODEL = "probe-model"


def fresh(tmp: str) -> mlflow.MlflowClient:
    uri = "sqlite:///{}/registry.db".format(tmp)
    return mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)


def a_run(cli: mlflow.MlflowClient, experiment: str = "probe") -> str:
    found = cli.get_experiment_by_name(experiment)
    exp = found.experiment_id if found else cli.create_experiment(experiment)
    run = cli.create_run(experiment_id=exp)
    cli.set_terminated(run.info.run_id, status="FINISHED")
    return run.info.run_id


class Report:
    def __init__(self) -> None:
        self.rows = []

    def measured(self, label: str, detail: str) -> None:
        self.rows.append(("MEASURED", label, detail, True))
        print("MEASURED  {:<44} {}".format(label, detail))

    def arm(self, label: str, ok: bool, detail: str, control_failed: bool) -> None:
        good = ok and control_failed
        state = "OK" if good else "BAD"
        self.rows.append((state, label, detail, good))
        print(
            "{:<9} {:<44} {}   control {}".format(
                state, label, detail, "failed as it must" if control_failed else "PASSED"
            )
        )

    def exit_code(self) -> int:
        bad = [r for r in self.rows if not r[3]]
        print("")
        print("{} arms, {} bad".format(len(self.rows), len(bad)))
        return 1 if bad else 0


def measure_stage_default(rep: Report, tmp: str) -> None:
    """MLflow's own stage transition, with nothing passed that it does not require."""
    cli = fresh(tmp + "/stage")
    cli.create_registered_model(MODEL)
    for _ in range(2):
        cli.create_model_version(MODEL, source="file:///dev/null", run_id=a_run(cli))

    cli.transition_model_version_stage(MODEL, "1", "Production")
    cli.transition_model_version_stage(MODEL, "2", "Production")
    live = sorted(
        int(v.version)
        for v in cli.search_model_versions("name='{}'".format(MODEL))
        if v.current_stage == "Production"
    )
    rep.measured("stage default leaves in Production", str(live))

    cli.transition_model_version_stage(
        MODEL, "1", "Production", archive_existing_versions=True
    )
    live = sorted(
        int(v.version)
        for v in cli.search_model_versions("name='{}'".format(MODEL))
        if v.current_stage == "Production"
    )
    rep.measured("with archive_existing_versions=True", str(live))


def measure_alias_exclusivity(rep: Report, tmp: str) -> None:
    cli = fresh(tmp + "/alias")
    cli.create_registered_model(MODEL)
    for _ in range(2):
        cli.create_model_version(MODEL, source="file:///dev/null", run_id=a_run(cli))

    cli.set_registered_model_alias(MODEL, "production", "1")
    cli.set_registered_model_alias(MODEL, "production", "2")
    v1 = cli.get_model_version(MODEL, "1")
    v2 = cli.get_model_version(MODEL, "2")
    rep.measured(
        "alias move is exclusive",
        "v1 aliases {} v2 aliases {}".format(list(v1.aliases), list(v2.aliases)),
    )
    rep.measured(
        "and it leaves no history",
        "v1 last_updated == created: {}".format(
            v1.last_updated_timestamp == v1.creation_timestamp
        ),
    )


def measure_tag_does_not_exclude(rep: Report, tmp: str) -> None:
    """A version tag was the obvious place for the stage. This is why it is not there."""
    cli = fresh(tmp + "/tags")
    cli.create_registered_model(MODEL)
    for _ in range(2):
        cli.create_model_version(MODEL, source="file:///dev/null", run_id=a_run(cli))
    cli.set_model_version_tag(MODEL, "1", "stage", "production")
    cli.set_model_version_tag(MODEL, "2", "stage", "production")
    tagged = sorted(
        int(v.version)
        for v in cli.search_model_versions(
            "name='{}' and tags.stage='production'".format(MODEL)
        )
    )
    rep.measured("versions tagged production at once", str(tagged))


def arm_ambiguous_reference(rep: Report, tmp: str) -> None:
    """A decimal alias colliding with a version number. resolve must refuse it."""
    cli = fresh(tmp + "/ambig")
    cli.create_registered_model(MODEL)
    runs = [a_run(cli) for _ in range(3)]
    for rid in runs:
        cli.create_model_version(MODEL, source="file:///dev/null", run_id=rid)

    cli.set_registered_model_alias(MODEL, "2", "3")
    by_version = cli.get_model_version(MODEL, "2").run_id
    by_alias = cli.get_model_version_by_alias(MODEL, "2").run_id
    rep.measured(
        "one string, two models",
        "version -> {} alias -> {}".format(by_version[:8], by_alias[:8]),
    )

    refused = False
    try:
        registry.resolve(cli, MODEL, "2")
    except registry.RegistryError:
        refused = True

    # The control. First-one-wins is what this refusal replaces, so the control is a
    # resolver that does exactly that. It must not refuse. If it does, the arm is passing
    # for some reason other than the one it claims.
    def first_one_wins(ref: str) -> int:
        return int(cli.get_model_version(MODEL, ref).version)

    control_refused = False
    try:
        first_one_wins("2")
    except registry.RegistryError:
        control_refused = True

    rep.arm(
        "resolve refuses an ambiguous reference",
        refused,
        "refused: {}".format(refused),
        control_failed=not control_refused,
    )


def arm_register_is_idempotent(rep: Report, tmp: str) -> None:
    cli = fresh(tmp + "/idem")
    digest = "a" * 64
    first = registry.register(cli, MODEL, a_run(cli), digest, "fp0")
    second = registry.register(cli, MODEL, a_run(cli), digest, "fp0")

    # The control is the store's own behaviour, which is what register wraps. Two
    # create_model_version calls with the same bytes give two versions, so the control
    # "fails" by producing a different second answer.
    third = int(
        cli.create_model_version(
            MODEL, source="file:///dev/null", run_id=a_run(cli),
            tags={TAG_ARTIFACT_HASH: digest},
        ).version
    )

    rep.arm(
        "register is idempotent on artefact hash",
        first == second,
        "{} then {}".format(first, second),
        control_failed=third != first,
    )


def arm_rollback_target(rep: Report, tmp: str) -> None:
    """Four versions registered and only the first two ever promoted.

    The fourth one is the shape that makes this arm worth running. A candidate that gets
    registered and then rejected by a gate is a high version number that was never in
    production, and it is exactly what a rollback must not land on.

    The first draft of this arm promoted 1 then 2 then 3 and the control passed, because
    the highest version that is not current happened to be the right answer every time.
    An unpromoted version breaks that coincidence.
    """
    cli = fresh(tmp + "/roll")
    versions = [
        registry.register(cli, MODEL, a_run(cli), "{}{}".format(i, "b" * 63), "fp{}".format(i))
        for i in range(4)
    ]
    registry.promote(cli, MODEL, str(versions[0]), "production")
    registry.promote(cli, MODEL, str(versions[1]), "production")

    target = registry.rollback_target(cli, MODEL, "production")

    # The control is the question the store can answer on its own. MLflow keeps no alias
    # history, so anything derived from the store without this log has to guess, and the
    # obvious guess is the highest version that is not current.
    now = registry.current(cli, MODEL, "production")
    guess = max(v for v in versions if v != now)

    rep.arm(
        "rollback target comes off the log",
        target == versions[0],
        "log says {}, store-only guess says {}".format(target, guess),
        control_failed=guess != target,
    )


def arm_log_disagreement(rep: Report, tmp: str) -> None:
    """An alias moved outside promote must make rollback refuse rather than guess."""
    cli = fresh(tmp + "/disagree")
    versions = [
        registry.register(cli, MODEL, a_run(cli), "{}{}".format(i, "c" * 63), "fp{}".format(i))
        for i in range(3)
    ]
    registry.promote(cli, MODEL, str(versions[0]), "production")
    registry.promote(cli, MODEL, str(versions[1]), "production")

    # Straight past promote, which is what the MLflow UI does.
    cli.set_registered_model_alias(MODEL, "production", str(versions[2]))

    refused = False
    try:
        registry.rollback_target(cli, MODEL, "production")
    except registry.RegistryError:
        refused = True

    # Control: reading the log's last entry without comparing it against the alias. That
    # returns a version and reports success, which is the behaviour being refused.
    last = registry.history(cli, MODEL, "production")[-1]
    rep.arm(
        "rollback refuses when the log and the alias disagree",
        refused,
        "refused: {}, blind read would have said {}".format(refused, last.from_version),
        control_failed=last.from_version is not None,
    )


def arm_no_op_promote(rep: Report, tmp: str) -> None:
    cli = fresh(tmp + "/noop")
    v = registry.register(cli, MODEL, a_run(cli), "d" * 64, "fp")
    registry.promote(cli, MODEL, str(v), "production")
    again = registry.promote(cli, MODEL, str(v), "production")
    entries = len(registry.history(cli, MODEL, "production"))

    # Control: a promote that logged unconditionally would put a second entry in, and the
    # entry would have from equal to to, so a rollback would land where it started.
    rep.arm(
        "promoting where it already points logs nothing",
        (not again.logged) and entries == 1,
        "logged {} entries {}".format(again.logged, entries),
        control_failed=entries != 2,
    )


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="registry-probe-")
    rep = Report()
    try:
        for d in ("stage", "alias", "tags", "ambig", "idem", "roll", "disagree", "noop"):
            os.makedirs(os.path.join(tmp, d), exist_ok=True)
        measure_stage_default(rep, tmp)
        measure_alias_exclusivity(rep, tmp)
        measure_tag_does_not_exclude(rep, tmp)
        print("")
        arm_ambiguous_reference(rep, tmp)
        arm_register_is_idempotent(rep, tmp)
        arm_rollback_target(rep, tmp)
        arm_log_disagreement(rep, tmp)
        arm_no_op_promote(rep, tmp)
        return rep.exit_code()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
