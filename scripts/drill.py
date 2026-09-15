"""Break the deployment on purpose and see what the recovery path does.

    python3 scripts/drill.py

The obvious way to test a rollback is under load. There is no serving process anywhere in
this project and no users, so load is not a thing that can be measured here, and pretending
otherwise would be the fabrication this whole repo is built to avoid.
What is real is the state machine. A rollback is a decision taken from a log against a
store, and both can be damaged. So the drills damage them.

Every drill is a pair. The damaged arm puts the store into a state a crash or a stray hand
could really produce, and the control runs the identical sequence with the damage left out.
The control must come back with the healthy answer. A drill whose control also reports the
failure has proved nothing, because then the sequence itself was broken rather than the
damage, and that mistake has shipped here before under a different name.

Two of the crash drills were how the settle rule in mcr/registry.py got written. Before it,
drill two deployed a version that had never served a single request.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow  # noqa: E402

from mcr import deploy, registry  # noqa: E402

MODEL = "drill-model"


class Drills:
    """One row per drill. The exit code is what CI reads."""

    def __init__(self) -> None:
        self.rows: List[Tuple[str, str, bool]] = []

    def refusal(self, label: str, damaged: str, control: str, want_control: str) -> None:
        """Damage that cannot be recovered. It has to refuse, and the control must not."""
        ok = control == want_control and damaged != control
        self.rows.append((label, damaged, ok))
        print(
            "{:<6} {:<38} damaged: {:<30} control: {}".format(
                "ok" if ok else "BAD", label, damaged, control
            )
        )
        if control != want_control:
            print(
                "       control gave '{}' and should have given '{}', so this drill "
                "graded nothing".format(control, want_control)
            )

    def recovery(self, label: str, naive: str, settled: str, control: str) -> None:
        """Damage the settle rule undoes.

        Three values rather than two, because the healthy answer alone proves nothing. The
        settled arm has to agree with the undamaged control, and the naive arm has to
        disagree with both. Without that third reading this drill would pass whether or not
        settle ran at all, which is a shape that has shipped green here before.
        """
        ok = settled == control and naive != control
        self.rows.append((label, settled, ok))
        print(
            "{:<6} {:<38} raw log: {:<8} settled: {:<8} control: {}".format(
                "ok" if ok else "BAD", label, naive, settled, control
            )
        )
        if naive == control:
            print(
                "       the raw log already gave the right answer, so this drill did not "
                "reach the thing it is about"
            )

    def invariant(self, label: str, got: str, want: str) -> None:
        """A plain fact about the state a repair left behind."""
        ok = got == want
        self.rows.append((label, got, ok))
        print(
            "{:<6} {:<38} {} (wanted {})".format(
                "ok" if ok else "BAD", label, got, want
            )
        )

    def exit_code(self) -> int:
        bad = [r for r in self.rows if not r[2]]
        print("")
        print("{} drills, {} bad".format(len(self.rows), len(bad)))
        return 1 if bad else 0


def fresh(tmp: str, slot: str, versions: int = 5) -> mlflow.MlflowClient:
    path = os.path.join(tmp, slot)
    os.makedirs(path, exist_ok=True)
    uri = "sqlite:///{}/drill.db".format(path)
    cli = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    registry.ensure_model(cli, MODEL)
    exp = cli.create_experiment("drill")
    for i in range(1, versions + 1):
        run = cli.create_run(experiment_id=exp)
        cli.set_terminated(run.info.run_id, status="FINISHED")
        registry.register(cli, MODEL, run.info.run_id, "hash%04d" % i, "fp%04d" % i)
    return cli


def half_promote(cli: mlflow.MlflowClient, to: int, frm: Optional[int], stage: str) -> None:
    """The first half of registry.promote and nothing else.

    This is the crash. The log entry lands and the process dies before the alias moves,
    which is the failure mode the write ordering in promote deliberately chose to have.
    """
    tags = dict(cli.get_registered_model(MODEL).tags)
    entry = registry.Transition(stage=stage, to_version=to, from_version=frm, at_ms=1)
    key = "{}.{}.{:0{}d}".format(
        registry.HISTORY_PREFIX, stage, registry._next_seq(tags, stage), registry.SEQ_WIDTH
    )
    cli.set_registered_model_tag(MODEL, key, entry.as_value())


def target_or_refusal(cli: mlflow.MlflowClient, stage: str = "production") -> str:
    try:
        return str(registry.rollback_target(cli, MODEL, stage))
    except registry.RegistryError:
        return "refused"


def naive_target(cli: mlflow.MlflowClient, stage: str = "production") -> str:
    """What the walk answers when it is handed the raw log.

    This is the shipped walk over an unsettled log rather than a second implementation of
    it, so the only difference between this and `rollback_target` is the settle call.
    """
    now = registry.current(cli, MODEL, stage)
    return str(registry.walk_back(registry.history(cli, MODEL, stage), now))


def drill_crash_same_version_retried(d: Drills, tmp: str) -> None:
    """Crash writing the entry for version 4, then retry version 4 and succeed.

    This one says where the defect stops. Retrying the same version leaves a duplicate
    entry in the raw log and the walk still lands on the right answer, because the version
    the phantom entry names is the version that went on to serve. So the crash on its own
    is harmless and it only becomes a wrong deploy once the retry moves somewhere else,
    which is drill two.

    The check is not that the answer is right, because it is right either way and a drill
    asserting that would pass with settle deleted. It is that settle dropped an entry while
    the answer stayed put.
    """
    cli = fresh(tmp, "c1a")
    for v in ("1", "2", "3"):
        registry.promote(cli, MODEL, v, "production")
    half_promote(cli, 4, 3, "production")
    registry.promote(cli, MODEL, "4", "production")

    raw = registry.history(cli, MODEL, "production")
    kept, dropped = registry.settle(raw, registry.current(cli, MODEL, "production"))

    d.invariant("crash then retry same version, answer", target_or_refusal(cli), "3")
    d.invariant("raw log agrees here, so no damage", naive_target(cli), "3")
    d.invariant(
        "and settle still dropped the phantom",
        "{} raw, {} kept, {} dropped".format(len(raw), len(kept), len(dropped)),
        "5 raw, 4 kept, 1 dropped",
    )


def drill_crash_different_version_retried(d: Drills, tmp: str) -> None:
    """Crash writing the entry for 4, work out that 4 was the problem, ship 5 instead.

    This is the one that mattered. Version 4 never served a request. It existed only as a
    log entry left behind by the crash, and the raw log gives no sign of that once the
    retry has written a correct entry on top of it.
    """
    cli = fresh(tmp, "c2a")
    for v in ("1", "2", "3"):
        registry.promote(cli, MODEL, v, "production")
    half_promote(cli, 4, 3, "production")
    registry.promote(cli, MODEL, "5", "production")

    ctl = fresh(tmp, "c2b")
    for v in ("1", "2", "3", "5"):
        registry.promote(ctl, MODEL, v, "production")

    d.recovery(
        "crash then ship a different version",
        naive_target(cli),
        target_or_refusal(cli),
        target_or_refusal(ctl),
    )


def drill_crash_after_alias_moved(d: Drills, tmp: str) -> None:
    """The other crash order. The alias moved and nothing recorded it.

    Refusing is the right answer and it is not a recovery. The version that served is not
    in the log at all, so there is nothing to settle and no way to tell a promote from a
    rollback after the fact.
    """
    for slot, damage in (("c3a", True), ("c3b", False)):
        cli = fresh(tmp, slot)
        for v in ("1", "2", "3"):
            registry.promote(cli, MODEL, v, "production")
        if damage:
            cli.set_registered_model_alias(MODEL, "production", "4")
            damaged = target_or_refusal(cli)
        else:
            control = target_or_refusal(cli)
    d.refusal("crash after the alias moved", damaged, control, "2")


def drill_out_of_band_alias(d: Drills, tmp: str) -> None:
    """Somebody moves the alias in the MLflow UI. The log stops describing the store."""
    for slot, damage in (("c4a", True), ("c4b", False)):
        cli = fresh(tmp, slot)
        for v in ("1", "2", "3"):
            registry.promote(cli, MODEL, v, "production")
        if damage:
            cli.set_registered_model_alias(MODEL, "production", "1")
            damaged = target_or_refusal(cli)
        else:
            control = target_or_refusal(cli)
    d.refusal("alias moved outside promote", damaged, control, "2")


def drill_rollback_does_not_oscillate(d: Drills, tmp: str) -> None:
    """Two rollbacks in a row must keep walking back rather than bouncing.

    The naive rule is "whatever the stage held immediately before", and after one rollback
    that is the version just rolled away from. The control here is the naive rule computed
    by hand, so the drill has something to be better than.
    """
    cli = fresh(tmp, "c5a")
    for v in ("1", "2", "3"):
        registry.promote(cli, MODEL, v, "production")
    registry.rollback(cli, MODEL, "production")
    registry.rollback(cli, MODEL, "production")
    damaged = str(registry.current(cli, MODEL, "production"))

    naive = fresh(tmp, "c5b")
    for v in ("1", "2", "3"):
        registry.promote(naive, MODEL, v, "production")
    for _ in range(2):
        log = registry.history(naive, MODEL, "production")
        now = registry.current(naive, MODEL, "production")
        prior = [e.to_version for e in log if e.to_version != now]
        registry.promote(naive, MODEL, str(prior[-1]), "production")
    control = str(registry.current(naive, MODEL, "production"))

    # The naive rule lands back on 3, which is the version the first rollback fled.
    d.refusal("two rollbacks reach the oldest", damaged, control, "3")


def drill_canary_open_crashed(d: Drills, tmp: str) -> None:
    """Crash between the fraction write and the canary alias. Abort is the repair."""
    cli = fresh(tmp, "c6a")
    registry.promote(cli, MODEL, "2", "production")
    cli.set_registered_model_tag(MODEL, deploy.FRACTION_TAG, "0.25")
    before = deploy.check_consistency(cli, MODEL) or "consistent"
    deploy.abort_canary(cli, MODEL)
    after = deploy.check_consistency(cli, MODEL) or "consistent"
    d.invariant("canary open crashed, then repaired", after, "consistent")
    d.invariant(
        "and it was really broken first",
        "broken" if before != "consistent" else "consistent",
        "broken",
    )


def drill_canary_land_crashed(d: Drills, tmp: str) -> None:
    """Crash after production moved and before the canary alias came off.

    Both aliases sit on one version. The repair must not move production, because
    production is already where it was meant to end up.
    """
    cli = fresh(tmp, "c7a")
    registry.promote(cli, MODEL, "2", "production")
    deploy.open_canary(cli, MODEL, "3", 0.25)
    registry.promote(cli, MODEL, "3", "production")
    before = deploy.check_consistency(cli, MODEL) or "consistent"
    deploy.abort_canary(cli, MODEL)
    after = deploy.check_consistency(cli, MODEL) or "consistent"
    d.invariant("canary land crashed, then repaired", after, "consistent")
    d.invariant(
        "and it was really broken first",
        "broken" if before != "consistent" else "consistent",
        "broken",
    )
    d.invariant(
        "that repair left production alone",
        str(registry.current(cli, MODEL, "production")),
        "3",
    )


def drill_abort_leaves_production(d: Drills, tmp: str) -> None:
    """The verdict the deleted --rollback flag got wrong.

    A rollback verdict about a canary must not move production. The damaged arm is what
    the old flag did, which is registry.rollback, and the control is what abort does.
    """
    cli = fresh(tmp, "c8a")
    for v in ("1", "2"):
        registry.promote(cli, MODEL, v, "production")
    deploy.open_canary(cli, MODEL, "3", 0.25)
    registry.rollback(cli, MODEL, "production")
    damaged = str(registry.current(cli, MODEL, "production"))

    other = fresh(tmp, "c8b")
    for v in ("1", "2"):
        registry.promote(other, MODEL, v, "production")
    deploy.open_canary(other, MODEL, "3", 0.25)
    deploy.abort_canary(other, MODEL)
    control = str(registry.current(other, MODEL, "production"))

    d.refusal("aborting a canary leaves production", damaged, control, "2")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="drill-")
    d = Drills()
    try:
        drill_crash_same_version_retried(d, tmp)
        drill_crash_different_version_retried(d, tmp)
        drill_crash_after_alias_moved(d, tmp)
        drill_out_of_band_alias(d, tmp)
        drill_rollback_does_not_oscillate(d, tmp)
        print("")
        drill_canary_open_crashed(d, tmp)
        drill_canary_land_crashed(d, tmp)
        drill_abort_leaves_production(d, tmp)
        return d.exit_code()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
