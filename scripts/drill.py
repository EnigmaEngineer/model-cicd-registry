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

import json
import os
import shutil
import sys
import tempfile
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow  # noqa: E402

from mcr import attest, deploy, registry  # noqa: E402

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
        registry.create_version(cli, MODEL, run.info.run_id, "hash%04d" % i, "fp%04d" % i)
    return cli


def half_promote(cli: mlflow.MlflowClient, to: int, frm: Optional[int], stage: str) -> None:
    """The first half of registry.promote and nothing else.

    This is the crash. The log entry lands and the process dies before the alias moves,
    which is the failure mode the write ordering in promote deliberately chose to have.
    """
    tags = dict(cli.get_registered_model(MODEL).tags)
    entry = registry.Transition(stage=stage, to_version=to, from_version=frm, at_ms=1)
    key = registry._slot(stage, registry._next_seq(tags, stage))
    # Sealed, because promote seals before it writes the tag. A crash lands after that, so
    # an unsealed phantom would be modelling a different event and would be refused as
    # tampering rather than recovered as a crash.
    cli.set_registered_model_tag(
        MODEL, key, attest.seal(key, entry.as_value(), attest.key_from_env())
    )


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


def _forge(cli: mlflow.MlflowClient, seq: int, value: str) -> None:
    """Overwrite one history slot with something this process did not write.

    A registered model tag write is unconditional, so this is the whole attack. It needs no
    privilege the deploy pipeline does not already have.
    """
    cli.set_registered_model_tag(MODEL, registry._slot("production", seq), value)


def _signed_three(tmp: str, slot: str) -> mlflow.MlflowClient:
    cli = fresh(tmp, slot)
    for v in ("1", "2", "3"):
        registry.promote(cli, MODEL, v, "production")
    return cli


def _wrong_order(cli: mlflow.MlflowClient) -> Tuple[str, int]:
    """Recovery first, then the integrity check. The order this code does not use.

    Both halves are the shipped functions rather than a second copy of them, so the only
    difference between this and `rollback_target` is which one runs first.
    """
    raw = registry.history(cli, MODEL, "production")
    now = registry.current(cli, MODEL, "production")
    kept, _ = registry.settle(raw, now)
    trusted, rejected = registry.authentic(kept)
    return str(registry.walk_back(trusted, now)), len(rejected)


def drill_forged_entry_evicts_a_genuine_one(d: Drills, tmp: str) -> None:
    """A tampered entry, and the reason the two filters run in the order they do.

    The forged entry claims a move to a version that was never promoted. That breaks the
    witness rule for the genuine entry beside it, so a recovery pass drops the genuine one
    and keeps a log that reads as healthy. The rollback target moves and nothing refuses.

    The third row is the one worth having. By the time a recovery first pass gets round to
    asking who wrote the log, the forged entry has already been discarded as a crash
    artefact, so the integrity check has nothing left to object to.
    """
    os.environ[attest.ENV_KEY] = "drill-key"
    try:
        cli = _signed_three(tmp, "f1a")
        phantom = registry.Transition(
            stage="production", to_version=9, from_version=1, at_ms=1
        )
        _forge(cli, 1, json.dumps(
            {"body": phantom.as_value(), "mac": "00" * 32},
            sort_keys=True, separators=(",", ":"),
        ))
        damaged = target_or_refusal(cli)
        wrong_order, rejected_late = _wrong_order(cli)

        control = target_or_refusal(_signed_three(tmp, "f1b"))

        d.refusal("forged log entry refused", damaged, control, "2")
        d.invariant("recovery first answers instead of refusing", wrong_order, "1")
        d.invariant(
            "because settle dropped the evidence first",
            "{} rejected after settle".format(rejected_late),
            "0 rejected after settle",
        )
    finally:
        os.environ.pop(attest.ENV_KEY, None)


def drill_unsigned_entry_in_a_signed_log(d: Drills, tmp: str) -> None:
    """The cheapest forgery there is, which is to leave the signature off.

    This is the one that decided the policy. An unsigned entry was accepted by the first
    rule written here, which made the signature worth nothing against anybody who could
    write a tag, because leaving it off was easier than faking it.
    """
    os.environ[attest.ENV_KEY] = "drill-key"
    try:
        cli = _signed_three(tmp, "f2a")
        phantom = registry.Transition(
            stage="production", to_version=9, from_version=1, at_ms=1
        )
        _forge(cli, 1, phantom.as_value())
        damaged = target_or_refusal(cli)
        wrong_order, _ = _wrong_order(cli)
        control = target_or_refusal(_signed_three(tmp, "f2b"))

        d.refusal("unsigned entry in a signed log refused", damaged, control, "2")
        d.invariant("and the wrong order still answers", wrong_order, "1")
    finally:
        os.environ.pop(attest.ENV_KEY, None)


def drill_genuine_entry_moved_to_another_slot(d: Drills, tmp: str) -> None:
    """Replay. The entry is real and this process wrote it, into a different slot.

    The log is ordered by that slot, so moving an entry changes which version the walk
    reaches. The slot is signed material for this reason and nothing else.
    """
    os.environ[attest.ENV_KEY] = "drill-key"
    try:
        cli = _signed_three(tmp, "f3a")
        tags = dict(cli.get_registered_model(MODEL).tags)
        _forge(cli, 3, tags[registry._slot("production", 2)])
        damaged = target_or_refusal(cli)
        control = target_or_refusal(_signed_three(tmp, "f3b"))
        d.refusal("replayed entry refused", damaged, control, "2")
    finally:
        os.environ.pop(attest.ENV_KEY, None)


def drill_an_unsigned_log_still_reads(d: Drills, tmp: str) -> None:
    """The control on the policy. A history written before any of this still answers.

    Without this row the three above would pass under a rule that refused every log, which
    would be the gate passing on nothing that this repo has now found in seven places.
    """
    cli = _signed_three(tmp, "f4a")
    states = {e.attestation for e in registry.history(cli, MODEL, "production")}
    d.invariant("an unsigned log answers normally", target_or_refusal(cli), "2")
    d.invariant(
        "and it really was unsigned",
        ",".join(sorted(states)),
        attest.UNATTESTED,
    )


def drill_a_wholly_unsigned_log_is_undefended(d: Drills, tmp: str) -> None:
    """The hole in the default policy, measured rather than described.

    The no downgrade rule needs an entry it verified before it can call an unsigned one a
    downgrade. A model the pipeline has never promoted offers no such entry, so a log
    written end to end by somebody else is read as a legacy log and answered normally.

    Both rows below are the weakness rather than the defence. The first is what the default
    does and the second is the setting that refuses it.
    """
    os.environ[attest.ENV_KEY] = "drill-key"
    try:
        cli = fresh(tmp, "f5a")
        for seq, (to, frm) in enumerate(((1, None), (2, 1), (3, 2))):
            entry = registry.Transition(
                stage="production", to_version=to, from_version=frm, at_ms=1
            )
            _forge(cli, seq, entry.as_value())
        cli.set_registered_model_alias(MODEL, "production", "3")

        raw = registry.history(cli, MODEL, "production")
        d.invariant("a log nobody signed answers anyway", target_or_refusal(cli), "2")
        d.invariant(
            "and only require_attested refuses it",
            "{} of {} rejected".format(
                len(registry.authentic(raw, require_attested=True)[1]), len(raw)
            ),
            "3 of 3 rejected",
        )
    finally:
        os.environ.pop(attest.ENV_KEY, None)


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
        drill_forged_entry_evicts_a_genuine_one(d, tmp)
        drill_unsigned_entry_in_a_signed_log(d, tmp)
        drill_genuine_entry_moved_to_another_slot(d, tmp)
        drill_an_unsigned_log_still_reads(d, tmp)
        drill_a_wholly_unsigned_log_is_undefended(d, tmp)
        print("")
        drill_canary_open_crashed(d, tmp)
        drill_canary_land_crashed(d, tmp)
        drill_abort_leaves_production(d, tmp)
        return d.exit_code()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
