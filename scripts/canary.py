"""Route a slice of traffic to a registered version and compare the arms.

    python3 scripts/canary.py --store sqlite:///mlflow.db --canary 2
    python3 scripts/canary.py --store sqlite:///mlflow.db --canary 2 --fraction 0.25
    python3 scripts/canary.py --store sqlite:///mlflow.db --canary 2 --no-shadow
    python3 scripts/canary.py --store sqlite:///mlflow.db --canary 2 --promote

Exit codes carry the verdict.

    0   promote. The canary arm was better over the served traffic.
    1   rollback. The canary arm was worse, so stop routing traffic to it.
    2   refuse. The comparison could not be made at all.
    3   hold. The comparison ran and did not separate the arms.

Four rather than the gate's three. A canary is already serving, so `hold` is a real
answer: keep it up and keep collecting. A pipeline mapping hold onto rollback tears down
a canary that had not finished answering, and one mapping it onto promote ships on noise.

THE TRAFFIC IS A REPLAY of a generated holdout. This project has no serving path and no
users. Every figure below is measured on rows a generator produced, and the split is over
request keys this script invents. What that makes real is the arithmetic of splitting a
sample, which is the thing the day is about. It does not make the loss numbers facts
about anybody's production.

Needs requirements-tracking.txt. mcr.registry is imported inside main so `--help` works
without MLflow installed.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODEL = "mcr-fraud"
DEFAULT_HOLDOUT = "configs/baseline.yml"
DEFAULT_FRACTION = 0.05

EXIT_PROMOTE = 0
EXIT_ROLLBACK = 1
EXIT_REFUSE = 2
EXIT_HOLD = 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="route a slice of traffic and compare the arms")
    p.add_argument("--store", required=True, help="mlflow tracking and registry uri")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--canary", required=True, help="a version number or an alias")
    p.add_argument("--stage", default="production")
    p.add_argument(
        "--fraction",
        type=float,
        default=DEFAULT_FRACTION,
        help="share of request keys routed to the canary arm",
    )
    p.add_argument("--salt", default="canary", help="changes which keys land in the slice")
    p.add_argument(
        "--holdout",
        default=DEFAULT_HOLDOUT,
        help="config whose data section and seed define the replayed requests",
    )
    p.add_argument(
        "--no-shadow",
        action="store_true",
        help="score only the routed arm per request, which is what a canary has when the "
        "metric depends on what was served",
    )
    p.add_argument(
        "--promote",
        action="store_true",
        help="move the stage to the canary version when the verdict is promote, and write "
        "the comparison to the run either way",
    )
    return p


def _payload(cli, tracking, gate, model, version, label):
    """Recover a version's config and rebuild the artefact the registry points at."""
    mv = cli.get_model_version(model, str(version))
    recovered = tracking.recover(cli, mv.run_id)
    payload = gate.rebuild(recovered.config, recovered.artifact_hash, label)
    return recovered, payload


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import mlflow

    from mcr import canary as canary_mod
    from mcr import gate, registry, tracking
    from mcr.config import load

    cli = mlflow.MlflowClient(tracking_uri=args.store, registry_uri=args.store)
    spec = gate.spec_from_config(load(args.holdout))
    fp = spec.fingerprint()

    try:
        router = canary_mod.Router(fraction=args.fraction, salt=args.salt)
    except canary_mod.CanaryError as exc:
        print("refused: {}".format(exc), file=sys.stderr)
        return EXIT_REFUSE

    try:
        canary_version = registry.resolve(cli, args.model, args.canary)
        held = registry.current(cli, args.model, args.stage)
    except registry.RegistryError as exc:
        print("refused: {}".format(exc), file=sys.stderr)
        return EXIT_REFUSE

    if held is None:
        print(
            "refused: nothing holds {}, so there is no control arm to route against".format(
                args.stage
            ),
            file=sys.stderr,
        )
        return EXIT_REFUSE

    try:
        can_rec, can_payload = _payload(
            cli, tracking, gate, args.model, canary_version,
            "canary version {}".format(canary_version),
        )
        con_rec, con_payload = _payload(
            cli, tracking, gate, args.model, held,
            "control version {}".format(held),
        )
    except (gate.GateError, tracking.TrackingError) as exc:
        print("refused: {}".format(exc), file=sys.stderr)
        return EXIT_REFUSE

    x, y = spec.rows()
    keys = canary_mod.replay_keys(len(y))

    try:
        arm_can, arm_con, pair = canary_mod.observe(
            router=router,
            keys=keys,
            x=x,
            y=y,
            canary_payload=can_payload,
            control_payload=con_payload,
            canary_name=can_rec.config.name,
            control_name=con_rec.config.name,
            canary_hash=can_rec.artifact_hash,
            control_hash=con_rec.artifact_hash,
            shadow=not args.no_shadow,
        )
    except canary_mod.CanaryError as exc:
        print("refused: {}".format(exc), file=sys.stderr)
        return EXIT_REFUSE

    result = canary_mod.decide(arm_can, arm_con, fp, args.fraction, pair)

    print("canary       version {} of {}".format(canary_version, args.model))
    print("control      version {}, holding {}".format(held, args.stage))
    print("")
    for line in canary_mod.report_lines(result):
        print(line)

    can, con, gap = canary_mod.slice_imbalance(router, keys, y)
    print("")
    print(
        "slice        positive rate {:.4f} against {:.4f}, gap {:+.4f}. The slice is a "
        "fixed set of keys, so this does not shrink as the canary runs longer.".format(
            can, con, gap
        )
    )

    if args.promote:
        # The comparison goes on the canary's run whatever the verdict, for the same
        # reason the gate writes a rejection: a canary that leaves no trace on the run is
        # one nobody can audit afterwards.
        run_id = cli.get_model_version(args.model, str(canary_version)).run_id
        for key, value in report_tags(result, args.stage, canary_mod).items():
            cli.set_tag(run_id, key, value)

        if result.verdict == canary_mod.PROMOTE:
            entry = registry.promote(cli, args.model, str(canary_version), args.stage)
            print("")
            print("{}: {} -> {}".format(args.stage, entry.from_version, entry.to_version))

    if result.verdict == canary_mod.ROLLBACK:
        # There is deliberately no --rollback flag, and the first draft of this script had
        # one that called registry.rollback. That was wrong in a way worth recording.
        #
        # The canary is a version being trialled. The control arm is whatever holds the
        # stage. So rolling the stage back in response to a bad canary moves production
        # off a model the canary result says nothing about, and onto whatever preceded it.
        # It reads as the obvious safety action and it is a change to the one thing that
        # was working.
        #
        # What a rollback verdict really means here is stop routing traffic to the canary,
        # and the canary's share is a command line flag rather than registry state, so
        # there is nothing in the registry to undo. The exit code is the action. Making
        # the canary's share into registry state is a later change and docs/adr-0005
        # says so.
        print("")
        print(
            "stop routing to the canary. Nothing in the registry changes, because the "
            "canary's share is a flag rather than a stage, and {} still holds {}.".format(
                held, args.stage
            )
        )

    if result.verdict == canary_mod.PROMOTE:
        return EXIT_PROMOTE
    if result.verdict == canary_mod.ROLLBACK:
        return EXIT_ROLLBACK
    if result.verdict == canary_mod.HOLD:
        return EXIT_HOLD
    return EXIT_REFUSE


def report_tags(result, stage: str, canary_mod) -> dict:
    """The canary comparison, flattened onto the run that produced the canary version.

    Tags rather than params, for the reason the gate uses tags: a run is written before it
    is ever canaried and a param refuses a rewrite, and one version can be canaried more
    than once at different fractions.

    Both intervals go on the run. Recording only the one the verdict came from would
    throw away the fact that made the day worth having, which is that they disagree.
    """
    tags = {
        "canary.stage": stage,
        "canary.verdict": result.verdict,
        "canary.reason": result.reason,
        "canary.holdout": result.holdout,
        "canary.metric": canary_mod.METRIC,
        "canary.fraction": repr(result.fraction),
    }
    if result.canary is not None:
        tags["canary.requests"] = repr(result.canary.n)
    if result.control is not None:
        tags["canary.control_requests"] = repr(result.control.n)
    if result.split is not None:
        tags["canary.split"] = "{!r},{!r},{!r}".format(*result.split)
    if result.shadow is not None:
        tags["canary.shadow"] = "{!r},{!r},{!r}".format(*result.shadow)
    if result.required is not None:
        tags["canary.required_rows"] = repr(result.required)
    return tags


if __name__ == "__main__":
    sys.exit(main())
