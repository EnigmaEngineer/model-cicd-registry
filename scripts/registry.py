"""Read and move the registry from a command line.

    python3 scripts/registry.py --store sqlite:///mlflow.db list
    python3 scripts/registry.py --store sqlite:///mlflow.db promote 3 production
    python3 scripts/registry.py --store sqlite:///mlflow.db history production
    python3 scripts/registry.py --store sqlite:///mlflow.db rollback production
    python3 scripts/registry.py --store sqlite:///mlflow.db report 2

`rollback` is the one command the project promises. It reads the transition log, finds what
production held before whatever it holds now, and points production back at it. It refuses
rather than guessing if the log does not describe the store, which happens the moment
somebody moves the alias in the MLflow UI.

Needs requirements-tracking.txt. mcr.registry is imported inside main for that reason, so
`--help` works on an install that has no MLflow in it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODEL = "mcr-fraud"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="read and move the model registry")
    parser.add_argument("--store", required=True, help="mlflow tracking and registry uri")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="every version, with the stage it holds")

    p = sub.add_parser("promote", help="point a stage at a version")
    p.add_argument("ref", help="a version number or the stage that currently holds it")
    p.add_argument("stage")
    p.add_argument(
        "--require-gate",
        action="store_true",
        help="refuse unless the gate's last verdict on that version was promote",
    )

    p = sub.add_parser("history", help="every transition of one stage, oldest first")
    p.add_argument("stage")

    p = sub.add_parser("rollback", help="put a stage back on the version it held before")
    p.add_argument("stage")

    p = sub.add_parser("report", help="the gate's last verdict on one version")
    p.add_argument("ref", help="a version number or the stage that currently holds it")

    return parser


GATE_VERDICT = "gate.verdict"
GATE_REASON = "gate.reason"
GATE_STAGE = "gate.stage"


def gate_refusal(cli, registry, model, ref, stage):
    """Why this version must not be promoted, or None if the gate cleared it.

    Asked for by `--require-gate` and off by default. Both halves of that need a reason.

    Why it exists. `promote` moves an alias. It does not know what a holdout is and it has
    never read a verdict, so until this flag a pipeline could promote a version the gate
    had just refused, and the transition log would record an ordinary promotion. Raised by
    a reader on 2026-09-17 and demonstrated on a clean store: a config that trains to a NaN
    is refused by the gate and promoted by hand in the next command.

    Why it is not the default, and why it is not in `mcr/registry.promote`. `rollback`
    calls `promote`. A version being rolled back to was gated against whatever was
    incumbent at the time, which is not what is incumbent now, so enforcing this in the
    library makes the recovery path depend on a stale verdict. The stage is the wrong
    place for a policy about how a stage is reached.

    What this is not. The verdict lives in MLflow tags and a tag overwrites silently, so
    anybody who can promote can also write `gate.verdict=promote`. This stops automation
    promoting something the gate refused. It does not stop a writer who means to. The
    transition log has the same property and `mcr/registry.settle` says so too.
    """
    version = registry.resolve(cli, model, ref)
    run_id = cli.get_model_version(model, str(version)).run_id
    tags = cli.get_run(run_id).data.tags

    verdict = tags.get(GATE_VERDICT)
    if verdict is None:
        return "version {} has never been gated, so there is no verdict to honour".format(
            version
        )
    if verdict != "promote":
        return "the gate's last verdict on version {} was {} ({})".format(
            version, verdict, tags.get(GATE_REASON, "no reason recorded")
        )

    # A verdict earned against one stage says nothing about another. Promoting to
    # production on the strength of a staging comparison is the same hole one level down.
    gated_stage = tags.get(GATE_STAGE)
    if gated_stage != stage:
        return (
            "version {} was gated against {} and this promotes it to {}".format(
                version, gated_stage or "an unrecorded stage", stage
            )
        )
    return None


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import mlflow

    from mcr import registry

    cli = mlflow.MlflowClient(tracking_uri=args.store, registry_uri=args.store)

    if args.command == "list":
        holders = {}
        for stage in registry.STAGES:
            v = registry.current(cli, args.model, stage)
            if v is not None:
                holders.setdefault(v, []).append(stage)
        rows = registry.versions(cli, args.model)
        if not rows:
            print("no versions of {}".format(args.model))
            return 0
        print("{:>7}  {:<12}  {:<18}  {}".format("version", "config", "stage", "artifact"))
        for v in rows:
            n = int(v.version)
            print(
                "{:>7}  {:<12}  {:<18}  {}".format(
                    n,
                    v.tags.get("config_fingerprint", "?"),
                    ",".join(holders.get(n, [])) or "-",
                    v.tags.get("artifact_hash", "?")[:16],
                )
            )
        return 0

    try:
        if args.command == "promote":
            if args.require_gate:
                refusal = gate_refusal(cli, registry, args.model, args.ref, args.stage)
                if refusal is not None:
                    print("refused: {}".format(refusal), file=sys.stderr)
                    return 2
            entry = registry.promote(cli, args.model, args.ref, args.stage)
            if not entry.logged:
                print("{} already points at version {}".format(args.stage, entry.to_version))
                return 0
            print(
                "{}: {} -> {}".format(
                    args.stage,
                    entry.from_version if entry.from_version is not None else "nothing",
                    entry.to_version,
                )
            )
            return 0

        if args.command == "history":
            log = registry.history(cli, args.model, args.stage)
            if not log:
                print("{} has never been pointed at anything".format(args.stage))
                return 0
            for e in log:
                print(
                    "{}  {} -> {}".format(
                        e.at_ms, e.from_version if e.from_version is not None else "-",
                        e.to_version
                    )
                )
            return 0

        if args.command == "rollback":
            entry = registry.rollback(cli, args.model, args.stage)
            print("{}: rolled back to version {}".format(args.stage, entry.to_version))
            return 0

        if args.command == "report":
            version = registry.resolve(cli, args.model, args.ref)
            run_id = cli.get_model_version(args.model, str(version)).run_id
            tags = {
                k: v for k, v in cli.get_run(run_id).data.tags.items()
                if k.startswith("gate.")
            }
            if not tags:
                # Not an error. A version nothing has gated yet is the normal state of a
                # freshly registered model, and saying so beats printing an empty block.
                print("version {} has never been gated".format(version))
                return 0
            for key in sorted(tags):
                print("{} = {}".format(key, tags[key]))
            return 0
    except registry.RegistryError as exc:
        print("refused: {}".format(exc), file=sys.stderr)
        return 2

    raise AssertionError("argparse accepted a command nothing handles")


if __name__ == "__main__":
    sys.exit(main())
