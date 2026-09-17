"""Open, abort and land a canary from a command line, and say what is deployed.

    python3 scripts/deploy.py status --store sqlite:///mlflow.db
    python3 scripts/deploy.py open   --store sqlite:///mlflow.db --canary 4 --fraction 0.05
    python3 scripts/deploy.py abort  --store sqlite:///mlflow.db
    python3 scripts/deploy.py land   --store sqlite:///mlflow.db

`abort` is what a rollback verdict from scripts/canary.py asks for. It takes the canary off
traffic and does not touch production, which is the distinction that the deleted
`--rollback` flag on the canary script got wrong. Rolling production back to an older
version is still `scripts/registry.py rollback` and it is a different decision about a
different model.

Exit codes. 0 did the thing. 1 refused because the state does not allow it. 2 could not
read the store at all. A refusal is not a crash and CI branches on the difference.

Needs requirements-tracking.txt. The imports are inside main so `--help` works without it.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_MODEL = "mcr-fraud"

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_UNREADABLE = 2


def build_parser() -> argparse.ArgumentParser:
    # --store and --model hang off each subcommand rather than off the top level, so
    # `deploy.py abort --store ...` works. Putting them on the parent reads fine in a
    # docstring and then refuses that order, which is how every caller writes it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", required=True, help="mlflow tracking and registry uri")
    common.add_argument("--model", default=DEFAULT_MODEL)

    parser = argparse.ArgumentParser(description="move a canary in and out of traffic")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "status",
        parents=[common],
        help="production, canary and the share, plus any disagreement",
    )

    p = sub.add_parser(
        "open", parents=[common], help="put a version on trial at a share of traffic"
    )
    p.add_argument("--canary", required=True, help="a version number or an alias")
    p.add_argument("--fraction", type=float, required=True)
    p.add_argument(
        "--require-gate",
        action="store_true",
        help="refuse unless the gate cleared that version against what holds production",
    )

    sub.add_parser(
        "abort",
        parents=[common],
        help="take the canary off traffic, leave production alone",
    )
    sub.add_parser(
        "land", parents=[common], help="make the canary production and end the trial"
    )

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import mlflow

    from mcr import deploy, gate, registry

    try:
        cli = mlflow.MlflowClient(tracking_uri=args.store, registry_uri=args.store)
        before = deploy.deployment(cli, args.model)
    except Exception as exc:
        print("cannot read {}: {}".format(args.store, exc))
        return EXIT_UNREADABLE

    if args.command == "status":
        print(before.describe())
        problem = deploy.check_consistency(cli, args.model)
        if problem is not None:
            # Printed and not raised. A status command that refuses to describe a broken
            # state is useless in exactly the situation somebody runs it.
            print("inconsistent: {}".format(problem))
            print("`abort` clears this without moving production")
            return EXIT_REFUSED
        return EXIT_OK

    try:
        if args.command == "open":
            # The canary is the other way into production. `land` promotes whatever is on
            # trial, so a version the gate refused reaches production in two commands
            # unless the trial itself is gated. Measured on 2026-09-17, before this
            # existed: open on a NaN model, then land, and production moved.
            #
            # The check goes on `open` and not on `land` because `land` acts on the
            # canary's own evidence. What has to be true is that the trial should have
            # started at all.
            if args.require_gate:
                refusal = gate.refusal_for_version(
                    cli, registry, args.model, args.canary, deploy.PRODUCTION
                )
                if refusal is not None:
                    print("refused: {}".format(refusal), file=sys.stderr)
                    return 2
            after = deploy.open_canary(cli, args.model, args.canary, args.fraction)
        elif args.command == "abort":
            after = deploy.abort_canary(cli, args.model)
        else:
            after = deploy.land_canary(cli, args.model)
    except (deploy.DeployError, registry.RegistryError) as exc:
        print("refused: {}".format(exc))
        print("before:  {}".format(before.describe()))
        return EXIT_REFUSED

    print("before: {}".format(before.describe()))
    print("after:  {}".format(after.describe()))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
