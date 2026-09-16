"""Judge a candidate against whatever holds a stage, and print the comparison.

    python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2
    python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2 --promote
    python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2 --holdout configs/baseline.yml

Exit codes carry the verdict, because this is meant to sit in a pipeline.

    0   promote. The candidate beat the incumbent, or nothing held the stage.
    1   reject. The comparison ran and the candidate did not win it.
    2   refuse. The gate could not compare the two at all.

One and two are different on purpose. A reject is a fact about the candidate. A refuse is a
fact about the gate's inputs, and a pipeline that treats them the same will retry a broken
incumbent forever.

Every comparison writes its verdict back to the candidate's run as `gate.*` tags. That is
the audit record, and it is the default because the verdicts worth reading later are mostly
the rejections. `--promote` moves the stage and does nothing else. `--no-report` compares
without writing, for when the store must not be touched.

Both models are scored on one holdout, which this script builds. It does not read the
metrics the runs recorded, because those were measured on whatever corpus each run
generated for itself. Those numbers are still printed, in the right hand column, so the
difference between them and the shared holdout is visible.

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

EXIT_PROMOTE = 0
EXIT_REJECT = 1
EXIT_REFUSE = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="judge a candidate against the incumbent")
    p.add_argument("--store", required=True, help="mlflow tracking and registry uri")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--candidate", required=True, help="a version number or an alias")
    p.add_argument("--stage", default="production")
    p.add_argument(
        "--holdout",
        default=DEFAULT_HOLDOUT,
        help="config whose data section and seed define the rows both models are judged on",
    )
    p.add_argument(
        "--promote",
        action="store_true",
        help="move the stage when the verdict is promote",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="compare without writing the verdict back to the candidate's run",
    )
    return p


def _scored(cli, tracking, gate, version, spec, label):
    """Score one registered version on the gate's holdout."""
    recovered = tracking.recover(cli, version.run_id)
    payload = gate.rebuild(recovered.config, recovered.artifact_hash, label)
    return gate.score(
        name=recovered.config.name,
        artifact_hash=recovered.artifact_hash,
        payload=payload,
        spec=spec,
        reported=recovered.metrics,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import mlflow

    from mcr import gate, registry, tracking
    from mcr.config import load

    cli = mlflow.MlflowClient(tracking_uri=args.store, registry_uri=args.store)
    spec = gate.spec_from_config(load(args.holdout))
    fp = spec.fingerprint()

    try:
        cand_version = registry.resolve(cli, args.model, args.candidate)
        held = registry.current(cli, args.model, args.stage)
    except registry.RegistryError as exc:
        print("refused: {}".format(exc), file=sys.stderr)
        return EXIT_REFUSE

    if held == cand_version:
        print("refused: version {} already holds {}".format(cand_version, args.stage), file=sys.stderr)
        return EXIT_REFUSE

    try:
        candidate = _scored(
            cli, tracking, gate, cli.get_model_version(args.model, str(cand_version)), spec,
            "candidate version {}".format(cand_version),
        )
        incumbent = None
        if held is not None:
            incumbent = _scored(
                cli, tracking, gate, cli.get_model_version(args.model, str(held)), spec,
                "incumbent version {}".format(held),
            )
    except (gate.GateError, tracking.TrackingError) as exc:
        print("refused: {}".format(exc), file=sys.stderr)
        return EXIT_REFUSE

    decision = gate.decide(candidate, incumbent, fp)

    print("candidate    version {} of {}".format(cand_version, args.model))
    print("incumbent    {}".format(
        "version {}".format(held) if held is not None else "nothing holds " + args.stage))
    print("")
    for line in gate.report_lines(decision):
        print(line)

    if not args.no_report:
        # Every comparison lands on the candidate's run, not only the ones somebody asked
        # to promote. This used to sit inside the --promote branch, which meant the only
        # verdicts on record were the ones from a run that expected to win. A rejection
        # leaving no trace is the rejection you most want to read back six weeks later.
        run_id = cli.get_model_version(args.model, str(cand_version)).run_id
        for key, value in report_tags(decision, args.stage).items():
            cli.set_tag(run_id, key, value)

    if args.promote and decision.promoted:
        entry = registry.promote(cli, args.model, str(cand_version), args.stage)
        print("")
        print("{}: {} -> {}".format(
            args.stage,
            entry.from_version if entry.from_version is not None else "nothing",
            entry.to_version))

    if decision.verdict == gate.PROMOTE:
        return EXIT_PROMOTE
    if decision.verdict == gate.REJECT:
        return EXIT_REJECT
    return EXIT_REFUSE


def report_tags(decision, stage: str) -> dict:
    """The comparison report, flattened onto the run that produced the candidate.

    Tags rather than params, because a run is written before it is ever gated and a param
    refuses a rewrite. A candidate can be gated more than once, against different
    incumbents, and the latest verdict is the one worth reading off the run.
    """
    from mcr import gate

    tags = {
        "gate.stage": stage,
        "gate.verdict": decision.verdict,
        "gate.reason": decision.reason,
        "gate.holdout": decision.holdout,
        "gate.metric": gate.METRIC,
    }
    if decision.candidate is not None:
        tags["gate.candidate_score"] = repr(decision.candidate.metric())
    if decision.incumbent is not None:
        tags["gate.incumbent_score"] = repr(decision.incumbent.metric())
    if decision.mean_diff is not None:
        tags["gate.mean_diff"] = repr(decision.mean_diff)
    if decision.interval is not None:
        tags["gate.interval"] = "{!r},{!r}".format(*decision.interval)
    return tags


if __name__ == "__main__":
    sys.exit(main())
