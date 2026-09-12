"""Can the tracking store rebuild the model, or does it only remember the score.

    python3 scripts/track_probe.py
    python3 scripts/track_probe.py configs/candidate-lr.yml --uri sqlite:///mlflow.db

scripts/repro_probe.py asks whether two runs of one config agree. That is a question about
the pipeline. This asks whether a run written into MLflow can be read back and retrained to
the same bytes, which is a question about the store. It is the one the promotion gate rests
on. The gate will never hold the config that trained the incumbent. It will hold a row in a
database.

Eight arms. Four are the round trip and four are controls, because a round trip check on
its own passes against a store that records nothing and a rebuild path that quietly fills
in defaults.

What this script cannot see, stated here rather than discovered later. Every arm loads a
config off disk and pushes it through the store, and MLflow hands every value back as a
string, so the rebuild only ever exercises the string branch of `config.coerce`. The
typing defect in docs/adr-0002 lives on the YAML load path and this script is structurally
blind to it. That is not a guess. The float rule was reverted deliberately and all seven
arms of the first version printed OK with `"recoverable": true`, while
`tests/test_config.py` failed on the check written for it. The suite owns that half.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcr import tracking  # noqa: E402
from mcr import train as train_mod  # noqa: E402
from mcr.config import ConfigError, load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _line(label, ok, detail):
    print("{:<40} {:<8} {}".format(label, "OK" if ok else "FAIL", detail))
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", default=os.path.join(ROOT, "configs", "baseline.yml")
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="mlflow tracking uri, default is a throwaway sqlite file",
    )
    parser.add_argument("--experiment", default="track-probe")
    args = parser.parse_args(argv)

    tmp = None
    uri = args.uri
    if uri is None:
        tmp = tempfile.mkdtemp(prefix="mcr-track-")
        uri = "sqlite:///{}".format(os.path.join(tmp, "mlflow.db"))

    cfg = load(args.config)
    result = train_mod.run(cfg)
    cli = tracking.client(uri)

    print("config {}  fingerprint {}".format(cfg.name, cfg.fingerprint()))
    print("artifact {}".format(result.content_hash))
    print("store    {}\n".format(uri))

    run_id = tracking.log_training_run(
        cli, cfg, result.content_hash, result.metrics, experiment=args.experiment
    )
    ok = True

    got = tracking.recover(cli, run_id)
    ok &= _line(
        "config rebuilds from stored params",
        got.config == cfg,
        "fingerprint {}".format(got.config_fingerprint),
    )

    # The arm this whole script exists for. Retrain from the recovered config and see
    # whether the bytes come back. Anything less than this is a dashboard.
    rebuilt = train_mod.run(got.config)
    ok &= _line(
        "retrain from the store matches bytes",
        rebuilt.content_hash == result.content_hash,
        "{} vs tagged {}".format(rebuilt.content_hash[:12], got.artifact_hash[:12]),
    )

    # Metrics are doubles in the sqlite backend and they come back exact. Worth an arm
    # because the two configs this repo ships differ on holdout log loss by 1.806e-11, so a
    # store that rounded on the way in would leave the gate comparing equal numbers.
    exact = {k: v for k, v in result.metrics.items() if got.metrics.get(k) != v}
    ok &= _line(
        "metrics round trip exactly",
        not exact,
        "{} metrics, {} inexact".format(len(result.metrics), len(exact)),
    )

    # The one invariant holding this boundary together. It is here because of what happened
    # when the coercion rule was reverted on purpose, which the module docstring describes.
    # MLflow hands back strings and nothing else, so `config_from_params` only ever reaches
    # the string branch of `coerce`. If a value ever arrived already typed, the rebuild would
    # take a different branch and the fingerprint would start depending on which one.
    types = sorted({type(v).__name__ for v in tracking.params_for(cfg).values()})
    ok &= _line(
        "every param crosses as a string",
        types == ["str"],
        "value types {}".format(types),
    )

    # Control one. A store that dropped a param would still pass every arm above as long
    # as the rebuild filled the gap from a default.
    thin = dict(tracking.params_for(cfg))
    dropped = thin.pop("model.epochs")
    try:
        tracking.config_from_params(thin)
        ok &= _line("control: a missing param is refused", False, "accepted it")
    except tracking.TrackingError as exc:
        ok &= _line(
            "control: a missing param is refused", True, str(exc).split(":")[-1].strip()
        )

    # Control two. The params have to carry the numbers rather than the shape. Move one
    # value and the rebuilt config must stop matching.
    bent = dict(tracking.params_for(cfg))
    bent["model.epochs"] = str(int(dropped) + 1)
    ok &= _line(
        "control: a changed param changes the fit",
        tracking.config_from_params(bent).fingerprint() != cfg.fingerprint(),
        "epochs {} -> {}".format(dropped, bent["model.epochs"]),
    )

    # Control three. Two runs of one recipe are two runs. If a run id were the identity of
    # a model then the registry could key on it, and it cannot.
    second = tracking.log_training_run(
        cli, cfg, result.content_hash, result.metrics, experiment=args.experiment
    )
    found = tracking.runs_for_fingerprint(cli, cfg.fingerprint(), experiment=args.experiment)
    ok &= _line(
        "control: one recipe, two run ids",
        second != run_id and len(found) == 2,
        "{} runs tagged {}".format(len(found), cfg.fingerprint()),
    )

    # Control four. A param the store hands back is a string, and the rebuild leans on the
    # config loader to put the type back. If a junk string were accepted the fingerprint
    # would be a hash of whatever the store happened to contain.
    junk = dict(tracking.params_for(cfg))
    junk["data.noise"] = "nine"
    try:
        tracking.config_from_params(junk)
        ok &= _line("control: junk param value is refused", False, "accepted 'nine'")
    except ConfigError as exc:
        ok &= _line("control: junk param value is refused", True, str(exc))

    print("")
    summary = {
        "config": cfg.name,
        "fingerprint": cfg.fingerprint(),
        "artifact_hash": result.content_hash,
        "run_id": run_id,
        "params_logged": len(tracking.params_for(cfg)),
        "metrics_logged": len(result.metrics),
        "recoverable": ok,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
