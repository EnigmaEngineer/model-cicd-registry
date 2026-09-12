"""Pipeline checks.

The reproducibility claim this whole day rests on is asserted here, and so is the control
that stops it being vacuous.
"""

from __future__ import annotations

import os

from mcr import train as T
from mcr.config import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _baseline():
    return load(os.path.join(ROOT, "configs", "baseline.yml"))


def _small():
    """A cut down config so the suite stays quick. The shipped one is 20,000 rows over
    400 epochs and this file runs it several times."""
    from mcr.config import from_dict

    return from_dict(
        {
            "name": "small",
            "seed": 5,
            "data": {
                "n_rows": 2000,
                "n_features": 6,
                "positive_rate": 0.2,
                "noise": 1.0,
                "holdout_frac": 0.25,
            },
            "model": {
                "learning_rate": 0.5,
                "epochs": 60,
                "l2": 0.001,
                "init": "zeros",
            },
        }
    )


def check_same_config_gives_the_same_bytes():
    cfg = _small()
    assert T.run(cfg).content_hash == T.run(cfg).content_hash


def check_a_different_seed_gives_different_bytes():
    """The control. Without it, a pipeline that ignored the seed entirely would pass the
    check above perfectly and the reproducibility claim would mean nothing."""
    from dataclasses import replace

    a = T.run(_small())
    b = T.run(replace(_small(), seed=6))
    assert a.content_hash != b.content_hash


def check_the_baseline_artefact_hash_is_the_pinned_one():
    """The only check here that is not circular, and the day's mutation pass is why.

    Every other reproducibility check runs the pipeline twice and compares the two
    results. All of them pass against a pipeline that computes the wrong thing
    consistently. A mutant flipped the sign of the noise term, from `centred @ w + noise`
    to `- noise`. That changes every row of the corpus and every byte of the artefact. The
    whole suite stayed green. The noise is symmetric, so nothing distributional moves.

    The consequence is not cosmetic. Every model ever stored under the old corpus becomes
    unreproducible the moment that edit lands, and the registry would be pointing at
    artefacts nobody can rebuild.

    So one external value gets pinned. Measured on this machine today.

    When this fails after a deliberate change to the generator or the model, that is
    correct and the new hash goes here in the same commit. When it fails and nobody meant
    to change either, that is the check doing its job.
    """
    assert T.run(_baseline()).content_hash == (
        "8b83b8d77ac9e226e996859a28e5725e80d6ba657e6a5eda7b513f46dda003b3"
    )


def check_the_artefact_carries_the_config_fingerprint():
    cfg = _small()
    assert T.run(cfg).artifact.payload["config_fingerprint"] == cfg.fingerprint()


def check_metrics_cover_both_splits():
    metrics = T.run(_small()).metrics
    for split in ("train", "holdout"):
        for name in ("log_loss", "accuracy", "roc_auc", "n_rows", "positive_rate"):
            key = "{}_{}".format(split, name)
            assert key in metrics, "missing {}".format(key)


def check_holdout_is_scored_on_rows_the_fit_never_saw():
    """n_rows on the holdout has to match the split, or the pipeline is scoring the
    training set twice under two names."""
    metrics = T.run(_small()).metrics
    assert metrics["holdout_n_rows"] == 500.0, metrics["holdout_n_rows"]
    assert metrics["train_n_rows"] == 1500.0, metrics["train_n_rows"]


def check_the_shipped_baseline_runs():
    """Slow and worth it. This is the config in the README quick start, so a reader hits
    this path first."""
    result = T.run(_baseline())
    assert result.metrics["holdout_roc_auc"] > 0.5
    assert len(result.content_hash) == 64


def check_no_timestamp_leaks_into_the_artefact():
    """A clock reading would make every run produce different bytes and the content hash
    would identify nothing. Checked by looking for the current year in the serialised
    form, which is crude and catches the realistic version of the mistake."""
    import datetime

    raw = T.run(_small()).artifact.to_bytes().decode("utf-8")
    year = str(datetime.date.today().year)
    # The seed is allowed to look like a date. Nothing else is.
    without_seed = raw.replace('"seed":5', "")
    assert year not in without_seed, "something time varying reached the artefact"
