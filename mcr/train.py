"""The training pipeline. Config in, artefact out, nothing else.

Everything this function needs comes from the config and the seed derived from it. It reads
no environment variable, no clock and no file. That is what makes the artefact a function
of the config, which is the property the registry and the
gate both rest on.

There is deliberately no timestamp in the artefact. A timestamp would make every run
produce different bytes and the content hash would identify nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from . import artifact as artifact_mod
from . import data as data_mod
from . import model as model_mod
from .config import TrainConfig


@dataclass(frozen=True)
class RunResult:
    artifact: artifact_mod.Artifact
    metrics: Dict[str, float]
    model: model_mod.Model

    @property
    def content_hash(self) -> str:
        return self.artifact.content_hash()


def evaluate(m: model_mod.Model, x, y) -> Dict[str, float]:
    p = m.predict_proba(x)
    return {
        "log_loss": model_mod.log_loss(y.astype(float), p),
        "accuracy": model_mod.accuracy(y, p),
        "roc_auc": model_mod.roc_auc(y, p),
        "n_rows": float(len(y)),
        "positive_rate": float(y.mean()),
    }


def run(cfg: TrainConfig) -> RunResult:
    ds = data_mod.generate(cfg.data, cfg.seed)
    m = model_mod.fit(ds.x_train, ds.y_train, cfg.model, cfg.seed)

    metrics = {}
    for split, x, y in (
        ("train", ds.x_train, ds.y_train),
        ("holdout", ds.x_holdout, ds.y_holdout),
    ):
        for k, v in evaluate(m, x, y).items():
            metrics["{}_{}".format(split, k)] = v

    art = artifact_mod.build(cfg, m, metrics)
    return RunResult(artifact=art, metrics=metrics, model=m)
