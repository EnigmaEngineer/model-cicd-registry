"""Artefact writing, and the reason the bytes are what they are.

An artefact has to be byte reproducible, because the registry keys on its content
hash, and the promotion gate compares a candidate against an incumbent that was written
on some earlier run. Two runs of the same config must land on the same bytes or
none of that works.

Floats are the whole problem. `json.dumps` on a numpy float64 writes Python's repr, which
is the shortest string that round trips, so it is already exact. What is not safe is
letting numpy types through unconverted, or letting dict order follow insertion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import TrainConfig
from .model import Model

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class Artifact:
    payload: Dict[str, Any]

    def to_bytes(self) -> bytes:
        # sort_keys and a fixed separator. Without both, the bytes follow whatever order
        # the dict happened to be built in and the hash stops meaning "same model".
        text = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        return (text + "\n").encode("utf-8")

    def content_hash(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def write(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.to_bytes())
        return self.content_hash()


def _floats(a: np.ndarray) -> list:
    """numpy scalars are not JSON serialisable and json.dumps says so at write time.

    tolist() converts the whole array to Python floats in one pass. Doing it elementwise
    with float() would be the same numbers and a slower loop.
    """
    return np.asarray(a, dtype=np.float64).tolist()


def build(cfg: TrainConfig, model: Model, metrics: Dict[str, float]) -> Artifact:
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "config_fingerprint": cfg.fingerprint(),
        "name": cfg.name,
        "seed": cfg.seed,
        "model": {
            "weights": _floats(model.weights),
            "bias": float(model.bias),
            "mean": _floats(model.mean),
            "scale": _floats(model.scale),
            "epochs_run": int(model.epochs_run),
            "final_loss": float(model.final_loss),
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    return Artifact(payload=payload)


def read(path: str | Path) -> Artifact:
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    art = Artifact(payload=payload)
    # A file that does not round trip means somebody hand edited it or wrote it with a
    # different serialiser. Either way its hash no longer identifies its contents, and the
    # registry that keys on that hash would be pointing at something else.
    if art.to_bytes() != raw:
        raise ValueError(
            "{} does not round trip, its bytes were not written by this module".format(path)
        )
    return art
