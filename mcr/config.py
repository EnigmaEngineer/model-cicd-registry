"""Training configuration.

Configs are YAML on disk and a frozen dataclass in memory. Every field a run depends on
lives here, because the promotion gate later has to answer "was this candidate trained the
same way as the incumbent" and that question is only answerable if the answer is one object.

The fingerprint is the point of this module. Two runs with the same fingerprint should
produce the same model, and a run whose fingerprint nobody recorded cannot be compared to
anything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(ValueError):
    """Raised on a config that cannot be trusted to describe a run."""


@dataclass(frozen=True)
class DataConfig:
    n_rows: int
    n_features: int
    positive_rate: float
    noise: float
    holdout_frac: float


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float
    epochs: int
    l2: float
    init: str


@dataclass(frozen=True)
class TrainConfig:
    name: str
    seed: int
    data: DataConfig
    model: ModelConfig

    def fingerprint(self) -> str:
        """Content hash over every field, stable across processes.

        sort_keys matters more than it looks. Without it the hash follows dict insertion
        order, so the same config loaded from two YAML files whose keys are in a different
        order would fingerprint differently and the gate would call them different runs.
        """
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


_INITS = ("zeros", "normal")


def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError("{} is missing '{}'".format(where, key))
    return mapping[key]


def _section(raw: Dict[str, Any], key: str, cls) -> Any:
    """Build a section and refuse anything the dataclass does not declare.

    Refusing unknown keys is deliberate. A typo like `epoch: 40` would otherwise be
    silently dropped and the run would use the default epochs, which is the kind of thing
    that makes two "identical" runs disagree for a reason nobody can find.
    """
    body = _require(raw, key, "config")
    if not isinstance(body, dict):
        raise ConfigError("config section '{}' must be a mapping".format(key))
    declared = {f.name for f in fields(cls)}
    unknown = sorted(set(body) - declared)
    if unknown:
        raise ConfigError("unknown keys in '{}': {}".format(key, ", ".join(unknown)))
    missing = sorted(declared - set(body))
    if missing:
        raise ConfigError("'{}' is missing: {}".format(key, ", ".join(missing)))
    return cls(**body)


def validate(cfg: TrainConfig) -> None:
    d, m = cfg.data, cfg.model

    if d.n_rows < 10:
        raise ConfigError("n_rows must be at least 10, got {}".format(d.n_rows))
    if d.n_features < 1:
        raise ConfigError("n_features must be positive, got {}".format(d.n_features))
    if not 0.0 < d.positive_rate < 1.0:
        raise ConfigError("positive_rate must be strictly between 0 and 1")
    if d.noise < 0.0:
        raise ConfigError("noise cannot be negative")
    if not 0.0 < d.holdout_frac < 1.0:
        raise ConfigError("holdout_frac must be strictly between 0 and 1")

    # A holdout that rounds to zero rows makes every gate downstream score an empty set,
    # and an empty set scores perfectly. Refuse it here rather than at the gate.
    if int(round(d.n_rows * d.holdout_frac)) < 5:
        raise ConfigError(
            "holdout_frac {} over {} rows leaves fewer than 5 holdout rows".format(
                d.holdout_frac, d.n_rows
            )
        )

    if m.learning_rate <= 0.0:
        raise ConfigError("learning_rate must be positive")
    if m.epochs < 1:
        raise ConfigError("epochs must be at least 1")
    if m.l2 < 0.0:
        raise ConfigError("l2 cannot be negative")
    if m.init not in _INITS:
        raise ConfigError("init must be one of {}, got '{}'".format(_INITS, m.init))

    if cfg.seed < 0:
        raise ConfigError("seed must be non negative, got {}".format(cfg.seed))
    if not cfg.name:
        raise ConfigError("name cannot be empty")


def from_dict(raw: Dict[str, Any]) -> TrainConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping at the top level")

    declared = {"name", "seed", "data", "model"}
    unknown = sorted(set(raw) - declared)
    if unknown:
        raise ConfigError("unknown top level keys: {}".format(", ".join(unknown)))

    cfg = TrainConfig(
        name=_require(raw, "name", "config"),
        seed=_require(raw, "seed", "config"),
        data=_section(raw, "data", DataConfig),
        model=_section(raw, "model", ModelConfig),
    )
    validate(cfg)
    return cfg


def load(path: str | Path) -> TrainConfig:
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if raw is None:
        raise ConfigError("config file {} is empty".format(path))
    return from_dict(raw)
