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
from typing import Any, Dict, get_type_hints

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


def declared_types(cls) -> Dict[str, type]:
    """Field name to real type for one config section.

    `dataclasses.fields(cls)[i].type` is the *string* "int" in this module, because of the
    `from __future__ import annotations` at the top. Calling it would call a string. So the
    types come from `get_type_hints`, which resolves them against the module namespace.
    """
    hints = get_type_hints(cls)
    return {f.name: hints[f.name] for f in fields(cls)}


def coerce(value: Any, want: type, where: str) -> Any:
    """Convert to the declared type, or refuse. Never convert with loss.

    This exists because the fingerprint was a function of how the YAML was typed rather
    than of the run. `noise: 1` and `noise: 1.0` describe the same training run and
    fingerprinted differently, because the dataclass stored whatever `yaml.safe_load`
    handed it and `json.dumps` then wrote `1` against `1.0`. Two spellings of one recipe
    got two registry keys. See docs/adr-0002.

    It also has to accept strings, because that is what the tracking store gives back. A
    param goes into MLflow as a float and comes out as "0.5", so the recovery path needs
    the same rule the loader uses. One function rather than two that drift apart.
    """
    if want is bool:
        raise ConfigError("{}: bool is not a config type".format(where))

    if want is str:
        # Deliberately not str(value). `name: 2026` is more likely a mistake than an
        # intent, and a silent stringify is how a typo survives to the registry.
        if not isinstance(value, str):
            raise ConfigError(
                "{} must be a string, got {}".format(where, type(value).__name__)
            )
        return value

    # bool before int, because bool is a subclass of int and `epochs: true` would pass.
    if isinstance(value, bool):
        raise ConfigError("{}: bool is not a number".format(where))

    if want is int:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value != int(value):
                raise ConfigError("{} must be a whole number, got {}".format(where, value))
            return int(value)
        if isinstance(value, str):
            try:
                return coerce(float(value), int, where)
            except ValueError:
                raise ConfigError("{} is not a number: {!r}".format(where, value))
        raise ConfigError("{} must be a number, got {}".format(where, type(value).__name__))

    if want is float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise ConfigError("{} is not a number: {!r}".format(where, value))
        raise ConfigError("{} must be a number, got {}".format(where, type(value).__name__))

    raise ConfigError("{}: no rule for declared type {}".format(where, want))


def _section(raw: Dict[str, Any], key: str, cls) -> Any:
    """Build a section and refuse anything the dataclass does not declare.

    Refusing unknown keys is deliberate. A typo like `epoch: 40` would otherwise be
    silently dropped and the run would use the default epochs, which is the kind of thing
    that makes two "identical" runs disagree for a reason nobody can find.
    """
    body = _require(raw, key, "config")
    if not isinstance(body, dict):
        raise ConfigError("config section '{}' must be a mapping".format(key))
    declared = declared_types(cls)
    unknown = sorted(set(body) - set(declared))
    if unknown:
        raise ConfigError("unknown keys in '{}': {}".format(key, ", ".join(unknown)))
    missing = sorted(set(declared) - set(body))
    if missing:
        raise ConfigError("'{}' is missing: {}".format(key, ", ".join(missing)))
    typed = {
        name: coerce(body[name], want, "{}.{}".format(key, name))
        for name, want in declared.items()
    }
    return cls(**typed)


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
        name=coerce(_require(raw, "name", "config"), str, "name"),
        seed=coerce(_require(raw, "seed", "config"), int, "seed"),
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
