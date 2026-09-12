"""The MLflow layer. Write a run, read it back, rebuild the config from what came back.

The reason this module exists is the last clause. A tracking store that records a number on
a dashboard is worth very little. One that holds enough to reconstruct the run is worth the
whole project, because the promotion gate has to compare a candidate against an incumbent
that some earlier process trained, and the only thing it will have is whatever went into
the store.

So `params_for` and `config_from_params` are inverses and there is a probe that says so.

Two measured facts about MLflow shape everything below.

A param refuses to be overwritten. `log_param` on a key that already holds a different
value raises. A tag overwrites silently. That is not a nuisance. It hands over the right
split for free. The recipe goes in params where nothing can rewrite it after the fact. The
mutable stage goes in tags, which is what the registry needs.

A long value is truncated rather than refused. A param over 6,000 characters and a tag over
8,000 come back shortened with a warning on stderr and no exception. Nothing here is close
to it. The longest value is a 64 character hash. But a silently shortened content hash is a
corrupt registry key, so `_short_enough` refuses rather than trusting the headroom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import mlflow

from .config import DataConfig, ModelConfig, TrainConfig, coerce, declared_types

EXPERIMENT = "model-cicd-registry"

# Measured on mlflow 3.16.0. Both are the point at which the value is truncated and the
# call still succeeds, so they are the interesting numbers rather than the documented ones.
PARAM_VALUE_LIMIT = 6000
TAG_VALUE_LIMIT = 8000

# Sections in the flat param namespace. Dotted keys rather than bare field names, because
# two sections are free to declare the same field name and a flat namespace would silently
# drop one of them. Derived from the dataclasses below rather than written out here.
_SECTIONS = {"data": DataConfig, "model": ModelConfig}

# Tag keys this module owns. The fingerprint is what "have I trained this recipe" keys on.
TAG_CONFIG_FINGERPRINT = "config_fingerprint"
TAG_ARTIFACT_HASH = "artifact_hash"


class TrackingError(RuntimeError):
    """Raised when the store cannot be trusted to describe the run it holds."""


def param_keys() -> List[str]:
    """Every param key a run logs, derived from the config dataclasses.

    Derived rather than listed. A field added to `DataConfig` with no matching entry here
    would otherwise be silently absent from every run in the store, and the recovery path
    would notice only by failing to rebuild.

    The dot is what makes this safe and it is the reason there is no collision guard below.
    Section names come out of a dict so they are unique, and a field name cannot contain a
    dot because it is a Python identifier, so two dotted keys can only match if one section
    appeared twice. The first draft had a guard for it. It could not fire, and a guard that
    cannot fire is a claim that something is being checked.
    """
    keys = ["name", "seed"]
    for section, cls in sorted(_SECTIONS.items()):
        keys.extend("{}.{}".format(section, f) for f in sorted(declared_types(cls)))
    return keys


def _short_enough(key: str, value: str, limit: int) -> str:
    if len(value) > limit:
        raise TrackingError(
            "{} is {} characters and MLflow truncates above {}".format(
                key, len(value), limit
            )
        )
    return value


def params_for(cfg: TrainConfig) -> Dict[str, str]:
    """The config as MLflow params. Strings, because that is what the store holds.

    Everything goes in as `str(value)`, which for a float is `repr` and therefore the
    shortest string that reads back to the same double. So the value survives and the type
    does not, and `config_from_params` puts the type back from the dataclass.
    """
    out = {"name": cfg.name, "seed": str(cfg.seed)}
    for section, cls in sorted(_SECTIONS.items()):
        body = getattr(cfg, section)
        for field in sorted(declared_types(cls)):
            out["{}.{}".format(section, field)] = str(getattr(body, field))

    # There was a guard here comparing these keys against param_keys(). It was decoration.
    # Both sides derive from _SECTIONS by the same walk, so nothing can make them disagree,
    # and a guard whose two sides share one source of truth cannot fail. A mutant broke the
    # expression inside its own error message and the suite never noticed, which is how it
    # was found. tests/test_tracking.py asserts the agreement from the outside instead.
    for k, v in out.items():
        _short_enough(k, v, PARAM_VALUE_LIMIT)
    return out


def config_from_params(params: Dict[str, str]) -> TrainConfig:
    """Rebuild the config from stored params, or refuse.

    The refusal matters more than the rebuild. A recovery path that filled in a default for
    a param the store never held would hand the gate a config that looks complete and
    describes a run nobody performed, and the artefact hash would then disagree with the
    store for a reason nothing reports.
    """
    missing = sorted(set(param_keys()) - set(params))
    if missing:
        raise TrackingError("run is missing params: {}".format(", ".join(missing)))

    sections = {}
    for section, cls in sorted(_SECTIONS.items()):
        body = {}
        for field, want in sorted(declared_types(cls).items()):
            key = "{}.{}".format(section, field)
            body[field] = coerce(params[key], want, key)
        sections[section] = cls(**body)

    return TrainConfig(
        name=coerce(params["name"], str, "name"),
        seed=coerce(params["seed"], int, "seed"),
        data=sections["data"],
        model=sections["model"],
    )


@dataclass(frozen=True)
class Recovered:
    run_id: str
    config: TrainConfig
    artifact_hash: str
    config_fingerprint: str
    metrics: Dict[str, float]


def client(tracking_uri: str) -> mlflow.MlflowClient:
    return mlflow.MlflowClient(tracking_uri=tracking_uri)


def experiment_id(cli: mlflow.MlflowClient, name: str = EXPERIMENT) -> str:
    found = cli.get_experiment_by_name(name)
    if found is not None:
        return found.experiment_id
    return cli.create_experiment(name)


def log_training_run(
    cli: mlflow.MlflowClient,
    cfg: TrainConfig,
    artifact_hash: str,
    metrics: Dict[str, float],
    experiment: str = EXPERIMENT,
    run_name: Optional[str] = None,
) -> str:
    """One run, one config, one artefact hash. Returns the run id.

    The run name carries the config name and the first eight of the hash. It is for a human
    reading the UI and nothing keys on it, which is why the hash also goes in as a tag.
    """
    fingerprint = cfg.fingerprint()
    name = run_name or "{}-{}".format(cfg.name, artifact_hash[:8])

    run = cli.create_run(
        experiment_id=experiment_id(cli, experiment),
        run_name=name,
        tags={
            TAG_CONFIG_FINGERPRINT: _short_enough(
                TAG_CONFIG_FINGERPRINT, fingerprint, TAG_VALUE_LIMIT
            ),
            TAG_ARTIFACT_HASH: _short_enough(
                TAG_ARTIFACT_HASH, artifact_hash, TAG_VALUE_LIMIT
            ),
        },
    )
    rid = run.info.run_id

    for key, value in sorted(params_for(cfg).items()):
        cli.log_param(rid, key, value)
    for key, value in sorted(metrics.items()):
        cli.log_metric(rid, key, float(value))

    cli.set_terminated(rid, status="FINISHED")
    return rid


def recover(cli: mlflow.MlflowClient, run_id: str) -> Recovered:
    run = cli.get_run(run_id)

    for tag in (TAG_CONFIG_FINGERPRINT, TAG_ARTIFACT_HASH):
        if tag not in run.data.tags:
            raise TrackingError("run {} has no {} tag".format(run_id, tag))

    cfg = config_from_params(run.data.params)
    stored = run.data.tags[TAG_CONFIG_FINGERPRINT]

    # The rebuilt config has to fingerprint to the value the run was tagged with. If it
    # does not, either the params do not describe the run or the fingerprint is not a
    # function of the config, and both of those make every comparison downstream wrong.
    if cfg.fingerprint() != stored:
        raise TrackingError(
            "run {} is tagged {} and its params rebuild to {}".format(
                run_id, stored, cfg.fingerprint()
            )
        )

    return Recovered(
        run_id=run_id,
        config=cfg,
        artifact_hash=run.data.tags[TAG_ARTIFACT_HASH],
        config_fingerprint=stored,
        metrics=dict(run.data.metrics),
    )


def runs_for_fingerprint(
    cli: mlflow.MlflowClient, fingerprint: str, experiment: str = EXPERIMENT
) -> List[str]:
    """Every run of one recipe, newest first.

    Two runs of one config get two run ids, which is why nothing here treats a run id as
    the identity of a model. The recipe is the fingerprint and the model is the artefact
    hash, and a run is just the occasion on which they met.
    """
    found = cli.search_runs(
        experiment_ids=[experiment_id(cli, experiment)],
        filter_string="tags.{} = '{}'".format(TAG_CONFIG_FINGERPRINT, fingerprint),
        order_by=["attributes.start_time DESC"],
    )
    return [r.info.run_id for r in found]
