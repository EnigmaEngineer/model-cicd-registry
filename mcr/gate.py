"""The promotion gate. Does this candidate beat what is in production.

The gate does not read the metrics either run recorded. That is the whole design and it
came out of a measurement, so it is worth stating before any of the code.

Every run here generates its own corpus from its own config, and the holdout is a slice of
that corpus. So `holdout_log_loss` in the tracking store is a number about the run's own
holdout, and two runs only share a holdout when their data config and their seed both
match. Nothing was checking that. With the model config held fixed and only the seed moved,
holdout log loss runs from 0.341545 to 0.449239 across twelve seeds. The gap this gate was
built to catch, a four hundred epoch model against a one epoch model on one corpus, is
0.192633. So the corpus alone is 55.9 percent of the signal, and AUC is far worse.
See docs/adr-0004 for the table.

A gate comparing stored metrics would therefore promote whichever model happened to draw an
easier holdout. So the gate builds one holdout, scores both models on it, and decides on
that. The stored numbers still go in the report, labelled as what each run said about its
own holdout, because the disagreement between the two is worth showing.

The second thing here is non finite metrics, and it is subtler than it looks. A comparison
against a NaN is False whichever way round it is written. `nan < good` is False and
`good < nan` is False. So a naive `candidate < incumbent` rejects a NaN candidate, which
looks correct, and rejects every candidate forever when the incumbent is the NaN. Same
silent False, and the second case is a production outage with a plausible message on it.
Both are refusals here, with different reasons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import data as data_mod
from . import model as model_mod
from .config import DataConfig, TrainConfig

# Lower is better, and it is the only metric of the three this project computes that can
# separate two models here. Measured over ten model configs on one corpus: log loss spans
# 0.239573, accuracy spans 1.4e-03 and AUC spans 2.06e-04. AUC reads the ranking only and
# the direction of the weight vector settles in the first few epochs, so a gate reading it
# passes a model that trained for one epoch. docs/adr-0004 carries the table.
METRIC = "log_loss"

# Only `bootstrap_interval` reads this, and nothing on the decision path calls that. It is
# kept so the interval the gate does use has a second implementation to be graded against.
RESAMPLES = 2000

# Two sided, on the mean of the paired difference. The gate promotes when the whole
# interval sits below zero and rejects otherwise, so this is the level at which a
# difference is allowed to move production.
CONFIDENCE = 0.95

PROMOTE = "promote"
REJECT = "reject"
REFUSE = "refuse"


class GateError(RuntimeError):
    """Raised when the gate cannot be handed what it needs to decide anything."""


@dataclass(frozen=True)
class HoldoutSpec:
    """The rows every candidate is judged on.

    Deliberately its own object rather than "whatever the candidate's config says". If the
    holdout came from the candidate, a candidate could choose its own exam.
    """

    data: DataConfig
    seed: int

    def fingerprint(self) -> str:
        """Identifies the rows, not the spec.

        Hashing the spec fields would be cheaper and would not survive a change to the
        generator, which would silently move every row while the fingerprint held still.
        This hashes the rows themselves.
        """
        import hashlib

        ds = data_mod.generate(self.data, self.seed)
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(ds.x_holdout, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(ds.y_holdout, dtype=np.int64).tobytes())
        return h.hexdigest()[:12]

    def rows(self) -> Tuple[np.ndarray, np.ndarray]:
        ds = data_mod.generate(self.data, self.seed)
        return ds.x_holdout, ds.y_holdout


def spec_from_config(cfg: TrainConfig) -> HoldoutSpec:
    return HoldoutSpec(data=cfg.data, seed=cfg.seed)


def model_from_artifact(payload: Dict[str, Any]) -> model_mod.Model:
    """Rebuild the scorer from artefact bytes.

    The artefact holds the fitted parameters, so the gate does not retrain anything to
    score a model. It also means the thing the gate scores is the thing whose hash the
    registry holds, which retraining could not promise on its own.
    """
    try:
        body = payload["model"]
        return model_mod.Model(
            weights=np.asarray(body["weights"], dtype=np.float64),
            bias=float(body["bias"]),
            mean=np.asarray(body["mean"], dtype=np.float64),
            scale=np.asarray(body["scale"], dtype=np.float64),
            epochs_run=int(body["epochs_run"]),
            final_loss=float(body["final_loss"]),
        )
    except (KeyError, TypeError) as exc:
        raise GateError("artefact has no model this gate can score: {}".format(exc))


@dataclass(frozen=True)
class Scored:
    """One model's results on the gate's holdout.

    `row_losses` is kept because the comparison is paired. Two models judged on the same
    rows can be compared row by row, which is a far tighter question than comparing two
    averages, and the interval below needs the rows.
    """

    name: str
    artifact_hash: str
    row_losses: np.ndarray
    log_loss: float
    roc_auc: float
    reported: Dict[str, float] = field(default_factory=dict)

    def finite(self) -> bool:
        # `log_loss` is the mean of `row_losses` so the first term is implied by the last,
        # and it is here because it is the value the refusal message prints. An accuracy
        # term was here too and came out, because it can only be NaN when the probabilities
        # already are, so it was a guard that could not fire on its own.
        return bool(
            np.isfinite(self.log_loss)
            and np.isfinite(self.roc_auc)
            and np.isfinite(self.row_losses).all()
        )

    def metric(self) -> float:
        return self.log_loss


def _row_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    # The same clip model.log_loss uses. It is here rather than imported because that
    # function returns the mean and the paired comparison needs the terms.
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    yf = y.astype(np.float64)
    return -(yf * np.log(p) + (1.0 - yf) * np.log(1.0 - p))


def score(
    name: str,
    artifact_hash: str,
    payload: Dict[str, Any],
    spec: HoldoutSpec,
    reported: Optional[Dict[str, float]] = None,
) -> Scored:
    x, y = spec.rows()
    m = model_from_artifact(payload)
    p = m.predict_proba(x)
    losses = _row_losses(y, p)

    # roc_auc raises when a class is absent, which cannot happen on a holdout this config
    # module allows, and a gate that dies on a degenerate holdout is worse than one that
    # reports a non finite number and refuses on it.
    try:
        auc = model_mod.roc_auc(y, p)
    except ValueError:
        auc = float("nan")

    return Scored(
        name=name,
        artifact_hash=artifact_hash,
        row_losses=losses,
        log_loss=float(np.mean(losses)),
        roc_auc=float(auc),
        reported=dict(reported or {}),
    )


def paired_interval(diff: np.ndarray, confidence: float = CONFIDENCE) -> Tuple[float, float]:
    """Interval on the mean of a paired difference. Closed form, no resampling.

    This started out as a percentile bootstrap and the mutation pass is why it is not one
    any more. A bootstrap needs a seed and a resample count, and neither is a number
    anybody chose. On a candidate sitting near the gate's resolution, four hundred epochs
    against forty, the verdict came back `worse` on some seeds and `not_separated` on
    others across thirty two of them. A gate whose answer depends on a default argument is
    not a gate, and pinning the seed would only have hidden that behind a constant.

    The mean of five thousand paired differences is normal enough for this, and the two
    agree on every case the probe carries, so the resampler was buying nothing and
    charging an arbitrary constant for it. `bootstrap_interval` is kept below and a check
    grades the two against each other, because one statistic computed two ways is worth
    having only when something asserts they say the same thing.
    """
    n = len(diff)
    if n < 2:
        raise GateError("cannot build an interval on {} rows".format(n))
    z = _z_for(confidence)
    mean = float(np.mean(diff))
    se = float(np.std(diff, ddof=1)) / math.sqrt(n)
    return mean - z * se, mean + z * se


def _z_for(confidence: float) -> float:
    """Two sided normal quantile, by bisection on the error function.

    `math.erf` is in the standard library and its inverse is not, and scipy is not a
    dependency of the training path. Bisection over thirty two rounds of a monotone
    function on a bracket of zero to ten lands well inside a float.
    """
    if not 0.0 < confidence < 1.0:
        raise GateError("confidence must be strictly between 0 and 1")
    target = confidence
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erf(mid / math.sqrt(2.0)) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    return 0.5 * (lo + hi)


def bootstrap_interval(
    diff: np.ndarray, resamples: int = RESAMPLES, confidence: float = CONFIDENCE, seed: int = 0
) -> Tuple[float, float]:
    """Percentile bootstrap on the same quantity. Not what the gate decides on.

    Here so the closed form above has something to be graded against, and so the
    instability that took it off the decision path can be shown rather than asserted.
    """
    if len(diff) == 0:
        raise GateError("cannot build an interval on zero rows")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    tail = 0.5 * (1.0 - confidence) * 100.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def rebuild(cfg: TrainConfig, expected_hash: str, label: str) -> Dict[str, Any]:
    """Retrain from a recovered config and refuse if the bytes moved.

    The registry stores an artefact hash and a run id. It does not store the artefact,
    because `source` on a version points at a path in the tracking store that nothing in
    this project writes. So the way to a scorable model is to rebuild it from the config
    the run recorded, which the tracking layer already proves round trips.

    The check is what makes that sound. If the rebuilt bytes hash to something other than
    what the version is tagged with, then either the pipeline stopped being a function of
    the config or the tag is describing a different model, and a gate that scored the
    rebuild anyway would be judging a model nobody registered.
    """
    from . import train as train_mod

    result = train_mod.run(cfg)
    if result.content_hash != expected_hash:
        raise GateError(
            "{} is registered as {} and its config rebuilds to {}, so the gate would be "
            "scoring a model the registry does not point at".format(
                label, expected_hash[:12], result.content_hash[:12]
            )
        )
    return result.artifact.payload


@dataclass(frozen=True)
class Decision:
    verdict: str
    reason: str
    detail: str
    holdout: str
    candidate: Optional[Scored] = None
    incumbent: Optional[Scored] = None
    mean_diff: Optional[float] = None
    interval: Optional[Tuple[float, float]] = None

    @property
    def promoted(self) -> bool:
        return self.verdict == PROMOTE


def decide(
    candidate: Scored,
    incumbent: Optional[Scored],
    holdout: str,
) -> Decision:
    """Promote, reject, or refuse to answer.

    Three verdicts rather than two. Refuse is not a stricter reject. A reject says the
    gate compared two models and the candidate lost. A refuse says the gate could not
    perform the comparison, and the two need different handling by whatever called it.
    """
    if not candidate.finite():
        return Decision(
            verdict=REFUSE,
            reason="candidate_not_finite",
            detail=(
                "candidate {} scores {} on {} and a comparison against a non finite value "
                "is False in both directions".format(
                    candidate.name, candidate.metric(), METRIC
                )
            ),
            holdout=holdout,
            candidate=candidate,
            incumbent=incumbent,
        )

    if incumbent is None:
        return Decision(
            verdict=PROMOTE,
            reason="no_incumbent",
            detail="nothing holds the stage, so there is nothing to beat",
            holdout=holdout,
            candidate=candidate,
        )

    if not incumbent.finite():
        # The dangerous one. Left to a naive comparison this rejects every candidate
        # forever, with a message saying the candidate failed to improve on a model that
        # is broken. The gate names the incumbent instead, which is the thing to fix.
        return Decision(
            verdict=REFUSE,
            reason="incumbent_not_finite",
            detail=(
                "the incumbent {} scores {} on {}, so nothing can be shown to beat it. "
                "Fix or unpoint the incumbent.".format(
                    incumbent.name, incumbent.metric(), METRIC
                )
            ),
            holdout=holdout,
            candidate=candidate,
            incumbent=incumbent,
        )

    if candidate.artifact_hash == incumbent.artifact_hash:
        return Decision(
            verdict=REFUSE,
            reason="same_artifact",
            detail=(
                "candidate and incumbent are the same bytes, {}".format(
                    candidate.artifact_hash[:12]
                )
            ),
            holdout=holdout,
            candidate=candidate,
            incumbent=incumbent,
        )

    diff = candidate.row_losses - incumbent.row_losses
    mean_diff = float(np.mean(diff))
    lo, hi = paired_interval(diff)

    if hi < 0.0:
        verdict, reason = PROMOTE, "better"
        detail = "candidate is better by {:.6f} on {}, interval [{:.6f}, {:.6f}]".format(
            -mean_diff, METRIC, -hi, -lo
        )
    elif lo > 0.0:
        verdict, reason = REJECT, "worse"
        detail = "candidate is worse by {:.6f} on {}, interval [{:.6f}, {:.6f}]".format(
            mean_diff, METRIC, lo, hi
        )
    else:
        # The interval covers zero. Not a tie and not a loss. The rows do not carry
        # enough to say which model is better, and holding the incumbent is the answer
        # that changes nothing when nothing is known.
        verdict, reason = REJECT, "not_separated"
        detail = (
            "difference {:+.6e} on {} with interval [{:+.6e}, {:+.6e}], which covers "
            "zero".format(mean_diff, METRIC, lo, hi)
        )

    return Decision(
        verdict=verdict,
        reason=reason,
        detail=detail,
        holdout=holdout,
        candidate=candidate,
        incumbent=incumbent,
        mean_diff=mean_diff,
        interval=(lo, hi),
    )


def _num(value: Optional[float]) -> str:
    """One cell of the report.

    A number the run never recorded and a number it recorded as NaN are different facts
    and they get different cells. The first draft printed a dash for both, which turned
    the most important value in the table into an absence.
    """
    if value is None:
        return "-"
    if not np.isfinite(value):
        return "nan"
    return "{:.6f}".format(value)


def report_lines(decision: Decision) -> List[str]:
    """The comparison report, as text.

    It carries what each run said about its own holdout beside what the gate measured on
    the shared one. Those two disagree whenever the runs did not share a corpus, and the
    disagreement is the reason this gate scores the models itself.
    """
    out = ["holdout      {}".format(decision.holdout)]
    out.append("metric       {}, lower is better".format(METRIC))
    out.append("")

    out.append("{:<32} {:>16} {:>10} {:>16}".format(
        "", "on this holdout", "roc auc", "as the run ran"))
    for role, s in (("incumbent", decision.incumbent), ("candidate", decision.candidate)):
        if s is None:
            out.append("{:<32} {:>16}".format(role, "none"))
            continue
        out.append(
            "{:<32} {:>16} {:>10} {:>16}".format(
                "{} {}".format(role, s.name)[:32],
                _num(s.metric()),
                _num(s.roc_auc),
                _num(s.reported.get("holdout_{}".format(METRIC))),
            )
        )

    out.append("")
    if decision.interval is not None:
        lo, hi = decision.interval
        out.append(
            "paired diff  {:+.6e}  interval [{:+.6e}, {:+.6e}]".format(
                decision.mean_diff, lo, hi
            )
        )
    out.append("verdict      {}  ({})".format(decision.verdict, decision.reason))
    out.append("             {}".format(decision.detail))
    return out


# The tag namespace the gate writes onto a candidate's run, and the one thing that reads it
# back. Both halves live here because they are one vocabulary. They were split across two
# scripts until 2026-09-17, which is how the reader came to check a different set of keys
# from the ones the writer produced.
TAG_PREFIX = "gate."
TAG_VERDICT = TAG_PREFIX + "verdict"
TAG_REASON = TAG_PREFIX + "reason"
TAG_STAGE = TAG_PREFIX + "stage"
TAG_INCUMBENT = TAG_PREFIX + "incumbent"

NO_INCUMBENT = "none"


def refusal_for(tags, stage: str, incumbent: Optional[int]) -> Optional[str]:
    """Why a version must not be promoted to `stage`, or None if the gate cleared it.

    Pure. It takes the run's tags, the stage being moved, and the version that holds that
    stage right now. Every caller does its own IO and this decides. Written that way after
    the first version needed a fake MLflow client to test at all, which meant the checks
    graded a stand in rather than this.

    Three refusals and the third is the one that matters.

    No verdict at all. A version nothing has gated is the normal state of a freshly
    registered model, and promoting it is the thing this exists to stop.

    A verdict that is not a promotion. Refuse and reject both land here.

    **A verdict earned against a different incumbent.** A comparison is against something.
    Clearing version 2 while a weak model held production says nothing about whether it
    beats the strong model that holds it now, and the tag is overwritten on the next gate
    run so a stale verdict looks exactly like a fresh one. This is what "passed the latest
    gate" has to mean, and the first version of this function missed it.
    """
    verdict = tags.get(TAG_VERDICT)
    if verdict is None:
        return "it has never been gated, so there is no verdict to honour"
    if verdict != PROMOTE:
        return "the gate's last verdict was {} ({})".format(
            verdict, tags.get(TAG_REASON, "no reason recorded")
        )

    gated_stage = tags.get(TAG_STAGE)
    if gated_stage != stage:
        return "it was gated against {} and this moves {}".format(
            gated_stage or "an unrecorded stage", stage
        )

    gated_against = tags.get(TAG_INCUMBENT)
    if gated_against is None:
        return (
            "the verdict predates this check and does not record which incumbent it beat, "
            "so it cannot be shown to be current"
        )
    now = NO_INCUMBENT if incumbent is None else str(incumbent)
    if gated_against != now:
        return "it was gated when {} held {} and {} holds it now".format(
            _held(gated_against), stage, _held(now)
        )
    return None


def _held(value: str) -> str:
    """`none` is a sentinel and reads as a version number if it is formatted like one.

    The first version printed "gated against version none", which a reader parses as a
    version literally named none before they parse it as nothing.
    """
    return "nothing" if value == NO_INCUMBENT else "version " + value


def refusal_for_version(cli, registry, model: str, ref: str, stage: str) -> Optional[str]:
    """`refusal_for` with the IO around it. Returns a reason or None.

    `registry` is passed in rather than imported, so this module still knows nothing about
    the registry and cannot be the reason a promote starts depending on a gate. That is
    the whole point of the split. The library can answer "did the gate clear this", and it
    is the caller that decides whether to ask. `mcr.registry.promote` never asks.
    """
    version = registry.resolve(cli, model, ref)
    run_id = cli.get_model_version(model, str(version)).run_id
    if run_id is None:
        return "version {} has no run behind it, so no verdict can be read".format(version)
    reason = refusal_for(
        cli.get_run(run_id).data.tags, stage, registry.current(cli, model, stage)
    )
    return None if reason is None else "version {}: {}".format(version, reason)
