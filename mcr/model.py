"""Logistic regression by full batch gradient descent, numpy only.

The model is deliberately small and hand written. This project is about the promotion
machinery rather than about modelling, and a model I implement myself has no hidden entropy
source and no library version to pin. When the gate has to answer whether a candidate beats an
incumbent, I want every difference between the two to come from the config.

Full batch rather than stochastic for the same reason. Minibatch order is one more thing
that has to be seeded and one more way two runs diverge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .seed import generator


@dataclass(frozen=True)
class Model:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    epochs_run: int
    final_loss: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self._standardise(x) @ self.weights + self.bias
        return _sigmoid(z)

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != self.weights.shape[0]:
            raise ValueError(
                "model expects {} features, got {}".format(
                    self.weights.shape[0], x.shape[1]
                )
            )
        return (np.log(x) - self.mean) / self.scale


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Branch on the sign. exp() of a large positive overflows and the naive form then
    # returns nan where it should return 0, which shows up as a loss of nan halfway
    # through training rather than as an error anybody can trace.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _initial_weights(cfg: ModelConfig, n_features: int, seed: int) -> np.ndarray:
    if cfg.init == "zeros":
        return np.zeros(n_features, dtype=np.float64)
    if cfg.init == "normal":
        rng = generator(seed, "model.init")
        return rng.normal(0.0, 0.01, size=n_features)
    raise ValueError("unknown init '{}'".format(cfg.init))


def fit(x: np.ndarray, y: np.ndarray, cfg: ModelConfig, seed: int) -> Model:
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            "x has {} rows and y has {}".format(x.shape[0], y.shape[0])
        )
    if x.shape[0] == 0:
        raise ValueError("cannot fit on zero rows")

    logx = np.log(x)
    mean = logx.mean(axis=0)
    scale = logx.std(axis=0)
    # A constant feature has zero spread and dividing by it gives inf. Leave it alone
    # instead. Its gradient is then zero and the weight stays where init put it.
    scale = np.where(scale < 1e-12, 1.0, scale)
    xs = (logx - mean) / scale

    w = _initial_weights(cfg, x.shape[1], seed)
    b = 0.0
    n = float(x.shape[0])
    yf = y.astype(np.float64)

    loss = float("nan")
    for _ in range(cfg.epochs):
        p = _sigmoid(xs @ w + b)
        err = p - yf
        grad_w = xs.T @ err / n + cfg.l2 * w
        grad_b = float(err.sum() / n)
        w = w - cfg.learning_rate * grad_w
        b = b - cfg.learning_rate * grad_b
        loss = log_loss(yf, _sigmoid(xs @ w + b)) + 0.5 * cfg.l2 * float(w @ w)

    return Model(
        weights=w,
        bias=b,
        mean=mean,
        scale=scale,
        epochs_run=cfg.epochs,
        final_loss=loss,
    )


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((p >= threshold).astype(np.int64) == y))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank based AUC with explicit tie handling.

    Written out rather than imported because scipy is not a dependency here. Ties get the
    average rank, which is what makes a model predicting one constant score 0.5 instead of
    1.0. A model that scores everything identically is a real candidate and the gate
    has to see it for what it is.
    """
    y = np.asarray(y)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("AUC needs both classes present, got {} positive".format(pos))

    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = np.asarray(p)[order]

    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
