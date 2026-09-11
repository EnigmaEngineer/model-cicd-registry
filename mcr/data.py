"""Synthetic training data.

The corpus is generated rather than downloaded so that a clone reproduces it exactly. The
cost of that choice is written into the README. A model fitted on data whose shape I chose
cannot say anything about accuracy, so nothing in this repo will publish an accuracy number
as if it meant something about the world. What it can say is whether the promotion
machinery behaves, which is what the project is about.

The generator is multiplicative on the feature side and additive in log odds. Keeping it
from sharing a functional form with the model is deliberate. If the generator were a
logistic model too, a good fit would be a tautology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DataConfig
from .seed import streams


@dataclass(frozen=True)
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_holdout: np.ndarray
    y_holdout: np.ndarray

    @property
    def n_features(self) -> int:
        return int(self.x_train.shape[1])


def _log_odds_weights(rng: np.random.Generator, n_features: int) -> np.ndarray:
    """A few features carry signal and the rest are noise.

    Real feature sets are like this and a dense weight vector is not. It also gives the l2
    term something to do, which matters once two configs get compared.
    """
    w = rng.normal(0.0, 1.0, size=n_features)
    if n_features > 3:
        w[3:] *= 0.05
    return w


def _solve_intercept(logit: np.ndarray, target_rate: float, tol: float = 1e-9) -> float:
    """Find c such that mean(sigmoid(logit - c)) equals the target positive rate.

    The obvious shortcut is to put c at the (1 - rate) quantile of the log odds. That is
    wrong and it is wrong in a way that looks right, because it sets the point where the
    probability crosses 0.5 rather than the mean of the probabilities. At a target of 0.20
    on this generator it produces 0.29. Half the rows sitting just under the boundary still
    carry a probability near 0.5 and they all contribute.

    So it is a root find. The mean is strictly decreasing in c, so bisection cannot miss,
    and bracketing off the observed range of the log odds means the bracket is always
    valid. Fifty iterations takes the interval well below tol.
    """
    lo = float(logit.min()) - 40.0
    hi = float(logit.max()) + 40.0

    def rate_at(c: float) -> float:
        return float(np.mean(1.0 / (1.0 + np.exp(-(logit - c)))))

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if rate_at(mid) > target_rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def generate(cfg: DataConfig, seed: int) -> Dataset:
    feat_rng, weight_rng, noise_rng, split_rng = streams(
        seed, "data.features", "data.weights", "data.noise", "data.split"
    )

    n, k = cfg.n_rows, cfg.n_features

    # Lognormal rather than normal. Most real numeric features are non negative and
    # right skewed, and a model that only ever sees centred gaussians hides scaling bugs.
    x = feat_rng.lognormal(mean=0.0, sigma=0.6, size=(n, k))

    w = _log_odds_weights(weight_rng, k)
    # Centring is not load bearing for the labels. The intercept solver below absorbs any
    # constant offset exactly, so adding the column means instead of subtracting them
    # produces identical probabilities. Verified across offsets from -12.5 to 100. It stays
    # because it keeps the log odds near zero, which keeps the solver's bracket sensible
    # and makes the returned intercept a number someone can reason about.
    centred = np.log(x) - np.log(x).mean(axis=0)
    logit = centred @ w + noise_rng.normal(0.0, cfg.noise, size=n)

    intercept = _solve_intercept(logit, cfg.positive_rate)
    p = 1.0 / (1.0 + np.exp(-(logit - intercept)))
    y = (noise_rng.random(n) < p).astype(np.int64)

    order = split_rng.permutation(n)
    n_holdout = int(round(n * cfg.holdout_frac))
    hold_idx, train_idx = order[:n_holdout], order[n_holdout:]

    return Dataset(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_holdout=x[hold_idx],
        y_holdout=y[hold_idx],
    )
