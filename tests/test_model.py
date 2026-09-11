"""Model checks.

AUC gets the most attention here because it is the number the promotion gate reads, and
the tie handling inside it is the part most likely to be wrong in a way
nobody notices. A model that predicts one constant is a real candidate and it has to score
0.5 rather than 1.0.
"""

from __future__ import annotations

import numpy as np

from mcr import model as M
from mcr.config import ModelConfig

CFG = ModelConfig(learning_rate=0.5, epochs=200, l2=0.0, init="zeros")


def _easy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0.0, 0.5, size=(n, 3))
    y = (np.log(x[:, 0]) > 0).astype(np.int64)
    return x, y


def check_sigmoid_does_not_overflow():
    """The naive 1/(1+exp(-z)) returns nan at z of -800 rather than 0.0, and it shows up
    as a nan loss partway through training instead of as an error."""
    z = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    out = M._sigmoid(z)
    assert np.isfinite(out).all(), out
    assert out[0] == 0.0 and out[-1] == 1.0
    assert abs(out[2] - 0.5) < 1e-12


def check_fit_learns_a_separable_problem():
    x, y = _easy()
    m = M.fit(x, y, CFG, seed=0)
    assert M.accuracy(y, m.predict_proba(x)) > 0.95


def check_loss_falls():
    x, y = _easy()
    short = M.fit(x, y, ModelConfig(0.5, 5, 0.0, "zeros"), seed=0)
    long = M.fit(x, y, ModelConfig(0.5, 200, 0.0, "zeros"), seed=0)
    assert long.final_loss < short.final_loss


def check_zeros_init_ignores_the_seed():
    """Recorded as a property rather than assumed. With a zeros init the fit is a
    deterministic function of the data alone, so the seed reaches the model only through
    the corpus. The probe in scripts/ is what measures the consequence."""
    x, y = _easy()
    a = M.fit(x, y, CFG, seed=1)
    b = M.fit(x, y, CFG, seed=999)
    assert np.array_equal(a.weights, b.weights)


def check_normal_init_uses_the_seed():
    """The other half. If this passed as well as the check above, `init` would be doing
    nothing at all."""
    x, y = _easy()
    cfg = ModelConfig(0.5, 10, 0.0, "normal")
    a = M.fit(x, y, cfg, seed=1)
    b = M.fit(x, y, cfg, seed=999)
    assert not np.array_equal(a.weights, b.weights)


def check_constant_feature_does_not_produce_nan():
    """Zero spread means dividing by zero in the standardiser. The guard clamps it to 1.0
    and the weight then stays where init put it."""
    x, y = _easy()
    x[:, 2] = 3.0
    m = M.fit(x, y, CFG, seed=0)
    assert np.isfinite(m.weights).all(), m.weights
    assert m.scale[2] == 1.0


def check_l2_shrinks_the_weights():
    x, y = _easy()
    loose = M.fit(x, y, ModelConfig(0.5, 200, 0.0, "zeros"), seed=0)
    tight = M.fit(x, y, ModelConfig(0.5, 200, 1.0, "zeros"), seed=0)
    assert float(tight.weights @ tight.weights) < float(loose.weights @ loose.weights)


def check_predict_refuses_a_width_mismatch():
    x, y = _easy()
    m = M.fit(x, y, CFG, seed=0)
    try:
        m.predict_proba(np.ones((4, 7)))
    except ValueError as exc:
        assert "expects 3 features" in str(exc)
    else:
        raise AssertionError("a 7 column matrix was scored by a 3 feature model")


def check_fit_refuses_empty_and_mismatched_input():
    try:
        M.fit(np.ones((0, 3)), np.array([], dtype=np.int64), CFG, seed=0)
    except ValueError as exc:
        assert "zero rows" in str(exc)
    else:
        raise AssertionError("fit accepted zero rows")

    try:
        M.fit(np.ones((5, 3)), np.zeros(4, dtype=np.int64), CFG, seed=0)
    except ValueError as exc:
        assert "5 rows" in str(exc)
    else:
        raise AssertionError("fit accepted mismatched lengths")


def check_auc_on_a_known_ordering():
    """Hand computable. Three positives and three negatives perfectly separated is 1.0,
    and reversing the scores is 0.0."""
    y = np.array([0, 0, 0, 1, 1, 1])
    assert M.roc_auc(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == 1.0
    assert M.roc_auc(y, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == 0.0


def check_auc_of_a_constant_predictor_is_half():
    """The one that matters. Without average ranks on ties this returns 1.0, and a model
    that predicts one number for every row would sail through the promotion gate."""
    y = np.array([0, 1, 0, 1, 1, 0])
    assert M.roc_auc(y, np.full(6, 0.42)) == 0.5


def check_auc_handles_partial_ties():
    """A tie spanning one positive and one negative. Four pairs, three clean wins and one
    tie worth 0.5, so 3.5 over 4.

    I wrote 0.75 here first and the implementation was right. Hence the pair check below,
    which does not depend on my arithmetic at all.
    """
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.5, 0.5, 0.9])
    assert abs(M.roc_auc(y, p) - 0.875) < 1e-12


def check_auc_matches_the_pairwise_definition():
    """The rank form is an optimisation of "count the pairs". Check it against the thing
    it optimises, on inputs with enough ties to matter.

    Scores are drawn from a small set of integers on purpose. With continuous scores ties
    never occur and the tie branch never runs, which is the fixture mistake that hides
    this class of bug.
    """
    rng = np.random.default_rng(4)
    for _ in range(30):
        n = int(rng.integers(4, 40))
        y = rng.integers(0, 2, size=n)
        if len(set(y.tolist())) < 2:
            continue
        p = rng.integers(0, 5, size=n).astype(float)
        pos, neg = p[y == 1], p[y == 0]
        brute = sum(
            1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg
        ) / (len(pos) * len(neg))
        assert abs(M.roc_auc(y, p) - brute) < 1e-12, (y, p)


def check_auc_refuses_a_single_class():
    """An all positive holdout makes AUC undefined. Returning 1.0 or nan would both be
    read as a result by whatever prints it."""
    try:
        M.roc_auc(np.ones(5, dtype=np.int64), np.linspace(0, 1, 5))
    except ValueError as exc:
        assert "both classes" in str(exc)
    else:
        raise AssertionError("AUC was computed on one class")


def check_log_loss_is_clipped():
    """A confident wrong prediction at exactly 0.0 gives log(0) and an infinite loss.
    Clipping keeps it large and finite, which is the difference between a bad number and
    an unusable one."""
    loss = M.log_loss(np.array([1.0]), np.array([0.0]))
    assert np.isfinite(loss) and loss > 30
