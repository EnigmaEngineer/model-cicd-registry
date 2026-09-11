"""Data checks.

The generator is the one thing in this repo whose output nothing else can check, so the
summary statistics get asserted rather than eyeballed. A generator that silently produces
one class, or puts every row in the holdout, would pass a shape check and destroy every
number downstream.
"""

from __future__ import annotations

import numpy as np

from mcr import data as D
from mcr.config import DataConfig

CFG = DataConfig(
    n_rows=4000, n_features=8, positive_rate=0.2, noise=1.0, holdout_frac=0.25
)


def check_split_sizes_match_the_config():
    ds = D.generate(CFG, seed=1)
    assert len(ds.y_holdout) == 1000, len(ds.y_holdout)
    assert len(ds.y_train) == 3000, len(ds.y_train)
    assert ds.x_train.shape == (3000, 8)


def check_same_seed_reproduces_exactly():
    a, b = D.generate(CFG, seed=1), D.generate(CFG, seed=1)
    assert np.array_equal(a.x_train, b.x_train)
    assert np.array_equal(a.y_train, b.y_train)
    assert np.array_equal(a.x_holdout, b.x_holdout)


def check_different_seed_changes_the_corpus():
    """The control on the check above. Without it, a generate() that ignored the seed
    entirely would pass the reproducibility check perfectly."""
    a, b = D.generate(CFG, seed=1), D.generate(CFG, seed=2)
    assert not np.array_equal(a.x_train, b.x_train)
    assert not np.array_equal(a.y_train, b.y_train)


def check_positive_rate_lands_near_the_target():
    """The check that caught the real bug in the generator.

    The first generator put the intercept at the (1 - rate) quantile of the log odds,
    which sets the 0.5 crossing point rather than the mean probability. At a target of
    0.20 it produced 0.2905. A shape check would have passed and every downstream metric
    would have been computed on a corpus that was not the one the config asked for.

    The band is binomial noise on 4,000 draws and nothing else now.
    """
    for seed in (3, 11, 29):
        ds = D.generate(CFG, seed=seed)
        y = np.concatenate([ds.y_train, ds.y_holdout])
        rate = float(y.mean())
        assert 0.18 < rate < 0.22, (seed, rate)


def check_the_intercept_solver_hits_its_target():
    """Direct on the solver rather than through the draw, so it has no binomial noise in
    it. Also covers rates the shipped configs do not use, since the promotion gate will
    eventually want a config with a rare positive."""
    rng = np.random.default_rng(0)
    logit = rng.normal(0.0, 2.0, size=5000)
    for target in (0.02, 0.1, 0.2, 0.5, 0.8):
        c = D._solve_intercept(logit, target)
        got = float(np.mean(1.0 / (1.0 + np.exp(-(logit - c)))))
        assert abs(got - target) < 1e-6, (target, got)


def check_the_quantile_shortcut_would_have_been_wrong():
    """Pins the defect itself, not just the fix.

    If somebody replaces the root find with the quantile again because it is shorter,
    this is the check that explains why not. The two disagree by a wide margin and the
    direction is always the same.
    """
    rng = np.random.default_rng(1)
    logit = rng.normal(0.0, 2.0, size=5000)
    target = 0.2
    shortcut = float(np.quantile(logit, 1.0 - target))
    rate_from_shortcut = float(np.mean(1.0 / (1.0 + np.exp(-(logit - shortcut)))))
    assert rate_from_shortcut > target + 0.05, rate_from_shortcut


def check_both_classes_are_present_in_both_splits():
    """A holdout with one class makes AUC undefined and makes accuracy meaningless.
    Cheap to check and it would be expensive to discover at the gate."""
    ds = D.generate(CFG, seed=4)
    for name, y in (("train", ds.y_train), ("holdout", ds.y_holdout)):
        assert set(np.unique(y)) == {0, 1}, "{} had {}".format(name, np.unique(y))


def check_train_and_holdout_do_not_overlap():
    """A leaked row inflates every holdout metric and the gate is built on holdout
    metrics. Compared by row bytes rather than by index, because the indices are internal
    to generate() and this check should not know about them."""
    ds = D.generate(CFG, seed=5)
    train_rows = {r.tobytes() for r in ds.x_train}
    hold_rows = {r.tobytes() for r in ds.x_holdout}
    assert not (train_rows & hold_rows)


def check_features_are_positive():
    """The model takes log(x). A non positive feature would give nan weights, and the
    lognormal draw is what guarantees it cannot happen."""
    ds = D.generate(CFG, seed=6)
    assert (ds.x_train > 0).all()
    assert (ds.x_holdout > 0).all()


def check_signal_is_concentrated_in_the_first_features():
    """The generator damps the tail features by 0.05 and that is the only reason the l2
    term has anything to bite on. If the damping went away this check is what says so."""
    rng = np.random.default_rng(0)
    w = D._log_odds_weights(rng, 8)
    assert np.abs(w[:3]).mean() > np.abs(w[3:]).mean() * 5


def check_the_damping_boundary_is_exactly_three():
    """A mean based check cannot see the boundary move by one.

    The first version of the check above compared the mean of the head against the mean of
    the tail at a factor of 5. A mutant moving the boundary from 3 to 4 left feature 3
    undamped, at 0.1049 against a real 0.0052, and the whole suite stayed green. The mean
    absorbed one extra term.

    So compare against the undamped draw directly. Same generator and same seed, so the
    ratio is exactly the damping factor wherever it applies and exactly 1 where it does
    not, with no averaging to hide behind.
    """
    plain = np.random.default_rng(0).normal(0.0, 1.0, size=8)
    damped = D._log_odds_weights(np.random.default_rng(0), 8)

    for i in range(3):
        assert damped[i] == plain[i], "feature {} should not be damped".format(i)
    for i in range(3, 8):
        assert abs(damped[i] - plain[i] * 0.05) < 1e-15, (
            "feature {} should be damped by 0.05".format(i)
        )


def check_damping_is_skipped_when_there_is_no_tail():
    """n_features of 3 or fewer means the slice would be empty. The guard exists for that
    and a mutant on its comparison has to break something."""
    plain = np.random.default_rng(2).normal(0.0, 1.0, size=3)
    assert np.array_equal(D._log_odds_weights(np.random.default_rng(2), 3), plain)

    plain4 = np.random.default_rng(2).normal(0.0, 1.0, size=4)
    got4 = D._log_odds_weights(np.random.default_rng(2), 4)
    assert got4[2] == plain4[2]
    assert abs(got4[3] - plain4[3] * 0.05) < 1e-15


def check_the_solver_bracket_holds_for_extreme_targets():
    """The bracket is the observed log odds range padded by 40. A mutant on either pad
    has to fail somewhere, and a narrow distribution with a far out target is where."""
    rng = np.random.default_rng(3)
    tight = rng.normal(0.0, 0.05, size=2000)
    for target in (0.001, 0.999):
        c = D._solve_intercept(tight, target)
        got = float(np.mean(1.0 / (1.0 + np.exp(-(tight - c)))))
        assert abs(got - target) < 1e-6, (target, got)


def check_holdout_fraction_is_honoured_at_an_odd_size():
    """round() rather than int() truncation. 999 rows at 0.25 is 249.75 and truncating
    would quietly hand the model an extra row every time."""
    cfg = DataConfig(
        n_rows=999, n_features=4, positive_rate=0.3, noise=1.0, holdout_frac=0.25
    )
    ds = D.generate(cfg, seed=7)
    assert len(ds.y_holdout) == 250, len(ds.y_holdout)
    assert len(ds.y_train) == 749, len(ds.y_train)
