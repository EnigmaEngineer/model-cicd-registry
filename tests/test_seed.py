"""Seed checks.

The important one here is `check_streams_do_not_collide_across_seeds`, because the obvious
implementation of named streams is `seed + offset` and it passes every other check in this
file.
"""

from __future__ import annotations

import numpy as np

from mcr import seed as S


def check_same_seed_and_stream_gives_same_draws():
    a = S.generator(11, "data.features").normal(size=50)
    b = S.generator(11, "data.features").normal(size=50)
    assert np.array_equal(a, b)


def check_different_streams_differ_under_one_seed():
    a = S.generator(11, "data.features").normal(size=50)
    b = S.generator(11, "data.noise").normal(size=50)
    assert not np.array_equal(a, b)


def check_different_seeds_differ_on_one_stream():
    a = S.generator(11, "data.features").normal(size=50)
    b = S.generator(12, "data.features").normal(size=50)
    assert not np.array_equal(a, b)


def check_streams_do_not_collide_across_seeds():
    """`seed + offset` would fail this and pass everything else in the file.

    Under addition, seed 1 with the second stream and seed 2 with the first are the same
    integer, so two runs one seed apart would share a random stream. The hash makes that
    impossible to arrange.
    """
    seen = {}
    for seed in range(40):
        for stream in ("data.features", "data.noise", "data.split", "model.init"):
            key = S._derive(seed, stream)
            assert key not in seen, "collision: {} and {}".format(
                seen[key], (seed, stream)
            )
            seen[key] = (seed, stream)


def check_derive_is_stable_across_processes():
    """Pinned literals, not a self comparison.

    Comparing two calls in one process would pass even if the derivation used Python's
    hash(), which is salted per process and changes between runs. These numbers were taken
    from a run and pasted back, so a change to the derivation shows up here as a failure
    rather than as every stored artefact silently becoming unreachable.
    """
    assert S._derive(0, "data.features") == 9032064831517925421
    assert S._derive(20260911, "model.init") == 16109260240651496545


def check_stream_version_is_in_the_derivation():
    """Bumping the version has to move every stream, or it buys nothing."""
    before = S._derive(5, "data.noise")
    original = S.STREAM_VERSION
    try:
        S.STREAM_VERSION = original + 1
        after = S._derive(5, "data.noise")
    finally:
        S.STREAM_VERSION = original
    assert before != after


def check_streams_helper_returns_in_order():
    a, b = S.streams(3, "x", "y")
    assert np.array_equal(a.normal(size=10), S.generator(3, "x").normal(size=10))
    assert np.array_equal(b.normal(size=10), S.generator(3, "y").normal(size=10))


def check_duplicate_stream_names_are_refused():
    """Two handles on one stream look independent and are not. The second consumer would
    get draws the first has already used, which is the correlation bug this module exists
    to prevent."""
    try:
        S.streams(3, "x", "x")
    except ValueError as exc:
        assert "distinct" in str(exc)
    else:
        raise AssertionError("duplicate stream names were accepted")


def check_seed_zero_is_accepted_and_minus_one_is_not():
    """The boundary from both sides. Testing only -1 leaves `seed < 0` free to drift to
    `seed < 1`, which refuses the most obvious seed anybody would pick."""
    S.generator(0, "x")
    try:
        S.generator(-1, "x")
    except ValueError:
        pass
    else:
        raise AssertionError("a negative seed was accepted")


def check_empty_and_negative_are_refused():
    for args in ((-1, "x"), (1, "")):
        try:
            S.generator(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted {}".format(args))

    try:
        S.streams(1)
    except ValueError:
        pass
    else:
        raise AssertionError("streams() accepted zero names")
