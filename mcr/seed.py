"""Seed control.

One rule here and it is the whole module: every consumer of randomness gets its own
generator derived from the run seed, and nothing reads a global.

The global approach is `np.random.seed(n)` at the top of main. It works right up until two
things draw from the stream in an order that changes. Add a shuffle, or reorder two calls,
and every draw after it moves. The run is still deterministic and it is no longer the same
run, which is the worst version because the artefact changes and the config does not.

Named streams cost four lines and remove the ordering dependency entirely.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Bumping this invalidates every stored artefact, so it is a deliberate act.
# If a stream's meaning changes, bump it and say so in the changelog.
STREAM_VERSION = 1


def _derive(seed: int, stream: str) -> int:
    """Map (seed, stream name) to a 64 bit integer.

    Hashing rather than `seed + offset` because adjacent seeds with adjacent offsets
    collide. Seed 1 stream b and seed 2 stream a would be the same integer under addition,
    and a collision between two streams is exactly the correlation the split exists to
    avoid. Python's own hash() is salted per process and cannot be used.
    """
    key = "{}:{}:{}".format(STREAM_VERSION, seed, stream).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def generator(seed: int, stream: str) -> np.random.Generator:
    if seed < 0:
        raise ValueError("seed must be non negative, got {}".format(seed))
    if not stream:
        raise ValueError("stream name cannot be empty")
    return np.random.default_rng(_derive(seed, stream))


def streams(seed: int, *names: str) -> tuple:
    """Several independent generators in one call, in the order asked for."""
    if not names:
        raise ValueError("ask for at least one stream")
    if len(set(names)) != len(names):
        raise ValueError("stream names must be distinct, got {}".format(list(names)))
    return tuple(generator(seed, n) for n in names)
