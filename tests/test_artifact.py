"""Artefact checks.

The registry keys on the content hash, so a hash that moves when nothing
meaningful changed, or holds still when something did, breaks the project rather than one
function.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from mcr import artifact as A
from mcr.config import from_dict
from mcr.model import Model

RAW = {
    "name": "t",
    "seed": 7,
    "data": {
        "n_rows": 500,
        "n_features": 3,
        "positive_rate": 0.2,
        "noise": 1.0,
        "holdout_frac": 0.2,
    },
    "model": {"learning_rate": 0.1, "epochs": 5, "l2": 0.0, "init": "zeros"},
}


def _model(bias=0.25):
    return Model(
        weights=np.array([0.1, -0.2, 0.3]),
        bias=bias,
        mean=np.array([0.0, 1.0, 2.0]),
        scale=np.array([1.0, 1.0, 1.0]),
        epochs_run=5,
        final_loss=0.5,
    )


def _art(bias=0.25, metrics=None):
    return A.build(from_dict(RAW), _model(bias), metrics or {"holdout_roc_auc": 0.8})


def check_hash_is_stable():
    assert _art().content_hash() == _art().content_hash()


def check_hash_moves_when_the_model_moves():
    assert _art(bias=0.25).content_hash() != _art(bias=0.26).content_hash()


def check_hash_moves_when_metrics_move():
    a = _art(metrics={"holdout_roc_auc": 0.80})
    b = _art(metrics={"holdout_roc_auc": 0.81})
    assert a.content_hash() != b.content_hash()


def check_numpy_scalars_do_not_reach_json():
    """json.dumps raises on a numpy float64, so this would be a write time crash rather
    than a wrong file. It is still worth pinning, because the fix is one tolist() call
    that a later edit could drop."""
    payload = _art().payload
    for key in ("weights", "mean", "scale"):
        for v in payload["model"][key]:
            assert type(v) is float, "{} held a {}".format(key, type(v))
    assert type(payload["model"]["bias"]) is float


def check_bytes_end_in_one_newline():
    """A file with no trailing newline shows as "\\ No newline at end of file" in every
    diff the registry produces later."""
    raw = _art().to_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


def check_round_trip_through_disk():
    path = os.path.join(tempfile.mkdtemp(), "m.json")
    written = _art().write(path)
    back = A.read(path)
    assert back.content_hash() == written


def check_a_hand_edited_file_is_refused():
    """The registry points at a hash. If somebody edits the file, the hash stops
    identifying its contents and every later comparison is against something that is not
    there any more. Refuse at read time instead."""
    path = os.path.join(tempfile.mkdtemp(), "m.json")
    _art().write(path)
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace('"bias":0.25', '"bias":  0.25'))
    try:
        A.read(path)
    except ValueError as exc:
        assert "round trip" in str(exc)
    else:
        raise AssertionError("a reformatted artefact was accepted")


def check_key_order_does_not_reach_the_bytes():
    """Two payloads with identical content built in a different order are one artefact.
    Without sort_keys they are two, and the registry would store both."""
    a = _art()
    flipped = A.Artifact(payload=dict(reversed(list(a.payload.items()))))
    assert flipped.to_bytes() == a.to_bytes()


def check_write_creates_the_directory():
    path = os.path.join(tempfile.mkdtemp(), "nested", "deeper", "m.json")
    _art().write(path)
    assert os.path.exists(path)


def check_float_precision_survives():
    """repr of a float64 is the shortest string that round trips, so this is exact rather
    than close. A serialiser that formatted to a fixed number of places would lose the
    last bits and two models that differ slightly would store as one."""
    m = _model()
    exact = float(np.float64(0.1) + np.float64(0.2))
    art = A.build(from_dict(RAW), m, {"odd": exact})
    path = os.path.join(tempfile.mkdtemp(), "m.json")
    art.write(path)
    assert A.read(path).payload["metrics"]["odd"] == exact
