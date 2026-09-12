"""Config checks.

The fixtures here build the collision in rather than hoping for it. A fixture whose two
configs differ in several places cannot tell you which field the fingerprint is actually
reading, so the pairs below differ in exactly one thing each.
"""

from __future__ import annotations

import os
import tempfile

from mcr import config as C

BASE = {
    "name": "t",
    "seed": 7,
    "data": {
        "n_rows": 500,
        "n_features": 4,
        "positive_rate": 0.2,
        "noise": 1.0,
        "holdout_frac": 0.2,
    },
    "model": {"learning_rate": 0.1, "epochs": 5, "l2": 0.0, "init": "zeros"},
}


def _copy(**over):
    raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE.items()}
    for dotted, value in over.items():
        section, _, field = dotted.partition("__")
        if field:
            raw[section][field] = value
        else:
            raw[section] = value
    return raw


def _expect_error(raw, fragment):
    try:
        C.from_dict(raw)
    except C.ConfigError as exc:
        assert fragment in str(exc), "wrong message: {}".format(exc)
        return
    raise AssertionError("expected ConfigError mentioning '{}'".format(fragment))


def check_baseline_config_loads():
    cfg = C.from_dict(_copy())
    assert cfg.name == "t"
    assert cfg.data.n_rows == 500
    assert cfg.model.init == "zeros"


def check_fingerprint_is_stable_across_calls():
    a = C.from_dict(_copy()).fingerprint()
    b = C.from_dict(_copy()).fingerprint()
    assert a == b, "{} != {}".format(a, b)


def check_the_fingerprint_is_a_function_of_the_run_and_not_of_the_yaml():
    """The defect docs/adr-0002 is about.

    `noise: 1` and `noise: 1.0` are the same training run. Before the loader coerced its
    own declared types they fingerprinted differently, because the dataclass stored
    whatever yaml handed it and json.dumps wrote `1` against `1.0`. That put two registry
    keys on one model.

    Every pair below is one run written two ways. Integral floats into int fields are the
    same case from the other side.
    """
    pairs = [
        (_copy(data__noise=1), _copy(data__noise=1.0)),
        (_copy(model__l2=0), _copy(model__l2=0.0)),
        (_copy(model__epochs=5), _copy(model__epochs=5.0)),
        (_copy(seed=7), _copy(seed=7.0)),
        (_copy(model__learning_rate="0.1"), _copy()),
        (_copy(data__n_rows="500"), _copy()),
    ]
    for left, right in pairs:
        a, b = C.from_dict(left).fingerprint(), C.from_dict(right).fingerprint()
        assert a == b, "{} fingerprints as {} and {}".format(left, a, b)


def check_coercion_refuses_a_cast_that_would_lose_something():
    """The other half. Coercing without a loss rule would turn 5.5 epochs into 5 quietly,
    which trades a visible fingerprint problem for an invisible training one."""
    _expect_error(_copy(model__epochs=5.5), "whole number")
    _expect_error(_copy(seed=7.5), "whole number")
    _expect_error(_copy(data__n_rows=500.5), "whole number")
    _expect_error(_copy(data__noise="abc"), "not a number")
    _expect_error(_copy(model__epochs="five"), "not a number")


def check_bool_is_refused_everywhere_it_could_pass_as_a_number():
    """`isinstance(True, int)` is True, so a bool reaches an int field unless something
    stops it, and `epochs: yes` is a real thing to type in YAML."""
    _expect_error(_copy(model__epochs=True), "bool")
    _expect_error(_copy(data__noise=False), "bool")
    _expect_error(_copy(seed=True), "bool")
    _expect_error(_copy(model__init=True), "must be a string")


def check_a_non_string_name_is_refused_rather_than_stringified():
    """`name: 2026` is more likely a mistake than an intent. str(value) would accept it and
    the run would be called "2026" with nothing saying so."""
    _expect_error(_copy(name=2026), "must be a string")
    _expect_error(_copy(model__init=1), "must be a string")


def check_declared_types_resolves_past_the_postponed_annotations():
    """`fields(cls)[i].type` is the string "int" in that module, because of the
    `from __future__ import annotations` at its top. Calling it would call a string."""
    types = C.declared_types(C.DataConfig)
    assert types["n_rows"] is int, types["n_rows"]
    assert types["noise"] is float, types["noise"]
    assert C.declared_types(C.ModelConfig)["init"] is str
    assert set(types) == {"n_rows", "n_features", "positive_rate", "noise", "holdout_frac"}


def check_coerce_refuses_a_type_it_has_no_rule_for():
    """A field declared as something this module does not handle has to fail loudly. The
    silent version returns the value untouched and the fingerprint goes back to depending
    on whatever the file happened to contain."""
    try:
        C.coerce([1, 2], list, "somewhere")
    except C.ConfigError as exc:
        assert "no rule" in str(exc), str(exc)
    else:
        raise AssertionError("coerced a value to a type with no rule")


def check_fingerprint_moves_on_every_field():
    """One field at a time. A fingerprint that ignores a field it should read is the
    failure that lets the gate call two different runs the same run."""
    base = C.from_dict(_copy()).fingerprint()
    for over in (
        {"name": "other"},
        {"seed": 8},
        {"data__n_rows": 501},
        {"data__n_features": 5},
        {"data__positive_rate": 0.21},
        {"data__noise": 1.1},
        {"data__holdout_frac": 0.25},
        {"model__learning_rate": 0.11},
        {"model__epochs": 6},
        {"model__l2": 0.001},
        {"model__init": "normal"},
    ):
        other = C.from_dict(_copy(**over)).fingerprint()
        assert other != base, "fingerprint ignored {}".format(over)


def check_fingerprint_ignores_key_order():
    """Two YAML files with the same content in a different order are the same run.

    Without sort_keys this fails, and it fails silently in the direction that matters,
    because the gate would treat a reordered config as an unrelated model.
    """
    raw = _copy()
    reordered = {
        "model": dict(reversed(list(raw["model"].items()))),
        "data": dict(reversed(list(raw["data"].items()))),
        "seed": raw["seed"],
        "name": raw["name"],
    }
    assert C.from_dict(raw).fingerprint() == C.from_dict(reordered).fingerprint()


def check_unknown_key_is_refused():
    _expect_error(_copy(model__epoch=40), "unknown keys in 'model'")
    _expect_error(_copy(nme="x"), "unknown top level keys")


def check_missing_key_is_refused():
    raw = _copy()
    del raw["model"]["l2"]
    _expect_error(raw, "'model' is missing: l2")


def check_ranges_are_refused():
    _expect_error(_copy(data__n_rows=9), "n_rows must be at least 10")
    _expect_error(_copy(data__positive_rate=0.0), "positive_rate")
    _expect_error(_copy(data__positive_rate=1.0), "positive_rate")
    _expect_error(_copy(data__noise=-0.1), "noise cannot be negative")
    _expect_error(_copy(model__learning_rate=0.0), "learning_rate must be positive")
    _expect_error(_copy(model__epochs=0), "epochs must be at least 1")
    _expect_error(_copy(model__l2=-1.0), "l2 cannot be negative")
    _expect_error(_copy(model__init="glorot"), "init must be one of")
    _expect_error(_copy(seed=-1), "seed must be non negative")
    _expect_error(_copy(name=""), "name cannot be empty")


def check_every_limit_is_tested_at_its_own_boundary():
    """Refusing 9 when the limit is 10 does not prove the limit is 10.

    The first version of this file tested each rule well away from its edge, with n_rows
    of 9 against a limit of 10 and a seed of -1 against a limit of 0. A mutation pass moved
    every threshold by one and moved every comparison between strict and non strict, and
    eleven of those mutants survived. Each pair below pins one limit from both sides, which
    is the only arrangement that can see a boundary shift.
    """
    accept_reject = [
        # field, smallest accepted, largest rejected
        ("data__n_rows", 10, 9),
        ("data__n_features", 1, 0),
        ("model__epochs", 1, 0),
        ("seed", 0, -1),
    ]
    for field, ok_value, bad_value in accept_reject:
        # n_rows of 10 at holdout 0.2 gives 2 rows, which the holdout rule refuses first,
        # so that pair is checked with a holdout fraction that clears it.
        extra = {"data__holdout_frac": 0.5} if field == "data__n_rows" else {}
        C.from_dict(_copy(**{field: ok_value}, **extra))
        try:
            C.from_dict(_copy(**{field: bad_value}, **extra))
        except C.ConfigError:
            pass
        else:
            raise AssertionError("{} accepted {}".format(field, bad_value))


def check_zero_is_allowed_where_zero_is_meaningful():
    """l2 of 0 is no regularisation and it is the default anybody starts from. noise of 0
    is a clean corpus. A `< 0` that drifted to `<= 0` would refuse both, and every other
    check in this file uses non zero values."""
    C.from_dict(_copy(model__l2=0.0))
    C.from_dict(_copy(data__noise=0.0))
    C.from_dict(_copy(seed=0))


def check_the_open_intervals_reject_both_ends():
    """positive_rate and holdout_frac are open on both sides. A strictness flip on either
    comparison lets 0.0 or 1.0 through, and both are degenerate. A positive rate of 0
    produces a single class corpus and a holdout fraction of 1 leaves nothing to train
    on."""
    for field in ("data__positive_rate", "data__holdout_frac"):
        for bad in (0.0, 1.0):
            try:
                C.from_dict(_copy(**{field: bad}))
            except C.ConfigError:
                pass
            else:
                raise AssertionError("{} accepted {}".format(field, bad))


def check_the_holdout_floor_is_exactly_five_rows():
    """100 rows at 0.05 is 5 and is accepted. 0.04 rounds to 4 and is refused."""
    C.from_dict(_copy(data__n_rows=100, data__holdout_frac=0.05))
    _expect_error(
        _copy(data__n_rows=100, data__holdout_frac=0.04), "fewer than 5 holdout"
    )


def check_learning_rate_must_be_strictly_positive():
    """A rate of 0 trains nothing and the loop would still run every epoch, so it returns
    the initial weights wearing a trained model's name."""
    C.from_dict(_copy(model__learning_rate=1e-9))
    _expect_error(_copy(model__learning_rate=0.0), "learning_rate must be positive")


def check_tiny_holdout_is_refused():
    """20 rows at 0.2 is 4 holdout rows. A gate scoring 4 rows is noise, and a gate
    scoring 0 rows scores every candidate as perfect."""
    _expect_error(_copy(data__n_rows=20, data__holdout_frac=0.2), "fewer than 5 holdout")


def check_config_is_frozen():
    """Assert the type, not the message.

    The first version of this matched on the word "frozen" appearing in the exception
    text. FrozenInstanceError says "cannot assign to field 'seed'", so the check failed
    against code that was behaving correctly. The type is the contract here.
    """
    import dataclasses

    cfg = C.from_dict(_copy())
    for obj, field, value in (
        (cfg, "seed", 9),
        (cfg.data, "n_rows", 1),
        (cfg.model, "l2", 1.0),
    ):
        try:
            setattr(obj, field, value)
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("{}.{} was mutable".format(type(obj).__name__, field))


def check_load_reads_a_file():
    handle, path = tempfile.mkstemp(suffix=".yml")
    os.close(handle)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(
                "name: f\nseed: 3\n"
                "data:\n  n_rows: 500\n  n_features: 4\n  positive_rate: 0.2\n"
                "  noise: 1.0\n  holdout_frac: 0.2\n"
                "model:\n  learning_rate: 0.1\n  epochs: 5\n  l2: 0.0\n  init: zeros\n"
            )
        cfg = C.load(path)
        assert cfg.name == "f" and cfg.seed == 3
    finally:
        os.unlink(path)


def check_empty_file_is_refused():
    handle, path = tempfile.mkstemp(suffix=".yml")
    os.close(handle)
    try:
        try:
            C.load(path)
        except C.ConfigError as exc:
            assert "empty" in str(exc)
        else:
            raise AssertionError("an empty config file was accepted")
    finally:
        os.unlink(path)


def check_shipped_configs_are_valid():
    """The files in configs/ are the ones a reader runs first. A repo whose own example
    config does not load is the fastest possible way to lose them."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = 0
    for entry in sorted(os.listdir(os.path.join(root, "configs"))):
        if entry.endswith(".yml"):
            C.load(os.path.join(root, "configs", entry))
            found += 1
    assert found >= 2, "expected at least 2 shipped configs, found {}".format(found)
