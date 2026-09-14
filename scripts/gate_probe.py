"""What the gate does, measured, with the answer key printed beside it.

    python3 scripts/gate_probe.py

Every arm carries a control: a stand in rule, run on the same inputs, that gets the answer
wrong in the way the arm exists to catch. An arm whose control also passes has proved
nothing, and the probe exits nonzero when that happens.

The last section is the floor. A gate that promotes everything and a gate that refuses
everything are both scored on the same cases, because a headline of "eight of nine" means
nothing until somebody says what doing nothing scores.

This needs no MLflow. It works on artefacts rather than on registry versions, which is what
makes it cheap enough to run on every change.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from mcr import gate, train as train_mod  # noqa: E402
from mcr.config import TrainConfig, load  # noqa: E402

CONFIGS = {
    "baseline": "configs/baseline.yml",
    "candidate-lr": "configs/candidate-lr.yml",
    "candidate-underfit": "configs/candidate-underfit.yml",
    "candidate-inverted": "configs/candidate-inverted.yml",
    "candidate-diverged": "configs/candidate-diverged.yml",
}


def root(rel: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel)


def build(spec: gate.HoldoutSpec, name: str, cfg: TrainConfig) -> gate.Scored:
    result = train_mod.run(cfg)
    return gate.score(
        name=name,
        artifact_hash=result.content_hash,
        payload=result.artifact.payload,
        spec=spec,
        reported={"holdout_{}".format(gate.METRIC): result.metrics["holdout_log_loss"]},
    )


def naive(candidate, incumbent) -> str:
    """The rule the gate replaced. Kept here so the arms have something to be compared to.

    One line, reads correctly, and it is wrong in two separate ways that the arms below
    pin. It compares the numbers the runs recorded rather than scoring anything, and its
    comparison against a non finite value is False whichever way round it is written.
    """
    if incumbent is None:
        return gate.PROMOTE
    key = "holdout_{}".format(gate.METRIC)
    if candidate.reported[key] < incumbent.reported[key]:
        return gate.PROMOTE
    return gate.REJECT


def main() -> int:
    cfgs = {k: load(root(v)) for k, v in CONFIGS.items()}
    spec = gate.spec_from_config(cfgs["baseline"])
    fp = spec.fingerprint()

    scored = {k: build(spec, k, c) for k, c in cfgs.items()}

    # A model trained on a different corpus. Same model config as the incumbent in every
    # respect except the seed, so it is genuinely no better and no worse, and its own
    # holdout happened to be an easier one.
    easy_cfg = TrainConfig(
        name="baseline-other-corpus",
        seed=20260920,
        data=cfgs["baseline"].data,
        model=cfgs["baseline"].model,
    )
    scored["other-corpus"] = build(spec, "baseline-other-corpus", easy_cfg)

    print("scored on holdout {}, {} rows".format(fp, len(spec.rows()[1])))
    print()
    print("{:<24} {:>16} {:>16} {:>12}".format(
        "", "shared holdout", "its own holdout", "roc auc"))
    for key in ("baseline", "candidate-lr", "candidate-underfit", "candidate-inverted",
                "candidate-diverged", "other-corpus"):
        s = scored[key]
        own = s.reported["holdout_{}".format(gate.METRIC)]
        print("{:<24} {:>16} {:>16} {:>12}".format(
            s.name[:24],
            "{:.6f}".format(s.metric()) if np.isfinite(s.metric()) else "nan",
            "{:.6f}".format(own) if np.isfinite(own) else "nan",
            "{:.6f}".format(s.roc_auc) if np.isfinite(s.roc_auc) else "nan",
        ))

    # Four fields per case. The incumbent and the candidate. What should happen. Why the
    # case is here at all, printed only when the gate gets it wrong.
    cases = [
        (None, "baseline", gate.PROMOTE, "nothing holds the stage"),
        ("baseline", "candidate-lr", gate.REJECT, "identical behaviour, no case to promote"),
        ("baseline", "candidate-underfit", gate.REJECT, "worse on calibration"),
        ("candidate-underfit", "baseline", gate.PROMOTE, "genuinely better"),
        ("baseline", "candidate-inverted", gate.REJECT, "ranks backwards and prints nothing"),
        ("baseline", "candidate-diverged", gate.REFUSE, "candidate is a NaN"),
        ("candidate-diverged", "baseline", gate.REFUSE, "the incumbent is the NaN"),
        ("baseline", "baseline", gate.REFUSE, "same bytes on both sides"),
        ("baseline", "other-corpus", gate.REJECT, "an easier holdout is not a better model"),
    ]

    print()
    print("{:<22} {:<22} {:<9} {:<9} {:<20}".format(
        "incumbent", "candidate", "want", "got", "reason"))

    bad = 0
    got_all = []
    for inc_key, cand_key, want, why in cases:
        inc = scored[inc_key] if inc_key else None
        cand = scored[cand_key]
        d = gate.decide(cand, inc, fp)
        got_all.append(d.verdict)
        ok = d.verdict == want
        bad += 0 if ok else 1
        print("{:<22} {:<22} {:<9} {:<9} {:<20} {}".format(
            inc_key or "-", cand_key, want, d.verdict, d.reason, "" if ok else "BAD"))
        if not ok:
            print("    {}".format(why))

    print()
    print("controls, the naive rule on the same cases")
    control_bad = 0
    for inc_key, cand_key, want, why in cases:
        inc = scored[inc_key] if inc_key else None
        cand = scored[cand_key]
        got = naive(cand, inc)
        if got == want:
            continue
        control_bad += 1
        print("  {:<22} {:<22} want {:<9} naive says {:<9}  {}".format(
            inc_key or "-", cand_key, want, got, why))
    if control_bad == 0:
        print("  NOTHING. The naive rule got every case right, so these cases prove nothing.")
        bad += 1
    else:
        print("  naive rule wrong on {} of {} cases".format(control_bad, len(cases)))

    print()
    print("the floor, because a score means nothing without one")
    right = sum(1 for (i, c, want, _), got in zip(cases, got_all) if got == want)
    for label, rule in (
        ("promote everything", lambda i, c: gate.PROMOTE),
        ("reject everything", lambda i, c: gate.REJECT),
        ("refuse everything", lambda i, c: gate.REFUSE),
        ("the naive rule", lambda i, c: naive(c, i)),
    ):
        n = sum(1 for (i, c, want, _) in cases
                if rule(scored[i] if i else None, scored[c]) == want)
        print("  {:<20} {} of {}".format(label, n, len(cases)))
    print("  {:<20} {} of {}".format("this gate", right, len(cases)))

    # What the gate can and cannot see. Every headline about a gate rejecting a bad model
    # is worth less than the number below it, which is the smallest real difference it
    # could have called. Undertrained models give a continuum to walk.
    print()
    print("resolution, walking a candidate toward the incumbent")
    print("  {:<12} {:>12} {:>12} {:>10}".format("epochs", "gap", "relative", "verdict"))
    called, declined = [], []
    for ep in (20, 40, 45, 100, 400):
        cand = build(spec, "epochs{}".format(ep), TrainConfig(
            name="epochs{}".format(ep),
            seed=cfgs["baseline"].seed,
            data=cfgs["baseline"].data,
            model=type(cfgs["baseline"].model)(
                learning_rate=cfgs["baseline"].model.learning_rate,
                epochs=ep,
                l2=cfgs["baseline"].model.l2,
                init=cfgs["baseline"].model.init,
            ),
        ))
        d = gate.decide(cand, scored["baseline"], fp)
        gap = cand.metric() - scored["baseline"].metric()
        (called if d.reason == "worse" else declined).append(gap)
        print("  {:<12} {:>12.4e} {:>11.4f}% {:>10}".format(
            ep, gap, 100.0 * gap / scored["baseline"].metric(), d.reason))
    if called and declined:
        print("  smallest gap called {:.4e}, largest declined {:.4e}".format(
            min(called), max(declined)))

    # The number the whole design rests on. Hold the model config completely still and
    # move only the seed, so every difference below is the corpus draw and nothing else.
    # Measured here rather than quoted, so the ratio at the end cannot go stale.
    print()
    print("what the corpus alone is worth, model config held still")
    own = []
    for s in range(12):
        seed = cfgs["baseline"].seed + s
        r = train_mod.run(TrainConfig(
            name="seedsweep", seed=seed,
            data=cfgs["baseline"].data, model=cfgs["baseline"].model))
        own.append(r.metrics["holdout_log_loss"])
    span = max(own) - min(own)
    print("  {} seeds, each on its own holdout: {:.6f} to {:.6f}, span {:.6f}".format(
        len(own), min(own), max(own), span))
    real_gap = scored["candidate-underfit"].metric() - scored["baseline"].metric()
    print("  the gap this gate exists to catch: {:.6f}".format(real_gap))
    print("  so the corpus draw is {:.1%} of that gap".format(span / real_gap))
    if called:
        print("  and {:.0f}x the smallest real difference the gate can call".format(
            span / min(called)))

    print()
    print("bad arms: {}".format(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
