"""Measure what splitting the traffic costs, and check the router does what it claims.

    python3 scripts/canary_probe.py
    python3 scripts/canary_probe.py --trials 400

No MLflow. Everything here trains from configs on disk and compares in memory, so this
runs on the core requirements file.

Three sections.

ONE. The router. Bias against its target. Stickiness across restarts. Salt sensitivity.
Then what a sticky slice does to composition that a per request coin flip does not.

TWO. The cost of the split. The same two models, the same rows, compared paired and
compared as two arms. Then the verdict each way, then how often the split gets the
direction wrong over many draws of the split itself.

THREE. The floor. What the trivial answers score on the same cases, so the section two
result has something to be read against.

EVERY NUMBER IS MEASURED ON A REPLAY OF A GENERATED HOLDOUT. There is no serving path in
this project and there are no users. The arithmetic of splitting a sample is real. The
loss values are a fact about a generator.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcr import canary as canary_mod  # noqa: E402
from mcr import config as config_mod  # noqa: E402
from mcr import gate  # noqa: E402
from mcr import train as train_mod  # noqa: E402

BASELINE = "configs/baseline.yml"

# Six candidates, chosen to span how far apart two models can be rather than to span the
# gate's resolution alone. One epoch is a model nobody would canary. Twenty is one the day
# 4 gate rejects with room to spare. Forty and forty five straddle the smallest gap the
# gate can call, and a hundred is one it cannot separate from the incumbent at all.
#
# Four hundred is deliberately absent. It reproduces the incumbent exactly, so the paired
# interval has zero width and the width ratio below divides by it.
CANDIDATE_EPOCHS = (1, 5, 20, 40, 45, 100)

FRACTIONS = (0.50, 0.10, 0.05)


def _fail(bad, msg):
    print("  BAD ARM: {}".format(msg))
    bad.append(msg)


def section_router(bad):
    print("=" * 78)
    print("ONE. THE ROUTER")
    print("=" * 78)

    keys = canary_mod.replay_keys(20000)

    print("")
    print("bias against the target, over {} keys".format(len(keys)))
    print("  {:>8} {:>10} {:>10} {:>12}".format("target", "achieved", "n canary", "error"))
    for f in (0.01, 0.05, 0.10, 0.25, 0.50):
        r = canary_mod.Router(fraction=f)
        mask = r.assign(keys)
        got = float(mask.mean())
        print("  {:>8.2f} {:>10.4f} {:>10d} {:>+12.4f}".format(
            f, got, int(mask.sum()), got - f))
        # A hash router's share is a binomial draw around the target, so three standard
        # errors is the bound worth asserting rather than exact equality.
        se = math.sqrt(f * (1.0 - f) / len(keys))
        if abs(got - f) > 3.0 * se:
            _fail(bad, "router at {} achieved {} which is beyond three standard errors".format(f, got))

    print("")
    r = canary_mod.Router(fraction=0.05)
    once = [r.arm(k) for k in keys[:500]]
    again = [canary_mod.Router(fraction=0.05).arm(k) for k in keys[:500]]
    sticky = once == again
    print("stickiness across a fresh Router object: {}".format("same arm every key" if sticky else "MOVED"))
    if not sticky:
        _fail(bad, "routing is not sticky across objects")

    salted = [canary_mod.Router(fraction=0.05, salt="other").arm(k) for k in keys[:500]]
    moved = sum(1 for a, b in zip(once, salted) if a != b)
    print("changing the salt moves {} of {} keys".format(moved, len(once)))
    if moved == 0:
        _fail(bad, "the salt does not change the assignment, so two canaries cannot be separated")

    print("")
    print("what a sticky slice does to composition that a coin flip does not.")
    print("the same key population, routed once and held, against reshuffled each round.")
    cfg = config_mod.load(BASELINE)
    spec = gate.spec_from_config(cfg)
    _, y = spec.rows()
    ks = canary_mod.replay_keys(len(y))
    router = canary_mod.Router(fraction=0.05)
    can, con, gap = canary_mod.slice_imbalance(router, ks, y)
    print("  sticky slice   positive rate {:.4f} against {:.4f}, gap {:+.4f}".format(can, con, gap))

    rng = np.random.default_rng(7)
    gaps = []
    k = int(round(len(y) * 0.05))
    for _ in range(200):
        idx = rng.permutation(len(y))
        gaps.append(float(y[idx[:k]].mean()) - float(y[idx[k:]].mean()))
    print("  coin flip      gap over 200 redraws: mean {:+.4f}, spread {:.4f}".format(
        float(np.mean(gaps)), float(np.std(gaps, ddof=1))))
    print("  The sticky gap is one draw and it stays. The coin flip's averages away, and")
    print("  that is the trade for serving a user the same model twice in a row.")


def _models():
    """Train the incumbent and each candidate once, and score them on one holdout."""
    cfg = config_mod.load(BASELINE)
    spec = gate.spec_from_config(cfg)
    x, y = spec.rows()

    base_run = train_mod.run(cfg)
    base_model = gate.model_from_artifact(base_run.artifact.payload)
    base_losses = gate._row_losses(y, base_model.predict_proba(x))

    out = []
    for ep in CANDIDATE_EPOCHS:
        cand_cfg = replace(cfg, name="canary-e{}".format(ep),
                           model=replace(cfg.model, epochs=ep))
        run = train_mod.run(cand_cfg)
        m = gate.model_from_artifact(run.artifact.payload)
        out.append((ep, gate._row_losses(y, m.predict_proba(x)), run.content_hash))
    return spec, x, y, base_losses, base_run.content_hash, out


def _split_verdict(a, b):
    mean, lo, hi = canary_mod.split_interval(a, b)
    return canary_mod._verdict_from(mean, lo, hi)[0], hi - lo


def section_cost(bad, trials):
    print("")
    print("=" * 78)
    print("TWO. WHAT THE SPLIT COSTS")
    print("=" * 78)

    spec, x, y, base_losses, base_hash, cands = _models()
    print("")
    print("{} holdout rows. Pairing exploits the correlation between the two arms' row".format(len(y)))
    print("losses and splitting discards it, so the cost of splitting is a function of how")
    print("close the two models are. That is the column to read.")
    print("")
    print("  {:>8} {:>14} {:>10} {:>13} {:>13} {:>9}".format(
        "epochs", "true diff", "corr", "paired width", "split 5% width", "ratio"))
    widths = {}
    for ep, losses, _ in cands:
        d = losses - base_losses
        lo, hi = gate.paired_interval(d)
        paired_w = hi - lo
        corr = float(np.corrcoef(losses, base_losses)[0, 1])
        mask = canary_mod.Router(fraction=0.05).assign(canary_mod.replay_keys(len(y)))
        _, l5, h5 = canary_mod.split_interval(losses[mask], base_losses[~mask])
        mask50 = canary_mod.Router(fraction=0.50).assign(canary_mod.replay_keys(len(y)))
        _, l50, h50 = canary_mod.split_interval(losses[mask50], base_losses[~mask50])
        if paired_w <= 0.0:
            _fail(bad, "epochs={} gave a paired width of {}, which cannot be divided by".format(
                ep, paired_w))
            continue
        # The ratio is computed from the two figures beside it rather than written down.
        ratio = (h5 - l5) / paired_w
        widths[ep] = {"paired": paired_w, "split5": h5 - l5, "split50": h50 - l50,
                      "ratio": ratio, "diff": float(d.mean())}
        print("  {:>8} {:>+14.4e} {:>10.6f} {:>13.4e} {:>13.4e} {:>9.1f}".format(
            ep, float(d.mean()), corr, paired_w, h5 - l5, ratio))
        if h5 - l5 <= paired_w:
            _fail(bad, "at epochs={} the split came out no wider than the paired interval".format(ep))

    lo_r, hi_r = widths[1]["ratio"], widths[100]["ratio"]
    print("")
    print("  The ratio runs from {:.1f} to {:.0f} across this table. Splitting is nearly free".format(
        lo_r, hi_r))
    print("  against a model that is obviously bad and it costs everything against one that")
    print("  is close. A canary only ever sees the second kind, because the promotion")
    print("  gate has")
    print("  already stopped the first kind before any traffic moves.")
    if not lo_r < hi_r:
        _fail(bad, "the ratio did not grow with closeness, so the claim above is unfounded")

    sd = float(np.std(base_losses, ddof=1))
    target = 0.5 * widths[40]["paired"]
    print("")
    print("to reach the paired half width of {:.4e} the split needs, in total requests:".format(target))
    for f in FRACTIONS:
        n = canary_mod.required_rows(sd, target, f)
        print("  at {:>5.1%} to the canary    {:>18,.0f}".format(f, n))
    print("  against {:,} rows for the paired comparison.".format(len(y)))

    print("")
    print("how often the split gets the direction wrong, over {} draws of the split".format(trials))
    print("  {:>8} {:>14} {:>16} {:>26}".format(
        "epochs", "true diff", "paired verdict", "split at 5%"))
    for ep, losses, _ in cands:
        d = losses - base_losses
        lo, hi = gate.paired_interval(d)
        pv = canary_mod._verdict_from(float(d.mean()), lo, hi)[0]
        rng = np.random.default_rng(1234)
        counts = {canary_mod.PROMOTE: 0, canary_mod.ROLLBACK: 0, canary_mod.HOLD: 0}
        k = int(round(len(y) * 0.05))
        for _ in range(trials):
            idx = rng.permutation(len(y))
            v, _w = _split_verdict(losses[idx[:k]], base_losses[idx[k:]])
            counts[v] += 1
        print("  {:>8} {:>+14.4e} {:>16} {:>10} promote {:>4} rollback {:>4} hold".format(
            ep, float(d.mean()), pv,
            counts[canary_mod.PROMOTE], counts[canary_mod.ROLLBACK], counts[canary_mod.HOLD]))

        # epochs=20 is the case that matters. It is worse by 8e-03, the gate rejects it
        # every time because the gate is deterministic on fixed rows, and the split
        # promotes it on some draws. An arm that never promoted it would mean the probe
        # had lost the effect it was built to show.
        if ep == 20 and counts[canary_mod.PROMOTE] == 0:
            _fail(bad, "the split never promoted the clearly worse model, so this probe shows nothing")
        if ep == 20 and pv != canary_mod.ROLLBACK:
            _fail(bad, "the paired comparison failed to reject epochs=20, so the case is not what it claims")

    print("")
    print("  a two sided 95% interval is wrong about 5% of the time by construction, so")
    print("  the epochs=100 row is the control: it is two models that really are alike and")
    print("  its promote count is the nominal rate. The epochs=20 row is the finding,")
    print("  because there the split is wrong about a difference the gate calls every time.")


def section_floor(bad):
    print("")
    print("=" * 78)
    print("THREE. THE FLOOR")
    print("=" * 78)
    print("")

    spec, x, y, base_losses, base_hash, cands = _models()

    # The answer key. A model that is worse on the paired comparison should be rolled
    # back, one that cannot be separated should be held. Written from the paired
    # comparison because that is the one with the resolution to have an opinion.
    truth = []
    for ep, losses, _ in cands:
        d = losses - base_losses
        lo, hi = gate.paired_interval(d)
        truth.append((ep, canary_mod._verdict_from(float(d.mean()), lo, hi)[0]))

    keys = canary_mod.replay_keys(len(y))

    def score_strategy(fn):
        return sum(1 for (ep, want), (_, losses, _h) in zip(truth, cands) if fn(ep, losses) == want)

    always = {
        "promote everything": lambda ep, l: canary_mod.PROMOTE,
        "rollback everything": lambda ep, l: canary_mod.ROLLBACK,
        "hold everything": lambda ep, l: canary_mod.HOLD,
    }

    def split_at(f):
        def go(ep, losses):
            mask = canary_mod.Router(fraction=f).assign(keys)
            return _split_verdict(losses[mask], base_losses[~mask])[0]
        return go

    def shadow(ep, losses):
        m, lo, hi = canary_mod.shadow_interval(losses, base_losses)
        return canary_mod._verdict_from(m, lo, hi)[0]

    rows = list(always.items()) + [
        ("split at 5%", split_at(0.05)),
        ("split at 50%", split_at(0.50)),
        ("shadow, both models on every request", shadow),
    ]
    scores = {}
    for label, fn in rows:
        scores[label] = score_strategy(fn)
        print("  {:<40} {} of {}".format(label, scores[label], len(truth)))

    print("")
    print("  The answer key is the paired comparison, so the shadow row scoring full marks")
    print("  is a tautology. It is printed to show the key is self consistent, not as a")
    print("  result. The rows worth reading are the two split ones against the three")
    print("  trivial ones, and on this board the split at 5% scores {} against {} for".format(
        scores["split at 5%"], scores["rollback everything"]))
    print("  rolling everything back without looking at anything.")
    print("  {} cases is a small board and the README says so.".format(len(truth)))
    best_trivial = max(scores[k] for k in always)
    if scores["split at 5%"] > best_trivial:
        print("  On this run the split beat every trivial answer, which it does not always do.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="measure the cost of splitting canary traffic")
    p.add_argument("--trials", type=int, default=400)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    bad = []
    section_router(bad)
    section_cost(bad, args.trials)
    section_floor(bad)

    print("")
    print("bad arms: {}".format(len(bad)))
    for b in bad:
        print("  {}".format(b))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
