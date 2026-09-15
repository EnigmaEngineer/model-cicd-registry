"""Does the same config really produce the same model, and does the seed really do anything.

    python3 scripts/repro_probe.py configs/baseline.yml

Five arms. Two ask whether the pipeline is reproducible and three are controls, because a
reproducibility check on its own passes just as well against a pipeline that ignores every
input it is given.

The separate process arm is the one that counts. Two runs inside one interpreter can agree
because they share module state, an import order or a warmed cache, and the failure this
arm exists to catch is a run that agrees with itself and disagrees with tomorrow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcr import train as train_mod  # noqa: E402
from mcr.config import load  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _in_process(cfg, times):
    return [train_mod.run(cfg).content_hash for _ in range(times)]


def _subprocess_hashes(config_path, times):
    """Fresh interpreter each time, through the real CLI.

    PYTHONHASHSEED is deliberately left alone. If anything in the pipeline reached for
    Python's salted hash() the artefact would move between processes, and pinning the seed
    here would hide exactly that.
    """
    out = []
    for _ in range(times):
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "train.py"),
             config_path, "--quiet"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        )
        out.append(proc.stdout.strip())
    return out


def _line(label, ok, detail):
    print("{:<34} {:<8} {}".format(label, "OK" if ok else "FAIL", detail))
    return ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=os.path.join(ROOT, "configs", "baseline.yml"))
    parser.add_argument("--runs", type=int, default=3)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load(args.config)
    print("config {}  fingerprint {}\n".format(cfg.name, cfg.fingerprint()))

    ok = True

    same = _in_process(cfg, args.runs)
    ok &= _line(
        "same config, one process",
        len(set(same)) == 1,
        "{} runs, {} distinct, {}".format(args.runs, len(set(same)), same[0][:12]),
    )

    across = _subprocess_hashes(args.config, args.runs)
    ok &= _line(
        "same config, fresh processes",
        len(set(across)) == 1 and across[0] == same[0],
        "{} runs, {} distinct, {}".format(args.runs, len(set(across)), across[0][:12]),
    )

    # Control one. A pipeline that ignored the seed would pass both arms above.
    other_seed = train_mod.run(replace(cfg, seed=cfg.seed + 1)).content_hash
    ok &= _line(
        "control: seed + 1 differs",
        other_seed != same[0],
        other_seed[:12],
    )

    # Control two. The config has to reach the model, not just the corpus.
    bumped = replace(cfg, model=replace(cfg.model, epochs=cfg.model.epochs + 1))
    other_cfg = train_mod.run(bumped).content_hash
    ok &= _line(
        "control: epochs + 1 differs",
        other_cfg != same[0],
        other_cfg[:12],
    )

    # Control three, and the interesting one. With a zeros init the model is a
    # deterministic function of the corpus, so the seed reaches it only through the data.
    # Under a normal init the seed also moves the starting point. If these two came out
    # the same, `init` would be a config field that does nothing.
    zeros = replace(cfg, model=replace(cfg.model, init="zeros"))
    normal = replace(cfg, model=replace(cfg.model, init="normal"))
    z_hash = train_mod.run(zeros).content_hash
    n_hash = train_mod.run(normal).content_hash
    ok &= _line(
        "control: init changes the fit",
        z_hash != n_hash,
        "zeros {} normal {}".format(z_hash[:8], n_hash[:8]),
    )

    print("")
    summary = {
        "config": cfg.name,
        "fingerprint": cfg.fingerprint(),
        "runs_per_arm": args.runs,
        "artifact_hash": same[0],
        "reproducible": ok,
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
