"""Train one model from one config.

    python3 scripts/train.py configs/baseline.yml
    python3 scripts/train.py configs/baseline.yml --out artifacts/baseline.json

Prints the config fingerprint, the artefact hash and the holdout metrics. The two hashes
are what the registry keys on, so they are the first thing on screen.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcr import train as train_mod  # noqa: E402
from mcr.config import load  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="train one model from one config")
    parser.add_argument("config")
    parser.add_argument("--out", default=None, help="where to write the artefact")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cfg = load(args.config)
    result = train_mod.run(cfg)

    if args.out:
        result.artifact.write(args.out)

    if not args.quiet:
        print("name         {}".format(cfg.name))
        print("config       {}".format(cfg.fingerprint()))
        print("artifact     {}".format(result.content_hash))
        print("seed         {}".format(cfg.seed))
        print("")
        for key in sorted(result.metrics):
            if key.startswith("holdout_"):
                print("{:<24} {:.6f}".format(key, result.metrics[key]))
        if args.out:
            print("\nwrote {}".format(args.out))
    else:
        print(result.content_hash)

    return 0


if __name__ == "__main__":
    sys.exit(main())
