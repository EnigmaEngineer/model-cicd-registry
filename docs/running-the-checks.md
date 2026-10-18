# Running the checks

Moved out of the README so the front page stays short. Back to the [README](../README.md).

## Run it

```
pip install -r requirements.txt
python3 scripts/train.py configs/baseline.yml --out artifacts/baseline.json
```

```
name         baseline
config       70a46feb1141
artifact     8b83b8d77ac9e226e996859a28e5725e80d6ba657e6a5eda7b513f46dda003b3
seed         20260911

holdout_accuracy         0.816000
holdout_log_loss         0.449239
holdout_n_rows           5000.000000
holdout_positive_rate    0.185200
holdout_roc_auc          0.675518

wrote artifacts/baseline.json
```

Then check the claim the rest of it rests on.

```
python3 scripts/repro_probe.py configs/baseline.yml
python3 tests/run_all.py
```

Tracking is a separate install, because training a model does not need it.

```
pip install -r requirements-tracking.txt
python3 scripts/train.py configs/baseline.yml --track sqlite:///mlflow.db
python3 scripts/track_probe.py
python3 tests/run_with_mlflow.py
```

The registry sits on the same store.

```
python3 scripts/train.py configs/baseline.yml --track sqlite:///mlflow.db --register mcr-fraud
python3 scripts/registry.py --store sqlite:///mlflow.db list
python3 scripts/registry.py --store sqlite:///mlflow.db promote 1 production
python3 scripts/registry.py --store sqlite:///mlflow.db promote 1 production --require-gate
python3 scripts/registry.py --store sqlite:///mlflow.db rollback production
python3 scripts/registry_probe.py
```

`promote` moves an alias and asks nothing about how the version got there. `--require-gate`
refuses unless the gate cleared that version, for that stage, **against whoever holds the
stage right now**. `deploy.py open` takes the same flag, because the canary is the other way
in. Both are opt in and the section below says what that does and does not buy.

And the gate sits in front of it.

```
python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2
python3 scripts/registry.py --store sqlite:///mlflow.db report 2
python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2 --promote
python3 scripts/gate_probe.py
```

Every comparison writes its verdict onto the candidate's run, whether or not anybody asked
to promote it. `report` reads it back. `--no-report` is the way to compare without writing.

And the canary sits behind it. Opening one is a change to the registry, not a flag on a
command, so the share survives the process that set it.

```
python3 scripts/deploy.py open   --store sqlite:///mlflow.db --canary 4 --fraction 0.05
python3 scripts/deploy.py open   --store sqlite:///mlflow.db --canary 4 --fraction 0.05 --require-gate
python3 scripts/deploy.py status --store sqlite:///mlflow.db
python3 scripts/canary.py --store sqlite:///mlflow.db --canary canary
python3 scripts/deploy.py abort  --store sqlite:///mlflow.db
python3 scripts/canary_probe.py
```

`canary.py` reads the share off the registry. Passing a `--fraction` that disagrees with
what was deployed is refused rather than believed.

The failure drills are the evidence for the rollback path and they run in seconds.

```
python3 scripts/drill.py
```

```
ok     crash then retry same version, answer  3 (wanted 3)
ok     raw log agrees here, so no damage      3 (wanted 3)
ok     and settle still dropped the phantom   5 raw, 4 kept, 1 dropped (wanted 5 raw, 4 kept, 1 dropped)
ok     crash then ship a different version    raw log: 4        settled: 3        control: 3
ok     crash after the alias moved            damaged: refused                        control: 2
ok     alias moved outside promote            damaged: refused                        control: 2
ok     two rollbacks reach the oldest         damaged: 1                              control: 3
ok     forged log entry refused               damaged: refused                        control: 2
ok     recovery first answers instead of refusing 1 (wanted 1)
ok     because settle dropped the evidence first 0 rejected after settle (wanted 0 rejected after settle)
ok     unsigned entry in a signed log refused damaged: refused                        control: 2
ok     and the wrong order still answers      1 (wanted 1)
ok     replayed entry refused                 damaged: refused                        control: 2
ok     an unsigned log answers normally       2 (wanted 2)
ok     and it really was unsigned             unattested (wanted unattested)
ok     a log nobody signed answers anyway     2 (wanted 2)
ok     and only require_attested refuses it   3 of 3 rejected (wanted 3 of 3 rejected)
ok     canary open crashed, then repaired     consistent (wanted consistent)
ok     and it was really broken first         broken (wanted broken)
ok     canary land crashed, then repaired     consistent (wanted consistent)
ok     and it was really broken first         broken (wanted broken)
ok     that repair left production alone      3 (wanted 3)
ok     aborting a canary leaves production    damaged: 1                              control: 2
23 drills, 0 bad
```

`gate_probe.py` and `canary_probe.py` need no MLflow at all.
