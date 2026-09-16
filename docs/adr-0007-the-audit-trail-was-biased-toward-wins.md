# The gate's audit trail was biased toward wins

Status: accepted, 2026-09-16.

## What was happening

`scripts/gate.py` wrote the comparison back to the candidate's run as `gate.*` tags. That
write sat inside the `if args.promote:` branch, under this comment:

    The report goes on the candidate's run whatever the verdict. A rejection that
    leaves no trace on the run is a rejection nobody can audit later.

The comment describes the inside of the branch correctly. Within `--promote`, a rejection
really did get tagged. The problem is which runs ever reach that branch.

`--promote` means move the stage when the verdict is promote. It is the flag a pipeline
passes when it intends to ship, and it is not the flag a person passes when checking
whether a model is any good yet. So the runs carrying a verdict were disproportionately the
ones that expected to win.

Measured on a clean store with three registered versions, `baseline` in production and
`candidate-underfit` as the candidate:

    python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2               0 tags
    python3 scripts/gate.py --store sqlite:///mlflow.db --candidate 2 --promote     9 tags

Both printed the same rejection. One of them is on the record.

## Why this is worth an ADR rather than a one line fix

The fix is small. The failure is not, and it is the kind that survives review because every
individual piece of it is defensible.

An audit trail exists for the decisions somebody will later dispute. Promotions are not
usually disputed, because the evidence for a promotion is the model sitting in production
doing its job. Rejections are disputed constantly, in the form "why is this not shipped
yet". A trail that records promotions well and rejections badly has its coverage exactly
inverted against the questions it will be asked.

It also degrades quietly. Nothing errors. The tags are there whenever anybody looks for
them deliberately, because looking deliberately means running the promote path.

## The decision

Writing the report is what a comparison does, not a side effect of asking for a promotion.

    --promote      moves the stage on a winning verdict, and nothing else
    --no-report    compares without writing to the store

Default on. The argument against a default write is that a read only query should not
mutate, which is a real principle and the reason `--no-report` exists rather than being
left out. It loses to the fact that a gate decision is not a query. It is an event, and the
store is where events about a model go.

## What this does not fix

The tags hold the latest verdict only. A candidate gated three times against three
incumbents keeps the third. Tags were chosen over params when the gate was built, because a run is written
before it is ever gated and a param refuses a rewrite, so the choice was between the latest
verdict and no verdict at all.

Keeping every verdict means a child run per comparison, or a table. Both are a bigger
change than this, and neither is worth making until somebody wants the history. Noted in
the README limitations rather than built.

`--no-report` is untested against a store that refuses writes, because there is no such
store here. It is checked by parsing, not by being exercised against a read only backend.
