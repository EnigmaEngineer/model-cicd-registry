"""What is deployed right now, including a canary and the share of traffic it holds.

The canary used to be a command line flag. `scripts/canary.py --canary 4 --fraction 0.25`
routed a quarter of a replay at version 4, printed a verdict and exited. Nothing in the
store knew any of it had happened, so a rollback verdict had nothing to act on and the exit
code was the whole action. That gap is written up at the end of docs/adr-0005.

A flag also made the wrong action look right. The first version of that script grew a
`--rollback` that called `registry.rollback`, which moves production. It ran end to end and
reported success while moving the one component the verdict said nothing about. The verdict
is about the canary. What it asks for is that the canary stops taking traffic.

So a deployment is three facts the registry holds:

    production        an alias holding the version that serves the control share
    canary            an alias holding the version on trial
    canary.fraction   a registered model tag holding the share the canary takes

and three operations over them. `open_canary` puts a version on trial. `abort_canary` takes
it off and leaves production alone, which is what a rollback verdict means. `land_canary`
makes the canary production and ends the trial. Each one goes through the registry's
transition log, so a canary that was opened and aborted is a thing the store can still
describe afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlflow

from . import registry

CANARY = "canary"
PRODUCTION = "production"

# The share lives on the registered model rather than on the version, because it is a
# property of the deployment and not of the model. The same version can be canaried twice
# at two fractions and neither reading belongs to the artefact.
FRACTION_TAG = "canary.fraction"

# Below this the canary arm is too small for the split comparison to say anything, and
# above it the control arm is. Both ends are refusals rather than clamps, because a clamp
# would answer a question the caller did not ask. The numbers are the range the sizing in
# mcr/canary.py was measured over and nothing here claims they are optimal.
MIN_FRACTION = 0.01
MAX_FRACTION = 0.5


class DeployError(RuntimeError):
    """Raised when a deployment move does not make sense against the current state."""


@dataclass(frozen=True)
class Deployment:
    production: Optional[int]
    canary: Optional[int]
    fraction: Optional[float]

    def is_canarying(self) -> bool:
        return self.canary is not None

    def describe(self) -> str:
        if self.production is None:
            return "nothing in production"
        if self.canary is None:
            return "production on version {}, no canary".format(self.production)
        return "production on version {}, canary on version {} at {:.4f}".format(
            self.production, self.canary, self.fraction
        )


def _read_fraction(cli: mlflow.MlflowClient, name: str) -> Optional[float]:
    try:
        tags = dict(cli.get_registered_model(name).tags)
    except mlflow.exceptions.MlflowException:
        return None
    raw = tags.get(FRACTION_TAG)
    if raw is None:
        return None
    # MLflow gives every tag back as a string. The float goes through the same coercion
    # discipline as the config fingerprint, for the reason in docs/adr-0002.
    try:
        return float(raw)
    except ValueError:
        raise DeployError(
            "{} on {} holds '{}', which is not a number".format(FRACTION_TAG, name, raw)
        )


def deployment(cli: mlflow.MlflowClient, name: str) -> Deployment:
    """Read the whole deployment in one go.

    The alias and the fraction tag are two reads with nothing tying them together, so they
    can disagree. `check_consistency` is where that is turned into a refusal. This function
    reports what is there and does not judge it, because a repair tool needs to see a state
    the operations refuse to produce.
    """
    return Deployment(
        production=registry.current(cli, name, PRODUCTION),
        canary=registry.current(cli, name, CANARY),
        fraction=_read_fraction(cli, name),
    )


def check_consistency(cli: mlflow.MlflowClient, name: str) -> Optional[str]:
    """Return a sentence naming the problem, or None when the state hangs together.

    A sentence rather than a bool. A caller that has to print something useful otherwise
    reconstructs the reason from the inputs, which is two places for one rule to live.
    """
    state = deployment(cli, name)

    if state.canary is not None and state.fraction is None:
        return "canary is on version {} and no fraction is recorded".format(state.canary)

    if state.canary is None and state.fraction is not None:
        return "no canary alias and {} still reads {}".format(FRACTION_TAG, state.fraction)

    if state.canary is not None and state.canary == state.production:
        return "version {} is both production and canary".format(state.canary)

    if state.canary is not None and state.production is None:
        return "canary is on version {} with nothing in production to compare it against".format(
            state.canary
        )

    return None


def open_canary(
    cli: mlflow.MlflowClient, name: str, ref: str, fraction: float
) -> Deployment:
    """Put a version on trial at a share of traffic.

    The fraction is written before the alias, matching `registry.promote`. A crash between
    them leaves a fraction with no canary, which `check_consistency` names and
    `abort_canary` clears. The other order leaves a canary taking an unrecorded share,
    which is the state nothing can describe.
    """
    if not MIN_FRACTION <= fraction <= MAX_FRACTION:
        raise DeployError(
            "fraction {} is outside [{}, {}]".format(fraction, MIN_FRACTION, MAX_FRACTION)
        )

    state = deployment(cli, name)
    if state.production is None:
        raise DeployError(
            "{} has nothing in production, so there is no control arm".format(name)
        )
    if state.canary is not None:
        raise DeployError(
            "{} is already canarying version {} at {}".format(
                name, state.canary, state.fraction
            )
        )

    version = registry.resolve(cli, name, ref)
    if version == state.production:
        raise DeployError(
            "version {} is already in production, so canarying it compares a model "
            "against itself".format(version)
        )

    cli.set_registered_model_tag(name, FRACTION_TAG, repr(float(fraction)))
    registry.promote(cli, name, str(version), CANARY)
    return deployment(cli, name)


def abort_canary(cli: mlflow.MlflowClient, name: str) -> Deployment:
    """Take the canary off traffic. Production does not move.

    This is what a rollback verdict from `scripts/canary.py` means, and it is the operation
    the deleted `--rollback` flag should have been calling. The distinction is the whole
    point: `registry.rollback` moves production to an earlier version, which is a statement
    about production. Aborting a canary is a statement about the canary.
    """
    state = deployment(cli, name)
    if state.canary is None:
        # The fraction tag can outlive the alias when a crash lands between the two writes
        # of open_canary. Clearing it here rather than refusing means abort is the repair
        # for that state as well as the normal ending, and an operator reaching for abort
        # after a failed open is reaching for the right thing.
        if state.fraction is not None:
            cli.delete_registered_model_tag(name, FRACTION_TAG)
            return deployment(cli, name)
        raise DeployError("{} is not canarying anything".format(name))

    registry.retire(cli, name, CANARY)
    cli.delete_registered_model_tag(name, FRACTION_TAG)
    return deployment(cli, name)


def land_canary(cli: mlflow.MlflowClient, name: str) -> Deployment:
    """Make the canary the production model and end the trial.

    Production moves first. The canary alias is removed second, so a crash in between
    leaves both aliases on the same version, which `check_consistency` names as a state and
    `abort_canary` clears without touching production. The other order would leave a
    window where the version is on trial at a share nothing is serving.
    """
    state = deployment(cli, name)
    if state.canary is None:
        raise DeployError("{} is not canarying anything".format(name))

    registry.promote(cli, name, str(state.canary), PRODUCTION)
    registry.retire(cli, name, CANARY)
    cli.delete_registered_model_tag(name, FRACTION_TAG)
    return deployment(cli, name)
