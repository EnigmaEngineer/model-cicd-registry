"""Checks on the signature over a transition log entry.

These are cheap and they do not need a store. The thing they are protecting is narrow and
easy to get wrong in a way that still looks like it works, which is a verifier that accepts
something it should not. So most of the checks below are about refusals, and the first one
is a control that stops the rest of them passing under a verifier that refuses everything.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcr import attest  # noqa: E402

KEY = b"a-key-for-checks"
SLOT = "transition.production.0001"
BODY = '{"at_ms":1,"from":1,"kind":"promote","stage":"production","to":2}'


def check_a_sealed_entry_verifies():
    """The control. Without it every refusal below passes on a verifier that says no."""
    state, body = attest.unseal(SLOT, attest.seal(SLOT, BODY, KEY), KEY)
    assert state == attest.ATTESTED, state
    assert body == BODY, body


def check_signing_is_deterministic():
    assert attest.sign(SLOT, BODY, KEY) == attest.sign(SLOT, BODY, KEY)


def check_a_changed_body_does_not_verify():
    sealed = json.loads(attest.seal(SLOT, BODY, KEY))
    sealed[attest.BODY] = BODY.replace('"to":2', '"to":9')
    state, _ = attest.unseal(SLOT, json.dumps(sealed), KEY)
    assert state == attest.FORGED, state


def check_a_changed_slot_does_not_verify():
    """Replay protection, and the only reason the slot is signed material.

    The log is ordered by slot, so lifting a genuine entry into a different slot changes
    which version the rollback walk reaches without changing a single byte of the entry.
    """
    sealed = attest.seal(SLOT, BODY, KEY)
    state, _ = attest.unseal("transition.production.0007", sealed, KEY)
    assert state == attest.FORGED, state


def check_a_different_key_does_not_verify():
    state, _ = attest.unseal(SLOT, attest.seal(SLOT, BODY, KEY), b"another-key")
    assert state == attest.FORGED, state


def check_a_bare_body_reads_as_unattested():
    """Every log written before this module existed is this shape, and it is not an attack."""
    state, body = attest.unseal(SLOT, BODY, KEY)
    assert state == attest.UNATTESTED, state
    assert body == BODY, body


def check_a_signed_entry_is_unattested_when_there_is_no_key():
    """A claim nobody can check is not a claim that passed.

    Returning ATTESTED here would mean the verifier accepts a signature on its own word,
    which is the failure this whole module is about. Returning FORGED would be a guess.
    """
    state, _ = attest.unseal(SLOT, attest.seal(SLOT, BODY, KEY), None)
    assert state == attest.UNATTESTED, state


def check_sealing_without_a_key_writes_the_bare_body():
    """A deploy path with no key still has to be able to record what it did."""
    assert attest.seal(SLOT, BODY, None) == BODY


def check_an_envelope_with_a_missing_signature_is_forged_not_legacy():
    """Shaped like an envelope and unusable as one. Something built this.

    Reading it as legacy would hand an attacker a way to be treated as pre-signature code
    by writing a half envelope, which is cheaper than forging a signature.
    """
    state, _ = attest.unseal(SLOT, json.dumps({attest.BODY: BODY}), KEY)
    assert state == attest.FORGED, state


def check_a_json_body_that_is_not_an_envelope_reads_as_unattested():
    """The legacy body is itself JSON, so the envelope test cannot be "does this parse"."""
    state, body = attest.unseal(SLOT, BODY, KEY)
    assert state == attest.UNATTESTED
    assert json.loads(body)["to"] == 2


def check_a_value_that_is_not_json_at_all_reads_as_unattested():
    state, body = attest.unseal(SLOT, "not json", KEY)
    assert state == attest.UNATTESTED, state
    assert body == "not json"


def check_the_key_comes_back_as_none_when_the_variable_is_empty():
    """An empty variable is not a key. A zero length key would verify things.

    `export MCR_LOG_KEY=` is a plausible way to end up here and it has to mean no key
    rather than a key everybody knows.
    """
    before = os.environ.get(attest.ENV_KEY)
    try:
        os.environ[attest.ENV_KEY] = ""
        assert attest.key_from_env() is None
        os.environ[attest.ENV_KEY] = "x"
        assert attest.key_from_env() == b"x"
    finally:
        if before is None:
            os.environ.pop(attest.ENV_KEY, None)
        else:
            os.environ[attest.ENV_KEY] = before


def check_the_three_states_are_distinct():
    """They get compared by value in registry.authentic, so two of them collapsing would
    silently merge a refusal into a pass."""
    assert len({attest.ATTESTED, attest.UNATTESTED, attest.FORGED}) == 3
