"""Who wrote a transition log entry, which is a different question from whether it happened.

`mcr/registry.py` keeps its transition log as tags on a registered model. The store has no
conditional write and no per tag owner, so anything holding store credentials can add a tag
or overwrite one. `registry.settle` reads such a log and drops entries the store cannot
confirm. That recovers a crash. It says nothing about who wrote what it kept.

The two questions need different inputs and that is the reason they are different functions.
Recovery needs the log and the live alias, both of which are already in the store.
Authorisation needs a secret the store does not hold. A function taking a key cannot be the
function that answers a question about the store alone.

What this buys and what it does not:

An entry carries a MAC over its own body and the slot it sits in. A writer without the key
cannot produce a body the verifier accepts, and cannot lift a genuine entry into a different
slot, because the slot is signed material. A writer holding the key can write anything it
likes and this will accept all of it. So this authenticates the holder of a key and not a
principal, and the honest claim is narrow.

An entry with no MAC is not a forgery. Every log written before this module existed is
unattested, and treating that as an attack would refuse every history already in a store.
Treating it as authentic is the laundering this module exists to stop. So it is a third
state and the caller decides what it is worth.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Optional, Tuple

ENV_KEY = "MCR_LOG_KEY"

# The default for a key argument, meaning "go and look in the environment". It cannot be
# None, because None is a real value here and means there is no key at all. A caller that
# wants verification off says None and means it.
USE_ENV = object()

# The three readings of one tag. UNATTESTED is not a weaker FORGED. It means the question
# was not asked, either because the entry predates the MAC or because no key was supplied.
ATTESTED = "attested"
UNATTESTED = "unattested"
FORGED = "forged"

BODY = "body"
MAC = "mac"


def resolve_key(key) -> Optional[bytes]:
    """Turn a key argument into a key, looking at the environment only when asked to.

    This exists so that nothing below the edge of the module reads the environment. A read
    path whose behaviour depends on a variable set three call frames up is the kind of
    implicit state that is impossible to test without mutating the process.
    """
    if key is USE_ENV:
        return key_from_env()
    return key


def key_from_env() -> Optional[bytes]:
    """The verification key, or None when there is not one.

    None is returned rather than a zero key so that a missing key cannot be mistaken for a
    key nobody guessed. A caller holding None has no way to verify and has to say so.
    """
    raw = os.environ.get(ENV_KEY)
    if not raw:
        return None
    return raw.encode("utf-8")


def _material(slot: str, body: str) -> bytes:
    """The bytes a MAC is taken over.

    The slot is in here with the body. Without it a genuine entry can be copied into
    another sequence key and still verify, and the log is ordered by that key, so moving
    one entry backwards changes which version a rollback walks to. Binding the slot costs
    nothing and removes that move.

    The separator cannot appear in a slot, which is why it is a newline. Slots are built
    from a fixed prefix, a stage name and a zero padded integer.
    """
    return ("{}\n{}".format(slot, body)).encode("utf-8")


def sign(slot: str, body: str, key: bytes) -> str:
    return hmac.new(key, _material(slot, body), hashlib.sha256).hexdigest()


def seal(slot: str, body: str, key: Optional[bytes]) -> str:
    """The tag value to write. Falls back to the bare body when there is no key.

    Writing an unattested entry is allowed on purpose. A deploy path without a key still
    has to be able to record what it did, and a registry that refuses to log without one
    would make the key a hard dependency of promotion rather than of verification.
    """
    if key is None:
        return body
    return json.dumps({BODY: body, MAC: sign(slot, body, key)}, sort_keys=True,
                      separators=(",", ":"))


def unseal(slot: str, value: str, key: Optional[bytes]) -> Tuple[str, str]:
    """Read one tag value. Returns the state and the body.

    The body comes back whatever the state is, because the caller needs to be able to say
    which entry it refused. Returning None for a forged entry would make the refusal
    harder to read than the attack.
    """
    body, mac = _split(value)

    if mac is None:
        return UNATTESTED, body
    if key is None:
        # The entry claims to be signed and there is nothing to check it against. Calling
        # that forged would be a guess, and calling it attested would accept the claim on
        # its own word.
        return UNATTESTED, body
    if hmac.compare_digest(mac, sign(slot, body, key)):
        return ATTESTED, body
    return FORGED, body


def _split(value: str) -> Tuple[str, Optional[str]]:
    """Separate an envelope from a bare body, tolerating either shape.

    The body is stored as a string rather than as nested JSON so that the bytes signed are
    the bytes verified. Keeping it as an object would mean re-serialising it to check a
    MAC, and two serialisers that disagree on key order or spacing would read as a forgery.
    """
    try:
        outer = json.loads(value)
    except ValueError:
        return value, None

    if not isinstance(outer, dict) or BODY not in outer:
        return value, None

    body = outer.get(BODY)
    mac = outer.get(MAC)
    if not isinstance(body, str) or not isinstance(mac, str):
        # Shaped like an envelope and not usable as one. This is the one case where a
        # malformed value is treated as an attack rather than as legacy, because a bare
        # body cannot reach here and something built this.
        return value, ""
    return body, mac
