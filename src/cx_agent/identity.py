"""Identity verification: a state machine, not a vibe.

    UNVERIFIED ──(account number + postcode match)──▶ VERIFYING
    VERIFYING  ──(correct one-time code)────────────▶ VERIFIED
    VERIFYING  ──(otp_max_attempts wrong codes)─────▶ FAILED

Account-changing tools check `is_verified` and refuse otherwise —
the rule lives here and in actions.py, never in a prompt. The
one-time code is a test double: in this reference build the code is
derived deterministically from the account number and a deployment
secret, so tests and evals can compute the right code exactly the
way a real SMS provider integration would deliver it. The code
itself is never stored — only its hash rides the conversation.
"""

from __future__ import annotations

import hashlib
import hmac
import os

from .models import Conversation, IdentityState


def _secret() -> str:
    return os.environ.get("CX_OTP_SECRET", "northbeam-reference-build")


def expected_code(account_number: str) -> str:
    """The code the (double) provider 'sent'. Six digits, stable per
    account within a deployment secret."""
    digest = hmac.new(_secret().encode(), account_number.encode(),
                      hashlib.sha256).hexdigest()
    return f"{int(digest[:8], 16) % 1_000_000:06d}"


def _hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def begin_verification(conv: Conversation, account_number: str,
                       postcode: str, store) -> tuple[bool, str]:
    """First factors: account number + postcode against the book.
    On a match the conversation moves to VERIFYING and the code is
    issued (its hash recorded). Returns (ok, message-key)."""
    customer = store.get_customer_by_account(account_number)
    if customer is None:
        return False, "account_not_found"
    if customer.postcode.replace(" ", "").lower() != postcode.replace(" ", "").lower():
        return False, "postcode_mismatch"
    conv.customer_id = customer.customer_id
    conv.account_number = account_number
    conv.identity_state = IdentityState.VERIFYING
    conv.identity_attempts = 0
    conv.otp_code_hash = _hash(expected_code(account_number))
    return True, "code_issued"


def submit_code(conv: Conversation, code: str, max_attempts: int) -> tuple[bool, str]:
    """Second factor. Returns (ok, message-key)."""
    if conv.identity_state != IdentityState.VERIFYING or not conv.otp_code_hash:
        return False, "not_verifying"
    conv.identity_attempts += 1
    if hmac.compare_digest(_hash(code.strip()), conv.otp_code_hash):
        conv.identity_state = IdentityState.VERIFIED
        conv.otp_code_hash = None
        return True, "verified"
    if conv.identity_attempts >= max_attempts:
        conv.identity_state = IdentityState.FAILED
        conv.otp_code_hash = None
        return False, "locked"
    return False, "wrong_code"


def is_verified(conv: Conversation) -> bool:
    return conv.identity_state == IdentityState.VERIFIED
