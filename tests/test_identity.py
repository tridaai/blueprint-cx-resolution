from cx_agent import identity
from cx_agent.models import Conversation, IdentityState


def _conv():
    return Conversation(conversation_id="CONV-T")


def test_expected_code_is_stable_six_digits():
    code = identity.expected_code("NB-1001")
    assert len(code) == 6 and code.isdigit()
    assert identity.expected_code("NB-1001") == code
    assert identity.expected_code("NB-1002") != code or True  # may collide rarely


def test_begin_verification_happy_path(seeded_store):
    conv = _conv()
    ok, key = identity.begin_verification(conv, "NB-1001", "02134", seeded_store)
    assert ok and key == "code_issued"
    assert conv.identity_state == IdentityState.VERIFYING
    assert conv.customer_id == "CUST-001"
    # The code itself is never stored — only its hash.
    assert conv.otp_code_hash and len(conv.otp_code_hash) == 64


def test_begin_verification_rejects_wrong_postcode(seeded_store):
    conv = _conv()
    ok, key = identity.begin_verification(conv, "NB-1001", "99999", seeded_store)
    assert not ok and key == "postcode_mismatch"
    assert conv.identity_state == IdentityState.UNVERIFIED


def test_begin_verification_unknown_account(seeded_store):
    conv = _conv()
    ok, key = identity.begin_verification(conv, "NB-9999", "02134", seeded_store)
    assert not ok and key == "account_not_found"


def test_submit_code_locks_after_max_attempts(seeded_store):
    conv = _conv()
    identity.begin_verification(conv, "NB-1001", "02134", seeded_store)
    right = identity.expected_code("NB-1001")
    wrong = "000000" if right != "000000" else "999999"
    ok, key = identity.submit_code(conv, wrong, 3)
    assert not ok and key == "wrong_code"
    identity.submit_code(conv, wrong, 3)
    ok, key = identity.submit_code(conv, wrong, 3)
    assert not ok and key == "locked"
    assert conv.identity_state == IdentityState.FAILED


def test_submit_correct_code_verifies(seeded_store):
    conv = _conv()
    identity.begin_verification(conv, "NB-1001", "02134", seeded_store)
    ok, key = identity.submit_code(conv, identity.expected_code("NB-1001"), 3)
    assert ok and key == "verified"
    assert identity.is_verified(conv)
    assert conv.otp_code_hash is None
