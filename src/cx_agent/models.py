"""The domain model. Everything the agent may rely on is a typed
record here — customers, accounts, bills, plans, the policy corpus,
troubleshooting procedures, conversations and the structured actions
specialists propose. Money is Decimal; at rest it is a string inside
the JSON payloads, re-validated into these models on the way out.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Carrier data (the Northbeam book)
# ---------------------------------------------------------------------------


class Customer(BaseModel):
    customer_id: str
    full_name: str
    account_number: str  # the number the customer knows, e.g. NB-48151623
    postcode: str  # second verification factor, with the one-time code
    email: str
    tenure_months: int
    vulnerable: bool = False  # flagged accounts escalate early, by rule


class Plan(BaseModel):
    plan_id: str
    name: str
    monthly_price: Decimal
    data_gb: int | None = None  # None = unlimited
    notes: str = ""


class Offer(BaseModel):
    offer_id: str
    name: str
    kind: str  # plan_upgrade | roaming_pass | loyalty_discount
    monthly_price: Decimal = Decimal("0")
    one_off_price: Decimal = Decimal("0")
    target_plan_id: str | None = None
    duration_days: int | None = None
    eligible_plan_ids: list[str] = Field(default_factory=list)
    blurb: str = ""


class BillCharge(BaseModel):
    charge_id: str
    kind: str  # plan | device | roaming | one_off | tax | fee
    description: str
    amount: Decimal
    # Evidence markers the billing specialist reasons over in code:
    duplicate_of: str | None = None  # charge_id this repeats in error
    after_cancellation: bool = False
    expected_amount: Decimal | None = None  # set when the rate was wrong


class Bill(BaseModel):
    bill_id: str
    account_number: str
    cycle_start: str  # ISO date
    cycle_end: str
    issued: str
    due: str
    charges: list[BillCharge] = Field(default_factory=list)

    @property
    def total(self) -> Decimal:
        return sum((c.amount for c in self.charges), Decimal("0"))


class Account(BaseModel):
    account_number: str
    customer_id: str
    plan_id: str
    status: str = "active"  # active | suspended | cancelled
    balance_due: Decimal = Decimal("0")
    payment_method_last4: str | None = None
    roaming_country: str | None = None  # set while the customer travels
    roaming_gb_used: Decimal = Decimal("0")
    cycle_day: int = 11  # days elapsed in the current cycle (of 30)
    cycle_days: int = 30
    open_case: str | None = None


class PolicyDocument(BaseModel):
    doc_id: str
    title: str
    section: str
    text: str


class ProcedureStep(BaseModel):
    step_id: str
    instruction: str
    check: str  # the condition the tool tests, e.g. "signal_restored"
    on_fail: str  # next | escalate | callback


class Procedure(BaseModel):
    procedure_id: str
    title: str
    symptom: str  # router key
    steps: list[ProcedureStep] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Conversation machinery
# ---------------------------------------------------------------------------


class IdentityState(str, Enum):
    UNVERIFIED = "unverified"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"


class Intent(str, Enum):
    BILLING = "billing"
    ROAMING = "roaming"
    OFFERS = "offers"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    CARE = "care"
    UNKNOWN = "unknown"


class Message(BaseModel):
    role: str  # customer | agent | system
    text: str
    turn: int = 0


class ActionKind(str, Enum):
    APPLY_CREDIT = "apply_credit"
    ACTIVATE_OFFER = "activate_offer"
    CHANGE_PLAN = "change_plan"
    UPDATE_PAYMENT_METHOD = "update_payment_method"
    BOOK_CALLBACK = "book_callback"
    PAYMENT_ARRANGEMENT = "payment_arrangement"


# Actions that change money or the account itself. They need verified
# identity + explicit customer confirmation + policy checks, all three.
CONSEQUENTIAL_ACTIONS = {
    ActionKind.APPLY_CREDIT,
    ActionKind.ACTIVATE_OFFER,
    ActionKind.CHANGE_PLAN,
    ActionKind.UPDATE_PAYMENT_METHOD,
    ActionKind.PAYMENT_ARRANGEMENT,
}


class ProposedAction(BaseModel):
    kind: ActionKind
    summary: str  # customer-readable, written by the specialist
    payload: dict = Field(default_factory=dict)  # kind-specific, code-built
    amount: Decimal | None = None  # money involved, if any
    basis: str = ""  # why: the evidence/rule the proposal rests on


class ActionRecord(BaseModel):
    action_id: str
    conversation_id: str
    kind: ActionKind
    summary: str
    payload: dict = Field(default_factory=dict)
    amount: Decimal | None = None
    basis: str = ""
    status: str = "proposed"  # proposed | executed | declined | blocked | superseded
    block_reason: str = ""
    turn: int = 0
    # The account the action belongs to and the day it was raised.
    # The credit-history rules read across conversations by account,
    # over a trailing window of days — neither is derivable from the
    # conversation alone, so both ride the record itself.
    account_number: str = ""
    created_date: str = ""  # ISO date


class DisputeLine(BaseModel):
    """One disputed charge, decided on its own evidence."""

    charge_id: str
    description: str
    amount: Decimal  # the charge as billed
    disputed_amount: Decimal  # the part in dispute (the excess on a wrong rate)
    evidence: str = ""  # what the bill itself proves, when it proves anything
    decision: str = "credited"  # credited | declined | escalated
    reason: str = ""  # plain-words reason, set when the line is declined


class DisputeCase(BaseModel):
    """A dispute as a record of its own — not a by-product of a chat.

    Opened when the billing desk assesses a dispute; each line is
    decided on its own evidence, so a case can end in a full credit,
    a partial credit (some lines credited, some declined with the
    reason in plain words), a decline, or an escalation when the
    evidenced total is above the agent's cap. The case outlives the
    conversation and is what the disputes view and the audit trail
    point at.
    """

    case_id: str
    conversation_id: str
    account_number: str
    bill_id: str
    state: str = "opened"  # opened | decided_credit | decided_partial
    #                        | declined_with_reason | escalated_above_cap
    lines: list[DisputeLine] = Field(default_factory=list)
    credit_applied: Decimal = Decimal("0")
    decision_basis: str = ""
    goodwill_refused_reason: str = ""
    created_date: str = ""  # ISO date
    updated_date: str = ""


class Finding(BaseModel):
    """One structured fact a specialist established with a tool. The
    supervisor's reply may only assert what findings (or retrieved
    policy) support — this list is the grounding contract."""

    topic: str
    text: str
    data: dict = Field(default_factory=dict)


class SpecialistResult(BaseModel):
    specialist: str
    findings: list[Finding] = Field(default_factory=list)
    proposed_action: ProposedAction | None = None
    # Policy doc ids the reply relies on (filled from retrieval).
    citations: list[str] = Field(default_factory=list)
    # True when the specialist made real progress this turn (answered
    # the question, advanced the procedure, proposed the fix). The
    # no-progress escalation counter reads this.
    progress: bool = True
    # The specialist needs identity before it can continue.
    needs_identity: bool = False
    # The specialist needs full verification (one-time code) — a
    # stronger state than mere identification — before it can
    # continue. The supervisor runs the identity flow and resumes.
    needs_verification: bool = False
    # The specialist determined this case is beyond its authority
    # (e.g. a dispute credit above its cap): escalate, with the
    # composed findings as the explanation.
    escalate_reason: str = ""
    escalate_queue: str = ""
    # A direct line the supervisor must include verbatim (numbers,
    # policy terms) — composition may wrap it, never alter it.
    must_include: list[str] = Field(default_factory=list)
    resolved: bool = False  # the customer's need is fully met


class HandoffPacket(BaseModel):
    """What the human agent receives. Built in code from the
    conversation record at the moment of escalation — never written
    by the model from memory."""

    conversation_id: str
    queue: str  # general | billing | fraud | vulnerable_care
    reason: str
    identity_state: str
    customer_name: str = ""
    account_number: str = ""
    summary: str
    attempts: list[str] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    open_items: list[str] = Field(default_factory=list)


class Escalation(BaseModel):
    escalation_id: str
    conversation_id: str
    reason: str
    queue: str
    packet: HandoffPacket
    turn: int


class Conversation(BaseModel):
    conversation_id: str
    customer_id: str | None = None
    account_number: str | None = None
    status: str = "active"  # active | escalated | resolved | closed
    identity_state: IdentityState = IdentityState.UNVERIFIED
    identity_attempts: int = 0
    otp_code_hash: str | None = None  # never the code itself
    messages: list[Message] = Field(default_factory=list)
    intent: Intent = Intent.UNKNOWN
    specialist_state: dict = Field(default_factory=dict)  # e.g. procedure walk
    pending_action: ProposedAction | None = None
    human_requests: int = 0
    no_progress_turns: int = 0
    turns: int = 0
    unhandled_intent: str | None = None  # care logged a need it cannot serve


class TurnResult(BaseModel):
    conversation_id: str
    reply: str
    intent: str
    identity_state: str
    status: str
    escalated: bool = False
    escalation_reason: str = ""
    action: ActionRecord | None = None
    citations: list[str] = Field(default_factory=list)


class AuditEvent(BaseModel):
    event_id: str
    conversation_id: str
    kind: str  # turn | identity | action | escalation | screening | reroute
    detail: dict = Field(default_factory=dict)
    turn: int = 0
