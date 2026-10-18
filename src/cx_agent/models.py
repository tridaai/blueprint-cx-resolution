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
    # What a roaming pass includes before per-GB billing starts
    # again (the Roam Pass: 1 GB a day). The trip-quote math reads
    # it from the offer, never from prose.
    included_gb_per_day: Decimal | None = None


class RoamingRate(BaseModel):
    """The destination rate table, as data: what pay-as-you-go
    roaming data costs per GB in a destination. Every roaming
    figure the agent quotes is priced from these rows — the model
    never supplies a price."""

    destination: str  # the country name customers use
    per_gb: Decimal


class RoamingNotice(BaseModel):
    """A fair-use warning as a stored record — the proactive
    summary artifact. Created when roaming usage crosses a
    fair-use mark (80%, then 100% of the ceiling), carrying the
    computed figures the customer was shown: usage, ceiling,
    pace, and the projected overage cost. The record is also the
    evidence a later roaming dispute reads: a charge the customer
    was warned about stands; a crossing with no warning on record
    before it is itself flagged here (`unwarned_crossing`) as a
    policy event."""

    notice_id: str
    account_number: str
    kind: str  # fair_use_80 | fair_use_100
    destination: str = ""
    trip_start: str = ""  # ISO date the trip began (the account's)
    used_gb: Decimal = Decimal("0")
    cap_gb: Decimal = Decimal("0")
    pct: Decimal = Decimal("0")  # used as a % of the ceiling
    projected_overage_cost: Decimal | None = None
    summary: str = ""  # the customer-facing text, composed from the figures
    conversation_id: str = ""  # "" when the network side raised it
    unwarned_crossing: bool = False  # 100% crossed with no prior warning
    created_date: str = ""  # ISO date


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
    # Money Northbeam owes the customer (an overpayment, a credit
    # left after cancellation). Refunds are computed from this —
    # never from a figure named in conversation.
    credit_balance: Decimal = Decimal("0")
    payment_method_last4: str | None = None
    # Autopay draws the balance from the method on file when the
    # bill lands; the state lives here, the attempts are records.
    autopay_enabled: bool = False
    # The processor double's reading for the method on file: the
    # decline code it returns ("" approves). A test double, like
    # the one-time code — a deployment swaps it for the real
    # processor behind payments_book.process_payment. Replacing
    # the card clears it: a new method has no history.
    processor_code: str = ""
    roaming_country: str | None = None  # set while the customer travels
    roaming_gb_used: Decimal = Decimal("0")
    roaming_since: str = ""  # ISO date the current trip began
    # The usage profile: how much data this customer typically
    # uses in a day. Trip quotes are priced from it (days × this
    # figure), so a projection is the customer's own pattern
    # priced at the destination rate — not a generic guess.
    typical_daily_data_gb: Decimal = Decimal("0.5")
    # What the network-side diagnostic tools read for this account
    # (e.g. {"line_test": "fault_found"}). Absent a reading, a tool
    # reports its neutral result — the book states facts, tools
    # never invent them.
    diagnostics: dict[str, str] = Field(default_factory=dict)
    cycle_day: int = 11  # days elapsed in the current cycle (of 30)
    cycle_days: int = 30
    # The current cycle as dates on the record — proration reads
    # these (via the shared clock), never the bare counts above:
    # the counts say where the account sits, the dates say when
    # that is, and money is computed from the when.
    cycle_start: str = ""  # ISO date
    cycle_end: str = ""  # ISO date, inclusive
    # A plan change scheduled for the next cycle boundary (a
    # downgrade, by policy): the plan it will move to, and the
    # date it takes effect. The live plan does not move until then.
    pending_plan_id: str | None = None
    pending_plan_effective: str = ""  # ISO date
    # A cancellation scheduled for the cycle boundary (round 14):
    # the ISO date it takes effect, set by the cancel_account
    # action through the same next-cycle scheduling the pending
    # plan machinery uses — the account stays active, and the
    # customer keeps what they paid for, until that date. Empty
    # until a cancellation is confirmed.
    pending_cancellation_effective: str = ""  # ISO date
    open_case: str | None = None
    # Accessibility and contact preference, recorded at the care
    # desk from the customer's own words (care_book): the note is
    # what they said, the preference is the derived instruction
    # ("large print", "email", "text message") later contacts
    # honour. Empty until the customer states a need.
    accessibility_note: str = ""
    preferred_contact: str = ""


class PolicyDocument(BaseModel):
    doc_id: str
    title: str
    section: str
    text: str


class ProcedureStep(BaseModel):
    """One step of a troubleshooting procedure, as data.

    The walker (procedures.py) interprets the step by `kind`;
    changing any of this in the stored record changes the walk
    without touching code:

    - **instruction** — the customer does something and reports the
      outcome (success / fail / unclear, classified in code).
      `check` names the condition their report is about; a check
      that *is* the symptom disappearing resolves the walk on
      success. `on_fail` says where a failure goes:
      next | escalate | callback.
    - **tool_check** — the agent tests a condition itself, as a
      tool (`check` names the tool). The tool's result selects the
      path through `branches` (result → step_id), falling back to
      the next step.
    - **measurement** — the customer reports a value (a speed
      test, a signal reading). The value is parsed and compared
      against `threshold` in code: below goes to `below_step`,
      at-or-above to `at_or_above_step`.
    - **branch** — the next step is chosen by a named rule
      (`rule`) over the run's recorded results; `branches` maps
      the rule's answer to a step_id.
    - **terminal** — the walk ends here: `outcome` is
      resolved | book_callback | escalate_field_visit.

    A step flagged `mandatory_safety` can never be skipped: a
    skip attempt — or an outright refusal — is refused in code,
    recorded on the run, and a second insistence escalates.
    """

    step_id: str
    kind: str = "instruction"  # instruction | tool_check | measurement
    #                            | branch | terminal
    instruction: str = ""  # what the customer is told / asked
    check: str = ""  # the condition tested, e.g. "signal_restored"
    on_fail: str = "next"  # next | escalate | callback (instruction steps)
    # measurement steps
    metric: str = ""  # e.g. "download_mbps"
    unit: str = ""  # e.g. "Mbps"
    threshold: Decimal | None = None
    below_step: str | None = None
    at_or_above_step: str | None = None
    # branch steps (and tool_check path selection)
    rule: str = ""  # named rule over the run's results (procedures.RULES)
    branches: dict[str, str] = Field(default_factory=dict)  # answer → step_id
    # terminal steps
    outcome: str = ""  # resolved | book_callback | escalate_field_visit
    mandatory_safety: bool = False
    next_step: str | None = None  # explicit successor; default: following step


class Procedure(BaseModel):
    """A troubleshooting procedure as a versioned data record.

    Procedures are stored per version (`procedure_id` + `version`);
    a run pins the version it started on, so editing the data
    mid-flight never moves a customer who is already walking it.
    """

    procedure_id: str
    version: int = 1
    title: str
    symptom: str  # router key
    # The phrases in a customer message that name this symptom —
    # data, like everything else about the procedure.
    symptom_phrases: list[str] = Field(default_factory=list)
    # The walk needs the account before it can start (its tools
    # test the line): the desk asks for identity first.
    requires_account: bool = False
    steps: list[ProcedureStep] = Field(default_factory=list)


class StepRecord(BaseModel):
    """One entry in a procedure run's history: a step as it
    actually happened — the tool's result, the customer's outcome,
    the value they reported, the branch taken. This is the
    transcript a callback or field visit is booked with."""

    step_id: str
    kind: str  # instruction | tool_check | measurement | branch
    #          | terminal | safety_refusal
    detail: str = ""  # the check / metric / rule the step turned on
    result: str = ""  # tool result | success | fail | reported value
    #                 | branch answer | terminal outcome
    turn: int = 0


class ProcedureRun(BaseModel):
    """An in-flight (or finished) procedure walk, as a record of
    its own — the procedure-side sibling of the dispute case.

    The conversation's specialist state holds only the run's id;
    the run itself lives in the store with its full step history,
    so leaving the conversation and coming back resumes at the
    exact step, and finishing hands the history to whoever picks
    the case up (callback, field visit, escalation packet).
    """

    run_id: str
    conversation_id: str
    account_number: str = ""
    procedure_id: str
    procedure_version: int = 1
    state: str = "active"  # active | resolved | dead_ended | abandoned
    current_step_id: str = ""  # the step awaiting the customer, when active
    history: list[StepRecord] = Field(default_factory=list)
    safety_refusals: int = 0
    outcome: str = ""  # resolved | callback_booked | field_visit
    #                  | safety_escalation | conversation_escalated
    created_date: str = ""  # ISO date
    updated_date: str = ""


class CallbackRecord(BaseModel):
    """A technician callback as a first-class record — not a
    by-product of a chat. Created the moment a procedure dead-ends,
    carrying the run's full step transcript, and settled when the
    customer answers the booking proposal (booked / declined /
    blocked at execution), exactly as a dispute case settles."""

    callback_id: str
    conversation_id: str
    account_number: str = ""
    procedure_id: str
    procedure_version: int = 1
    run_id: str = ""
    reason: str = ""  # why the procedure dead-ended, in plain words
    status: str = "proposed"  # proposed | booked | declined | blocked
    transcript: list[StepRecord] = Field(default_factory=list)
    action_id: str = ""  # the executed book_callback action, once booked
    created_date: str = ""  # ISO date


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
    TAKE_PAYMENT = "take_payment"
    REFUND_PAYMENT = "refund_payment"
    SET_AUTOPAY = "set_autopay"
    CANCEL_ACCOUNT = "cancel_account"


# Actions that change money or the account itself. They need verified
# identity + explicit customer confirmation + policy checks, all three.
CONSEQUENTIAL_ACTIONS = {
    ActionKind.APPLY_CREDIT,
    ActionKind.ACTIVATE_OFFER,
    ActionKind.CHANGE_PLAN,
    ActionKind.UPDATE_PAYMENT_METHOD,
    ActionKind.PAYMENT_ARRANGEMENT,
    ActionKind.TAKE_PAYMENT,
    ActionKind.REFUND_PAYMENT,
    ActionKind.SET_AUTOPAY,
    ActionKind.CANCEL_ACCOUNT,
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


class RetentionCase(BaseModel):
    """A customer's stated intent to leave, as a record of its
    own (round 14) — the retention sibling of the dispute case.

    Opened the moment the intent is expressed and the account is
    known. The `reason` is classified in code against the
    governed list in retention_book; `reason_text` keeps the
    customer's words verbatim. The case carries the ONE save
    offer the policy allows (its mechanism, its summary, and the
    action record that carries it — empty until one is made),
    whether the customer asked for no offers at all, how many
    times they have asked to cancel, and where a fault-blamed
    exit was routed. It settles into exactly one outcome:
    saved_accepted_offer (the offer executed), cancelled (the
    cancellation executed, effective at the cycle boundary on
    the account), or escalated_retention_team (policy puts the
    case with humans — a fault-blamed exit, which escalates
    through the billing desk's dispute path with `routed`
    recording that)."""

    case_id: str
    conversation_id: str = ""
    account_number: str = ""
    reason: str = "other"  # the governed list lives in retention_book
    reason_text: str = ""  # the customer's words, verbatim
    state: str = "open"  # open | offer_proposed | saved
    #                    | cancelled | escalated_retention_team
    outcome: str = ""  # "" | saved_accepted_offer | cancelled
    #                  | escalated_retention_team
    routed: str = ""  # "" | billing_dispute_path (a fault-blamed exit)
    no_offers_requested: bool = False
    cancel_requests: int = 1
    save_offer_kind: str = ""  # the underlying action kind, once made
    save_offer_mechanism: str = ""  # plan_downgrade | goodwill_credit
    #                               | loyalty_upgrade
    save_offer_summary: str = ""
    save_offer_action_id: str = ""
    save_offer_outcome: str = ""  # "" | proposed | accepted
    #                             | declined | blocked
    cancellation_action_id: str = ""
    cancellation_effective: str = ""  # ISO date, once executed
    created_date: str = ""  # ISO date
    updated_date: str = ""


class PaymentAttempt(BaseModel):
    """One run of a payment against the method on file — approved
    or declined, with the processor's reason code when it declined
    and the class the code falls into. The attempt log is what the
    desk diagnoses from, and what the retry rules read: a HARD
    decline on a method is a fact about that method, and no later
    conversation may spend a customer's patience retrying it."""

    attempt_id: str
    conversation_id: str = ""
    account_number: str
    method_last4: str = ""
    amount: Decimal = Decimal("0")
    outcome: str = "approved"  # approved | declined
    reason_code: str = ""  # the processor's code, when declined
    decline_class: str = ""  # soft | hard, when declined
    source: str = "manual"  # manual | autopay
    action_id: str = ""  # the action record that ran it, if any
    created_date: str = ""  # ISO date


class ArrangementInstallment(BaseModel):
    """One dated slice of a payment arrangement. paid_amount is
    what payments have covered so far; the installment is paid
    when the two meet, and missed when its date passes short."""

    number: int
    amount: Decimal
    due_date: str  # ISO date
    paid_amount: Decimal = Decimal("0")
    paid_date: str = ""  # ISO date the slice was fully covered


class PaymentArrangement(BaseModel):
    """A promise-to-pay as a record of its own — the sibling of
    the dispute case. The schedule (amounts and dates) is computed
    in code under the arrangement policy; the state is derived
    from the installments and the calendar, never asserted:
    completed when every slice is covered, broken when a date
    passes with its slice short, active otherwise. A broken
    arrangement is a human's case — the agent does not
    renegotiate it."""

    arrangement_id: str
    conversation_id: str = ""
    account_number: str
    total: Decimal = Decimal("0")
    installments: list[ArrangementInstallment] = Field(default_factory=list)
    state: str = "active"  # active | completed | broken
    created_date: str = ""  # ISO date


class PaymentNotice(BaseModel):
    """A failed automatic payment as a stored record — the
    payments sibling of the roaming fair-use notice. The billing
    run raises it when autopay declines; it carries the computed
    figures (amount, method, reason) and the exact summary the
    customer is shown when they next make contact, so the failure
    is explained from the record, not reconstructed."""

    notice_id: str
    account_number: str
    kind: str = "autopay_failed"
    amount: Decimal = Decimal("0")
    method_last4: str = ""
    reason_code: str = ""
    summary: str = ""  # the customer-facing text, composed from the figures
    conversation_id: str = ""
    created_date: str = ""  # ISO date


class ComplaintRecord(BaseModel):
    """A complaint as a record of its own — the care desk's sibling
    of the dispute case. Written the moment the complaint is
    recognised, with the customer's words verbatim, a category from
    the governed list, and the response deadline policy promises
    (acknowledged in the conversation at once; the substantive
    response is a person's, due by `response_due`). The SLA watch
    (care_book.complaint_watch) reads these records: a complaint
    still open past its deadline is breached, and the watch raises
    it to the escalation queue — `sla_escalation_id` is that queue
    entry, set once."""

    complaint_id: str
    conversation_id: str = ""
    account_number: str = ""  # "" when the customer never identified
    category: str = "other"  # the governed list lives in care_book
    text: str = ""  # the customer's words, verbatim
    status: str = "open"  # open | responded | closed
    created_date: str = ""  # ISO date
    response_due: str = ""  # ISO date the substantive response is due
    sla_escalation_id: str = ""  # the watch's queue entry, once raised


class PromiseEntry(BaseModel):
    """One commitment on the promises ledger (round 16) — a
    DERIVED entry, never stored: the ledger is a read over the
    records that carry the commitment (a complaint's response
    deadline, a booked technician callback, an arrangement
    instalment), so it can never drift from the books it
    summarises. `state` is derived from the backing record plus
    the shared clock: kept (the commitment was met), pending
    (still within its time), broken (its date passed with the
    item still open). `source_id` is the backing record's id —
    every entry resolves to it; there is no entry without one."""

    promise_id: str  # PRM-<source id> — derived, stable per source
    kind: str  # complaint_response | callback | arrangement_instalment
    account_number: str
    source_id: str  # the backing record's id
    conversation_id: str = ""  # where the commitment was made
    summary: str = ""
    due_date: str = ""  # ISO date the commitment falls due
    state: str = "pending"  # pending | kept | broken


class ContinuityBriefing(BaseModel):
    """What a returning customer is opened with (round 16),
    computed in code from the account's records at the moment
    identity verifies: the open items with their deadlines, the
    promises already broken, the standing flags, the last
    conversation's outcome in one line, a proposal left pending,
    and the resumable marker a previous conversation left behind
    (an active procedure run). Every element names the record it
    came from (`ref_id` / `source_id`); `lines` are the
    customer-facing opening lines composed from those elements
    in code — the supervisor puts them in front of the reply
    verbatim. An account with no history gets an empty briefing:
    no lines, no opening beyond the ordinary greeting."""

    account_number: str
    open_items: list = []  # {kind, ref_id, summary, due_date}
    broken_promises: list = []  # PromiseEntry dicts, state == broken
    flags: dict = {}  # accessibility_note, preferred_contact, vulnerable
    last_outcome: str = ""  # one line; "" when there is no last conversation
    last_outcome_source: str = ""  # the conversation the line came from
    resumable_run_id: str = ""  # an active ProcedureRun left by another conversation
    resumable_summary: str = ""
    pending_proposal: dict = {}  # {kind, summary, conversation_id}
    lines: list = []  # the opening lines, in the order they're spoken


class FeedbackRecord(BaseModel):
    """A compliment or a piece of general feedback, recorded the
    moment it's recognised — data the business mines, not
    conversation filler. The customer's words ride verbatim; the
    kind is the category (compliment | feedback)."""

    record_id: str
    conversation_id: str = ""
    account_number: str = ""
    kind: str = "feedback"  # compliment | feedback
    text: str = ""  # the customer's words, verbatim
    created_date: str = ""  # ISO date


class AttributionRecord(BaseModel):
    """Why a conversation ended the way it did — computed in code
    from its own audit trail at close, never judged by a model.

    `outcome` is what happened (resolved, with or without an
    executed action; escalated; or ended unresolved). For a
    non-resolved outcome, `cause` names the one primary cause from
    the governed taxonomy in attribution.py — or is None when the
    ending was the designed behaviour (a threat, a second ask for
    a human) or no rule had anything true to say. `rule` always
    names the exact rule that fired, and `evidence` carries the
    signals it read — the reroute count, the escalation reason,
    the block reasons — so the attribution is explainable from the
    record alone."""

    conversation_id: str
    outcome: str  # resolved | resolved_with_action | escalated
    #             | unresolved_ended
    cause: str | None = None  # the taxonomy lives in attribution.py
    rule: str = ""  # the rule that fired, e.g. "dispute_above_cap"
    desk: str = ""  # the desk that owned the conversation at the end
    evidence: dict = Field(default_factory=dict)
    attributed_date: str = ""  # ISO date the record was computed


class AnnotationRecord(BaseModel):
    """One human's judgement on one conversation's attribution —
    the annotation queue's unit of work. The annotator says
    whether the attributed cause is right; when it isn't, the
    corrected cause comes from the same taxonomy (required then,
    forbidden otherwise). A second annotation must come from a
    different annotator; when the first two disagree, a third —
    the verdict, from a third person — settles it, and the record
    is closed. `seq` is the store-wide creation order, so "the
    first two" means the same thing on every store."""

    annotation_id: str
    conversation_id: str
    annotator: str
    cause_correct: bool
    corrected_cause: str | None = None
    note: str = ""
    is_verdict: bool = False
    seq: int = 0
    created_date: str = ""  # ISO date


class DailyDigest(BaseModel):
    """One day's quality digest (round 12): the snapshot the
    daily pass composes from that day's records — volume,
    outcomes and causes, the signal aggregates, the complaint
    book's state at day end, the mining candidates, the
    escalations, and the regression spot-check's verdict.

    The digest is a snapshot, not a running total: re-running
    the pass for a date REPLACES the stored digest for that date
    and raises `version` by one, so a replacement is always on
    the record, never silent. It informs humans — nothing
    anywhere acts on it automatically."""

    date: str  # ISO date the digest covers
    version: int = 1
    conversations: int = 0  # closed conversations attributed that day
    by_outcome: dict = {}
    by_entry_desk: dict = {}
    resolution_share: float | None = None
    cause_mix: dict = {}  # attributed cause -> count (designed endings excluded)
    escalated_by_design: int = 0
    no_cause: int = 0
    signals: dict = {}  # the day's signal aggregates (round 11 measures)
    complaints: dict = {}  # SLA states at day end: open/due_soon/breached/settled
    promises: dict = {}  # the promises ledger at day end (round 16)
    mining_candidates: list = []  # top unhandled-intent candidates
    escalations_by_reason: dict = {}
    handoffs: dict = {}  # handoff quality (round 17): the day's
    #                      incomplete packets with their gaps named,
    #                      and the book's unaccepted, overdue queue
    costs: dict = {}  # cost & latency (round 18): the day's model
    #                   calls folded from their records — calls by
    #                   kind, tokens by basis, latency, closings
    spot_check: dict = {}  # the regression spot-check's findings


class SignalRecord(BaseModel):
    """One turn's conversation signals, computed in code from the
    conversation's own records (signals.py) — how the customer is
    experiencing the conversation, as opposed to what it is
    about. Frustration is the governed lexicon's score of the
    customer's latest message and whether the series is rising;
    effort is the counted cost of the conversation so far (turns,
    identity attempts, reroutes, repeated asks — the
    components ride beside the score so the number explains
    itself); progress is whether the turn landed; confusion names
    the topic the customer failed to follow, when they did.
    `interventions` lists what the signal rules did on this turn
    — an offer made, an offer answered, a second explanation form
    served — each also an audit event."""

    record_id: str  # SIG-<conversation>-<turn>
    conversation_id: str
    turn: int
    desk: str = ""  # the desk that owned the turn
    frustration_score: int = 0
    frustration_rising: bool = False
    effort_score: int = 0
    effort_turns: int = 0
    effort_identity_attempts: int = 0
    effort_reroutes: int = 0
    effort_repeats: int = 0
    progress: bool = False
    findings_count: int = 0
    action_executed: bool = False
    confusion: bool = False
    confusion_topic: str = ""
    interventions: list[dict] = Field(default_factory=list)


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
    # How many verification attempts the customer spent getting
    # here (round 17): the state alone does not say whether the
    # human inherits a fresh case or a customer who has already
    # failed three codes.
    identity_attempts: int = 0
    # The issue in the customer's own words (round 17): their
    # first message, verbatim — the human reads what the customer
    # actually said the problem was, not a paraphrase.
    issue: str = ""
    # The account carries a vulnerability flag (seeded, or set from
    # the customer's own plainly stated circumstances in this
    # conversation) — the human sees it without digging.
    vulnerable: bool = False
    customer_name: str = ""
    account_number: str = ""
    summary: str
    attempts: list[str] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    open_items: list[str] = Field(default_factory=list)
    # What was tried, at the level the human continues from
    # (round 17): procedure steps with their outcomes, dispute
    # lines as they were decided, attempted actions — each line
    # from a record, in the order they happened.
    tried: list[str] = Field(default_factory=list)
    # The money the case turns on (round 17): action amounts and
    # disputed figures, each named with its record — the human
    # never has to re-derive what is at stake.
    figures: list[str] = Field(default_factory=list)
    # The account's open promises and complaints at handoff
    # (round 17), read from the round 16 promises ledger:
    # pending and broken commitments, oldest first. Empty can
    # mean "none open" or "never computed" — `open_state_known`
    # is the difference, and the completeness grade reads it.
    open_promises: list[str] = Field(default_factory=list)
    open_state_known: bool = False
    # The completeness grade (round 17), computed in code by
    # handoffs.grade_packet at the moment of handoff: which class
    # of escalation this is, and which of the elements its class
    # requires are missing. An incomplete packet still transfers
    # the customer — the grade names the gap for the human and
    # the daily digest; it never blocks the transfer.
    handoff_class: str = ""
    complete: bool = True
    missing: list[str] = Field(default_factory=list)
    # The customer's experience of the conversation, from the
    # signal records (signals.packet_summary): effort score and
    # its components, the frustration trend, the interventions
    # already tried and how they were answered, the turns taken.
    # The human sees what it cost the customer to get here, not
    # just the facts of the case.
    signals: dict = Field(default_factory=dict)
    # The known issues this conversation matched (round 15), read
    # live from the incident feed at the moment of handoff: only
    # incidents still active or monitoring ride the packet, each
    # as the record states it (id, title, status, the operator's
    # summary) — the human picks up a customer who was already
    # told about a known fault, and sees exactly what they were
    # told.
    known_incidents: list[dict] = Field(default_factory=list)


class Escalation(BaseModel):
    escalation_id: str
    conversation_id: str
    reason: str
    queue: str
    packet: HandoffPacket
    turn: int
    # The day the handoff was raised (round 17), by the shared
    # clock — the age an unaccepted handoff is judged by. Empty
    # on records written before the field existed; those cannot
    # be aged and are never reported as overdue.
    created_date: str = ""  # ISO date
    # Acceptance (round 17): the receiving side is a recorded
    # act, not an assumption — who picked the handoff up, and
    # the day they did. Both empty until someone does.
    accepted_by: str = ""
    accepted_date: str = ""  # ISO date


class Incident(BaseModel):
    """One known issue on the operator's incident feed (round 15)
    — a governed record, not a rumour the model heard.

    The feed is how the agent learns that a fault is already
    known: an operator writes the record (or the dataset seeds
    it), and deterministic matching in incidents.py decides
    whether a customer's report is explained by it. `service` is
    the troubleshooting vocabulary the procedures already use
    (their `symptom` keys — no_signal, slow_data, wifi_calling,
    broadband_drops), `affected_postcodes` scopes the issue to
    areas (empty means every area), and `symptom_signals` are the
    declared phrases a customer's report must carry for the
    incident to explain it. `summary` is the operator's own
    customer-safe wording: it is the only account of the issue
    the agent may give, verbatim — the agent never invents a
    cause or a fix time beyond this record. A `resolved` incident
    no longer matches anything; `active` and `monitoring` do."""

    incident_id: str
    title: str
    service: str = ""  # a procedure symptom key (incidents.SERVICES)
    affected_postcodes: list[str] = Field(default_factory=list)
    symptom_signals: list[str] = Field(default_factory=list)
    status: str = "active"  # active | monitoring | resolved
    started_at: str = ""  # ISO date the record states the issue began
    summary: str = ""  # operator-written, customer-safe, served verbatim
    updated_date: str = ""  # ISO date of the last operator act


class IncidentAct(BaseModel):
    """One recorded operator act on the incident feed: an
    incident added, updated, or resolved. Append-only, `seq` in
    store-wide creation order like the prompt registry's act log,
    so the feed's history reads the same on every store."""

    act_id: str
    seq: int = 0
    kind: str = ""  # add | update | resolve
    incident_id: str = ""
    actor: str = ""
    note: str = ""
    created_date: str = ""  # ISO date


class PromptEntry(BaseModel):
    """One version of one governed text (round 13).

    The registry's unit: every text the model is instructed by,
    or that the agent serves to the customer as governed wording,
    lives here as a versioned record — the compose instructions
    the supervisor drafts under (surface `compose`), the router's
    classification prompt (`intent`), the second explanation forms
    (`explanation`), and the policy corpus documents themselves
    (`policy`). Entries are keyed (`prompt_id`, `version`) and
    never edited or deleted: a change is a new version, and the
    active version is the highest unless an operator pins another
    (see PromptPin). `prompts.py` owns the registry's rules."""

    prompt_id: str  # compose.system | intent.user | explain.POL-ID-01 | policy.POL-BILL-01 | ...
    version: int = 1
    surface: str = ""  # compose | intent | explanation | policy
    text: str = ""


class PromptPin(BaseModel):
    """An operator's pin on one registry entry: while a pin
    stands, the pinned version is the active one, however many
    newer versions exist. A rollback IS a pin (to an older
    version); unpinning is its own recorded act."""

    prompt_id: str
    version: int
    actor: str = ""
    note: str = ""


class PromptAct(BaseModel):
    """One recorded act on the registry: a version added, a
    rollback (pin), an unpin. Append-only, `seq` in store-wide
    creation order like the annotation records, so the registry's
    history reads the same on every store."""

    act_id: str
    seq: int = 0
    kind: str = ""  # add_version | rollback | unpin
    prompt_id: str = ""
    version: int = 0  # the version the act concerns (0 for unpin)
    actor: str = ""
    note: str = ""
    created_date: str = ""  # ISO date


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
    # The prompt-registry versions that served this conversation
    # (round 13): prompt_id -> version, accumulated turn by turn —
    # the compose instructions in force, the intent prompt when a
    # hint was taken, each policy document at the version cited,
    # each second form at the version served. Historical replies
    # can be traced to the exact texts that produced them.
    prompt_versions: dict = Field(default_factory=dict)


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
