"""The payments book: every payment decision, computed here in
code — the payments sibling of the billing and roaming books.

- **The processor double** — a payment runs against the method on
  file and comes back approved or declined with a reason code.
  The double reads the code off the account record (a deployment
  swaps it for a real processor behind `process_payment`); every
  attempt, approved or not, lands on the attempt log.
- **Decline classes** — a code is SOFT (a retry can succeed:
  funds, a processor timeout) or HARD (the method itself is dead:
  expired, reported lost, account closed). The class is decided
  here, and it drives the remedy the customer is given and the
  retry rule: a hard decline standing against a method refuses
  any further attempt on that method, in the desk and again at
  execution.
- **Arrangements** — a promise-to-pay schedule computed under
  policy: how many installments, the exact amounts (they sum to
  the balance to the cent), and their dates. The state of an
  arrangement is derived from its installments and the calendar,
  never asserted. Payments pour into the earliest open slice.
- **Refunds** — money Northbeam owes goes back to the method on
  file, read from the record. Any other destination is refused;
  that rule lives here and again at execution.
- **Autopay** — the billing run charges the method on file; a
  decline raises a PaymentNotice carrying the figures, so the
  failure is explained from the record at the next contact.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from decimal import ROUND_FLOOR, Decimal

from .clock import today
from .config import Policy
from .money import CENT, fmt, money
from .models import (
    Account,
    ArrangementInstallment,
    PaymentArrangement,
    PaymentAttempt,
    PaymentNotice,
)

ZERO = Decimal("0")

# code -> (class, what happened in plain words, the remedy)
# SOFT means the method may still work; HARD means it never will.
DECLINES: dict[str, tuple[str, str, str]] = {
    "insufficient_funds": (
        "soft",
        "the card was declined for insufficient funds — there wasn't "
        "enough money in the account behind it",
        "Once funds are in that account, the same card can be tried "
        "again — nothing about the card itself is wrong.",
    ),
    "processor_timeout": (
        "soft",
        "the payment processor didn't answer in time, so nothing "
        "was taken",
        "Nothing is wrong with the card — trying again usually "
        "works.",
    ),
    "card_expired": (
        "hard",
        "the card on file has expired",
        "That card can never be charged again — the way forward "
        "is a new card on the account.",
    ),
    "card_reported_lost": (
        "hard",
        "the card on file has been reported lost or stolen",
        "That card can never be charged again — the way forward "
        "is a new card on the account.",
    ),
    "account_closed": (
        "hard",
        "the account behind the card on file has been closed",
        "That method can never be charged again — the way forward "
        "is a new card on the account.",
    ),
}


def decline_class(code: str) -> str:
    """soft | hard for a processor code. A code the table predates
    reads as soft: the non-destructive class — a retry is not
    forbidden on evidence the book doesn't have."""
    entry = DECLINES.get(code)
    return entry[0] if entry else "soft"


def decline_reason(code: str) -> str:
    entry = DECLINES.get(code)
    if entry:
        return entry[1]
    return f"the processor declined the card (code {code})"


def decline_remedy(code: str) -> str:
    entry = DECLINES.get(code)
    if entry:
        return entry[2]
    return "Trying again may work; if it keeps failing, a person can look at it."


# ---------------------------------------------------------------------------
# The processor double and the attempt log
# ---------------------------------------------------------------------------


def process_payment(store, account: Account, amount: Decimal, *,
                    source: str = "manual", conversation_id: str = "",
                    action_id: str = "") -> PaymentAttempt:
    """Run one payment against the method on file and record the
    attempt. The double's answer is the code on the account record
    ("" approves); the attempt lands on the log either way. Money
    does not move here — the caller applies an approval."""
    code = (account.processor_code or "").strip()
    attempt = PaymentAttempt(
        attempt_id=f"ATT-{uuid.uuid4().hex[:10]}",
        conversation_id=conversation_id,
        account_number=account.account_number,
        method_last4=account.payment_method_last4 or "",
        amount=money(amount),
        outcome="declined" if code else "approved",
        reason_code=code,
        decline_class=decline_class(code) if code else "",
        source=source, action_id=action_id,
        created_date=today().isoformat(),
    )
    store.save_payment_attempt(attempt)
    return attempt


def attempts_for(store, account_number: str) -> list[PaymentAttempt]:
    return store.list_payment_attempts(account_number)


def last_attempt(store, account_number: str) -> PaymentAttempt | None:
    rows = attempts_for(store, account_number)
    return rows[-1] if rows else None


def hard_decline_block(store, account: Account) -> PaymentAttempt | None:
    """The HARD decline standing against the method on file, if
    one stands: the latest attempt on the current method declined
    hard, with no approval on that method since. While it stands,
    no payment on that method may be attempted — the desk will
    not propose one and execution refuses one."""
    if not account.payment_method_last4:
        return None
    rows = [t for t in attempts_for(store, account.account_number)
            if t.method_last4 == account.payment_method_last4]
    for attempt in reversed(rows):
        if attempt.outcome == "approved":
            return None
        if attempt.decline_class == "hard":
            return attempt
    return None


def diagnosis_text(attempt: PaymentAttempt) -> str:
    """A declined attempt, explained from the record: what the
    processor said, in plain words, and the remedy for its class."""
    return (f"Your payment of {fmt(attempt.amount)} on "
            f"{attempt.created_date} didn't go through — "
            f"{decline_reason(attempt.reason_code)}. "
            f"{decline_remedy(attempt.reason_code)}")


def payment_outcome_text(store, record) -> tuple[str, bool]:
    """What the customer is told after a confirmed payment runs:
    (text, resolved). Read from the attempt the execution wrote —
    the reply asserts the outcome the record holds, nothing else."""
    attempts = [t for t in store.list_payment_attempts(
        record.account_number, record.conversation_id)
        if t.action_id == record.action_id]
    attempt = attempts[-1] if attempts else None
    account = store.get_account(record.account_number)
    if attempt is not None and attempt.outcome == "approved":
        balance = account.balance_due if account else ZERO
        return (f"Paid — {fmt(attempt.amount)} taken from the card "
                f"ending {attempt.method_last4}. Your balance is "
                f"now {fmt(balance)}.", True)
    if attempt is not None:
        return (diagnosis_text(attempt)
                + " Nothing was charged, and your balance is "
                "unchanged.", False)
    return ("The payment didn't go through, and nothing was "
            "charged.", False)


# ---------------------------------------------------------------------------
# Arrangements — schedule maths and derived state
# ---------------------------------------------------------------------------


def split_amounts(total: Decimal, policy: Policy) -> list[Decimal]:
    """The installment amounts for a balance: as many installments
    as policy allows (count capped, each at least the policy
    minimum), summing to the balance exactly — the floor share for
    each, with the residue cents dealt one each to the earliest
    installments."""
    total = money(total)
    by_minimum = int(total / policy.arrangement_min_installment)
    count = max(1, min(policy.arrangement_max_installments, by_minimum))
    base = (total / count).quantize(CENT, rounding=ROUND_FLOOR)
    residue = int((total - base * count) / CENT)
    return [base + CENT if i < residue else base for i in range(count)]


def plan_installments(total: Decimal, policy: Policy,
                      start: date | None = None) -> list[ArrangementInstallment]:
    """The schedule: amounts from the split, dates on the policy
    calendar — the first inside the policy window, the rest
    monthly after it."""
    start = start or today()
    out = []
    for idx, amount in enumerate(split_amounts(total, policy)):
        due = start + timedelta(
            days=policy.arrangement_first_due_days
            + idx * policy.arrangement_interval_days)
        out.append(ArrangementInstallment(
            number=idx + 1, amount=amount, due_date=due.isoformat()))
    return out


def arrangement_state(arr: PaymentArrangement, on: date | None = None) -> str:
    """active | completed | broken, derived from the installments
    and the calendar: completed when every slice is covered,
    broken when a due date has passed with its slice short."""
    on = on or today()
    if all(i.paid_amount >= i.amount for i in arr.installments):
        return "completed"
    if any(i.paid_amount < i.amount and date.fromisoformat(i.due_date) < on
           for i in arr.installments):
        return "broken"
    return "active"


def refresh_arrangements(store, account_number: str,
                         on: date | None = None) -> list[PaymentArrangement]:
    """Re-derive every arrangement state on the account and persist
    the ones the calendar has moved. Returns the refreshed list."""
    out = []
    for arr in store.list_payment_arrangements(account_number):
        state = arrangement_state(arr, on)
        if state != arr.state:
            arr.state = state
            store.save_payment_arrangement(arr)
        out.append(arr)
    return out


def active_arrangement(store, account_number: str) -> PaymentArrangement | None:
    for arr in refresh_arrangements(store, account_number):
        if arr.state == "active":
            return arr
    return None


def broken_arrangement(store, account_number: str) -> PaymentArrangement | None:
    for arr in refresh_arrangements(store, account_number):
        if arr.state == "broken":
            return arr
    return None


def apply_payment(store, account_number: str, amount: Decimal,
                  on: date | None = None) -> PaymentArrangement | None:
    """Pour an approved payment into the account's active
    arrangement, earliest open slice first (policy allows one
    active arrangement, so there is at most one target). Returns
    the arrangement touched, if any."""
    on = on or today()
    remaining = money(amount)
    if remaining <= 0:
        return None
    arr = active_arrangement(store, account_number)
    if arr is None:
        return None
    for inst in arr.installments:
        if remaining <= 0:
            break
        owed = inst.amount - inst.paid_amount
        if owed <= 0:
            continue
        paid = min(owed, remaining)
        inst.paid_amount += paid
        remaining -= paid
        if inst.paid_amount >= inst.amount:
            inst.paid_date = on.isoformat()
    arr.state = arrangement_state(arr, on)
    store.save_payment_arrangement(arr)
    return arr


def _join(words: list[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + f" and {words[-1]}"


def schedule_text(arr: PaymentArrangement) -> str:
    """The schedule as one customer-readable line, figures exact."""
    amounts = [i.amount for i in arr.installments]
    dates = [i.due_date for i in arr.installments]
    if len(set(amounts)) == 1 and len(amounts) > 1:
        return (f"{len(amounts)} monthly instalments of "
                f"{fmt(amounts[0])}, due on {_join(dates)}")
    parts = [f"{fmt(i.amount)} on {i.due_date}" for i in arr.installments]
    return f"{len(parts)} instalments: {_join(parts)}"


def arrangement_summary(arr: PaymentArrangement) -> str:
    """The arrangement in one line for findings and handoffs: its
    total, its state, and — when broken — the slice that broke it."""
    text = (f"a payment arrangement over {fmt(arr.total)} "
            f"({schedule_text(arr)}), currently {arr.state}")
    if arr.state == "broken":
        missed = next(i for i in arr.installments
                      if i.paid_amount < i.amount
                      and date.fromisoformat(i.due_date) < today())
        text += (f" — instalment {missed.number} of {fmt(missed.amount)} "
                 f"was due on {missed.due_date} and is "
                 f"{fmt(missed.amount - missed.paid_amount)} short")
    return text


# ---------------------------------------------------------------------------
# Refunds — to the method on file, and nowhere else
# ---------------------------------------------------------------------------


def refund_destination_allowed(account: Account, last4: str) -> bool:
    """A refund destination is allowed only when it is the method
    on the account record. Anything else — another card, a bank
    account, a cheque — is refused, every time: the rule that
    keeps a refund from becoming a payout to a stranger."""
    return bool(last4) and last4 == (account.payment_method_last4 or "")


# ---------------------------------------------------------------------------
# Autopay — the billing run and its failure artifact
# ---------------------------------------------------------------------------


def autopay_failure_summary(account: Account, attempt: PaymentAttempt) -> str:
    """The failed-autopay summary, composed from the figures: the
    amount, the method, the reason in plain words, what's still
    due, and the way forward for the decline's class."""
    remedy = ("Once the card can be charged, you can pay the balance "
              "here, and autopay will run again on the next bill."
              if attempt.decline_class == "soft" else
              "Autopay needs a working card — add a new card and the "
              "next bill will be taken automatically.")
    return (f"Your automatic payment of {fmt(attempt.amount)} to the "
            f"card ending {attempt.method_last4} didn't go through "
            f"on {attempt.created_date}: "
            f"{decline_reason(attempt.reason_code)}. Your balance "
            f"of {fmt(account.balance_due)} is still due. {remedy}")


def run_autopay(store, account_number: str,
                policy: Policy) -> tuple[PaymentAttempt | None,
                                         PaymentNotice | None]:
    """The billing-side run: charge the balance to the method on
    file for an autopay account. An approval settles the balance
    (and pours into any active arrangement, like any payment); a
    decline raises the PaymentNotice the next contact will be
    shown. Returns (attempt, notice) — both None when autopay
    isn't on or nothing is due."""
    account = store.get_account(account_number)
    if account is None or not account.autopay_enabled \
            or account.balance_due <= 0:
        return None, None
    amount = account.balance_due
    attempt = process_payment(store, account, amount, source="autopay")
    if attempt.outcome == "approved":
        account.balance_due = max(ZERO, account.balance_due - amount)
        store.save_account(account)
        apply_payment(store, account_number, amount)
        return attempt, None
    notice = PaymentNotice(
        notice_id=f"NTF-{uuid.uuid4().hex[:10]}",
        account_number=account.account_number,
        kind="autopay_failed",
        amount=attempt.amount,
        method_last4=attempt.method_last4,
        reason_code=attempt.reason_code,
        summary=autopay_failure_summary(account, attempt),
        created_date=today().isoformat(),
    )
    store.save_payment_notice(notice)
    return attempt, notice


def latest_autopay_failure(store, account_number: str
                           ) -> tuple[PaymentAttempt, PaymentNotice | None] | None:
    """The failure the next contact opens with, when one is
    standing: the account's latest attempt is a declined autopay
    run. Returns (attempt, its notice) or None."""
    attempt = last_attempt(store, account_number)
    if attempt is None or attempt.source != "autopay" \
            or attempt.outcome != "declined":
        return None
    notices = [n for n in store.list_payment_notices(account_number)
               if n.kind == "autopay_failed"
               and n.created_date == attempt.created_date
               and n.amount == attempt.amount]
    return attempt, (notices[-1] if notices else None)
