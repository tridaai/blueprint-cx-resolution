"""The six specialists.

Each is a function over (conversation, customer text, context) that
works with tools — the store, the retriever, the money module — and
returns a SpecialistResult: structured findings the supervisor may
assert, at most one proposed action, and the citations the reply
rests on. Specialists never format the final reply and never
execute anything; execution is actions.py, composition is the
supervisor's, and both re-check the rules at their own moment.

The desks work their happy paths; their deepening is scheduled
in the round map, not improvised here. Rerouting between desks
lives in the supervisor (round 2), not here.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from decimal import Decimal

from . import care_book, payments_book, roaming_book
from . import explanations as expl_mod
from .routing import analyze, classify_confirmation
from .billing_book import (
    assess_dispute,
    bill_shock,
    compare_bills,
    decompose_bill,
    goodwill_offer,
    line_evidence,
    note_goodwill_refusal,
    open_dispute_case,
)
from .clock import today
from .config import Policy
from .money import fmt, money
from .models import (
    ActionKind,
    AuditEvent,
    Conversation,
    Finding,
    PaymentArrangement,
    ProposedAction,
    SpecialistResult,
)
from . import plan_changes, procedures


@dataclass
class Ctx:
    store: object
    policy: Policy
    retriever: object


def _cite(ctx: Ctx, section: str, query: str, limit: int = 2) -> list[str]:
    return [d["doc_id"] for d in ctx.retriever.search(query, limit=limit, section=section)]


def _account(ctx: Ctx, conv: Conversation):
    if not conv.account_number:
        return None
    return ctx.store.get_account(conv.account_number)


def _needs_identity(specialist: str, ask: str) -> SpecialistResult:
    return SpecialistResult(
        specialist=specialist,
        findings=[Finding(topic="identity", text=ask)],
        needs_identity=True,
        progress=True,
    )


_IDENTITY_ASK = (
    "I can help with that. So I can pull up the right account, tell me "
    "your account number (it starts NB-) and the postcode on the account."
)


# ---------------------------------------------------------------------------
# Billing
# ---------------------------------------------------------------------------


def _evidenced_total(bill, verdicts=None) -> tuple[Decimal, list[str]]:
    """Charges proven wrong, as (total, evidence in words) — the
    quick scan the explanation and the goodwill gate use. The
    evidence is the bill's own markers plus, for roaming lines,
    the record verdicts roaming_book computed. The per-line
    decisions live in billing_book.assess_dispute."""
    total = Decimal("0")
    items: list[str] = []
    if bill is None:
        return total, items
    for charge in bill.charges:
        amount, evidence = line_evidence(charge)
        if amount > 0:
            total += amount
            items.append(evidence)
    for verdict in (verdicts or {}).values():
        if verdict.creditable > 0:
            total += verdict.creditable
            items.append(verdict.evidence)
    return total, items


def _goodwill(conv: Conversation, ctx: Ctx, case=None):
    """Goodwill, decided by the policy's memory.

    Returns (findings, action-or-None) when goodwill is in play for
    this customer — an offer, or a refusal with the reason said
    plainly and recorded — and None when it isn't (short tenure, or
    no customer record to read). The offer is the smaller of the
    per-offer cap and the headroom the trailing credit history
    leaves under the rolling cap."""
    customer = ctx.store.get_customer(conv.customer_id) if conv.customer_id else None
    if customer is None or customer.tenure_months < 12:
        return None
    offer, history = goodwill_offer(ctx.store, conv, ctx.policy)
    if history.offered_here:
        return None  # once per conversation (POL-BILL-02)
    citations = _cite(ctx, "billing", "goodwill credit poor experience")
    if offer <= 0:
        days = ctx.policy.credit_history_days
        note_goodwill_refusal(
            ctx.store, conv, case,
            reason=(f"rolling goodwill cap reached: {fmt(history.executed_total)} "
                    f"credited in the last {days} days against a "
                    f"{fmt(ctx.policy.goodwill_rolling_cap)} limit"),
            detail={"history_total": str(history.executed_total),
                    "rolling_cap": str(ctx.policy.goodwill_rolling_cap),
                    "window_days": days})
        return ([Finding(
            topic="goodwill_refused",
            text=(f"I'd add a goodwill credit if I could, but I can see "
                  f"we've already credited {fmt(history.executed_total)} in "
                  f"goodwill to your account in the last {days} days, and "
                  f"{fmt(ctx.policy.goodwill_rolling_cap)} is the most we can "
                  "add in that period — so I can't add another today. That "
                  "limit is only about goodwill: a charge that's actually "
                  "wrong still gets put right in full."))],
            None, citations)
    findings = [Finding(
        topic="goodwill",
        text=(f"Even so, you've been with us {customer.tenure_months} months "
              "and this has clearly been a frustrating stretch."))]
    basis = "Goodwill under POL-BILL-02; bill itself is correct."
    if offer < ctx.policy.goodwill_credit_cap:
        basis += (" Offer reduced to the headroom left under the rolling "
                  "goodwill cap.")
    return (findings, ProposedAction(
        kind=ActionKind.APPLY_CREDIT,
        summary=f"apply a goodwill credit of {fmt(offer)}",
        payload={"credit_class": "goodwill"},
        amount=offer, basis=basis), citations)


def _dispute_result(conv: Conversation, ctx: Ctx, bill,
                    assessment, unhappy: bool) -> SpecialistResult:
    """A dispute with identified lines: open the case, say the
    per-line decisions, and propose the credited sum — or escalate
    with the case record intact when the evidenced total is above
    the tier cap."""
    citations = _cite(ctx, "billing",
                      "billing dispute duplicate charge credited in full")
    credited_n = len(assessment.credited)
    declined_n = len(assessment.declined)
    basis = (f"Per-line assessment on bill {bill.bill_id} under POL-BILL-01: "
             f"{credited_n} line(s) credited on evidence, {declined_n} "
             "declined, the rest escalated." if assessment.escalated else
             f"Per-line assessment on bill {bill.bill_id} under POL-BILL-01: "
             f"{credited_n} line(s) credited on evidence, {declined_n} declined.")
    case = open_dispute_case(ctx.store, conv, bill, assessment, basis)

    if assessment.escalated:
        evidence_items = [l.evidence for l in assessment.lines if l.evidence]
        return SpecialistResult(
            specialist="billing",
            findings=[Finding(
                topic="dispute",
                text=(f"You're right to query this: "
                      f"{'; '.join(evidence_items)}. That comes to "
                      f"{fmt(assessment.evidenced_total)}, which is above the "
                      f"{fmt(ctx.policy.dispute_credit_cap)} I can credit from "
                      "here, so I'm passing this to our billing team with the "
                      f"evidence attached. It's logged as dispute case "
                      f"{case.case_id}, with every line's decision on it."))],
            citations=citations, progress=True,
            escalate_reason="dispute_above_agent_cap", escalate_queue="billing")

    findings: list[Finding] = []
    for line in assessment.lines:
        if line.decision == "credited":
            findings.append(Finding(
                topic="dispute_line",
                text=f"You're right about this one: {line.evidence}."))
        elif line.decision == "declined":
            findings.append(Finding(
                topic="dispute_line_declined",
                text=(f"On '{line.description}' of {fmt(line.amount)}: "
                      f"{line.reason}, so I can't credit that one.")))
    if assessment.credit_total > 0:
        findings.append(Finding(
            topic="dispute_total",
            text=(f"The amount that shouldn't be there is "
                  f"{fmt(assessment.credit_total)}.")))
        return SpecialistResult(
            specialist="billing", findings=findings,
            proposed_action=ProposedAction(
                kind=ActionKind.APPLY_CREDIT,
                summary=f"credit {fmt(assessment.credit_total)} to your account for the incorrect charge(s)",
                payload={"credit_class": "dispute",
                         "dispute_case_id": case.case_id},
                amount=assessment.credit_total,
                basis=(f"Evidenced on bill {bill.bill_id}, dispute case "
                       f"{case.case_id} (POL-BILL-01).")),
            citations=citations, progress=True)

    # Every disputed line declined: the case stands as the record,
    # and goodwill may still be in play for the experience.
    if unhappy:
        goodwill = _goodwill(conv, ctx, case=case)
        if goodwill is not None:
            gw_findings, gw_action, gw_citations = goodwill
            return SpecialistResult(
                specialist="billing", findings=findings + gw_findings,
                proposed_action=gw_action,
                citations=citations + [c for c in gw_citations
                                       if c not in citations],
                progress=True)
    return SpecialistResult(
        specialist="billing", findings=findings,
        citations=citations, progress=True)


def _shock_text(shock: dict) -> str:
    """The bill-shock lead: the jump and its computed driver. Every
    figure in it comes out of billing_book.bill_shock."""
    driver = shock["driver"]
    lead = (f"This bill is {fmt(shock['increase'])} higher than your recent "
            f"average of {fmt(shock['average'])}.")
    top = driver["top_line"]
    if driver["family"] == "one_off" and top is not None:
        if driver["is_new"]:
            return (f"{lead} One charge drives it: '{top.description}' at "
                    f"{fmt(top.amount)} — a one-off that wasn't on your "
                    "previous bill.")
        return (f"{lead} The main driver is a one-off charge: "
                f"'{top.description}' at {fmt(top.amount)}.")
    if driver["is_new"]:
        return (f"{lead} The main driver is {driver['label'].lower()} — "
                f"{fmt(driver['current'])} this month, where your previous "
                "bill had none.")
    return (f"{lead} The main driver is {driver['label'].lower()} — "
            f"{fmt(driver['current'])} this month against "
            f"{fmt(driver['previous'])} last month.")


def _comparison_text(current, previous) -> str:
    """The month-on-month read, from the computed deltas only."""
    changed = [r for r in compare_bills(current, previous) if r["delta"] != 0]
    if not changed:
        return (f"Against your previous bill of {fmt(previous.total)}, "
                "nothing moved — every charge family is the same.")
    parts = [
        f"{r['label'].lower()} {'up' if r['delta'] > 0 else 'down'} "
        f"{fmt(abs(r['delta']))}"
        for r in changed
    ]
    return (f"Against your previous bill of {fmt(previous.total)}: "
            + "; ".join(parts) + ".")


def billing(conv: Conversation, text: str, ctx: Ctx) -> SpecialistResult:
    if not conv.account_number:
        return _needs_identity("billing", _IDENTITY_ASK)
    bill = ctx.store.get_bill_for_account(conv.account_number)
    bills = ctx.store.list_bills_for_account(conv.account_number)
    previous_bills = bills[:-1] if len(bills) > 1 else []
    lowered = text.lower()
    dispute = any(w in lowered for w in (
        "dispute", "wrong", "shouldn't", "shouldnt", "charged twice",
        "refund", "fix it", "duplicate", "error", "charged after",
        "not fair", "remove the charge"))
    unhappy = any(w in lowered for w in ("expensive", "too much", "goodwill",
                                         "unhappy", "loyal", "ridiculous",
                                         "fed up"))
    # Roaming lines are decided from the account's records (pass
    # windows, warnings given, unwarned ceiling crossings) as
    # well as the bill's own markers — one evidence read for the
    # whole desk, computed before anything is decided.
    account = _account(ctx, conv)
    verdicts = roaming_book.roaming_dispute_verdicts(
        ctx.store, account, bill, ctx.policy, conv.conversation_id) if bill else {}
    evidence_total, evidence_items = _evidenced_total(bill, verdicts)

    # A dispute with identified lines becomes a case, decided line
    # by line: evidenced lines credit, named-but-clean lines
    # decline with reasons, an over-cap total escalates.
    if dispute and bill is not None:
        assessment = assess_dispute(bill, text, ctx.policy, verdicts)
        if assessment.lines:
            return _dispute_result(conv, ctx, bill, assessment, unhappy)

    # Goodwill: the bill is right, the experience wasn't. Available
    # whether or not the customer used the word "dispute" — and
    # bounded by the account's credit history, not just this chat.
    if unhappy and evidence_total == 0:
        goodwill = _goodwill(conv, ctx)
        if goodwill is not None:
            gw_findings, gw_action, gw_citations = goodwill
            findings = []
            if dispute:
                findings.append(Finding(
                    topic="dispute_checked",
                    text=("I've checked every line against the rates and dates on "
                          "the account, and each charge matches what the bill "
                          "should be — there isn't an incorrect charge to credit.")))
            findings += gw_findings
            return SpecialistResult(
                specialist="billing", findings=findings,
                proposed_action=gw_action, citations=gw_citations,
                progress=True)

    if dispute and evidence_total == 0:
        return SpecialistResult(
            specialist="billing",
            findings=[Finding(
                topic="dispute_checked",
                text=("I've checked every line against the rates and dates on "
                      "the account, and each charge matches what the bill "
                      "should be — there isn't an incorrect charge to credit."))],
            citations=_cite(ctx, "billing", "bill itemised lines correct charges"),
            progress=True)

    # Explain the bill: family by family, against last month's —
    # and when the jump is a shock, the driver leads, not the lines.
    if bill is None:
        return SpecialistResult(
            specialist="billing",
            findings=[Finding(topic="no_bill",
                              text="There's no bill issued on this account yet.")],
            progress=True)
    # The explanation is the topic the confusion signal pins to
    # this desk: the bill itself, marked here rather than left to
    # the citation, because the reply's subject is the customer's
    # own bill, not the policy about bills.
    sig = conv.specialist_state.setdefault("signals", {})
    sig["explained_topic"] = "bill"
    sig["explained_turn"] = conv.turns
    summary_sentence = (
        f"Your bill issued {bill.issued} for {bill.cycle_start} to "
        f"{bill.cycle_end} totals {fmt(bill.total)} and is due {bill.due}.")
    shock = bill_shock(bill, previous_bills, ctx.policy)
    findings: list[Finding] = []
    if (sig.get("second_form") or "") == "bill" \
            and "bill" not in sig.get("second_forms_used", []):
        # The second form (round 11): the same computed figures,
        # summary-first — the total and the driver lead, the lines
        # follow read slowly — for a customer the family-by-family
        # read did not land with.
        sig["second_form_served"] = "bill"
        findings.append(Finding(
            topic="bill_second_form",
            text=expl_mod.bill_second_form(
                issued_line=summary_sentence[0].lower()
                + summary_sentence[1:],
                driver_line=_shock_text(shock) if shock is not None
                else None)))
    else:
        if shock is not None:
            findings.append(Finding(topic="bill_shock",
                                    text=_shock_text(shock)))
        findings.append(Finding(
            topic="bill_summary",
            text=f"{summary_sentence} Here is what makes it up:"))
    must_include = [f"{c.description}: {fmt(c.amount)}" for c in bill.charges]
    for group in decompose_bill(bill):
        must_include.append(f"{group['label']} total: {fmt(group['total'])}")
    if previous_bills:
        previous = previous_bills[-1]
        findings.append(Finding(
            topic="bill_comparison", text=_comparison_text(bill, previous)))
        for row in compare_bills(bill, previous):
            if row["delta"] != 0:
                must_include.append(
                    f"{row['label']}: {fmt(row['current'])} "
                    f"(was {fmt(row['previous'])})")
    if evidence_total > 0:
        findings.append(Finding(
            topic="bill_flag",
            text=(f"One thing stands out: {'; '.join(evidence_items)}. "
                  "Say the word and I'll credit that back.")))
    return SpecialistResult(
        specialist="billing", findings=findings,
        citations=_cite(ctx, "billing", "itemised bill lines total due date"),
        must_include=must_include, progress=True, resolved=True)


# ---------------------------------------------------------------------------
# Roaming
# ---------------------------------------------------------------------------


_DAYS_RE = re.compile(r"(\d+)\s*-?\s*(?:days?|nights?)")
_WEEKS_RE = re.compile(r"(\d+)\s*weeks?")
_WEEK_PHRASES = (("a week", 7), ("one week", 7), ("two weeks", 14),
                 ("a fortnight", 14))
_COST_WORDS = ("cost", "cheaper", "cheapest", "worth", "price", "how much",
               "charge", "expensive")


def _parse_days(text: str) -> int | None:
    lowered = text.lower()
    match = _DAYS_RE.search(lowered)
    if match:
        return int(match.group(1))
    match = _WEEKS_RE.search(lowered)
    if match:
        return int(match.group(1)) * 7
    for phrase, days in _WEEK_PHRASES:
        if phrase in lowered:
            return days
    return None


def _quote_text(quote) -> str:
    """The trip quote as one finding: the projection first, the
    comparison second, the recommendation last — every figure
    from roaming_book's quote, none composed by a model."""
    text = (f"For {quote.days} days in {quote.destination} at your typical "
            f"{roaming_book.gb(quote.daily_gb)} a day, expect about "
            f"{roaming_book.gb(quote.expected_gb)}. Pay-as-you-go at "
            f"{fmt(quote.per_gb)}/GB comes to {fmt(quote.payg_cost)}.")
    if not quote.has_pass:
        return (f"{text} The Roam Pass isn't available on your current "
                "plan, so pay-as-you-go is the way to pay for this trip.")
    if quote.pass_overage_gb > 0:
        text += (f" The {quote.pass_name} is {fmt(quote.pass_price)} and "
                 f"covers {roaming_book.gb(quote.pass_allowance_gb)} over "
                 f"the trip (1 GB a day); the remaining "
                 f"{roaming_book.gb(quote.pass_overage_gb)} at "
                 f"{fmt(quote.per_gb)}/GB adds {fmt(quote.pass_overage_cost)}, "
                 f"so {fmt(quote.pass_cost)} all-in.")
    else:
        text += (f" The {quote.pass_name} is {fmt(quote.pass_price)} and "
                 f"covers {roaming_book.gb(quote.pass_allowance_gb)} over "
                 f"the trip (1 GB a day) — your whole expected usage fits "
                 f"inside it, so {fmt(quote.pass_cost)} all-in.")
    if quote.cheaper == "pass":
        text += f" The pass is cheaper by {fmt(quote.saving)}."
    elif quote.cheaper == "payg":
        text += (f" Pay-as-you-go is cheaper by {fmt(quote.saving)} — "
                 "I wouldn't add the pass for this trip.")
    else:
        text += " The two come out the same — either way works."
    return text


def _pass_proposal(offer, basis: str) -> ProposedAction:
    return ProposedAction(
        kind=ActionKind.ACTIVATE_OFFER,
        summary=f"activate the {offer.name} for {fmt(offer.one_off_price)}",
        payload={"offer_id": offer.offer_id},
        amount=offer.one_off_price, basis=basis)


def roaming(conv: Conversation, text: str, ctx: Ctx) -> SpecialistResult:
    lowered = text.lower()
    strong_buy = any(w in lowered for w in ("buy", "add", "activate", "purchase")) \
        and "pass" in lowered
    soft_buy = "get" in lowered and "pass" in lowered
    account = _account(ctx, conv)

    # Fair-use bookkeeping rides every turn that can see the
    # account: if usage has crossed a mark since the last warning
    # on record, the notice is raised now and its summary leads
    # whatever this turn goes on to say.
    notices = []
    if account is not None:
        notices = roaming_book.sync_fair_use_notices(
            ctx.store, account, ctx.policy, conv.conversation_id)
    summaries = [n.summary for n in notices]

    def result(**kwargs) -> SpecialistResult:
        kwargs["must_include"] = summaries + list(kwargs.get("must_include", []))
        return SpecialistResult(specialist="roaming", **kwargs)

    rate_row = roaming_book.find_destination(ctx.store, text)
    days = _parse_days(text)
    state = conv.specialist_state.setdefault("roaming", {})

    # A guard turn ends by inviting the customer to decide on the
    # pass — and it holds the floor for that decision (the
    # supervisor routes the next turn back here). An assent or a
    # buy verb completes the purchase path the guard deferred; a
    # refusal closes it with nothing added; anything else is a
    # new topic and falls through to be handled on its own terms.
    if state.get("awaiting_decision") == "activate_pass":
        verdict = classify_confirmation(text)
        if verdict == "other" and not strong_buy and not soft_buy \
                and rate_row is None and days is None \
                and not analyze(text).scores:
            # An answer we cannot read — a hesitant "um, yes I
            # think so", a fragment — is not a new topic: it
            # carries no signal of one (no destination, no day
            # count, no desk phrases at all). Falling through
            # would spend the marker all the same, and the
            # decision the guard deferred would be gone: the
            # cheaper pass the customer was mid-answer to could
            # never be bought, and their plain "yes" a turn later
            # would land nowhere. The decision stays open, the
            # floor stays ours, and the question is asked again,
            # plainly.
            conv.specialist_state["force_intent"] = "roaming"
            return result(
                findings=[Finding(
                    topic="bill_shock_guard",
                    text=("Just so I have it straight — shall I add "
                          "the Roam Pass, the cheaper option from "
                          "those figures? A yes or a no is all I "
                          "need."))],
                progress=False)
        state.pop("awaiting_decision", None)
        if verdict == "decline":
            return result(
                findings=[Finding(
                    topic="guard_declined",
                    text=("No problem — I haven't added anything, and "
                          "nothing has been charged. The figures stand "
                          "if you change your mind."))],
                progress=True)
        if verdict == "confirm" or strong_buy or soft_buy:
            strong_buy = True

    # Planning: a destination is named and the customer is asking
    # what it costs — with a length, or in cost words. A bare
    # "get" doesn't make it a purchase when a trip is being
    # priced; a real buy verb does, and is handled below.
    planning = rate_row is not None and not strong_buy and (
        days is not None or any(w in lowered for w in _COST_WORDS))
    if planning:
        if account is None:
            return _needs_identity("roaming", _IDENTITY_ASK)
        quote = roaming_book.quote_trip(
            ctx.store, account, rate_row.destination, days or 7, ctx.policy)
        if quote is None:
            return result(
                findings=[Finding(
                    topic="no_rate",
                    text=(f"I don't have a published roaming rate for "
                          f"{rate_row.destination}, so I won't guess at a "
                          "figure — a person can price that trip for you."))],
                progress=True)
        findings = [Finding(topic="trip_quote", text=_quote_text(quote))]
        citations = _cite(ctx, "roaming", "roam pass price projection warning")
        if quote.shock and not state.get("projection_shown"):
            # The bill-shock guard: the projection and comparison
            # lead, and nothing chargeable is proposed until the
            # customer has seen them. The withholding is the rule,
            # not a style — the proposal is simply not made.
            state["projection_shown"] = True
            if quote.cheaper == "pass":
                state["awaiting_decision"] = "activate_pass"
                conv.specialist_state["force_intent"] = "roaming"
            findings.append(Finding(
                topic="bill_shock_guard",
                text=("That pay-as-you-go projection is past the point "
                      "where I have to flag bill shock before anything "
                      "chargeable, so I won't propose adding anything "
                      "this turn — look at the two figures, and if you "
                      "want the cheaper option, say the word and I'll "
                      "set it up.")))
            return result(findings=findings, citations=citations,
                          must_include=[_quote_text(quote)], progress=True)
        if quote.cheaper == "pass":
            offer = ctx.store.get_offer("OFF-ROAM10")
            return result(
                findings=findings,
                proposed_action=_pass_proposal(
                    offer, "Recommended by the trip quote: cheaper than "
                    "pay-as-you-go at the destination rate (POL-ROAM-03)."),
                citations=citations, progress=True)
        return result(findings=findings, citations=citations, progress=True)

    if strong_buy or soft_buy:
        if account is None:
            return _needs_identity("roaming", _IDENTITY_ASK)
        offer = ctx.store.get_offer("OFF-ROAM10")
        findings: list[Finding] = []
        must_include: list[str] = []
        progress = roaming_book.fair_use_progress(account, ctx.policy)
        if account.roaming_country and progress["used"] > progress["cap"]:
            rate = roaming_book.rate_for(ctx.store, account.roaming_country)
            over_gb = progress["used"] - progress["cap"]
            over_cost = (money(over_gb * rate.per_gb)
                         if rate is not None else None)
            warning = (
                f"Fair-use warning: you have used {roaming_book.gb(progress['used'])} "
                f"roaming this trip, past the {roaming_book.gb(progress['cap'])} "
                "fair-use ceiling, so speeds are already reduced and "
                "further data is charged per GB.")
            if over_cost is not None and over_cost > ctx.policy.roaming_bill_shock_amount:
                if not state.get("projection_shown"):
                    # The same guard, fired from actual usage: the
                    # overage already run up leads, and the sale
                    # waits a turn.
                    state["projection_shown"] = True
                    state["awaiting_decision"] = "activate_pass"
                    conv.specialist_state["force_intent"] = "roaming"
                    guard = (
                        f"{warning} The {roaming_book.gb(over_gb)} past the "
                        f"ceiling at {account.roaming_country} rates "
                        f"({fmt(rate.per_gb)}/GB) already comes to "
                        f"{fmt(over_cost)} in pay-as-you-go charges. Look "
                        "at that figure before we add anything else "
                        "chargeable — a Roam Pass from here covers the "
                        "next 10 days, but it can't undo the overage "
                        "already run up. Say the word and I'll add it.")
                    return result(
                        findings=[Finding(topic="bill_shock_guard", text=guard)],
                        citations=_cite(ctx, "roaming",
                                        "roaming fair use projection warning"),
                        must_include=[warning], progress=True)
            must_include.append(warning)
            findings.append(Finding(
                topic="fair_use",
                text="Before anything chargeable, you should have that warning in full."))
        if account.plan_id not in offer.eligible_plan_ids:
            findings.append(Finding(
                topic="not_eligible",
                text=f"The Roam Pass isn't available on your current plan."))
            return result(findings=findings,
                          citations=_cite(ctx, "roaming", "roam pass countries price"),
                          must_include=must_include, progress=True)
        findings.append(Finding(
            topic="roam_pass",
            text=(f"{offer.name}: {fmt(offer.one_off_price)} one-off, and it "
                  f"covers {offer.duration_days} days. {offer.blurb}")))
        return result(
            findings=findings,
            proposed_action=_pass_proposal(
                offer, "Customer requested the pass in conversation (POL-ROAM-01)."),
            citations=_cite(ctx, "roaming", "roam pass price days activate"),
            must_include=must_include, progress=True)

    # Informational: how roaming works, or the customer's usage. A
    # question about the customer's OWN usage needs their account —
    # there is no honest generic answer to "how much have I used".
    if account is None and any(w in lowered for w in
                               ("my ", "have i", "i have", "i've", "did i")):
        return _needs_identity("roaming", _IDENTITY_ASK)
    findings = []
    if account is not None and account.roaming_country:
        progress = roaming_book.fair_use_progress(account, ctx.policy)
        findings.append(Finding(
            topic="roaming_status",
            text=(f"You're currently roaming in {account.roaming_country} and have "
                  f"used {roaming_book.gb(progress['used'])} this trip. The fair-use "
                  f"ceiling is {roaming_book.gb(progress['cap'])} — that's "
                  f"{roaming_book.num(progress['pct'])}% of it.")))
    docs = ctx.retriever.search(text, limit=2, section="roaming")
    for doc in docs:
        findings.append(Finding(topic="policy", text=doc["text"]))
    if not findings:
        findings.append(Finding(
            topic="roaming_general",
            text=("Roaming outside a pass is charged per GB at the destination "
                  "rate. Tell me where you're heading and I'll work out the "
                  "cheapest way to stay connected.")))
    return result(
        findings=findings,
        citations=[d["doc_id"] for d in docs], progress=True, resolved=bool(docs))


# ---------------------------------------------------------------------------
# Offers & plan changes — quoted, versioned, reversible (round 5:
# see plan_changes.py for the transaction rules themselves)
# ---------------------------------------------------------------------------


def _target_plan(text: str, ctx: Ctx):
    lowered = text.lower()
    for plan in ctx.store.list_plans():
        if plan.name.lower() in lowered:
            return plan
    if "unlimited" in lowered:
        return ctx.store.get_plan("PLAN-UNL")
    if "family" in lowered:
        return ctx.store.get_plan("PLAN-FAM")
    if "essential" in lowered:
        return ctx.store.get_plan("PLAN-ESS")
    if "basic" in lowered:
        return ctx.store.get_plan("PLAN-BAS")
    return None


_REVERSAL_WORDS = ("undo", "reverse", "revert", "change it back",
                   "change back", "put it back", "put me back")

# A downgrade or cancellation the customer blames on a fault is not
# a price question at all: the billing desk's dispute path owns it
# (POL-PLAN-02), and this desk routes it there instead of pricing
# an exit the fault may yet fix.
_FAULT_MARKERS = ("fault", "broken", "breaking", "not working",
                 "never worked", "no service", "outage", "unusable",
                 "keeps dropping")


def _fault_cited(lowered: str) -> bool:
    return any(marker in lowered for marker in _FAULT_MARKERS)


def _proposal_version(conv: Conversation, ctx: Ctx) -> int:
    """Proposals are versioned per conversation: the count of
    plan-change PROPOSAL records already on it, plus one. A
    proposal record is the one that starts life as "proposed"
    (it may later be superseded or declined); execution attempts
    write their own records and don't mint versions."""
    prior = [r for r in ctx.store.list_actions(conv.conversation_id)
             if r.kind == ActionKind.CHANGE_PLAN
             and r.status in ("proposed", "superseded", "declined")]
    return len(prior) + 1


def _net_line(net: Decimal) -> str:
    if net > 0:
        return f"Net adjustment on your next bill: {fmt(net)} due."
    if net < 0:
        return f"Net adjustment on your next bill: a {fmt(-net)} credit."
    return "Net adjustment on your next bill: $0.00."


def _change_result(conv: Conversation, ctx: Ctx, account,
                   target) -> SpecialistResult:
    """The plan-change proposal, quoted off the cycle calendar.

    The quote (plan_changes.quote_change) decides the shape by
    policy: an upgrade starts now, prorated, with the projected
    next bill stated; a downgrade is scheduled at the next cycle
    boundary with no proration. Either way the proposal is a
    versioned record stamped with the fingerprint of the account
    state it was priced against, and execution re-checks it."""
    current = ctx.store.get_plan(account.plan_id)
    if current.plan_id == target.plan_id:
        return SpecialistResult(
            specialist="offers",
            findings=[Finding(topic="same_plan",
                              text=f"You're already on {target.name}.")],
            progress=True, resolved=True)
    customer = ctx.store.get_customer(conv.customer_id)
    loyalty = ctx.store.get_offer("OFF-UPG-LOYAL")
    new_price = target.monthly_price
    basis = "Standard plan change under POL-PLAN-01."
    if (loyalty and loyalty.target_plan_id == target.plan_id
            and account.plan_id in loyalty.eligible_plan_ids
            and customer and customer.tenure_months >= 12):
        new_price = loyalty.monthly_price
        basis = (f"Loyalty upgrade OFF-UPG-LOYAL: {fmt(new_price)}/month "
                 f"instead of {fmt(target.monthly_price)} (tenure "
                 f"{customer.tenure_months} months).")
    quote = plan_changes.quote_change(account, current, target, new_price)
    payload = {
        "new_plan_id": target.plan_id,
        "old_plan_id": current.plan_id,
        "new_price": str(new_price), "net": str(quote.net),
        "credit": str(quote.credit), "charge": str(quote.charge),
        "direction": quote.direction, "effective": quote.effective,
        "effective_date": quote.effective_date,
        "cycle_start": quote.cycle_start, "cycle_end": quote.cycle_end,
        "days_remaining": quote.days_remaining,
        "days_in_cycle": quote.days_in_cycle,
        "next_bill_impact": str(quote.next_bill_impact),
        "proposal_version": _proposal_version(conv, ctx),
        "issued_date": today().isoformat(),
        "fingerprint": plan_changes.account_fingerprint(account),
        "fingerprint_parts": plan_changes.fingerprint_parts(account),
    }
    if quote.effective == "next_cycle":
        findings = [Finding(
            topic="plan_change",
            text=(f"Moving from {current.name} "
                  f"({fmt(current.monthly_price)}/month) down to "
                  f"{target.name} ({fmt(new_price)}/month) is a downgrade, "
                  "and downgrades start at the next cycle rather than "
                  "today: there is no proration on a downgrade, and you "
                  "keep the plan you've already paid for until this "
                  "cycle ends."))]
        must_include = [
            f"{target.name} takes effect on {quote.effective_date} "
            "(the start of your next cycle).",
            f"Until then you stay on {current.name}; from "
            f"{quote.effective_date}, {target.name} is "
            f"{fmt(new_price)}/month.",
            "No charge or credit applies to this change.",
        ]
        return SpecialistResult(
            specialist="offers", findings=findings,
            proposed_action=ProposedAction(
                kind=ActionKind.CHANGE_PLAN,
                summary=(f"change your plan to {target.name} at "
                         f"{fmt(new_price)}/month from "
                         f"{quote.effective_date}"),
                payload=payload, amount=Decimal("0"),
                basis=basis + " Downgrade under POL-PLAN-02: next-cycle "
                              "start, no proration."),
            citations=_cite(ctx, "plans",
                            "downgrade next cycle no proration plan change"),
            must_include=must_include, progress=True)
    findings = [Finding(
        topic="plan_change",
        text=(f"Switching from {current.name} ({fmt(current.monthly_price)}/month) "
              f"to {target.name} ({fmt(new_price)}/month) now, with "
              f"{quote.days_remaining} days left in this cycle "
              f"(it runs to {quote.cycle_end}), works out as follows."))]
    must_include = [
        f"Credit for the unused part of {current.name}: {fmt(quote.credit)}",
        f"Charge for {target.name} for the remaining "
        f"{quote.days_remaining} days: {fmt(quote.charge)}",
        _net_line(quote.net),
        f"Projected next bill: {fmt(quote.next_bill_impact)} — the net "
        f"adjustment above plus the first full month of {target.name} "
        f"at {fmt(new_price)}.",
        f"From the next cycle, {target.name} is {fmt(new_price)}/month.",
    ]
    return SpecialistResult(
        specialist="offers", findings=findings,
        proposed_action=ProposedAction(
            kind=ActionKind.CHANGE_PLAN,
            summary=(f"change your plan to {target.name} at "
                     f"{fmt(new_price)}/month, effective now"),
            payload=payload, amount=quote.net, basis=basis),
        citations=_cite(ctx, "plans", "plan change prorated credit charge confirm"),
        must_include=must_include, progress=True)


def _reversal_result(conv: Conversation, ctx: Ctx,
                     account) -> SpecialistResult:
    """'Undo that change': inside the reversal window the desk
    proposes the compensating pair; outside it, the request goes
    to the offers team with the change record attached — the agent
    does not improvise a reversal policy won't cover."""
    citations = _cite(ctx, "plans", "plan change reversed same day undo")
    orig = plan_changes.latest_executed_change(ctx.store,
                                                account.account_number)
    if orig is None:
        return SpecialistResult(
            specialist="offers",
            findings=[Finding(
                topic="no_change_to_reverse",
                text=("There's no plan change on your account for me to "
                      "undo — your plan is what it has been."))],
            citations=citations, progress=True)
    old_plan = ctx.store.get_plan(orig.payload.get("old_plan_id", ""))
    new_plan = ctx.store.get_plan(orig.payload.get("new_plan_id", ""))
    old_name = old_plan.name if old_plan else "your previous plan"
    new_name = new_plan.name if new_plan else "the new plan"
    pair = plan_changes.reversal_of(ctx.store, orig)
    if pair:
        return SpecialistResult(
            specialist="offers",
            findings=[Finding(
                topic="already_reversed",
                text=(f"That change was already reversed on "
                      f"{pair[0].created_date}: {old_name} was restored "
                      "and the money unwound then, so there's nothing "
                      "left to undo."))],
            citations=citations, progress=True)
    if not plan_changes.within_reversal_window(orig, ctx.policy):
        return SpecialistResult(
            specialist="offers",
            findings=[Finding(
                topic="reversal_outside_window",
                text=(f"You changed to {new_name} on {orig.created_date} "
                      f"(change record {orig.action_id}: {orig.summary}). "
                      "A change can be reversed from here on the day it "
                      "was made, and that day has passed — so I'm passing "
                      "this to our offers team with that change record "
                      "attached, and they can still sort it out."))],
            citations=citations, progress=True,
            escalate_reason="plan_reversal_outside_window",
            escalate_queue="offers")
    orig_net = Decimal(str(orig.payload.get("net", "0")))
    scheduled = orig.payload.get("effective") == "next_cycle"
    if scheduled:
        summary = (f"cancel the scheduled change to {new_name} and "
                   f"keep {old_name}")
        must_include = [
            f"The scheduled change to {new_name} is cancelled — you "
            f"stay on {old_name}, and no money was ever moved for it."]
        amount = Decimal("0")
    else:
        summary = (f"reverse the change to {new_name} and put you "
                   f"back on {old_name}")
        must_include = [
            f"Reversal: {old_name} is restored from today, and the "
            f"{fmt(orig_net)} net adjustment from the change is "
            "unwound — your balance goes back to what it was "
            "before it."]
        amount = -orig_net
    payload = {
        "reversal_of": orig.action_id,
        "restore_plan_id": orig.payload.get("old_plan_id", ""),
        "new_plan_id": orig.payload.get("new_plan_id", ""),
        "orig_net": str(orig_net), "unwind_net": str(-orig_net),
        "net": str(-orig_net),
        "effective": orig.payload.get("effective", "immediate"),
        "proposal_version": _proposal_version(conv, ctx),
        "issued_date": today().isoformat(),
        "fingerprint": plan_changes.account_fingerprint(account),
        "fingerprint_parts": plan_changes.fingerprint_parts(account),
    }
    return SpecialistResult(
        specialist="offers", findings=[Finding(
            topic="plan_reversal",
            text=(f"I can undo that change — it was made today, so "
                  f"it's inside the reversal window."))],
        proposed_action=ProposedAction(
            kind=ActionKind.CHANGE_PLAN, summary=summary,
            payload=payload, amount=amount,
            basis=(f"Reversal of {orig.action_id} ({orig.summary}) "
                   "inside the reversal window (POL-PLAN-02).")),
        citations=citations, must_include=must_include, progress=True)


def refresh_plan_change(conv: Conversation, ctx: Ctx,
                        pending) -> SpecialistResult:
    """Rebuild a plan-change proposal that execution refused as
    stale. The figures are re-quoted from the live account — the
    same desk logic, not a re-read of the stale payload — and the
    result leads with what moved, so the customer re-confirms
    knowing exactly why they're being asked again."""
    account = _account(ctx, conv)
    lead = Finding(
        topic="stale_proposal",
        text=("That proposal went stale before it could run: your "
              "account changed after I priced it, so its figures no "
              "longer match your account. Nothing was applied."))
    if account is None:
        return SpecialistResult(specialist="offers", findings=[lead],
                                progress=True)
    moved = plan_changes.fingerprint_diff(
        account, pending.payload.get("fingerprint_parts", {}))
    if moved:
        lead = Finding(
            topic="stale_proposal",
            text=("That proposal went stale before it could run — "
                  + "; ".join(moved) +
                  " after I priced it, so its figures no longer match "
                  "your account. Nothing was applied. Here is the same "
                  "request priced fresh:"))
    if pending.payload.get("reversal_of"):
        fresh = _reversal_result(conv, ctx, account)
    else:
        target = ctx.store.get_plan(pending.payload.get("new_plan_id", ""))
        if target is None:
            return SpecialistResult(specialist="offers", findings=[lead],
                                    progress=True)
        fresh = _change_result(conv, ctx, account, target)
    fresh.findings = [lead] + fresh.findings
    return fresh


def offers(conv: Conversation, text: str, ctx: Ctx) -> SpecialistResult:
    # "How does a plan change work?" is a question, not an order:
    # answer it from the plan policy instead of listing offers.
    lowered0 = text.lower()
    if any(q in lowered0 for q in ("how does", "how do i", "how is",
                                   "what happens", "explain")):
        docs = ctx.retriever.search(text, limit=2, section="plans")
        if docs:
            return SpecialistResult(
                specialist="offers",
                findings=[Finding(topic="policy", text=d["text"]) for d in docs],
                citations=[d["doc_id"] for d in docs],
                progress=True, resolved=True)
    lowered = lowered0
    # "Undo that change" is its own transaction, not a new change:
    # it is priced off the change record, inside or outside the
    # reversal window — never off the catalogue.
    if any(w in lowered for w in _REVERSAL_WORDS):
        if not conv.account_number:
            return _needs_identity("offers", _IDENTITY_ASK)
        account = _account(ctx, conv)
        if account is None:
            return _needs_identity("offers", _IDENTITY_ASK)
        return _reversal_result(conv, ctx, account)
    target = _target_plan(text, ctx)
    wants_change = target is not None or any(
        w in lowered for w in ("upgrade", "change", "switch", "downgrade"))
    if wants_change and target is not None:
        if not conv.account_number:
            return _needs_identity("offers", _IDENTITY_ASK)
        account = _account(ctx, conv)
        current = ctx.store.get_plan(account.plan_id)
        if _fault_cited(lowered) and \
                target.monthly_price < current.monthly_price:
            return SpecialistResult(
                specialist="offers",
                findings=[Finding(
                    topic="fault_handoff",
                    text=("If the service itself is at fault, that's "
                          "not a plan change to price here — a "
                          "downgrade because of a fault belongs to "
                          "our billing team under the dispute policy, "
                          "so the fault and the charges get looked at "
                          "together. I'm passing this to them now "
                          "with what you've described attached."))],
                citations=_cite(ctx, "plans",
                                "fault downgrade cancellation billing dispute"),
                progress=True,
                escalate_reason="fault_cancellation_dispute_path",
                escalate_queue="billing")
        return _change_result(conv, ctx, account, target)
    if "cancel" in lowered and _fault_cited(lowered):
        if not conv.account_number:
            return _needs_identity("offers", _IDENTITY_ASK)
        return SpecialistResult(
            specialist="offers",
            findings=[Finding(
                topic="fault_handoff",
                text=("Leaving because of a fault is the billing "
                      "team's case, not a plan change — they own "
                      "fault cancellations under the dispute policy, "
                      "so the fault and the charges get looked at "
                      "together. I'm passing this to them now with "
                      "what you've described attached."))],
            citations=_cite(ctx, "plans",
                            "fault downgrade cancellation billing dispute"),
            progress=True,
            escalate_reason="fault_cancellation_dispute_path",
            escalate_queue="billing")

    # Browse: what's available.
    findings = []
    for offer in ctx.store.list_offers():
        findings.append(Finding(
            topic="offer",
            text=f"{offer.name}: {offer.blurb}"))
    if conv.account_number:
        findings.append(Finding(
            topic="eligibility",
            text=("Those are the headline terms; tell me which one you want and "
                  "I'll check it against your account and give you exact figures.")))
    else:
        findings.append(Finding(
            topic="eligibility",
            text=("If you give me your account number and postcode I can check "
                  "which of these you're eligible for and price a change exactly.")))
    return SpecialistResult(specialist="offers", findings=findings, progress=True)


# ---------------------------------------------------------------------------
# Technical support — guided procedures
# ---------------------------------------------------------------------------


def technical(conv: Conversation, text: str, ctx: Ctx) -> SpecialistResult:
    """The desk is a thin adapter over the procedure engine
    (procedures.py): procedures are versioned data, a walk is a
    stored run with its full step history, and a dead end books
    its callback as a record carrying the transcript. The desk
    owns the conversation-shaped edges — the callback proposal
    that follows identity, the account a procedure needs before
    its tools can run, and the symptom that starts a walk."""
    state = conv.specialist_state.setdefault("technical", {})

    if state.get("want_callback") and conv.account_number:
        state.pop("want_callback")
        proc = ctx.store.get_procedure(state.get("procedure_id", ""))
        if proc is not None:
            return procedures.callback_proposal(ctx, conv, proc)

    run = None
    if state.get("run_id"):
        candidate = ctx.store.get_procedure_run(state["run_id"])
        if candidate is not None and candidate.state == "active":
            run = candidate
        elif candidate is not None and candidate.state == "dead_ended" \
                and state.get("want_callback"):
            run = None  # the dead end is parked on its identity ask
        else:
            state.pop("run_id", None)
            state.pop("procedure_id", None)
    if run is not None:
        return procedures.advance(ctx, conv, run, text)

    if state.get("want_callback"):
        # The dead end is waiting on identity the supervisor is
        # already collecting; hold the turn rather than starting
        # a second walk over the top of it.
        return SpecialistResult(
            specialist="technical",
            findings=[Finding(
                topic="identity",
                text=("Once I have your account number and postcode I "
                      "can book that technician callback straight "
                      "away."))],
            needs_identity=True, progress=True)

    # Start a procedure from the symptom — matched on the phrases
    # the procedure's own data carries.
    proc = procedures.match_procedure(ctx.store, text)
    if proc is None:
        return SpecialistResult(
            specialist="technical",
            findings=[Finding(
                topic="unknown_fault",
                text=("I don't have a guided fix for that one yet. Describe "
                      "exactly what the phone does — any error message, when "
                      "it started — and I'll either find the right steps or "
                      "book you a technician."))],
            progress=False)
    if proc.requires_account and not conv.account_number:
        return _needs_identity(
            "technical",
            "I can run a line test on that straight away — it finds "
            "most of these. " + _IDENTITY_ASK)
    return procedures.start_run(ctx, conv, proc)


# ---------------------------------------------------------------------------
# Account & payments
# ---------------------------------------------------------------------------


def account(conv: Conversation, text: str, ctx: Ctx) -> SpecialistResult:
    lowered = text.lower()
    acct = _account(ctx, conv)
    state = conv.specialist_state.setdefault("account", {})

    # A broken arrangement is a human's case, decided before
    # anything else this desk might say: the schedule was missed,
    # the record says so, and the agent does not renegotiate it.
    # The escalation carries the record — its id rides the audit
    # trail, its facts ride the findings.
    if acct is not None:
        broken = payments_book.broken_arrangement(
            ctx.store, acct.account_number)
        if broken is not None:
            summary = payments_book.arrangement_summary(broken)
            ctx.store.save_audit(AuditEvent(
                event_id=f"EVT-{uuid.uuid4().hex[:10]}",
                conversation_id=conv.conversation_id, kind="payments",
                detail={"event": "broken_arrangement_routed",
                        "arrangement_id": broken.arrangement_id,
                        "account_number": acct.account_number,
                        "summary": summary},
                turn=conv.turns))
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="broken_arrangement",
                    text=(f"I can see {summary}. Once an arrangement "
                          "breaks, it's handled by a person on our "
                          "team — they can look at the whole picture "
                          "and agree a way forward with you. I'm "
                          "passing you over with the arrangement "
                          "record attached, so you won't have to "
                          "explain it again."))],
                escalate_reason="broken_payment_arrangement",
                escalate_queue="general",
                progress=True)

    # A HARD decline standing against the method on file ends any
    # retry conversation before it starts: the desk refuses here,
    # and execution refuses again if a proposal ever slips past.
    block = (payments_book.hard_decline_block(ctx.store, acct)
             if acct is not None else None)
    if block is not None and any(w in lowered for w in
                                 ("try", "again", "retry", "same card")):
        return SpecialistResult(
            specialist="account",
            findings=[Finding(
                topic="hard_decline_refusal",
                text=(payments_book.diagnosis_text(block)
                      + " I can't try that card again — a hard "
                      "decline means it will never go through. The "
                      "way forward is a new card: tell me the last "
                      "four digits of a different card and I'll put "
                      "it on the account, then we can pay with "
                      "that instead."))],
            citations=_cite(ctx, "payments",
                            "declined payment retry new card"),
            progress=True)

    # A failed autopay run is the first thing a payments contact
    # hears about: its notice summary leads, verbatim, on the
    # branches where the customer is asking after their money.
    failure = (payments_book.latest_autopay_failure(
        ctx.store, acct.account_number) if acct is not None else None)
    failure_lead: list[str] = []
    if failure is not None:
        attempt, notice = failure
        failure_lead = [notice.summary if notice is not None
                        else payments_book.diagnosis_text(attempt)]

    if "refund" in lowered or (state.get("refund_open") and any(
            w in lowered for w in ("card on file", "method on file",
                                   "send it back", "to the card"))):
        if acct is None:
            return _needs_identity("account", _IDENTITY_ASK)
        credit = acct.credit_balance
        if credit <= 0:
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="refund_none",
                    text=("There's no credit balance on your account "
                          "— nothing is owed back right now. If a "
                          "payment has left your bank twice, the "
                          "second one may still be settling; what "
                          f"shows here is a balance of "
                          f"{fmt(acct.balance_due)} due."))],
                citations=_cite(ctx, "payments", "refund overpayment"),
                progress=True)
        digits = [d for d in re.findall(r"(?<!\d)(\d{4})(?!\d)", text)
                  if d != acct.payment_method_last4]
        foreign = bool(digits) or any(w in lowered for w in (
            "bank account", "bank transfer", "cheque", "another card",
            "different card", "new card", "cash"))
        if foreign:
            # The refund stays open on the corrected destination:
            # the customer's next message ("the card on file")
            # belongs to this branch, not the card-update branch.
            state["refund_open"] = True
            named = ("a bank account"
                     if any(w in lowered for w in
                            ("bank account", "bank transfer"))
                     else "another card" if digits or "card" in lowered
                     else "a different destination")
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="refund_destination_refused",
                    text=(f"I can't send a refund to {named}. "
                          "Refunds go back to the payment method on "
                          f"your account — the card ending "
                          f"{acct.payment_method_last4} — and nowhere "
                          "else; that's a fraud-protection rule and "
                          "it applies every time. The amount owed "
                          f"back to you is {fmt(credit)}. Say the "
                          "word and I'll send it to the card ending "
                          f"{acct.payment_method_last4}."))],
                citations=_cite(ctx, "payments",
                                "refund payment method on file fraud"),
                progress=True)
        state["refund_open"] = False
        return SpecialistResult(
            specialist="account",
            findings=[Finding(
                topic="refund",
                text=(f"You're owed {fmt(credit)} back — it sits on "
                      "your account as a credit balance. Refunds go "
                      f"to the card ending {acct.payment_method_last4} "
                      "on your account."))],
            proposed_action=ProposedAction(
                kind=ActionKind.REFUND_PAYMENT,
                summary=(f"refund {fmt(credit)} to the card ending "
                         f"{acct.payment_method_last4}"),
                payload={"amount": str(credit),
                         "method_last4": acct.payment_method_last4 or ""},
                amount=credit,
                basis="Credit balance on the account; refunds go "
                      "to the method on file only (POL-PAY-03)."),
            citations=_cite(ctx, "payments",
                            "refund payment method on file fraud"),
            progress=True)

    if any(w in lowered for w in ("card", "payment method", "direct debit")):
        from .identity import is_verified
        if not is_verified(conv):
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="verify_first",
                    text=("Changing the card on file needs a fully verified "
                          "identity first — the account check plus the one-time "
                          "code. Let's do that now."))],
                needs_verification=True,
                citations=_cite(ctx, "payments", "payment method card verified identity"),
                progress=True)
        digits = re.findall(r"(?<!\d)(\d{4})(?!\d)", text)
        if digits:
            last4 = digits[-1]
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="card_update",
                    text=f"I have the new card ending {last4}. Only the last "
                         "four digits are kept — never the full number.")],
                proposed_action=ProposedAction(
                    kind=ActionKind.UPDATE_PAYMENT_METHOD,
                    summary=f"replace the card on file with the one ending {last4}",
                    payload={"last4": last4},
                    basis="Verified customer supplied the new card's last four (POL-PAY-02)."),
                citations=_cite(ctx, "payments", "card replaced last four digits next bill"),
                progress=True)
        return SpecialistResult(
            specialist="account",
            findings=[Finding(
                topic="card_ask",
                text=("Tell me the last four digits of the new card — only "
                      "those four, never the full number."))],
            citations=_cite(ctx, "payments", "card replaced last four digits"),
            progress=True)

    if any(w in lowered for w in ("can't pay", "cannot pay", "can not pay",
                                  "arrangement", "split", "instalment",
                                  "installment", "behind")):
        if acct is None:
            return _needs_identity("account", _IDENTITY_ASK)
        active = payments_book.active_arrangement(
            ctx.store, acct.account_number)
        if active is not None:
            # One arrangement at a time: the answer is its state,
            # never a second schedule written over it.
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="arrangement_active",
                    text=(f"You already have an arrangement running: "
                          f"{payments_book.arrangement_summary(active)}. "
                          "A second one can't start while this one is "
                          "active — the payments go toward these "
                          "instalments first."))],
                citations=_cite(ctx, "payments",
                                "payment arrangement split balance instalments"),
                progress=True)
        balance = acct.balance_due
        if balance >= ctx.policy.arrangement_min_balance:
            installments = payments_book.plan_installments(
                balance, ctx.policy)
            schedule = PaymentArrangement(
                arrangement_id="", account_number=acct.account_number,
                total=balance, installments=installments)
            first = installments[0]
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="arrangement",
                    text=(f"Your balance is {fmt(balance)}. Under the "
                          f"payment arrangement policy you can split "
                          f"it into "
                          f"{payments_book.schedule_text(schedule)}, "
                          f"with no fee. The first instalment falls "
                          f"due on {first.due_date}."))],
                proposed_action=ProposedAction(
                    kind=ActionKind.PAYMENT_ARRANGEMENT,
                    summary=(f"split your {fmt(balance)} balance into "
                             f"{payments_book.schedule_text(schedule)}"),
                    payload={
                        "installments": [
                            {"number": i.number, "amount": str(i.amount),
                             "due_date": i.due_date}
                            for i in installments],
                        "instalments": len(installments),
                        "each": str(first.amount)},
                    amount=balance,
                    basis=("Balance at or over the arrangement "
                           "minimum, no arrangement active "
                           "(POL-PAY-01).")),
                citations=_cite(ctx, "payments", "payment arrangement split balance instalments"),
                progress=True)
        return SpecialistResult(
            specialist="account",
            findings=[Finding(
                topic="arrangement_small",
                text=(f"Your balance is {fmt(balance)}. Arrangements start at "
                      f"a {fmt(ctx.policy.arrangement_min_balance)} balance, "
                      "so this one is due as a single payment "
                      f"by the bill's due date."))],
            citations=_cite(ctx, "payments", "payment arrangement split balance instalments"),
            progress=True)

    if any(w in lowered for w in ("pay my", "pay the", "make a payment",
                                  "payment again", "pay in full",
                                  "pay it off", "retry", "try again")) \
            or state.get("await_retry"):
        state.pop("await_retry", None)
        if acct is None:
            return _needs_identity("account", _IDENTITY_ASK)
        if acct.balance_due <= 0:
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="nothing_due",
                    text="There's nothing due on your account right "
                         "now — the balance is $0.00.")],
                progress=True, resolved=True)
        if block is not None:
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="hard_decline_refusal",
                    text=(payments_book.diagnosis_text(block)
                          + " I won't run that card again — it can't "
                          "go through. Tell me the last four digits "
                          "of a new card and we'll pay with that "
                          "instead."))],
                citations=_cite(ctx, "payments",
                                "declined payment retry new card"),
                progress=True)
        findings = []
        last = payments_book.last_attempt(ctx.store, acct.account_number)
        if last is not None and last.outcome == "declined":
            findings.append(Finding(
                topic="diagnosis",
                text=payments_book.diagnosis_text(last)))
        findings.append(Finding(
            topic="pay",
            text=(f"Your balance is {fmt(acct.balance_due)}. I can "
                  f"take that now from the card ending "
                  f"{acct.payment_method_last4} on your account.")))
        return SpecialistResult(
            specialist="account",
            findings=findings,
            proposed_action=ProposedAction(
                kind=ActionKind.TAKE_PAYMENT,
                summary=(f"pay your balance of {fmt(acct.balance_due)} "
                         f"with the card ending "
                         f"{acct.payment_method_last4}"),
                payload={"amount": str(acct.balance_due),
                         "method_last4": acct.payment_method_last4 or ""},
                amount=acct.balance_due,
                basis="Customer asked to pay; the charge goes to "
                      "the method on file (POL-PAY-03)."),
            citations=_cite(ctx, "payments", "pay balance card on file"),
            must_include=failure_lead,
            progress=True)

    if any(w in lowered for w in ("autopay", "auto pay",
                                  "automatic payment")):
        from .identity import is_verified
        if not is_verified(conv):
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="verify_first",
                    text=("Changing autopay needs a fully verified "
                          "identity first — the account check plus "
                          "the one-time code. Let's do that now."))],
                needs_verification=True,
                citations=_cite(ctx, "payments", "autopay automatic payment"),
                progress=True)
        if acct is None:
            return _needs_identity("account", _IDENTITY_ASK)
        if any(w in lowered for w in ("turn off", "switch off", "cancel",
                                      "disable", "stop autopay")):
            if not acct.autopay_enabled:
                return SpecialistResult(
                    specialist="account",
                    findings=[Finding(
                        topic="autopay_state",
                        text="Autopay is already off on your account.")],
                    progress=True, resolved=True)
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="autopay_off",
                    text=("Autopay is on at the moment — each bill "
                          "is taken from the card ending "
                          f"{acct.payment_method_last4}."))],
                proposed_action=ProposedAction(
                    kind=ActionKind.SET_AUTOPAY,
                    summary="turn autopay off",
                    payload={"enabled": False},
                    basis="Customer asked to stop automatic "
                          "payments (POL-PAY-03)."),
                citations=_cite(ctx, "payments", "autopay automatic payment"),
                must_include=failure_lead,
                progress=True)
        if any(w in lowered for w in ("turn on", "switch on", "set up",
                                      "enable", "activate", "start")):
            if acct.autopay_enabled:
                return SpecialistResult(
                    specialist="account",
                    findings=[Finding(
                        topic="autopay_state",
                        text=("Autopay is already on — each bill is "
                              "taken from the card ending "
                              f"{acct.payment_method_last4}."))],
                    progress=True, resolved=True)
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="autopay_on",
                    text=("With autopay on, each bill is taken "
                          "automatically from the card ending "
                          f"{acct.payment_method_last4} when it "
                          "lands — no need to remember due dates."))],
                proposed_action=ProposedAction(
                    kind=ActionKind.SET_AUTOPAY,
                    summary=(f"turn autopay on, paying each bill "
                             f"from the card ending "
                             f"{acct.payment_method_last4}"),
                    payload={"enabled": True},
                    basis="Customer asked for automatic payments "
                          "from the method on file (POL-PAY-03)."),
                citations=_cite(ctx, "payments", "autopay automatic payment"),
                must_include=failure_lead,
                progress=True)
        state_text = ("on — each bill is taken from the card ending "
                      f"{acct.payment_method_last4}"
                      if acct.autopay_enabled else "off")
        return SpecialistResult(
            specialist="account",
            findings=[Finding(
                topic="autopay_state",
                text=f"Autopay is {state_text} on your account.")],
            must_include=failure_lead,
            progress=True, resolved=True)

    if acct is None:
        return _needs_identity("account", _IDENTITY_ASK)
    bill = ctx.store.get_bill_for_account(conv.account_number)
    due = bill.due if bill else "the date on your bill"
    findings = [Finding(
        topic="balance",
        text=f"Your current balance is {fmt(acct.balance_due)}, due {due}.")]
    active = payments_book.active_arrangement(ctx.store,
                                              acct.account_number)
    if active is not None:
        findings.append(Finding(
            topic="arrangement_state",
            text=f"You have {payments_book.arrangement_summary(active)}."))
    if acct.autopay_enabled:
        findings.append(Finding(
            topic="autopay_state",
            text=("Autopay is on — this balance will be taken from "
                  f"the card ending {acct.payment_method_last4}.")))
    return SpecialistResult(
        specialist="account",
        findings=findings,
        must_include=failure_lead,
        progress=True, resolved=True)


# ---------------------------------------------------------------------------
# Customer care — the catch-all
# ---------------------------------------------------------------------------

# The words of a customer asking how they'll be reached — the
# question the recorded contact preference answers.
_CONTACT_WORDS = ("contact", "write to", "reach me", "hear from",
                  "get in touch", "send me", "letters", "by post")


def _recorded_need_result(account) -> SpecialistResult:
    """The reply for a need just written on the account: what was
    recorded, stated as the instruction it became."""
    if account.preferred_contact:
        line = (f"I've recorded it on your account: from now on we'll "
                f"contact you in {account.preferred_contact}.")
    else:
        line = ("I've recorded it on your account, in your own words, "
                "and we'll follow it at every contact.")
    return SpecialistResult(
        specialist="care",
        findings=[Finding(
            topic="accessibility_recorded",
            text=f"Thank you for telling me. {line}")],
        must_include=[line], progress=True, resolved=True)


def care(conv: Conversation, text: str, ctx: Ctx) -> SpecialistResult:
    lowered = text.lower().strip()
    care_state = conv.specialist_state.setdefault("care", {})
    sig = conv.specialist_state.setdefault("signals", {})

    # A need stated a turn ago, held for want of the account
    # number: if this message names the account, the note lands
    # now, before anything else the message might also be.
    pending_note = care_state.get("pending_accessibility")
    if pending_note:
        account = care_book.record_accessibility(
            ctx.store, conv, pending_note,
            account_number=care_book.account_number_in(text))
        if account is not None:
            care_state.pop("pending_accessibility", None)
            return _recorded_need_result(account)

    # A complaint becomes a record the moment it's recognised —
    # category in code, the customer's words verbatim, the
    # response deadline policy promises — and the reply is the
    # acknowledgment: reference, category, date. A customer
    # chasing the complaint they already made is answered from
    # its record, not recorded a second time.
    category = care_book.complaint_category(text)
    if category is not None:
        open_here = [c for c in ctx.store.list_complaints(
            conversation_id=conv.conversation_id) if c.status == "open"]
        chasing = any(w in lowered for w in (
            "news", "update", "any word", "progress", "happening",
            "chasing", "on my complaint", "about my complaint",
            "still waiting"))
        if open_here and chasing:
            record = open_here[-1]
            if care_book.complaint_sla_state(record, ctx.policy) \
                    == "breached":
                said = (f"Your complaint {record.complaint_id} was due "
                        f"a substantive response by {record.response_due} "
                        f"— that date has passed, so it's been raised "
                        f"with the team as a priority. I'm sorry you're "
                        f"still waiting.")
            else:
                said = (f"Your complaint {record.complaint_id} is with "
                        f"the team — it's due a substantive response by "
                        f"{record.response_due}, and that still stands.")
            return SpecialistResult(
                specialist="care",
                findings=[Finding(topic="complaint_chase", text=said)],
                must_include=[record.complaint_id],
                progress=True, resolved=True)
        record = care_book.open_complaint(
            ctx.store, conv, text, category, ctx.policy)
        due_line = (f"We'll respond in substance by "
                    f"{record.response_due}.")
        ref_line = f"Your complaint reference is {record.complaint_id}."
        words = category.replace("_", " ")
        article = "an" if words[0] in "aeiou" else "a"
        return SpecialistResult(
            specialist="care",
            findings=[Finding(
                topic="complaint",
                text=("I'm sorry you've had this experience — thank you "
                      f"for telling me plainly. {ref_line} I've recorded "
                      f"it as {article} {words} complaint, "
                      f"in your own words, and a person will pick it up "
                      f"from here. {due_line}"))],
            citations=_cite(ctx, "general",
                            "complaint recorded response within five days"),
            must_include=[ref_line, due_line],
            progress=True, resolved=True)

    # A stated accessibility or contact need is recorded on the
    # account. Without an account to write it on, the desk asks
    # for the number and holds the words — the need is never
    # dropped because the paperwork lagged.
    if care_book.accessibility_need(text):
        account = care_book.record_accessibility(ctx.store, conv, text)
        if account is not None:
            return _recorded_need_result(account)
        care_state["pending_accessibility"] = text
        return SpecialistResult(
            specialist="care",
            findings=[Finding(
                topic="accessibility_ask_account",
                text=("Thank you for telling me — I want to record that "
                      "on your account so every contact follows it. "
                      "What's your account number (it starts NB-)?"))],
            progress=True)

    # Compliments and feedback are records too — the business
    # mines them. A bare thank-you stays a pleasantry (below).
    kind = care_book.feedback_kind(text)
    if kind is not None:
        care_book.record_feedback(ctx.store, conv, text, kind)
        if kind == "compliment":
            said = ("Thank you — I've recorded that as a compliment, in "
                    "your own words, and the team will see it.")
        else:
            said = ("Thank you — I've recorded your feedback, in your "
                    "own words, and the team reads it.")
        return SpecialistResult(
            specialist="care",
            findings=[Finding(topic="feedback", text=said)],
            progress=True, resolved=True)

    # A contact preference on the record answers a contact
    # question from the record — honoured, not approximated.
    account = _account(ctx, conv)
    if account is None:
        number = care_book.account_number_in(text)
        if number:
            account = ctx.store.get_account(number)
    if account is not None and account.preferred_contact and \
            any(w in lowered for w in _CONTACT_WORDS):
        line = care_book.contact_preference_line(account)
        return SpecialistResult(
            specialist="care",
            findings=[Finding(topic="contact_preference", text=line)],
            must_include=[line], progress=True, resolved=True)

    if lowered in ("hello", "hi", "hey", "good morning", "good afternoon") or \
            lowered.startswith(("hello", "hi there")) and len(lowered) < 24:
        return SpecialistResult(
            specialist="care",
            findings=[Finding(
                topic="greeting",
                text=("Hello — I can help with your bill, roaming, plan "
                      "changes, faults, and payments. What's happened?"))],
            progress=True)
    if any(w in lowered for w in ("thank", "thanks", "great", "perfect")) and \
            len(lowered) < 40:
        return SpecialistResult(
            specialist="care",
            findings=[Finding(topic="closing",
                              text="You're welcome. Anything else, just ask.")],
            progress=True, resolved=True)
    # The second explanation form (round 11): the signals say the
    # customer did not follow the last explanation of this topic,
    # and the supervisor armed the topic for this turn. Serve the
    # topic's other shape — the summary-first restatement or the
    # worked example — never the first form's words again. The
    # citation is unchanged: the second form says the same policy,
    # differently.
    flag = sig.get("second_form") or ""
    if flag and flag != "bill" \
            and expl_mod.has_second_form(ctx.store, flag) \
            and flag not in sig.get("second_forms_used", []):
        sig["second_form_served"] = flag
        return SpecialistResult(
            specialist="care",
            findings=[Finding(topic="policy_second_form",
                              text=expl_mod.second_form(ctx.store, flag))],
            citations=[flag], progress=True, resolved=True)

    docs = ctx.retriever.search(text, limit=2)
    if docs:
        findings = [Finding(topic="policy", text=d["text"]) for d in docs]
        return SpecialistResult(
            specialist="care", findings=findings,
            citations=[d["doc_id"] for d in docs], progress=True, resolved=True)
    conv.unhandled_intent = text[:120]
    return SpecialistResult(
        specialist="care",
        findings=[Finding(
            topic="unhandled",
            text=("That's not something I have in my guides, and I'd rather "
                  "say so than guess. I've noted it so the team can pick it "
                  "up. If it's about your bill, a plan, roaming or a fault, "
                  "tell me a little more — or ask for a person and I'll "
                  "transfer you."))],
        progress=True)
