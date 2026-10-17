"""The six specialists.

Each is a function over (conversation, customer text, context) that
works with tools — the store, the retriever, the money module — and
returns a SpecialistResult: structured findings the supervisor may
assert, at most one proposed action, and the citations the reply
rests on. Specialists never format the final reply and never
execute anything; execution is actions.py, composition is the
supervisor's, and both re-check the rules at their own moment.

The desks work their happy paths; their deepening — dispute
tiers, the full procedure engine, proration edge cases — is
scheduled in the round map, not improvised here. Rerouting
between desks lives in the supervisor (round 2), not here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

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
from .config import Policy
from .money import fmt, plan_change_net
from .models import (
    ActionKind,
    Conversation,
    Finding,
    ProposedAction,
    SpecialistResult,
)
from .routing import classify_step_outcome


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


def _evidenced_total(bill) -> tuple[Decimal, list[str]]:
    """Charges the bill itself proves wrong, as (total, evidence in
    words) — the quick scan the explanation and the goodwill gate
    use. The per-line decisions live in billing_book.assess_dispute."""
    total = Decimal("0")
    items: list[str] = []
    if bill is None:
        return total, items
    for charge in bill.charges:
        amount, evidence = line_evidence(charge)
        if amount > 0:
            total += amount
            items.append(evidence)
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
    evidence_total, evidence_items = _evidenced_total(bill)

    # A dispute with identified lines becomes a case, decided line
    # by line: evidenced lines credit, named-but-clean lines
    # decline with reasons, an over-cap total escalates.
    if dispute and bill is not None:
        assessment = assess_dispute(bill, text, ctx.policy)
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
    findings: list[Finding] = []
    shock = bill_shock(bill, previous_bills, ctx.policy)
    if shock is not None:
        findings.append(Finding(topic="bill_shock", text=_shock_text(shock)))
    findings.append(Finding(
        topic="bill_summary",
        text=(f"Your bill issued {bill.issued} for {bill.cycle_start} to "
              f"{bill.cycle_end} totals {fmt(bill.total)} and is due {bill.due}. "
              "Here is what makes it up:")))
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


def roaming(conv: Conversation, text: str, ctx: Ctx) -> SpecialistResult:
    lowered = text.lower()
    wants_pass = any(w in lowered for w in ("buy", "add", "activate", "get", "purchase")) \
        and "pass" in lowered
    account = _account(ctx, conv)

    if wants_pass:
        if account is None:
            return _needs_identity("roaming", _IDENTITY_ASK)
        offer = ctx.store.get_offer("OFF-ROAM10")
        findings: list[Finding] = []
        must_include: list[str] = []
        used = account.roaming_gb_used
        if used >= ctx.policy.roaming_fair_use_gb:
            must_include.append(
                f"Fair-use warning: you have used {used} GB roaming this trip, past "
                f"the {ctx.policy.roaming_fair_use_gb} GB fair-use ceiling, so speeds "
                "are already reduced and further data is charged per GB.")
            findings.append(Finding(
                topic="fair_use",
                text="Before anything chargeable, you should have that warning in full."))
        if account.plan_id not in offer.eligible_plan_ids:
            findings.append(Finding(
                topic="not_eligible",
                text=f"The Roam Pass isn't available on your current plan."))
            return SpecialistResult(specialist="roaming", findings=findings,
                                    citations=_cite(ctx, "roaming", "roam pass countries price"),
                                    must_include=must_include, progress=True)
        findings.append(Finding(
            topic="roam_pass",
            text=(f"{offer.name}: {fmt(offer.one_off_price)} one-off, and it "
                  f"covers {offer.duration_days} days. {offer.blurb}")))
        return SpecialistResult(
            specialist="roaming", findings=findings,
            proposed_action=ProposedAction(
                kind=ActionKind.ACTIVATE_OFFER,
                summary=f"activate the {offer.name} for {fmt(offer.one_off_price)}",
                payload={"offer_id": offer.offer_id},
                amount=offer.one_off_price,
                basis="Customer requested the pass in conversation (POL-ROAM-01)."),
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
        findings.append(Finding(
            topic="roaming_status",
            text=(f"You're currently roaming in {account.roaming_country} and have "
                  f"used {account.roaming_gb_used} GB this trip. The fair-use "
                  f"ceiling is {ctx.policy.roaming_fair_use_gb} GB.")))
    docs = ctx.retriever.search(text, limit=2, section="roaming")
    for doc in docs:
        findings.append(Finding(topic="policy", text=doc["text"]))
    if not findings:
        findings.append(Finding(
            topic="roaming_general",
            text=("Roaming outside a pass is charged per GB at the destination "
                  "rate. Tell me where you're heading and I'll work out the "
                  "cheapest way to stay connected.")))
    return SpecialistResult(
        specialist="roaming", findings=findings,
        citations=[d["doc_id"] for d in docs], progress=True, resolved=bool(docs))


# ---------------------------------------------------------------------------
# Offers & plan changes
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
    target = _target_plan(text, ctx)
    wants_change = target is not None or any(
        w in text.lower() for w in ("upgrade", "change", "switch"))
    if wants_change and target is not None:
        if not conv.account_number:
            return _needs_identity("offers", _IDENTITY_ASK)
        account = _account(ctx, conv)
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
        basis = f"Standard plan change under POL-PLAN-01."
        if (loyalty and loyalty.target_plan_id == target.plan_id
                and account.plan_id in loyalty.eligible_plan_ids
                and customer and customer.tenure_months >= 12):
            new_price = loyalty.monthly_price
            basis = (f"Loyalty upgrade OFF-UPG-LOYAL: {fmt(new_price)}/month "
                     f"instead of {fmt(target.monthly_price)} (tenure "
                     f"{customer.tenure_months} months).")
        days_remaining = account.cycle_days - account.cycle_day
        proration = plan_change_net(current.monthly_price, new_price,
                                    days_remaining, account.cycle_days)
        net = proration["net"]
        net_line = (
            f"Net adjustment on your next bill: {fmt(net)} due."
            if net > 0 else
            f"Net adjustment on your next bill: a {fmt(-net)} credit."
            if net < 0 else "Net adjustment on your next bill: $0.00.")
        findings = [Finding(
            topic="plan_change",
            text=(f"Switching from {current.name} ({fmt(current.monthly_price)}/month) "
                  f"to {target.name} ({fmt(new_price)}/month) now, with "
                  f"{days_remaining} days left in this cycle, works out as follows."))]
        must_include = [
            f"Credit for the unused part of {current.name}: {fmt(proration['credit_for_old_plan'])}",
            f"Charge for {target.name} for the remaining {days_remaining} days: {fmt(proration['charge_for_new_plan'])}",
            net_line,
            f"From the next cycle, {target.name} is {fmt(new_price)}/month.",
        ]
        return SpecialistResult(
            specialist="offers", findings=findings,
            proposed_action=ProposedAction(
                kind=ActionKind.CHANGE_PLAN,
                summary=(f"change your plan to {target.name} at "
                         f"{fmt(new_price)}/month, effective now"),
                payload={"new_plan_id": target.plan_id,
                         "new_price": str(new_price), "net": str(net),
                         "credit": str(proration["credit_for_old_plan"]),
                         "charge": str(proration["charge_for_new_plan"])},
                amount=net, basis=basis),
            citations=_cite(ctx, "plans", "plan change prorated credit charge confirm"),
            must_include=must_include, progress=True)

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

_AUTO_CHECKS = {"coverage_ok", "allowance_ok"}  # the agent tests these itself

# Checks that mean "the symptom is gone" when they pass, as opposed
# to set-up steps (switching Wi-Fi calling on) that pass and advance.
_RESOLUTION_CHECKS = {"signal_restored", "speed_restored", "call_connected"}


def _present_step(proc, index: int, ctx: Ctx, conv: Conversation,
                  account) -> tuple[int, list[Finding]]:
    """Skip agent-side checks (running the tool), land on the next
    customer step, and return (index, findings)."""
    findings: list[Finding] = []
    while index < len(proc.steps) and proc.steps[index].check in _AUTO_CHECKS:
        step = proc.steps[index]
        if step.check == "coverage_ok":
            findings.append(Finding(
                topic="check",
                text="I've checked coverage at your postcode: no outage is "
                     "reported there, so let's work through the device side."))
        elif step.check == "allowance_ok":
            findings.append(Finding(
                topic="check",
                text="I've checked your data allowance: that isn't what's "
                     "slowing you down."))
        index += 1
    if index < len(proc.steps):
        findings.append(Finding(
            topic="step", text=proc.steps[index].instruction))
    return index, findings


def technical(conv: Conversation, text: str, ctx: Ctx) -> SpecialistResult:
    state = conv.specialist_state.setdefault("technical", {})
    lowered = text.lower()

    if state.get("want_callback") and conv.account_number:
        proc_id = state.pop("want_callback")
        return SpecialistResult(
            specialist="technical",
            findings=[Finding(
                topic="steps_exhausted",
                text="Thank you — identity confirmed. As promised, this one "
                     "goes to a technician rather than more guesswork.")],
            proposed_action=ProposedAction(
                kind=ActionKind.BOOK_CALLBACK,
                summary="book a technician callback within one working day",
                payload={"reason": proc_id},
                basis="Guided procedure exhausted without resolution (POL-TECH-01)."),
            citations=_cite(ctx, "technical", "troubleshooting steps callback technician"),
            progress=True)

    if state.get("procedure_id"):
        proc = ctx.store.get_procedure(state["procedure_id"])
        index = state.get("step_index", 0)
        step = proc.steps[index]
        outcome = classify_step_outcome(text)
        if outcome == "unclear":
            return SpecialistResult(
                specialist="technical",
                findings=[Finding(topic="step_repeat",
                                  text=f"Just to be sure we're on the same step: {step.instruction} "
                                       "Tell me yes if it worked, or no if nothing changed.")],
                progress=False)
        if outcome == "success":
            if step.check in _RESOLUTION_CHECKS:
                # The step's check IS the symptom disappearing —
                # success here ends the procedure, it doesn't advance it.
                state.pop("procedure_id", None)
                state.pop("step_index", None)
                return SpecialistResult(
                    specialist="technical",
                    findings=[Finding(
                        topic="resolved",
                        text="That's the fix — your service should stay up now. "
                             "If it drops again, quote this conversation and "
                             "we'll pick up right here.")],
                    progress=True, resolved=True)
            nxt, findings = _present_step(proc, index + 1, ctx, conv,
                                          _account(ctx, conv))
            if nxt >= len(proc.steps):
                state.pop("procedure_id", None)
                state.pop("step_index", None)
                findings.append(Finding(
                    topic="resolved",
                    text="That's the fix — your service should stay up now. "
                         "If it drops again, quote this conversation and we'll "
                         "pick up right here."))
                return SpecialistResult(specialist="technical", findings=findings,
                                        progress=True, resolved=True)
            state["step_index"] = nxt
            return SpecialistResult(specialist="technical", findings=findings,
                                    progress=True)
        # outcome == fail
        if step.on_fail == "next" and index + 1 < len(proc.steps):
            nxt, findings = _present_step(proc, index + 1, ctx, conv,
                                          _account(ctx, conv))
            state["step_index"] = nxt
            findings.insert(0, Finding(
                topic="step_failed",
                text="Understood — that step didn't do it. Next one:"))
            return SpecialistResult(specialist="technical", findings=findings,
                                    progress=True)
        # on_fail == callback (or the steps ran out)
        state.pop("procedure_id", None)
        state.pop("step_index", None)
        if not conv.account_number:
            result = _needs_identity(
                "technical",
                "We've worked through the guided steps without restoring "
                "service. I'd like to book a technician callback — " + _IDENTITY_ASK)
            result.specialist = "technical"
            conv.specialist_state["technical"]["want_callback"] = proc.procedure_id
            return result
        return SpecialistResult(
            specialist="technical",
            findings=[Finding(
                topic="steps_exhausted",
                text=(f"We've worked through the {proc.title} steps and the "
                      "problem is still there, so this needs a technician."))],
            proposed_action=ProposedAction(
                kind=ActionKind.BOOK_CALLBACK,
                summary="book a technician callback within one working day",
                payload={"reason": proc.procedure_id},
                basis="Guided procedure exhausted without resolution (POL-TECH-01)."),
            citations=_cite(ctx, "technical", "troubleshooting steps callback technician"),
            progress=True)

    # Start a procedure from the symptom.
    symptom = None
    if "no signal" in lowered or "no service" in lowered or "signal" in lowered:
        symptom = "no_signal"
    elif "slow" in lowered:
        symptom = "slow_data"
    elif "wifi calling" in lowered or "wi-fi calling" in lowered:
        symptom = "wifi_calling"
    proc = next((p for p in ctx.store.list_procedures() if p.symptom == symptom), None)
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
    nxt, findings = _present_step(proc, 0, ctx, conv, _account(ctx, conv))
    if nxt >= len(proc.steps):
        return SpecialistResult(specialist="technical", findings=findings,
                                progress=True, resolved=True)
    state["procedure_id"] = proc.procedure_id
    state["step_index"] = nxt
    findings.insert(0, Finding(
        topic="procedure_start",
        text=f"Let's work through the {proc.title} steps in order — they fix "
             "most cases like this."))
    return SpecialistResult(specialist="technical", findings=findings, progress=True)


# ---------------------------------------------------------------------------
# Account & payments
# ---------------------------------------------------------------------------


def account(conv: Conversation, text: str, ctx: Ctx) -> SpecialistResult:
    lowered = text.lower()
    acct = _account(ctx, conv)

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
        balance = acct.balance_due
        if balance >= Decimal("100"):
            each = (balance / 3).quantize(Decimal("0.01"))
            return SpecialistResult(
                specialist="account",
                findings=[Finding(
                    topic="arrangement",
                    text=(f"Your balance is {fmt(balance)}. Under the payment "
                          f"arrangement policy you can split it into 3 equal "
                          f"monthly instalments of {fmt(each)}, with no fee."))],
                proposed_action=ProposedAction(
                    kind=ActionKind.PAYMENT_ARRANGEMENT,
                    summary=f"split your {fmt(balance)} balance into 3 monthly instalments of {fmt(each)}",
                    payload={"instalments": 3, "each": str(each)},
                    amount=balance,
                    basis="Balance ≥ $100, account not in collections (POL-PAY-01)."),
                citations=_cite(ctx, "payments", "payment arrangement split balance instalments"),
                progress=True)
        return SpecialistResult(
            specialist="account",
            findings=[Finding(
                topic="arrangement_small",
                text=(f"Your balance is {fmt(balance)}. Arrangements start at "
                      "a $100 balance, so this one is due as a single payment "
                      f"by the bill's due date."))],
            citations=_cite(ctx, "payments", "payment arrangement split balance instalments"),
            progress=True)

    if acct is None:
        return _needs_identity("account", _IDENTITY_ASK)
    bill = ctx.store.get_bill_for_account(conv.account_number)
    due = bill.due if bill else "the date on your bill"
    return SpecialistResult(
        specialist="account",
        findings=[Finding(
            topic="balance",
            text=f"Your current balance is {fmt(acct.balance_due)}, due {due}.")],
        progress=True, resolved=True)


# ---------------------------------------------------------------------------
# Customer care — the catch-all
# ---------------------------------------------------------------------------


def care(conv: Conversation, text: str, ctx: Ctx) -> SpecialistResult:
    lowered = text.lower().strip()
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
