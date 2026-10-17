"""The Northbeam Mobile book, generated in code.

Every figure the evals assert on is built here, once, so the gold
outcomes are true by construction: Amara Osei's bill really does
carry the same device-protection charge twice (the second marked
`duplicate_of` the first), Daniel Reyes really has used 31 GB
roaming against a 25 GB fair-use ceiling, Marcus Webb's add-on fee
really is marked as charged after cancellation. The specialists
reason over these records with tools; nothing is staged inside a
prompt.
"""

from __future__ import annotations

from decimal import Decimal

from .documents import embed
from .models import (
    Account,
    Bill,
    BillCharge,
    Customer,
    Offer,
    Plan,
    PolicyDocument,
    Procedure,
    ProcedureStep,
)

D = Decimal


def build_plans() -> list[Plan]:
    return [
        Plan(plan_id="PLAN-BAS", name="Basic Talk & Text", monthly_price=D("25.00"),
             data_gb=2, notes="Calls, texts and a little data."),
        Plan(plan_id="PLAN-ESS", name="Essential 20GB", monthly_price=D("45.00"),
             data_gb=20, notes="Everyday plan for one line."),
        Plan(plan_id="PLAN-UNL", name="Unlimited Max", monthly_price=D("75.00"),
             data_gb=None, notes="Unlimited data, 4K streaming, hotspot included."),
        Plan(plan_id="PLAN-FAM", name="Family Share 100GB", monthly_price=D("110.00"),
             data_gb=100, notes="Shared pool for up to four lines."),
    ]


def build_offers() -> list[Offer]:
    return [
        Offer(offer_id="OFF-ROAM10", name="Roam Pass — 10 days", kind="roaming_pass",
              one_off_price=D("30.00"), duration_days=10,
              eligible_plan_ids=["PLAN-BAS", "PLAN-ESS", "PLAN-UNL", "PLAN-FAM"],
              blurb="Unlimited calls and texts plus 1 GB of data a day in 40+ countries."),
        Offer(offer_id="OFF-UPG-LOYAL", name="Loyalty upgrade to Unlimited Max",
              kind="plan_upgrade", monthly_price=D("60.00"), target_plan_id="PLAN-UNL",
              eligible_plan_ids=["PLAN-ESS", "PLAN-BAS"],
              blurb="Unlimited Max at $60/month instead of $75, for customers with us 12+ months."),
        Offer(offer_id="OFF-DATA10", name="Extra data — 10 GB", kind="data_topup",
              one_off_price=D("10.00"),
              eligible_plan_ids=["PLAN-BAS", "PLAN-ESS", "PLAN-FAM"],
              blurb="A one-off 10 GB top-up that lasts to the end of the cycle."),
    ]


def build_customers() -> list[Customer]:
    rows = [
        ("CUST-001", "Amara Osei", "NB-1001", "02134", 26, False),
        ("CUST-002", "Daniel Reyes", "NB-1002", "90210", 8, False),
        ("CUST-003", "Priya Nair", "NB-1003", "10001", 41, False),
        ("CUST-004", "Tom Beckett", "NB-1004", "60614", 3, False),
        ("CUST-005", "Grace Liu", "NB-1005", "94107", 15, False),
        ("CUST-006", "Marcus Webb", "NB-1006", "30301", 19, False),
        ("CUST-007", "Elena Petrova", "NB-1007", "02139", 6, False),
        ("CUST-008", "Sam Whitfield", "NB-1008", "75201", 12, True),
        ("CUST-009", "June Park", "NB-1009", "98101", 30, False),
        ("CUST-010", "Oliver Grant", "NB-1010", "33101", 5, False),
        ("CUST-011", "Ruth Callaghan", "NB-1011", "55401", 34, False),
        ("CUST-012", "Felix Adeyemi", "NB-1012", "80202", 17, False),
    ]
    return [
        Customer(customer_id=cid, full_name=name, account_number=acct,
                 postcode=pc, email=f"{name.split()[0].lower()}@example.com",
                 tenure_months=tenure, vulnerable=vuln)
        for cid, name, acct, pc, tenure, vuln in rows
    ]


def build_accounts() -> list[Account]:
    return [
        Account(account_number="NB-1001", customer_id="CUST-001", plan_id="PLAN-ESS",
                balance_due=D("94.35"), payment_method_last4="4417", cycle_day=19),
        Account(account_number="NB-1002", customer_id="CUST-002", plan_id="PLAN-UNL",
                balance_due=D("152.52"), payment_method_last4="9034", cycle_day=19,
                roaming_country="Spain", roaming_gb_used=D("31")),
        Account(account_number="NB-1003", customer_id="CUST-003", plan_id="PLAN-FAM",
                balance_due=D("118.60"), payment_method_last4="2210", cycle_day=11),
        Account(account_number="NB-1004", customer_id="CUST-004", plan_id="PLAN-ESS",
                balance_due=D("45.00"), payment_method_last4="7781", cycle_day=7),
        Account(account_number="NB-1005", customer_id="CUST-005", plan_id="PLAN-ESS",
                balance_due=D("76.77"), payment_method_last4="5150", cycle_day=14),
        Account(account_number="NB-1006", customer_id="CUST-006", plan_id="PLAN-ESS",
                balance_due=D("61.10"), payment_method_last4="6602", cycle_day=19),
        Account(account_number="NB-1007", customer_id="CUST-007", plan_id="PLAN-ESS",
                balance_due=D("45.00"), payment_method_last4="3098", cycle_day=5),
        Account(account_number="NB-1008", customer_id="CUST-008", plan_id="PLAN-BAS",
                balance_due=D("25.00"), payment_method_last4="8845", cycle_day=9),
        Account(account_number="NB-1009", customer_id="CUST-009", plan_id="PLAN-UNL",
                balance_due=D("75.00"), payment_method_last4="1173", cycle_day=22),
        Account(account_number="NB-1010", customer_id="CUST-010", plan_id="PLAN-ESS",
                balance_due=D("212.40"), payment_method_last4="4029", cycle_day=19),
        Account(account_number="NB-1011", customer_id="CUST-011", plan_id="PLAN-ESS",
                balance_due=D("72.61"), payment_method_last4="7305", cycle_day=19),
        Account(account_number="NB-1012", customer_id="CUST-012", plan_id="PLAN-ESS",
                balance_due=D("130.38"), payment_method_last4="2214", cycle_day=19),
    ]


_CURRENT_CYCLE = ("2026-09-12", "2026-10-11", "2026-10-12", "2026-11-02")
_PREVIOUS_CYCLE = ("2026-08-13", "2026-09-11", "2026-09-12", "2026-10-03")


def _bill(bill_id: str, acct: str, charges: list[BillCharge],
          cycle: tuple[str, str, str, str] = _CURRENT_CYCLE) -> Bill:
    start, end, issued, due = cycle
    return Bill(bill_id=bill_id, account_number=acct, cycle_start=start,
                cycle_end=end, issued=issued, due=due, charges=charges)


def build_bills() -> list[Bill]:
    return [
        _bill("BILL-1001", "NB-1001", [
            BillCharge(charge_id="CH-1001-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1001-2", kind="one_off",
                       description="Device protection — monthly", amount=D("9.99")),
            BillCharge(charge_id="CH-1001-3", kind="one_off",
                       description="Device protection — monthly", amount=D("9.99"),
                       duplicate_of="CH-1001-2"),
            BillCharge(charge_id="CH-1001-4", kind="device",
                       description="Handset instalment 14 of 24", amount=D("22.50")),
            BillCharge(charge_id="CH-1001-5", kind="tax",
                       description="Taxes and regulatory fees", amount=D("6.87")),
        ]),
        _bill("BILL-1002", "NB-1002", [
            BillCharge(charge_id="CH-1002-1", kind="plan",
                       description="Unlimited Max — monthly plan", amount=D("75.00")),
            BillCharge(charge_id="CH-1002-2", kind="roaming",
                       description="Roaming data — Spain (pay per GB)", amount=D("68.40")),
            BillCharge(charge_id="CH-1002-3", kind="tax",
                       description="Taxes and regulatory fees", amount=D("9.12")),
        ]),
        _bill("BILL-1006", "NB-1006", [
            BillCharge(charge_id="CH-1006-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1006-2", kind="fee",
                       description="Premium voicemail add-on", amount=D("12.00"),
                       after_cancellation=True),
            BillCharge(charge_id="CH-1006-3", kind="tax",
                       description="Taxes and regulatory fees", amount=D("4.10")),
        ]),
        _bill("BILL-1010", "NB-1010", [
            BillCharge(charge_id="CH-1010-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1010-2", kind="device",
                       description="Handset instalment 9 of 24", amount=D("18.75")),
            BillCharge(charge_id="CH-1010-3", kind="fee",
                       description="Late payment fee", amount=D("8.00")),
            BillCharge(charge_id="CH-1010-4", kind="tax",
                       description="Taxes and regulatory fees", amount=D("5.02")),
        ]),
        _bill("BILL-1011", "NB-1011", [
            BillCharge(charge_id="CH-1011-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1011-2", kind="device",
                       description="Handset instalment 6 of 24", amount=D("22.50")),
            BillCharge(charge_id="CH-1011-3", kind="tax",
                       description="Taxes and regulatory fees", amount=D("5.11")),
        ]),
        _bill("BILL-1012", "NB-1012", [
            BillCharge(charge_id="CH-1012-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1012-2", kind="roaming",
                       description="Roaming data — Greece (pay per GB)",
                       amount=D("78.00"), expected_amount=D("20.00")),
            BillCharge(charge_id="CH-1012-3", kind="tax",
                       description="Taxes and regulatory fees", amount=D("7.38")),
        ]),
        # The previous cycle for every billed account — explanations
        # compare against these, and bill shock is measured against
        # the trailing average they form.
        _bill("BILL-1001-PREV", "NB-1001", [
            BillCharge(charge_id="CH-1001P-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1001P-2", kind="one_off",
                       description="Device protection — monthly", amount=D("9.99")),
            BillCharge(charge_id="CH-1001P-3", kind="device",
                       description="Handset instalment 13 of 24", amount=D("22.50")),
            BillCharge(charge_id="CH-1001P-4", kind="tax",
                       description="Taxes and regulatory fees", amount=D("6.86")),
        ], cycle=_PREVIOUS_CYCLE),
        _bill("BILL-1002-PREV", "NB-1002", [
            BillCharge(charge_id="CH-1002P-1", kind="plan",
                       description="Unlimited Max — monthly plan", amount=D("75.00")),
            BillCharge(charge_id="CH-1002P-2", kind="tax",
                       description="Taxes and regulatory fees", amount=D("9.12")),
        ], cycle=_PREVIOUS_CYCLE),
        _bill("BILL-1006-PREV", "NB-1006", [
            BillCharge(charge_id="CH-1006P-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1006P-2", kind="tax",
                       description="Taxes and regulatory fees", amount=D("4.10")),
        ], cycle=_PREVIOUS_CYCLE),
        _bill("BILL-1010-PREV", "NB-1010", [
            BillCharge(charge_id="CH-1010P-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1010P-2", kind="device",
                       description="Handset instalment 8 of 24", amount=D("18.75")),
            BillCharge(charge_id="CH-1010P-3", kind="tax",
                       description="Taxes and regulatory fees", amount=D("5.02")),
        ], cycle=_PREVIOUS_CYCLE),
        _bill("BILL-1011-PREV", "NB-1011", [
            BillCharge(charge_id="CH-1011P-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1011P-2", kind="device",
                       description="Handset instalment 5 of 24", amount=D("22.50")),
            BillCharge(charge_id="CH-1011P-3", kind="tax",
                       description="Taxes and regulatory fees", amount=D("5.11")),
        ], cycle=_PREVIOUS_CYCLE),
        _bill("BILL-1012-PREV", "NB-1012", [
            BillCharge(charge_id="CH-1012P-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1012P-2", kind="tax",
                       description="Taxes and regulatory fees", amount=D("3.71")),
        ], cycle=_PREVIOUS_CYCLE),
    ]


def build_policies() -> list[PolicyDocument]:
    rows = [
        ("POL-BILL-01", "Billing disputes", "billing",
         "When a charge on a bill is wrong — a duplicate charge, a charge for a "
         "service after it was cancelled, or a charge at the wrong rate — the "
         "incorrect amount is credited in full once the evidence is on the bill "
         "itself. Each disputed charge is decided on its own evidence, and the "
         "dispute is recorded as a case with the decision for every line. "
         "Evidenced dispute credits are capped at $50 per bill; above that the "
         "case goes to the billing team with its record intact. The customer "
         "confirms the credit before it is applied."),
        ("POL-BILL-02", "Goodwill credits", "billing",
         "When the bill is correct but the customer has had a poor experience, "
         "the agent may offer a goodwill credit of up to $25. The credit needs "
         "the customer's agreement and is applied to the next bill. Goodwill "
         "credits are offered once per conversation, and no more than $50 of "
         "goodwill may be credited to an account in any 90-day period — past "
         "that, goodwill is refused, though an incorrect charge is still "
         "credited in full under the dispute policy."),
        ("POL-BILL-03", "Reading your bill", "billing",
         "Every bill is itemised: the plan charge, device instalments, one-off "
         "charges, roaming, and taxes and regulatory fees are separate lines. "
         "The total is the sum of the lines. Due dates fall 21 days after the "
         "bill is issued."),
        ("POL-ROAM-01", "Roaming and the Roam Pass", "roaming",
         "Outside a pass, roaming data is charged per GB at the destination "
         "rate. The Roam Pass costs $30 for 10 days and includes unlimited "
         "calls and texts plus 1 GB of data per day in more than 40 countries. "
         "A pass can be activated at any point in a trip and starts when the "
         "customer confirms it."),
        ("POL-ROAM-02", "Roaming fair use", "roaming",
         "Roaming fair use is 25 GB per trip. Beyond 25 GB, speeds reduce and "
         "additional data is charged per GB. The agent must tell a customer "
         "who is near or past the fair-use ceiling before recommending any "
         "further chargeable roaming product."),
        ("POL-PLAN-01", "Changing your plan", "plans",
         "A plan change can start immediately or at the next cycle. An "
         "immediate change is prorated: the customer is credited for the "
         "unused part of the old plan and charged for the remaining part of "
         "the new plan, to the day. The new monthly price applies in full "
         "from the next cycle. The customer confirms the change, and the "
         "prorated figures, before anything is applied."),
        ("POL-ID-01", "Verifying your identity", "identity",
         "Before any change to an account or its money, identity is verified "
         "with three things: the account number, the postcode on the account, "
         "and a one-time code sent to the contact on file. Three failed code "
         "attempts lock verification for the conversation and the customer "
         "is offered a human agent."),
        ("POL-PAY-01", "Payment arrangements", "payments",
         "A customer who cannot pay in full may split a balance of $100 or "
         "more into up to three equal monthly instalments, with no fee, as "
         "long as the account is not already in collections. The arrangement "
         "is confirmed by the customer and recorded on the account."),
        ("POL-PAY-02", "Payment methods", "payments",
         "The card on file can be replaced once identity is verified. Only "
         "the last four digits are ever shown or stored in the conversation. "
         "A replacement card takes effect for the next bill."),
        ("POL-ESC-01", "Speaking to a human", "escalation",
         "A customer can always reach a human agent. The agent will first "
         "offer to solve the problem itself; if the customer asks again, or "
         "the problem cannot be solved in chat, the conversation transfers "
         "with a summary so the customer does not repeat themselves."),
        ("POL-CARE-01", "Stores and opening hours", "general",
         "Northbeam stores are open Monday to Saturday, 9am to 6pm local "
         "time. The support line is staffed 24 hours a day. Store staff can "
         "help with SIM swaps and device setup in person."),
        ("POL-TECH-01", "Fixing connection problems", "technical",
         "Connection problems are worked through in a fixed order: confirm "
         "coverage, restart the device, reseat the SIM, then reset network "
         "settings. Each step is checked before moving on. If the steps do "
         "not restore service, a technician callback is booked."),
    ]
    return [PolicyDocument(doc_id=d, title=t, section=s, text=x) for d, t, s, x in rows]


def build_procedures() -> list[Procedure]:
    return [
        Procedure(procedure_id="PROC-NOSIGNAL", title="No signal",
                  symptom="no_signal",
                  steps=[
                      ProcedureStep(step_id="NS-1",
                                    instruction="Check the coverage at your postcode — one moment while I look.",
                                    check="coverage_ok", on_fail="escalate"),
                      ProcedureStep(step_id="NS-2",
                                    instruction="Restart your phone and tell me whether the signal bars come back.",
                                    check="signal_restored", on_fail="next"),
                      ProcedureStep(step_id="NS-3",
                                    instruction="Power off, take the SIM out, reseat it firmly, and power back on. Any signal now?",
                                    check="signal_restored", on_fail="next"),
                      ProcedureStep(step_id="NS-4",
                                    instruction="Reset network settings (Settings → General → Reset). This forgets saved Wi-Fi passwords. Did the signal return?",
                                    check="signal_restored", on_fail="callback"),
                  ]),
        Procedure(procedure_id="PROC-SLOWDATA", title="Slow data",
                  symptom="slow_data",
                  steps=[
                      ProcedureStep(step_id="SD-1",
                                    instruction="Let me check how much of your data allowance is left this cycle.",
                                    check="allowance_ok", on_fail="next"),
                      ProcedureStep(step_id="SD-2",
                                    instruction="Turn airplane mode on for 30 seconds, then off, and run a quick page load. Faster now?",
                                    check="speed_restored", on_fail="next"),
                      ProcedureStep(step_id="SD-3",
                                    instruction="Move near a window or outside if you can — building materials block signal. Any improvement?",
                                    check="speed_restored", on_fail="callback"),
                  ]),
        Procedure(procedure_id="PROC-WIFICALL", title="Wi-Fi calling setup",
                  symptom="wifi_calling",
                  steps=[
                      ProcedureStep(step_id="WC-1",
                                    instruction="Open Settings → Phone → Wi-Fi Calling and switch it on; confirm your emergency address if asked.",
                                    check="wifi_calling_on", on_fail="next"),
                      ProcedureStep(step_id="WC-2",
                                    instruction="Connect to your Wi-Fi network and place a short test call. Did it connect over Wi-Fi?",
                                    check="call_connected", on_fail="callback"),
                  ]),
    ]


def _seed_credit_history(store) -> None:
    """Executed credits from before 'now' — the trailing history the
    goodwill policy reads. Ruth Callaghan (NB-1011) has already had
    $50 of goodwill inside the window, the full rolling allowance;
    the dates are pinned relative to the shared clock so the window
    means the same thing whenever the book is seeded."""
    from datetime import timedelta

    from .clock import today
    from .models import ActionKind, ActionRecord

    for i, days_ago in enumerate((52, 18), start=1):
        store.save_action(ActionRecord(
            action_id=f"ACT-HIST-1011-{i}",
            conversation_id="CONV-HIST-1011",
            kind=ActionKind.APPLY_CREDIT,
            summary="apply a goodwill credit of $25.00",
            payload={"credit_class": "goodwill"},
            amount=D("25.00"),
            basis="Goodwill under POL-BILL-02; bill itself is correct.",
            status="executed",
            account_number="NB-1011",
            created_date=(today() - timedelta(days=days_ago)).isoformat(),
        ))


def seed_store(store) -> dict:
    """Write the whole book into a store and embed the policy corpus.
    Returns counts for readiness/demo output."""
    plans = build_plans()
    offers = build_offers()
    customers = build_customers()
    accounts = build_accounts()
    bills = build_bills()
    policies = build_policies()
    procedures = build_procedures()
    for plan in plans:
        store.save_plan(plan)
    for offer in offers:
        store.save_offer(offer)
    for customer in customers:
        store.save_customer(customer)
    for account in accounts:
        store.save_account(account)
    for bill in bills:
        store.save_bill(bill)
    for doc in policies:
        store.save_policy(doc)
        store.set_embedding(doc.doc_id, embed(f"{doc.title}. {doc.text}"))
    for proc in procedures:
        store.save_procedure(proc)
    _seed_credit_history(store)
    return {
        "plans": len(plans), "offers": len(offers), "customers": len(customers),
        "accounts": len(accounts), "bills": len(bills),
        "policies": len(policies), "procedures": len(procedures),
    }
