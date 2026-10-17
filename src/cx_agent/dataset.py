"""The Northbeam Mobile book, generated in code.

Every figure the evals assert on is built here, once, so the gold
outcomes are true by construction: Amara Osei's bill really does
carry the same device-protection charge twice (the second marked
`duplicate_of` the first), Daniel Reyes really has used 31 GB
roaming against a 25 GB fair-use ceiling, Marcus Webb's add-on fee
really is marked as charged after cancellation. The specialists
reason over these records with tools; nothing is staged inside a
prompt.

Round 6 adds the roaming book's facts: a destination rate table,
a usage profile on every account, Nadia Hassan past the fair-use
ceiling in Greece with no warning on record anywhere, and Chris
Novak's Roam Pass — activated on the record before his Italy
trip, so his roaming charge is covered by it.

Round 7 adds the payments book's facts: Marta Nowak's payment
arrangement, written before 'now' with its first instalment
already past due and unpaid — broken by derivation whenever the
record is read, which is what her next contact runs into.
"""

from __future__ import annotations

from decimal import Decimal

from .documents import embed
from .prompts import POLICY_TEXTS, seed_registry
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
    RoamingRate,
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
              included_gb_per_day=D("1"),
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


def build_roaming_rates() -> list[RoamingRate]:
    """Pay-as-you-go data per GB, by destination. Trip quotes,
    bill-shock projections and fair-use overage costs are all
    priced from this table and nowhere else."""
    rows = [
        ("Spain", "8.50"), ("Italy", "8.50"), ("France", "8.50"),
        ("Portugal", "8.50"), ("Germany", "8.50"), ("Ireland", "8.50"),
        ("Netherlands", "8.50"), ("United Kingdom", "8.50"),
        ("Canada", "9.00"), ("Greece", "10.00"), ("Mexico", "11.00"),
        ("Australia", "12.00"), ("Brazil", "13.00"), ("Japan", "14.00"),
        ("Thailand", "16.00"), ("United Arab Emirates", "22.00"),
    ]
    return [RoamingRate(destination=name, per_gb=D(rate))
            for name, rate in rows]


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
        ("CUST-013", "Nadia Hassan", "NB-1013", "10025", 14, False),
        ("CUST-014", "Chris Novak", "NB-1014", "60607", 9, False),
        ("CUST-015", "Marta Nowak", "NB-1015", "11215", 22, False),
    ]
    return [
        Customer(customer_id=cid, full_name=name, account_number=acct,
                 postcode=pc, email=f"{name.split()[0].lower()}@example.com",
                 tenure_months=tenure, vulnerable=vuln)
        for cid, name, acct, pc, tenure, vuln in rows
    ]


def _cycle_dates(account: Account) -> Account:
    """Stamp the account's current cycle as dates, derived from its
    cycle position against the shared clock — the same pinning the
    seeded credit history uses, so the calendar means the same
    thing whenever the book is seeded: the cycle started
    (cycle_day - 1) days ago and runs cycle_days days in all."""
    from datetime import timedelta

    from .clock import today

    start = today() - timedelta(days=account.cycle_day - 1)
    account.cycle_start = start.isoformat()
    account.cycle_end = (start + timedelta(days=account.cycle_days - 1)).isoformat()
    return account


def build_accounts() -> list[Account]:
    return [_cycle_dates(account) for account in [
        Account(account_number="NB-1001", customer_id="CUST-001", plan_id="PLAN-ESS",
                balance_due=D("94.35"), payment_method_last4="4417", cycle_day=19,
                typical_daily_data_gb=D("0.6")),
        Account(account_number="NB-1002", customer_id="CUST-002", plan_id="PLAN-UNL",
                balance_due=D("152.52"), payment_method_last4="9034", cycle_day=19,
                roaming_country="Spain", roaming_gb_used=D("31"),
                roaming_since="2026-09-25", typical_daily_data_gb=D("2.5")),
        Account(account_number="NB-1003", customer_id="CUST-003", plan_id="PLAN-FAM",
                balance_due=D("118.60"), payment_method_last4="2210", cycle_day=11,
                typical_daily_data_gb=D("1.8")),
        Account(account_number="NB-1004", customer_id="CUST-004", plan_id="PLAN-ESS",
                balance_due=D("45.00"), payment_method_last4="7781", cycle_day=7,
                typical_daily_data_gb=D("0.2")),
        Account(account_number="NB-1005", customer_id="CUST-005", plan_id="PLAN-ESS",
                balance_due=D("76.77"), payment_method_last4="5150", cycle_day=14,
                typical_daily_data_gb=D("0.7"),
                # The network-side line test finds a fault on Grace's
                # line: her broadband procedure branches straight to
                # a field visit, by the book.
                diagnostics={"line_test": "fault_found"}),
        Account(account_number="NB-1006", customer_id="CUST-006", plan_id="PLAN-ESS",
                balance_due=D("61.10"), payment_method_last4="6602", cycle_day=19,
                typical_daily_data_gb=D("0.5")),
        Account(account_number="NB-1007", customer_id="CUST-007", plan_id="PLAN-ESS",
                balance_due=D("45.00"), payment_method_last4="3098", cycle_day=5,
                typical_daily_data_gb=D("0.9")),
        Account(account_number="NB-1008", customer_id="CUST-008", plan_id="PLAN-BAS",
                balance_due=D("25.00"), payment_method_last4="8845", cycle_day=9,
                typical_daily_data_gb=D("0.1")),
        Account(account_number="NB-1009", customer_id="CUST-009", plan_id="PLAN-UNL",
                balance_due=D("75.00"), payment_method_last4="1173", cycle_day=22,
                typical_daily_data_gb=D("1.2")),
        Account(account_number="NB-1010", customer_id="CUST-010", plan_id="PLAN-ESS",
                balance_due=D("212.40"), payment_method_last4="4029", cycle_day=19,
                typical_daily_data_gb=D("0.8")),
        Account(account_number="NB-1011", customer_id="CUST-011", plan_id="PLAN-ESS",
                balance_due=D("72.61"), payment_method_last4="7305", cycle_day=19,
                typical_daily_data_gb=D("0.4")),
        Account(account_number="NB-1012", customer_id="CUST-012", plan_id="PLAN-ESS",
                balance_due=D("130.38"), payment_method_last4="2214", cycle_day=19,
                typical_daily_data_gb=D("1.1")),
        # Nadia is mid-trip in Greece, past the fair-use ceiling,
        # and no fair-use warning exists anywhere on her record —
        # the gap her roaming dispute will turn on.
        Account(account_number="NB-1013", customer_id="CUST-013", plan_id="PLAN-ESS",
                balance_due=D("92.40"), payment_method_last4="6631", cycle_day=19,
                roaming_country="Greece", roaming_gb_used=D("28"),
                roaming_since="2026-09-24", typical_daily_data_gb=D("1.6")),
        # Chris roamed in Italy under a Roam Pass he activated
        # before the trip (seeded on the action log below): his
        # bill carries the per-GB overage the pass doesn't cover.
        Account(account_number="NB-1014", customer_id="CUST-014", plan_id="PLAN-ESS",
                balance_due=D("85.86"), payment_method_last4="9021", cycle_day=19,
                roaming_country="Italy", roaming_gb_used=D("12"),
                roaming_since="2026-09-20", typical_daily_data_gb=D("0.8")),
        # Marta's arrangement (seeded below) was written over
        # this balance; its first instalment is already past due.
        Account(account_number="NB-1015", customer_id="CUST-015", plan_id="PLAN-ESS",
                balance_due=D("187.50"), payment_method_last4="3355", cycle_day=19,
                typical_daily_data_gb=D("0.5")),
    ]]


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
        _bill("BILL-1013", "NB-1013", [
            BillCharge(charge_id="CH-1013-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1013-2", kind="roaming",
                       description="Roaming data — Greece (pay per GB)",
                       amount=D("42.00")),
            BillCharge(charge_id="CH-1013-3", kind="tax",
                       description="Taxes and regulatory fees", amount=D("5.40")),
        ]),
        _bill("BILL-1014", "NB-1014", [
            BillCharge(charge_id="CH-1014-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1014-2", kind="roaming",
                       description="Roaming data — Italy (pay per GB)",
                       amount=D("36.00")),
            BillCharge(charge_id="CH-1014-3", kind="tax",
                       description="Taxes and regulatory fees", amount=D("4.86")),
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
        _bill("BILL-1013-PREV", "NB-1013", [
            BillCharge(charge_id="CH-1013P-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1013P-2", kind="tax",
                       description="Taxes and regulatory fees", amount=D("5.40")),
        ], cycle=_PREVIOUS_CYCLE),
        _bill("BILL-1014-PREV", "NB-1014", [
            BillCharge(charge_id="CH-1014P-1", kind="plan",
                       description="Essential 20GB — monthly plan", amount=D("45.00")),
            BillCharge(charge_id="CH-1014P-2", kind="tax",
                       description="Taxes and regulatory fees", amount=D("4.86")),
        ], cycle=_PREVIOUS_CYCLE),
    ]


def build_policies() -> list[PolicyDocument]:
    """The policy corpus records. The documents' TEXT lives in
    the prompt registry (prompts.POLICY_TEXTS) — the corpus and
    the registry are one wording, never two copies: this builder
    wraps each registered text in its record (id, title, section),
    and activating a new registry version of a policy rewrites
    the corpus document to match (prompts._sync_policy_corpus)."""
    rows = [
        ("POL-BILL-01", "Billing disputes", "billing"),
        ("POL-BILL-02", "Goodwill credits", "billing"),
        ("POL-BILL-03", "Reading your bill", "billing"),
        ("POL-ROAM-01", "Roaming and the Roam Pass", "roaming"),
        ("POL-ROAM-02", "Roaming fair use", "roaming"),
        ("POL-ROAM-03", "Roaming projections and warnings", "roaming"),
        ("POL-PLAN-01", "Changing your plan", "plans"),
        ("POL-PLAN-02", "When a plan change starts, and undoing one", "plans"),
        ("POL-ID-01", "Verifying your identity", "identity"),
        ("POL-PAY-01", "Payment arrangements", "payments"),
        ("POL-PAY-02", "Payment methods", "payments"),
        ("POL-PAY-03", "Failed payments, refunds and autopay", "payments"),
        ("POL-ESC-01", "Speaking to a human", "escalation"),
        ("POL-CARE-01", "Stores and opening hours", "general"),
        ("POL-CARE-02", "Complaints, feedback and how we contact you", "general"),
        ("POL-TECH-01", "Fixing connection problems", "technical"),
    ]
    return [PolicyDocument(doc_id=d, title=t, section=s,
                           text=POLICY_TEXTS[d])
            for d, t, s in rows]


def build_procedures() -> list[Procedure]:
    """The procedures, as data. The engine (procedures.py) walks
    these records — kinds, thresholds, branch tables, safety
    flags — so a procedure is edited here (as a new version),
    never in the walker."""
    return [
        Procedure(procedure_id="PROC-NOSIGNAL", title="No signal",
                  symptom="no_signal",
                  symptom_phrases=["no signal", "no service", "signal"],
                  steps=[
                      ProcedureStep(step_id="NS-1", kind="tool_check",
                                    instruction="Check the coverage at your postcode — one moment while I look.",
                                    check="coverage_ok"),
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
                  symptom_phrases=["slow"],
                  steps=[
                      ProcedureStep(step_id="SD-1", kind="tool_check",
                                    instruction="Let me check how much of your data allowance is left this cycle.",
                                    check="allowance_ok"),
                      ProcedureStep(step_id="SD-2",
                                    instruction="Turn airplane mode on for 30 seconds, then off, and run a quick page load. Faster now?",
                                    check="speed_restored", on_fail="next"),
                      ProcedureStep(step_id="SD-3",
                                    instruction="Move near a window or outside if you can — building materials block signal. Any improvement?",
                                    check="speed_restored", on_fail="callback"),
                  ]),
        Procedure(procedure_id="PROC-WIFICALL", title="Wi-Fi calling setup",
                  symptom="wifi_calling",
                  symptom_phrases=["wifi calling", "wi-fi calling"],
                  steps=[
                      ProcedureStep(step_id="WC-1",
                                    instruction="Open Settings → Phone → Wi-Fi Calling and switch it on; confirm your emergency address if asked.",
                                    check="wifi_calling_on", on_fail="next"),
                      ProcedureStep(step_id="WC-2",
                                    instruction="Connect to your Wi-Fi network and place a short test call. Did it connect over Wi-Fi?",
                                    check="call_connected", on_fail="callback"),
                  ]),
        Procedure(procedure_id="PROC-BROADBAND", title="Broadband keeps dropping",
                  symptom="broadband_drops",
                  symptom_phrases=["broadband", "keeps dropping",
                                   "dropping out", "keeps disconnecting"],
                  requires_account=True,
                  steps=[
                      ProcedureStep(step_id="BB-1", kind="tool_check",
                                    instruction="Run a line test on the line.",
                                    check="line_test",
                                    branches={"fault_found": "BB-6"}),
                      ProcedureStep(step_id="BB-2", mandatory_safety=True,
                                    instruction="First, a safety step we never skip: unplug the router from the wall socket and check the plug and cable as you do — if anything looks damaged or smells of burning, stop and tell me. Otherwise leave it unplugged for a full minute, then tell me it's unplugged.",
                                    check="router_power_cycled", on_fail="next"),
                      ProcedureStep(step_id="BB-3", kind="measurement",
                                    instruction="Plug the router back in and give the lights a minute to settle. Then run a speed test — any speed-test app or site — and tell me the download number it shows, in Mbps.",
                                    metric="download_mbps", unit="Mbps",
                                    threshold=D("25"),
                                    below_step="BB-4",
                                    at_or_above_step="BB-7"),
                      ProcedureStep(step_id="BB-4", kind="branch",
                                    rule="speed_critical",
                                    threshold=D("10"),
                                    branches={"critical": "BB-6",
                                              "normal": "BB-5"}),
                      ProcedureStep(step_id="BB-5",
                                    instruction="Reseat the cable between the wall socket and the router — push both ends in until they click — then run the speed test once more. Better now?",
                                    check="speed_restored", on_fail="next"),
                      ProcedureStep(step_id="BB-6", kind="terminal",
                                    outcome="escalate_field_visit",
                                    instruction="The readings point to a fault on the line itself, so this needs an engineer visit rather than more steps here."),
                      ProcedureStep(step_id="BB-7", kind="terminal",
                                    outcome="resolved",
                                    instruction="That reading is healthy — the power cycle cleared the fault state. Your broadband should stay up now; if it drops again, quote this conversation and we'll pick up right here."),
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


def _seed_roaming_history(store) -> None:
    """Executed roaming actions from before 'now' — the record a
    roaming dispute reads. Chris Novak (NB-1014) activated the
    Roam Pass on 2026-09-20, the day his Italy trip began, so the
    pass window covers the roaming on his current bill."""
    from .models import ActionKind, ActionRecord

    store.save_action(ActionRecord(
        action_id="ACT-HIST-1014-1",
        conversation_id="CONV-HIST-1014",
        kind=ActionKind.ACTIVATE_OFFER,
        summary="activate the Roam Pass — 10 days for $30.00",
        payload={"offer_id": "OFF-ROAM10"},
        amount=D("30.00"),
        basis="Customer requested the pass in conversation (POL-ROAM-01).",
        status="executed",
        account_number="NB-1014",
        created_date="2026-09-20",
    ))


def _seed_payment_history(store) -> None:
    """A payment arrangement from before 'now' — the record the
    broken-arrangement rule reads. Marta Nowak (NB-1015) split her
    $187.50 balance into three instalments of $62.50 on
    2026-09-05; the first fell due on 2026-09-19 and was never
    paid, so the arrangement derives as broken on any later read.
    """
    from .models import ArrangementInstallment, PaymentArrangement

    store.save_payment_arrangement(PaymentArrangement(
        arrangement_id="ARR-HIST-1015",
        conversation_id="CONV-HIST-1015",
        account_number="NB-1015",
        total=D("187.50"),
        installments=[
            ArrangementInstallment(number=1, amount=D("62.50"),
                                   due_date="2026-09-19"),
            ArrangementInstallment(number=2, amount=D("62.50"),
                                   due_date="2026-10-19"),
            ArrangementInstallment(number=3, amount=D("62.50"),
                                   due_date="2026-11-18"),
        ],
        state="active",
        created_date="2026-09-05",
    ))


def seed_store(store) -> dict:
    """Write the whole book into a store and embed the policy corpus.
    Returns counts for readiness/demo output."""
    plans = build_plans()
    offers = build_offers()
    rates = build_roaming_rates()
    customers = build_customers()
    accounts = build_accounts()
    bills = build_bills()
    policies = build_policies()
    procedures = build_procedures()
    for plan in plans:
        store.save_plan(plan)
    for offer in offers:
        store.save_offer(offer)
    for rate in rates:
        store.save_roaming_rate(rate)
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
    prompt_count = seed_registry(store)
    _seed_credit_history(store)
    _seed_roaming_history(store)
    _seed_payment_history(store)
    return {
        "plans": len(plans), "offers": len(offers), "customers": len(customers),
        "accounts": len(accounts), "bills": len(bills),
        "policies": len(policies), "procedures": len(procedures),
        "roaming_rates": len(rates), "prompts": prompt_count,
    }
