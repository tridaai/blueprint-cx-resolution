from decimal import Decimal

from cx_agent.dataset import (
    build_accounts,
    build_bills,
    build_customers,
    build_policies,
    build_procedures,
    seed_store,
)
from cx_agent.store import InMemoryStore


def _latest_bills() -> dict:
    """The current bill per account — build_bills now returns the
    previous cycle too, and the balance speaks of the latest."""
    latest = {}
    for bill in build_bills():
        current = latest.get(bill.account_number)
        if current is None or bill.issued > current.issued:
            latest[bill.account_number] = bill
    return latest


def test_book_is_internally_consistent():
    customers = {c.customer_id: c for c in build_customers()}
    accounts = build_accounts()
    bills = _latest_bills()
    for account in accounts:
        assert account.customer_id in customers
        assert customers[account.customer_id].account_number == account.account_number
    # Billed accounts: the bill total is the balance the customer sees.
    for number, bill in bills.items():
        account = next(a for a in accounts if a.account_number == number)
        if number != "NB-1010":  # 1010's balance carries an older unpaid bill too
            assert bill.total == account.balance_due


def test_gold_facts_the_evals_lean_on():
    bills = _latest_bills()
    duplicates = [c for c in bills["NB-1001"].charges if c.duplicate_of]
    assert len(duplicates) == 1 and duplicates[0].amount == Decimal("9.99")
    after_cancel = [c for c in bills["NB-1006"].charges if c.after_cancellation]
    assert len(after_cancel) == 1 and after_cancel[0].amount == Decimal("12.00")
    accounts = {a.account_number: a for a in build_accounts()}
    assert accounts["NB-1002"].roaming_gb_used == Decimal("31")
    assert accounts["NB-1010"].balance_due == Decimal("212.40")
    # Round 3 gold: Felix's roaming line is billed $58 over its rate
    # (above the evidenced cap, so his dispute escalates), Ruth's
    # bill is exactly right, and every billed account has a previous
    # bill to compare against.
    roaming = next(c for c in bills["NB-1012"].charges if c.kind == "roaming")
    assert roaming.amount - roaming.expected_amount == Decimal("58.00")
    assert all(c.expected_amount is None and not c.duplicate_of
               and not c.after_cancellation for c in bills["NB-1011"].charges)
    billed = {b.account_number for b in build_bills()}
    history = {}
    for bill in build_bills():
        history.setdefault(bill.account_number, []).append(bill)
    assert all(len(history[n]) == 2 for n in billed)


def test_corpus_and_procedures_cover_the_desks():
    sections = {p.section for p in build_policies()}
    assert {"billing", "roaming", "plans", "identity", "payments",
            "escalation", "general", "technical"} <= sections
    symptoms = {p.symptom for p in build_procedures()}
    assert symptoms == {"no_signal", "slow_data", "wifi_calling",
                        "broadband_drops"}
    # Round 4 gold: Grace Liu's line test finds a network fault
    # (her broadband walk branches to a field visit); every other
    # account's line reads clear by default.
    accounts = {a.account_number: a for a in build_accounts()}
    assert accounts["NB-1005"].diagnostics == {"line_test": "fault_found"}
    assert all(a.diagnostics == {} for n, a in accounts.items()
               if n != "NB-1005")


def test_seed_writes_everything_with_embeddings():
    store = InMemoryStore()
    counts = seed_store(store)
    assert counts == {"plans": 4, "offers": 3, "customers": 14, "accounts": 14,
                      "bills": 16, "policies": 14, "procedures": 4,
                      "roaming_rates": 16}
    embeddings = store.list_embeddings()
    assert len(embeddings) == 14
    assert all(len(vec) == 384 for _doc, vec in embeddings)
