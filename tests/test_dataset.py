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


def test_book_is_internally_consistent():
    customers = {c.customer_id: c for c in build_customers()}
    accounts = build_accounts()
    bills = {b.account_number: b for b in build_bills()}
    for account in accounts:
        assert account.customer_id in customers
        assert customers[account.customer_id].account_number == account.account_number
    # Billed accounts: the bill total is the balance the customer sees.
    for number, bill in bills.items():
        account = next(a for a in accounts if a.account_number == number)
        if number != "NB-1010":  # 1010's balance carries an older unpaid bill too
            assert bill.total == account.balance_due


def test_gold_facts_the_evals_lean_on():
    bills = {b.account_number: b for b in build_bills()}
    duplicates = [c for c in bills["NB-1001"].charges if c.duplicate_of]
    assert len(duplicates) == 1 and duplicates[0].amount == Decimal("9.99")
    after_cancel = [c for c in bills["NB-1006"].charges if c.after_cancellation]
    assert len(after_cancel) == 1 and after_cancel[0].amount == Decimal("12.00")
    accounts = {a.account_number: a for a in build_accounts()}
    assert accounts["NB-1002"].roaming_gb_used == Decimal("31")
    assert accounts["NB-1010"].balance_due == Decimal("212.40")


def test_corpus_and_procedures_cover_the_desks():
    sections = {p.section for p in build_policies()}
    assert {"billing", "roaming", "plans", "identity", "payments",
            "escalation", "general", "technical"} <= sections
    symptoms = {p.symptom for p in build_procedures()}
    assert symptoms == {"no_signal", "slow_data", "wifi_calling"}


def test_seed_writes_everything_with_embeddings():
    store = InMemoryStore()
    counts = seed_store(store)
    assert counts == {"plans": 4, "offers": 3, "customers": 10, "accounts": 10,
                      "bills": 4, "policies": 12, "procedures": 3}
    embeddings = store.list_embeddings()
    assert len(embeddings) == 12
    assert all(len(vec) == 384 for _doc, vec in embeddings)
