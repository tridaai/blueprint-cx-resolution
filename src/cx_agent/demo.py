"""A scripted walk through the agent, offline, in about a minute.

Seeds the Northbeam book into an in-memory store and runs six
conversations that between them touch every desk: a billing dispute
with an evidenced credit, a plan upgrade with proration, a guided
troubleshooting walk, a roaming pass bought past the fair-use
ceiling (warning first), a customer who asks for a human twice, and
an identity lockout. It prints the turns so a reader can watch the
rules fire — the same conversations the eval gate scores.
"""

from __future__ import annotations

from .identity import expected_code
from .service import ConversationService
from .store import InMemoryStore


def _run(service: ConversationService, title: str, script: list[str]) -> None:
    conv = service.start_conversation()
    print(f"\n=== {title} ({conv.conversation_id}) ===")
    for line in script:
        result = service.post_message(conv.conversation_id, line)
        print(f"  customer: {line}")
        print(f"  agent:    {result.reply}")
        if result.escalated:
            print(f"  ** escalated: {result.escalation_reason}")


def main() -> int:
    service = ConversationService(store=InMemoryStore())
    counts = service.seed()
    print(f"Seeded the Northbeam book: {counts}")

    _run(service, "Billing dispute — the duplicated charge", [
        "Why is my bill so high this month?",
        "My account is NB-1001 and postcode 02134",
        expected_code("NB-1001"),
        "I was charged twice for device protection, that's wrong",
        "yes",
    ])
    _run(service, "Loyalty plan upgrade with proration", [
        "I want to change my plan to Unlimited Max",
        "account NB-1001, postcode 02134",
        expected_code("NB-1001"),
        "yes go ahead",
    ])
    _run(service, "No signal — the guided steps fix it", [
        "I have no signal at all on my phone",
        "ok done, still nothing",
        "done that too, still no signal",
        "yes! signal is back, thank you",
    ])
    _run(service, "Roaming past fair use — warning before any sale", [
        "I'm in Spain and want to buy a roam pass",
        "NB-1002 postcode 90210",
        expected_code("NB-1002"),
        "yes",
    ])
    _run(service, "Asking for a human — once is an offer, twice is a transfer", [
        "Let me talk to a real person",
        "No, I said get me a human",
    ])
    _run(service, "Identity lockout — three wrong codes", [
        "update my card please",
        "NB-1005 postcode 94107",
        "111111",
        "222222",
        "333333",
    ])
    print("\nDemo complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
