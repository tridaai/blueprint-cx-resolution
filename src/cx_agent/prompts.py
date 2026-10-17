"""The prompt registry (round 13).

Every governed text in the system lives here, as versioned data:

- **compose** — the instructions the supervisor's drafter works
  under: the system prompt and the user-message template the
  facts are poured into (`llm.py` resolves both from here; there
  is no second copy in the boundary code).
- **intent** — the router's classification prompt, used for the
  live tie-break hint.
- **explanation** — the second explanation forms (round 11): the
  same policy said differently, served verbatim when the signals
  say the first form did not land.
- **policy** — the policy corpus documents themselves. They are
  the texts the agent retrieves, cites and serves, and they state
  the governed figures (caps, thresholds, windows) — so they are
  versioned and linted like everything else. `dataset.py` builds
  the corpus records from these texts; activating a new policy
  version rewrites the corpus document and its embedding to
  match.

Versioning follows the procedures' pattern: entries are stored
per version (`prompt_id` + `version`), an entry is never edited
or deleted, a change is a new version, and the ACTIVE version is
the highest unless an operator pin stands (a rollback is a pin;
unpinning is its own act). Every operator act is recorded in the
act log, and every activation passes the lint gate first
(`promptlint.py`) — a version whose text would break the lint
cannot become active.

Loading is fail-closed: resolving an entry the registry does
not have, or one whose record is malformed (a version below 1,
blank text, an unknown surface), raises `PromptRegistryError`
naming the entry and the defect. There is no inline fallback
text anywhere — that is the point of the registry.

What is NOT here, deliberately: the deterministic template text
the code composes itself — the supervisor's confirmation ask
and offer lines, the specialists' computed findings sentences,
the procedure step instructions (already versioned with the
procedures), the billing desk's second form (composed from the
customer's own figures at runtime, in `explanations.py`). None
of it is an instruction to a model; it is code-composed text,
reviewed as code.
"""

from __future__ import annotations

import uuid

from .clock import today
from .models import PolicyDocument, PromptAct, PromptEntry, PromptPin

SURFACES = ("compose", "intent", "explanation", "policy")

COMPOSE_SYSTEM_ID = "compose.system"
COMPOSE_USER_ID = "compose.user"
INTENT_SYSTEM_ID = "intent.system"
INTENT_USER_ID = "intent.user"

COMPOSE_USER_TEMPLATE = (
    "Customer said: {customer_message}\n\n"
    "Facts established:\n{facts}\n\n"
    "Lines that must appear verbatim:\n{verbatim}\n\n"
    "Confirmation ask to include, if any: {action_prompt}{sources}"
)

INTENT_SYSTEM = "Classify a telecom support message into exactly one label."

INTENT_USER_TEMPLATE = (
    "Labels: {labels}. Reply with the label only.\n\nMessage: {text}"
)


def policy_prompt_id(doc_id: str) -> str:
    return f"policy.{doc_id}"


def explanation_prompt_id(topic: str) -> str:
    return f"explain.{topic}"


# === SEED TEXTS ===

COMPOSE_SYSTEM = "You are the voice of Northbeam Mobile's customer support agent. Write one clear, warm reply using ONLY the facts provided. Never invent figures, dates, policy terms or account details. Keep every verbatim line exactly as given. If a Sources line is provided, end the reply with it unchanged. Do not promise anything the facts do not state."

POLICY_TEXTS: dict[str, str] = {'POL-BILL-01': 'When a charge on a bill is wrong — a duplicate charge, a charge for a service '
                'after it was cancelled, or a charge at the wrong rate — the incorrect amount is '
                'credited in full once the evidence is on the bill itself. Each disputed charge is '
                'decided on its own evidence, and the dispute is recorded as a case with the '
                'decision for every line. Evidenced dispute credits are capped at $50 per bill; '
                'above that the case goes to the billing team with its record intact. The customer '
                'confirms the credit before it is applied.',
 'POL-BILL-02': 'When the bill is correct but the customer has had a poor experience, the agent '
                "may offer a goodwill credit of up to $25. The credit needs the customer's "
                'agreement and is applied to the next bill. Goodwill credits are offered once per '
                'conversation, and no more than $50 of goodwill may be credited to an account in '
                'any 90-day period — past that, goodwill is refused, though an incorrect charge is '
                'still credited in full under the dispute policy.',
 'POL-BILL-03': 'Every bill is itemised: the plan charge, device instalments, one-off charges, '
                'roaming, and taxes and regulatory fees are separate lines. The total is the sum '
                'of the lines. Due dates fall 21 days after the bill is issued.',
 'POL-CARE-01': 'Northbeam stores are open Monday to Saturday, 9am to 6pm local time. The support '
                'line is staffed 24 hours a day. Store staff can help with SIM swaps and device '
                'setup in person.',
 'POL-CARE-02': 'A complaint is recorded the moment a customer makes one — their words, a '
                'category, and a reference the customer is given there and then. A person responds '
                'in substance within five days; a complaint still open after that is escalated to '
                "the care team. Compliments and feedback are recorded too, in the customer's own "
                'words. When a customer tells us how they need to be contacted — large print, '
                'braille, easy read, email or text instead of letters or calls — the need is '
                'recorded on the account and honoured at every later contact. When a customer '
                'tells us plainly about a bereavement, a serious illness or money hardship, that '
                'is recorded on the account from their own statement and a person from the care '
                'team handles the contact.',
 'POL-ESC-01': 'A customer can always reach a human agent. The agent will first offer to solve the '
               'problem itself; if the customer asks again, or the problem cannot be solved in '
               'chat, the conversation transfers with a summary so the customer does not repeat '
               'themselves.',
 'POL-ID-01': 'Before any change to an account or its money, identity is verified with three '
              'things: the account number, the postcode on the account, and a one-time code sent '
              'to the contact on file. Three failed code attempts lock verification for the '
              'conversation and the customer is offered a human agent.',
 'POL-PAY-01': 'A customer who cannot pay in full may split a balance of $100 or more into up to '
               'three monthly instalments, each of at least $25, with no fee, as long as the '
               'account is not already in collections and has no arrangement already running. The '
               'first instalment falls due within 14 days and the rest follow monthly. The '
               'arrangement is confirmed by the customer and recorded on the account, with its '
               'amounts and dates. An instalment that passes its date unpaid breaks the '
               'arrangement, and a broken arrangement is handled by a person — it is not '
               'renegotiated in chat.',
 'POL-PAY-02': 'The card on file can be replaced once identity is verified. Only the last four '
               'digits are ever shown or stored in the conversation. A replacement card takes '
               'effect for the next bill.',
 'POL-PAY-03': 'When a payment is declined, the reason decides what happens next. A soft decline — '
               'insufficient funds, or the processor not answering — leaves the card usable, and '
               'the payment can be tried again once the cause is fixed. A hard decline — an '
               'expired card, a card reported lost, or a closed account — means that card can '
               'never be charged again: it is not retried, and the customer is asked for a new '
               'card instead. Refunds go back to the payment method on the account, and nowhere '
               'else — not to another card, not to a bank account, not by cheque. Money owed back '
               'is read from the account, never from the conversation. With autopay on, each bill '
               'is taken automatically from the method on file; if that payment fails, the '
               'customer is told what failed, and why, at their next contact.',
 'POL-PLAN-01': 'A plan change can start immediately or at the next cycle. An immediate change is '
                'prorated: the customer is credited for the unused part of the old plan and '
                'charged for the remaining part of the new plan, to the day. The new monthly price '
                'applies in full from the next cycle. The customer confirms the change, and the '
                'prorated figures, before anything is applied.',
 'POL-PLAN-02': 'An upgrade — moving to a higher-priced plan — starts straight away and is '
                'prorated to the day. A downgrade starts at the next cycle instead: there is no '
                'proration on a downgrade, and the customer keeps the plan they have paid for '
                'until the cycle ends. A customer who wants to leave or downgrade because of a '
                'fault is handled by the billing team under the dispute policy, not by the '
                'plan-change desk. A plan change can be reversed on the day it was made; after '
                "that day, reversing it is the offers team's decision.",
 'POL-RET-01': 'When a customer says they want to cancel, the reason is recorded in their own '
                'words against a governed list and a retention case is opened. The agent may make '
                'one save offer, and only one, built from the mechanisms already on the account: a '
                'downgrade priced under the plan-change policy, a goodwill credit under the '
                'goodwill policy, or the loyalty upgrade price where the customer is eligible for '
                'it. If the customer says they want no offers, none is made. If they decline the '
                'offer, or ask to cancel again, the cancellation goes ahead without a second '
                'offer. A cancellation takes effect at the end of the current billing cycle, so '
                'the customer keeps the service they have paid for until then. A customer who '
                'wants to leave because of a fault is handled by the billing team under the '
                'dispute policy, and the retention case records that routing.',
 'POL-ROAM-01': 'Outside a pass, roaming data is charged per GB at the destination rate. The Roam '
                'Pass costs $30 for 10 days and includes unlimited calls and texts plus 1 GB of '
                'data per day in more than 40 countries. A pass can be activated at any point in a '
                'trip and starts when the customer confirms it.',
 'POL-ROAM-02': 'Roaming fair use is 25 GB per trip. Beyond 25 GB, speeds reduce and additional '
                'data is charged per GB. The agent must tell a customer who is near or past the '
                'fair-use ceiling before recommending any further chargeable roaming product.',
 'POL-ROAM-03': 'Before a trip, the agent prices both ways of paying from the destination rate and '
                "the customer's typical daily use — pay-as-you-go and the Roam Pass — and states "
                'both figures and the saving before recommending either. When the projected '
                'pay-as-you-go cost passes $50, the projection and the comparison lead the '
                'conversation: no chargeable roaming product is proposed until the customer has '
                'seen them. While a customer roams, the agent warns as fair use runs out — a '
                'summary at 80% of the ceiling and again at 100%, stating the usage, the ceiling, '
                'and what continuing at the same pace is projected to cost. A roaming charge '
                'disputed afterwards is decided on the record: a charge covered by an active pass '
                'stands, a charge the customer was warned about stands, and a charge that passed '
                'the ceiling with no warning on record is credited under the dispute policy.',
 'POL-TECH-01': 'Connection problems are worked through in a fixed order: confirm coverage, '
                'restart the device, reseat the SIM, then reset network settings. Each step is '
                'checked before moving on. If the steps do not restore service, a technician '
                'callback is booked. Broadband faults start with a line test: a fault on the '
                'network side goes straight to an engineer visit. The router power-cycle is a '
                'safety step and is never skipped. A healthy home line holds at least 25 Mbps on a '
                'speed test; below that, the fault is treated as a line fault, not a device fault.'}

EXPLANATION_TEXTS: dict[str, str] = {'POL-BILL-01': 'The short version: if a charge looks wrong, name it and we check it line by line '
                '— anything the bill itself proves wrong comes back to you as a credit, anything '
                "that checks out stays, and anything over the agent's limit goes to a person with "
                'the case already written up. Say your bill shows Device protection at $9.99 twice '
                'in one month: that second line is proven wrong by the bill itself, so it is '
                'credited in full — you do not have to argue it, and you do not pay it while it is '
                'checked.',
 'POL-BILL-03': 'Take a sample bill. It totals $58.54 for the month: a $45.00 plan line, a $9.99 '
                'one-off device protection charge, and $3.55 of tax — the total is just those '
                'lines added together. Your bill reads the same way, so the three things worth '
                'checking each month are the total, the due date, and any line that was not on '
                "last month's bill: a new line is usually the one that explains a jump.",
 'POL-CARE-01': 'The short version: if you need a person, the support line is the round-the-clock '
                'option — any hour, any day, including account changes. Stores are the daytime '
                'option, 9am to 6pm Monday to Saturday, and they are the only place for SIM swaps '
                'and hands-on device checks. So: urgent or after hours, call the line; hands-on '
                'with a device, visit a store in the day.',
 'POL-ID-01': 'The short version: two steps. First your account number and the postcode on the '
              'account, so we open the right record; then the 6-digit code we send to the contact '
              'on file, so we know it is you holding it. Three wrong codes lock the chat and a '
              'person finishes verifying with you instead — the lock is there so nobody can keep '
              'guessing at your account.',
 'POL-PAY-01': 'Say you owe $120 and agree an arrangement over three instalments: $40 lands on the '
               'first date, $40 a month later, $40 a month after that — each amount and date is on '
               'the record before the first payment is due. Keep to the dates and nothing else is '
               'owed; miss one and the arrangement breaks, the full balance falls due, and a '
               'person picks the case up.',
 'POL-PLAN-01': 'The short version: upgrading starts today, and you pay only for the extra days — '
                'the part-month charge is worked out from your real cycle dates, not a flat fee. '
                'Downgrading waits until the cycle turns over, so you keep everything you have '
                'already paid for until then. Either way, the exact figures are quoted and '
                'confirmed with you before anything changes.',
 'POL-ROAM-02': 'Say you use 8 GB in France on a plan with a 10 GB fair-use amount: you are at '
                '80%, so you get a heads-up message. Use the full 10 GB and you get a second '
                'message, and from there pay-as-you-go rates apply until the cycle resets. The '
                'clock that matters is your billing cycle, not the length of the trip — and both '
                'messages land before extra charges do.'}


def seed_entries() -> list[PromptEntry]:
    """The registry as shipped: version 1 of every governed text.
    The seed writes the versions a store does not have yet and
    never overwrites one it does — an operator's newer version
    survives a re-seed."""
    entries = [
        PromptEntry(prompt_id=COMPOSE_SYSTEM_ID, version=1,
                    surface="compose", text=COMPOSE_SYSTEM),
        PromptEntry(prompt_id=COMPOSE_USER_ID, version=1,
                    surface="compose", text=COMPOSE_USER_TEMPLATE),
        PromptEntry(prompt_id=INTENT_SYSTEM_ID, version=1,
                    surface="intent", text=INTENT_SYSTEM),
        PromptEntry(prompt_id=INTENT_USER_ID, version=1,
                    surface="intent", text=INTENT_USER_TEMPLATE),
    ]
    entries += [
        PromptEntry(prompt_id=explanation_prompt_id(topic), version=1,
                    surface="explanation", text=text)
        for topic, text in sorted(EXPLANATION_TEXTS.items())
    ]
    entries += [
        PromptEntry(prompt_id=policy_prompt_id(doc_id), version=1,
                    surface="policy", text=text)
        for doc_id, text in sorted(POLICY_TEXTS.items())
    ]
    return entries


class PromptRegistryError(RuntimeError):
    """The registry cannot serve what was asked of it: an entry
    is missing, malformed, or an act names something that does
    not exist. The message names the entry and the defect — a
    registry failure is a deployment bug, never a runtime mood."""


def _validate_entry(entry: PromptEntry) -> None:
    if not isinstance(entry.version, int) or entry.version < 1:
        raise PromptRegistryError(
            f"prompt registry: entry '{entry.prompt_id}' has a "
            f"malformed version ({entry.version!r}); versions are "
            f"positive integers")
    if not entry.text or not entry.text.strip():
        raise PromptRegistryError(
            f"prompt registry: entry '{entry.prompt_id}' v{entry.version} "
            f"has blank text")
    if entry.surface not in SURFACES:
        raise PromptRegistryError(
            f"prompt registry: entry '{entry.prompt_id}' v{entry.version} "
            f"has unknown surface {entry.surface!r} "
            f"(expected one of {', '.join(SURFACES)})")


class PromptRegistry:
    """A read view over the registry in one store. Stateless —
    the store is the registry; this class is its rules."""

    def __init__(self, store) -> None:
        self.store = store

    def known_ids(self) -> list[str]:
        return sorted({e.prompt_id
                       for e in self.store.list_prompt_entries()})

    def versions(self, prompt_id: str) -> list[PromptEntry]:
        return self.store.list_prompt_entries(prompt_id)

    def entry_at(self, prompt_id: str, version: int) -> PromptEntry | None:
        entry = self.store.get_prompt_version(prompt_id, version)
        if entry is not None:
            _validate_entry(entry)
        return entry

    def pin(self, prompt_id: str) -> PromptPin | None:
        return self.store.get_prompt_pin(prompt_id)

    def active_entry(self, prompt_id: str) -> PromptEntry:
        versions = self.versions(prompt_id)
        if not versions:
            raise PromptRegistryError(
                f"prompt registry: no entry for '{prompt_id}' — the "
                f"registry is seeded with the book; a missing entry "
                f"means the seed did not run or the registry was "
                f"damaged, and there is no fallback text")
        pin = self.pin(prompt_id)
        if pin is not None:
            entry = self.store.get_prompt_version(prompt_id, pin.version)
            if entry is None:
                raise PromptRegistryError(
                    f"prompt registry: '{prompt_id}' is pinned to "
                    f"v{pin.version}, which does not exist")
        else:
            entry = versions[-1]
        _validate_entry(entry)
        return entry

    def active_version(self, prompt_id: str) -> int:
        return self.active_entry(prompt_id).version

    def active_text(self, prompt_id: str) -> str:
        return self.active_entry(prompt_id).text

    def active_entries(self) -> list[PromptEntry]:
        return [self.active_entry(pid) for pid in self.known_ids()]

    def version_or_none(self, prompt_id: str) -> int | None:
        """The active version, or None when the registry cannot
        serve the entry. For provenance recording, which notes
        what served a conversation without ever failing its turn
        over bookkeeping."""
        try:
            return self.active_version(prompt_id)
        except PromptRegistryError:
            return None


def seed_registry(store) -> int:
    """Write the seed versions the store lacks, then validate the
    whole seeded set — the fail-closed load: a registry that
    cannot serve every shipped entry refuses here, at seed time,
    with the defect named. Returns the number of governed texts
    in the registry."""
    for entry in seed_entries():
        if store.get_prompt_version(entry.prompt_id, entry.version) is None:
            store.save_prompt_entry(entry)
    registry = PromptRegistry(store)
    for entry in seed_entries():
        registry.active_entry(entry.prompt_id)
    return len({e.prompt_id for e in seed_entries()})


def render_template(template: str, **slots: str) -> str:
    """Pour values into a registry template. The templates carry
    named slots ({facts}, {verbatim}, ...); substitution is a
    plain token replacement, so a value can contain anything —
    braces included — without escaping games."""
    out = template
    for key, value in slots.items():
        out = out.replace("{" + key + "}", value)
    return out


# -- operator acts ----------------------------------------------------
# Each act passes the lint gate over the active set AS IT WOULD
# STAND after the act, then writes, then records itself in the act
# log. A refused act changes nothing and records nothing.


def _record_act(store, kind: str, prompt_id: str, version: int,
                actor: str, note: str) -> PromptAct:
    seq = max((a.seq for a in store.list_prompt_acts()), default=0) + 1
    act = PromptAct(
        act_id=f"PACT-{uuid.uuid4().hex[:10]}", seq=seq, kind=kind,
        prompt_id=prompt_id, version=version, actor=actor, note=note,
        created_date=today().isoformat())
    store.save_prompt_act(act)
    return act


def _lint_gate(store, policy, candidate: dict[str, PromptEntry]) -> None:
    from . import promptlint

    registry = PromptRegistry(store)
    actives = {e.prompt_id: e for e in registry.active_entries()}
    actives.update(candidate)
    findings = promptlint.lint_entries(list(actives.values()), policy)
    if findings:
        raise promptlint.PromptLintError(findings)


def _sync_policy_corpus(store, prompt_id: str) -> None:
    """A policy entry's active text IS the corpus document's
    text: when an act changes which version is active, the corpus
    record and its embedding follow, so retrieval and citation
    serve the same words the registry calls active."""
    if not prompt_id.startswith("policy."):
        return
    doc_id = prompt_id[len("policy."):]
    doc = store.get_policy(doc_id)
    if doc is None:
        return
    text = PromptRegistry(store).active_text(prompt_id)
    if doc.text == text:
        return
    from .documents import embed

    store.save_policy(PolicyDocument(
        doc_id=doc.doc_id, title=doc.title, section=doc.section,
        text=text))
    store.set_embedding(doc_id, embed(f"{doc.title}. {text}"))


def add_version(store, policy, prompt_id: str, text: str,
                actor: str = "", note: str = "") -> PromptEntry:
    """Add the next version of an existing entry. Activation is
    immediate when no pin stands — the new version is the highest
    — and gated by the lint either way the pin leaves it. New
    prompts are born in code, as seed entries; this act versions
    the ones that exist."""
    registry = PromptRegistry(store)
    versions = registry.versions(prompt_id)
    if not versions:
        raise PromptRegistryError(
            f"prompt registry: unknown prompt '{prompt_id}' — new "
            f"prompts are added in code, as seed entries, not at "
            f"runtime")
    entry = PromptEntry(
        prompt_id=prompt_id, version=versions[-1].version + 1,
        surface=versions[-1].surface, text=text)
    _validate_entry(entry)
    candidate = {} if registry.pin(prompt_id) is not None \
        else {prompt_id: entry}
    _lint_gate(store, policy, candidate)
    store.save_prompt_entry(entry)
    _record_act(store, "add_version", prompt_id, entry.version,
                actor, note)
    _sync_policy_corpus(store, prompt_id)
    return entry


def rollback(store, policy, prompt_id: str, version: int,
             actor: str = "", note: str = "") -> PromptPin:
    """Pin an older version active again — the rollback. The
    newer versions stay on the books, readable forever; the pin
    is what changes, and unpinning (its own act) hands activation
    back to the highest version."""
    registry = PromptRegistry(store)
    target = registry.entry_at(prompt_id, version)
    if target is None:
        raise PromptRegistryError(
            f"prompt registry: '{prompt_id}' has no version "
            f"{version} to roll back to")
    _lint_gate(store, policy, {prompt_id: target})
    pin = PromptPin(prompt_id=prompt_id, version=version,
                    actor=actor, note=note)
    store.save_prompt_pin(pin)
    _record_act(store, "rollback", prompt_id, version, actor, note)
    _sync_policy_corpus(store, prompt_id)
    return pin


def unpin(store, policy, prompt_id: str,
          actor: str = "", note: str = "") -> None:
    """Lift a pin: the highest version becomes active again. The
    gate checks that set before the pin moves."""
    registry = PromptRegistry(store)
    versions = registry.versions(prompt_id)
    if not versions:
        raise PromptRegistryError(
            f"prompt registry: unknown prompt '{prompt_id}'")
    if registry.pin(prompt_id) is None:
        raise PromptRegistryError(
            f"prompt registry: '{prompt_id}' is not pinned")
    _lint_gate(store, policy, {prompt_id: versions[-1]})
    store.delete_prompt_pin(prompt_id)
    _record_act(store, "unpin", prompt_id, 0, actor, note)
    _sync_policy_corpus(store, prompt_id)
