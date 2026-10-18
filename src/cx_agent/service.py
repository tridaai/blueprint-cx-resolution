"""ConversationService: the one door into the agent.

The API, the demo, the console and the eval runner all come through
here. The service owns conversation records (load → run one graph
turn → save) and refuses work the record forbids — an escalated
conversation is closed to the agent; a human owns it now.
"""

from __future__ import annotations

import logging
import uuid

from . import attribution, care_book, continuity, handoffs, incidents, mining
from . import retention_book, signals
from . import prompts as prompts_mod
from .clock import today
from .config import LLMConfig, Policy
from .dataset import seed_store
from .llm import LiveLLM, make_llm
from .models import (AuditEvent, Conversation, Escalation, HandoffPacket,
                    TurnResult)
from .prompts import PromptRegistry
from .retrieval import PolicyRetriever
from .store import Store, make_store
from .supervisor import Supervisor

log = logging.getLogger("cx_agent.service")


class ConversationService:
    def __init__(self, store: Store | None = None, policy: Policy | None = None,
                 llm=None, llm_config: LLMConfig | None = None) -> None:
        self.store = store or make_store()
        self.policy = policy or Policy.from_env()
        self.llm = llm or make_llm(llm_config)
        # The prompt registry is a view over this service's store;
        # a live drafter resolves its prompts from it (round 13).
        self.prompts = PromptRegistry(self.store)
        if isinstance(self.llm, LiveLLM):
            self.llm.prompts = self.prompts
        self.retriever = PolicyRetriever(self.store)
        self.supervisor = Supervisor(self.store, self.policy, self.llm,
                                     self.retriever)

    # -- lifecycle ------------------------------------------------------
    def seed(self) -> dict:
        return seed_store(self.store)

    def start_conversation(self) -> Conversation:
        conv = Conversation(conversation_id=f"CONV-{uuid.uuid4().hex[:10]}")
        self.store.save_conversation(conv)
        return conv

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        return self.store.get_conversation(conversation_id)

    def list_conversations(self) -> list[Conversation]:
        return self.store.list_conversations()

    def list_escalations(self):
        return self.store.list_escalations()

    # -- handoff acceptance (round 17) ----------------------------------

    def accept_handoff(self, escalation_id: str, receiver: str):
        """A person picks a handoff up: the receiver and the
        day are stamped on the escalation record, and the act
        is written to the conversation's audit trail (see
        handoffs.accept_handoff)."""
        return handoffs.accept_handoff(self.store, escalation_id, receiver)

    def unaccepted_handoffs(self) -> list[dict]:
        """The handoffs nobody has picked up past the governed
        threshold — the queue's fallen-on-the-floor view. A
        read: it names them, a person acts."""
        return handoffs.unaccepted_overdue(self.store, self.policy)

    def list_actions(self, conversation_id: str | None = None):
        return self.store.list_actions(conversation_id)

    def list_disputes(self, conversation_id: str | None = None,
                      account_number: str | None = None):
        return self.store.list_dispute_cases(conversation_id, account_number)

    def get_dispute(self, case_id: str):
        return self.store.get_dispute_case(case_id)

    def list_retention_cases(self, conversation_id: str | None = None,
                             account_number: str | None = None):
        return self.store.list_retention_cases(conversation_id,
                                               account_number)

    def get_retention_case(self, case_id: str):
        return self.store.get_retention_case(case_id)

    def retention_report(self) -> dict:
        """The retention report over every stored case — counts
        from the case records only (retention_book)."""
        return retention_book.retention_report(self.store)

    def list_audit(self, conversation_id: str):
        return self.store.list_audit(conversation_id)

    def list_callbacks(self, conversation_id: str | None = None):
        return self.store.list_callbacks(conversation_id)

    # -- continuity (round 16) -------------------------------------

    def promise_ledger(self, account_number=None):
        """The promises ledger: every commitment on the book with
        its derived due state (pending / kept / broken), from the
        records — optionally for one account."""
        from . import continuity

        return continuity.promise_ledger(self.store, account_number)

    def conversation_continuity(self, conversation_id):
        """One conversation's continuity briefing: the snapshot
        delivered at verification when there is one, otherwise
        the briefing computed live from the account's records.
        None when the conversation doesn't exist."""
        conv = self.store.get_conversation(conversation_id)
        if conv is None:
            return None
        stored = (conv.specialist_state.get("continuity") or {}).get(
            "briefing")
        if stored is not None:
            return {"conversation_id": conversation_id,
                    "account_number": conv.account_number,
                    "briefing": stored, "source": "delivered"}
        if not conv.account_number:
            return {"conversation_id": conversation_id,
                    "account_number": "", "briefing": None,
                    "source": "none"}
        from . import continuity

        briefing = continuity.build_briefing(
            self.store, conv.account_number,
            exclude_conversation_id=conversation_id)
        return {"conversation_id": conversation_id,
                "account_number": conv.account_number,
                "briefing": briefing.model_dump(mode="json"),
                "source": "computed"}

    def list_procedure_runs(self, conversation_id: str | None = None):
        return self.store.list_procedure_runs(conversation_id)

    def list_roaming_notices(self, account_number: str | None = None):
        return self.store.list_roaming_notices(account_number)

    def list_payment_attempts(self, account_number: str | None = None):
        return self.store.list_payment_attempts(account_number)

    def list_payment_arrangements(self, account_number: str | None = None):
        return self.store.list_payment_arrangements(account_number)

    def list_payment_notices(self, account_number: str | None = None):
        return self.store.list_payment_notices(account_number)

    def list_complaints(self, conversation_id: str | None = None,
                        account_number: str | None = None):
        return self.store.list_complaints(conversation_id, account_number)

    def get_complaint(self, complaint_id: str):
        return self.store.get_complaint(complaint_id)

    def list_feedback(self, conversation_id: str | None = None,
                      account_number: str | None = None):
        return self.store.list_feedback(conversation_id, account_number)

    # -- the complaint watch and the mining pass ---------------------------

    def complaint_watch(self) -> dict:
        """The watch as a read: breached and near-breach open
        complaints, from the records and the calendar."""
        return care_book.complaint_watch(self.store, self.policy)

    def run_complaint_watch(self) -> dict:
        """The watch as an act: every breached complaint not
        already raised goes to the human queue with its record
        attached — the summary carries the category, the missed
        deadline and the customer's words — and the complaint
        record keeps the queue entry's id, so the watch never
        raises the same breach twice."""
        watch = care_book.complaint_watch(self.store, self.policy)
        raised = []
        for record in watch["breached"]:
            if record.sla_escalation_id:
                continue
            conv = self.store.get_conversation(record.conversation_id)
            customer = (self.store.get_customer_by_account(
                record.account_number) if record.account_number else None)
            # The watch's packet carries the round 17 elements
            # it honestly can: the customer's words are the
            # complaint itself, and the account's open promises
            # come from the ledger — this complaint among them.
            open_promises: list[str] = []
            if record.account_number:
                for entry in continuity.promise_ledger(
                        self.store, record.account_number):
                    if entry.state in ("pending", "broken"):
                        suffix = (" (overdue)" if entry.state == "broken"
                                  else "")
                        open_promises.append(
                            f"{entry.summary} — due {entry.due_date}{suffix}")
            packet = HandoffPacket(
                conversation_id=record.conversation_id,
                queue="general",
                reason="complaint_sla_breach",
                identity_state=(conv.identity_state.value if conv
                                else "unverified"),
                identity_attempts=(conv.identity_attempts if conv else 0),
                issue=record.text[:200],
                vulnerable=bool(customer and customer.vulnerable),
                customer_name=customer.full_name if customer else "",
                account_number=record.account_number,
                summary=(
                    f"Complaint {record.complaint_id} "
                    f"({record.category.replace('_', ' ')}) recorded "
                    f"{record.created_date} passed its response deadline "
                    f"{record.response_due} still open. The customer's "
                    f"words: {record.text[:160]}"),
                open_items=[
                    f"Complaint {record.complaint_id} awaits its "
                    f"substantive response "
                    f"(deadline {record.response_due})."],
                open_promises=open_promises,
                open_state_known=bool(record.account_number),
            )
            # Graded like every packet (round 17), as the watch
            # class: no live conversation, so no signals element
            # is required of it.
            handoffs.grade_packet(packet, conv.intent.value if conv else "")
            esc = Escalation(
                escalation_id=f"ESC-{uuid.uuid4().hex[:10]}",
                conversation_id=record.conversation_id,
                queue="general",
                reason="complaint_sla_breach",
                turn=conv.turns if conv else 0,
                created_date=today().isoformat(),
                packet=packet)
            self.store.save_escalation(esc)
            record.sla_escalation_id = esc.escalation_id
            self.store.save_complaint(record)
            raised.append(esc.escalation_id)
        return {"on": watch["on"], "raised": raised,
                "breached": [c.complaint_id for c in watch["breached"]],
                "due_soon": [c.complaint_id for c in watch["due_soon"]]}

    def mining_report(self) -> dict:
        """The unhandled-intent mining pass over every stored
        conversation — a report for the humans designing new
        intents. It creates nothing."""
        return mining.mine_unhandled_intents(self.store, self.policy)

    # -- failure attribution and the annotation queue ----------------------

    def run_attribution(self) -> dict:
        """The attribution pass: every conversation that has
        closed gets its outcome + cause record, computed from
        its audit trail. Conversations still open are left
        alone."""
        return attribution.run_attribution(self.store)

    def attribute_conversation(self, conversation_id: str):
        """Attribute one conversation — including one that
        merely stopped, which the operator declares ended by
        asking. Returns None when no such conversation exists."""
        return attribution.attribute_conversation(self.store, conversation_id)

    def list_attributions(self):
        return self.store.list_attributions()

    def attribution_report(self, days: int = 7) -> dict:
        """Cause × desk counts over a window with the previous
        window beside them, and the mining join: top unhandled
        candidates with their members' attributed causes."""
        return attribution.attribution_report(self.store, self.policy, days)

    def annotation_queue(self) -> list[dict]:
        """The review queue: escalated and unresolved
        conversations in full, resolved ones by the deterministic
        sample — each with its attribution and review state."""
        return attribution.annotation_queue(self.store, self.policy)

    def annotate(self, conversation_id: str, annotator: str,
                 cause_correct: bool, corrected_cause: str | None = None,
                 note: str = "", is_verdict: bool = False):
        """Write one annotation. Raises ValueError with the rule
        it broke — the API turns that into a 400."""
        return attribution.annotate(
            self.store, conversation_id, annotator, cause_correct,
            corrected_cause, note, is_verdict)

    def annotation_agreement(self) -> dict:
        """Annotator agreement (simple rate + Cohen's kappa over
        cause labels) and the disagreements awaiting a verdict."""
        return attribution.annotation_agreement(self.store)

    def list_annotations(self, conversation_id: str | None = None):
        return self.store.list_annotations(conversation_id)

    # -- conversation signals -------------------------------------------

    def list_signals(self, conversation_id: str):
        """One conversation's signal series, turn by turn — the
        latest record is the snapshot."""
        return self.store.list_signal_records(conversation_id)

    def signal_report(self) -> dict:
        """The signal aggregates over every stored conversation:
        effort by entry desk, rising-frustration share, the
        interventions' outcomes, and the descriptive
        frustration/escalation table."""
        return signals.signal_report(self.store, self.policy)

    # -- the daily quality pass (round 12) --------------------------
    def run_daily_quality(self, on=None, with_spot_check: bool = True):
        """Run the daily quality pass for one date (default: the
        shared clock's today) and store its digest. Re-running
        for a date replaces the digest and raises its version.
        The digest informs humans; it triggers no action."""
        from . import quality

        return quality.run_daily_pass(
            self.store, self.policy, on, with_spot_check=with_spot_check)

    def list_digests(self):
        return self.store.list_digests()

    def get_digest(self, digest_date: str):
        return self.store.get_digest(digest_date)

    def quality_trend(self) -> list[dict]:
        """Headline metrics across the stored digests."""
        from . import quality

        return quality.quality_trend(self.store)

    # -- the prompt registry (round 13) -----------------------------

    def list_prompts(self) -> list[dict]:
        """The registry's active set: one row per governed text —
        its surface, its active version, how many versions exist,
        and the pin standing on it, if any."""
        rows = []
        for entry in self.prompts.active_entries():
            pin = self.prompts.pin(entry.prompt_id)
            rows.append({
                "prompt_id": entry.prompt_id,
                "surface": entry.surface,
                "active_version": entry.version,
                "versions": len(self.prompts.versions(entry.prompt_id)),
                "pinned_version": pin.version if pin else None,
            })
        return rows

    def get_prompt(self, prompt_id: str) -> dict | None:
        """One entry in full: every version (old versions stay
        readable forever), the pin, and the act log for it."""
        versions = self.prompts.versions(prompt_id)
        if not versions:
            return None
        pin = self.prompts.pin(prompt_id)
        return {
            "prompt_id": prompt_id,
            "surface": versions[-1].surface,
            "active_version": self.prompts.active_version(prompt_id),
            "pinned_version": pin.version if pin else None,
            "versions": [e.model_dump(mode="json") for e in versions],
            "acts": [a.model_dump(mode="json")
                     for a in self.store.list_prompt_acts(prompt_id)],
        }

    def add_prompt_version(self, prompt_id: str, text: str,
                           actor: str = "", note: str = ""):
        """Add a version of an entry. The lint gates activation;
        a refusal raises PromptLintError and writes nothing."""
        return prompts_mod.add_version(
            self.store, self.policy, prompt_id, text, actor, note)

    def rollback_prompt(self, prompt_id: str, version: int,
                        actor: str = "", note: str = ""):
        """Roll an entry back: pin an older version active. The
        newer versions stay on the books."""
        return prompts_mod.rollback(
            self.store, self.policy, prompt_id, version, actor, note)

    def unpin_prompt(self, prompt_id: str, actor: str = "",
                     note: str = ""):
        """Lift a pin: the highest version activates again."""
        return prompts_mod.unpin(
            self.store, self.policy, prompt_id, actor, note)

    def conversation_prompts(self, conversation_id: str):
        """The exact registry texts that served one conversation,
        at the versions recorded on it — the trace from a
        historical reply to the words behind it. None when the
        conversation does not exist."""
        conv = self.store.get_conversation(conversation_id)
        if conv is None:
            return None
        rows = []
        for pid, version in sorted(conv.prompt_versions.items()):
            entry = self.prompts.entry_at(pid, version)
            rows.append({
                "prompt_id": pid, "version": version,
                "surface": entry.surface if entry else "",
                "text": entry.text if entry else "",
            })
        return rows

    # -- the incident feed (round 15) -------------------------------

    def list_incidents(self, status: str | None = None):
        """The operator incident feed, in feed order — optionally
        only the incidents in one status."""
        rows = self.store.list_incidents()
        if status:
            rows = [i for i in rows if i.status == status]
        return rows

    def get_incident(self, incident_id: str):
        return self.store.get_incident(incident_id)

    def add_incident(self, **fields):
        """Add an incident to the feed — a recorded operator act.
        A refused act (a duplicate id, a value outside the
        governed vocabularies) raises IncidentError and writes
        nothing."""
        return incidents.add_incident(self.store, **fields)

    def update_incident(self, incident_id: str, actor: str = "",
                        note: str = "", **fields):
        """Update an incident's governed fields — a recorded
        operator act."""
        return incidents.update_incident(
            self.store, incident_id, actor, note, **fields)

    def resolve_incident(self, incident_id: str, actor: str = "",
                         note: str = ""):
        """Resolve an incident — a recorded operator act. From
        this act on it matches nothing, including in
        conversations already open."""
        return incidents.resolve_incident(
            self.store, incident_id, actor, note)

    def list_incident_acts(self, incident_id: str | None = None):
        """The feed's act log — every add, update and resolve,
        in the order they happened."""
        return self.store.list_incident_acts(incident_id)

    def conversation_incidents(self, conversation_id: str):
        """One conversation's incident matches (incident id, the
        matched signals, the turn), each with the incident's live
        status — or None when the conversation does not exist."""
        conv = self.store.get_conversation(conversation_id)
        if conv is None:
            return None
        return incidents.conversation_matches(self.store, conv)

    # -- the turn ---------------------------------------------------------
    def post_message(self, conversation_id: str, text: str) -> TurnResult:
        conv = self.store.get_conversation(conversation_id)
        if conv is None:
            conv = Conversation(conversation_id=conversation_id)
        if conv.status == "escalated":
            return TurnResult(
                conversation_id=conv.conversation_id,
                reply=("This conversation has been transferred to a person "
                       "on our team — they'll pick it up from here with the "
                       "full summary attached."),
                intent=conv.intent.value,
                identity_state=conv.identity_state.value,
                status=conv.status, escalated=True,
                escalation_reason="already_escalated")
        try:
            result = self.supervisor.run_turn(conv, text)
        except Exception as exc:
            # A crashed turn is evidence, not silence: the trail
            # gets an internal-error event (failure attribution
            # reads it as the system_error cause), the conversation
            # is saved as far as it got, and the error still
            # propagates — nothing here swallows it.
            self.store.save_conversation(conv)
            self.store.save_audit(AuditEvent(
                event_id=f"EVT-{uuid.uuid4().hex[:10]}",
                conversation_id=conv.conversation_id, kind="system",
                detail={"event": "internal_error",
                        "error": f"{type(exc).__name__}: {exc}"},
                turn=conv.turns))
            raise
        self.store.save_conversation(conv)
        return result

    def export_transcript(self, conversation_id: str) -> bool:
        """Write the conversation transcript to object storage (the
        document of record a QA reviewer pulls). Returns False when
        S3 is not configured — the caller reports that honestly
        rather than pretending the export landed."""
        from .config import S3Config
        from .objects import put_text

        conv = self.store.get_conversation(conversation_id)
        if conv is None:
            return False
        lines = [f"Conversation {conv.conversation_id} — status {conv.status}, "
                 f"intent {conv.intent.value}, identity {conv.identity_state.value}"]
        lines += [f"[{m.turn}] {m.role}: {m.text}" for m in conv.messages]
        return put_text(S3Config.from_env(),
                        f"transcripts/{conversation_id}.txt", "\n".join(lines))

    # -- readiness ---------------------------------------------------------
    def readiness(self) -> dict:
        from .config import S3Config
        s3 = S3Config.from_env()
        return {
            "store": self.store.profile,
            "customers_seeded": self.store.count_customers(),
            "llm_backend": self.llm.backend,
            "s3_configured": s3.configured,
        }
