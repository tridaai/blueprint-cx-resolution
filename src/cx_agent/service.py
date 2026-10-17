"""ConversationService: the one door into the agent.

The API, the demo, the console and the eval runner all come through
here. The service owns conversation records (load → run one graph
turn → save) and refuses work the record forbids — an escalated
conversation is closed to the agent; a human owns it now.
"""

from __future__ import annotations

import logging
import uuid

from . import care_book, mining
from .clock import today
from .config import LLMConfig, Policy
from .dataset import seed_store
from .llm import make_llm
from .models import Conversation, Escalation, HandoffPacket, TurnResult
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

    def list_actions(self, conversation_id: str | None = None):
        return self.store.list_actions(conversation_id)

    def list_disputes(self, conversation_id: str | None = None,
                      account_number: str | None = None):
        return self.store.list_dispute_cases(conversation_id, account_number)

    def get_dispute(self, case_id: str):
        return self.store.get_dispute_case(case_id)

    def list_audit(self, conversation_id: str):
        return self.store.list_audit(conversation_id)

    def list_callbacks(self, conversation_id: str | None = None):
        return self.store.list_callbacks(conversation_id)

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
            packet = HandoffPacket(
                conversation_id=record.conversation_id,
                queue="general",
                reason="complaint_sla_breach",
                identity_state=(conv.identity_state.value if conv
                                else "unverified"),
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
            )
            esc = Escalation(
                escalation_id=f"ESC-{uuid.uuid4().hex[:10]}",
                conversation_id=record.conversation_id,
                queue="general",
                reason="complaint_sla_breach",
                turn=conv.turns if conv else 0,
                packet=packet,
                created_date=today().isoformat())
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
        result = self.supervisor.run_turn(conv, text)
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
