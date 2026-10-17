"""ConversationService: the one door into the agent.

The API, the demo, the console and the eval runner all come through
here. The service owns conversation records (load → run one graph
turn → save) and refuses work the record forbids — an escalated
conversation is closed to the agent; a human owns it now.
"""

from __future__ import annotations

import logging
import uuid

from .config import LLMConfig, Policy
from .dataset import seed_store
from .llm import make_llm
from .models import Conversation, TurnResult
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
