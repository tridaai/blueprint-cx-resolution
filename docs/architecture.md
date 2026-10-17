# Architecture — CX Resolution Agent

How the agent is put together, and why. The design follows the
convergent pattern in published production CX write-ups (a
supervisor with specialist agents, behavior rubrics as evals,
escalation as a designed behavior, observability from day one) —
implemented here from scratch, for a fictional carrier, on the
standard Trida blueprint spine.

## The shape

```
customer ─▶ FastAPI ─▶ ConversationService ─▶ Supervisor (LangGraph)
                                                   │
        ┌──────────────┬──────────────┬────────────┼────────────┬──────────────┐
        ▼              ▼              ▼            ▼            ▼              ▼
     billing       roaming        offers      technical    account         care
        └──────────────┴──────────────┴────────────┴────────────┴──────────────┘
                                                   │
                              structured findings + proposed action
                                                   │
                     supervisor composes ONE reply (LLM boundary)
                                                   │
                     action gates (identity · confirmation · policy)
                                                   │
        store (PostgreSQL + pgvector │ SQLite │ in-memory) · audit · S3 exports
```

Two ideas do most of the work:

1. **Specialists return data, the supervisor returns prose.**
   A specialist works with tools and returns a `SpecialistResult`:
   structured findings, at most one typed proposed action, the
   policy citations its claims rest on, and verbatim lines (figures,
   warnings) that must survive composition. Only the supervisor
   drafts the customer-facing reply, exactly once per turn. This is
   the production lesson about duplicated formatting: when every
   specialist formats its own output, cost and inconsistency
   compound; when the supervisor formats once, the reply has one
   voice and one place where guarantees are enforced.

2. **Every consequential decision is a code path.**
   Routing is a weighted phrase table. Identity is a state
   machine. Escalation is an ordered rule list evaluated before
   anything composes. Money is Decimal arithmetic in `money.py`.
   Actions execute through one gate function that re-checks
   identity, confirmation and policy at execution time. The model
   drafts language and (in live mode) may hint at intent when the
   router is tied — it decides nothing.

## The supervisor graph

Nodes, in turn order:

- **screen** — `screening.screen()` reads the raw customer text for
  prompt-injection patterns and for the signal classes the
  escalation rules consume (threat, fraud, legal, vulnerability,
  ask-for-a-human). Injection attempts are audit-logged; the turn
  then proceeds as ordinary service, so an injected instruction is
  never routed, quoted into a prompt, or acted on.
- **dispatch** — an ordered choice: an escalation decision already
  made → `escalate`; an identity flow in progress → `identity`; a
  proposed action awaiting its answer → `confirm`; otherwise →
  `route`.
- **identity** — runs the verification conversation. Expected
  input is parsed with closed patterns (account number, 5-digit
  postcode, 6-digit code). Success at the code step resumes the
  original request in the same turn, at the desk that asked
  (`force_intent`), so the customer never repeats themselves.
- **confirm** — classifies the reply against the pending action
  (confirm / decline / other, closed vocabularies in
  `routing.py`). Confirming a consequential action while
  unverified detours into the identity flow and re-asks the
  confirmation afterwards — verification never silently satisfies
  a confirmation given before it.
- **route** — the phrase-table router, read as evidence
  (`routing.analyze`): matched signals, the leaders, the runner-up.
  A live procedure owns short outcome replies ("still nothing")
  that carry no routing signal. Two desks tied on a real signal
  short-circuit to `post` with the supervisor's own clarifying
  question — no desk is guessed, and no specialist runs that turn.
  Every decision lands on the turn's audit event.
- **specialist** — the desk runs (see below). One behaviour is the
  supervisor's own: on a first ask-for-a-human routed to the
  catch-all, the reply is a genuine solve attempt (an invitation
  to describe the problem), per the escalation rubric.
- **post** — bookkeeping: identity hand-offs (awaiting state +
  resume text), specialist-driven escalation, action proposal
  recording, the no-progress counter, and the no-progress
  escalation when the counter reaches the policy cap.
- **compose** — builds a `DraftContext` (findings, verbatim lines,
  the confirmation ask, citations) and calls the LLM boundary.
  Afterwards the code enforces three guarantees on whatever came
  back: every verbatim line is present, the confirmation ask is
  present when an action pends, and cited policy documents appear
  as a Sources line. A live drafter that drops a figure gets it
  appended, not trusted.
- **escalate** — builds the handoff packet from the record, saves
  the escalation, closes the conversation to the agent, and tells
  the customer their reference. The packet is never written by the
  model: it is assembled from the conversation, the action log
  and the audit trail.
- **finalize** — appends both messages, writes the turn audit
  event, and emits the `TurnResult`.

Conversation state lives in the store, not in graph memory: the
API is stateless, the graph is invoked per turn over the loaded
record, and any process can serve any turn.

## Rerouting and clarification

The conversation's **desk** is state of its own, distinct from the
per-turn intent: the desk is the specialist that owns the
conversation until a routed turn says otherwise. When a routed
turn lands on a different desk — a billing chat that turns into a
fault, a roaming question inside an account chat — the move is a
**reroute**, written to the audit as an event of its own
(from-desk, to-desk, trigger turn, the signals that caused it).
What survives a reroute is everything that belongs to the
conversation rather than the desk: the identity state (a verified
customer is never asked to verify again because the topic moved),
the message history and specialist findings so far, and the
executed-action record.

One thing deliberately does not survive: a proposal the old desk
left awaiting its answer. The moment the desk changes, the
pending proposal is retired — its record stays on the books as
`superseded`, with the reason — and it can only ever run after
being proposed again and confirmed again at the new desk. This
closes the stale-yes hole in any design where confirmations are
conversational: a "yes" spoken to the new desk's question can
never be spent on the old desk's price.

When the router cannot honestly separate two readings — exactly
two desks tie at the top on a real signal (the threshold is in
`routing.CLARIFY_MIN_SCORE`, and the care catch-all never takes
part in a tie) — the supervisor asks **one** clarifying question
offering the two readings in plain words ("your bill", "a fault
with your service"), and holds the desk unchanged while it is
open. The next routed turn answers it first: a reply that picks
a reading (by signal, or by the words people name desks with)
routes there, and the resolution counts as progress in the
no-progress bookkeeping. A reply that settles nothing is
committed on — the router takes its best reading, or hands the
turn to care when there is nothing to read — and no second
question is asked. Symmetrically, a turn with no clear signal at
all (at most one weak phrase matched) routes to the care desk
with `low_confidence_care` on its routing record, never to a
transactional desk on a guess.

## The desks

| Desk | Tools it works with | What it can propose |
|---|---|---|
| billing | bill history, dispute cases, credit policy + trailing credit history | `apply_credit` (dispute or goodwill class) |
| roaming | account roaming state, offer catalogue, fair-use policy | `activate_offer` (Roam Pass) |
| offers | plan catalogue, loyalty offer, proration engine | `change_plan` |
| technical | procedure step-graphs, coverage/allowance checks | `book_callback` |
| account | account record, balance, arrangement policy | `update_payment_method`, `payment_arrangement` |
| care | policy corpus retrieval | — (answers, or logs an unhandled intent) |

Dispute evidence is a property of the bill, computed in code:
a charge marked as a duplicate of another, a charge levied after
cancellation, or a charge above its expected rate. The billing
desk decides each disputed line on that evidence — evidenced
lines credit for exactly the evidenced amount (the excess only,
on a wrong rate); lines the customer names but the record clears
decline with the reason in plain words — and persists the whole
assessment as a dispute case that settles when the customer
answers the proposed credit (decided_credit, decided_partial,
declined_with_reason, escalated_above_cap). Explanations are
computed the same way: the bill decomposes into charge families
with totals and month-on-month deltas against the previous bill,
and a total that jumps the bill-shock threshold over the trailing
average leads with the family the deltas indict. Goodwill runs on
a separate tier with a memory: the account's executed credits in
the trailing 90 days bound what may be offered, a refusal is
recorded with its reason, and the evidenced path is never
throttled by the goodwill history.

Procedures are data (`dataset.build_procedures`): ordered steps,
each with the condition that resolves it and an `on_fail` path
(next / callback). Steps the agent can test itself (coverage,
allowance) run as tools and never reach the customer as questions.
A step whose check *is* the symptom disappearing (signal restored,
speed restored, call connected) ends the procedure on success;
set-up steps (switching Wi-Fi calling on) advance on success.

## Identity

```
UNVERIFIED ── account + postcode match ──▶ VERIFYING
VERIFYING  ── correct one-time code ────▶ VERIFIED
VERIFYING  ── max attempts wrong ───────▶ FAILED ──▶ escalate
```

Two disclosure tiers fall out of the machine: **identification**
(the VERIFYING state, factors matched) unlocks account-specific
answers; **verification** (VERIFIED) unlocks consequential
actions. The one-time code in this build is a test double —
`identity.expected_code` derives it deterministically from the
account number and a deployment secret so the full flow is
testable end to end; only its hash is stored on the conversation.
Production swaps the double for an SMS/email sender behind the
same two functions.

## Actions

A proposal is a typed record with a customer-readable summary, a
code-built payload, the money involved, and its basis (the
evidence or rule it rests on). Execution (`actions.execute`)
re-checks at the moment it runs:

1. consequential kinds require `VERIFIED` identity;
2. consequential kinds require `confirmed=True`, which the
   supervisor passes only for the explicit confirmation turn;
3. kind-specific policy: credit amount against the cap for its
   class (dispute vs goodwill), offer eligibility against the
   live plan, arrangement against the live balance floor, card
   last-4 shape.

Blocks return a record with a reason — a normal outcome the
conversation explains — never an exception. Executions mutate the
account through the store and write audit events; the console's
escalation queue and the action log are views over those records.

## Grounding

Policy retrieval is scoped by corpus section: a desk searches its
own section (billing searches billing), the way a human agent
reaches for the right shelf. The composer may only cite documents
retrieval returned for that turn, and the citation is enforced in
code after drafting. The embedder is a deterministic hashing
embedder with stopword removal and suffix stemming (384
dimensions, one contract on every store); pgvector holds the
vectors in production, JSON in SQLite, with the same cosine in
code — rankings, and therefore eval results, do not drift between
profiles. A model embedder can replace it behind `embed`/`cosine`.

## Evals as behavior rubrics

The eval pack grades trajectories, not replies: 41 scripted
multi-turn conversations, each attributed to a persona and a
rubric, with assertions on the observable behavior per turn and on
final store state (balance after a credit, plan after a change,
the escalation queue, the reroute audit count, action-record
statuses). The trajectories include the round-2 desk discipline:
a proposal retired by a reroute and never executed, verification
surviving a change of desk, a clarification resolved by its
answer and one ignored and committed on without a second
question. Round 3 adds the billing outcomes: a partial dispute
settling its case with exactly the evidenced line credited, a
goodwill refusal that moves no money, a bill-shock lead, and an
over-cap dispute escalated with its case intact. The rubrics
encode the production rules:

- **escalation** — first ask is an attempt, second is a transfer;
  stuck is a transfer; threat/fraud/legal/vulnerable are immediate.
- **education** — the agent explains and teaches (bill anatomy,
  proration, guided fixes) before it ever needs to escalate.
- **grounding** — policy claims carry the right citation; a
  question with no policy basis gets an honest "not in my guides",
  never an invention.
- **transaction** — proposals price exactly, executions land
  exactly, declines change nothing, the unverified get nothing.
- **identity** — the machine's states, including lockout.
- **care** — the catch-all answers what it can and logs what it
  can't.

Zero-tolerance assertions (escalation decisions, action
execution, money/plan state, identity state) must pass at 100%;
the overall bar is 95%. The pack runs offline by default in about
four seconds, which is what makes it a gate rather than a report.

## Stores and deployment

One store protocol, three implementations (PostgreSQL + pgvector
in production via numbered SQL migrations; SQLite as the zero-
setup local default; in-memory as a test double). Selection is a
single env decision (`DATABASE_URL`). The compose topology runs
the stateless API beside Postgres and SeaweedFS's S3 gateway
(service name `minio`, the S3 API on :8333); transcript exports
are the object store's first job. Health is liveness; readiness
reports the store profile, whether the book is seeded, the live
LLM backend, and whether S3 is configured.

## Failure posture

- A live drafting failure degrades that turn to the offline
  composer; the rules do not degrade with it.
- A blocked action is explained, not retried silently.
- An escalated conversation refuses further agent turns; a human
  owns it, and the refusal says so.
- Retrieval below its similarity floor is treated as "no policy
  basis" — the honest answer is an unhandled-intent log and an
  offer of a human, never a plausible-sounding stretch.

## Round map

This document describes rounds 1–3 (core; rerouting and
clarification; billing depth). The scheduled deepening — the
retention flow, failure attribution and the annotation queue,
conversation-signal extraction, the daily quality pass, the
prompt registry, handoff scoring, cost budgets — is laid out in
the blueprint backlog; the architecture above is shaped so each
lands as a new node, desk behavior, or report over the same
records, not as a rewrite.
