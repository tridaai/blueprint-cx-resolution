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
  a confirmation given before it. An "other" reply with no desk
  signal at all ("um, yes I think so") is an unread answer, not
  a new topic: the proposal stays open and the question is
  re-asked, rather than the turn routing to another desk and
  the reroute retiring the proposal unanswered (round 9 — the
  roaming guard applies the same rule to its deferred decision).
  A reply carrying any real signal still routes, and a desk
  change still retires.
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
| roaming | account roaming state, rate table, usage profile, offer catalogue, fair-use notices | `activate_offer` (Roam Pass) |
| offers | plan catalogue, loyalty offer, cycle calendar, change history | `change_plan` (and its reversal) |
| technical | procedure step-graphs, coverage/allowance checks | `book_callback` |
| account | account record, balance, attempt log, arrangements, processor double, autopay state | `update_payment_method`, `payment_arrangement`, `take_payment`, `refund_payment`, `set_autopay` |
| care | policy corpus retrieval, complaint records + SLA watch, feedback records, the account's accessibility record | — (answers, records, or logs an unhandled intent) |

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
throttled by the goodwill history. Roaming lines carry a second
evidence source — the account's own records — covered next.

## Roaming depth

Roaming works from a rate table, not from memory
(`roaming_book.py`). A trip quote multiplies the customer's
recorded typical daily use by the trip length and prices the
trip twice: pay-as-you-go at the destination rate, and under
the Roam Pass — its price plus the overage beyond its daily
allowance at the same rate. The desk recommends whichever is
cheaper and states both figures and the saving; every number is
computed in Decimal from the table, the profile and the
catalogue, so there is no figure for a model to invent.

Two guards hold in code. The **bill-shock guard**: when the
pay-as-you-go projection passes the policy threshold, the quote
turn carries no proposed action at all — the projection and the
comparison lead, and the purchase waits for the customer's next
message (the desk leaves an awaiting-decision marker that holds
the floor for that answer; an assent completes the deferred
purchase, a refusal closes it). The same guard fires from
actuals for a customer already abroad past the fair-use
ceiling, leading with the overage already run up. And
**fair-use notices**: usage accrues on the account record, and
crossing 80% or 100% of the ceiling raises a stored notice
whose summary — used, cap, projected overage cost at the
current pace — is composed from computed figures and shown in
conversation in the same act. A 100% crossing with no earlier
warning on record is flagged on the notice as a policy event.

That record is also the roaming dispute evidence. A disputed
roaming line is decided by reading the account: an executed
pass whose window covers the billed cycle stands the charge; a
fair-use warning dated on or before the cycle's close stands
it too (a warning raised inside the disputing conversation
itself does not count — it post-dates the usage); usage past
the ceiling with neither on record makes the line evidenced,
and it credits — or escalates above the dispute cap — through
the same machinery as any bill-marked line. The dispute case
records which evidence applied, line by line.

## Payments depth

A payment is an attempt on the record. The processor is a test
double, as the one-time code is: it reads the method on file and
answers approved, or declined with a reason code. Every run —
manual or autopay — writes a `PaymentAttempt` carrying the
amount, the method's last four, the code, and the class the code
falls into, decided in `payments_book.py`: **soft** (insufficient
funds, processor timeout — the card lives, and a retry can
succeed once the cause is fixed) or **hard** (card expired, card
reported lost, account closed — that method can never be charged
again). The class decides the conversation: the desk reads the
latest decline and gives that class's remedy, and a hard decline
standing against the current method is refused twice — the desk
will not propose the retry, and execution re-checks the attempt
log at the moment of the attempt and blocks it
(`hard_decline_retry_refused`) without the processor ever being
reached. Replacing the card clears the standing decline: it was
a fact about the old method, not about the customer. The
supervisor composes the payment reply from the attempt record —
paid, with the new balance, or the diagnosis — never from the
generic action text.

**Arrangements are records, not promises in prose.** Confirming
one writes a `PaymentArrangement` whose schedule is computed in
code under policy: at most three instalments, each at least the
policy minimum (the count falls until every slice clears it),
residue cents dealt to the earliest slices so the amounts sum
to the balance exactly, the first due within the policy window
and the rest monthly. No second arrangement opens while one is
active — re-checked at execution, not just at proposal. An
approved payment pours into the arrangement, earliest open
slice first. The state is derived from the record and the
calendar, never asserted: every slice covered is *completed*; a
due date past with its slice short is *broken*; otherwise
*active*. A broken arrangement is a human's case — the next
contact at the account desk escalates to the care queue with
the record attached (its id on the audit trail, its summary in
the handoff) before any other business, and the agent does not
renegotiate it.

**Refunds go to source.** Money owed back lives on the account
as a credit balance; a refund's amount is read from that record
and its destination is the payment method on file, named in the
proposal from the record. A destination the customer supplies —
another card, a bank transfer, a cheque — is refused in the
conversation with the fraud reason, and execution refuses it
again against the live account (`refund_destination_not_on_file`),
so no version of the request can move money anywhere else.

**Autopay is state on the account**, changed only as a
confirmed action, and its failures are proactive: the
billing-side run writes its attempt like any other, and a
decline raises a `PaymentNotice` whose summary — amount,
method, date, reason, balance still due, the class's remedy —
is composed from the computed figures, following the fair-use
notice pattern. At the customer's next contact the notice leads
the payments conversation, verbatim, and the diagnosis path
opens from it.

## Care depth

The care desk is the catch-all, and round 8 makes it the desk
that records (`care_book.py`). Three of the things a customer
says there are data, not filler:

**Complaints are records.** Recognition is a rule, not a model:
the customer's words carry a complaint marker, a governed
category list (staff conduct, appointment, billing, coverage,
service quality, other — first matching class in code order
decides) classifies it, and the record is written at once —
words verbatim, status open, and the response deadline computed
from policy (created + `complaint_response_days`). The reply is
the acknowledgment: reference, category, date. A customer
chasing an open complaint in the same conversation is answered
from its record — reference and standing deadline — rather than
recorded a second time. The **SLA watch** is a pure scan of the
open records against the shared clock, following the repo's
pattern for time-derived state (the arrangement book derives
its states the same way): breached and near-breach are computed,
never stored. Running the watch (API, CLI, console) raises each
breached complaint to the human queue as an escalation whose
summary carries the record — category, missed deadline, the
customer's words — and the record keeps the queue entry's id,
so a breach is raised exactly once. Near-breach complaints are
flagged in the complaints views for the team to see coming.

**Feedback is kept verbatim.** A compliment or a piece of
general feedback is recognised by an explicit marker, or by
praise aimed at a named subject, and recorded with its kind.
A bare "thanks, great" stays what it is — a pleasantry the
closing branch answers — because a record of every thank-you
would be noise, and the business mines the records.

**Accessibility needs live on the account.** A stated need
("send my letters in large print", "don't phone me, text me")
is recorded on the account record itself: the words verbatim in
`accessibility_note`, the derived instruction in
`preferred_contact`. If no account can be resolved, the desk
asks for the number and holds the words in the conversation
state until it arrives — the need is never dropped for lagging
paperwork. Later contacts read the record: a contact-shaped
question is answered with the honoured preference line, stated
as the commitment it became.

**Vulnerability is a discipline about what may be written
down.** The persistent flag on the customer record is set in
exactly one place — the supervisor's screen node — and from
exactly one evidence class: a circumstance the customer states
plainly, matched against the governed phrase classes in
`screening.py` (bereavement, serious illness, financial
hardship stated in the first person). The setting writes an
audit event with the class and its source. The older broad
keyword screen still routes *this* conversation to a person on
a mention, but it never writes the lasting record: tone, age,
apologies for slow typing and other indirect hints set nothing,
and the test suite proves both directions. A set flag routes
the conversation to the priority care queue, the handoff packet
carries `vulnerable` as a field with an open item naming where
the flag came from, and the flag persists — a later contact
with the same account escalates on the record alone.

**Unhandled-intent mining is a report, and only a report**
(`mining.py`). The pass reads every stored conversation and
takes one utterance from each that ended unhandled — the care
desk's own log first, else the first turn the router sent to
care on low confidence (read from the turn audit's routing
evidence), else the last ask before a no-progress escalation.
Utterances cluster greedily in conversation-id order against
the first member's embedding, using the repo's own hashing
embedder and cosine — no new machinery — at the policy
similarity; a cluster at or above the policy count is a
**candidate intent**, labelled by its most characteristic word
pair, with up to three verbatim samples. Determinism is a
property of the construction (fixed order, fixed tie-breaks),
and the suite asserts two runs agree exactly. The report is
served at `/care/mining`, in the console, and as
`cx-agent intent-mining`; nothing anywhere creates an intent
from it. Designing the intent stays a human's job — the pass
exists so the need is visible, counted, and quoted in the
customers' own words.

## Plan-change transactions

A plan change moves a customer between price points, so the
offers desk works to transaction rules the other desks don't
need (`plan_changes.py`). The **quote** is computed from the
account's real cycle calendar — `cycle_start` / `cycle_end` on
the record, the shared clock for today — by
`money.plan_change_for_cycle`: credit for the unused days of the
old plan at its daily rate, charge for the remaining days of the
new one, the net, and the projected next bill (the net adjustment
plus the first full month at the new price), every figure Decimal
and stated in the proposal before anything is confirmed.

**Effective dates are policy.** An upgrade (the effective price
rises) starts immediately, prorated as above. A downgrade starts
at the next cycle boundary: no proration, the customer keeps the
plan they've paid for, and confirming the proposal *schedules*
the change on the account (`pending_plan_id`,
`pending_plan_effective`) — the live plan does not move. A
downgrade or cancellation the customer blames on a fault is not
priced at all: it escalates to the billing queue, whose dispute
path owns fault-driven exits.

A proposal is a **versioned record stamped with a fingerprint**
of the account state it was priced against (plan, cycle dates,
balance, any scheduled change — the parts ride the payload next
to their hash). Execution re-derives the fingerprint from the
live account and refuses a mismatch as `stale_proposal`: the
supervisor marks the proposal superseded with the reason, and
the desk re-quotes the same request from live state, leading
with what moved, for a fresh confirmation. A "yes" can therefore
never spend figures the account has outgrown.

A completed change can be **reversed inside the policy window**
(`plan_change_reversal_days`; the same day by default) as a
**compensating pair**: one record restores the old plan (or
cancels the schedule, for a downgrade that never landed), a
second unwinds the money by the exact inverse of the original
net — both stamped with the original's id, which is itself never
edited. The window is re-checked at execution, not trusted from
proposal time. Outside the window the desk escalates to the
offers queue with the change record attached; the agent does not
improvise a reversal policy won't cover.

## The troubleshooting engine

Procedures are versioned data (`dataset.build_procedures`), and
one walker (`procedures.py`) interprets them: editing a
procedure's record — its steps, thresholds, branch table — is a
data change (a new version), never a code change. A walk is a
**procedure run**, a store record of its own holding the
procedure and version it pinned, the step it waits on, and a
step record for everything that happened; the conversation's
specialist state holds only the run's id. The step kinds:

- **tool_check** — the agent tests a condition itself (coverage,
  allowance, the network line test). Tools read the book and
  report a result word; the step's branch table routes on it.
  A tool with nothing to read reports its neutral result — it
  never invents one.
- **instruction** — the customer acts and reports success / fail
  / unclear, classified in code. A step whose check *is* the
  symptom disappearing (signal restored, speed restored, call
  connected) ends the walk on success; set-up steps (switching
  Wi-Fi calling on) advance.
- **measurement** — the customer reports a value (a speed test);
  it is parsed and compared against the step's threshold in
  code, and the comparison selects the path.
- **branch** — a named rule over the run's recorded results
  (e.g. the last speed against a hard threshold) picks the
  next step.
- **terminal** — resolved, book_callback, or
  escalate_field_visit (an engineer visit is a human's booking;
  the conversation escalates with the record).

A step flagged `mandatory_safety` (the broadband power-cycle)
completes only on success: a skip attempt or an outright refusal
is refused in code, recorded on the run, and the step stays
current; a second insistence ends the walk and escalates. If the
conversation leaves the desk mid-procedure — a billing question,
say — the run simply waits: routing hands outcome-shaped replies
back to the technical desk, an identity detour raised by another
desk is dropped when the customer returns to the walk, and the
walk resumes at its exact step with its history intact. If the
conversation escalates, the run closes as abandoned, history
attached.

Exhausting a procedure books its callback as a **callback
record** — the sibling of the dispute case — created at the
dead end with the run's full step transcript and settled when
the customer answers the booking proposal (booked, with the
executed action's id, or declined / blocked). The console and
`/callbacks` show the record, so the technician starts from the
transcript, not from zero.

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

The eval pack grades trajectories, not replies: 127 scripted
multi-turn conversations, each attributed to a persona and a
rubric, with assertions on the observable behavior per turn and on
final store state (balance after a credit, plan after a change,
the escalation queue, the reroute audit count, action-record
statuses). Eighty-one are hand-written; the rest are generated
and attacked (round 9, below). The report breaks the results
down by rubric, by persona, and by adversarial class, so a
regression localises to a customer style or an attack type. The trajectories include the round-2 desk discipline:
a proposal retired by a reroute and never executed, verification
surviving a change of desk, a clarification resolved by its
answer and one ignored and committed on without a second
question. Round 3 adds the billing outcomes: a partial dispute
settling its case with exactly the evidenced line credited, a
goodwill refusal that moves no money, a bill-shock lead, and an
over-cap dispute escalated with its case intact. Round 4 adds
the technical rubric: a line-test result branching to a field
visit, a reported speed taking the fault path below threshold
and the resolved path above it, a dead-ended procedure whose
callback record carries every step's result, a walk resumed at
its exact step after a billing detour, and a safety step that
refuses a skip. Round 5 adds the plan-change transactions: an
upgrade priced to the cent off the cycle calendar, a downgrade
that schedules at the boundary and moves nothing, a proposal
the world outruns — refused as stale, re-quoted, and only then
executed — and a reversal inside its window restoring plan and
money as a compensating pair, with the same request outside the
window routed to the offers queue. Round 7 adds the payment
trajectories: a soft decline diagnosed by class and retried to
success after the fix, a hard decline whose same-card retry is
refused in the conversation and never executes, an arrangement
whose three instalments match the balance to the cent and whose
second arrangement is refused, a missed instalment deriving the
arrangement broken and routing the next contact to a human with
the record, a refund refused to a customer-named bank account
and executed to the card on file, and a failed autopay whose
notice figures lead the next contact. Round 8 adds the care
trajectories: a complaint recorded with its category and
deadline and acknowledged in the reply, the same complaint
breach-aged by a poke and raised by the watch, a compliment
kept verbatim, an accessibility need honoured at a later
contact, an explicit vulnerability statement that sets the flag
and routes to the care queue with the flag in the packet, a
hinted one that sets nothing, and a planted set of repeated
unhandled asks surfacing as one mined candidate with the right
count. Round 9 adds the persona matrix and the adversarial set.
The matrix (`evals/personas.py`) applies seven persona
archetypes — confused and repetitive, terse, rambling, non-native,
mind-changing, wall-of-text, careful elderly — to five core flows
and grades the identical invariants per pair: how a customer talks
must not change what the agent does. The adversarial set
(`evals/adversarial.jsonl`) works four attack classes —
injection (a fake system notice in a dispute, a fake email, a
prompt-extraction attempt, a claimed staff identity), impersonation
(a spouse, an "on behalf of" caller), social engineering (a
claimed regulator, deadline urgency), and cross-account probes
(fishing while verified, while unverified, and a mid-chat account
switch) — and is zero-tolerance as a set: one failed assertion
anywhere in it fails the gate. The matrix earned its keep on its
first run: the confused persona's hesitant "um, yes I think so"
destroyed every pending decision it touched — routed as a new
topic, the desk change retired the proposal, and the customer's
later plain "yes" confirmed nothing. Two decision points shared
the flaw (the supervisor's confirmation protocol and the roaming
guard's deferred assent); both now re-ask a signal-less
non-answer and keep the decision open, and the persona's four
transaction flows are the regression proof. The rubrics encode the
production rules:

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
- **care** — the catch-all answers what it can, records what it
  must (complaints with their deadlines, feedback verbatim,
  needs on the account), and logs what it can't — and the
  unhandled logs mine into candidate intents.
- **technical** — procedures branch on tool results, measurements
  decide in code, dead ends book transcript-carrying callbacks,
  safety steps cannot be skipped.

Zero-tolerance assertions (escalation decisions, action
execution, money/plan state, identity state, and the procedure
record-keeping — step order, callback presence, transcript
contents) must pass at 100%; the overall bar is 95%. The pack runs offline by default in about
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

## Failure attribution and the annotation queue

A conversation's audit trail already records how it ended; round
10 makes that a record of its own. Attribution is a rule chain
over the trail and the sibling records — audit events, action
records, dispute cases, procedure runs, the closing escalation —
evaluated in a fixed order, because the order is the argument: an
internal-error event (written by the service when a turn crashes)
outranks everything; a verification lockout outranks the
desk-level stories; a policy cap or refusal outranks the
procedure's account of itself; a walk that exhausted, escalated
from inside a procedure, or stalled into a no-progress escalation
is a procedure dead end; and only then may an escalation be
called what it often is — designed behaviour (a threat, a fraud
report, a legal matter, a vulnerability, the customer's second
ask for a human), which earns no cause. What remains reads, in
order: verification abandoned mid-step, a conversation rerouted
twice or more with nothing executed, a need the care desk logged
as unhandled (or the router's low-confidence fallback), and work
left pending — a proposal unanswered, a walk mid-step — which is
the customer leaving. The outcome is read alongside: escalated
from the record; resolved-with-action when something executed
and nothing is pending; resolved; and unresolved-ended for
everything else, including a record marked resolved whose last
word is still an unanswered proposal. Each record names the rule
that fired and carries the signals it read, so the attribution
can be re-derived — or disputed — from the record alone.

Two operator acts write records, in the pattern of the complaint
watch: a pass attributes every conversation that has closed and
has no record yet (a record, once written, stands), and a
single-conversation attribution covers the ones that merely
stopped, which is how a walk-away gets its record. The report
aggregates by cause × desk over a window with the previous
window beside it, reconciles exactly against the records, and
joins the unhandled-intent mining candidates to the attributed
causes of their member conversations — the two failure lenses
reading the same evidence.

The annotation queue is the human loop over the records: every
escalated or unresolved conversation is queued, plus resolved
ones by a deterministic sample (a stable hash of the
conversation id against a policy modulus), so the successes are
audited too without drowning the reviewers. An annotation marks
the attributed cause right, or corrects it from the same
taxonomy (a correction is required exactly then); a second
annotation must come from a different annotator; agreement
between the first two is reported as a simple rate and as
Cohen's kappa over the cause labels; and where they disagree, a
third person's annotation — the verdict — settles the record
and closes it. No automation reads any of it. It is the
evidence base from which humans fix routing, raise a cap, or
design the intent the mining pass keeps pointing at.

## Round map

This document describes rounds 1–10 (core; rerouting and
clarification; billing depth; the troubleshooting engine;
plan-change transactions; roaming depth; payments depth;
customer care depth and unhandled-intent mining; persona and
adversarial eval expansion; failure attribution and the
annotation queue). The scheduled deepening — the retention
flow, conversation-signal extraction, the daily quality pass,
the prompt registry, handoff scoring, cost budgets — is laid
out in the blueprint backlog; the architecture above is shaped
so each lands as a new node, desk behavior, or report over the
same records, not as a rewrite.
