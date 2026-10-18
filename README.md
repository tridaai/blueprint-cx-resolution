<p align="center">
  <img src="https://trida.ai/logos-rounded/trida-rounded.png" alt="Trida AI" width="96">
</p>

# Blueprint — CX Resolution Agent

A customer support agent that only answers questions is the easy
version of this problem. The production version — the one the
published field guides from large CX teams keep circling back to —
has to *resolve*: explain a bill and credit what's wrong on it,
change a plan at the right prorated price, walk a customer through
a fault step by step, and know, in code rather than in vibes, the
exact moment to stop and hand the conversation to a person.

This blueprint is that agent, built for a fictional regional
carrier, **Northbeam Mobile**. A LangGraph supervisor owns each
conversation and routes it to one of six specialist desks —
billing, roaming, offers & plan changes, technical support,
account & payments, and a customer-care catch-all — which work with
tools and return structured findings. Only the supervisor speaks to
the customer, and everything that matters is decided in code:
identity is a state machine, escalation follows written rules,
money is Decimal arithmetic, and a consequential action executes
only with a verified identity, the customer's explicit
confirmation, and a fresh policy check at the moment it runs.

> Reference prototype. Not connected to any real carrier, billing
> system, or customer.

> **Work in progress** — this blueprint is built and passing its automated test and eval gates, but it has not yet finished hands-on local testing, and its companion blog post is not yet published. Expect changes. This notice comes off when both are done.

## Run it with a real model

The agent's language work — drafting the reply from the findings,
reading free-text nuance — is built for a real LLM. Set one
provider and it uses it; the rules around the model are code
either way, and every figure in a reply was computed by a tool,
never by the model.

```bash
cp .env.example .env   # then set ONE provider
```

- **OpenAI-compatible** — `OPENAI_API_KEY`, plus `OPENAI_BASE_URL`
  for any compatible endpoint: NVIDIA NIM hosted
  (`https://integrate.api.nvidia.com/v1`) or self-hosted
  (`http://localhost:8000/v1`), vLLM, and so on. `OPENAI_MODEL`
  picks the model.
- **Anthropic** — `ANTHROPIC_API_KEY` (and optionally
  `ANTHROPIC_MODEL`).
- **Ollama** — `OLLAMA_BASE_URL` for a local model.

With nothing set, the agent runs on a labelled **offline
fallback**: a deterministic composer that assembles the same
findings into a reply. It exists so the system, the tests and the
eval gate run with no key — it is not a substitute for a model on
real customer language, and `/readiness` reports which mode is
live.

## Quickstart

With [uv](https://docs.astral.sh/uv/):

```bash
make install     # uv sync --extra dev
make demo        # six scripted conversations, offline, ~1 minute
make test        # pytest
make evals       # the rubric trajectory gate
make serve       # API + console on http://localhost:8000
```

No uv? The pip fallback installs the frozen, hash-verified
dependency set first and the package second (the two steps never
mix — pip refuses an editable install under a hashed constraints
file):

```bash
python3 -m venv .venv && source .venv/bin/activate
make install-pip
```

Production topology, locally:

```bash
docker compose up --build -d
docker compose exec api cx-agent seed   # write the Northbeam book
open http://localhost:8000              # the operator console
```

## What the agent does

- **Billing** — decomposes the latest bill by charge family from
  the record and compares it with the previous bill; when the
  total jumps past the bill-shock threshold over the trailing
  average, the explanation leads with the computed driver.
  Disputes become cases: each disputed line is decided on its own
  evidence (a duplicate charge, a charge after cancellation, a
  wrong rate) and credited, declined with the reason in plain
  words, or escalated above the agent's cap with the case record
  intact. Goodwill credits follow their own smaller cap — and a
  rolling cap over the account's trailing 90 days of credit
  history, read in code before any offer is made.
- **Roaming** — prices a trip before it's taken: the destination
  rate, the trip length and the customer's own usage profile give
  a pay-as-you-go figure and a Roam Pass figure (allowance and
  overage included), and the cheaper one is recommended with the
  saving stated. A projection past the bill-shock threshold
  leads the conversation and defers any chargeable proposal a
  turn; the same rule fires from actual usage when the customer
  is already past the fair-use ceiling. Usage accrues on the
  account, and crossing 80% or 100% of the ceiling raises a
  stored fair-use notice whose summary — used, cap, projected
  overage at the current pace — reaches the customer in the same
  act. A crossing with no earlier warning on record is flagged
  on the notice, and it decides roaming disputes: a charge past
  the ceiling with no warning given is evidenced and credits
  through the dispute machinery; a charge inside an active pass
  window, or after a recorded warning, stands — with the
  evidence noted on the case either way.
- **Offers & plan changes** — eligibility and proration computed
  in code, off the account's real cycle calendar: the customer
  sees the credit, the charge, the net and the projected next
  bill before confirming anything. Upgrades start now; a
  downgrade schedules at the next cycle boundary with no
  proration. A change can be reversed the same day it was made;
  a downgrade blamed on a fault is routed to the billing desk's
  dispute path, not priced as an exit.
- **Retention** — wanting to leave opens a case: the reason
  classified in code against a governed list (price, coverage,
  service quality, moving house, competitor, no longer needed,
  other), the customer's words kept verbatim. At most one save
  offer per case, ever, and built only from mechanisms already
  in the book — a downgrade priced by the plan-change
  machinery, a goodwill credit under the credit policy, or the
  loyalty upgrade price where the customer is eligible — and
  none at all when the customer says they want none, or when
  no mechanism honestly answers the reason (a house move gets
  no invented discount). Declining the offer, or asking to
  cancel a second time, goes straight to the cancellation:
  a confirmed action that takes effect at the end of the
  cycle, so the customer keeps what they paid for until then.
  A fault-blamed exit follows the billing desk's dispute path
  with the routing recorded on the case, and the cases policy
  puts with humans settle as escalated to the retention team.
  The report (`cx-agent retention-report`, `GET
  /retention/report`) counts cases by reason, offers made and
  accepted, the save rate over decided cases, and
  cancellations completed — from the case records only.
- **Technical support** — guided procedures as versioned data,
  walked by one engine: agent-side checks (coverage, allowance,
  a network line test) run as tools whose results branch the
  walk, customer-reported measurements are compared against
  thresholds in code, and mandatory safety steps refuse to be
  skipped. Every walk is a stored run with its full step
  history — leave mid-procedure and it resumes at the exact
  step — and exhausting the steps books a technician callback
  as a record carrying the whole step transcript, instead of
  improvising.
- **Known-issue awareness** — operations keeps an incident
  feed: governed records carrying the affected service, the
  postcodes covered, the declared symptom signals, the status,
  and a customer-safe summary in the operator's own words.
  Matching is deterministic code, not model judgement: when a
  customer's report is explained by a live incident (active or
  monitoring — never a resolved one), the agent acknowledges
  the known issue up front in the record's words, starts no
  troubleshooting procedure for that symptom, proposes nothing
  unrelated, and offers the book's callback follow-up instead.
  The match is audited with its signals, rides the handoff
  packet when the conversation escalates, and stops the moment
  the incident is resolved — even mid-conversation.
- **Account & payments** — card replacement behind full
  identity verification, and payments worked as records: every
  charge attempt lands on a log with the processor's reason
  code, and the code's class decides what happens next — a soft
  decline (insufficient funds, a processor timeout) may be
  retried once the cause is fixed, while a hard decline (an
  expired or lost card, a closed account) is never retried, in
  the conversation or at execution. Payment arrangements are
  stored schedules — amounts and dates computed in code, the
  total equal to the balance to the cent — that payments pour
  into and the calendar marks completed or broken; a broken
  arrangement goes to a human with its record attached.
  Refunds go back to the payment method on file and nowhere
  else — a destination the customer names is refused with the
  fraud reason — and the amount is read from the account, never
  from the conversation. Autopay is a confirmed setting on the
  account; a failed automatic payment raises a notice carrying
  its figures, which is the first thing the customer hears
  about at their next contact. Only the last four digits of a
  card ever appear.
- **Customer care** — the catch-all desk for legitimate needs
  outside the five, answering from the policy corpus with
  citations; a need it cannot serve is logged as an unhandled
  intent, not papered over. The desk also records: a complaint
  becomes a stored record the moment it's recognised —
  category from a governed list, the customer's words verbatim,
  and a response deadline computed from policy (five days) that
  an SLA watch reads; a complaint still open past its deadline
  is raised to the human queue with its record attached, and a
  customer chasing their complaint is answered from the record,
  not recorded twice. Compliments and feedback are kept
  verbatim, as data. An accessibility need a customer states —
  large print, email instead of letters — is written on the
  account and honoured at every later contact. And the
  conversations that ended unhandled are mined offline into
  candidate intents for a human to design (`cx-agent
  intent-mining`) — a report, never an auto-created intent.
## How the conversation is going

Outcomes are only half of what a conversation records. After
every customer turn the agent also computes — in code, from the
conversation's own records, never by asking a model how the chat
feels — four **signals**: **frustration**, scored per message
against a governed negative-affect lexicon, with the trend read
off the series (rising means the latest message is the
customer's most frustrated yet, and it built over at least two
turns); **effort**, the counted cost of the conversation so far
— a point per turn, per identity attempt, per reroute, per
repeated ask, the components kept beside the score; **progress**,
whether the turn landed; and **confusion**, said outright or the
same question asked again, pinned to the topic of the last
explanation. The series persists per conversation
(`GET /signals/{conversation}`), and three rules act on it:

- **Frustration rising, with no progress** → the supervisor
  **offers a person now**, in the reply. An offer, not a
  transfer: yes takes it, no closes it for the rest of the
  conversation, anything else lets it lapse. It fires once, and
  never on a turn carrying a decision of its own.
- **Identity effort at the governed threshold** (two failed
  codes) → the same early offer, mid-verification. Policy
  defines no alternative verification path, so a human is the
  governed alternative to grinding through codes — and a third
  failure still locks, offer or not.
- **Confusion on a topic** → the desk serves that topic's
  **second explanation form** — a summary-first restatement or a
  worked example, written for the top topics — never the first
  form's words again, and never twice.

Every firing and every answer is an audit event. When a
conversation does escalate, the handoff packet carries the
signal summary — effort, the frustration trend, the
interventions tried and how they were answered — so the human
sees the customer's experience, not just the case. Operators
read the aggregates (`cx-agent signals`, `GET /signals/report`,
the console's signals panel): average effort by entry desk, the
share of conversations whose frustration rose, offer outcomes,
and the frustration × escalation counts — descriptive, and
labelled as such.

## Why conversations end

Every closed conversation gets a record of how it ended,
computed in code from its own audit trail: the outcome
(resolved, resolved with an action, escalated, or ended
unresolved) and, for the endings that weren't resolutions, one
primary cause from a governed taxonomy — routing error,
identity friction, policy limit, missing capability, procedure
dead end, the customer leaving, a system error — with the rule
that fired and the signals it read carried on the record. An
escalation the rules demanded (a threat, a second ask for a
human) is recorded as designed, not as a failure. The report
aggregates cause × desk over a window against the previous one
(`cx-agent attribution`), and joins the mining candidates to
the causes their conversations earned. Humans review the calls
in an **annotation queue**: every escalated or unresolved
conversation, plus a deterministic sample of the resolved ones;
two annotators who can't be the same person, agreement and
Cohen's kappa over the cause labels, and a third person's
verdict where they disagree. Nothing in the loop acts on its
own — it is evidence for the humans designing desks and
intents.

## The day, read back whole

Once a day the pieces above are composed into one stored
**daily quality digest** (`cx-agent daily-quality --date
YYYY-MM-DD`, `POST /quality/run`, the console's digest panel):
the day's closed conversations — volume by entry desk and
outcome, the conversations counted by channel, the cause mix
with designed escalations counted apart, the signal aggregates (average effort, how often
frustration rose, what the offers of a person did), the
complaint book's SLA state at day end, the promises ledger's
state at day end with broken promises listed in full, the
day's top unhandled-intent candidates, and escalation counts
by reason.
Every figure is derived from the day's records in code; the
digest is a snapshot, and re-running the pass for a date
replaces it — the new digest's version number records that it
did, so a replacement is never silent. One function renders
the stored record to markdown, and the API, the CLI and the
console all show that same rendering. A trend view lines the
stored digests up — volume, resolution share, top cause,
average effort, drift — so a bad day is visible against the
days around it.

The pass also re-runs a deterministic sample of the eval
trajectories against the current build — every scenario whose
id hashes into one-in-four, topped up to at least one per
rubric — and compares them with the committed baseline in
`evals/results.json`. An assertion that was green in the
baseline and fails now is reported as **drift**, with the
scenario, the assertion, and the expected and actual values;
no drift, and the digest says so in one line. The baseline
records per-scenario counts rather than per-assertion history,
so drift is defined at exactly that granularity — a sampled
scenario failing now that the baseline never recorded as green
is listed as unbaselined, not dressed up as drift.

One honest scope note: conversations carry no closing date of
their own, so a digest's day is the conversations *attributed*
that day. The pass runs attribution first, which means a
digest run for today is complete; a conversation that merely
stopped joins the day an operator attributes it. And the
digest informs humans — nothing anywhere acts on it.

## Coming back

A returning customer should never have to start over, and the
agent should never have to guess what happened last time. The
moment identity verifies, the supervisor computes a
**continuity briefing** from the account's records — in code,
never from a model's memory of the last chat: the open items
with their deadlines (a complaint still open and the date its
response is due, a dispute in flight, an arrangement's next
instalment, a technician callback on the book), the standing
flags (the accessibility note and contact preference, honoured
from the first turn; the vulnerability flag, carried for the
humans), the last conversation's outcome in one line, and
anything left mid-flight. The briefing's lines open the reply —
"I can see your complaint from Tuesday is still open, and
we're due to respond by Friday" — and every element of it
names the record it came from; a test walks the briefing and
resolves each element to its record, so an opening can never
claim an item that isn't on the book. An account with no
history gets an empty briefing and an ordinary opening:
continuity is computed, never fabricated.

The commitments behind those items stand as a **promises
ledger** (`cx-agent promises`, `GET /promises`, the console's
ledger panel): every complaint response deadline, booked
callback and arrangement instalment, with a due state derived
from its record and the shared clock — kept, pending, or
broken (a callback made is kept; a deadline passed with the
item still open is broken). The ledger is a read over the
records, not a second store, so it cannot drift from the books
it summarises. Broken promises surface three ways: in the next
conversation's briefing, in the daily digest — listed in full
at day end — and on the ledger views. Nothing anywhere
auto-apologises or auto-compensates for one: a person sees it
and decides.

Continuity crosses conversations, not just contacts. A
conversation that ends mid-procedure leaves its run active on
the store; the next conversation for the account adopts that
run at verification — rebound to the new conversation, its
history intact — and the walk continues from the step it
stopped at instead of starting over. A proposal left
unconfirmed is named from its record ("you had this waiting
for a decision") but never carried over: acting on it still
takes a fresh proposal from live figures and a fresh yes.

## Every word is versioned

The texts the agent works under are data with versions, in a
**prompt registry** (`prompts.py`): the compose instructions
the supervisor drafts under and the template the facts are
poured into, the router's classification prompt, the second
explanation forms, and the policy documents themselves — 28
governed texts, seeded with the book and stored per version.
The active version is the highest unless an operator pins
another; a change is a new version, old versions stay readable
forever, and a rollback is a recorded pin, not an edit. The
model boundary resolves its prompts from the registry at call
time and refuses — loudly, with the entry named — if the
registry cannot serve one: there is no fallback prompt in the
code, and no second copy of any of these texts anywhere.

Every activation passes a **lint** (`cx-agent prompt-lint`,
`promptlint.py`) that runs in the test gate and inside the
operator acts themselves. It checks the words against the
code: the figures a text states — the dispute cap, the
goodwill tiers, the fair-use ceiling, the arrangement terms,
the code attempts, the complaint window — must equal the
Policy values the desks enforce, under a mapping declared in
the lint itself (worked-example figures state no policy and
are deliberately unmapped); promise phrases the agent may
never make fail on sight; no entry outside the compose surface
may instruct contacting the customer directly, because only
the supervisor writes to the customer; and a compose entry
must carry the verbatim-figures and Sources instructions it
exists to enforce. Activating a policy text rewrites the
corpus document and its embedding to match, so retrieval
serves the same words the registry calls active.

Each conversation records the registry versions that served
it — the compose pair, each policy document at the version
cited, each second form at the version served — on its record
(`GET /conversations/{id}/prompts`), so any historical reply
can be traced to the exact texts behind it. What the registry
deliberately does not hold: the deterministic text the code
composes itself (the confirmation ask, the offer lines, the
desks' computed findings, procedure instructions, the bill's
summary-first form) — code-composed templates, reviewed as
code, never instructions to a model.

## The rules are code

- **Identity** — unverified → verifying → verified / failed.
  Account number + postcode issue a one-time code (a deterministic
  test double in this build — see the architecture doc); three
  wrong codes lock verification and escalate. Consequential actions
  refuse to run unverified, full stop.
- **Escalation** — threats, fraud reports and legal matters
  escalate immediately; a flagged-vulnerable customer goes to the
  care queue; asking for a human earns one genuine solve attempt,
  and a second ask transfers at once; three consecutive
  no-progress turns transfer. Every transfer carries a handoff
  packet built from the record — summary, attempts, actions taken,
  open items — and, since round 11, the signal summary: what the
  conversation cost the customer and what was already tried.
- **Handoff quality** — since round 17 the packet also carries
  the issue in the customer's own words, the verification
  attempts spent, what was tried at step level (procedure
  outcomes, dispute lines as decided), the figures the case
  turns on, and the account's open promises and complaints from
  the promises ledger — and it is **graded in code** against the
  elements its class of escalation requires (an immediate
  fraud transfer is not graded on history it cannot have; a
  standard handoff is graded on all of it). A packet missing an
  element is marked incomplete with the gaps named — on the
  packet, in the audit trail, and in the daily digest — but the
  transfer is never blocked on paperwork. Picking a handoff up
  is a recorded act (receiver and day on the escalation), and
  handoffs nobody has picked up past a governed threshold are
  named by the digest and the queue views; nothing reassigns
  them on its own. What the customer is told was passed on is
  generated from the packet itself, element by element — an
  incomplete packet produces a shorter line, never a false one.
- **Cost & latency** — every model call is a record before it
  is a number: the compose call (a *decision* call) and the
  router's tie-break hint (an *advisory* call) each land on the
  conversation's audit trail with their tokens and wall-clock
  latency. Tokens are **measured** where the provider reported
  usage and **estimated** by a deterministic estimate everywhere
  else — the offline backend always estimates, and the two are
  never summed without their labels. Budgets are governed
  Policy values, read from those records at the start of a
  turn: when a conversation has spent its decision calls or its
  tokens, the current turn has already completed — the *next*
  one closes it gracefully, one plain line that a person will
  pick it up, a graded handoff packet, reason `budget_exhausted`.
  A model call carries a governed timeout and exactly one
  retry; a second timeout closes the turn the same way, reason
  `model_timeout` — no answer is ever composed out of a call
  that never answered. Attribution reads both closings as the
  design working. The report folds from the records — per desk,
  per day, overall — on the API (`GET /costs/report`), the CLI
  (`cx-agent costs`), the console, and as a section of the
  daily digest; the totals reconcile by construction, because
  nothing keeps a second set of books.
- **Signals** — frustration, effort, progress and confusion are
  computed per turn from the conversation's own records (a
  governed lexicon, counted effort components, a repeat-overlap
  measure), never judged by a model. They drive three
  interventions — an early offer of a person on rising
  frustration with no progress, the same offer at the
  identity-effort threshold, and a second explanation form on
  confusion — each recorded as an audit event. None of them
  touches the escalation rules above: an accepted offer
  escalates under its own reason, and the safety rules fire
  exactly as before, offer or no offer.
- **Grounding** — a policy claim in a reply must cite the policy
  document retrieval returned for that turn; the composer enforces
  the citation in code after drafting. Account facts come only
  from tools.
- **Rerouting** — a conversation can change desks mid-stream (a
  billing chat that turns into a fault). The move is written to
  the audit with its trigger turn; identity, context and the
  executed-action history carry over untouched. A proposal the
  old desk left awaiting an answer is retired in the move — it
  can only ever run after being re-made and re-confirmed at the
  new desk, so a stray "yes" to a later question can never fire
  a stale action.
- **Fresh figures only** — a plan proposal is a versioned record
  stamped with a fingerprint of the account state it was priced
  against. If the account moves before the customer answers —
  a payment lands, another action executes — execution refuses
  the proposal as stale, marks it on the books, and the desk
  re-quotes the same request from live state for a fresh
  confirmation. Reversals never edit history: a reversal is a
  compensating pair of records (the plan restored, the money
  unwound) pointing back at the original change.
- **Asking, not guessing** — when two desks tie on a real signal,
  the agent asks one targeted clarifying question offering both
  readings in plain words; if the customer doesn't answer it, the
  router commits to its best reading and never asks twice. A
  turn with no clear signal goes to the care desk, never to a
  transactional desk on a guess. Every routing decision is
  recorded with its matched signals and the runner-up intent.
- **Vulnerability** — the flag on an account record is set only
  from a circumstance the customer states plainly, in one of
  three governed classes (bereavement, serious illness,
  financial hardship), and the setting is audited with its
  class. Tone, age, slow typing and sad phrasing set nothing —
  the tests prove both directions. Setting it routes the
  conversation to the priority care queue with the flag visible
  in the handoff packet, and it persists for later contacts.
- **Screening** — customer text is screened for prompt injection
  before any model sees it; attempts are audit-logged and answered
  as ordinary service.
- **Channels** — a conversation belongs to a channel, recorded
  at creation: `web_chat`, `email` or `voice_transcript`. The
  channel governs **presentation only** — identity, screening,
  policy, money and escalation read the customer's words exactly
  the same way on every channel, and the eval gate plays one
  dispute down all three and grades identical state. What the
  channel changes is the rendering, applied in code after the
  reply is composed: email goes out as plain text (markdown
  stripped deterministically) under a subject line derived from
  the conversation's governed state — its desk and its status,
  never the customer's words — and a length cap that truncates
  whole sentences only, never a figure, a guaranteed line or
  the Sources line (the cap yields to those, in that order);
  the voice channel gets a spoken-form view — figures spelled
  for speech, URLs and the Sources line replaced by a spoken
  pointer to the customer's account messages, sentences split
  at clause boundaries under a word cap — while the stored
  reply keeps its digits verbatim. Inbound, the envelope comes
  off by rule: an email's body is the message (quoted history
  dropped) and its subject is screened on arrival, audit-logged
  and filed on the record — context, never an instruction; a
  transcript keeps the customer-side lines with the diarisation
  stripped. The daily digest counts the day's conversations by
  channel.

## The eval gate

CX agents fail in behavior, not in unit tests, so the gate is
behavioral: **163 scripted multi-turn conversations** played
against the real service, graded with pass/fail rubric
assertions — escalation, education, grounding, transaction,
identity, care, attribution, signals, retention, continuity,
handoffs, costs, channel. One hundred and seventeen are hand-written trajectories
(the refund seeker, the human-demander, the fragment writer, the
AI skeptic, the confused elder, non-native phrasing). Thirty-five
more are the **persona matrix**: seven difficult-customer
archetypes (confused and repetitive, terse, rambling, non-native,
mind-changing, wall-of-text, careful elderly) generated across
five core flows, each pair grading the identical outcome — same
money to the cent, same executed actions — because how a
customer talks must not change what the agent does. The last
eleven are an **adversarial set** — prompt injection, impersonation,
social engineering, and cross-account probes — graded
zero-tolerance as a set. The gate requires **100% on
zero-tolerance assertions** (escalation decisions, whether an
action executed, exact money and identity state), **≥95%
overall**, and a clean adversarial set; the report breaks down
by rubric, by persona, and by attack class. Current state of
the tree: 1530/1530 assertions, 1193/1193 zero-tolerance. The
matrix found a real defect on its first run — a hesitant
"um, yes I think so" silently destroyed pending decisions at
two decision points; both now re-ask instead, and the matrix
is the regression proof.

```bash
make evals   # writes evals/results.json
```

The pytest suite (396 tests) covers the deterministic core —
identity machine, escalation rules, money math, action gates,
routing, rerouting and clarification, billing depth, plan-change
transactions, the retention flow (the case record, the one
policy-bound save offer, the no-dark-pattern rules, the
cycle-end cancellation), care records and the mining pass, failure
attribution and the annotation queue, the confirmation
re-ask regressions, the prompt registry and its lint,
continuity (the briefing's traceability to its records, the
promises ledger's due states, the adopted procedure run),
handoff quality (the completeness grade per escalation class,
acceptance stamped and singular, the overdue read by the
shared clock, the digest naming both, the transfer line held
to the packet), cost & latency discipline (the call records
on the audit trail, the fold's reconciliation, both caps
closing the next turn, the retry-once timeout closing the
turn to a person), channel discipline (the envelopes parsed
by rule, the subject screened and filed, the email rendering's
plain text and its protected cut, the spoken form's spelled
figures and pointer line, the same dispute landing identically
on all three channels), store
conformance across all three store profiles — the
PostgreSQL leg runs when `CX_TEST_DATABASE_URL` names a
scratch database, and on 2026-10-18 it ran green end to end
against PostgreSQL 16.15 with pgvector 0.8.7: 408 tests, no
skips. Its first live run found real defects (a migration
check that never reached the database, payload reads that
assumed SQLite's wire shape); the fixes are in `store.py`
and the round map in `docs/architecture.md` tells the story.

## Layout

```
src/cx_agent/
  supervisor.py    the LangGraph graph: screen, identity, route,
                   confirm, compose, escalate
  specialists.py   the six desks; structured findings + proposals
  escalation.py    the escalation rules + handoff packet builder
  identity.py      the verification state machine + code double
  actions.py       structured action execution behind the gates
  routing.py       the deterministic intent router, its recorded
                   evidence (signals, runner-up), the clarification
                   reads + the reply classifiers
  screening.py     prompt-injection and signal screening
  retrieval.py     policy-corpus retrieval (pgvector / JSON)
  documents.py     the deterministic hashing embedder (384-dim)
  llm.py           the provider boundary + offline composer
  money.py         Decimal money math: proration, credits
  plan_changes.py  plan-change transactions: cycle-calendar quotes,
                   effective dates, proposal fingerprints, reversals
  billing_book.py  the billing ledger: decomposition, comparison,
                   bill shock, per-line dispute cases, credit history
  care_book.py     the care ledger: complaint records + the SLA
                   watch, feedback records, accessibility on the
                   account
  retention_book.py the retention ledger: cancellation-intent
                   cases (governed reasons, words verbatim), the
                   one policy-bound save offer assembled from real
                   mechanisms, the no-dark-pattern rules, the
                   cycle-end cancellation, the retention report
  mining.py        the unhandled-intent mining pass: clusters
                   the conversations that ended unhandled into
                   candidate intents (a report; creates nothing)
  attribution.py   failure attribution + the annotation queue:
                   outcome and cause computed from the audit
                   trail in code, the windowed cause × desk
                   report with the mining join, and the human
                   review loop (annotations, agreement, verdicts)
  incidents.py     known-issue awareness: the operator incident
                   feed, the deterministic matching (live status,
                   declared signals, service, postcode scope),
                   the record-verbatim acknowledgement, the
                   recorded operator acts (add / update / resolve)
  signals.py       conversation signals: the frustration lexicon
                   and its rising rule, the counted effort score,
                   confusion detection, the per-turn snapshot, the
                   packet summary and the operator report
  quality.py       the daily quality pass: the digest composed
                   from a day's records, the regression
                   spot-check against the committed baseline,
                   the trend view, the markdown rendering
  continuity.py    the continuity briefing computed from the
                   account's records at verification (open items,
                   flags, last outcome, the resumable run), and
                   the promises ledger: due states derived from
                   the commitment records + the shared clock
  handoffs.py      handoff quality: the completeness grade per
                   escalation class, the customer line generated
                   from the packet, acceptance as a recorded act,
                   the unaccepted-and-overdue read
  costs.py         cost & latency discipline: the fold over the
                   model-call records — per-conversation usage
                   (what the budget reads), the operator report
                   per desk and per day, the digest's section
  channels.py      channel discipline: the inbound envelope
                   parsers (email subject/body, transcript
                   diarisation), the per-channel renderings
                   (email plain text + governed subject + cap,
                   the spoken-form view), presentation only
  prompts.py       the prompt registry: every governed text as
                   versioned data (compose + intent prompts, the
                   explanation forms, the policy corpus), the
                   fail-closed resolver, the recorded operator
                   acts (add / rollback / unpin)
  promptlint.py    the prompt lint: figure consistency against
                   the Policy, banned promises, side-channel
                   sends, composition discipline — the test gate
                   and the activation gate
  explanations.py  second explanation forms, resolved from the
                   registry at serve time, plus the bill's
                   summary-first form (a computed template)
  procedures.py    the troubleshooting engine: procedures as
                   versioned data, runs, tools, branches, safety
                   steps, transcript-carrying callbacks
  clock.py         the one shared clock (CX_TODAY pins it)
  dataset.py       the Northbeam book, generated in code
  store.py         store port: PostgreSQL, SQLite, in-memory
  objects.py       S3 transcript exports (SigV4, no SDK)
  api.py           FastAPI surface + console
  static/console.html  the single-file operator console
migrations/        numbered SQL migrations (PostgreSQL + SQLite)
evals/             scenarios.jsonl (117 trajectories) + personas.py
                   (the 35-scenario matrix) + adversarial.jsonl
                   (11 attacks) + run_evals.py
tests/             the pytest suite
docs/architecture.md  the full design
```

The bundled dataset under `dataset.py` is generated in code for
this blueprint — customers, bills, plans and policies are invented
for Northbeam, with the gold outcomes the evals assert on built
into the records themselves.

## Honest limits

- The one-time code is a **test double**: derived deterministically
  per account so tests and evals can verify the flow. A production
  deployment sends a real code over SMS/email and stores only its
  hash — the state machine around it does not change.
- The desks are deep on the paths the book covers — billing,
  troubleshooting, plan changes, roaming, payments, care,
  retention — but the book is finite: a need outside the
  seeded procedures, policies and offers is an unhandled
  intent by design, logged and mined rather than improvised.
  Incident matching is deterministic
  and bounded by the incidents' declared signals: a report
  worded outside them follows the normal troubleshooting
  path, and the agent has no incident intelligence beyond the
  feed — no inferred causes, no estimated fix times, only
  what the operator's record states.
- The offline fallback drafts competently from findings because
  the findings are written as sentences; it does not understand
  free-form customer language the way a model does. Routing and
  step classification are closed-vocabulary code and behave the
  same in both modes.
- Transcript export to S3 is the only object-storage write;
  there is no attachment or document intake — everything the
  agent knows arrives as conversation text or a record.

## License

Apache 2.0. See `LICENSE`. A Trida AI blueprint: patterns are
yours to take; the code is the deliverable.
