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

- **Billing** — decomposes the latest bill line by line from the
  record; disputes are checked against evidence on the bill
  itself (a duplicate charge, a charge after cancellation, a wrong
  rate) and credited in full up to the agent's cap, or escalated
  to the billing queue above it. Goodwill credits follow their own
  smaller cap.
- **Roaming** — explains rates and usage from the account; sells
  the Roam Pass only after the fair-use warning when the customer
  is past the ceiling. The warning precedes any chargeable offer,
  by rule.
- **Offers & plan changes** — eligibility and proration computed
  in code; the customer sees the credit, the charge and the net
  before confirming anything.
- **Technical support** — guided procedures walked as step-graphs:
  agent-side checks run as tools, each customer step is confirmed
  before the next, and exhausting the steps books a technician
  callback instead of improvising.
- **Account & payments** — card replacement and payment
  arrangements behind full identity verification; only the last
  four digits of a card ever appear.
- **Customer care** — the catch-all desk for legitimate needs
  outside the five, answering from the policy corpus with
  citations; a need it cannot serve is logged as an unhandled
  intent, not papered over.

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
  open items.
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
- **Asking, not guessing** — when two desks tie on a real signal,
  the agent asks one targeted clarifying question offering both
  readings in plain words; if the customer doesn't answer it, the
  router commits to its best reading and never asks twice. A
  turn with no clear signal goes to the care desk, never to a
  transactional desk on a guess. Every routing decision is
  recorded with its matched signals and the runner-up intent.
- **Screening** — customer text is screened for prompt injection
  before any model sees it; attempts are audit-logged and answered
  as ordinary service.

## The eval gate

CX agents fail in behavior, not in unit tests, so the gate is
behavioral: **37 scripted multi-turn conversations** played by
persona drivers (the refund seeker, the human-demander, the
fragment writer, the AI skeptic, the confused elder, non-native
phrasing) against the real service, graded with pass/fail rubric
assertions — escalation, education, grounding, transaction,
identity, care. The gate requires **100% on zero-tolerance
assertions** (escalation decisions, whether an action executed,
exact money and identity state) and **≥95% overall**. Current
state of the tree: 196/196 assertions, 109/109 zero-tolerance.

```bash
make evals   # writes evals/results.json
```

The pytest suite (60 tests) covers the deterministic core —
identity machine, escalation rules, money math, action gates,
routing, rerouting and clarification, store conformance across
the SQLite and in-memory profiles (the PostgreSQL leg runs when
`CX_TEST_DATABASE_URL` names a scratch database).

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
  dataset.py       the Northbeam book, generated in code
  store.py         store port: PostgreSQL, SQLite, in-memory
  objects.py       S3 transcript exports (SigV4, no SDK)
  api.py           FastAPI surface + console
  static/console.html  the single-file operator console
migrations/        numbered SQL migrations (PostgreSQL + SQLite)
evals/             scenarios.jsonl (31 trajectories) + run_evals.py
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
- Specialists are still on their happy paths in places. Deeper
  dispute tiers, retention flows and the daily quality pass are
  designed (see `docs/architecture.md`) and land in later
  rounds.
- The offline fallback drafts competently from findings because
  the findings are written as sentences; it does not understand
  free-form customer language the way a model does. Routing and
  step classification are closed-vocabulary code and behave the
  same in both modes.
- Transcript export to S3 is the only object-storage write in this
  round; attachments and document intake are later work.

## License

Apache 2.0. See `LICENSE`. A Trida AI blueprint: patterns are
yours to take; the code is the deliverable.
