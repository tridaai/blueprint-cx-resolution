# Contributing

This is a Trida AI reference blueprint. Contributions follow the
house rules that keep every blueprint in the series comparable:

## Setup

```bash
make install        # uv sync --extra dev
# or: make install-pip  (frozen set first, package second)
cp .env.example .env  # optional; the offline fallback needs no key
```

## The gates

Every change must pass both, in this order, on the final tree:

```bash
make test    # pytest — the deterministic core
make evals   # the rubric trajectory gate (100% zero-tolerance, ≥95% overall)
```

A change that makes a rubric assertion fail is a behavior change:
either the behavior is wrong (fix the code) or the rubric is
wrong (change the scenario *and* say why in the commit). Never
tune an assertion to a number the agent happens to produce.

## Rules

- **Deterministic in code, language in the model.** Routing,
  identity, escalation, money and action gates are code paths with
  tests. If a rule can be expressed as a check, it does not belong
  in a prompt.
- **Data is generated, never real.** The Northbeam book is built
  in `dataset.py` with gold outcomes by construction. Never commit
  a real customer record, bill, or conversation.
- **Dependencies are pinned exactly** (`==` in `pyproject.toml`)
  and the full set is frozen in `constraints.txt`. Regenerate it
  with `uv export --format requirements-txt --extra dev
  --no-emit-project -o constraints.txt` whenever pins change, and
  commit `uv.lock` with it.
- **Migrations are numbered and append-only.** A schema change is
  a new file in `migrations/` *and* its SQLite dialect twin in
  `migrations/sqlite/` — never an edit to a shipped migration.
- **Commits** are small, in a human voice, and say what changed and
  why. Author: Trida AI <github@trida.ai>.
- No CI configuration lives in this repo; the gates above are the
  whole contract, run locally.
