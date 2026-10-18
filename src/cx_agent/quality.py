"""The daily quality pass (round 12).

Once a day an operator runs the pass over one day's closed
conversations, and it composes a single stored digest from the
records those conversations left behind: volume by entry desk
and outcome, the attribution cause mix (round 10), the signal
aggregates (round 11), the complaint book's SLA state at day
end (round 8), the day's top unhandled-intent candidates (the
round 8 mining pass, scoped to the day), escalation counts by
reason, handoff quality (round 17: the day's incomplete
packets with their gaps named, and the book's unaccepted
handoffs past the acceptance threshold),
the day's model-call costs folded from their records (round 18:
calls by kind, tokens split by basis, latency, and the budget
and timeout closings) — and a regression
spot-check that re-runs a
deterministic sample of the eval trajectories against the
current build and compares them with the committed baseline.

THE DAY. Conversations carry no closing date of their own; the
one day-scope the records support is the attribution record's
`attributed_date`. A day's digest therefore covers exactly the
conversations attributed that day. The pass runs the attribution
pass first (the round 10 operator act — records stand once
written), so everything closed and unattributed joins the
digest for the day the pass runs. A conversation that merely
stopped stays out until an operator attributes it singly; a
digest for a past date reports what was attributed that day,
and nothing pretends otherwise.

THE SNAPSHOT. The digest is stored, one record per date.
Re-running the pass for a date REPLACES the digest and raises
its version by one — the replacement is on the record, never
silent, and never an accumulation. Every figure in it was
derived from records in code at pass time; the markdown
rendering (`render_digest_markdown`) renders the stored record
and recomputes nothing.

THE SPOT-CHECK. The sample rule is named and deterministic:
every scenario whose id hashes (SHA-256, the repo's sampling
convention) into one-in-`daily_spot_check_modulus`, topped up
to at least one scenario per rubric. Each sampled scenario is
re-run through the eval runner's own machinery against the
current build, and its assertion failures are compared with the
committed baseline, `evals/results.json`. The baseline's
granularity is the scenario — it records how many of a
scenario's assertions passed, not which ones — so the honest
comparison is: an assertion failing NOW in a scenario the
baseline records as fully green is DRIFT, reported with the
scenario, the assertion, and the expected vs actual values the
runner produced. A scenario failing now that the baseline does
not record as fully green (or does not record at all) is listed
separately as unbaselined — visible, but never dressed up as
drift. No drift anywhere: the digest says so in one line.

The digest informs humans. Nothing anywhere acts on it: no
alert fires, no threshold trips, no workflow starts. Its job is
to be read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

from . import attribution, care_book, continuity, costs, handoffs, mining
from .clock import today
from .models import DailyDigest

BASELINE_PATH = (Path(__file__).resolve().parents[2]
                 / "evals" / "results.json")
EVAL_RUNNER_PATH = (Path(__file__).resolve().parents[2]
                    / "evals" / "run_evals.py")

_TURN_FAILURE = re.compile(
    r"^(?P<sid>.+?) turn (?P<turn>\d+): expected "
    r"(?P<key>\w+)=(?P<want>.*), got (?P<got>.*)$")
_FINAL_FAILURE = re.compile(
    r"^(?P<sid>.+?): final expected "
    r"(?P<key>\w+)=(?P<want>.*), got (?P<got>.*)$")


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------

def _signal_section(store, conversation_ids: list[str]) -> dict:
    """The round 11 measures over the day's conversations only,
    folded with the signal report's own counting rules: average
    effort from the latest record per conversation, the
    rising-frustration share (any record in the series rising),
    and the intervention outcomes — second forms counted by
    kind, offers by outcome, the acceptance rate over the
    offers that were decided."""
    efforts: list[int] = []
    rising = 0
    offered = accepted = declined = lapsed = second_forms = 0
    for cid in conversation_ids:
        records = store.list_signal_records(cid)
        if not records:
            continue
        efforts.append(records[-1].effort_score)
        if any(r.frustration_rising for r in records):
            rising += 1
        for record in records:
            for entry in record.interventions:
                kind = entry.get("kind", "")
                outcome = entry.get("outcome", "")
                if kind.startswith("second_form:"):
                    second_forms += 1
                elif outcome == "offered":
                    offered += 1
                elif outcome == "accepted":
                    accepted += 1
                elif outcome == "declined":
                    declined += 1
                elif outcome == "lapsed":
                    lapsed += 1
    decided = accepted + declined
    return {
        "conversations": len(efforts),
        "avg_effort": (round(sum(efforts) / len(efforts), 2)
                       if efforts else None),
        "rising_frustration_conversations": rising,
        "rising_frustration_share": (
            round(rising / len(efforts), 3) if efforts else None),
        "interventions": {
            "offered": offered,
            "accepted": accepted,
            "declined": declined,
            "lapsed": lapsed,
            "second_forms": second_forms,
            "acceptance_rate": (round(accepted / decided, 3)
                                if decided else None),
        },
    }


def _complaint_section(store, policy, on: date) -> dict:
    """The complaint book's SLA state at day end: every complaint
    recorded on or before the date, counted by the state the
    round 8 SLA clock reads for it on that date."""
    counts: Counter = Counter()
    for complaint in store.list_complaints():
        if complaint.created_date > on.isoformat():
            continue  # not yet recorded at this day's end
        counts[care_book.complaint_sla_state(
            complaint, policy, on=on)] += 1
    return {"open": counts.get("open", 0),
            "due_soon": counts.get("due_soon", 0),
            "breached": counts.get("breached", 0),
            "settled": counts.get("settled", 0),
            "total": sum(counts.values())}


def run_daily_pass(store, policy, on: date | None = None, *,
                   with_spot_check: bool = True,
                   spot_scenario_ids: list[str] | None = None
                   ) -> DailyDigest:
    """Run the pass for one date and store the digest.

    Attribution runs first, so the day's set is complete for
    today; the digest then composes every figure from records.
    A stored digest for the date is replaced, its version
    raised by one."""
    on = on or today()
    day = on.isoformat()
    attribution.run_attribution(store)
    records = [r for r in store.list_attributions()
               if r.attributed_date == day]
    conversation_ids = [r.conversation_id for r in records]
    id_set = set(conversation_ids)

    by_outcome = {outcome: 0 for outcome in attribution.OUTCOMES}
    cause_mix: Counter = Counter()
    by_design = no_cause = 0
    by_entry_desk: Counter = Counter()
    for record in records:
        by_outcome[record.outcome] += 1
        if record.cause:
            cause_mix[record.cause] += 1
        elif record.rule.startswith("escalated_by_design:"):
            by_design += 1
        else:
            no_cause += 1
        series = store.list_signal_records(record.conversation_id)
        by_entry_desk[series[0].desk if series else record.desk] += 1
    total = len(records)
    resolved = (by_outcome["resolved"]
                + by_outcome["resolved_with_action"])

    handoff_section = _handoff_section(store, policy, id_set, on)

    report = mining.mine_unhandled_intents(
        store, policy, conversation_ids=id_set)
    candidates = [
        {"label": row["label"], "count": row["count"],
         "samples": row["samples"]}
        for row in report["candidates"][:policy.daily_mining_top]]

    escalations = Counter(
        e.reason for e in store.list_escalations()
        if e.conversation_id in id_set)

    spot = (spot_check(policy, scenario_ids=spot_scenario_ids)
            if with_spot_check else
            {"skipped": True, "sampled": [], "drift": [],
             "drift_count": 0, "unbaselined_failing": [],
             "summary": "Spot-check skipped for this run."})

    previous = store.get_digest(day)
    digest = DailyDigest(
        date=day,
        version=(previous.version + 1 if previous is not None else 1),
        conversations=total,
        by_outcome=by_outcome,
        by_entry_desk=dict(sorted(by_entry_desk.items())),
        resolution_share=(round(resolved / total, 3) if total else None),
        cause_mix=dict(sorted(cause_mix.items())),
        escalated_by_design=by_design,
        no_cause=no_cause,
        signals=_signal_section(store, conversation_ids),
        complaints=_complaint_section(store, policy, on),
        promises=continuity.promise_summary(store, on),
        mining_candidates=candidates,
        escalations_by_reason=dict(sorted(escalations.items())),
        handoffs=handoff_section,
        costs=costs.digest_section(store, id_set),
        spot_check=spot)
    store.save_digest(digest)
    return digest


def _handoff_section(store, policy, conversation_ids: set, on: date) -> dict:
    """Handoff quality (round 17): the day's incomplete packets
    with their gaps named — scoped to the day's conversations,
    like the escalation counts — and, as a day-end state of the
    whole book (the complaints and promises sections' pattern),
    the handoffs nobody has picked up past the acceptance
    threshold. The digest names both; a person acts."""
    incomplete = [
        {"escalation_id": e.escalation_id,
         "conversation_id": e.conversation_id,
         "queue": e.queue, "reason": e.reason,
         "missing": list(e.packet.missing)}
        for e in store.list_escalations()
        if e.conversation_id in conversation_ids and not e.packet.complete]
    return {
        "incomplete": incomplete,
        "incomplete_count": len(incomplete),
        "unaccepted_overdue": handoffs.unaccepted_overdue(store, policy, on=on),
    }


# ---------------------------------------------------------------------------
# The regression spot-check
# ---------------------------------------------------------------------------

def _eval_module():
    """Load the eval runner as a module: the spot-check re-runs
    sampled scenarios through the runner's own checks, so the
    comparison is against the gate's own grading, not a copy."""
    spec = importlib.util.spec_from_file_location(
        "cx_eval_runner", EVAL_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scenarios(module) -> list[dict]:
    """Every scenario the eval gate runs, in the runner's order."""
    import personas  # the runner puts evals/ on sys.path at import
    scenarios = [json.loads(line) for line in
                 (EVAL_RUNNER_PATH.parent / "scenarios.jsonl")
                 .read_text().splitlines() if line.strip()]
    adversarial = [json.loads(line) for line in
                   (EVAL_RUNNER_PATH.parent / "adversarial.jsonl")
                   .read_text().splitlines() if line.strip()]
    return scenarios + personas.build_matrix_scenarios() + adversarial


def sample_scenario_ids(scenarios: list[dict], modulus: int) -> list[str]:
    """The named sampling rule: every scenario whose id hashes
    into one-in-`modulus`, topped up with the first scenario (by
    id) of any rubric the hash left out, so every rubric is
    spot-checked every day."""
    def _in(scenario_id: str) -> bool:
        digest = hashlib.sha256(scenario_id.encode()).hexdigest()
        return int(digest, 16) % modulus == 0

    picked = {s["id"] for s in scenarios if _in(s["id"])}
    rubrics = {s.get("rubric") for s in scenarios}
    covered = {s.get("rubric") for s in scenarios if s["id"] in picked}
    for rubric in sorted(r for r in rubrics if r is not None):
        if rubric not in covered:
            first = min(s["id"] for s in scenarios
                        if s.get("rubric") == rubric)
            picked.add(first)
    return sorted(picked)


def _run_scenario(module, scenario: dict) -> list[str]:
    """Play one scenario against the current build, exactly as
    the eval gate does, on its own scratch store."""
    from .service import ConversationService
    from .store import InMemoryStore

    failures: list[str] = []
    service = ConversationService(store=InMemoryStore())
    service.seed()
    for seed_conv in scenario.get("seed_conversations", []):
        seeded = service.start_conversation()
        for seed_turn in seed_conv["turns"]:
            service.post_message(
                seeded.conversation_id,
                module._substitute(seed_turn["say"]))
    conv = service.start_conversation()
    for idx, turn in enumerate(scenario["turns"], start=1):
        if turn.get("poke"):
            module.apply_poke(service, turn["poke"])
        result = service.post_message(
            conv.conversation_id, module._substitute(turn["say"]))
        module.check_turn(result, turn.get("expect") or {}, failures,
                          f"{scenario['id']} turn {idx}",
                          scenario.get("zt", False))
    module.check_final(service, scenario, conv.conversation_id,
                       failures, scenario["id"],
                       scenario.get("zt", False))
    return failures


def _bare_actual(key: str, got: str) -> str:
    """The runner renders a scalar actual as `key=value`; the
    digest already names the assertion, so the repeated key
    comes off. Anything else the runner produced is kept as-is."""
    prefix = f"{key}="
    return got[len(prefix):] if got.startswith(prefix) else got


def _parse_failure(scenario_id: str, failure: str) -> dict:
    """One runner failure string → the digest's finding shape:
    where it failed, the assertion, what was expected, what the
    build produced. The runner's strings carry all four; a shape
    it doesn't recognise is kept verbatim rather than parsed
    into something it isn't."""
    first_line = failure.splitlines()[0] if failure else ""
    match = _TURN_FAILURE.match(first_line)
    if match:
        return {"scenario_id": scenario_id,
                "where": f"turn {match.group('turn')}",
                "assertion": match.group("key"),
                "expected": match.group("want"),
                "actual": _bare_actual(match.group("key"),
                                       match.group("got"))}
    match = _FINAL_FAILURE.match(first_line)
    if match:
        return {"scenario_id": scenario_id, "where": "final",
                "assertion": match.group("key"),
                "expected": match.group("want"),
                "actual": _bare_actual(match.group("key"),
                                       match.group("got"))}
    return {"scenario_id": scenario_id, "where": "",
            "assertion": first_line, "expected": "", "actual": ""}


def spot_check(policy, *, scenario_ids: list[str] | None = None,
               baseline_path: Path | None = None) -> dict:
    """Re-run the sampled scenarios and compare with the
    committed baseline. DRIFT is an assertion failing now in a
    scenario the baseline records as fully green — the
    baseline's granularity is the scenario, and the comparison
    says so rather than pretending to per-assertion history it
    doesn't have."""
    baseline_file = baseline_path or BASELINE_PATH
    module = _eval_module()
    scenarios = load_scenarios(module)
    by_id = {s["id"]: s for s in scenarios}
    sampled = (sorted(scenario_ids) if scenario_ids is not None
               else sample_scenario_ids(
                   scenarios, policy.daily_spot_check_modulus))
    rule = (f"sha256(scenario id) mod {policy.daily_spot_check_modulus}"
            " == 0, topped up to at least one scenario per rubric")

    baseline: dict[str, dict] = {}
    baseline_found = baseline_file.exists()
    if baseline_found:
        data = json.loads(baseline_file.read_text())
        baseline = {row["id"]: row for row in data["scenarios"]}

    drift: list[dict] = []
    unbaselined: list[str] = []
    for scenario_id in sampled:
        scenario = by_id.get(scenario_id)
        if scenario is None:
            unbaselined.append(scenario_id)  # sampled id no longer exists
            continue
        failures = _run_scenario(module, scenario)
        if not failures:
            continue
        row = baseline.get(scenario_id)
        green_at_baseline = (row is not None
                             and row["total"] > 0
                             and row["passed"] == row["total"])
        if green_at_baseline:
            drift.extend(_parse_failure(scenario_id, f)
                         for f in failures)
        else:
            unbaselined.append(scenario_id)

    if not baseline_found:
        summary = ("No committed baseline found at evals/results.json"
                   " — nothing was compared, and no drift is claimed.")
    elif drift:
        summary = (f"Drift: {len(drift)} assertion(s) that were green"
                   f" in the committed baseline fail now, across"
                   f" {len({d['scenario_id'] for d in drift})} sampled"
                   " scenario(s).")
    else:
        summary = (f"No drift — all {len(sampled)} sampled scenarios"
                   " match the committed baseline.")
    return {"rule": rule, "sampled": sampled,
            "baseline_file": "evals/results.json",
            "baseline_found": baseline_found,
            "drift": drift, "drift_count": len(drift),
            "unbaselined_failing": sorted(set(unbaselined)),
            "summary": summary}


# ---------------------------------------------------------------------------
# The trend view and the markdown rendering — stored records only
# ---------------------------------------------------------------------------

def top_cause(digest: DailyDigest) -> str | None:
    """The day's most frequent attributed cause; ties break
    alphabetically so the trend table never depends on dict
    order."""
    if not digest.cause_mix:
        return None
    return sorted(digest.cause_mix.items(),
                  key=lambda kv: (-kv[1], kv[0]))[0][0]


def quality_trend(store) -> list[dict]:
    """Digest over digest, headline metrics only, computed from
    the stored digests — a date with no pass run has no row."""
    rows = []
    for digest in store.list_digests():
        rows.append({
            "date": digest.date,
            "version": digest.version,
            "conversations": digest.conversations,
            "resolution_share": digest.resolution_share,
            "top_cause": top_cause(digest),
            "avg_effort": digest.signals.get("avg_effort"),
            "drift_count": digest.spot_check.get("drift_count", 0),
        })
    return rows


def _counts_block(title: str, counts: dict) -> list[str]:
    lines = [f"## {title}", ""]
    if not counts:
        lines += ["None.", ""]
        return lines
    for key in sorted(counts, key=lambda k: (-counts[k], k)):
        lines.append(f"- {key}: {counts[key]}")
    lines.append("")
    return lines


def render_digest_markdown(digest: DailyDigest) -> str:
    """Render the stored digest as markdown. One function, the
    record only — the console, the API export, and the CLI all
    show exactly this, and it recomputes nothing."""
    share = ("—" if digest.resolution_share is None
             else f"{round(digest.resolution_share * 100)}%")
    lines = [
        f"# Daily quality digest — {digest.date}",
        "",
        f"Version {digest.version} · {digest.conversations} closed"
        f" conversation(s) · resolution share {share} ·"
        f" {sum(digest.escalations_by_reason.values())} escalation(s)",
        "",
    ]
    lines += _counts_block("Volume by outcome", digest.by_outcome)
    lines += _counts_block("Volume by entry desk", digest.by_entry_desk)

    lines += ["## Why conversations ended", ""]
    if digest.cause_mix:
        for cause in sorted(digest.cause_mix,
                            key=lambda c: (-digest.cause_mix[c], c)):
            lines.append(f"- {cause}: {digest.cause_mix[cause]}")
    else:
        lines.append("- No attributed causes.")
    lines.append(f"- Escalated by design (not failures):"
                 f" {digest.escalated_by_design}")
    lines.append(f"- Ended with no cause fired: {digest.no_cause}")
    lines.append("")

    signals = digest.signals
    lines += ["## The customer's experience (signals)", ""]
    if signals.get("conversations"):
        iv = signals["interventions"]
        lines.append(
            f"- Average effort {signals['avg_effort']} across"
            f" {signals['conversations']} conversation(s); frustration"
            f" rose in {signals['rising_frustration_conversations']}"
            f" ({round((signals['rising_frustration_share'] or 0) * 100)}%).")
        lines.append(
            f"- Offers of a person: {iv['offered']} made,"
            f" {iv['accepted']} accepted, {iv['declined']} declined,"
            f" {iv['lapsed']} lapsed; second explanation forms"
            f" served: {iv['second_forms']}.")
    else:
        lines.append("- No signal records for the day.")
    lines.append("")

    complaints = digest.complaints
    lines += ["## Complaints at day end", "",
              f"- Open {complaints.get('open', 0)} · due soon"
              f" {complaints.get('due_soon', 0)} · breached"
              f" {complaints.get('breached', 0)} · settled"
              f" {complaints.get('settled', 0)}"
              f" (of {complaints.get('total', 0)} on the book).",
              ""]

    # Promises (round 16): the ledger's state at day end. Broken
    # promises are listed in full — a person reads them here and
    # acts; the digest itself apologises to no one and compensates
    # no one.
    promises = digest.promises or {}
    lines += ["## Promises at day end", "",
              f"- Broken {promises.get('broken_count', 0)} · pending"
              f" {promises.get('pending_count', 0)} · kept"
              f" {promises.get('kept_count', 0)} on the book."]
    for entry in promises.get("broken") or []:
        lines.append(f"- BROKEN: {entry['summary']} — due"
                     f" {entry['due_date']}, account"
                     f" {entry['account_number']}.")
    lines.append("")

    lines += ["## Unhandled-intent candidates", ""]
    if digest.mining_candidates:
        for candidate in digest.mining_candidates:
            sample = (f' — e.g. "{candidate["samples"][0]}"'
                      if candidate["samples"] else "")
            lines.append(f"- “{candidate['label']}”"
                         f" ×{candidate['count']}{sample}")
    else:
        lines.append("- None at candidate size.")
    lines.append("")

    lines += _counts_block("Escalations by reason",
                           digest.escalations_by_reason)

    # Handoffs (round 17): the day's incomplete packets with
    # their gaps named, and the book's unaccepted queue past
    # the threshold. The digest acts on neither — it makes
    # both impossible to miss.
    hf = digest.handoffs or {}
    lines += ["## Handoffs", ""]
    if hf.get("incomplete"):
        for row in hf["incomplete"]:
            lines.append(
                f"- INCOMPLETE {row['escalation_id']} ({row['reason']}): "
                f"missing {', '.join(row['missing'])}.")
    else:
        lines.append("- Every packet handed off today was complete.")
    overdue = hf.get("unaccepted_overdue") or []
    if overdue:
        for row in overdue:
            lines.append(
                f"- UNACCEPTED {row['escalation_id']} — queue "
                f"{row['queue']}, raised {row['created_date']}, "
                f"{row['age_days']} day(s) without a pickup.")
    else:
        lines.append("- No handoff is past the acceptance threshold.")
    lines.append("")

    # Costs (round 18): the day's model calls, folded from
    # their records by the same fold the standing report runs —
    # the tokens split by basis, the latency, and the closings
    # the discipline produced inside the day's scope.
    cst = digest.costs or {}
    lines += ["## Cost and latency", ""]
    if cst.get("calls"):
        lines.append(
            f"- {cst['calls']} model call(s):"
            f" {cst['decision_calls']} decision,"
            f" {cst['advisory_calls']} advisory;"
            f" {cst['tokens']} tokens (estimated"
            f" {cst['tokens_estimated']}, measured"
            f" {cst['tokens_measured']}).")
        lines.append(
            f"- Latency mean {cst['mean_latency_ms']} ms · p95"
            f" {cst['p95_latency_ms']} ms; timed-out calls"
            f" {cst['timeouts']}, errored {cst['errors']}.")
        lines.append(
            f"- Closings: {cst['budget_exhaustions']} on budget,"
            f" {cst['timeout_escalations']} on timeout.")
    else:
        lines.append("- No model calls recorded for the day.")
    lines.append("")

    spot = digest.spot_check
    lines += ["## Regression spot-check", ""]
    if spot.get("skipped"):
        lines += ["Skipped for this run.", ""]
    else:
        lines.append(f"Sample rule: {spot.get('rule', '')}. Sampled"
                     f" {len(spot.get('sampled', []))} scenario(s)"
                     f" against the committed baseline"
                     f" ({spot.get('baseline_file', '')}).")
        lines.append("")
        lines.append(spot.get("summary", ""))
        for finding in spot.get("drift", []):
            lines.append(
                f"- DRIFT {finding['scenario_id']}"
                f" ({finding['where']}): assertion"
                f" `{finding['assertion']}` — expected"
                f" {finding['expected']}, got {finding['actual']}")
        if spot.get("unbaselined_failing"):
            lines.append(
                "- Also failing, but not green in the baseline (not"
                " counted as drift): "
                + ", ".join(spot["unbaselined_failing"]))
        lines.append("")
    lines.append("This digest informs the team; it triggers no"
                 " automated action.")
    lines.append("")
    return "\n".join(lines)
