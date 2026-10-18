"""Operator CLI: seed the book, show readiness, list the queue.

    cx-agent seed            write the Northbeam book into the store
    cx-agent readiness       print the profile the app will run with
    cx-agent escalations     list open escalations (the human queue),
                             each with its packet's completeness
                             grade and who has picked it up; the
                             unaccepted-and-overdue are named last
    cx-agent accept-handoff  record a person picking a handoff up:
        <escalation-id> <receiver>
    cx-agent complaints      list complaint records with SLA state
    cx-agent complaint-watch run the watch: raise breached complaints
    cx-agent intent-mining   the unhandled-intent mining report
    cx-agent attribution-run attribute every closed conversation
    cx-agent attribution     the attribution report (cause x desk,
                             this window against the previous one)
    cx-agent annotation-queue the human review queue + agreement
    cx-agent signals         the conversation-signal report (effort,
                             frustration, interventions)
    cx-agent daily-quality   run the daily quality pass and print
        [--date YYYY-MM-DD]  its digest (default date: today)
    cx-agent quality-trend   headline metrics across stored digests
    cx-agent retention       list retention cases (reason, the one
                             save offer, the outcome)
    cx-agent retention-report the retention report: cases by
                             reason, offers made/accepted, the
                             save rate over decided cases
    cx-agent prompts         the prompt registry's active set
    cx-agent prompt-lint     lint the active registry (figures,
                             promises, side channels, composition)
    cx-agent prompt-add      add a version of a registry entry:
        <prompt-id> <file>   its text read from the file
    cx-agent prompt-rollback pin an older version active again:
        <prompt-id> <version>
    cx-agent prompt-unpin    lift an entry's pin:
        <prompt-id>
    cx-agent incidents       the incident feed (known issues, their
                             status, scope and declared signals)
    cx-agent incident-add    add an incident from a JSON file:
        <file>
    cx-agent incident-update update an incident from a JSON file:
        <incident-id> <file>
    cx-agent incident-resolve resolve an incident (it stops
        <incident-id>        matching from this act on)
    cx-agent promises        the promises ledger: every commitment
        [NB-####]            on the book with its derived due state
                             (broken first) — one account, or all
"""

from __future__ import annotations

import json
import sys

from . import care_book
from .logging_setup import configure_logging
from .service import ConversationService


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "readiness"
    service = ConversationService()
    if cmd == "seed":
        print(json.dumps(service.seed(), indent=2))
        return 0
    if cmd == "readiness":
        print(json.dumps(service.readiness(), indent=2))
        return 0
    if cmd == "escalations":
        for esc in service.list_escalations():
            packet = esc.packet
            grade = ("complete" if packet.complete
                     else f"INCOMPLETE (missing: {', '.join(packet.missing)})")
            pickup = (f"accepted by {esc.accepted_by} on {esc.accepted_date}"
                      if esc.accepted_by else "not yet accepted")
            print(f"{esc.escalation_id}  queue={esc.queue}  reason={esc.reason}  "
                  f"conv={esc.conversation_id}  packet={grade}  {pickup}")
        overdue = service.unaccepted_handoffs()
        for row in overdue:
            print(f"OVERDUE {row['escalation_id']}  queue={row['queue']}  "
                  f"raised {row['created_date']}  {row['age_days']} day(s) "
                  "unaccepted")
        return 0
    if cmd == "accept-handoff":
        if len(argv) < 3:
            print("usage: cx-agent accept-handoff <escalation-id> <receiver>")
            return 2
        try:
            esc = service.accept_handoff(argv[1], argv[2])
        except (KeyError, ValueError) as exc:
            print(f"refused: {exc}")
            return 1
        print(f"{esc.escalation_id} accepted by {esc.accepted_by} "
              f"on {esc.accepted_date}")
        return 0
    if cmd == "complaints":
        for c in service.list_complaints():
            state = care_book.complaint_sla_state(c, service.policy)
            print(f"{c.complaint_id}  {c.category}  status={c.status}  "
                  f"sla={state}  due={c.response_due}  "
                  f"account={c.account_number or '-'}  conv={c.conversation_id}")
        return 0
    if cmd == "complaint-watch":
        print(json.dumps(service.run_complaint_watch(), indent=2))
        return 0
    if cmd == "intent-mining":
        print(json.dumps(service.mining_report(), indent=2))
        return 0
    if cmd == "attribution-run":
        print(json.dumps(service.run_attribution(), indent=2))
        return 0
    if cmd == "attribution":
        days = int(argv[1]) if len(argv) > 1 else 7
        report = service.attribution_report(days)
        print(f"Attribution — last {report['window_days']} days "
              f"({report['current_window'][0]} to {report['current_window'][1]}) "
              f"vs previous ({report['previous_window'][0]} to "
              f"{report['previous_window'][1]})")
        print(f"  conversations attributed: {report['current']['total']} "
              f"(previous {report['previous']['total']})")
        for row in report["cause_desk_rows"]:
            print(f"  {row['cause']:<20} {row['desk']:<10} "
                  f"now {row['current']}  was {row['previous']}  "
                  f"({row['delta']:+d})")
        print(f"  escalated by design: {report['current']['by_design']} "
              f"(previous {report['previous']['by_design']})   "
              f"no cause fired: {report['current']['no_cause']} "
              f"(previous {report['previous']['no_cause']})")
        for join in report["mining_evidence"]:
            print(f"  mining: {join['label']} ×{join['count']} — "
                  f"missing_capability {join['missing_capability']}, "
                  f"routing_error {join['routing_error']}, "
                  f"other {join['other_cause']}, "
                  f"unattributed {join['unattributed']}")
        return 0
    if cmd == "signals":
        report = service.signal_report()
        print(f"Conversation signals — {report['conversations']} "
              f"conversation(s) with signal records")
        print(f"  average effort: {report['avg_effort']}")
        for desk, avg in report["avg_effort_by_entry_desk"].items():
            print(f"    entered at {desk:<10} {avg}")
        print(f"  frustration rose in "
              f"{report['rising_frustration_conversations']} "
              f"({report['rising_frustration_share']})")
        iv = report["interventions"]
        print(f"  offers: {iv['offered']} made, {iv['accepted']} accepted, "
              f"{iv['declined']} declined, {iv['lapsed']} lapsed "
              f"(acceptance rate {iv['acceptance_rate']}); "
              f"second forms served: {iv['second_forms']}")
        cells = report["frustration_escalation"]
        print(f"  frustration x escalation (descriptive): "
              f"rising+escalated {cells['rising_and_escalated']}, "
              f"rising only {cells['rising_not_escalated']}, "
              f"escalated only {cells['not_rising_and_escalated']}, "
              f"neither {cells['not_rising_not_escalated']}")
        return 0
    if cmd == "annotation-queue":
        for row in service.annotation_queue():
            print(f"{row['conversation_id']}  {row['outcome']:<19} "
                  f"cause={row['cause'] or '—':<19} desk={row['desk']:<10} "
                  f"state={row['state']:<10} annotations={row['annotations']}")
        agreement = service.annotation_agreement()
        print(f"agreement: {agreement['agreements']}/{agreement['pairs']} "
              f"pairs, rate={agreement['agreement_rate']}, "
              f"kappa={agreement['kappa']}, "
              f"disputed={len(agreement['disagreements'])}, "
              f"settled={len(agreement['settled'])}")
        return 0
    if cmd == "daily-quality":
        from datetime import date as _date

        from . import quality

        raw = None
        if "--date" in argv:
            raw = argv[argv.index("--date") + 1]
        elif len(argv) > 1:
            raw = argv[1]
        try:
            on = _date.fromisoformat(raw) if raw else None
        except ValueError:
            print(f"not an ISO date: {raw}", file=sys.stderr)
            return 2
        digest = service.run_daily_quality(on)
        print(quality.render_digest_markdown(digest))
        return 0
    if cmd == "retention":
        for c in service.list_retention_cases():
            offer = (f"offer={c.save_offer_mechanism}:"
                     f"{c.save_offer_outcome}" if c.save_offer_summary
                     else "offer=none")
            print(f"{c.case_id}  reason={c.reason}  state={c.state}  "
                  f"outcome={c.outcome or '-'}  {offer}  "
                  f"account={c.account_number or '-'}  "
                  f"conv={c.conversation_id}")
        return 0
    if cmd == "retention-report":
        report = service.retention_report()
        print(f"Retention — {report['cases']} case(s), "
              f"{report['decided_cases']} decided")
        for reason, count in report["by_reason"].items():
            if count:
                print(f"  reason {reason:<18} {count}")
        print(f"  offers: {report['offers_made']} made, "
              f"{report['offers_accepted']} accepted, "
              f"{report['offers_declined']} declined; "
              f"no offers requested in {report['no_offers_requested']}")
        rate = report["save_rate"]
        print(f"  save rate over decided cases: "
              f"{'—' if rate is None else f'{rate:.0%}'}  "
              f"cancellations completed: "
              f"{report['cancellations_completed']}  "
              f"escalated to humans: "
              f"{report['escalated_retention_team']}")
        return 0
    if cmd == "prompts":
        rows = service.list_prompts()
        if not rows:
            print("The prompt registry is empty — run cx-agent seed first.")
            return 0
        for row in rows:
            pin = (f"  pinned v{row['pinned_version']}"
                   if row["pinned_version"] else "")
            print(f"{row['prompt_id']:<28} {row['surface']:<12} "
                  f"active v{row['active_version']} "
                  f"({row['versions']} version(s)){pin}")
        return 0
    if cmd == "prompt-lint":
        from .promptlint import PromptLintError, lint_store
        from .prompts import PromptRegistryError

        if not service.prompts.known_ids():
            print("prompt lint: the registry is empty — "
                  "run cx-agent seed first", file=sys.stderr)
            return 2
        try:
            findings = lint_store(service.store, service.policy)
        except PromptRegistryError as exc:
            print(f"prompt lint: {exc}", file=sys.stderr)
            return 2
        if findings:
            for finding in findings:
                print(str(finding))
            print(f"prompt lint: {len(findings)} finding(s) — "
                  f"the registry does not pass")
            return 1
        print(f"prompt lint: OK — {len(service.prompts.known_ids())} "
              f"active entries, no findings")
        return 0
    if cmd in ("prompt-add", "prompt-rollback", "prompt-unpin"):
        from .promptlint import PromptLintError
        from .prompts import PromptRegistryError

        try:
            if cmd == "prompt-add":
                if len(argv) < 3:
                    print("usage: cx-agent prompt-add <prompt-id> <file>",
                          file=sys.stderr)
                    return 2
                with open(argv[2], encoding="utf-8") as fh:
                    text = fh.read()
                entry = service.add_prompt_version(
                    argv[1], text, actor="cli")
                print(f"{entry.prompt_id}: v{entry.version} added and "
                      f"active (unless pinned)")
            elif cmd == "prompt-rollback":
                if len(argv) < 3:
                    print("usage: cx-agent prompt-rollback "
                          "<prompt-id> <version>", file=sys.stderr)
                    return 2
                pin = service.rollback_prompt(
                    argv[1], int(argv[2]), actor="cli")
                print(f"{pin.prompt_id}: rolled back — v{pin.version} "
                      f"is active (pinned)")
            else:
                if len(argv) < 2:
                    print("usage: cx-agent prompt-unpin <prompt-id>",
                          file=sys.stderr)
                    return 2
                service.unpin_prompt(argv[1], actor="cli")
                detail = service.get_prompt(argv[1])
                print(f"{argv[1]}: unpinned — v{detail['active_version']} "
                      f"is active")
        except PromptLintError as exc:
            for finding in exc.findings:
                print(str(finding), file=sys.stderr)
            print("the act was refused by the lint gate; nothing "
                  "was written", file=sys.stderr)
            return 1
        except (PromptRegistryError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    if cmd == "incidents":
        rows = service.list_incidents()
        if not rows:
            print("The incident feed is empty — run cx-agent seed first.")
            return 0
        for i in rows:
            areas = (", ".join(i.affected_postcodes)
                     if i.affected_postcodes else "all areas")
            print(f"{i.incident_id}  {i.status:<10} {i.service:<16} "
                  f"{i.title}  [{areas}]  since {i.started_at}")
            print(f"    signals: {', '.join(i.symptom_signals)}")
            print(f"    summary: {i.summary}")
        return 0
    if cmd in ("incident-add", "incident-update", "incident-resolve"):
        from .incidents import IncidentError

        try:
            if cmd == "incident-add":
                if len(argv) < 2:
                    print("usage: cx-agent incident-add <file>",
                          file=sys.stderr)
                    return 2
                with open(argv[1], encoding="utf-8") as fh:
                    fields = json.load(fh)
                actor = fields.pop("actor", "cli")
                note = fields.pop("note", "")
                incident = service.add_incident(
                    actor=actor, note=note, **fields)
                print(f"{incident.incident_id}: added to the feed "
                      f"({incident.status}, {incident.service})")
            elif cmd == "incident-update":
                if len(argv) < 3:
                    print("usage: cx-agent incident-update "
                          "<incident-id> <file>", file=sys.stderr)
                    return 2
                with open(argv[2], encoding="utf-8") as fh:
                    fields = json.load(fh)
                actor = fields.pop("actor", "cli")
                note = fields.pop("note", "")
                incident = service.update_incident(
                    argv[1], actor, note, **fields)
                print(f"{incident.incident_id}: updated "
                      f"({incident.status}, {incident.service})")
            else:
                if len(argv) < 2:
                    print("usage: cx-agent incident-resolve "
                          "<incident-id>", file=sys.stderr)
                    return 2
                incident = service.resolve_incident(argv[1], actor="cli")
                print(f"{incident.incident_id}: resolved — it no "
                      f"longer matches any report")
        except IncidentError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    if cmd == "promises":
        account = argv[1] if len(argv) > 1 else None
        entries = service.promise_ledger(account)
        if not entries:
            print("No promises on the book.")
            return 0
        for e in entries:
            print(f"{e.promise_id}  {e.state:<8} {e.kind:<24} "
                  f"due={e.due_date}  account={e.account_number or '-'}  "
                  f"{e.summary}")
        broken = sum(1 for e in entries if e.state == "broken")
        print(f"{len(entries)} promise(s), {broken} broken — "
              f"broken promises are for a person to act on.")
        return 0
    if cmd == "quality-trend":
        rows = service.quality_trend()
        if not rows:
            print("No digests stored yet — run cx-agent daily-quality.")
            return 0
        print(f"{'date':<12}{'version':>8}{'convs':>7}"
              f"{'resolved':>10}{'top cause':>20}{'effort':>8}"
              f"{'drift':>7}")
        for row in rows:
            share = ("—" if row["resolution_share"] is None
                     else f"{round(row['resolution_share'] * 100)}%")
            effort = ("—" if row["avg_effort"] is None
                      else str(row["avg_effort"]))
            print(f"{row['date']:<12}{row['version']:>8}"
                  f"{row['conversations']:>7}{share:>10}"
                  f"{(row['top_cause'] or '—'):>20}{effort:>8}"
                  f"{row['drift_count']:>7}")
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
