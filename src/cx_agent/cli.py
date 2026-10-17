"""Operator CLI: seed the book, show readiness, list the queue.

    cx-agent seed            write the Northbeam book into the store
    cx-agent readiness       print the profile the app will run with
    cx-agent escalations     list open escalations (the human queue)
    cx-agent complaints      list complaint records with SLA state
    cx-agent complaint-watch run the watch: raise breached complaints
    cx-agent intent-mining   the unhandled-intent mining report
    cx-agent attribution-run attribute every closed conversation
    cx-agent attribution     the attribution report (cause x desk,
                             this window against the previous one)
    cx-agent annotation-queue the human review queue + agreement
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
            print(f"{esc.escalation_id}  queue={esc.queue}  reason={esc.reason}  "
                  f"conv={esc.conversation_id}")
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
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
