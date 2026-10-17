"""Operator CLI: seed the book, show readiness, list the queue.

    cx-agent seed        write the Northbeam book into the store
    cx-agent readiness   print the profile the app will run with
    cx-agent escalations list open escalations (the human queue)
"""

from __future__ import annotations

import json
import sys

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
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
