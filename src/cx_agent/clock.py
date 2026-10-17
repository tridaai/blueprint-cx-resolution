"""One clock for the whole book.

Domain rules that speak of days — the trailing-90-day credit
history, the date a dispute case was opened — read the date here
and nowhere else, so a deployment (or a test) can pin "today"
with CX_TODAY and every rule agrees on what it is.
"""

from __future__ import annotations

import os
from datetime import date


def today() -> date:
    pinned = os.environ.get("CX_TODAY", "").strip()
    if pinned:
        return date.fromisoformat(pinned)
    return date.today()
