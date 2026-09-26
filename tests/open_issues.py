"""The issues a user-facing reason is allowed to point at, and how a reason names one.

Two tables in this suite show a tracker reference to a *user* rather than leaving it as provenance in a
comment: :data:`~digitalearth.base.contract.PENDING`, which answers "when does this tier get the method",
and `KEYWORD_SHORTFALLS` in `tests/test_contract_names.py`, which answers "why does this tier spell the
keyword differently". In both, a closed issue answers the question with "it is already done" — which is how
fifteen `PENDING` reasons came to point at #300 and #303 after both seams landed (#319).

`PENDING` has been held to this allowlist since #319; the keyword table was shape-checked only, against
``#\\d+|order \\d+|unscheduled``, so a row naming a closed issue passed (review R-L2). The same rot, one
table over. The allowlist lives here rather than in either module so both are held to one list, and so the
check that nothing in it has gone stale can see every reason that names an issue.

It is an allowlist and not a lookup on purpose: a test that asked GitHub would need a network and a token to
run, and would go red for an outage rather than for a defect. So the offline stand-in for "this issue is
open" is a short list a human maintains, and extending it is the deliberate act the checks exist to force.
A reason that names a roadmap **order** instead needs no entry here — which is the point, and why order is
the preferred form: orders outlive the issues that implement them.
"""

import re
from types import MappingProxyType
from typing import List, Mapping

__all__ = ["ISSUE_REFERENCE", "KNOWN_OPEN_ISSUES", "issues_named_in"]

#: An issue reference inside a reason, as a reader would follow it.
ISSUE_REFERENCE = re.compile(r"#(\d+)")

#: The issues a user-facing reason may name, with the title each carried when it was last checked against
#: the tracker **by hand, on 2026-09-25**. Checked with ``gh issue view <n> --json state,title``; every one
#: below answered ``OPEN``.
KNOWN_OPEN_ISSUES: Mapping[int, str] = MappingProxyType(
    {
        201: "The 3-D tier cannot draw line geometries — no rivers, roads, tracks or trajectories",
        226: "feat(static): add Map.lines for plain line geometry, matching the web tier",
        261: (
            "fix(api): legend() now has three incompatible shapes, and web still has no colorbar()"
        ),
        264: (
            "fix(api): layer_control() shares zero parameters between the two tiers that have it"
        ),
        265: (
            "fix(web): title(), fit_bounds() and labels() introduce third names for existing concepts"
        ),
        268: (
            "refactor(api): tiles() takes a name on one tier and a URL on another, through one "
            "overloaded provider="
        ),
        331: (
            "feat(interactive): polygons() cannot classify — no scheme= or k=, unlike the web tier"
        ),
        332: (
            "refactor(api): static and interactive builders spell opacity and column differently from "
            "the Core contract"
        ),
    }
)


def issues_named_in(reason: str) -> List[int]:
    """Return every issue number a reason names, in the order it names them.

    Args:
        reason: The user-facing string — a `PENDING` reason, or a keyword shortfall's.

    Returns:
        The numbers. A reason naming two issues yields both, so neither goes unchecked.
    """
    return [int(number) for number in ISSUE_REFERENCE.findall(reason)]
