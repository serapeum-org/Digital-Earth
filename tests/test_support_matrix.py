"""Generate the backend support matrix from the tiers' capabilities, and ratchet that it stays current — U-5.

:meth:`~digitalearth.base.capabilities.Capabilities.to_dict` is labelled "for the support matrix the docs
build", but nothing built the matrix. This does: it renders the four tiers' declared capabilities into one
markdown page (:data:`SUPPORT_MATRIX_PATH`) — a table per dimension (layer kinds, encoding channels, data-driven
channels, classification schemes, features) showing which backend supports each, and a section listing what a
backend omits and the reason it gave. The page is in the docs ``nav``, so the site carries a single honest
answer to "which backend can do what", derived from the code rather than hand-maintained.

Like the API snapshot (``tests/test_api_snapshot.py``), it reads only the engine-free capability declarations,
so it runs in the lean ``dev`` matrix, and it is guarded as a ratchet: the committed page must equal what the
live capabilities render, or the test fails with the diff. **To change it on purpose:** change the capability,
run ``python -m tests.test_support_matrix`` to rewrite the page, and commit it.
"""

import textwrap
from pathlib import Path
from typing import Dict, List

from tests.capability_surface import tier_capabilities
from tests.ratchet import assert_matches_committed

#: The committed page, under the docs reference section (and in ``mkdocs.yml``'s ``nav``).
SUPPORT_MATRIX_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "reference" / "support-matrix.md"
)

#: The backends as columns, left to right. The default tier first, then the three extras. Guarded against the
#: package's discovered tier set by ``tests/test_api_snapshot.py::test_the_snapshot_and_matrix_cover_every_tier``.
BACKENDS: List[str] = ["matplotlib", "web", "interactive", "3d"]

#: The set-valued capability dimensions to tabulate, as (heading, capability key).
_DIMENSIONS = (
    ("Layer kinds", "kinds"),
    ("Encoding channels", "channels"),
    ("Data-driven channels", "data_driven"),
    ("Classification schemes", "schemes"),
    ("Features", "features"),
)

_PREAMBLE = (
    "<!-- This page is generated from each tier's Capabilities declaration. Do not edit it by hand; run\n"
    "`python -m tests.test_support_matrix` to regenerate it, and tests/test_support_matrix.py fails if it\n"
    "drifts. -->\n\n"
    "# Backend support matrix\n\n"
    "What each rendering backend can draw and style, read straight from its `Capabilities`\n"
    "declaration. `✓` means the backend supports it; `—` means it does not.\n"
)


def _dimension_table(heading: str, key: str, caps: Dict[str, dict]) -> str:
    """Render one capability dimension as a markdown table, backends across the columns.

    Args:
        heading: The section heading.
        key: The capability-dict key whose set is tabulated (e.g. ``"kinds"``).
        caps: Each backend's ``to_dict()``.

    Returns:
        The ``## heading`` section with one row per value (the sorted union across backends).
    """
    values = sorted({value for backend in BACKENDS for value in caps[backend][key]})
    header = "| capability | " + " | ".join(BACKENDS) + " |"
    rule = "| --- | " + " | ".join("---" for _ in BACKENDS) + " |"
    rows = [
        "| `"
        + value
        + "` | "
        + " | ".join(
            "✓" if value in caps[backend][key] else "—" for backend in BACKENDS
        )
        + " |"
        for value in values
    ]
    return "\n".join([f"## {heading}", "", header, rule, *rows])


def _absent_section(caps: Dict[str, dict]) -> str:
    """Render the per-backend "what it omits, and why" section from each tier's ``absent`` reasons.

    Args:
        caps: Each backend's ``to_dict()``.

    Returns:
        The ``## What a backend omits, and why`` section; a backend with nothing absent is skipped.
    """
    blocks = ["## What a backend omits, and why", ""]
    for backend in BACKENDS:
        absent = caps[backend]["absent"]
        if not absent:
            continue
        blocks.append(f"### {backend}")
        blocks.append("")
        for name in sorted(absent):
            blocks.append(
                textwrap.fill(
                    f"- **`{name}`** — {absent[name]}",
                    width=112,
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def render_support_matrix() -> str:
    """Build the whole support-matrix page from the live, engine-free capability declarations.

    Returns:
        The markdown page: a preamble, one table per dimension, then the absent-and-why section. Deterministic
        (backends fixed, values sorted) so two machines render it byte-identically.
    """
    caps = {backend: tier_capabilities(backend) for backend in BACKENDS}
    sections = [_dimension_table(heading, key, caps) for heading, key in _DIMENSIONS]
    body = "\n\n".join([_PREAMBLE.rstrip(), *sections, _absent_section(caps).rstrip()])
    return body + "\n"


def test_the_support_matrix_page_is_current():
    """The ratchet: the committed support-matrix page must equal what the live capabilities render.

    Test scenario:
        Render the page from the live capability declarations and compare it to the committed
        ``docs/reference/support-matrix.md``. A capability added, removed or re-explained with no regenerate
        fails here with the unified diff inline, so the docs cannot silently fall out of step with the code.
    """
    assert_matches_committed(
        render_support_matrix(),
        SUPPORT_MATRIX_PATH,
        what="support matrix",
        regenerate="python -m tests.test_support_matrix",
    )


if __name__ == "__main__":  # regenerate the page after a deliberate capability change
    SUPPORT_MATRIX_PATH.write_text(render_support_matrix(), encoding="utf-8")
    print(f"wrote {SUPPORT_MATRIX_PATH}")
