"""The one comparison every U-5 ratchet makes: a committed artifact against the live value, diff on drift.

The three U-5 ratchets — the API snapshot, the support matrix and the known-deviations allowlist — each render a
piece of the package to text and hold a committed file to it, failing with a unified diff and how to regenerate
when they disagree. That compare-and-diff is one behaviour; this holds it once so the three do not repeat it.
"""

import difflib
from pathlib import Path


def assert_matches_committed(
    current: str, committed_path: Path, *, what: str, regenerate: str
) -> None:
    """Assert the committed file equals ``current``; on drift, fail with a unified diff and the regen command.

    Args:
        current: The freshly rendered or serialised text.
        committed_path: The committed artifact to compare against.
        what: A short noun phrase naming what drifted, for the message (e.g. ``"public API surface"``).
        regenerate: The command that rewrites ``committed_path`` after a deliberate change.

    Raises:
        AssertionError: when the committed file differs from ``current``, carrying the diff and the hint.
    """
    committed = committed_path.read_text(encoding="utf-8")
    if current == committed:
        return
    diff = "".join(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=f"{committed_path.name} (committed)",
            tofile="the live value",
        )
    )
    raise AssertionError(
        f"the {what} drifted from {committed_path.name}. If the change is intended, regenerate it with "
        f"`{regenerate}` and commit the diff; otherwise it is an unintended change.\n\n{diff}"
    )
