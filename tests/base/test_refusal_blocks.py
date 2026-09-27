"""One throwing call per refusal block, held by a guard rather than by another sweep (R2-L1, SonarCloud S5778).

A ``with pytest.raises(...)`` block holding two calls that can raise does not test what it says: the fixture
can fail and the test passes for the wrong reason, which is why SonarCloud reports it as a BUG. Four commits
have swept this tree for it — ``cb7c171f`` (base), ``c9d421d7`` (interactive), ``1007637a`` (3-D) and
``6a4e5871`` (static) — and round 2 still found three sites, one of them *added* by a later commit to a file an
earlier sweep had already cleaned. A sweep cannot hold a rule; this can.

**ruff's ``PT012`` does not catch this shape**, measured rather than assumed: ``ruff check --select PT012 src
tests`` reports "All checks passed!" over this tree while the sites this guard reports are still present. PT012
flags a block of several *statements*; S5778 counts the *calls* inside it, and every site found here was one
statement with a call nested in the arguments of another. So the rule the finding suggested would not have
held, and an AST check is what is left.

It started as a **ratchet** — the violators had to be a subset of a `KNOWN` list — because its two outstanding
sites belonged to findings other branches owned, and a guard that went red the moment one of them was cleared
would have made their fix require an edit to this file. Both are cleared now
(``tests/static/test_static_contours.py``, the site ``f9bf8a8c`` added, and ``tests/web/test_web_raster.py``),
so the exception list is **gone** rather than left empty: an empty list is permission nothing needs, and the
check that policed the list could no longer fail. What is left is the stronger statement — *no* refusal block
in the tree holds two calls that can raise — which is what the sweeps were trying to say.
"""

import ast
from pathlib import Path

import pytest

#: The tree scanned: every test module, not merely the tier that was last swept.
TESTS_ROOT = Path(__file__).resolve().parents[1]

#: The context managers whose body may hold only one call that can raise. ``pytest.warns`` is here for the
#: same reason (S9088): a second warning-raising call inside the block makes it ambiguous which was matched.
REFUSAL_MANAGERS = ("raises", "warns")

#: This module, which quotes the pattern in prose and must not be scanned for it.
GUARD_MODULE = Path(__file__).resolve()


def _blocks_with_two_throwing_calls(path: Path) -> list:
    """Return every refusal block in `path` whose body holds more than one call.

    Args:
        path: The test module to read.

    Returns:
        One ``"<relative path>:<line>"`` per offending block, with the called names, so a failure says where
        to hoist and what to hoist.
    """
    found = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        managed = [
            item
            for item in node.items
            if any(name in ast.unparse(item.context_expr) for name in REFUSAL_MANAGERS)
        ]
        if not managed:
            continue
        called = [
            inner
            for statement in node.body
            for inner in ast.walk(statement)
            if isinstance(inner, ast.Call)
        ]
        if len(called) > 1:
            names = [ast.unparse(call.func) for call in called]
            found.append(
                f"{path.relative_to(TESTS_ROOT).as_posix()}:{node.lineno} -> {names}"
            )
    return found


def _test_modules() -> list:
    """Return every test module the audit reads, this guard excepted.

    One walk, shared by the audit and by the check that the audit reaches every tier — so narrowing the walk
    cannot leave that check reporting on a tree the audit never opened.

    Returns:
        The paths, sorted.
    """
    return [
        path
        for path in sorted(TESTS_ROOT.rglob("*.py"))
        if path.resolve() != GUARD_MODULE
    ]


@pytest.fixture(scope="module")
def offenders() -> dict:
    """Every offending block in the test tree, grouped by the module it is in.

    Returns:
        ``{relative module path: [site, ...]}`` — read by AST, so no test module is imported and every tier's
        files are seen from the lean ``dev`` environment.
    """
    grouped: dict = {}
    for path in _test_modules():
        sites = _blocks_with_two_throwing_calls(path)
        if sites:
            grouped[path.relative_to(TESTS_ROOT).as_posix()] = sites
    return grouped


class TestOneThrowingCallPerRefusalBlock:
    """No refusal block anywhere in the tree holds two calls that can raise, with nothing exempted."""

    def test_no_module_holds_a_block_with_two_throwing_calls(self, offenders):
        """A block with two calls that can raise passes when its fixture fails, which is the whole defect.

        Args:
            offenders: The audit of the tree.

        Test scenario:
            An equality rather than a subset: the two sites this guard was written alongside are fixed, so
            there is no exception list left to be a subset of. A new site anywhere under ``tests/`` fails
            here, and the message names the module, the line and what to hoist.
        """
        assert offenders == {}, (
            "these refusal blocks hold more than one call that can raise, so a failing fixture would pass "
            f"the test: {offenders}. Hoist everything but the refused call above the `with`"
        )

    def test_the_audit_reaches_every_tier_s_tests(self):
        """A guard scoped to one tier is how three sweeps each missed the next tier's copy.

        Test scenario:
            Measured on the audit's own walk rather than on a second one: the check above is only worth what
            it opened, and `TestNoMessageNamesAMethodTheTierLost` globbing `three_d/` alone is the finding
            (R2-M6) that says so.
        """
        seen = {
            path.relative_to(TESTS_ROOT).parts[0]
            for path in _test_modules()
            if len(path.relative_to(TESTS_ROOT).parts) > 1
        }
        missing = {"base", "static", "interactive", "three_d", "web"} - seen
        assert missing == set(), f"the audit never reaches {sorted(missing)}"
