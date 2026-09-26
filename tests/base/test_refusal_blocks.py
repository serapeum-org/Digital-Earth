"""One throwing call per refusal block, held by a guard rather than by another sweep (R2-L1, SonarCloud S5778).

A ``with pytest.raises(...)`` block holding two calls that can raise does not test what it says: the fixture
can fail and the test passes for the wrong reason, which is why SonarCloud reports it as a BUG. Four commits
have swept this tree for it — ``cb7c171f`` (base), ``c9d421d7`` (interactive), ``1007637a`` (3-D) and
``6a4e5871`` (static) — and round 2 still found three sites, one of them *added* by a later commit to a file an
earlier sweep had already cleaned. A sweep cannot hold a rule; this can.

**ruff's ``PT012`` does not catch this shape**, measured rather than assumed: ``ruff check --select PT012 src
tests`` reports "All checks passed!" over this tree while the sites in :data:`KNOWN` are still present. PT012
flags a block of several *statements*; S5778 counts the *calls* inside it, and every site here is one
statement with a call nested in the arguments of another. So the rule the finding suggested would not have
held, and an AST check is what is left.

It is written as a **ratchet**: the violators must be a subset of :data:`KNOWN`. A new site fails here, and
clearing a listed one needs no edit to this file — which is what keeps this guard from turning someone else's
fix red. The price of that is a cleared entry lingering as permission nothing needs; what is refused instead is
an entry naming a module that is no longer there, which is how a list of exceptions rots unnoticed.
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

#: The sites still carrying two throwing calls, each with the finding that owns it. Both are outside the trees
#: this guard was added with, and both are named in round 2: `tests/static/test_static_contours.py` is R2-L1's
#: third site, and `tests/web/test_web_raster.py` is the same shape found by this check. Listed so the guard
#: can be green today and still refuse a *new* one.
KNOWN = {
    "static/test_static_contours.py",
    "web/test_web_raster.py",
}


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
    """The ratchet: no new site, and the listed ones may be cleared without touching this file."""

    def test_no_module_outside_the_known_list_holds_two_throwing_calls(self, offenders):
        """A block with two calls that can raise passes when its fixture fails, which is the whole defect.

        Args:
            offenders: The audit of the tree.
        """
        unlisted = {
            module: sites for module, sites in offenders.items() if module not in KNOWN
        }
        assert unlisted == {}, (
            "these refusal blocks hold more than one call that can raise, so a failing fixture would pass "
            f"the test: {unlisted}. Hoist everything but the refused call above the `with`"
        )

    def test_every_known_entry_names_a_module_that_is_still_there(self):
        """Permission granted to a file that has been renamed away is permission nobody can see the end of.

        Test scenario:
            Deliberately *not* "the entry still offends". Both listed sites belong to findings other branches
            own, and a check that went red the moment one of them was cleared would make their fix require an
            edit to this file — the cross-branch collision this guard exists to avoid. Clearing an entry
            therefore keeps the subset check green, and the entry may be dropped whenever someone is next in
            here. What is refused is an entry naming nothing, which is how a list of exceptions rots.
        """
        missing = sorted(
            module for module in KNOWN if not (TESTS_ROOT / module).is_file()
        )
        assert missing == [], (
            f"{missing} are listed as known offenders and no longer exist, so the entries grant permission "
            "to nothing — drop them"
        )

    def test_the_audit_reaches_every_tier_s_tests(self):
        """A guard scoped to one tier is how three sweeps each missed the next tier's copy.

        Test scenario:
            Measured on the audit's own walk rather than on a second one: a subset check is only worth what
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
