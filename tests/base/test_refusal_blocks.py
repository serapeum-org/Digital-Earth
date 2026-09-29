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

**A property read is not a call, and that was the hole (R2-M12).** Counting `ast.Call` alone left a block
whose second raising thing is an attribute read invisible here, and one such block was *exploited* to prove
it: with `Map.figure_spec` — read inside the block, not hoisted — made to raise the very message the block
matches, `test_the_drawing_comes_off_and_the_record_stays_empty` passed while `draw_layer`, the call it
exists to refuse, never ran. So the second half below counts a read of a name the package serves through
`@property` as what it is: a call, spelled without parentheses. Plain attribute reads are *not* counted — a
stored field cannot run code — which is what keeps the reading from flagging every `self.DEM` in an
argument list (measured: 34 blocks hold an attribute read of some kind, 7 hold a property read).

That second half carries a **named list**, for the reason the first half's is gone: five of the seven sites
are in `interactive/`, `three_d/` and `test_quickplot.py`, which the change that widened this could neither
fix nor verify — the tiers' suites need environments it does not run. The list is a *subset* check keyed on
the read rather than on a line number, so clearing a site never reddens this file and a line shift above one
never matters. Each is reported to the reviewer rather than excused silently.
"""

import ast
from pathlib import Path

import pytest

import digitalearth

#: The tree scanned: every test module, not merely the tier that was last swept.
TESTS_ROOT = Path(__file__).resolve().parents[1]

#: The context managers whose body may hold only one call that can raise. ``pytest.warns`` is here for the
#: same reason (S9088): a second warning-raising call inside the block makes it ambiguous which was matched.
REFUSAL_MANAGERS = ("raises", "warns")

#: This module, which quotes the pattern in prose and must not be scanned for it.
GUARD_MODULE = Path(__file__).resolve()

#: The package's own source. Only its *path* comes from the import; it is read by AST like the tests are, so
#: a property declared in an engine-backed tier is still seen from the lean ``dev`` environment.
PACKAGE_ROOT = Path(digitalearth.__file__).resolve().parent

#: The property reads this guard's second half already found, by module, with the reason each is not fixed
#: here rather than merely excused.
#:
#: Keyed on the read and not on a line number: an entry has to survive every edit above the site it names,
#: and it is a **subset** check, so fixing one of these never turns this file red for its owner. Each read is
#: a genuine instance of the shape — `scene.figure_spec` in the 3-D seam test is the sharpest, since
#: `figure_spec` can itself raise `OffLimbError`, which is exactly what that block matches — and each lives in
#: a tier whose suite needs an environment the `dev` gate does not run, so a fix here could not be verified:
#:
#: - ``interactive/`` and ``test_quickplot.py``'s ``crs=dataset.epsg`` / ``crs=fc.epsg``: a *pyramids*
#:   property, read into the argument list of the refused `quickmap` call.
#: - ``three_d/test_seam3d.py``'s ``scene.figure_spec``: the same shape as the static site this commit fixed.
KNOWN_PROPERTY_READS = {
    "interactive/test_interactive_autostyle.py": {"dataset.epsg"},
    "test_quickplot.py": {"dataset.epsg", "fc.epsg"},
    "three_d/test_seam3d.py": {"scene.figure_spec"},
}


def _property_names() -> set:
    """Return every attribute name the package computes on read rather than stores.

    Returns:
        The names declared under ``@property`` or ``@cached_property`` anywhere in the package. A read of one
        runs package code and can raise, which is the whole reason it counts beside a call.
    """
    names = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                spelled = ast.unparse(decorator)
                if spelled == "property" or spelled.endswith("cached_property"):
                    names.add(node.name)
    return names


def _refusal_blocks(path: Path) -> list:
    """Return every ``pytest.raises``/``warns`` block in `path`, with the calls its body makes.

    Args:
        path: The test module to read.

    Returns:
        One ``(node, calls)`` pair per block, `calls` being every `ast.Call` its body holds.
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
        found.append(
            (
                node,
                [
                    inner
                    for statement in node.body
                    for inner in ast.walk(statement)
                    if isinstance(inner, ast.Call)
                ],
            )
        )
    return found


def _property_reads(node: ast.With, calls: list, properties: set) -> list:
    """Return the property reads in a block's body that sit beside its calls.

    Args:
        node: The refusal block.
        calls: Every call its body makes.
        properties: The names the package serves through a property.

    Returns:
        The reads, unparsed. The dotted path *leading to* a call is excluded — ``m.colorbar(...)`` is one
        thing that can raise, not two — so what is left is a read the block performs in its own right,
        typically a fixture handed to the refused call as an argument.
    """
    leading = {
        id(inner)
        for call in calls
        for inner in ast.walk(call.func)
        if isinstance(inner, ast.Attribute)
    }
    return [
        ast.unparse(inner)
        for statement in node.body
        for inner in ast.walk(statement)
        if isinstance(inner, ast.Attribute)
        and id(inner) not in leading
        and inner.attr in properties
    ]


def _blocks_with_two_throwing_calls(path: Path) -> list:
    """Return every refusal block in `path` whose body holds more than one call.

    Args:
        path: The test module to read.

    Returns:
        One ``"<relative path>:<line>"`` per offending block, with the called names, so a failure says where
        to hoist and what to hoist.
    """
    found = []
    for node, called in _refusal_blocks(path):
        if len(called) > 1:
            names = [ast.unparse(call.func) for call in called]
            found.append(
                f"{path.relative_to(TESTS_ROOT).as_posix()}:{node.lineno} -> {names}"
            )
    return found


def _blocks_whose_second_raise_is_a_property_read(path: Path, properties: set) -> list:
    """Return every refusal block in `path` holding a property read beside its one call.

    Args:
        path: The test module to read.
        properties: The names the package serves through a property.

    Returns:
        One ``(site, reads)`` pair per offending block. Blocks with two *calls* are left to the check above,
        so the two halves cannot report the same site twice.
    """
    found = []
    for node, called in _refusal_blocks(path):
        if len(called) > 1:
            continue
        reads = _property_reads(node, called, properties)
        if len(called) + len(reads) > 1:
            site = f"{path.relative_to(TESTS_ROOT).as_posix()}:{node.lineno}"
            found.append((site, tuple(reads)))
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


@pytest.fixture(scope="module")
def property_read_offenders() -> dict:
    """Every refusal block whose second thing that can raise is a property read, by module.

    Returns:
        ``{relative module path: {"sites": [...], "reads": {...}}}`` — the sites for the failure message, the
        reads for the subset the named list is checked against.
    """
    properties = _property_names()
    grouped: dict = {}
    for path in _test_modules():
        found = _blocks_whose_second_raise_is_a_property_read(path, properties)
        if found:
            grouped[path.relative_to(TESTS_ROOT).as_posix()] = {
                "sites": [site for site, _ in found],
                "reads": {read for _, reads in found for read in reads},
            }
    return grouped


class TestTheSecondRaiseIsNotAPropertyRead:
    """The half R2-M12 asks for: a read that runs package code counts beside the call it sits next to."""

    def test_no_module_holds_an_unnamed_property_read_in_a_refusal_block(
        self, property_read_offenders
    ):
        """A raising property inside the block passes the test while the refused call never runs.

        Args:
            property_read_offenders: The audit of the tree.

        Test scenario:
            Exploited before this check existed: `Map.figure_spec`, read inside a block that matched
            "needs 2 labels; got 1", was made to raise that message, and the test passed with `draw_layer`
            unreached. Anything not in `KNOWN_PROPERTY_READS` fails here; the named reads are a subset check,
            so clearing one of them cannot redden this file for whoever clears it.
        """
        unnamed = {
            module: found["sites"]
            for module, found in property_read_offenders.items()
            if found["reads"] - KNOWN_PROPERTY_READS.get(module, set())
        }
        assert unnamed == {}, (
            "these refusal blocks read a property beside the call they refuse, so a raise from the read "
            f"would pass the test: {unnamed}. Hoist the read above the `with`"
        )

    def test_every_named_read_is_in_a_module_the_audit_still_reads(self):
        """A list of exceptions outlives the files it names, and then reads as a rule nobody checks.

        Test scenario:
            The one way this list can rot without a test failing: the module is renamed or deleted and the
            permission stays. Not an equality with what the audit finds — that would go red the moment one of
            the five sites is fixed, which is the thing this list exists to avoid.
        """
        modules = {path.relative_to(TESTS_ROOT).as_posix() for path in _test_modules()}
        stale = sorted(set(KNOWN_PROPERTY_READS) - modules)
        assert stale == [], f"{stale} are named here but no longer in the tree"

    def test_the_property_registry_reads_the_package(self):
        """An empty registry would leave the check above green over a tree it never measured.

        Test scenario:
            The reading turns entirely on this set: no names, no property reads, no finding. Measured when it
            was added — 50 names across the five tiers — and held to a floor plus four names that are what the
            audit actually reports on, one of them (`figure_spec`) the exploited one.
        """
        properties = _property_names()
        assert len(properties) > 30, f"only {sorted(properties)} were found"
        missing = {"figure_spec", "epsg", "drawn", "layers"} - properties
        assert missing == set(), f"the registry missed {sorted(missing)}"


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
