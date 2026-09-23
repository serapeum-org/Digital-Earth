"""The numbered contract and the citations of it must not drift apart.

The four tiers are held to a numbered contract, and until #326 it existed only as docstring bullets spread
across the four tiers' contract-test files. A clause therefore had no home: nothing said what a given number
*was*, nothing listed where it was asserted, and nothing checked that two tiers still agreed about it.
Amending one clause took edits at seven statement sites, two of them found by grepping.

:mod:`digitalearth.base.contract_clauses` is the home. This is the guard over it, in the same shape as the
``UNDRAWN_KINDS`` guard in ``tests/base/test_renderer_conformance.py``: a declaration nothing checks is a
declaration that rots, so drift is checked **both ways**.

1. **Every cited number is a clause that exists.** A citation of a number nobody ever wrote down reads as
   authority it does not have, and is the typo this half catches.
2. **Every clause is pinned by some test.** A clause no test names is a rule nothing holds a tier to. "Pinned"
   is read as: named in a test module's own docstring, or in the docstring of a ``Test``-prefixed class in
   one. That is a proxy — a docstring is not an assertion — but it is the honest one available here, because
   the tier files that carry the assertions ``importorskip`` their engines, so importing them to ask would
   skip three tiers out of four in the lean ``dev`` env. The files are read with :mod:`ast` instead, which
   costs no import and sees every tier.

**Three namespaces spell an identifier the same way.** This contract numbers its clauses ``C<n>``; the review
rounds number their findings by severity and rank, so a review's first critical is also written that way; and
Python's method-resolution order is computed by the C3 linearisation. A token whose own line or either
neighbour names a review or a linearisation is read as belonging to those namespaces rather than to this
contract — a window rather than a line, because the sentence that marks it wraps. The filter is narrow on
purpose, and it is the reason a new citation of this contract should keep clear of both words.
"""

import ast
import re
from pathlib import Path

import pytest

from digitalearth.base.contract_clauses import CLAUSES, clause

#: The repository root, from which the scanned trees hang.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The trees a citation may live in. Everything the package ships and everything that tests it.
SCANNED_TREES = (REPO_ROOT / "src", REPO_ROOT / "tests")

#: A citation of this contract: the bare token. It may open a docstring, so a quote before it is fine; a quote
#: *after* it makes the token a string of its own, which is matplotlib's ``colors="C0"`` colour-cycle name and
#: not a citation of anything.
CITATION = re.compile(r"(?<!\w)C(\d+)\b(?![\"'])")

#: What marks a window as belonging to one of the other two namespaces — a review finding, or the C3
#: linearisation. Read the module docstring before widening this: every term here is one a citation of *this*
#: contract must keep clear of.
OTHER_NAMESPACE = re.compile(r"\breview\b|C\d+/[HMLN]\d+|lineari[sz]", re.IGNORECASE)

#: This module cites numbers while explaining the rule, and must not count as anybody's pin.
GUARD_MODULE = Path(__file__).resolve()

#: The module that *states* the clauses. It spells numbers in order to define them, and spells one that is
#: deliberately not a clause to show what the lookup does with it, so reading it as a citation site would have
#: the direction backwards.
CANONICAL_MODULE = REPO_ROOT / "src" / "digitalearth" / "base" / "contract_clauses.py"


def _cited_in(text: str, origin: str) -> dict[int, list[str]]:
    """Collect the clause numbers a block of text cites, with where each was found.

    A citation is judged against the line it sits on **and its two neighbours**, because the sentence that
    marks it as another namespace's often wraps: ``tests/test_mixin_contract.py`` writes "and C3 cannot" on
    one line and "linearize it" on the next, and a line-at-a-time reading would take that for this contract's
    third clause.

    Args:
        text: The text to read.
        origin: How to name this text in a failure message — a path, or a path and a line offset.

    Returns:
        Clause number to the sites citing it, each site as ``"<origin>:<line>"``.
    """
    found: dict[int, list[str]] = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        context = "\n".join(lines[max(index - 1, 0) : index + 2])
        if OTHER_NAMESPACE.search(context):
            continue
        for match in CITATION.finditer(line):
            found.setdefault(int(match.group(1)), []).append(f"{origin}:{index + 1}")
    return found


def _merge(into: dict[int, list[str]], more: dict[int, list[str]]) -> None:
    """Fold one citation map into another.

    Args:
        into: The map being accumulated. Changed in place.
        more: The citations to add.
    """
    for number, sites in more.items():
        into.setdefault(number, []).extend(sites)


def _python_files() -> list[Path]:
    """Every Python file a citation could live in.

    Returns:
        The files under :data:`SCANNED_TREES`, byte-compiled caches excluded, in a stable order.
    """
    return sorted(
        path
        for tree in SCANNED_TREES
        for path in tree.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _relative(path: Path) -> str:
    """Name a file the way a failure message should print it.

    Args:
        path: The file.

    Returns:
        Its path relative to the repository root, with forward slashes on every platform.
    """
    return path.relative_to(REPO_ROOT).as_posix()


def _all_citations() -> dict[int, list[str]]:
    """Collect every citation of the contract in the scanned trees.

    Returns:
        Clause number to the sites citing it.
    """
    citations: dict[int, list[str]] = {}
    for path in _python_files():
        if path == CANONICAL_MODULE:
            continue
        _merge(citations, _cited_in(path.read_text(encoding="utf-8"), _relative(path)))
    return citations


def _pinning_docstrings(path: Path) -> list[str]:
    """Return the docstrings of `path` that can pin a clause.

    A test module's own docstring and the docstring of every ``Test``-prefixed class in it. Method docstrings
    are deliberately left out: a clause named only in passing, while explaining one assertion, is a mention
    rather than a pin, and counting it would let a clause survive here after the class that held it went.

    Args:
        path: The test module to read. Parsed, never imported, so an engine-gated module is still read.

    Returns:
        The docstrings, in source order, with the empty ones dropped.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = [ast.get_docstring(tree)]
    docstrings += [
        ast.get_docstring(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test")
    ]
    return [text for text in docstrings if text]


def _pins() -> dict[int, list[str]]:
    """Collect the clauses the test suite pins, and where.

    Returns:
        Clause number to the test modules pinning it.
    """
    pinned: dict[int, list[str]] = {}
    for path in _python_files():
        if path == GUARD_MODULE or REPO_ROOT / "tests" not in path.parents:
            continue
        for text in _pinning_docstrings(path):
            _merge(pinned, _cited_in(text, _relative(path)))
    return pinned


class TestTheClauseTable:
    """The canonical list is well formed before anything is compared against it."""

    def test_every_clause_is_filed_under_its_own_number(self):
        """A clause reached by the wrong key would be cited correctly and read wrongly.

        Test scenario:
            The table is keyed by each clause's own ``number``, so a typo in one or the other is the only
            way the two can disagree — and it would hand `clause(7)` a rule that is not C7's.
        """
        misfiled = sorted(
            key for key, stated in CLAUSES.items() if stated.number != key
        )
        assert misfiled == [], (
            f"these keys of CLAUSES hold a clause numbered something else: {misfiled}"
        )

    def test_every_clause_states_a_rule(self):
        """An empty clause is the thing this module exists to stop: a number with no wording.

        Test scenario:
            Each entry carries both a handle and the rule itself, so a citation can be resolved to text a
            reader can act on rather than to a placeholder.
        """
        wordless = sorted(
            stated.name
            for stated in CLAUSES.values()
            if not (stated.title and stated.rule)
        )
        assert wordless == [], (
            f"these clauses are a number and nothing else: {wordless}"
        )

    def test_a_number_that_is_not_a_clause_is_refused(self):
        """Asking for a clause nobody wrote says so, and says what does exist.

        Test scenario:
            The lookup is how a test resolves a citation, so the number that was never a clause has to fail
            loudly there rather than return something plausible.
        """
        absent = max(CLAUSES) + 1
        with pytest.raises(KeyError, match=f"C{absent} is not a clause"):
            clause(absent)


class TestTheCitationsAndTheClausesDoNotDrift:
    """Drift either way is a defect: a citation with no clause, or a clause with no test."""

    def test_every_cited_number_is_a_clause_that_exists(self):
        """A citation of a number nobody wrote down borrows authority the contract never gave it.

        Test scenario:
            Every Python file under ``src/`` and ``tests/`` is read for citations, and each number is looked
            up in the canonical table. A mistyped or invented number fails here, naming the site, which is
            what grepping for it used to be relied on to do.
        """
        undefined = {
            number: sites
            for number, sites in sorted(_all_citations().items())
            if number not in CLAUSES
        }
        assert undefined == {}, (
            f"these numbers are cited but are not clauses: {undefined}; state each in "
            "digitalearth.base.contract_clauses.CLAUSES, or correct the citation"
        )

    def test_every_clause_is_pinned_by_some_test(self):
        """A clause no test names is a rule nothing holds a tier to.

        Test scenario:
            Each clause in the canonical table must be named by a test module's docstring or by a
            ``Test``-prefixed class's docstring somewhere under ``tests/``. Deleting the last class that
            pinned a clause fails here rather than leaving the clause standing with nothing behind it.
        """
        pinned = _pins()
        unpinned = sorted(
            CLAUSES[number].name for number in CLAUSES if number not in pinned
        )
        assert unpinned == [], (
            f"these clauses are stated but no test names them: {unpinned}; pin each in the tier's contract "
            "file, or take the clause out of digitalearth.base.contract_clauses"
        )
