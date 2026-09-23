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

**Four namespaces used to spell an identifier the same way**, and that ambiguity is now resolved at the
source rather than guessed at here. This contract numbers its clauses ``C<n>``. The review rounds numbered
their findings the same way, so a review's first critical read as clause 1; they are now written ``R-C1``.
Python's method-resolution order was spelled the same way, after the algorithm that computes it; it is now
named in words. And matplotlib's colour cycle spells a colour ``"C0"``, which is a quoted string and never
prose.

The guard therefore matches on **shape** — see :data:`CITATION`. It previously read a window of surrounding
text and dropped any token near the words "review" or "linearise", which guessed wrong in both directions: a
real citation in a sentence mentioning a review was silently unchecked, and a review finding phrased without
either word was reported as a clause that does not exist. A guard against silent drift must not itself fail
silently, so the heuristic is gone.
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

#: A citation of this contract: the bare token, with nothing joined to its left.
#:
#: The exclusions are **shape**, not prose. An earlier version read a window of surrounding text and dropped any
#: token near the words "review" or "linearise", which guessed wrong in both directions: a real citation in a
#: sentence that happened to mention a review round was silently not checked, and a review finding phrased
#: without either word was reported as a clause that does not exist. Both failures are silent, and the guard
#: exists to stop exactly that kind of silence.
#:
#: So the other namespaces are now spelled so they cannot collide, and this matches only what is left:
#:
#: * ``(?<![\w-])`` — a review finding is written ``R-C1``/``R-H2``, so a hyphen to the left is not a citation.
#: * ``(?![\"'])`` — a quote to the right makes the token a string of its own, which is matplotlib's
#:   ``colors="C0"`` colour-cycle name.
#: * Python's method-resolution order is described by name rather than as "C3".
CITATION = re.compile(r"(?<![\w-])C(\d+)\b(?![\"'])")

#: A review-round finding written in the clause spelling — the one collision that keeps coming back, because
#: a review round generates fresh ``C<n>`` findings every time one is run.
#:
#: The shape rules above cannot tell this apart: the token really is a bare ``C<n>`` in prose, and only the
#: word in front of it says it belongs to a review round rather than to the contract. What *can* be checked is
#: that word, so this matches the phrase and not the number — a finding cited as ``review C1`` is caught here
#: and rewritten ``review R-C1``, after which :data:`CITATION` no longer sees it at all.
#:
#: The first sweep (commit ``7d98a6e0``) migrated ``tests/`` and left three sites under ``src/`` writing a
#: finding this way, where they were silently counted as citations of clause 1 (review R-M3). Nothing said so,
#: because clause 1 exists. This is what says so next time.
REVIEW_FINDING = re.compile(r"\breviews?\s+C\d")

#: This module cites numbers while explaining the rule, and must not count as anybody's pin.
GUARD_MODULE = Path(__file__).resolve()

#: The module that *states* the clauses. It spells numbers in order to define them, and spells one that is
#: deliberately not a clause to show what the lookup does with it, so reading it as a citation site would have
#: the direction backwards.
CANONICAL_MODULE = REPO_ROOT / "src" / "digitalearth" / "base" / "contract_clauses.py"

#: The two modules that spell clause numbers to **explain** the rule rather than to cite it, excluded from
#: both directions of the guard.
#:
#: The exclusions used to be asymmetric — :func:`_all_citations` skipped the canonical module only, and
#: :func:`_pins` the guard module only — so this module's own prose was scanned for citations and reported as
#: one (review R-N2). Skipping the canonical module in :func:`_pins` is moot, since that function already
#: keeps to ``tests/``; it is stated anyway, because a rule with a hole on one side is the shape that let the
#: first hole through, and a reader comparing the two functions should find the same answer in both.
SELF_REFERRING = (GUARD_MODULE, CANONICAL_MODULE)


def _cited_in(text: str, origin: str) -> dict[int, list[str]]:
    """Collect the clause numbers a block of text cites, with where each was found.

    A citation is judged **line by line**, on its own shape. It used to be judged against a window of
    surrounding prose, because the other namespaces spelled themselves the same way and only the sentence
    around a token said which one it was. They no longer do, so there is nothing left to infer.

    Args:
        text: The text to read.
        origin: How to name this text in a failure message — a path, or a path and a line offset.

    Returns:
        Clause number to the sites citing it, each site as ``"<origin>:<line>"``.
    """
    found: dict[int, list[str]] = {}
    for index, line in enumerate(text.splitlines()):
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
        if path in SELF_REFERRING:
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
        if path in SELF_REFERRING or REPO_ROOT / "tests" not in path.parents:
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


#: What :data:`CITATION` must and must not read as a citation, as `(line, the numbers it yields)`.
#:
#: The guard is a regular expression over every line of `src/` and `tests/`, and nothing held it to the four
#: namespaces its docstring says it separates (review R-M3, gap 5). The last two rows are the shape rules'
#: **cost**, pinned rather than hidden: an upper bound written after a hyphen and a possessive written with an
#: apostrophe are real citations that the exclusions drop. Both are false negatives, which is the safe
#: direction — the guard reports a citation of a clause that does not exist, so a missed one is a citation
#: nobody checked rather than a failure nobody can explain.
CITATION_CASES = [
    ("a change to contract C4 is a change to every tier", [4]),
    ("clause C13 and clause C7 both bind the display CRS", [13, 7]),
    ("the finding R-C1 was raised by a review round, not by this contract", []),
    ('cleopatra draws the first series in colors="C0"', []),
    ("the clauses run C1-C14", [1]),
    ("C7's other half is a kind the tier does not draw", []),
]


class TestTheCitationPatternIsTheShapeRuleItDocuments:
    """The regular expression is the whole guard, and nothing held it to the rules it claims.

    Four namespaces spell ``C<n>``; three of them respell themselves so this pattern skips them, and that
    arrangement is only as good as the pattern. A change to it that silently stopped matching — a stray
    anchor, a lost word boundary — would empty :func:`_all_citations` and turn both drift checks green with
    nothing behind them.
    """

    @pytest.mark.parametrize(
        ("line", "expected"),
        CITATION_CASES,
        ids=[row[0][:40] for row in CITATION_CASES],
    )
    def test_a_line_yields_exactly_the_numbers_it_cites(self, line, expected):
        """Each namespace, and each cost of telling them apart by shape.

        Args:
            line: The line to read.
            expected: The clause numbers it should yield, in the order they appear.
        """
        found = [int(match.group(1)) for match in CITATION.finditer(line)]
        assert found == expected, f"{line!r} yielded {found}"


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

    def test_no_review_finding_is_written_in_the_clause_spelling(self):
        """The collision that regenerates: every review round mints fresh `C<n>` findings.

        Test scenario:
            The spelling rule is stated in `digitalearth.base.contract_clauses` and was applied to `tests/`
            alone, leaving three sites under `src/` citing a finding as a bare `C1` — counted as citations
            of clause 1, and passing because clause 1 exists (review R-M3). The number cannot be judged by
            shape here, so the *phrase* is: a finding introduced by the word "review" is a finding, whatever
            number follows, and it belongs in the `R-` spelling that the citation pattern skips. The two
            modules that explain the rule are excluded, as they are from the citation scan, because quoting
            the spelling to forbid it is not using it.
        """
        offenders = []
        for path in _python_files():
            if path in SELF_REFERRING:
                continue
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if REVIEW_FINDING.search(line):
                    offenders.append(f"{_relative(path)}:{index + 1}")
        assert offenders == [], (
            f"these sites cite a review finding in the contract's own spelling: {sorted(offenders)}; write "
            "it R-C1 / R-H2 / R-M9, which the citation pattern skips"
        )

    def test_the_guard_does_not_read_its_own_prose_as_a_citation(self):
        """A guard that cites the numbers it explains must not count itself among the citers.

        Test scenario:
            The two self-exclusions were asymmetric (review R-N2): `_pins` skipped this module and
            `_all_citations` did not, so every number this file spells while documenting the rule was
            collected as a citation — and the docstring's own example of the spelling to avoid was
            attributed to the clause it names. Both directions now skip both modules, which is what this
            asserts: a sentence added here explaining a clause cannot make this file a citation site.
        """
        mine = f"{_relative(GUARD_MODULE)}:"
        cited_here = sorted(
            site
            for sites in _all_citations().values()
            for site in sites
            if site.startswith(mine)
        )
        assert cited_here == [], (
            f"the guard reads its own prose as citing the contract: {cited_here}; the module explains the "
            "spelling rather than using it, so it belongs in SELF_REFERRING"
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
