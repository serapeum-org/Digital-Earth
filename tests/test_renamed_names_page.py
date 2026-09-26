"""The page that teaches the renames must say what :data:`ALIASES` says (#318, order 27a).

`docs/reference/renamed-names.md` is where a caller is sent when a `DeprecationWarning` names a replacement,
and until this guard existed nothing held it to the code. It showed: it documented twelve **keyword** renames
and then dismissed the rest as "plus one method and one return type" — written when there was one. Order 27a
adopted six agreed renames and `set_extent` besides, taking `ALIASES` to eighteen method rows across four
tiers, and the page said one. A reader following it wrote the old spelling and got a warning the page did not
explain.

So the method table is now **derived-checked** rather than maintained by hand-in-hope, in the same shape as
the citation guard in ``tests/test_contract_clauses.py``: the prose is parsed and compared with the code, and
drift is checked **both ways**.

1. **Every alias is on the page.** An alias nobody documented is a warning a reader cannot act on — the
   direction that went wrong.
2. **Every row is an alias.** A row naming a rename that is not in `ALIASES` teaches a spelling the package
   does not forward, which is worse than silence: the call fails rather than warns.
3. **Every row names the replacement the code forwards to.** The two set checks above are keyed on the old
   spelling, so a row with the right old name and the wrong new one passes both. This reads the third column.

The parser is held to its own cases too (:class:`TestReadingTheTable`). A guard that silently reads **no**
rows would pass direction 2 for any page at all, so the check that it can read a row, and that it steps over
the header and the separator, is part of the guard rather than an afterthought.

Only the **method** table is guarded. The keyword renames have no declaration in the package to be held
against — each is a `renamed_parameter` call inside the tier that keeps it, and several are reached through
``**opts`` where no signature names them — so there is nothing here to compare a keyword row with. Saying so
is better than a guard that reads the tiers' signatures and quietly skips the three tiers whose engine the
lean ``dev`` environment does not install.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from digitalearth.base.contract import ALIASES

#: The repository root, from which the page hangs.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The page this module guards. Renamed from ``renamed-keywords.md`` when it grew the method table, because a
#: page covering both was still called after one of them.
PAGE = REPO_ROOT / "docs" / "reference" / "renamed-names.md"

#: The heading the method table lives under. The page carries a second table — the keyword renames — with a
#: **four**-column shape, so the section is found by its heading rather than by counting pipes: a parser that
#: read every table on the page would take a keyword row's `Method` column for a method rename.
METHOD_HEADING = "## The renamed methods"

#: A row of a markdown table: the cells between the outer pipes. Matched rather than split on ``|`` so a line
#: that is not a table row at all yields nothing instead of a one-cell row.
TABLE_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$")

#: The separator under a table's header, which is a row by shape and not by meaning.
SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")


def _cells(line: str) -> List[str]:
    """Return a markdown table row's cells, stripped of whitespace and of code-span backticks.

    Args:
        line: One line of the page.

    Returns:
        The row's cells, or an empty list when the line is not a table row or is the header separator.
    """
    matched = TABLE_ROW.match(line.strip())
    if matched is None:
        return []
    cells = [cell.strip().strip("`").strip() for cell in matched["cells"].split("|")]
    if all(SEPARATOR_CELL.match(cell) for cell in cells):
        return []
    return cells


def _rows_in(text: str, heading: str) -> List[Tuple[str, str, str]]:
    """Return the three-column table rows under ``heading``, header row excluded.

    The section runs from its heading to the next one at the same level or above, so a table further down the
    page cannot be read into this one.

    Args:
        text: The whole page.
        heading: The markdown heading the wanted table sits under.

    Returns:
        ``(backend, old spelling, current spelling)`` per row, in the order the page lists them.
    """
    lines = text.splitlines()
    try:
        start = lines.index(heading) + 1
    except ValueError:
        return []
    rows: List[Tuple[str, str, str]] = []
    for line in lines[start:]:
        if line.startswith("## "):
            break
        cells = _cells(line)
        if len(cells) != 3 or cells == ["Backend", "Old spelling", "Current spelling"]:
            continue
        rows.append((cells[0], cells[1], cells[2]))
    return rows


@pytest.fixture(scope="module")
def documented() -> List[Tuple[str, str, str]]:
    """The method renames the page teaches, read out of its prose."""
    return _rows_in(PAGE.read_text(encoding="utf-8"), METHOD_HEADING)


@pytest.fixture(scope="module")
def declared() -> Dict[Tuple[str, str], str]:
    """The method renames the package keeps, read out of :data:`ALIASES` as ``{(backend, old): new}``."""
    return {
        (backend, old): new
        for backend, renames in ALIASES.items()
        for old, new in renames.items()
    }


class TestThePageAndTheCodeAgree:
    """Drift between the page and :data:`ALIASES`, checked in both directions and on all three columns."""

    def test_every_alias_the_package_keeps_is_on_the_page(self, documented, declared):
        """An alias the page does not mention is a warning a reader is sent here and cannot resolve.

        Test scenario:
            This is the direction that failed: order 27a moved six renames into `ALIASES` and added
            `set_extent`, and the page went on describing "one method". A reader who wrote `imshow` got a
            warning naming `field`, came here, and found nothing about it.
        """
        listed = {(backend, old) for backend, old, _ in documented}
        missing = sorted(key for key in declared if key not in listed)
        assert missing == [], (
            f"these aliases in ALIASES have no row in {PAGE.name}: {missing}"
        )

    def test_every_row_on_the_page_is_an_alias_the_package_keeps(
        self, documented, declared
    ):
        """A row for a rename the package does not forward teaches a call that raises rather than warns.

        Test scenario:
            The page outliving a rename is the other way it rots — a spelling retired, the alias dropped at
            the release it was kept for, and the row left behind. That row reads as a live alias.
        """
        invented = sorted(
            (backend, old)
            for backend, old, _ in documented
            if (backend, old) not in declared
        )
        assert invented == [], (
            f"these rows in {PAGE.name} name no alias in ALIASES: {invented}"
        )

    def test_every_row_names_the_replacement_the_package_forwards_to(
        self, documented, declared
    ):
        """The third column is the answer a reader came for, so it is compared and not merely counted.

        Test scenario:
            Both checks above are keyed on the *old* spelling, so a row that names the right deprecated name
            and the wrong replacement — `imshow` to `points`, say — satisfies each of them. Only reading the
            current-spelling cell catches it.
        """
        wrong = sorted(
            f"{backend} {old}: page says {new}, ALIASES says {declared[(backend, old)]}"
            for backend, old, new in documented
            if (backend, old) in declared and new != declared[(backend, old)]
        )
        assert wrong == [], f"{PAGE.name} names the wrong replacement: {wrong}"

    def test_the_page_names_every_backend_that_has_an_alias(self, documented, declared):
        """The Backend column is spelled the way `quickmap` and `ALIASES` spell it, not prettified.

        Test scenario:
            The page used to write the 3-D tier "3-D", which is what a reader says and not what
            `alias_table("3d")` takes. A label nothing can be looked up by makes the table unmatchable
            against the code — and this is the check that fails if a fifth backend arrives with aliases and
            only the first table is extended.
        """
        labelled = {backend for backend, _, _ in documented}
        unlabelled = sorted({backend for backend, _ in declared} - labelled)
        assert unlabelled == [], (
            f"{PAGE.name} has no row for these backends: {unlabelled}"
        )


class TestReadingTheTable:
    """The parser, held to its own cases — a guard that reads nothing would pass half the checks above."""

    #: A table shaped like the page's, with the one row shape the parser must keep and the two it must drop.
    SAMPLE = "\n".join(
        [
            "## The renamed methods",
            "",
            "| Backend | Old spelling | Current spelling |",
            "|---|---|---|",
            "| web | `add_raster` | `field` |",
            "",
            "## Something else",
            "",
            "| web | `not_a_rename` | `nonsense` |",
        ]
    )

    def test_a_row_is_read_with_its_backticks_stripped(self):
        """The cells reach the comparison as bare identifiers, the way `ALIASES` spells them.

        Test scenario:
            The page writes a method name as a code span. A parser that kept the backticks would report every
            single alias as missing *and* every single row as invented, which is a failure nobody would read
            as "the parser is wrong".
        """
        read = _rows_in(self.SAMPLE, "## The renamed methods")
        assert read == [("web", "add_raster", "field")], read

    def test_the_header_and_its_separator_are_not_rows(self):
        """Both are table rows by shape, and neither names a rename.

        Test scenario:
            ``|---|---|---|`` has three cells and would be read as a rename of ``---`` to ``---``; the header
            would be read as a rename of "Old spelling". Either one lands in the "invented" list and buries
            the real finding.
        """
        read = _rows_in(self.SAMPLE, "## The renamed methods")
        assert "---" not in {cell for row in read for cell in row}, read

    def test_the_next_heading_ends_the_section(self):
        """A table elsewhere on the page belongs to its own section.

        Test scenario:
            The page carries the keyword table and the shared-defaults table below this one. Reading past the
            heading would pull their rows in as method renames.
        """
        read = _rows_in(self.SAMPLE, "## The renamed methods")
        assert [old for _, old, _ in read] == ["add_raster"], read

    def test_a_heading_the_page_does_not_have_reads_as_no_rows(self):
        """A restructure that renames the heading must fail the drift checks, not crash the module.

        Test scenario:
            An absent section is indistinguishable from an empty one, and both mean the guard has nothing to
            compare — which `test_every_alias_the_package_keeps_is_on_the_page` reports as every alias
            missing. An exception here would instead read as a broken test.
        """
        read = _rows_in(self.SAMPLE, "## A heading nobody wrote")
        assert read == [], read


class TestThePageIsWhereTheDocsSendReaders:
    """The page's own path, since the guard above would pass just as happily against a file nobody links."""

    def test_the_page_exists_under_the_name_the_nav_uses(self):
        """The rename from ``renamed-keywords.md`` has to reach `mkdocs.yml`, or the nav entry 404s.

        Test scenario:
            `mkdocs.yml` lists the page by path. Renaming the file and not the entry leaves a nav link to
            nothing, which a docs build reports only under `--strict`.
        """
        nav = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        entry = f"reference/{PAGE.name}"
        assert entry in nav, f"mkdocs.yml does not list {entry}"

    def test_the_old_page_name_is_gone_from_the_tree(self):
        """A stale link would resolve for as long as both names existed, and then stop.

        Test scenario:
            The page was ``reference/renamed-keywords.md`` and two other pages linked it by that name. A
            leftover reference is a dead link the moment the old file goes.
        """
        stale = sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in list((REPO_ROOT / "docs").rglob("*.md"))
            + [REPO_ROOT / "mkdocs.yml", REPO_ROOT / "README.md"]
            if "renamed-keywords" in path.read_text(encoding="utf-8")
        )
        assert stale == [], f"these still point at the old page name: {stale}"
