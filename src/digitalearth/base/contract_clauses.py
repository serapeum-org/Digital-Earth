"""The numbered cross-backend contract — every clause stated once, so no tier has to restate it.

The four rendering tiers answer to a numbered contract. ``C1``, ``C7``, ``C13`` and the rest are cited
throughout ``src/`` and ``tests/`` as though a canonical list existed; until this module, none did. Each
clause lived only as a docstring bullet in whichever tier's contract-test file happened to mention it, which
meant a clause had no home: nothing said what ``C7`` *was*, nothing listed where it was asserted, and nothing
checked that the four tiers were still saying the same thing about it.

The cost was measured while settling #320. Amending one clause — C7 — took edits at seven statement sites
across ``base/``, ``web/`` and four tiers' tests, two of which were found by grepping rather than by looking
anything up. The failure mode of that exercise is a stale statement left behind on one tier, and nothing
would have caught it.

**How to cite a clause, and how not to.** Write the bare number — ``C7``, or ``contract C7`` — and do not
restate the clause beside it; the whole point is that the wording lives here. ``tests/test_contract_clauses.py``
scans for that spelling and holds it both ways: a number nobody wrote down fails, and a clause no test names
fails.

That makes the spelling load bearing, because three other things in this repository used to be written the
same way and one of them is generated continuously:

* **A review-round finding.** The review passes number findings by severity and rank, so a round's first
  critical was also written ``C1``. Write these ``R-C1`` / ``R-H2`` / ``R-M9`` — the leading ``R-`` is what
  keeps them out of the scan, and findings already spelled ``H<n>`` or ``M<n>`` never collided.
* **Python's method resolution order**, once written "the C3 linearisation". Name it in words instead.
* **matplotlib's colour cycle**, ``colors="C0"``. A quoted string, so it never reads as prose.

A citation that ignores this is not merely untidy: a review finding written ``C1`` is indistinguishable from
clause 1 and would be counted as a citation of it, which is the drift this module exists to stop.

So the clause text lives here, once, and the tiers **cite** it: a tier's contract test names the number and
keeps its own scenario prose, rather than paraphrasing the rule in a fifth wording.
:mod:`tests.test_contract_clauses` is the drift guard over that arrangement — every cited number must be
defined here, and every clause defined here must be pinned by some test.

**This is not** :mod:`digitalearth.base.contract`, which is the *method vocabulary* every tier answers to
(``field``, ``points``, ``save`` and their keywords). That module says which names exist; this one says which
behaviours they are held to. They are deliberately separate files so neither grows the other's job.

**Three namespaces spell an identifier ``C<n>``** and only one of them is this contract. The review rounds
number their findings by severity and rank (``C1``/``H2``/``M9``), and Python's method-resolution order is
computed by the *C3 linearisation*. The guard reads a line naming a review or a linearisation as belonging to
those namespaces, not this one; a new citation of this contract should avoid both words on the same line.

Nothing here imports a renderer, or anything outside the standard library, so ``base/`` stays engine-neutral
and reading a clause costs no engine import.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

__all__ = ["CLAUSES", "Clause", "cite", "clause"]


@dataclass(frozen=True)
class Clause:
    """One numbered clause of the cross-backend contract.

    Attributes:
        number: The clause's number — the ``7`` that callers cite as ``C7``.
        title: A short handle for the clause, for a listing or a failure message that has no room for the
            rule itself.
        rule: The one wording of the clause. Every tier is held to this text; none of them restates it.
        issues: The issues the clause was settled under, in the order they settled it. Empty where the clause
            came out of the Wave-0 batch (#246–#257) without a number of its own recorded beside it.

    Examples:
        - A clause states itself, so a citation never has to:
            ```python
            >>> from digitalearth.base.contract_clauses import clause
            >>> c7 = clause(7)
            >>> c7.name, c7.title
            ('C7', 'a layer with nothing to draw is skipped')

            ```
    """

    number: int
    title: str
    rule: str
    issues: tuple[int, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        """The clause as it is cited in prose.

        Returns:
            The number with its ``C`` prefix, e.g. ``"C7"``.
        """
        return f"C{self.number}"

    def __str__(self) -> str:
        """Render the clause the way a failure message wants it.

        Returns:
            The citation and the rule, joined by an em dash.
        """
        return f"{self.name} — {self.rule}"


def _clauses() -> tuple[Clause, ...]:
    """Build the clause table.

    A function rather than a literal so the wordings can carry their own indentation without fighting the
    formatter, and so :data:`CLAUSES` can be keyed by each clause's own number rather than by a key somebody
    has to keep in step with it.

    Returns:
        Every clause of the contract, in number order.
    """
    return (
        Clause(
            number=1,
            title="save returns the path it wrote",
            rule=(
                "Every tier's `save` hands back the `pathlib.Path` it wrote, so a caller can stat, read or "
                "re-open the file without re-wrapping the return value."
            ),
            issues=(248,),
        ),
        Clause(
            number=2,
            title="the frame rate is fps",
            rule=(
                "The frame rate is `fps: float` on every animation entry point, over one shared default of "
                "3.0; the spellings it replaced survive as deprecated aliases."
            ),
            issues=(256,),
        ),
        Clause(
            number=3,
            title="size is a marker's size",
            rule=(
                "`size` is the visual size of a marker and nothing else: text size is `text_size`, the "
                "column that varies a marker is `column`, and the spellings they replaced survive as "
                "deprecated aliases."
            ),
            issues=(251,),
        ),
        Clause(
            number=4,
            title="scheme and k classify the same way everywhere",
            rule=(
                "A classifiable layer takes `scheme: str | None = None` and `k: int = 5`, `scheme=None` is a "
                "continuous ramp on every tier, and one `scheme`/`k` pair cuts the same classes whichever "
                "tier draws it."
            ),
            issues=(246,),
        ),
        Clause(
            number=5,
            title="cmap=None resolves through auto_style",
            rule=(
                "`cmap=None` on a raster builder resolves through `auto_style`, never through a colormap "
                "literal written into the signature."
            ),
            issues=(249,),
        ),
        Clause(
            number=6,
            title="auto_style's levels and units are consumed",
            rule=(
                "`auto_style` carries more than a colormap: its `levels` and `units` are consumed too, and a "
                "caller who names either gets exactly what they named."
            ),
            issues=(230,),
        ),
        Clause(
            number=7,
            title="a layer with nothing to draw is skipped",
            rule=(
                "A layer with nothing to draw — an off-limb raster, an empty geometry set, a custom layer "
                "whose object this process does not hold — is skipped with a warning, and raises only under "
                "`strict=True`. A kind the tier does not draw at all is not this clause's case: it is a "
                "caller error, and `drawer_for` refuses it by name in every mode, lenient or strict."
            ),
            issues=(257, 320),
        ),
        Clause(
            number=8,
            title="one big_data_threshold, two reaches",
            rule=(
                "One `big_data_threshold` over one shared default: an attribute on the map and a per-call "
                "override on the builder, with one deprecated alias."
            ),
            issues=(250,),
        ),
        Clause(
            number=9,
            title="one shared basemap default",
            rule=(
                "The default basemap provider is one shared constant read from `digitalearth.base.basemaps`, "
                "not a literal spelled per tier."
            ),
            issues=(247,),
        ),
        Clause(
            number=10,
            title="a keyed provider's coverage reaches the engine",
            rule=(
                "Where a keyed basemap provider declares the area it covers, those bounds are declared to "
                "the engine rather than dropped."
            ),
            issues=(233,),
        ),
        Clause(
            number=11,
            title="quickmap refuses what a backend cannot honour",
            rule=(
                "`quickmap` refuses by name a parameter the chosen backend cannot honour, and refuses it "
                "before that backend is imported, rather than accepting it and dropping it in silence."
            ),
        ),
        Clause(
            number=12,
            title="six named Natural-Earth layers",
            rule=(
                "The Natural-Earth reference layers are six named methods — `coastlines`, `borders`, `land`, "
                "`ocean`, `lakes`, `rivers` — over whatever flags a tier keeps underneath them."
            ),
            issues=(253,),
        ),
        Clause(
            number=13,
            title="every tier declares its display CRS",
            rule=(
                "Every tier declares the display CRS it places its layers in, as something a caller reads "
                "off the scene rather than a convention they have to infer, and refuses one it cannot draw "
                "in."
            ),
            issues=(291,),
        ),
        Clause(
            number=14,
            title="animation state is declared, not conjured",
            rule=(
                "The animation state a scene carries is declared in `__init__`, not conjured on first use by "
                "`getattr`, so the attribute is visible on the class that owns it."
            ),
        ),
    )


#: Every clause of the cross-backend contract, by its number. This is the single statement of each: a tier
#: cites the number and keeps its own scenario prose, and `tests/test_contract_clauses.py` fails if a cited
#: number is missing here or a clause here is pinned by no test.
CLAUSES: Mapping[int, Clause] = MappingProxyType({c.number: c for c in _clauses()})


def clause(number: int) -> Clause:
    """Return the clause with this number.

    Args:
        number: The clause number, as cited — ``7`` for ``C7``.

    Returns:
        The clause.

    Raises:
        KeyError: If no clause has that number. The message lists the numbers that do exist, because the way
            this goes wrong is a citation of a number nobody ever wrote down.

    Examples:
        - A number that was never a clause says so:
            ```python
            >>> from digitalearth.base.contract_clauses import clause
            >>> clause(99)
            Traceback (most recent call last):
            ...
            KeyError: 'C99 is not a clause of the contract; it states C1-C14'

            ```
    """
    try:
        return CLAUSES[number]
    except KeyError:
        known = sorted(CLAUSES)
        raise KeyError(
            f"C{number} is not a clause of the contract; it states C{known[0]}-C{known[-1]}"
        ) from None


def cite(number: int) -> str:
    """Return a clause as a failure message quotes it.

    Args:
        number: The clause number, as cited.

    Returns:
        The citation and the rule, so an assertion that fails says which clause it was holding the tier to.

    Examples:
        - The wording comes from one place, including in a message:
            ```python
            >>> from digitalearth.base.contract_clauses import cite
            >>> cite(5).startswith("C5 — `cmap=None` on a raster builder")
            True

            ```
    """
    return str(clause(number))
