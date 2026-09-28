"""The contract every backend answers to — as data, so a test can hold a facade against it.

Four tiers grew four vocabularies. A raster field was `imshow` on static, `image` on interactive, `add_raster`
on web and `terrain` on 3-D; a colour key was a builder on two tiers and a toggle on a third; `render` returned
the engine's object on two and `None` on one; and eight names meant two different things depending on which
facade you were holding. Every one of those was defensible where it grew and indefensible across four.

This module is the agreement that ends it, written down rather than described: :data:`CORE` is the vocabulary
every tier answers to where it can draw the thing at all, :data:`TIER2` the names that must agree wherever they
appear, and :data:`PENDING` what a tier has not built yet and which roadmap order builds it. A contract test
per tier reads this and compares; nothing here imports a renderer, so the comparison costs no engine.

**A spelling never means two things.** Where a tier called a contract method something else, that spelling was
deleted rather than kept working: nothing here is released, so there is no caller to keep a promise to.

**Nor does a call shape.** A name and its keyword set are not enough to call it: `set_bounds` accepted
`padding=` on every tier, satisfied every contract test, and still read the same four bare numbers as two
different rectangles — matplotlib's `[xmin, xmax, ymin, ymax]` on one tier and bbox `(west, south, east,
north)` on the next — under a first parameter named `bbox` on one and `bounds` on the rest (#344). So
:class:`Method` declares the shape as well: `first_argument` and `sequence_order`.
"""

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import FrozenSet, Iterable, Mapping, Optional, Tuple

__all__ = [
    "CORE",
    "PENDING",
    "ROADMAP_ORDERS",
    "TIER2",
    "Method",
    "core_method",
    "orders_named_in",
    "pending_for",
    "roadmap_order",
]


@dataclass(frozen=True)
class Method:
    """One method of the contract: its name, what it takes, and what it hands back.

    **A keyword set is not a call shape**, and the difference cost a breaking bug (#344). This class recorded
    the keywords a name accepts and nothing else about how it is called — not the spelling of its first
    positional parameter, not the order it reads a sequence in. So `set_bounds` read four bare numbers as
    matplotlib's `[xmin, xmax, ymin, ymax]` on the static tier and as bbox `(west, south, east, north)` on the
    other two, under a parameter named `bbox` on one and `bounds` on the rest: one call framed two different
    rectangles, a keyword call ported in neither direction, and every contract test passed, because both tiers
    accepted `padding=` and that was all the declaration said. :attr:`first_argument` and
    :attr:`sequence_order` are what say the rest.

    Both are optional, and that is deliberate rather than convenient: a required field would have to be given
    at every construction site in this module before the file imported at all, and the contract cannot honestly
    fix a spelling its tiers do not agree on yet. `None` therefore means *undeclared* — see the note above
    :data:`CORE` for the names in that position and what settles them — and a declaration is added when the
    tiers agree, not by choosing a winner here and failing the losers.

    Attributes:
        name: The canonical spelling, the same on every tier.
        doc: One line saying what it does — the text a support matrix shows beside it.
        keywords: The canonical keywords it accepts beyond its data argument. A tier may take more; it may not
            spell these differently.
        returns: What a caller gets: `"self"` for a chainable builder, `"engine"` for the renderer's own
            object, `"path"` for a written file, `"value"` for a description.
        builds_in: The roadmap order that builds it, for a name the contract declares before any tier has it.
            `None` for a name that is already drawn somewhere.
        first_argument: The canonical spelling of the **first positional parameter** — the data a builder
            draws, the id a layer method acts on, the region a framing call frames. A tier may name later
            parameters as it likes; it may not spell this one differently, or the same positional call means
            two things and the same keyword call means neither. `None` where the contract does not fix it: a
            name that takes nothing positionally, or one whose tiers still disagree.
        sequence_order: Where that argument may be a bare sequence, the components it holds **in the order
            they are read** — `("west", "south", "east", "north")` for a framing rectangle. The tuple of names
            rather than a flag, because it is the ordering *and* its documentation: a reader needs no second
            lookup, and a fourth tier is told what to implement. `None` for an argument that is not a
            sequence, or a sequence whose ordering the contract has not fixed.

    Examples:
        - The contract says what a name means before any tier is consulted:
            ```python
            >>> from digitalearth.base.contract import core_method
            >>> field = core_method("field")
            >>> field.returns, "cmap" in field.keywords
            ('self', True)

            ```
        - And how it is called, not only what it accepts:
            ```python
            >>> from digitalearth.base.contract import core_method
            >>> framing = core_method("set_bounds")
            >>> framing.first_argument, framing.sequence_order
            ('bounds', ('west', 'south', 'east', 'north'))

            ```
        - Nothing is waiting: every Core name is drawn on at least one tier, so no `builds_in` is set. It
          stays a field because the *next* name the contract declares ahead of its tiers needs one:
            ```python
            >>> from digitalearth.base.contract import CORE
            >>> [method.name for method in CORE if method.builds_in]
            []

            ```
    """

    name: str
    doc: str
    keywords: FrozenSet[str] = field(default_factory=frozenset)
    returns: str = "self"
    builds_in: Optional[str] = None
    first_argument: Optional[str] = None
    sequence_order: Optional[Tuple[str, ...]] = None


# `_LAYER_MANAGEMENT_ORDER`, and the three reasons interpolated into it — `_PENDING_LAYERS`,
# `_PENDING_LIVE_LAYERS` and `_PENDING_IDENTITY` — stood here while nine `PENDING` rows and three `builds_in`
# citations named order 23. That order has built them: `add_layer`, `get_layer`, `remove_layer`,
# `set_visible`, `move_layer` and `replace_layer` are on all four facades, each over its tier's own renderer.
# The constants go with the rows rather than staying as a spelling nobody interpolates, and the entry in
# `ROADMAP_ORDERS` goes with them — a vendored order nothing cites is the next thing to go stale, which
# `tests/base/test_contract.py::TestAPendingReasonPointsAtLiveWork` refuses.

#: Where a tier adopts a Core spelling it has already agreed to. Order 27a is U-3's remainder — the
#: "canonical names" half of the contract (DE-26, folded into U-3), which PR #304 froze without any tier
#: adopting.
_RENAME_ORDER = "order 27a"

#: `_GUIDES_ORDER = "order 24"` stood here — where a colour key stops being the most recent classification
#: and becomes a guide on its own layer's encoding. That order has built it on all four tiers: `colorbar` and
#: `legend` record a `Guide` through `Symbology.with_guide` onto the layer's colour encoding, so the key moves
#: with its layer, goes away with it, and travels in the figure. The two 3-D `PENDING` rows and the four
#: `KEYWORD_SHORTFALLS` rows that cited it are all off, so the constant and its `ROADMAP_ORDERS` entry go with
#: them — `TestAPendingReasonPointsAtLiveWork` refuses a declared order no reason points at, which is what
#: makes finishing an order include deleting its citation rather than leaving a pointer nobody follows.

#: How a reason points at the roadmap. The letter is part of the number — order 27a is a step of its own that
#: sits after 27, not a variant of it — so a pattern that stopped at the digits would read two orders as one.
_ORDER_REFERENCE = re.compile(r"\border \d+[a-z]?")

#: The letters an order's number may be suffixed with — the `a` of `order 27a`. Named rather than inlined
#: because :func:`_counted_order` strips them from the right with `str.rstrip`, which takes the characters
#: themselves rather than a pattern.
_ORDER_SUFFIX_LETTERS = "abcdefghijklmnopqrstuvwxyz"

#: Every roadmap order a user-facing reason may name, with one line saying what it builds.
#:
#: **The roadmap is not in this repository** — it is the maintainer's planning document, and a reason naming
#: `order 27a` is a pointer into it. Nothing checked that the order on the other end existed: a reason reading
#: "order 99" satisfied every guard, because the guards only ever asked whether the *form* was an order rather
#: than a wave (review R-L3). Measured before this table: a `PENDING` row rewritten to name order 99 passed all
#: three checks in `TestAPendingReasonPointsAtLiveWork`.
#:
#: A test cannot read the document — it is not here to read, and a suite that reached outside the repository
#: for it would fail for everyone who does not have it. So what is vendored is the part a citation needs: the
#: orders this contract points at, and what each one builds. That is enough for a reader to know what they are
#: being promised without going and finding the plan, and enough for the guard to refuse a number nobody wrote
#: down. It is the bargain :data:`~tests.open_issues.KNOWN_OPEN_ISSUES` strikes for issue numbers, for the same
#: reason and with the same cost: extending it is a deliberate act. **Taking a row out is the same act in
#: reverse**, and is what finishing an order looks like here: the framing order held auto-framing and the
#: camera round-trip, both of which are built, so nothing cites it any more and its row is gone.
#: `TestAPendingReasonPointsAtLiveWork` is what makes that a requirement rather than tidying — it refuses a
#: declared order no reason points at, in the same breath as a reason pointing at an order nobody declared.
#:
#: The keys are the constants above rather than repeated strings, so a citation and its entry cannot disagree.
ROADMAP_ORDERS: Mapping[str, str] = MappingProxyType(
    {
        _RENAME_ORDER: (
            "the Core contract's remainder — the renames each tier agreed to and had not adopted, plus the "
            "divergences the frozen contract did not settle"
        ),
    }
)


# **Where `first_argument` is left undeclared, and why it is not simply chosen here.** Measured across the
# four facades, eight declared names take a differently-spelled first positional parameter depending on which
# one you hold: `field` (`data` on web and interactive, `dataset` on static), `add_layer` (`layer`, `element`,
# `artist`), `colorbar` (`layer_id`, `show`, `layer`), `legend` (nothing positional, `show`, `colors`),
# `contours` (`dataset`, `data`, `dataset`), `set_title` (`heading`, `title`), `tiles` (`url`, `provider`) and
# `basemap` (`provider`, `source`). Declaring one side of each would fail the others for a divergence this
# module did not cause and cannot fix — the rename belongs to order 27a, which is named for exactly this
# remainder, and the last four already have a `KEYWORD_SHORTFALLS` row tracking the keyword half of the same
# gap. So the shape is declared where the tiers already agree, which is what a guard can hold them to today,
# and the divergences are recorded here rather than settled by fiat. A name gains its declaration in the
# change that makes the tiers agree.

#: The Core vocabulary: what every tier answers to, where it can draw the thing at all. A tier that cannot —
#: a 3-D scene has no extent to frame — declares that in its `Capabilities` (#294) rather than growing a method
#: that raises.
CORE: Tuple[Method, ...] = (
    Method(
        "field",
        "Draw a raster band as a coloured field.",
        frozenset({"band", "cmap", "limits", "opacity", "name", "visible"}),
    ),
    Method(
        "points",
        "Draw point features.",
        frozenset(
            {"column", "scheme", "k", "cmap", "size", "opacity", "name", "visible"}
        ),
        first_argument="features",
    ),
    Method(
        "lines",
        "Draw line features.",
        frozenset(
            {"column", "scheme", "k", "cmap", "width", "opacity", "name", "visible"}
        ),
        first_argument="features",
    ),
    Method(
        "polygons",
        "Draw polygon features.",
        frozenset({"column", "scheme", "k", "cmap", "opacity", "name", "visible"}),
        first_argument="features",
    ),
    Method(
        "choropleth",
        "Draw polygons coloured by a column.",
        frozenset({"column", "scheme", "k", "cmap", "opacity", "name", "visible"}),
        first_argument="features",
    ),
    Method(
        "colorbar",
        "Show the continuous colour key of a layer.",
        frozenset({"layer_id", "label", "visible"}),
    ),
    Method(
        "legend",
        "Show the keyed colour list of a layer.",
        frozenset({"layer_id", "title", "labels", "visible"}),
    ),
    Method(
        "add_layer",
        "Add an object the caller built themselves, as a custom layer.",
        frozenset({"name", "band"}),
    ),
    Method(
        "get_layer",
        "Return the description of one layer, by id.",
        frozenset(),
        returns="value",
        first_argument="layer_id",
    ),
    Method(
        "remove_layer",
        "Take a layer off the figure, by id.",
        first_argument="layer_id",
    ),
    Method(
        "set_visible",
        "Show or hide a layer, by id.",
        frozenset({"visible"}),
        first_argument="layer_id",
    ),
    Method(
        "move_layer",
        "Move a layer in draw order, by id.",
        frozenset({"index"}),
        first_argument="layer_id",
    ),
    Method(
        "replace_layer",
        (
            "Swap a layer's description for another, keeping its id and its place — except where the new "
            "description changes the layer's band, which re-places it at the top of the band it now "
            "belongs to, as adding it there would."
        ),
        frozenset(),
        first_argument="layer",
    ),
    Method(
        "layer_ids",
        "The ids of the figure's layers, in draw order.",
        frozenset(),
        returns="value",
    ),
    Method(
        "set_bounds",
        (
            "Frame the figure on a region — bounds=Bounds, or (west, south, east, north) in the display "
            "CRS; None fits the data."
        ),
        frozenset({"padding"}),
        first_argument="bounds",
        # The one sequence argument the contract declares, and the reason the field exists (#344): these four
        # numbers were read in two orders across the 2-D tiers, so one call framed two rectangles. It is the
        # order `Bounds` itself holds — `Bounds.as_bbox` writes it and `from_bbox` reads it — which is what
        # `tests/base/test_contract.py` holds the declaration to. matplotlib's `[xmin, xmax, ymin, ymax]` is
        # reachable through `Bounds.as_mpl`/`from_mpl` and nowhere else.
        sequence_order=("west", "south", "east", "north"),
    ),
    Method(
        "render",
        "Build and return the renderer's own object.",
        frozenset(),
        returns="engine",
    ),
    Method(
        "save",
        "Write the figure to a file.",
        frozenset(),
        returns="path",
        first_argument="path",
    ),
    Method("show", "Display the figure.", frozenset(), returns="engine"),
)

#: The names that are not Core — not every tier has them — but that must mean one thing wherever they appear.
#: Each was a live collision until this contract: `tiles` took a URL on one tier and a provider name on
#: another, `contours` filled on one and not on another, `text` took a CRS on two tiers and not on the third.
TIER2: Tuple[Method, ...] = (
    Method(
        "text",
        "Place a string at a coordinate.",
        frozenset({"s", "crs", "name"}),
        first_argument="lon",
    ),
    Method(
        "labels",
        "Label features from a column.",
        frozenset({"column", "crs", "name"}),
        first_argument="features",
    ),
    Method(
        "graticule",
        "Draw meridians and parallels.",
        frozenset({"lon_step", "lat_step", "spacing"}),
        first_argument="lon_step",
    ),
    Method("tiles", "Draw raster tiles from a URL template.", frozenset({"url"})),
    Method(
        "basemap", "Draw a named basemap beneath the data.", frozenset({"provider"})
    ),
    Method(
        "contours",
        "Trace iso-value lines, or fill between them.",
        frozenset({"levels", "interval", "filled"}),
    ),
    Method(
        "layer_control",
        "Offer the viewer a switch per layer.",
        frozenset({"layers", "position"}),
    ),
    Method("set_title", "Give the figure a heading.", frozenset({"subtitle"})),
    Method(
        "save_animation",
        "Write a sequence of frames to a file.",
        frozenset({"fps"}),
        returns="path",
        first_argument="path",
    ),
    Method(
        "projection",
        "Draw in another projection, by name.",
        frozenset(),
        first_argument="name",
    ),
)

#: What a tier has not built yet, as `{backend: {name: why}}`. This is the honest half of the contract: a name
#: absent because the tier cannot draw it at all reads differently from one absent because nobody has written
#: it, and only the tier can say which. A contract test holds each facade against `CORE` minus its pending list.
#:
#: **What a reason may point at.** A roadmap **order**, or an issue that is open — and nothing else, because
#: these strings are the one place in the package where a tracker reference is shown to a *user* rather than
#: left as provenance in a comment. A closed issue answers "when?" with "already done" (#319), and a wave
#: number rots the moment a wave is inserted ahead of it and the rest renumber (#317). Where neither exists,
#: the reason says the work is **unscheduled** rather than naming a plan that does not hold it — three of the
#: 3-D rows are in that position, and saying so is the whole point of the table.
PENDING: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "web": MappingProxyType(
            {
                # `set_visible`, `move_layer` and `replace_layer` were listed here against order 23, which
                # has now built all three: the tier answers to each over its own renderer, through
                # `WebMapBase._change`, and the queue the page is built from follows. The table is empty for
                # this tier as a result, and stays here so the next unbuilt name has a home.
            }
        ),
        "3d": MappingProxyType(
            {
                "field": "a raster becomes terrain, a volume or a globe here, each its own builder",
                "points": (
                    "positioned 3-D points are point_cloud() here, which takes a z per point; a flat points "
                    "builder is unscheduled"
                ),
                "add_layer": (
                    "a caller's own object is a PyVista mesh or volume, so it is added with add_mesh() or "
                    "add_volume(), each of which records a custom:pyvista layer"
                ),
                "lines": "line features in three dimensions — #201",
                "polygons": "polygons are drawn extruded here; a flat fill is unscheduled",
                "choropleth": "a classified fill follows polygons, and is unscheduled with them",
                # `colorbar` ("the scalar bar is PyVista's, and becomes a guide on the encoding") and
                # `legend` ("a keyed list beside a scene") were listed here against order 24, which has now
                # built both: `digitalearth.three_d.guides` records a `Guide` on the layer's own colour
                # encoding and draws from it — PyVista's scalar bar for a ramp, its keyed legend for classes,
                # derived from the layer's `Scale` — so the key follows the layer it explains. Both are
                # declared features of the tier as well (`three_d/capabilities.py`).
                "set_bounds": "a scene is framed by its camera, not by an extent (see Capabilities.absent)",
            }
        ),
        "interactive": MappingProxyType(
            {
                # `field`, `lines` and `add_layer` were listed here, each against the tier's own older
                # spelling. All three are adopted at order 27a, so none is pending any more.
                # `get_layer`, `remove_layer`, `set_visible`, `move_layer` and `replace_layer` were listed
                # here against order 23, which has now built all five: the tier answers to each over its own
                # renderer, through `InteractiveMapBase._change`.
                # `set_bounds` was listed here, "no framing method here under any spelling", against the
                # framing order — which has now built it: the tier frames on a region or on its own data,
                # takes `padding`, and reports the region through its viewport. The table is empty for this
                # tier as a result, and stays here so the next unbuilt name has a home.
            }
        ),
        "matplotlib": MappingProxyType(
            {
                # `field`, `points` and `polygons` were listed here, each against the tier's own older
                # spelling. All three are adopted at order 27a, so none is pending any more.
                "lines": "line features on the static tier — #226",
                # `add_layer`, `get_layer`, `remove_layer`, `set_visible`, `move_layer` and `replace_layer`
                # were listed here against order 23, which has now built all six: the tier answers to each
                # over its own renderer, through `Scene._change`.
                # `set_bounds` was listed here, "framed by set_extent(bbox) here, which neither pads nor
                # fits the data". Orders 27a and 26 between them built all of it: the tier answers to
                # `set_bounds`, returns `self`, takes `padding`, and fits the figure to its own data on
                # `None` (`static/maps/projection.py`'s `_fitted_box`). The `KEYWORD_SHORTFALLS` row that
                # recorded the missing half came off with it, so nothing is owed on this name — this comment
                # went on citing that record after it was gone, and claiming the two capabilities were still
                # missing, for a release (round 3, L3).
                # What was owing after that was the *argument*, not a capability: the tier read matplotlib's
                # `[xmin, xmax, ymin, ymax]` under the name `bbox` while the other two read
                # `bounds=(west, south, east, north)`, so one call framed two rectangles. That is fixed
                # rather than tracked — a Core name meaning two things is a broken contract, and there is no
                # order to wait for. What the declaration could not say at the time, it now says:
                # `set_bounds` declares `first_argument` and `sequence_order`, so the next tier is held to
                # both without a probe having to remember (#344).
            }
        ),
    }
)


def core_method(name: str) -> Method:
    """Return one Core method's declaration.

    Args:
        name: The canonical spelling.

    Returns:
        Its :class:`Method`.

    Raises:
        KeyError: for a name the Core does not declare, naming the ones it does.

    Examples:
        - What a name means, before any tier is consulted:
            ```python
            >>> from digitalearth.base.contract import core_method
            >>> core_method("save").returns
            'path'

            ```
        - A misspelling is answered with the vocabulary:
            ```python
            >>> from digitalearth.base.contract import core_method
            >>> core_method("add_raster")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            KeyError: "'add_raster' is not a Core method; the Core is ['add_layer', ...]"

            ```
    """
    for method in CORE:
        if method.name == name:
            return method
    raise KeyError(
        f"{name!r} is not a Core method; the Core is {sorted(entry.name for entry in CORE)}"
    )


def pending_for(backend: str) -> Mapping[str, str]:
    """Return the Core names one tier has not built, and why.

    Args:
        backend: The tier, as `quickmap(backend=...)` spells it.

    Returns:
        `{name: reason}`, empty for a tier that answers to every Core name.

    Examples:
        - A 3-D scene has no extent to frame, and says so rather than raising from a method:
            ```python
            >>> from digitalearth.base.contract import pending_for
            >>> pending_for("3d")["set_bounds"]
            'a scene is framed by its camera, not by an extent (see Capabilities.absent)'

            ```
    """
    return PENDING.get(backend, MappingProxyType({}))


def orders_named_in(reason: str) -> Tuple[str, ...]:
    """Return every roadmap order a reason points at, in the order it names them.

    This is the one reading of "an order reference", so a guard over a reason and the reason itself cannot
    disagree about where one ends. In particular the letter belongs to the number: `order 27a` is a step of
    its own, and a pattern stopping at the digits would read it as order 27.

    Args:
        reason: A user-facing string — a `PENDING` reason, or a keyword shortfall's.

    Returns:
        The orders named, each spelled as :data:`ROADMAP_ORDERS` keys it. Empty for a reason that names none,
        which is allowed: an open issue and the word "unscheduled" are the other two honest answers.

    Examples:
        - A reason that names one. Written out rather than read off a live `PENDING` row, and deliberately
          naming an order that is **gone**: this parses a string and does not ask whether the order is still
          declared, which is what lets a reason be read before the table is consulted. An example pinned to
          whichever row happens to cite a live order breaks every time one lands — order 24 built the two 3-D
          rows that used to be quoted here, and took its own entry off `ROADMAP_ORDERS` with them:
            ```python
            >>> from digitalearth.base.contract import orders_named_in
            >>> orders_named_in("a keyed list beside a scene — order 24")
            ('order 24',)

            ```
        - The letter is part of the number:
            ```python
            >>> from digitalearth.base.contract import orders_named_in
            >>> orders_named_in("adopting the Core spelling is order 27a")
            ('order 27a',)

            ```
    """
    return tuple(_ORDER_REFERENCE.findall(reason))


def _counted_order(order: str) -> Tuple[int, str]:
    """Return the key an order sorts by, so a list of them reads as the roadmap counts them.

    The spelling is read from the right rather than matched, because the pattern that used to match it —
    ``(\\d+)([a-z]*)$`` — backtracks quadratically (`python:S8786`). Anchored at the end, ``\\d+`` has to give
    a digit back and retry once per start position inside any digit run that does not reach the end, so a near
    miss costs O(n²): measured on ``"order " + "1" * n + "!"``, 14 ms at n=1,000, 999 ms at n=8,000 and 15.9 s
    at n=32,000 — quadrupling on every doubling. The scan below is one pass over the tail and stayed at 1 µs
    across all of those. No order is long today, so nothing was slow; a module that validates user-facing
    strings should not carry the shape at all.

    It is the same function, not an approximation of it: ``str.isdecimal`` is true for exactly the Unicode
    category ``\\d`` matches (``Nd``), and a single trailing newline is dropped first because ``$`` matches
    before one. Measured over 222,652 inputs — every string up to length four over ``"0o1a2 b\\n9zA-"``, the
    real spellings, Arabic-Indic digits, a superscript two, and 200,000 random strings — the two agreed on
    every one.

    Args:
        order: An order, spelled as :data:`ROADMAP_ORDERS` keys it.

    Returns:
        Its number and its letter suffix, so ``order 27a`` follows ``order 27`` and neither is read as the
        other. An order whose spelling carries no number sorts first, under its own text: there is no right
        place for it, and putting it where a reader will see it is better than hiding it in the middle.

    Examples:
        - The number and the letter are counted apart, so one cannot be read as the other:
            ```python
            >>> from digitalearth.base.contract import _counted_order
            >>> _counted_order("order 27"), _counted_order("order 27a")
            ((27, ''), (27, 'a'))

            ```
        - A spelling carrying no number sorts first, under its own text:
            ```python
            >>> from digitalearth.base.contract import _counted_order
            >>> _counted_order("order next")
            (0, 'order next')

            ```
    """
    probe = order[:-1] if order.endswith("\n") else order
    letters = probe[len(probe.rstrip(_ORDER_SUFFIX_LETTERS)) :]
    end = len(probe) - len(letters)
    start = end
    while start and probe[start - 1].isdecimal():
        start -= 1
    if start == end:
        return (0, order)
    return (int(probe[start:end]), letters)


def _orders_named(orders: Iterable[str]) -> str:
    """List the roadmap orders a refusal names, counted rather than spelled.

    `sorted()` over the keys compares them character by character, which was right for every order the
    contract names today — they all have two digits — and wrong for the next one-digit or three-digit order
    to be added (`R2-N1`). A refusal exists to point a reader at the orders that *are* written down, so the
    list it hands them should be in the order they are numbered.

    Args:
        orders: The orders to name.

    Returns:
        The orders, comma-separated, in roadmap order.

    Examples:
        - Ten does not come before nine, and a letter is a step after the bare number:
            ```python
            >>> from digitalearth.base.contract import _orders_named
            >>> _orders_named(["order 10", "order 9", "order 27a", "order 27"])
            'order 9, order 10, order 27, order 27a'

            ```
    """
    return ", ".join(sorted(orders, key=_counted_order))


def roadmap_order(order: str) -> str:
    """Return what one roadmap order builds.

    Args:
        order: The order, spelled as a reason names it — `"order 24"`.

    Returns:
        One line saying what that order builds.

    Raises:
        KeyError: for an order this contract does not point at, naming the ones it does. A reason may only
            cite an order that is written down here, so a number nobody wrote down fails loudly rather than
            reading as a plan (review R-L3).

    Examples:
        - What a caller waiting on a tier's adoption of a Core spelling is waiting for:
            ```python
            >>> from digitalearth.base.contract import roadmap_order
            >>> roadmap_order("order 27a").split("—")[0].strip()
            "the Core contract's remainder"

            ```
        - An order nobody wrote down is refused with the ones there are:
            ```python
            >>> from digitalearth.base.contract import roadmap_order
            >>> try:
            ...     roadmap_order("order 99")
            ... except KeyError as error:
            ...     print(error.args[0])
            'order 99' is not a roadmap order this contract names; it names order 27a

            ```
    """
    try:
        return ROADMAP_ORDERS[order]
    except KeyError:
        raise KeyError(
            f"{order!r} is not a roadmap order this contract names; it names "
            + _orders_named(ROADMAP_ORDERS)
        ) from None
