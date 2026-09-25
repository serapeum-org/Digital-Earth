"""The contract every backend answers to — as data, so a test can hold a facade against it.

Four tiers grew four vocabularies. A raster field was `imshow` on static, `image` on interactive, `add_raster`
on web and `terrain` on 3-D; a colour key was a builder on two tiers and a toggle on a third; `render` returned
the engine's object on two and `None` on one; and eight names meant two different things depending on which
facade you were holding. Every one of those was defensible where it grew and indefensible across four.

This module is the agreement that ends it, written down rather than described: :data:`CORE` is the vocabulary
every tier answers to where it can draw the thing at all, :data:`TIER2` the names that must agree wherever they
appear, :data:`ALIASES` what each tier used to call them, and :data:`PENDING` what a tier has not built yet and
which roadmap order builds it. A contract test per tier reads this and compares; nothing here imports a
renderer, so the comparison costs no engine.

**A spelling never changes meaning in the release that deprecates it.** An old name keeps working, warns once
through :func:`~digitalearth.base.deprecation.renamed_method`, and names its replacement — and where the two
would mean different things (`globe(True)` against `projection("globe")`), the tier translates rather than
forwarding blindly.
"""

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import FrozenSet, Iterable, Mapping, Optional, Tuple

__all__ = [
    "ALIASES",
    "PLANNED_RENAMES",
    "CORE",
    "PENDING",
    "ROADMAP_ORDERS",
    "TIER2",
    "Method",
    "alias_table",
    "core_method",
    "orders_named_in",
    "pending_for",
    "planned_renames",
    "roadmap_order",
]


@dataclass(frozen=True)
class Method:
    """One method of the contract: its name, what it takes, and what it hands back.

    Attributes:
        name: The canonical spelling, the same on every tier.
        doc: One line saying what it does — the text a support matrix shows beside it.
        keywords: The canonical keywords it accepts beyond its data argument. A tier may take more; it may not
            spell these differently.
        returns: What a caller gets: `"self"` for a chainable builder, `"engine"` for the renderer's own
            object, `"path"` for a written file, `"value"` for a description.
        builds_in: The roadmap order that builds it, for a name the contract declares before any tier has it.
            `None` for a name that is already drawn somewhere.

    Examples:
        - The contract says what a name means before any tier is consulted:
            ```python
            >>> from digitalearth.base.contract import core_method
            >>> field = core_method("field")
            >>> field.returns, "cmap" in field.keywords
            ('self', True)

            ```
        - A name nobody has built yet says which order builds it:
            ```python
            >>> from digitalearth.base.contract import core_method
            >>> core_method("move_layer").builds_in
            'order 23'

            ```
    """

    name: str
    doc: str
    keywords: FrozenSet[str] = field(default_factory=frozenset)
    returns: str = "self"
    builds_in: Optional[str] = None


#: When the three live layer-management methods are built. Named once because it is one decision — the order
#: that adds toggle, reorder and replace to the tiers still without them — rather than nine independent notes
#: that happen to agree today.
#:
#: It names the **order** and not the wave. This said "Wave 5, order 23" until a wave was inserted ahead of it
#: and every later wave renumbered, after which it told callers their methods were coming in a wave that had
#: already shipped without them (#317). Orders keep their numbers when the plan moves; waves do not, so a wave
#: number in a message a user reads is a fact with a shelf life.
#:
#: Three tiers are waiting, and only one of them has an open issue: #216 for the static half. The web half's
#: issue, #188, is **closed** — it closed with the identity work, while `set_visible`, `move_layer` and
#: `replace_layer` are still absent from `WebMap`, so nothing open tracks that half. The interactive half was
#: never filed. The 3-D tier is not waiting at all: measured, `Scene3D` answers to all three already, which is
#: why `PENDING` lists none of them against it. The issue numbers stay out of the reasons below deliberately —
#: a reason is held to :data:`~tests.open_issues.KNOWN_OPEN_ISSUES`, and a closed issue could not pass it.
_LAYER_MANAGEMENT_ORDER = "order 23"

#: Why a tier has none of them yet. The two spellings differ in what "layer management" means on that tier:
#: on a live page the layers are there to be toggled while the viewer watches, which is the harder half.
_PENDING_LAYERS = f"layer management — {_LAYER_MANAGEMENT_ORDER}"
_PENDING_LIVE_LAYERS = f"layer management on a live page — {_LAYER_MANAGEMENT_ORDER}"

#: Why a tier cannot add, read back or drop a layer by id. The same order, because it is the same decision seen
#: from its other side: every renderer already indexes its layers, so what order 23 adds is the **public**
#: method on each tier rather than the machinery underneath it.
_PENDING_IDENTITY = f"layer identity — {_LAYER_MANAGEMENT_ORDER}"

#: Where a tier adopts a Core spelling it has already agreed to. Order 27a is U-3's remainder — the "canonical
#: names + deprecated aliases" half of the contract (DE-26, folded into U-3), which PR #304 froze without any
#: tier adopting. :data:`PLANNED_RENAMES` is the record of what each tier agreed to call these.
_RENAME_ORDER = "order 27a"

#: Where a figure learns to frame itself: order 26, auto-framing and camera round-trip.
_FRAMING_ORDER = "order 26"

#: Where a colour key stops being the most recent classification and becomes a guide on its own layer's
#: encoding. Named here so the two 3-D rows that cite it are held to the same table as every other citation.
_GUIDES_ORDER = "order 24"

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
#: `order 26` is a pointer into it. Nothing checked that the order on the other end existed: a reason reading
#: "order 99" satisfied every guard, because the guards only ever asked whether the *form* was an order rather
#: than a wave (review R-L3). Measured before this table: a `PENDING` row rewritten to name order 99 passed all
#: three checks in `TestAPendingReasonPointsAtLiveWork`.
#:
#: A test cannot read the document — it is not here to read, and a suite that reached outside the repository
#: for it would fail for everyone who does not have it. So what is vendored is the part a citation needs: the
#: orders this contract points at, and what each one builds. That is enough for a reader to know what they are
#: being promised without going and finding the plan, and enough for the guard to refuse a number nobody wrote
#: down. It is the bargain :data:`~tests.open_issues.KNOWN_OPEN_ISSUES` strikes for issue numbers, for the same
#: reason and with the same cost: extending it is a deliberate act.
#:
#: The keys are the constants above rather than repeated strings, so a citation and its entry cannot disagree.
ROADMAP_ORDERS: Mapping[str, str] = MappingProxyType(
    {
        _LAYER_MANAGEMENT_ORDER: (
            "layer management — the public toggle, reorder and replace on each tier, over the renderers' "
            "own set_visible/is_visible that Wave 6 landed"
        ),
        _GUIDES_ORDER: (
            "a legend and a colorbar that follow their own layer, as guides on its encoding rather than on "
            "the most recent classification"
        ),
        _FRAMING_ORDER: (
            "auto-framing and camera round-trip: a figure that frames itself on its data, and a viewport "
            "that carries the bounds it was framed on"
        ),
        _RENAME_ORDER: (
            "the Core contract's remainder — the renames each tier has agreed to and not adopted "
            "(:data:`PLANNED_RENAMES`), plus the divergences the frozen contract did not settle"
        ),
    }
)


def _drawn_as(old: str) -> str:
    """Say that only the spelling is missing, and where the Core one is adopted.

    A tier that draws the thing under its own name is not missing the capability, and a reason that implies it
    is sends the reader looking for work nobody is going to do. The old spelling is the one
    :data:`PLANNED_RENAMES` records, so the two tables answer consistently.

    Args:
        old: What the tier calls the method today.

    Returns:
        The reason to list against the Core name.
    """
    return f"drawn as {old}() here; adopting the Core spelling is {_RENAME_ORDER}"


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
    ),
    Method(
        "lines",
        "Draw line features.",
        frozenset(
            {"column", "scheme", "k", "cmap", "width", "opacity", "name", "visible"}
        ),
    ),
    Method(
        "polygons",
        "Draw polygon features.",
        frozenset({"column", "scheme", "k", "cmap", "opacity", "name", "visible"}),
    ),
    Method(
        "choropleth",
        "Draw polygons coloured by a column.",
        frozenset({"column", "scheme", "k", "cmap", "opacity", "name", "visible"}),
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
    ),
    Method("remove_layer", "Take a layer off the figure, by id."),
    Method(
        "set_visible",
        "Show or hide a layer, by id.",
        frozenset({"visible"}),
        builds_in=_LAYER_MANAGEMENT_ORDER,
    ),
    Method(
        "move_layer",
        "Move a layer in draw order, by id.",
        frozenset({"index"}),
        builds_in=_LAYER_MANAGEMENT_ORDER,
    ),
    Method(
        "replace_layer",
        (
            "Swap a layer's description for another, keeping its id and its place — except where the new "
            "description changes the layer's band, which re-places it at the top of the band it now "
            "belongs to, as adding it there would."
        ),
        frozenset(),
        builds_in=_LAYER_MANAGEMENT_ORDER,
    ),
    Method(
        "layer_ids",
        "The ids of the figure's layers, in draw order.",
        frozenset(),
        returns="value",
    ),
    Method(
        "set_bounds",
        "Frame the figure on a region; None fits the data.",
        frozenset({"padding"}),
    ),
    Method(
        "render",
        "Build and return the renderer's own object.",
        frozenset(),
        returns="engine",
    ),
    Method("save", "Write the figure to a file.", frozenset(), returns="path"),
    Method("show", "Display the figure.", frozenset(), returns="engine"),
)

#: The names that are not Core — not every tier has them — but that must mean one thing wherever they appear.
#: Each was a live collision until this contract: `tiles` took a URL on one tier and a provider name on
#: another, `contours` filled on one and not on another, `text` took a CRS on two tiers and not on the third.
TIER2: Tuple[Method, ...] = (
    Method("text", "Place a string at a coordinate.", frozenset({"s", "crs", "name"})),
    Method(
        "labels",
        "Label features from a column.",
        frozenset({"column", "crs", "name"}),
    ),
    Method(
        "graticule",
        "Draw meridians and parallels.",
        frozenset({"lon_step", "lat_step", "spacing"}),
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
    ),
    Method("projection", "Draw in another projection, by name.", frozenset()),
)

#: What each tier used to call a contract name, as `{backend: {old: new}}`. **Every entry here is live**: the
#: old spelling works, warns once and names its replacement, and a contract test holds each tier to that.
#:
#: A rename that has been agreed but not adopted lives in :data:`PLANNED_RENAMES` instead. Both were in this
#: table once, with a docstring that claimed liveness in one sentence and disclaimed it in the next — so
#: `alias_table("matplotlib")` told a caller that `imshow` was deprecated in favour of a `field` that does not
#: exist, and `imshow` itself warned nobody (review M5). Two claims, two tables.
ALIASES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "web": MappingProxyType(
            {
                "add_raster": "field",
                "fit_bounds": "set_bounds",
                "title": "set_title",
                "animate": "save_animation",
                "to_gif": "save_animation",
                "save_gif": "save_animation",
                "terrain": "terrain_tiles",
                "globe": "projection",
            }
        ),
        "3d": MappingProxyType({"animate": "record"}),
        # The six adopted at order 27a. Each was in `PLANNED_RENAMES` — agreed and not adopted — and moving
        # it here is what turns the liveness checks on: the old spelling has to exist, warn, and name its
        # replacement. Every recipe key underneath is unchanged (`via="imshow"`, `"scatter"`, `"shapes"`,
        # `"image"`; the interactive line layer's kind was already the neutral `"lines"`), so a figure
        # written before the rename still reads back into the drawer that made it.
        "matplotlib": MappingProxyType(
            {
                "imshow": "field",
                "scatter": "points",
                "shapes": "polygons",
                # Not one of the six: `set_extent` was never a `PLANNED_RENAMES` row, because it was not the
                # Core method under an older name (see that table). Order 27a adopted the *name* anyway, so
                # the old spelling is live here and the capability gap is a keyword shortfall.
                "set_extent": "set_bounds",
            }
        ),
        "interactive": MappingProxyType(
            {
                "image": "field",
                "path": "lines",
                "add_element": "add_layer",
            }
        ),
    }
)

#: The renames a tier has agreed to and not yet adopted, as `{backend: {old: new}}`. Nothing here warns and
#: nothing here forwards: the old name is simply what the tier still calls the method, and the new one is what
#: it will be called once the tier adopts it.
#:
#: **It is empty, and that is a result rather than an omission.** It held six — `image`/`path`/`add_element` on
#: the interactive tier and `imshow`/`scatter`/`shapes` on the static one — and all six were adopted at order
#: 27a, which moved them to :data:`ALIASES` where their liveness is checked. `alias_table` is where a caller
#: looks now; this is where the *next* agreed-but-unadopted spelling is recorded, kept because the agreement
#: is part of the contract — a tier should not have to re-decide a spelling — and kept apart from
#: :data:`ALIASES` because "you may still write this" and "we intend to rename this" are answers to different
#: questions. :func:`_drawn_as` is the `PENDING` reason such a row explains itself by, and has no caller for
#: the same reason: nothing is waiting.
#:
#: Two names deliberately never joined it, and the reasons outlive the rows:
#:
#: - `set_extent` was **not** `set_bounds` under an older name, and listing it as one told a caller the
#:   method already existed under another spelling while `PENDING` dated it at order 26 — the same arrival,
#:   two answers (review R-L4). Order 27a resolved that by splitting the two claims rather than by adopting
#:   the row: the tier answers to `set_bounds` now, returning `self`, with `set_extent` a live alias in
#:   :data:`ALIASES`; the `padding` it still does not take, and the `None` that would fit the data, are
#:   auto-framing at order 26 and are recorded as a keyword shortfall against that order.
#: - `point_cloud` is a second *current* spelling of `grid_points`, which the static tier offers and
#:   deprecates neither of. A rename is a name on its way out, and nothing has been agreed about that one
#:   (review M5).
PLANNED_RENAMES: Mapping[str, Mapping[str, str]] = MappingProxyType({})

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
                "set_visible": _PENDING_LIVE_LAYERS,
                "move_layer": _PENDING_LIVE_LAYERS,
                "replace_layer": _PENDING_LIVE_LAYERS,
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
                "colorbar": "the scalar bar is PyVista's, and becomes a guide on the encoding — order 24",
                "legend": "a keyed list beside a scene — order 24",
                "set_bounds": "a scene is framed by its camera, not by an extent (see Capabilities.absent)",
            }
        ),
        "interactive": MappingProxyType(
            {
                # `field`, `lines` and `add_layer` were listed here, each `_drawn_as` the tier's own
                # spelling. All three are adopted at order 27a: the tier answers to the Core name and the old
                # spelling is a live alias, so neither is pending any more.
                "get_layer": _PENDING_IDENTITY,
                "remove_layer": _PENDING_IDENTITY,
                "set_visible": _PENDING_LAYERS,
                "move_layer": _PENDING_LAYERS,
                "replace_layer": _PENDING_LAYERS,
                "set_bounds": f"no framing method here under any spelling — {_FRAMING_ORDER}",
            }
        ),
        "matplotlib": MappingProxyType(
            {
                # `field`, `points` and `polygons` were listed here, each `_drawn_as` the tier's own
                # spelling. All three are adopted at order 27a: the tier answers to the Core name and the old
                # spelling is a live alias, so neither is pending any more.
                "lines": "line features on the static tier — #226",
                "add_layer": _PENDING_IDENTITY,
                "get_layer": _PENDING_IDENTITY,
                "remove_layer": _PENDING_IDENTITY,
                "set_visible": _PENDING_LAYERS,
                "move_layer": _PENDING_LAYERS,
                "replace_layer": _PENDING_LAYERS,
                # `set_bounds` was listed here, "framed by set_extent(bbox) here, which neither pads nor
                # fits the data". Order 27a took the *name*: the tier answers to `set_bounds` and returns
                # `self`, and `set_extent` is a live alias. What it still does not take is `padding`, and it
                # has no `None` that fits the data — that half is auto-framing, and `KEYWORD_SHORTFALLS`
                # records it against the framing order so the rename cannot be read as the capability.
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


def alias_table(backend: str) -> Mapping[str, str]:
    """Return what one tier used to call the contract's names.

    Args:
        backend: The tier, as `quickmap(backend=...)` spells it.

    Returns:
        `{old: new}` for the aliases that are **live** — every one of them forwards and warns. A tier whose
        renames are agreed but unadopted answers `{}` here and names them through
        :func:`planned_renames`, because a caller reads this table to know what still works.

    Examples:
        - The web tier's raster builder was `add_raster`:
            ```python
            >>> from digitalearth.base.contract import alias_table
            >>> alias_table("web")["add_raster"]
            'field'

            ```
        - The static tier's was `imshow`, since order 27a adopted the Core spelling there:
            ```python
            >>> from digitalearth.base.contract import alias_table
            >>> alias_table("matplotlib")["imshow"]
            'field'

            ```
        - A tier that has renamed nothing answers empty, rather than raising:
            ```python
            >>> from digitalearth.base.contract import alias_table
            >>> dict(alias_table("nobody"))
            {}

            ```
    """
    return ALIASES.get(backend, MappingProxyType({}))


def planned_renames(backend: str) -> Mapping[str, str]:
    """Return the renames one tier has agreed to and not yet adopted.

    Args:
        backend: The tier, as `quickmap(backend=...)` spells it.

    Returns:
        `{old: new}`, empty for a tier with none. Nothing here forwards or warns today: the old name is what
        the tier still calls the method.

    Examples:
        - **Every tier answers empty right now**, because order 27a adopted all six rows this table held;
          they are live aliases in :data:`ALIASES` instead, which is where a caller looks for what still
          works. The next agreed-but-unadopted spelling is recorded here:
            ```python
            >>> from digitalearth.base.contract import planned_renames
            >>> dict(planned_renames("matplotlib"))
            {}

            ```
        - A tier the table has never held answers the same way, rather than raising:
            ```python
            >>> from digitalearth.base.contract import planned_renames
            >>> dict(planned_renames("web"))
            {}

            ```
    """
    return PLANNED_RENAMES.get(backend, MappingProxyType({}))


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
        - A reason that names one:
            ```python
            >>> from digitalearth.base.contract import orders_named_in, pending_for
            >>> orders_named_in(pending_for("interactive")["set_bounds"])
            ('order 26',)

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
        order: The order, spelled as a reason names it — `"order 26"`.

    Returns:
        One line saying what that order builds.

    Raises:
        KeyError: for an order this contract does not point at, naming the ones it does. A reason may only
            cite an order that is written down here, so a number nobody wrote down fails loudly rather than
            reading as a plan (review R-L3).

    Examples:
        - What a caller waiting on `set_bounds` is waiting for:
            ```python
            >>> from digitalearth.base.contract import roadmap_order
            >>> roadmap_order("order 26").split(":")[0]
            'auto-framing and camera round-trip'

            ```
        - An order nobody wrote down is refused with the ones there are:
            ```python
            >>> from digitalearth.base.contract import roadmap_order
            >>> try:
            ...     roadmap_order("order 99")
            ... except KeyError as error:
            ...     print(error.args[0])
            'order 99' is not a roadmap order this contract names; it names order 23, order 24, order 26, order 27a

            ```
    """
    try:
        return ROADMAP_ORDERS[order]
    except KeyError:
        raise KeyError(
            f"{order!r} is not a roadmap order this contract names; it names "
            + _orders_named(ROADMAP_ORDERS)
        ) from None
