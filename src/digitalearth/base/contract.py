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

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import FrozenSet, Mapping, Optional, Tuple

__all__ = [
    "ALIASES",
    "PLANNED_RENAMES",
    "CORE",
    "PENDING",
    "TIER2",
    "Method",
    "alias_table",
    "core_method",
    "pending_for",
    "planned_renames",
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
            'Wave 5, order 23'

            ```
    """

    name: str
    doc: str
    keywords: FrozenSet[str] = field(default_factory=frozenset)
    returns: str = "self"
    builds_in: Optional[str] = None


#: When the three live layer-management methods are built. Named once because it is one decision — the wave
#: that adds toggle, reorder and replace to every tier — rather than nine independent notes that happen to
#: agree today.
_LAYER_MANAGEMENT_WAVE = "Wave 5, order 23"

#: Why a tier has none of them yet. The two spellings differ in what "layer management" means on that tier:
#: on a live page the layers are there to be toggled while the viewer watches, which is the harder half.
_PENDING_LAYERS = f"layer management — {_LAYER_MANAGEMENT_WAVE}"
_PENDING_LIVE_LAYERS = f"layer management on a live page — {_LAYER_MANAGEMENT_WAVE}"

#: Why a tier answers to less of the Core than it will: its renderer seam has not landed. One string per tier,
#: so the issue number is corrected in one place when the seam does land.
_PENDING_INTERACTIVE_SEAM = "the tier's seam — #300"
_PENDING_STATIC_SEAM = "the tier's seam — #303"


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
        builds_in=_LAYER_MANAGEMENT_WAVE,
    ),
    Method(
        "move_layer",
        "Move a layer in draw order, by id.",
        frozenset({"index"}),
        builds_in=_LAYER_MANAGEMENT_WAVE,
    ),
    Method(
        "replace_layer",
        (
            "Swap a layer's description for another, keeping its id and its place — except where the new "
            "description changes the layer's band, which re-places it at the top of the band it now "
            "belongs to, as adding it there would."
        ),
        frozenset(),
        builds_in=_LAYER_MANAGEMENT_WAVE,
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
                "terrain": "terrain_tiles",
                "globe": "projection",
            }
        ),
        "3d": MappingProxyType({"animate": "record"}),
    }
)

#: The renames a tier has agreed to and not yet adopted, as `{backend: {old: new}}`. Nothing here warns and
#: nothing here forwards: the old name is simply what the tier still calls the method, and the new one is what
#: it will be called when its seam lands (#300 interactive, #303 static). Kept in the contract because the
#: agreement is part of it — a tier that seams later should not have to re-decide the spelling — and kept
#: apart from :data:`ALIASES` because "you may still write this" and "we intend to rename this" are answers to
#: different questions.
PLANNED_RENAMES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "interactive": MappingProxyType(
            {
                "image": "field",
                "path": "lines",
                "add_element": "add_layer",
            }
        ),
        "matplotlib": MappingProxyType(
            {
                "imshow": "field",
                "scatter": "points",
                "shapes": "polygons",
                "set_extent": "set_bounds",
                # `point_cloud` is not listed: it is a second *current* spelling of `grid_points`, which the
                # tier offers and deprecates neither of. A rename is a name on its way out, and nothing has
                # been agreed about that one (review M5).
            }
        ),
    }
)

#: What a tier has not built yet, as `{backend: {name: why}}`. This is the honest half of the contract: a name
#: absent because the tier cannot draw it at all reads differently from one absent because nobody has written
#: it, and only the tier can say which. A contract test holds each facade against `CORE` minus its pending list.
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
                    "builder is Wave 5"
                ),
                "add_layer": (
                    "a caller's own object is a PyVista mesh or volume, so it is added with add_mesh() or "
                    "add_volume(), which record a custom:pyvista layer (#293)"
                ),
                "lines": "line features in three dimensions — TD, Wave 5",
                "polygons": "polygons are drawn extruded here; a flat fill is Wave 5",
                "choropleth": "a classified fill follows polygons — Wave 5",
                "colorbar": "the scalar bar is PyVista's, and is a guide on the encoding (#292)",
                "legend": "a keyed list beside a scene — Wave 5, order 24",
                "set_bounds": "a scene is framed by its camera, not by an extent (see Capabilities.absent)",
            }
        ),
        "interactive": MappingProxyType(
            {
                "field": _PENDING_INTERACTIVE_SEAM,
                "lines": _PENDING_INTERACTIVE_SEAM,
                "add_layer": _PENDING_INTERACTIVE_SEAM,
                "get_layer": _PENDING_INTERACTIVE_SEAM,
                "remove_layer": _PENDING_INTERACTIVE_SEAM,
                "set_visible": _PENDING_LAYERS,
                "move_layer": _PENDING_LAYERS,
                "replace_layer": _PENDING_LAYERS,
                "layer_ids": _PENDING_INTERACTIVE_SEAM,
                "set_bounds": _PENDING_INTERACTIVE_SEAM,
            }
        ),
        "matplotlib": MappingProxyType(
            {
                "field": _PENDING_STATIC_SEAM,
                "points": _PENDING_STATIC_SEAM,
                "lines": "line features on the static tier — #226",
                "polygons": _PENDING_STATIC_SEAM,
                "add_layer": _PENDING_STATIC_SEAM,
                "get_layer": _PENDING_STATIC_SEAM,
                "remove_layer": _PENDING_STATIC_SEAM,
                "set_visible": _PENDING_LAYERS,
                "move_layer": _PENDING_LAYERS,
                "replace_layer": _PENDING_LAYERS,
                "layer_ids": _PENDING_STATIC_SEAM,
                "set_bounds": _PENDING_STATIC_SEAM,
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
        - A tier whose seam has not landed has renamed nothing yet:
            ```python
            >>> from digitalearth.base.contract import alias_table
            >>> dict(alias_table("matplotlib"))
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
        - The static tier's raster builder is still `imshow`, and is to become `field`:
            ```python
            >>> from digitalearth.base.contract import planned_renames
            >>> planned_renames("matplotlib")["imshow"]
            'field'

            ```
        - A tier that has adopted its renames has none planned:
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
