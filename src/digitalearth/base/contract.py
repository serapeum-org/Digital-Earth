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
    "CORE",
    "PENDING",
    "TIER2",
    "Method",
    "alias_table",
    "core_method",
    "pending_for",
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
        builds_in="Wave 5, order 23",
    ),
    Method(
        "move_layer",
        "Move a layer in draw order, by id.",
        frozenset({"index"}),
        builds_in="Wave 5, order 23",
    ),
    Method(
        "replace_layer",
        "Swap a layer's description for another, keeping its id and place.",
        frozenset(),
        builds_in="Wave 5, order 23",
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

#: What each tier used to call a contract name, as `{backend: {old: new}}`. Every entry is a live alias: the
#: old spelling works, warns once and names its replacement. A tier whose seam has not landed carries its
#: entries here too, so the rename is agreed before it is adopted (#300 interactive, #303 static).
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
                "point_cloud": "grid_points",
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
                "set_visible": "layer management on a live page — Wave 5, order 23",
                "move_layer": "layer management on a live page — Wave 5, order 23",
                "replace_layer": "layer management on a live page — Wave 5, order 23",
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
                "field": "the tier's seam — #300",
                "lines": "the tier's seam — #300",
                "add_layer": "the tier's seam — #300",
                "get_layer": "the tier's seam — #300",
                "remove_layer": "the tier's seam — #300",
                "set_visible": "layer management — Wave 5, order 23",
                "move_layer": "layer management — Wave 5, order 23",
                "replace_layer": "layer management — Wave 5, order 23",
                "layer_ids": "the tier's seam — #300",
                "set_bounds": "the tier's seam — #300",
            }
        ),
        "matplotlib": MappingProxyType(
            {
                "field": "the tier's seam — #303",
                "points": "the tier's seam — #303",
                "lines": "line features on the static tier — #226",
                "polygons": "the tier's seam — #303",
                "add_layer": "the tier's seam — #303",
                "get_layer": "the tier's seam — #303",
                "remove_layer": "the tier's seam — #303",
                "set_visible": "layer management — Wave 5, order 23",
                "move_layer": "layer management — Wave 5, order 23",
                "replace_layer": "layer management — Wave 5, order 23",
                "layer_ids": "the tier's seam — #303",
                "set_bounds": "the tier's seam — #303",
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
        `{old: new}`, empty for a tier that renamed nothing.

    Examples:
        - The web tier's raster builder was `add_raster`:
            ```python
            >>> from digitalearth.base.contract import alias_table
            >>> alias_table("web")["add_raster"]
            'field'

            ```
    """
    return ALIASES.get(backend, MappingProxyType({}))


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
