"""DecorationMixin — what a 3-D scene says about itself, rather than what it draws (#203).

This tier had no decoration at all. A title, a label at a place, the extent of the data and which way is up
were reachable only as `scene.plotter.add_text(...)` / `scene.plotter.show_bounds(...)` — which is the
abstraction the whole package rests on being stepped around, and which left none of it covered by the tier's
tests or reachable from `quickmap(backend="3d")`. The other three tiers each ship a `decoration` module; this
is the fourth, and it follows their names wherever the concept is the same (`set_title`, `text`).

**Two homes, not one.** `base/spec/furniture.py` names the four places a decoration can live, and this module
uses two of them:

- **A layer in the tree.** :meth:`DecorationMixin.text` writes a `LayerSpec` of the registered `text` kind, so
  a label has an id, is hidden with `set_visible`, taken off with `remove_layer`, and drawn again from
  `to_dict()`/`from_dict()` like a terrain or a point cloud. That is what the other two tiers that have `text`
  do with it, and a string placed at a coordinate really is drawn in the scene's coordinates.
- **Figure furniture.** :meth:`DecorationMixin.set_title` is the panel's own `title`, which `PanelSpec` already
  carries, so it is described and is drawn again from the description — the web tier's answer for the same
  name. :meth:`DecorationMixin.axes` (the labelled bounds box) and
  :meth:`DecorationMixin.orientation_axes` (the corner triad) are furniture too, but **drawn view state**
  rather than described: they are held on the scene and re-applied to whichever plotter it is drawing on,
  exactly as its vertical exaggeration and its camera are. All three go through :func:`redraw_decoration`,
  which :meth:`~digitalearth.three_d.base.Scene3DBase._dress_plotter` calls for the same reason it applies the
  view scale.

**Why the box and the triad are not described, and what would change that.** The only description that fits a
coordinate frame is a :class:`~digitalearth.base.spec.Furniture` on the panel, and `Furniture` refuses a kind
nobody registered. Registering one means two entries in `base/`: a `FurnitureInfo` in `base/registry.py` and the
matching name in `base/capabilities.py`'s feature table, since `tests/base/test_capabilities.py` refuses
registered furniture no tier could declare. Both are outside this tier, so they are **reported rather than
made** — with the names `axes_box` and `orientation_axes`. Until then the two follow the static tier's own answer
for figure-level decoration (`Scene.set_title`, `Scene.colorbar`, `Scene.legend` are drawn and not described):
the scene keeps them, the figure does not carry them, and removing a layer never takes them with it.

**What this module deliberately does not build.** A scalar bar and a legend — and not because they are
missing. Both are Core names keyed to a layer's `Encoding`, so they are **guides** rather than decoration: they
explain something the layer publishes, where a title, a label and a coordinate frame explain nothing. They live
in :mod:`digitalearth.three_d.guides` as `colorbar()` and `legend()` (order 24), under those spellings and no
others — a differently shaped `scalar_bar()` here would put two spellings on one concept, which is the single
thing the Core contract exists to prevent. A north arrow is not built at all: the tier declares it absent, with
its reason (a scene can be looked at from any direction, so there is no fixed north on screen).
"""

import logging
import math
from dataclasses import replace as with_fields
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from digitalearth.base.crs import is_geographic
from digitalearth.base.spec import LayerSpec
from digitalearth.base.spec._serial import crs_to_json
from digitalearth.base.spec.bounds import same_crs
from digitalearth.three_d.layer import drawing_props

logger = logging.getLogger(__name__)

#: The Natural-Earth resolutions :mod:`cleopatra.basemap.reference` publishes — the spelling the static and
#: web tiers' coastlines/borders use too, so one resolution means one dataset on every tier.
_NATURAL_EARTH_RESOLUTIONS: frozenset[str] = frozenset({"110m", "50m", "10m"})

__all__ = [
    "DecorationMixin",
    "axis_labels",
    "axis_titles",
    "draw_reference_fill",
    "draw_reference_lines",
    "draw_text",
    "redraw_decoration",
]

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object

#: The corner annotation index PyVista's `add_title` writes into (`vtkCornerAnnotation`'s upper edge), and the
#: one a subtitle is written under it at. Named because reading a title back off the plotter is how a test
#: proves the title landed, and the number is meaningless on its own.
TITLE_SLOT: int = 7

#: The actor name PyVista files a subtitle under, so a second `set_title` replaces it rather than stacking.
SUBTITLE_ACTOR: str = "digitalearth-subtitle"

#: What the axes are called when the scene's display CRS says nothing useful: the plain survey words, which are
#: what #203 asked for in place of PyVista's `X`/`Y`/`Z`. A bare array has no CRS to read, and `X`/`Y`/`Z`
#: labels a figure with the letters of its own array indices.
FALLBACK_AXIS_TITLES: tuple[str, str, str] = ("Easting", "Northing", "Elevation")

#: The axis titles of a geographic display CRS, used when the CRS resolves but names no axes of its own.
GEOGRAPHIC_AXIS_TITLES: tuple[str, str, str] = ("Longitude", "Latitude", "Elevation")

#: The short labels a corner triad is drawn with when the scene is projected. The triad is a widget a couple of
#: centimetres across, so it takes abbreviations rather than the box's full titles — but never `X`/`Y`/`Z`.
#: **Not** the CRS's own `abbrev` fields: measured, EPSG:3857 abbreviates its axes `X` and `Y`, which is exactly
#: the labelling #203 asked to be rid of, so the short forms are chosen from geographic-or-projected instead.
FALLBACK_AXIS_LABELS: tuple[str, str, str] = ("E", "N", "Up")

#: The short labels a geographic scene's triad is drawn with.
GEOGRAPHIC_AXIS_LABELS: tuple[str, str, str] = ("Lon", "Lat", "Up")

#: Which of a CRS's axes is the scene's x, y and z, by the direction the axis points. A CRS declares its axes
#: in its own order — EPSG:4326 is (latitude, longitude) — while a scene always draws x east and y north, so
#: the axes are matched by direction rather than by position. Taking them in CRS order labelled a lon/lat scene
#: back to front.
_AXIS_DIRECTIONS: tuple[tuple[str, ...], ...] = (
    ("east", "west"),
    ("north", "south"),
    ("up", "down"),
)


def axis_titles(crs: Any) -> tuple[str, str, str]:
    """Return what to call a scene's three axes, read off its display CRS.

    A 3-D figure labelled `X`/`Y`/`Z` says nothing about what is being looked at, which is what PyVista's own
    default does and what #203 asked for instead. The CRS knows the answer — a projected CRS calls its axes
    Easting and Northing, a geographic one calls them longitude and latitude, and a 3-D CRS names its vertical
    axis too — so the titles are read from it through pyramids rather than written out per tier.

    Args:
        crs: The scene's display CRS, in any spelling pyramids reads, or `None` for a scene whose layers
            declared none.

    Returns:
        The `(x, y, z)` titles. From the CRS's own axis names where it has them, matched by the direction each
        axis points rather than by the order the CRS declares them in; else the geographic or the plain survey
        words; and :data:`FALLBACK_AXIS_TITLES` for a scene with no CRS at all, which is a bare array drawn in
        whatever units it was handed in.

    Examples:
        - A projected CRS names its own axes, and the vertical one is filled in:
            ```python
            >>> from digitalearth.three_d.decoration import axis_titles
            >>> axis_titles(32618)
            ('Easting', 'Northing', 'Elevation')

            ```
        - A geographic CRS is labelled in its own terms, east first however the CRS orders its axes:
            ```python
            >>> from digitalearth.three_d.decoration import axis_titles
            >>> axis_titles(4326)
            ('Geodetic longitude', 'Geodetic latitude', 'Elevation')

            ```
        - A scene with no CRS is labelled with the survey words rather than with array indices:
            ```python
            >>> from digitalearth.three_d.decoration import axis_titles
            >>> axis_titles(None)
            ('Easting', 'Northing', 'Elevation')

            ```
    """
    if is_geographic(crs) is None:
        return FALLBACK_AXIS_TITLES
    default = GEOGRAPHIC_AXIS_TITLES if is_geographic(crs) else FALLBACK_AXIS_TITLES
    return tuple(  # type: ignore[return-value]
        _named_axis(crs, directions) or fallback
        for directions, fallback in zip(_AXIS_DIRECTIONS, default)
    )


def axis_labels(crs: Any) -> tuple[str, str, str]:
    """Return the short axis labels a corner triad is drawn with, for a scene's display CRS.

    :func:`axis_titles`' companion for the widget that has no room for a title. The two are chosen the same
    way — from the data rather than from the engine's defaults — but a triad gets `Lon`/`Lat`/`Up` where the
    box gets `Geodetic longitude`.

    Args:
        crs: The scene's display CRS, in any spelling pyramids reads, or `None`.

    Returns:
        :data:`GEOGRAPHIC_AXIS_LABELS` for a geographic CRS, :data:`FALLBACK_AXIS_LABELS` otherwise — which
        covers a projected CRS and a scene with none, both of which are drawn in a plane's own coordinates.

    Examples:
        - A lon/lat scene and a projected one are labelled in their own terms:
            ```python
            >>> from digitalearth.three_d.decoration import axis_labels
            >>> axis_labels(4326), axis_labels(32618)
            (('Lon', 'Lat', 'Up'), ('E', 'N', 'Up'))

            ```
    """
    return GEOGRAPHIC_AXIS_LABELS if is_geographic(crs) else FALLBACK_AXIS_LABELS


def _named_axis(crs: Any, directions: tuple[str, ...]) -> str | None:
    """Return the name of the CRS axis pointing one way, or `None` when it declares none.

    Args:
        crs: The display CRS, in any spelling pyramids reads.
        directions: The directions that count as this axis — `("east", "west")` for the scene's x.

    Returns:
        The axis's own name, or `None` when the CRS has no axis pointing that way (a 2-D CRS has no vertical
        axis) or cannot be read at all.
    """
    from pyramids.base.crs import crs_from_user_input

    try:
        axes = crs_from_user_input(crs).axis_info
    # Any CRS-resolution failure means "no name to read", never a guess — the rule `base.crs` follows.
    except Exception:  # noqa: BLE001
        return None
    for axis in axes:
        if str(getattr(axis, "direction", "")).lower() in directions:
            name = getattr(axis, "name", None)
            return str(name) if name else None
    return None


def _finite(value: Any, name: str, caller: str) -> float:
    """Return `value` as a float, refusing anything that could not be written down.

    Args:
        value: The number the caller gave.
        name: The parameter it arrived as, for the message.
        caller: The method it was written on, for the message.

    Returns:
        `value` as a `float`.

    Raises:
        ValueError: when `value` is not a finite number — a figure holding `NaN` or infinity could not be
            stored, and a label anchored there is drawn nowhere while still counting as a layer.

    Examples:
        - A finite number passes through as a float:
            ```python
            >>> from digitalearth.three_d.decoration import _finite
            >>> _finite(3, "lon", "Scene3D.text()")
            3.0

            ```
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{caller}: {name} must be a finite number; got {value!r}"
        ) from None
    if not math.isfinite(number):
        raise ValueError(f"{caller}: {name} must be a finite number; got {value!r}")
    return number


def _display_point(
    scene: Any, x: float, y: float, crs: Any
) -> tuple[float, float] | None:
    """Return `(x, y)` where the scene draws it, reprojected from `crs` through pyramids.

    Args:
        scene: The scene whose display CRS the point is placed in.
        x: The anchor's x in `crs`.
        y: The anchor's y in `crs`.
        crs: What `x`/`y` are measured in, or `None` for "the scene's own".

    Returns:
        The `(x, y)` in the scene's display CRS — the pair unchanged when there is nothing to reproject — or
        `None` when the reprojection puts the point nowhere, which is a far-side point on a clipped CRS.
    """
    display = getattr(scene, "display_crs", None)
    if display is None or crs is None or same_crs(crs, display):
        return x, y
    from pyramids.feature.geometry import reproject_coordinates

    placed_x, placed_y = reproject_coordinates([x], [y], from_crs=crs, to_crs=display)
    if not (math.isfinite(placed_x[0]) and math.isfinite(placed_y[0])):
        return None
    return float(placed_x[0]), float(placed_y[0])


def _draw_title(plotter: Any, heading: str, style: dict[str, Any]) -> None:
    """Put a heading, and an optional second line, onto a plotter.

    Args:
        plotter: The plotter to draw on.
        heading: The title text.
        style: How it is drawn — `font_size`, `color` and `subtitle`, as
            :meth:`DecorationMixin.set_title` recorded them.
    """
    font_size = int(style.get("font_size", 18))
    colour = style.get("color")
    plotter.add_title(heading, font_size=font_size, color=colour)
    # A render window has exactly one centred title band, and the two lines in it cannot be sized apart, so a
    # subtitle is drawn smaller in the upper-left corner instead of under the heading. It is a separate actor,
    # filed under a name of its own, so a second `set_title` replaces it rather than stacking a second line.
    plotter.remove_actor(SUBTITLE_ACTOR, render=False)
    subtitle = style.get("subtitle")
    if subtitle:
        plotter.add_text(
            subtitle,
            position="upper_left",
            font_size=max(int(font_size * 0.6), 1),
            color=colour,
            name=SUBTITLE_ACTOR,
        )


def _draw_axes_box(plotter: Any, crs: Any, style: dict[str, Any]) -> None:
    """Put a labelled bounds box onto a plotter, titled for the CRS the scene draws in.

    Args:
        plotter: The plotter to draw on.
        crs: The scene's display CRS, which the untitled axes are named from.
        style: The keywords :meth:`DecorationMixin.axes` recorded — the caller's own titles among them.
    """
    titles = axis_titles(crs)
    options = dict(style)
    for key, title in zip(("xtitle", "ytitle", "ztitle"), titles):
        # The caller's own title wins; `None` means "name it from the data", which is the whole point.
        if options.get(key) is None:
            options[key] = title
    # PyVista replaces the renderer's one `CubeAxesActor` rather than stacking a second, so re-applying this
    # on every plotter the scene is dressed with leaves one box (measured, PyVista 0.48.4).
    plotter.show_bounds(**options)


def _draw_orientation_axes(plotter: Any, crs: Any, style: dict[str, Any]) -> None:
    """Put the corner orientation triad onto a plotter, labelled for the CRS the scene draws in.

    Args:
        plotter: The plotter to draw on.
        crs: The scene's display CRS, which the unlabelled axes are named from.
        style: The keywords :meth:`DecorationMixin.orientation_axes` recorded.
    """
    options = dict(style)
    for key, label in zip(("xlabel", "ylabel", "zlabel"), axis_labels(crs)):
        if options.get(key) is None:
            options[key] = label
    plotter.add_axes(**options)


def redraw_decoration(scene: Any) -> None:
    """Put the scene's own decoration onto the plotter it is about to draw on.

    The decoration this tier keeps as **figure furniture** — the panel's title, and whether an axes box or an
    orientation triad was asked for — belongs to the scene rather than to whichever plotter happens to exist.
    So it is applied to a window the scene has just been given, whether it built it or a caller handed one
    over, exactly as its vertical exaggeration and its camera are
    (:meth:`~digitalearth.three_d.base.Scene3DBase._dress_plotter`), and again when a figure is drawn into the
    scene, so a stored heading survives the round trip.

    Args:
        scene: The scene whose decoration is being applied. Nothing is drawn when it has no plotter yet — a
            described scene builds no render window, and decoration must not be what forces one.
    """
    plotter = getattr(scene, "_plotter", None)
    # A stand-in plotter need not implement the whole annotation API, and it is still a valid plotter to draw
    # layers on — the same rule `_dress_plotter` applies to `set_scale`.
    if plotter is None or not hasattr(plotter, "add_title"):
        return
    held = dict(getattr(scene, "_decoration", {}))
    # The figure straight off the scene, not `figure_spec`: that property reads the live camera off the
    # plotter, and this runs while the plotter is still being built.
    heading = scene._figure.panels[0].title
    if heading is not None:
        _draw_title(plotter, heading, held.get("title", {}))
    # An absence is the "off" state: `axes(False)` drops the entry, so a window the scene is dressed with next
    # does not get the box back. That is what makes turning it off a property of the scene rather than of one
    # call on one plotter.
    if "axes" in held:
        _draw_axes_box(plotter, scene.display_crs, held["axes"])
    if "orientation_axes" in held:
        _draw_orientation_axes(plotter, scene.display_crs, held["orientation_axes"])


def draw_reference_lines(
    scene: Any, _data: Any, layer: LayerSpec
) -> tuple[Any, Any] | None:
    """Draw the Natural-Earth coastline or border lines a `reference_lines` layer describes.

    Args:
        scene: The scene being drawn into.
        _data: The source slot every drawer takes, unread here — the lines come from Natural Earth, named by
            the layer's props, so a figure holding this layer is storable with no in-memory source.
        layer: The layer's description, whose props carry the dataset, resolution and line style.

    Returns:
        The `(mesh, actor)` pair, or `None` when Natural Earth returned no drawable line and the scene is not
        `strict`.

    Raises:
        OffLimbError: when there is nothing to draw and the scene is `strict`.
    """
    from cleopatra.basemap.reference import natural_earth

    from digitalearth.three_d.vector import _polyline

    props = drawing_props(layer.symbology.props)
    dataset = props.pop("dataset")
    resolution = props.pop("resolution")
    color = props.pop("color", "#000000")
    width = props.pop("width", 1.0)
    opacity = props.pop("opacity", 1.0)
    # The lines are lon/lat; reprojecting them into a projected display CRS is pyramids' job, not this tier's,
    # so draw them as given and say so rather than silently misplacing them (the fills half of #205 waits on
    # cleopatra#384 regardless).
    if scene.display_crs is not None and not is_geographic(scene.display_crs):
        logger.warning(
            "%s: Natural-Earth reference lines are in EPSG:4326 and may not align with the scene's %r display "
            "CRS; reproject the scene's data to a geographic CRS, or wait on the pyramids reprojection path",
            layer.kind,
            scene.display_crs,
        )

    import pyvista as pv

    parts = [
        np.asarray(part, dtype="float64") for part in natural_earth(dataset, resolution)
    ]
    lines = [_polyline(part) for part in parts if len(part) >= 2]
    if not lines:
        scene._skip_empty(layer.kind, f"natural_earth({dataset!r}) returned no lines")
        return None
    merged = pv.MultiBlock(lines).combine()
    return merged, scene.plotter.add_mesh(
        merged, color=color, line_width=float(width), opacity=float(opacity), **props
    )


def draw_reference_fill(
    scene: Any, _data: Any, layer: LayerSpec
) -> tuple[Any, Any] | None:
    """Draw the Natural-Earth land/ocean/lake fill a `reference_fill` layer describes.

    The fill counterpart of :func:`draw_reference_lines`. The polygons are read **hole-aware** from
    cleopatra's `natural_earth_polygons` (each part `[exterior, *holes]`) and triangulated into flat caps at
    `z=0` with their holes carved (:func:`~digitalearth.three_d.vector._cap_with_holes`), so an ocean fill
    shows the continents through it rather than painting over them (cleopatra#384).

    Args:
        scene: The scene being drawn into.
        _data: The source slot every drawer takes, unread here — the polygons come from Natural Earth, named
            by the layer's props, so a figure holding this layer is storable with no in-memory source.
        layer: The layer's description, whose props carry the dataset, resolution and fill style.

    Returns:
        The `(mesh, actor)` pair, or `None` when Natural Earth returned no fillable polygon and the scene is
        not `strict`.

    Raises:
        OffLimbError: when there is nothing to draw and the scene is `strict`.
    """
    from cleopatra.basemap.reference import natural_earth_polygons

    from digitalearth.three_d.vector import _cap_with_holes

    props = drawing_props(layer.symbology.props)
    dataset = props.pop("dataset")
    resolution = props.pop("resolution")
    color = props.pop("color", "#cccccc")
    opacity = props.pop("opacity", 1.0)
    # The polygons are lon/lat; as with the reference lines, reprojecting them into a projected display CRS
    # is pyramids' job, not this tier's, so draw them as given and say so rather than silently misplacing
    # them.
    if scene.display_crs is not None and not is_geographic(scene.display_crs):
        logger.warning(
            "%s: Natural-Earth reference fills are in EPSG:4326 and may not align with the scene's %r "
            "display CRS; reproject the scene's data to a geographic CRS, or wait on the pyramids "
            "reprojection path",
            layer.kind,
            scene.display_crs,
        )

    import pyvista as pv

    caps = []
    for rings in natural_earth_polygons(dataset, resolution):
        # The first ring is the exterior; a part whose exterior is degenerate is nothing to fill, and a
        # degenerate hole is skipped rather than carved.
        if not rings or len(rings[0]) < 3:
            continue
        exterior = np.asarray(rings[0], dtype="float64")
        interiors = [
            np.asarray(hole, dtype="float64") for hole in rings[1:] if len(hole) >= 3
        ]
        caps.append(_cap_with_holes(exterior, interiors))
    if not caps:
        scene._skip_empty(
            layer.kind, f"natural_earth_polygons({dataset!r}) returned no polygons"
        )
        return None
    merged = pv.MultiBlock(caps).combine()
    return merged, scene.plotter.add_mesh(
        merged, color=color, opacity=float(opacity), **props
    )


def draw_text(scene: Any, _data: Any, layer: LayerSpec) -> tuple[Any, Any] | None:
    """Draw the label a `text` layer describes, anchored in the scene's coordinates.

    Args:
        scene: The scene being drawn into.
        _data: The source slot every drawer takes, unread here — a label carries its own anchor and points at
            no source, so a figure holding one is storable even when every other layer was built in memory.
        layer: The layer's description, whose props carry the anchor, the string and the text style.

    Returns:
        The `(anchor, actor)` pair — the anchor is the one-point `PolyData` the label hangs off, so
        `mesh_of()` answers for a label as it does for any other layer — or `None` when the anchor does not
        reproject into the scene's display CRS and the scene is not `strict`.

    Raises:
        OffLimbError: when the anchor cannot be placed and the scene is `strict`.
    """
    props = drawing_props(layer.symbology.props)
    label = props.pop("s")
    anchor_x = props.pop("x")
    anchor_y = props.pop("y")
    anchor_z = props.pop("z")
    placed = _display_point(scene, anchor_x, anchor_y, props.pop("crs", None))
    if placed is None:
        scene._skip_empty(
            layer.id, "the anchor does not reproject into the scene's display CRS"
        )
        return None
    # The tier's spelling on the way in, PyVista's on the way out: `text_size` is contract C3's name for how
    # big text is drawn (`size` is a marker's size and nothing else), and `color` is what every tier calls a
    # colour. Storing the contract's spellings is what lets another tier read this layer's description.
    style = {"font_size": int(props.pop("text_size"))}
    colour = props.pop("color", None)
    if colour is not None:
        style["text_color"] = colour
    import pyvista as pv

    anchor = pv.PolyData(
        np.asarray([[placed[0], placed[1], anchor_z]], dtype="float64")
    )
    # A default rather than a pinned value: a caller who wants the anchor itself marked says so, and a
    # `show_points=True` among the keywords would otherwise be a duplicate-keyword TypeError.
    props.setdefault("show_points", False)
    drawn = scene.plotter.add_point_labels(anchor, [label], **style, **props)
    return anchor, drawn


class DecorationMixin(_MixinBase):
    """Title, labels and the coordinate frame for a :class:`~digitalearth.three_d.scene3d.Scene3D`.

    Composed into `Scene3D` beside the capability mixins, so the composed class stays a thin composition and
    the methods live here. Every method returns `self`, so decoration reads as one expression:
    `scene.terrain(dem).set_title("Elevation").axes().orientation_axes()`.

    Note:
        The class is declared against :class:`~digitalearth.three_d.base.Scene3DBase` under `TYPE_CHECKING`
        only — at runtime the base is `object`, so the composed MRO is what it was before the annotation.

    See Also:
        digitalearth.three_d.scene3d.Scene3D: the composition that supplies the state these methods use.
        digitalearth.three_d.base.Scene3DBase: the typing-only base declared above the class.
    """

    def set_title(
        self,
        heading: str,
        *,
        subtitle: str | None = None,
        font_size: int = 18,
        color: Any = None,
    ) -> Self:
        """Give the scene a heading, and record it on the figure.

        The Core contract's Tier-2 `set_title`, with the `subtitle` it declares: the same name, the same
        chainable return and the same meaning as on the other three tiers. A rendered scene travels alone —
        a screenshot, an exported page — so whatever context the notebook around it had is gone, and the
        heading has to be part of the picture.

        The title is **figure furniture, and described**: it is the panel's own `title`, which `PanelSpec`
        already carries, so it survives `figure_spec.to_dict()` and is drawn again by
        :meth:`~digitalearth.three_d.base.Scene3DBase.from_figure`. It is not a layer — it has no place on the
        ground, it is not in the layer tree, and removing a layer never takes it with it.

        Args:
            heading: The title text, drawn centred along the top of the render window.
            subtitle: A smaller second line — a date, a source, a unit. Drawn in the **upper-left corner**
                rather than under the heading: a render window has one centred title band and the two lines in
                it cannot be sized apart. Recorded as drawing state rather than on the figure, exactly as the
                web tier records its own subtitle.
            font_size: Size of the heading in points; the subtitle is drawn at 60% of it.
            color: Colour of both lines. `None` leaves PyVista's theme colour.

        Returns:
            This scene, so decoration chains.

        Raises:
            TypeError: when `heading` is not a string. A title is the one piece of a figure a reader is
                certain to read, and `None` drawn as `"None"` is worse than no title.

        Examples:
            - A heading is drawn and is part of what the scene describes:
                ```python
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.set_title("Elevation (m)", subtitle="SRTM, 2024")
                >>> scene.figure_spec.panels[0].title
                'Elevation (m)'
                >>> scene.close()

                ```
            - The heading survives the round trip the seam exists for:
                ```python
                >>> from digitalearth.base.spec import FigureSpec
                >>> from digitalearth.three_d import Scene3D
                >>> first = Scene3D(off_screen=True)
                >>> stored = first.set_title("Elevation (m)").figure_spec.to_dict()
                >>> first.close()
                >>> second = Scene3D.from_figure(FigureSpec.from_dict(stored), off_screen=True)
                >>> second.figure_spec.panels[0].title
                'Elevation (m)'
                >>> second.close()

                ```

        See Also:
            digitalearth.web.decoration.DecorationMixin.set_title: the same name on the web tier.
        """
        if not isinstance(heading, str):
            raise TypeError(
                f"set_title() needs the heading as a string; got {type(heading).__name__}"
            )
        panel = with_fields(self._figure.panels[0], title=heading)
        self._figure = with_fields(self._figure, panels=(panel,))
        self._decoration["title"] = {
            "font_size": int(font_size),
            "color": color,
            "subtitle": subtitle,
        }
        redraw_decoration(self)
        return self

    def axes(
        self,
        show: bool = True,
        *,
        xtitle: str | None = None,
        ytitle: str | None = None,
        ztitle: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Draw a labelled box round the data, so the scene says what its coordinates are.

        A 3-D render is otherwise an object floating in space: nothing on the window says how wide the ground
        is or how high the relief goes. This is PyVista's `show_bounds` — the box with ticks and axis titles —
        reached in the tier's own vocabulary, and **titled from the scene's display CRS** rather than from
        PyVista's `X`/`Y`/`Z` (see :func:`axis_titles`). The box's extent follows the renderer, so calling this
        before the data is added is fine: the box grows with what is drawn.

        It is figure furniture, not a layer: it frames the data rather than being data, so it is not in the
        layer tree and removing a layer never takes it with it. It is **not** written into the figure either —
        see the module docstring for the two `base/` registrations that would let it be.

        Args:
            show: Whether the box is drawn. `False` takes it off, and keeps it off the next window the scene
                is given.
            xtitle: What to call the x axis. `None` names it from the display CRS.
            ytitle: What to call the y axis. `None` names it from the display CRS.
            ztitle: What to call the z axis. `None` names it from the display CRS.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.show_bounds` — `grid`, `location`, `ticks`,
                `n_xlabels`, `font_size`, `all_edges`, ….

        Returns:
            This scene, so decoration chains.

        Examples:
            - A projected scene names its own axes rather than showing letters:
                ```python
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True, crs=32618)
                >>> _ = scene.axes()
                >>> box = scene.plotter.renderer.cube_axes_actor
                >>> box.GetXTitle(), box.GetYTitle(), box.GetZTitle()
                ('Easting', 'Northing', 'Elevation')
                >>> scene.close()

                ```
            - Off again, in one keyword rather than a second method:
                ```python
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.axes().axes(False)
                >>> scene.plotter.renderer.cube_axes_actor is None
                True
                >>> scene.close()

                ```

        See Also:
            orientation_axes: the corner triad, which says which way the axes point.
        """
        if not show:
            self._decoration.pop("axes", None)
            if self._plotter is not None:
                self._plotter.remove_bounds_axes()
            return self
        self._decoration["axes"] = {
            "xtitle": xtitle,
            "ytitle": ytitle,
            "ztitle": ztitle,
            **kwargs,
        }
        redraw_decoration(self)
        return self

    def orientation_axes(
        self,
        show: bool = True,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Show the corner triad, so a reader can tell which way the scene is being looked at from.

        A 3-D scene has no fixed orientation — the camera decides — which is why this tier declares a north
        arrow absent. The triad is the answer that does hold: three arrows in the corner of the window,
        turning with the camera, labelled for the CRS the scene draws in rather than `X`/`Y`/`Z` (see
        :func:`axis_labels`).

        Figure furniture, like :meth:`axes`: it sits in a corner of the window rather than in the scene's
        coordinates, so it is not a layer.

        Args:
            show: Whether the triad is drawn. `False` hides it, and keeps it hidden on the next window.
            xlabel: What to label the x arrow. `None` labels it from the display CRS.
            ylabel: What to label the y arrow. `None` labels it from the display CRS.
            zlabel: What to label the z arrow. `None` labels it from the display CRS.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_axes` — `line_width`, `x_color`, `viewport`,
                `labels_off`, ….

        Returns:
            This scene, so decoration chains.

        Examples:
            - A lon/lat scene's triad is labelled in its own terms:
                ```python
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True, crs=4326)
                >>> _ = scene.orientation_axes()
                >>> triad = scene.plotter.renderer.axes_actor
                >>> triad.GetXAxisLabelText(), triad.GetZAxisLabelText()
                ('Lon', 'Up')
                >>> scene.close()

                ```

        See Also:
            axes: the labelled box, which says how far the coordinates reach.
        """
        if not show:
            self._decoration.pop("orientation_axes", None)
            if self._plotter is not None:
                self._plotter.hide_axes()
            return self
        self._decoration["orientation_axes"] = {
            "xlabel": xlabel,
            "ylabel": ylabel,
            "zlabel": zlabel,
            **kwargs,
        }
        redraw_decoration(self)
        return self

    def text(
        self,
        lon: Any,
        lat: Any,
        s: str | None = None,
        *,
        z: Any = 0.0,
        crs: Any = 4326,
        text_size: float = 14.0,
        color: Any = None,
        name: Any = None,
        **kwargs: Any,
    ) -> Self:
        """Place a single string at a coordinate, as a layer of its own.

        The "label this spot" case, and the tier's answer to the Core contract's Tier-2 `text`: the same name,
        the same `s`/`crs`/`name` arguments and the same meaning as on the static, interactive and web tiers,
        plus the `z` a scene drawn in three dimensions needs.

        The label is a **layer**, not furniture: it has an id, `set_visible` hides it, `remove_layer` takes it
        off, and it is drawn again from `figure_spec.to_dict()`. It does **not** set the scene's display CRS —
        an annotation is decoration, and decoration must not decide what coordinates the scene is drawn in — so
        a label given in another CRS is reprojected into the display CRS the data established. Add it after the
        layer that sets that CRS, or give the scene `crs=`, or there is nothing yet to reproject into.

        Args:
            lon: X of the anchor, in `crs` — a longitude by default.
            lat: Y of the anchor, in `crs` — a latitude by default.
            s: The string to draw. Named `s` as it is on the other three tiers, which follow matplotlib's own
                spelling.
            z: Height of the anchor, in the scene's vertical units. `0.0` puts the label on the ground plane.
            crs: What `lon`/`lat` are measured in. EPSG:4326 by default; a point in another CRS is reprojected
                into the scene's display CRS through pyramids, exactly as a layer's data is.
            text_size: Text size in points. Named for the text rather than `size`, which is the visual size of
                a marker and nothing else (contract C3).
            color: Text colour. `None` leaves PyVista's theme colour, so a label matches the scene it is drawn
                on.
            name: The caller's name for the layer, used as its id; `None` numbers it by kind (`text-1`), and a
                name already on the scene is suffixed.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_point_labels` — `bold`, `italic`, `shape`,
                `shape_color`, `always_visible`, ….

        Returns:
            This scene, so decoration chains.

        Raises:
            TypeError: when the string to draw is not given.
            ValueError: when `lon`, `lat`, `z` or `text_size` is not a finite number — a figure holding `NaN`
                or infinity could not be written down.

        Examples:
            - Label a peak, and the label is a layer with an id like any other:
                ```python
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.text(0.0, 0.0, "origin", z=1.0, name="mark")
                >>> scene.layer_ids
                ['mark']
                >>> scene.get_layer("mark").kind
                'text'
                >>> scene.close()

                ```
            - A label is hidden and taken off by id, as a data layer is:
                ```python
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.text(0.0, 0.0, "origin").set_visible("text-1", False)
                >>> scene.renderer.is_visible("text-1")
                False
                >>> _ = scene.remove_layer("text-1")
                >>> scene.layer_ids
                []
                >>> scene.close()

                ```

        See Also:
            digitalearth.static.maps.decoration.DecorationMixin.text: the same name on the static tier.
        """
        call = "Scene3D.text()"
        if s is None:
            raise TypeError(
                "text() needs the string to draw; pass it as the third argument"
            )
        self._add_described_layer(
            kind="text",
            name=name,
            s=s,
            x=_finite(lon, "lon", call),
            y=_finite(lat, "lat", call),
            z=_finite(z, "z", call),
            # In the shared CRS spelling: a caller may hand in a CRS object, which a figure written to JSON
            # has no form for, and this is how every other spec field holds one.
            crs=crs_to_json(crs, f"{call} crs="),
            text_size=_finite(text_size, "text_size", call),
            color=color,
            **kwargs,
        )
        return self

    def coastlines(
        self,
        resolution: str = "110m",
        *,
        name: Any = None,
        color: str = "#000000",
        width: float = 1.0,
        opacity: float = 1.0,
        visible: bool = True,
    ) -> Any:
        """Draw Natural-Earth coastlines as reference lines in the scene (TD-5b, #205).

        Reference geography was reachable only inside :meth:`~digitalearth.three_d.globe.GlobeMixin.globe`,
        where geovista traces the shoreline onto the sphere. This draws it in an ordinary (non-globe) scene —
        a terrain, say — from the same ``cleopatra.basemap.reference`` Natural-Earth coordinates the static and
        web tiers read, as 3-D polylines on the ground plane. The lines are in geographic coordinates
        (EPSG:4326); over a scene whose display CRS is projected they are drawn but warned about, because
        reprojecting them is pyramids' job and the land/ocean **polygon fills** half of #205 is still blocked
        upstream on cleopatra#384.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: Layer id; ``None`` numbers it ``coastlines``.
            color: Line colour.
            width: Line width in screen pixels.
            opacity: Line opacity in ``[0, 1]``.
            visible: Whether the layer is drawn when added.

        Returns:
            The registered actor, or ``None`` when Natural Earth returned no lines (see ``strict``).

        Raises:
            ValueError: if ``resolution`` is not one of ``"110m"``, ``"50m"`` or ``"10m"``.
        """
        return self._reference_lines(
            "coastline",
            "coastlines",
            resolution,
            name=name,
            color=color,
            width=width,
            opacity=opacity,
            visible=visible,
        )

    def borders(
        self,
        resolution: str = "110m",
        *,
        name: Any = None,
        color: str = "#777777",
        width: float = 1.0,
        opacity: float = 1.0,
        visible: bool = True,
    ) -> Any:
        """Draw Natural-Earth country borders as reference lines in the scene (TD-5b, #205).

        The shoreline counterpart :meth:`coastlines` for country boundaries — the same line path over the
        ``borders`` Natural-Earth dataset.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: Layer id; ``None`` numbers it ``borders``.
            color: Line colour.
            width: Line width in screen pixels.
            opacity: Line opacity in ``[0, 1]``.
            visible: Whether the layer is drawn when added.

        Returns:
            The registered actor, or ``None`` when Natural Earth returned no lines (see ``strict``).

        Raises:
            ValueError: if ``resolution`` is not one of ``"110m"``, ``"50m"`` or ``"10m"``.
        """
        return self._reference_lines(
            "borders",
            "borders",
            resolution,
            name=name,
            color=color,
            width=width,
            opacity=opacity,
            visible=visible,
        )

    def _reference_lines(
        self,
        dataset: str,
        prefix: str,
        resolution: str,
        *,
        name: Any,
        color: str,
        width: float,
        opacity: float,
        visible: bool,
    ) -> Any:
        """Register one Natural-Earth line layer — a coastline or a border.

        Args:
            dataset: The Natural-Earth dataset name cleopatra takes — ``"coastline"`` or ``"borders"``.
            prefix: The id the layer is numbered under when ``name`` is ``None``.
            resolution: One of ``"110m"``, ``"50m"`` or ``"10m"``.
            name: The caller's layer id, or ``None`` to use ``prefix``.
            color: Line colour.
            width: Line width in screen pixels.
            opacity: Line opacity in ``[0, 1]``.
            visible: Whether the layer is drawn when added.

        Returns:
            The registered actor, or ``None`` when Natural Earth returned no lines.

        Raises:
            ValueError: for an unknown resolution.
        """
        if resolution not in _NATURAL_EARTH_RESOLUTIONS:
            raise ValueError(
                f"{prefix}() resolution={resolution!r} must be one of "
                f"{sorted(_NATURAL_EARTH_RESOLUTIONS)} — the resolutions Natural Earth publishes"
            )
        return self._add_described_layer(
            kind="reference_lines",
            data=None,
            name=name if name is not None else prefix,
            visible=visible,
            dataset=dataset,
            resolution=resolution,
            color=color,
            width=width,
            opacity=opacity,
        )

    def land(
        self,
        resolution: str = "110m",
        *,
        name: Any = None,
        color: str = "#e9e4d8",
        opacity: float = 1.0,
        visible: bool = True,
    ) -> Any:
        """Fill Natural-Earth land polygons in the scene (TD-5b, #205).

        The fill counterpart of :meth:`coastlines`: where the shoreline is a line, this paints the land. The
        polygons are **hole-aware** — cleopatra 0.42.0's ``natural_earth_polygons`` returns each part as
        ``[exterior, *holes]``, so an inland lake is carved out of the fill rather than painted over
        (cleopatra#384). They are drawn as flat caps on the ground plane (``z=0``) from the same Natural-Earth
        source the static and web tiers read, so a figure holding the layer is storable with no in-memory
        source. The polygons are geographic (EPSG:4326); over a projected display CRS they are drawn but
        warned about, because reprojecting them is pyramids' job (as for :meth:`coastlines`).

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: Layer id; ``None`` numbers it ``land``.
            color: Fill colour.
            opacity: Fill opacity in ``[0, 1]``.
            visible: Whether the layer is drawn when added.

        Returns:
            The registered actor, or ``None`` when Natural Earth returned no polygon (see ``strict``).

        Raises:
            ValueError: if ``resolution`` is not one of ``"110m"``, ``"50m"`` or ``"10m"``.
        """
        return self._reference_fill(
            "land",
            "land",
            resolution,
            name=name,
            color=color,
            opacity=opacity,
            visible=visible,
        )

    def ocean(
        self,
        resolution: str = "110m",
        *,
        name: Any = None,
        color: str = "#aad3df",
        opacity: float = 1.0,
        visible: bool = True,
    ) -> Any:
        """Fill Natural-Earth ocean polygons in the scene (TD-5b, #205).

        The same hole-aware fill path as :meth:`land`, for the ocean — the polygon that carries one
        continent-shaped hole per landmass, so the hole-aware ``natural_earth_polygons`` matters most here.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: Layer id; ``None`` numbers it ``ocean``.
            color: Fill colour.
            opacity: Fill opacity in ``[0, 1]``.
            visible: Whether the layer is drawn when added.

        Returns:
            The registered actor, or ``None`` when Natural Earth returned no polygon (see ``strict``).

        Raises:
            ValueError: if ``resolution`` is not one of ``"110m"``, ``"50m"`` or ``"10m"``.
        """
        return self._reference_fill(
            "ocean",
            "ocean",
            resolution,
            name=name,
            color=color,
            opacity=opacity,
            visible=visible,
        )

    def lakes(
        self,
        resolution: str = "110m",
        *,
        name: Any = None,
        color: str = "#aad3df",
        opacity: float = 1.0,
        visible: bool = True,
    ) -> Any:
        """Fill Natural-Earth lake polygons in the scene (TD-5b, #205).

        The same hole-aware fill path as :meth:`land` and :meth:`ocean`, for inland water.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: Layer id; ``None`` numbers it ``lakes``.
            color: Fill colour.
            opacity: Fill opacity in ``[0, 1]``.
            visible: Whether the layer is drawn when added.

        Returns:
            The registered actor, or ``None`` when Natural Earth returned no polygon (see ``strict``).

        Raises:
            ValueError: if ``resolution`` is not one of ``"110m"``, ``"50m"`` or ``"10m"``.
        """
        return self._reference_fill(
            "lakes",
            "lakes",
            resolution,
            name=name,
            color=color,
            opacity=opacity,
            visible=visible,
        )

    def _reference_fill(
        self,
        dataset: str,
        prefix: str,
        resolution: str,
        *,
        name: Any,
        color: str,
        opacity: float,
        visible: bool,
    ) -> Any:
        """Register one Natural-Earth polygon fill — land, ocean or lakes.

        :meth:`land`, :meth:`ocean` and :meth:`lakes` are one mechanism over three Natural-Earth datasets, so
        the body lives here; the geometry is read hole-aware and triangulated by :func:`draw_reference_fill`.

        Args:
            dataset: The Natural-Earth polygon dataset — ``"land"``, ``"ocean"`` or ``"lakes"``.
            prefix: The id the layer is numbered under when ``name`` is ``None``.
            resolution: One of ``"110m"``, ``"50m"`` or ``"10m"``.
            name: The caller's layer id, or ``None`` to use ``prefix``.
            color: Fill colour.
            opacity: Fill opacity in ``[0, 1]``.
            visible: Whether the layer is drawn when added.

        Returns:
            The registered actor, or ``None`` when Natural Earth returned no polygon.

        Raises:
            ValueError: for an unknown resolution.
        """
        if resolution not in _NATURAL_EARTH_RESOLUTIONS:
            raise ValueError(
                f"{prefix}() resolution={resolution!r} must be one of "
                f"{sorted(_NATURAL_EARTH_RESOLUTIONS)} — the resolutions Natural Earth publishes"
            )
        return self._add_described_layer(
            kind="reference_fill",
            data=None,
            name=name if name is not None else prefix,
            visible=visible,
            dataset=dataset,
            resolution=resolution,
            color=color,
            opacity=opacity,
        )
