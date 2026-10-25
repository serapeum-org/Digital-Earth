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
  which :meth:`~digitalearth.three_d.core.base.Scene3DBase._dress_plotter` calls for the same reason it applies the
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
in :mod:`digitalearth.three_d.legends.legend` as `colorbar()` and `legend()` (order 24), under those spellings and no
others — a differently shaped `scalar_bar()` here would put two spellings on one concept, which is the single
thing the Core contract exists to prevent. A north arrow *is* built, as :meth:`DecorationMixin.north_arrow`: a
camera-aware corner marker (not a fixed on-screen arrow, since a scene can be looked at from any direction), the
single-direction companion to the orientation triad — see its method docstring.
"""

import logging
import math
from contextlib import suppress
from dataclasses import replace as with_fields
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from digitalearth.base.crs import is_geographic
from digitalearth.base.spec import LayerSpec
from digitalearth.base.spec._serial import crs_to_json
from digitalearth.base.spec.bounds import same_crs
from digitalearth.three_d.core.layer import drawing_props

logger = logging.getLogger(__name__)

#: The Natural-Earth resolutions :mod:`cleopatra.basemap.reference` publishes — the spelling the static and
#: web tiers' coastlines/borders use too, so one resolution means one dataset on every tier.
_NATURAL_EARTH_RESOLUTIONS: frozenset[str] = frozenset({"110m", "50m", "10m"})

__all__ = [
    "DecorationMixin",
    "axis_labels",
    "axis_titles",
    "draw_basemap",
    "draw_graticule",
    "draw_labels",
    "draw_reference_fill",
    "draw_reference_lines",
    "draw_text",
    "fetch_basemap_image",
    "redraw_decoration",
]

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.core.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object

#: The corner annotation index PyVista's `add_title` writes into (`vtkCornerAnnotation`'s upper edge), and the
#: one a subtitle is written under it at. Named because reading a title back off the plotter is how a test
#: proves the title landed, and the number is meaningless on its own.
TITLE_SLOT: int = 7

#: The actor name PyVista files a subtitle under, so a second `set_title` replaces it rather than stacking.
SUBTITLE_ACTOR: str = "digitalearth-subtitle"

#: The actor names the corner furniture is filed under, so a second call replaces rather than stacks it — the
#: same discipline as `SUBTITLE_ACTOR`.
ATTRIBUTION_ACTOR: str = "digitalearth-attribution"
SCALE_BAR_ACTOR: str = "digitalearth-scale-bar"

#: The four panel anchors (contract C, :data:`~digitalearth.base.registry.FURNITURE_ANCHORS`) mapped to the
#: corner names :meth:`pyvista.Plotter.add_text` places text at. One map, so attribution and any later corner
#: text agree on where "bottom-right" is.
_ANCHOR_TO_TEXT_POSITION: dict[str, str] = {
    "top-left": "upper_left",
    "top-right": "upper_right",
    "bottom-left": "lower_left",
    "bottom-right": "lower_right",
}

#: The same four anchors as a normalised ``(xmin, ymin, xmax, ymax)`` viewport box, for the orientation-marker
#: widget the north arrow is drawn as — which is placed by viewport fraction rather than by a corner name.
_ANCHOR_TO_VIEWPORT: dict[str, tuple[float, float, float, float]] = {
    "top-left": (0.0, 0.8, 0.2, 1.0),
    "top-right": (0.8, 0.8, 1.0, 1.0),
    "bottom-left": (0.0, 0.0, 0.2, 0.2),
    "bottom-right": (0.8, 0.0, 1.0, 0.2),
}

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
            >>> from digitalearth.three_d.decoration.decoration import axis_titles
            >>> axis_titles(32618)
            ('Easting', 'Northing', 'Elevation')

            ```
        - A geographic CRS is labelled in its own terms, east first however the CRS orders its axes:
            ```python
            >>> from digitalearth.three_d.decoration.decoration import axis_titles
            >>> axis_titles(4326)
            ('Geodetic longitude', 'Geodetic latitude', 'Elevation')

            ```
        - A scene with no CRS is labelled with the survey words rather than with array indices:
            ```python
            >>> from digitalearth.three_d.decoration.decoration import axis_titles
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
            >>> from digitalearth.three_d.decoration.decoration import axis_labels
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
            >>> from digitalearth.three_d.decoration.decoration import _finite
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


def _draw_attribution(plotter: Any, style: dict[str, Any]) -> None:
    """Put a credit line in a corner of the render window.

    Args:
        plotter: The plotter to draw on.
        style: The `text`, `anchor`, `font_size` and `color` :meth:`DecorationMixin.attribution` recorded. It
            is filed under :data:`ATTRIBUTION_ACTOR`, so a second call replaces the line rather than stacking a
            second one — the same discipline the subtitle follows.
    """
    plotter.add_text(
        style["text"],
        position=_ANCHOR_TO_TEXT_POSITION[style["anchor"]],
        font_size=int(style["font_size"]),
        color=style.get("color"),
        name=ATTRIBUTION_ACTOR,
    )


def _draw_scale_bar(plotter: Any, _style: dict[str, Any]) -> None:
    """Put a ground-distance scale bar on the plotter, as VTK's legend-scale ruler.

    A scale bar is honest only under a **parallel (orthographic)** projection: in perspective, one screen
    distance is a different ground distance at every depth, which is why this tier declared a scale bar absent.
    So this turns parallel projection on — the mode the bar is true in — rather than draw a bar that lies, and
    adds a :class:`vtkLegendScaleActor` configured to label the bottom edge in distance units of the scene's
    coordinates. Filed under :data:`SCALE_BAR_ACTOR`, so a second call replaces it.

    Args:
        plotter: The plotter to draw on.
        _style: The keywords :meth:`DecorationMixin.scale_bar` recorded; none is read today (the actor labels
            itself from the view), so the slot is kept for a later option rather than read.
    """
    # Orthographic is the only projection a single screen-to-ground scale is true in, so switch it on here.
    if hasattr(plotter, "enable_parallel_projection"):
        plotter.enable_parallel_projection()
    from vtkmodules.vtkRenderingAnnotation import vtkLegendScaleActor

    actor = vtkLegendScaleActor()
    # Labels as distances along one edge, not as x/y coordinate values at every edge — the scale-bar reading.
    actor.SetLabelModeToDistance()
    actor.TopAxisVisibilityOff()
    actor.LeftAxisVisibilityOff()
    actor.RightAxisVisibilityOff()
    actor.BottomAxisVisibilityOn()
    actor.LegendVisibilityOn()
    plotter.add_actor(actor, name=SCALE_BAR_ACTOR, render=False)


def _draw_north_arrow(scene: Any, plotter: Any, style: dict[str, Any]) -> None:
    """Put a camera-aware north arrow in a corner, as an orientation-marker widget.

    The tier declared a north arrow absent because a 3-D scene can be looked at from any direction, so a fixed
    on-screen arrow would point the wrong way. This is the answer that holds: an arrow pointing along ``+y`` —
    grid north in the scene's display CRS (true north too for a north-up CRS such as EPSG:4326/3857) — shown in
    a corner viewport that **turns with the camera**, so it reads correctly from every viewpoint. It is the
    single-direction companion to the orientation triad.

    Held on the scene (``_north_arrow_widget``) because an orientation widget is interactive machinery VTK
    drops the moment nothing references it; a second call turns the first off so a marker does not stack.

    Args:
        scene: The scene, which holds the widget so it is not garbage-collected.
        plotter: The plotter to draw on.
        style: The `anchor` and `color` :meth:`DecorationMixin.north_arrow` recorded.
    """
    import pyvista as pv

    existing = getattr(scene, "_north_arrow_widget", None)
    if existing is not None:
        # A stand-in plotter's widget need not implement VTK's toggle; turning the old one off is best-effort.
        with suppress(Exception):
            existing.EnabledOff()
    arrow = pv.Arrow(
        start=(0.0, -1.0, 0.0),
        direction=(0.0, 1.0, 0.0),
        tip_length=0.35,
        tip_radius=0.12,
        shaft_radius=0.04,
    )
    mapper = pv.DataSetMapper(arrow)
    actor = pv.Actor(mapper=mapper)
    with suppress(Exception):
        actor.prop.color = style.get("color") or "#d62728"
    scene._north_arrow_widget = plotter.add_orientation_widget(
        actor, interactive=False, viewport=_ANCHOR_TO_VIEWPORT[style["anchor"]]
    )


def fetch_basemap_image(
    bbox_4326: tuple[float, float, float, float], provider: Any, zoom: Any
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Return a stitched web-map image and its EPSG:3857 extent for a lon/lat box, through cleopatra.

    The headless seam :func:`draw_basemap` fetches tiles through. It composes cleopatra's tile helpers —
    `get_provider`, `auto_zoom`, the bbox→tiles enumerator, `fetch_tiles` and `stitch_tiles` — which is **image
    I/O, not GIS** (the tier's one carve-out, as for the Natural-Earth reference layers). Isolated as one
    function so a test can replace it with a synthetic image and never touch the network, exactly as
    `FeatureCollection._fetch_vectortileserver_tile` is isolated upstream.

    Note:
        cleopatra exposes tile fetching only bound to a matplotlib axes (`add_tiles`); there is no public
        "give me the stitched image for this box" call, so this reaches one private helper
        (`_tiles_for_bbox`). A public headless-stitch entry point is tracked upstream as cleopatra#401; the
        swap to make here once it lands is tracked as Digital-Earth#483. Until then this seam keeps the private
        reach isolated to this one call.

    Args:
        bbox_4326: The area to cover, ``(west, south, east, north)`` in EPSG:4326 degrees.
        provider: A cleopatra/xyzservices provider, or a dotted provider name such as ``"CartoDB.Positron"``.
        zoom: The tile zoom level, or ``None`` to derive one from the box with cleopatra's `auto_zoom`.

    Returns:
        The stitched RGBA image ``(H, W, 4)`` and its ``(west, south, east, north)`` extent in EPSG:3857.
    """
    from cleopatra.basemap import tiles as cleo_tiles

    resolved = cleo_tiles.get_provider(provider)
    box = tuple(float(v) for v in bbox_4326)
    level = int(zoom) if zoom is not None else cleo_tiles.auto_zoom(box)
    grid = cleo_tiles._tiles_for_bbox(box, level)
    data = cleo_tiles.fetch_tiles(grid, resolved)
    image, extent = cleo_tiles.stitch_tiles(data, grid, level)
    west, south, east, north = extent
    return np.asarray(image), (float(west), float(south), float(east), float(north))


def draw_basemap(scene: Any, _data: Any, layer: LayerSpec) -> tuple[Any, Any] | None:
    """Drape web-map tiles under the scene as a textured plane (the `basemap` kind).

    The 3-D answer to the web/static tiers' tile basemap: the stitched image from :func:`fetch_basemap_image`
    is draped onto a flat `StructuredGrid` at the layer's `z` using the same texture path `terrain(texture=)`
    uses (`_as_texture`/`_drape_uv`), so tiles sit under the data as ground cover (contract band `underlay`).

    Args:
        scene: The scene being drawn into.
        _data: The source slot every drawer takes, unread here — the image is fetched from the tile provider
            named in the layer's props, so a figure holding this layer is storable with no in-memory source.
        layer: The layer's description, whose props carry the lon/lat `bbox`, the `provider`, the `zoom`, the
            ground-plane `z` and the draw style.

    Returns:
        The `(plane, actor)` pair.
    """
    import pyvista as pv

    from digitalearth.three_d.builders.terrain import _as_texture, _drape_uv

    props = drawing_props(layer.symbology.props)
    bbox = props.pop("bbox")
    provider = props.pop("provider")
    zoom = props.pop("zoom", None)
    z = float(props.pop("z", 0.0))
    # Tiles are EPSG:3857; aligning them to a non-mercator display CRS is pyramids' reprojection job, not this
    # tier's, so draw them in their own extent and say so — the rule the reference layers follow (#205).
    if scene.display_crs is not None and not same_crs(scene.display_crs, 3857):
        logger.warning(
            "basemap tiles are in EPSG:3857 and may not align with the scene's %r display CRS; draw the "
            "scene in EPSG:3857, or wait on the pyramids reprojection path",
            scene.display_crs,
        )
    image, (west, south, east, north) = fetch_basemap_image(bbox, provider, zoom)
    xx, yy = np.meshgrid(
        np.array([west, east], dtype="float64"),
        np.array([south, north], dtype="float64"),
    )
    plane = pv.StructuredGrid(xx, yy, np.full(xx.shape, z, dtype="float64"))
    _drape_uv(plane)
    return plane, scene.plotter.add_mesh(plane, texture=_as_texture(image), **props)


def redraw_decoration(scene: Any) -> None:
    """Put the scene's own decoration onto the plotter it is about to draw on.

    The decoration this tier keeps as **figure furniture** — the panel's title, and whether an axes box or an
    orientation triad was asked for — belongs to the scene rather than to whichever plotter happens to exist.
    So it is applied to a window the scene has just been given, whether it built it or a caller handed one
    over, exactly as its vertical exaggeration and its camera are
    (:meth:`~digitalearth.three_d.core.base.Scene3DBase._dress_plotter`), and again when a figure is drawn into the
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
    if "attribution" in held:
        _draw_attribution(plotter, held["attribution"])
    if "scale_bar" in held:
        _draw_scale_bar(plotter, held["scale_bar"])
    if "north_arrow" in held:
        _draw_north_arrow(scene, plotter, held["north_arrow"])
    _apply_lighting(scene, plotter, held)


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

    from digitalearth.three_d.builders.vector import _polyline

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
    `z=0` with their holes carved (:func:`~digitalearth.three_d.builders.vector._cap_with_holes`), so an ocean fill
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

    from digitalearth.three_d.builders.vector import _cap_with_holes

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


#: The readable degree spacings a graticule's automatic step is chosen from, smallest first — the familiar
#: map intervals from half a degree to thirty. The step is the first of these that draws no more than about ten
#: lines across the wider span of the box, so the grid reads without crowding at any extent.
_GRATICULE_STEPS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0)

#: How many vertices each meridian/parallel is densified to before reprojection, so a line that is straight in
#: lon/lat is drawn as the smooth curve it becomes in a projected display CRS rather than as one chord.
_GRATICULE_DENSITY: int = 60


def _nice_graticule_step(span: float) -> float:
    """Return a readable grid spacing in degrees for a box spanning `span` degrees.

    Args:
        span: The wider of the box's two extents, in degrees.

    Returns:
        The first of :data:`_GRATICULE_STEPS` that draws at most about ten lines across `span`, or the coarsest
        one for a very wide box; the finest for a non-positive or non-finite span.
    """
    if not math.isfinite(span) or span <= 0:
        return _GRATICULE_STEPS[0]
    for step in _GRATICULE_STEPS:
        if span / step <= 10.0:
            return step
    return _GRATICULE_STEPS[-1]


def _grid_values(lo: float, hi: float, step: float) -> list[float]:
    """Return the grid-line coordinates at multiples of `step` within ``[lo, hi]``.

    Args:
        lo: The low edge of the range.
        hi: The high edge of the range.
        step: The spacing between lines.

    Returns:
        The multiples of `step` that fall within the range, low to high; empty when `step` is not a positive
        finite number or the range holds none.
    """
    if not math.isfinite(step) or step <= 0:
        return []
    start = math.ceil(lo / step) * step
    values: list[float] = []
    value = start
    # Cap the count so a tiny step over a wide box cannot spin: a graticule is a reading aid, not a raster.
    while value <= hi + 1e-9 and len(values) < 1000:
        values.append(round(value, 6))
        value += step
    return values


def _reproject_xy(
    xs: Any, ys: Any, crs: Any, display: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return `xs`/`ys` in the scene's display CRS, reprojected from `crs` through pyramids.

    Args:
        xs: The x coordinates, in `crs`.
        ys: The y coordinates, in `crs`.
        crs: What the coordinates are measured in, or `None` for "the scene's own".
        display: The scene's display CRS, or `None` when it has none yet.

    Returns:
        The `(xs, ys)` arrays unchanged when there is nothing to reproject, else the reprojected pair — the
        reprojection is pyramids' job, never this tier's (the rule the reference layers keep, #205).
    """
    xs_arr = np.asarray(xs, dtype="float64")
    ys_arr = np.asarray(ys, dtype="float64")
    if display is None or crs is None or same_crs(crs, display):
        return xs_arr, ys_arr
    from pyramids.base.crs import reproject_coordinates

    px, py = reproject_coordinates(
        xs_arr.tolist(), ys_arr.tolist(), from_crs=crs, to_crs=display
    )
    return np.asarray(px, dtype="float64"), np.asarray(py, dtype="float64")


def _graticule_box(
    scene: Any, bbox: Any, crs: Any
) -> tuple[float, float, float, float] | None:
    """Return the ``(west, south, east, north)`` box a graticule spans, in `crs`.

    Args:
        scene: The scene, whose drawn extent frames the grid when no `bbox` is given.
        bbox: An explicit ``(west, south, east, north)`` box in `crs`, or `None` to read the scene's extent.
        crs: What the box is measured in — the grid is built in this CRS, then reprojected to the display CRS.

    Returns:
        The box in `crs`, or `None` when no `bbox` was given and the scene has drawn nothing to frame. A box
        read from the plotter is in the display CRS, so its corners are reprojected back into `crs` for the
        grid values (pyramids does the reprojection).
    """
    if bbox is not None:
        west, south, east, north = (float(value) for value in bbox)
        return west, south, east, north
    plotter = getattr(scene, "_plotter", None)
    bounds = plotter.bounds if plotter is not None else None
    if not bounds:
        return None
    xmin, xmax, ymin, ymax = (
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3]),
    )
    if xmin >= xmax or ymin >= ymax:
        return None
    display = scene.display_crs
    if display is None or crs is None or same_crs(crs, display):
        return xmin, ymin, xmax, ymax
    from pyramids.base.crs import reproject_coordinates

    xs, ys = reproject_coordinates(
        [xmin, xmax, xmin, xmax],
        [ymin, ymin, ymax, ymax],
        from_crs=display,
        to_crs=crs,
    )
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _format_graticule_label(value: float, axis: str, geographic: bool) -> str:
    """Return the text a graticule line is labelled with.

    Args:
        value: The line's constant coordinate.
        axis: ``"lon"`` for a meridian, ``"lat"`` for a parallel.
        geographic: Whether the grid CRS is geographic, so a hemisphere letter is meaningful.

    Returns:
        A degree label with a hemisphere letter for a geographic CRS (``"10°E"``, ``"35°N"``), else the plain
        number of a projected grid.
    """
    if not geographic:
        return f"{value:g}"
    if axis == "lon":
        hemisphere = "E" if value >= 0 else "W"
    else:
        hemisphere = "N" if value >= 0 else "S"
    return f"{abs(value):g}°{hemisphere}"


def _draw_graticule_labels(
    scene: Any,
    lon_values: list[float],
    lat_values: list[float],
    box: tuple[float, float, float, float],
    crs: Any,
    display: Any,
    z: float,
    label_step: int,
    name: str,
) -> None:
    """Label the graticule's meridians along the south edge and its parallels along the west edge.

    Args:
        scene: The scene being drawn into.
        lon_values: The meridian coordinates, in `crs`.
        lat_values: The parallel coordinates, in `crs`.
        box: The ``(west, south, east, north)`` box the grid spans, in `crs`.
        crs: What the grid is measured in.
        display: The scene's display CRS the labels are reprojected into.
        z: The height the labels are drawn at.
        label_step: Label every `label_step`-th line, so a dense grid is not labelled on every line.
        name: The actor name the labels are filed under, so a redraw replaces rather than stacks them.
    """
    west, south, _east, _north = box
    geographic = bool(is_geographic(crs))
    points: list[list[float]] = []
    texts: list[str] = []
    for index, lon in enumerate(lon_values):
        if index % label_step:
            continue
        px, py = _reproject_xy([lon], [south], crs, display)
        points.append([float(px[0]), float(py[0]), z])
        texts.append(_format_graticule_label(lon, "lon", geographic))
    for index, lat in enumerate(lat_values):
        if index % label_step:
            continue
        px, py = _reproject_xy([west], [lat], crs, display)
        points.append([float(px[0]), float(py[0]), z])
        texts.append(_format_graticule_label(lat, "lat", geographic))
    if not points:
        return
    import pyvista as pv

    anchors = pv.PolyData(np.asarray(points, dtype="float64"))
    scene.plotter.add_point_labels(
        anchors, texts, show_points=False, font_size=10, name=name
    )


def draw_graticule(scene: Any, _data: Any, layer: LayerSpec) -> tuple[Any, Any] | None:
    """Draw the meridian/parallel grid a `graticule` layer describes (lat/lon lines + edge labels).

    The tier's answer to forge3d's graticules: constant-longitude and constant-latitude lines across the
    framed extent, **reprojected into the scene's display CRS through pyramids** so the grid aligns with the
    data (unlike the reference lines, which are drawn as-given and warned about — a graticule is generated here,
    so it is cheap to project). The coordinate work is pyramids'; this tier only draws the polylines.

    Args:
        scene: The scene being drawn into — its display CRS places the grid, its extent frames it when the
            layer named no `bbox`, and its `strict` policy decides what an empty grid does.
        _data: The source slot every drawer takes, unread here — a graticule is generated from the box and the
            step, so a figure holding one is storable with no in-memory source.
        layer: The layer's description, whose props carry the `bbox`, `step`, `crs`, line style and labelling.

    Returns:
        The `(mesh, actor)` pair for the merged grid lines, or `None` when there was no extent to draw over, or
        the box spanned less than one step, and the scene is not `strict`.

    Raises:
        OffLimbError: when there is nothing to draw and the scene is `strict`.

    Note:
        The edge labels are a second actor, filed under a name derived from the layer id so a redraw of the
        same scene replaces them. :meth:`DecorationMixin.graticule` documents that `remove_layer` takes the
        grid lines off and leaves the labels, the same auxiliary-actor shape the globe's coastline keeps.
    """
    props = drawing_props(layer.symbology.props)
    crs = props.pop("crs", 4326)
    bbox = props.pop("bbox", None)
    lon_step = props.pop("lon_step", None)
    lat_step = props.pop("lat_step", None)
    color = props.pop("color", "#888888")
    width = float(props.pop("width", 0.5))
    opacity = float(props.pop("opacity", 1.0))
    labels = bool(props.pop("labels", True))
    label_step = props.pop("label_step", None)
    z = float(props.pop("z", 0.0))
    display = scene.display_crs

    box = _graticule_box(scene, bbox, crs)
    if box is None:
        scene._skip_empty(
            layer.kind,
            "the scene has drawn nothing to frame a graticule over; pass bbox=",
        )
        return None
    west, south, east, north = box
    lon_spacing = (
        float(lon_step) if lon_step is not None else _nice_graticule_step(east - west)
    )
    lat_spacing = (
        float(lat_step) if lat_step is not None else _nice_graticule_step(north - south)
    )
    lon_values = _grid_values(west, east, lon_spacing)
    lat_values = _grid_values(south, north, lat_spacing)
    if not lon_values and not lat_values:
        scene._skip_empty(layer.kind, "the box spans less than one grid step")
        return None

    import pyvista as pv

    from digitalearth.three_d.builders.vector import _polyline

    lines = []
    for lon in lon_values:
        ys = np.linspace(south, north, _GRATICULE_DENSITY)
        px, py = _reproject_xy(np.full(_GRATICULE_DENSITY, lon), ys, crs, display)
        lines.append(_polyline(np.column_stack([px, py, np.full(len(px), z)])))
    for lat in lat_values:
        xs = np.linspace(west, east, _GRATICULE_DENSITY)
        px, py = _reproject_xy(xs, np.full(_GRATICULE_DENSITY, lat), crs, display)
        lines.append(_polyline(np.column_stack([px, py, np.full(len(px), z)])))
    merged = pv.MultiBlock(lines).combine()
    actor = scene.plotter.add_mesh(
        merged, color=color, line_width=width, opacity=opacity, **props
    )
    if labels:
        _draw_graticule_labels(
            scene,
            lon_values,
            lat_values,
            box,
            crs,
            display,
            z,
            int(label_step) if label_step else 1,
            f"{layer.id}-graticule-labels",
        )
    return merged, actor


def _apply_sun(plotter: Any, sun: dict[str, Any]) -> None:
    """Place a single distant directional light along a sun azimuth/elevation on the plotter.

    Args:
        plotter: The plotter to light.
        sun: The ``azimuth``, ``elevation`` (degrees), ``intensity`` and ``color`` :meth:`DecorationMixin.sun`
            recorded. Azimuth is measured from grid north (``+y``) clockwise; elevation from the horizon.
    """
    import pyvista as pv

    azimuth = math.radians(float(sun["azimuth"]))
    elevation = math.radians(float(sun["elevation"]))
    # x east, y north, z up — the tier's axis convention (see `_AXIS_DIRECTIONS`).
    dx = math.cos(elevation) * math.sin(azimuth)
    dy = math.cos(elevation) * math.cos(azimuth)
    dz = math.sin(elevation)
    bounds = plotter.bounds
    centre = (
        (bounds[0] + bounds[1]) / 2.0,
        (bounds[2] + bounds[3]) / 2.0,
        (bounds[4] + bounds[5]) / 2.0,
    )
    diagonal = max(
        math.dist((bounds[0], bounds[2], bounds[4]), (bounds[1], bounds[3], bounds[5])),
        1.0,
    )
    distance = diagonal * 2.0
    # A controlled directional sun: drop the default lights so the azimuth/elevation is what lights the scene,
    # then place one distant light along the sun direction (positional=False makes it directional).
    plotter.remove_all_lights()
    light = pv.Light(
        position=(
            centre[0] + dx * distance,
            centre[1] + dy * distance,
            centre[2] + dz * distance,
        ),
        focal_point=centre,
        color=sun.get("color") or "white",
    )
    light.positional = False
    light.intensity = float(sun.get("intensity", 1.0))
    plotter.add_light(light)


def _toggle_pass(
    plotter: Any, flag: str, enable_name: str, disable_name: str, on: bool
) -> None:
    """Turn a render pass (shadows / ambient occlusion) on or off, once per plotter.

    Args:
        plotter: The plotter to toggle the pass on.
        flag: The attribute the plotter records this pass's state under, so a later decoration redraw on the
            same plotter does not stack a second copy of the pass.
        enable_name: The plotter method that turns the pass on.
        disable_name: The plotter method that turns it off.
        on: Whether the pass should be on.
    """
    enable = getattr(plotter, enable_name, None)
    if enable is None:
        return
    if on and not getattr(plotter, flag, False):
        enable()
        setattr(plotter, flag, True)
    elif not on and getattr(plotter, flag, False):
        disable = getattr(plotter, disable_name, None)
        if disable is not None:
            disable()
        setattr(plotter, flag, False)


def _apply_lighting(scene: Any, plotter: Any, held: dict[str, Any]) -> None:
    """Apply the scene's sun and shadow/ambient-occlusion settings to the plotter it is drawing on.

    Lighting is **drawn view state**, re-applied to whichever plotter the scene is dressed with, exactly as the
    vertical exaggeration and the camera are — so a sun set before anything is drawn survives the render, and a
    plotter swapped in later gets it too.

    Args:
        scene: The scene whose lighting is applied (unused today beyond the signature the redraw hook shares).
        plotter: The plotter being dressed.
        held: The scene's decoration state, which carries ``sun`` / ``shadows`` / ``ssao`` when set.
    """
    sun = held.get("sun")
    if sun is not None and hasattr(plotter, "add_light"):
        _apply_sun(plotter, sun)
    _toggle_pass(
        plotter,
        "_de_shadows",
        "enable_shadows",
        "disable_shadows",
        bool(held.get("shadows")),
    )
    _toggle_pass(
        plotter, "_de_ssao", "enable_ssao", "disable_ssao", bool(held.get("ssao"))
    )


def draw_labels(scene: Any, data: Any, layer: LayerSpec) -> tuple[Any, Any] | None:
    """Draw a label per feature a `labels` layer describes, each with a readable halo.

    The tier's Core ``labels`` verb (the plural, styled companion to :func:`draw_text`): each feature in the
    collection is labelled with the text in its ``column``, at a representative point, drawn on a filled box so
    the text reads against any scene, with occlusion so labels behind geometry can be hidden. The features are
    placed in the scene's display CRS through pyramids (:meth:`~digitalearth.three_d.core.base.Scene3DBase._place`),
    exactly as the extrusion and line builders place theirs; a feature with no finite representative point or no
    ``column`` value is not labelled.

    Args:
        scene: The scene being drawn into — its display CRS places the features, its `strict` policy decides
            what an empty collection does.
        data: The layer's source object: the feature collection to label.
        layer: The layer's description, whose props carry the `column`, the text style and the halo/occlusion
            settings.

    Returns:
        The `(anchors, actor)` pair — the anchors are the one-point-per-label `PolyData` the labels hang off,
        so `mesh_of()` answers for the layer — or `None` when nothing was labelled and the scene is not
        `strict`.

    Raises:
        KeyError: when ``column`` is not a property of the features.
        OffLimbError: when nothing could be labelled and the scene is `strict`.
    """
    props = drawing_props(layer.symbology.props)
    column = props.pop("column")
    text_size = int(props.pop("text_size", 12))
    color = props.pop("color", None)
    halo = bool(props.pop("halo", True))
    halo_color = props.pop("halo_color", "white")
    occlude = bool(props.pop("occlude", True))
    z = float(props.pop("z", 0.0))
    crs = props.pop("crs", None)

    gdf = data
    # The Core `crs` keyword names what a collection declaring none is measured in — the one case pyramids
    # cannot reproject — so it is applied before `_place`, which otherwise draws a CRS-less frame as given.
    if (
        crs is not None
        and getattr(gdf, "crs", None) is None
        and hasattr(gdf, "set_crs")
    ):
        gdf = gdf.set_crs(crs)
    placed = scene._place(gdf, layer="labels")
    if column not in placed.columns:
        raise KeyError(
            f"labels(): {column!r} is not a property of the features; it has {sorted(placed.columns)}"
        )
    # A point labels at the point, a line or polygon at a representative interior point — the fallback every
    # reader of vector geometry in this package uses. geopandas does the geometry op; no shapely is imported.
    reps = placed.geometry.representative_point()
    xs = np.asarray(reps.x, dtype="float64")
    ys = np.asarray(reps.y, dtype="float64")
    values = placed[column].to_numpy()

    points: list[list[float]] = []
    kept: list[str] = []
    for x, y, value in zip(xs, ys, values):
        # A feature with no finite representative point (an empty geometry) or a missing value is not labelled,
        # the same quiet drop the static tier's labels make rather than drawing text at NaN.
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        points.append([float(x), float(y), z])
        kept.append(str(value))
    if not points:
        scene._skip_empty(
            layer.kind, f"no feature carried a geometry and a {column!r} value to label"
        )
        return None

    import pyvista as pv

    anchors = pv.PolyData(np.asarray(points, dtype="float64"))
    style: dict[str, Any] = {"font_size": text_size}
    if color is not None:
        style["text_color"] = color
    # A filled rounded box behind each label is the halo that keeps it readable against any scene. `occlude`
    # maps to PyVista's `always_visible`: when it is False, geometry in front of a label hides it.
    if halo:
        style["shape"] = "rounded_rect"
        style["shape_color"] = halo_color
        style["fill_shape"] = True
    else:
        style["shape"] = None
        style["fill_shape"] = False
    props.setdefault("show_points", False)
    drawn = scene.plotter.add_point_labels(
        anchors, kept, always_visible=not occlude, **style, **props
    )
    return anchors, drawn


class DecorationMixin(_MixinBase):
    """Title, labels and the coordinate frame for a :class:`~digitalearth.three_d.scene3d.Scene3D`.

    Composed into `Scene3D` beside the capability mixins, so the composed class stays a thin composition and
    the methods live here. Every method returns `self`, so decoration reads as one expression:
    `scene.terrain(dem).set_title("Elevation").axes().orientation_axes()`.

    Note:
        The class is declared against :class:`~digitalearth.three_d.core.base.Scene3DBase` under `TYPE_CHECKING`
        only — at runtime the base is `object`, so the composed MRO is what it was before the annotation.

    See Also:
        digitalearth.three_d.scene3d.Scene3D: the composition that supplies the state these methods use.
        digitalearth.three_d.core.base.Scene3DBase: the typing-only base declared above the class.
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
        :meth:`~digitalearth.three_d.core.base.Scene3DBase.from_figure`. It is not a layer — it has no place on the
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
            digitalearth.web.decoration.decoration.DecorationMixin.set_title: the same name on the web tier.
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

        A 3-D scene has no fixed orientation — the camera decides — so the triad shows all three axes at once:
        three arrows in the corner of the window, turning with the camera, labelled for the CRS the scene
        draws in rather than `X`/`Y`/`Z` (see :func:`axis_labels`). For the one direction a map most wants
        named on its own, :meth:`north_arrow` draws a camera-aware single arrow instead.

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
            digitalearth.static.builders.decoration.DecorationMixin.text: the same name on the static tier.
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

    def attribution(
        self,
        text: str,
        *,
        anchor: str = "bottom-right",
        font_size: int = 10,
        color: Any = None,
    ) -> Self:
        """Draw a data or tile credit line in a corner of the scene.

        A rendered 3-D scene travels alone — a screenshot, an exported page — so the credit a basemap or a
        dataset carries has to be part of the picture, not a line in the notebook around it. The tier once
        declared this absent ("a caller writes one beside the image"); drawing it is less work than that was.

        It is **figure furniture, drawn**: kept on the scene and re-applied to whichever plotter it draws on
        (:func:`redraw_decoration`), exactly as the title's subtitle is, and filed under one actor name so a
        second call replaces the line rather than stacking a second one. It is not a layer — it has no place on
        the ground and `remove_layer` never touches it.

        Args:
            text: The credit line. Drawn verbatim in the chosen corner.
            anchor: Which corner — one of ``"top-left"``, ``"top-right"``, ``"bottom-left"``,
                ``"bottom-right"`` (:data:`~digitalearth.base.registry.FURNITURE_ANCHORS`). ``"bottom-right"``
                by default, where a credit line conventionally sits.
            font_size: Size of the line in points.
            color: Text colour. ``None`` leaves PyVista's theme colour.

        Returns:
            This scene, so decoration chains.

        Raises:
            TypeError: when `text` is not a string.
            ValueError: when `anchor` is not one of the four corners.

        Examples:
            - A credit line is drawn and recorded as scene decoration:
                ```python
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.attribution("© OpenStreetMap contributors")
                >>> _ = scene.screenshot()
                >>> scene.close()

                ```
        """
        if not isinstance(text, str):
            raise TypeError(
                f"attribution() needs the credit as a string; got {type(text).__name__}"
            )
        if anchor not in _ANCHOR_TO_TEXT_POSITION:
            raise ValueError(
                f"attribution() anchor={anchor!r} must be one of {sorted(_ANCHOR_TO_TEXT_POSITION)}"
            )
        self._decoration["attribution"] = {
            "text": text,
            "anchor": anchor,
            "font_size": int(font_size),
            "color": color,
        }
        redraw_decoration(self)
        return self

    def scale_bar(self) -> Self:
        """Draw a ground-distance scale bar — and switch the view to orthographic, the only mode it is true in.

        A scale bar answers "how far is this on the ground?", and that has one answer only under a **parallel
        (orthographic)** projection: in perspective, a fixed screen length is a different ground distance at
        every depth, which is why this tier declared a scale bar absent. Rather than draw a bar that lies under
        the default perspective camera, this turns parallel projection on and adds VTK's legend-scale ruler,
        which labels the bottom edge in the distance units of the scene's coordinates and updates as the view
        is zoomed. Call it after the data is added, so the ruler has a framed extent to measure.

        Figure furniture, drawn and re-applied like :meth:`attribution`; filed under one actor name so a second
        call replaces it.

        Returns:
            This scene, so decoration chains.

        Note:
            Switching to orthographic changes how the scene looks (parallel lines stop converging). That is the
            cost of an honest bar, and it **persists for as long as the scale bar is on the scene**: the bar is
            re-applied on every decoration redraw (each :meth:`attribution`/:meth:`north_arrow` call, and each
            plotter rebuild), so a perspective camera set in between is reasserted to orthographic on the next
            redraw — a scale bar read under perspective would misstate the distance. To return to a perspective
            view, do not add a scale bar; there is no per-call toggle to drop it while keeping the scene.

        Examples:
            - A scale bar is drawn over a framed scene:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(6.0), np.arange(6.0))))
                >>> _ = scene.scale_bar()
                >>> scene.close()

                ```
        """
        self._decoration["scale_bar"] = {}
        redraw_decoration(self)
        return self

    def north_arrow(self, *, anchor: str = "top-right", color: Any = None) -> Self:
        """Draw a camera-aware north arrow in a corner of the scene.

        A 3-D scene has no fixed up-is-north, because the camera decides the view — which is why this tier
        declared a north arrow absent and offers :meth:`orientation_axes` instead. This is the single-direction
        companion to that triad: an arrow pointing along ``+y`` (grid north in the display CRS — true north too
        for a north-up CRS such as EPSG:4326/3857), shown in a corner viewport that **turns with the camera**,
        so it points the right way from any viewpoint rather than lying the moment the scene is rotated.

        Figure furniture, drawn and re-applied like :meth:`attribution`; a second call turns the first arrow
        off so a marker does not stack.

        Args:
            anchor: Which corner — one of the four
                :data:`~digitalearth.base.registry.FURNITURE_ANCHORS`. ``"top-right"`` by default, where a
                north arrow conventionally sits.
            color: Arrow colour. ``None`` draws it in a default red so it reads against most scenes.

        Returns:
            This scene, so decoration chains.

        Raises:
            ValueError: when `anchor` is not one of the four corners.

        Examples:
            - A north arrow is drawn and tracks the camera:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True, crs=4326)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(6.0), np.arange(6.0))))
                >>> _ = scene.north_arrow()
                >>> _ = scene.screenshot()
                >>> scene.close()

                ```
        """
        if anchor not in _ANCHOR_TO_VIEWPORT:
            raise ValueError(
                f"north_arrow() anchor={anchor!r} must be one of {sorted(_ANCHOR_TO_VIEWPORT)}"
            )
        self._decoration["north_arrow"] = {"anchor": anchor, "color": color}
        redraw_decoration(self)
        return self

    def sun(
        self,
        *,
        azimuth: float,
        elevation: float,
        intensity: float = 1.0,
        color: Any = None,
    ) -> Self:
        """Light the scene with a directional sun at a given azimuth and elevation (forge3d parity, #493).

        The rasterization-grade half of forge3d's lighting: a single distant directional light, placed along a
        compass azimuth and an elevation above the horizon, which is what gives relief its shading. It is
        **not** a physically-based sun and there is no path tracing here — VTK is a rasterizer — but paired with
        :meth:`shadows` and :meth:`ambient_occlusion` it reads as real relief rather than flat colour.

        Lighting is **drawn view state**, like the vertical exaggeration and the camera: it is re-applied to
        whichever plotter the scene is dressed with, so a sun set before anything is drawn survives the render
        and a plotter swapped in later gets it. Setting a sun drops the default lights, so the azimuth and
        elevation are what light the scene.

        Args:
            azimuth: Compass bearing of the sun in degrees, measured from grid north (``+y``) clockwise — ``90``
                is due east, ``180`` due south.
            elevation: Height of the sun above the horizon in degrees — ``0`` grazing, ``90`` overhead.
            intensity: Brightness of the light; ``1.0`` is PyVista's default.
            color: Light colour. ``None`` is white.

        Returns:
            This scene, so decoration chains.

        Raises:
            ValueError: when `azimuth`, `elevation` or `intensity` is not a finite number.

        Examples:
            - A low morning sun from the east over relief:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(6.0), np.arange(6.0))))
                >>> _ = scene.sun(azimuth=90.0, elevation=20.0)
                >>> _ = scene.screenshot()
                >>> scene.close()

                ```
        """
        call = "Scene3D.sun()"
        self._decoration["sun"] = {
            "azimuth": _finite(azimuth, "azimuth", call),
            "elevation": _finite(elevation, "elevation", call),
            "intensity": _finite(intensity, "intensity", call),
            "color": color,
        }
        redraw_decoration(self)
        return self

    def shadows(self, on: bool = True) -> Self:
        """Turn cast shadows on or off (VTK's shadow-map pass).

        Shadows need a light to cast them, so pair this with :meth:`sun`. Like the sun, it is drawn view state,
        re-applied on each plotter the scene is dressed with; ``on=False`` turns it off and keeps it off.

        Args:
            on: Whether shadows are drawn.

        Returns:
            This scene, so decoration chains.

        Examples:
            - Shadows under a sun:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(6.0), np.arange(6.0))))
                >>> _ = scene.sun(azimuth=135.0, elevation=30.0).shadows()
                >>> _ = scene.screenshot()
                >>> scene.close()

                ```
        """
        if on:
            self._decoration["shadows"] = True
        else:
            self._decoration.pop("shadows", None)
        redraw_decoration(self)
        return self

    def ambient_occlusion(self, on: bool = True) -> Self:
        """Turn screen-space ambient occlusion on or off (VTK's SSAO pass).

        Ambient occlusion darkens crevices and contact points, which reads as depth without a second light. It
        is drawn view state like :meth:`sun` and :meth:`shadows`, re-applied on each plotter; ``on=False`` turns
        it off and keeps it off.

        Args:
            on: Whether ambient occlusion is drawn.

        Returns:
            This scene, so decoration chains.

        Examples:
            - Ambient occlusion over relief:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(get_source(np.add.outer(np.arange(6.0), np.arange(6.0))))
                >>> _ = scene.ambient_occlusion()
                >>> _ = scene.screenshot()
                >>> scene.close()

                ```
        """
        if on:
            self._decoration["ssao"] = True
        else:
            self._decoration.pop("ssao", None)
        redraw_decoration(self)
        return self

    def graticule(
        self,
        *,
        lon_step: Any = None,
        lat_step: Any = None,
        spacing: Any = None,
        bbox: Any = None,
        crs: Any = 4326,
        color: Any = "#888888",
        width: float = 0.5,
        labels: bool = True,
        label_step: Any = None,
        z: float = 0.0,
        name: Any = None,
        **kwargs: Any,
    ) -> Self:
        """Draw a latitude/longitude grid across the scene, labelled along its edges.

        The tier's answer to forge3d's graticules, and the Core ``graticule`` verb: meridians (constant
        longitude, ``lon_step`` apart) and parallels (constant latitude, ``lat_step`` apart) over the framed
        extent, **reprojected into the display CRS through pyramids** so the grid lies on the data rather than
        beside it. A graticule is generated here rather than read from a file, so — unlike the Natural-Earth
        reference lines — it is reprojected rather than warned about.

        It is a **layer** of the ``graticule`` kind, not furniture: it has an id, :meth:`set_visible` hides it,
        :meth:`remove_layer` takes it off, and it is drawn again from ``figure_spec.to_dict()``.

        Args:
            lon_step: Degrees between meridians. ``None`` chooses a readable spacing for the extent
                (:func:`_nice_graticule_step`); the keyword is the Core contract's (the same name the web and
                interactive tiers use).
            lat_step: Degrees between parallels. ``None`` chooses a readable spacing.
            spacing: One step for both axes, for a square grid; overrides ``lon_step`` and ``lat_step``.
            bbox: The ``(west, south, east, north)`` box to grid, in `crs` — a 3-D extra beyond the Core
                keywords. ``None`` reads the scene's drawn extent, so draw the data first (or pass a box): a
                scene with nothing drawn has no extent to frame and the layer is skipped (or raises under
                ``strict``).
            crs: What `bbox` and the steps are measured in, and the CRS the grid is generated in before being
                reprojected to the display CRS. EPSG:4326 by default.
            color: Line colour.
            width: Line width in screen pixels (PyVista's ``line_width``).
            labels: Whether to label meridians along the south edge and parallels along the west edge.
            label_step: Label every `label_step`-th line, to keep a dense grid legible. ``None`` labels every
                line.
            z: The height the grid is drawn at, in the scene's vertical units. ``0.0`` puts it on the ground.
            name: The layer id; ``None`` numbers it ``graticule``.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_mesh`.

        Returns:
            This scene, so decoration chains.

        Raises:
            ValueError: when `bbox` is given but is not four numbers, a step or `z` is not a finite number, or
                a step is not positive.

        Note:
            The edge labels are an auxiliary actor. :meth:`remove_layer` takes the grid lines off; the labels
            are re-applied (and so replaced) on every redraw of the scene, the same shape the globe's coastline
            keeps beside its sphere.

        Examples:
            - A grid over an explicit box is a layer with an id like any other:
                ```python
                >>> from digitalearth.three_d import Scene3D
                >>> scene = Scene3D(off_screen=True, crs=4326)
                >>> _ = scene.graticule(bbox=(-10.0, 35.0, 30.0, 60.0), spacing=10.0)
                >>> scene.get_layer(scene.layer_ids[-1]).kind
                'graticule'
                >>> scene.close()

                ```
        """
        call = "Scene3D.graticule()"
        if spacing is not None:
            lon_step = lat_step = spacing
        extra: dict[str, Any] = {}
        if bbox is not None:
            edges = tuple(bbox)
            if len(edges) != 4:
                raise ValueError(
                    f"{call} bbox must be (west, south, east, north); got {bbox!r}"
                )
            extra["bbox"] = tuple(_finite(value, "bbox", call) for value in edges)
        for step_name, step_value in (("lon_step", lon_step), ("lat_step", lat_step)):
            if step_value is not None:
                resolved_step = _finite(step_value, step_name, call)
                if resolved_step <= 0:
                    raise ValueError(
                        f"{call} {step_name} must be positive; got {step_value!r}"
                    )
                extra[step_name] = resolved_step
        self._add_described_layer(
            kind="graticule",
            name=name,
            crs=crs_to_json(crs, f"{call} crs="),
            color=color,
            width=float(width),
            labels=bool(labels),
            label_step=int(label_step) if label_step else None,
            z=_finite(z, "z", call),
            **extra,
            **kwargs,
        )
        return self

    def labels(
        self,
        features: Any,
        column: str,
        *,
        text_size: float = 12.0,
        color: Any = None,
        halo: bool = True,
        halo_color: Any = "white",
        occlude: bool = True,
        z: float = 0.0,
        crs: Any = None,
        name: Any = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Label every feature with the text in ``column``, each on a readable halo (forge3d parity, #491).

        The tier's Core ``labels`` verb — "label features from a column" — and the plural, styled companion to
        :meth:`text` (which places one string at one coordinate). It is the tier's answer to forge3d's labels
        with halos and occlusion handling: each feature is labelled at a representative point, on a filled box
        that keeps the text legible against busy geometry, with occlusion so labels behind the terrain can be
        hidden. The keywords are the web and static tiers' wherever the three can mean the same thing (the Core
        contract's point — one spelling per concept).

        Like :meth:`text`, it is a **layer**: it has an id, :meth:`set_visible` hides it, :meth:`remove_layer`
        takes it off, and it is drawn again from ``figure_spec.to_dict()``. The features are placed in the
        display CRS through pyramids, exactly as a vector layer's data is.

        Args:
            features: A pyramids ``FeatureCollection`` or GeoDataFrame (or a path/URL to one, the only form a
                figure can be written down with). Points label at the point, lines and polygons at a
                representative interior point.
            column: The property to read each label's text from.
            text_size: Text size in points. Named for the text rather than ``size``, which is a marker's size
                everywhere else (contract C3).
            color: Text colour. ``None`` leaves PyVista's theme colour.
            halo: Whether each label is drawn on a filled rounded box, so it reads against busy geometry.
            halo_color: The halo's fill colour.
            occlude: Whether geometry in front of a label hides it (``True``) or labels are always drawn on top
                (``False``).
            z: The height the labels are anchored at, in the scene's vertical units. ``0.0`` is the ground.
            crs: What the features are in when they carry no CRS of their own — the one case pyramids cannot
                reproject. ``None`` leaves a self-describing collection alone (matches the static/web tiers).
            name: The layer id; ``None`` numbers it ``labels``.
            visible: Whether the layer is drawn when added.
            **kwargs: Forwarded to :meth:`pyvista.Plotter.add_point_labels` — `bold`, `italic`, `margin`, ….

        Returns:
            This scene, so decoration chains.

        Raises:
            ValueError: when `text_size` or `z` is not a finite number.
            KeyError: when `column` is not a property of the features (raised when they are placed).

        Examples:
            - Label two polygons by an attribute (needs geopandas at the call site — only ever as the type
              pyramids hands back):
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Polygon
                >>> from digitalearth.three_d import Scene3D
                >>> gdf = gpd.GeoDataFrame(
                ...     {"name": ["A", "B"]},
                ...     geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                ...               Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])],
                ...     crs=4326,
                ... )
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.labels(gdf, "name")
                >>> scene.get_layer(scene.layer_ids[-1]).kind
                'labels'
                >>> scene.close()

                ```
        """
        call = "Scene3D.labels()"
        self._add_described_layer(
            kind="labels",
            data=features,
            name=name,
            visible=visible,
            column=column,
            text_size=_finite(text_size, "text_size", call),
            color=color,
            halo=bool(halo),
            halo_color=halo_color,
            occlude=bool(occlude),
            z=_finite(z, "z", call),
            crs=crs,
            **kwargs,
        )
        return self

    def basemap(
        self,
        provider: Any = None,
        *,
        bbox: Any,
        zoom: Any = None,
        name: Any = None,
        z: float = 0.0,
        crs: Any = 4326,
        opacity: float = 1.0,
        visible: bool = True,
    ) -> Any:
        """Drape web-map tiles under the scene as ground cover (the `basemap` kind).

        The 3-D answer to the web and static tiers' tile basemap, and what this tier once declared absent
        ("no map tiles to drape under a scene drawn in three dimensions"): the tiles covering ``bbox`` are
        fetched through cleopatra (image I/O, never GIS — the tier's one carve-out, as for the Natural-Earth
        reference layers) and draped onto a flat plane at ``z`` with the same texture path
        ``terrain(texture=)`` uses. It is a layer of the ``basemap`` kind, banded ``underlay``, so it sits
        beneath the data in the layer tree with an id, visibility and removal like any other.

        The tiles are EPSG:3857 (Web Mercator). Over a scene whose display CRS is also 3857 they align; over
        another projected CRS they are drawn in their own extent and warned about, because reprojecting them is
        pyramids' job, not this tier's — the same boundary the reference lines keep (#205).

        Args:
            provider: The tile provider — a cleopatra/xyzservices provider object, or a dotted name such as
                ``"CartoDB.Positron"`` or ``"Esri.WorldImagery"``. ``None`` uses cleopatra's default
                (OpenStreetMap). First positional, as the Core `basemap` contract spells it.
            bbox: The area to cover, ``(west, south, east, north)``, in `crs`. Keyword-only and required — a
                3-D scene has no map extent of its own to default to, so the area is named explicitly.
            zoom: The tile zoom level (``0``–``19``), or ``None`` to let cleopatra pick one for the extent.
            name: The layer id; ``None`` numbers it ``basemap``.
            z: The ground-plane height the tiles are draped at, in the scene's vertical units. ``0.0`` puts
                them on the ground, under elevated data.
            crs: What `bbox` is measured in. EPSG:4326 by default; a box in another CRS is reprojected to
                EPSG:4326 (what the tile math needs) through pyramids.
            opacity: Tile opacity in ``[0, 1]`` — below 1 lets flat data drawn at the same ``z`` show through.
            visible: Whether the layer is drawn when added.

        Returns:
            The registered actor.

        Raises:
            ValueError: when `bbox` is not four numbers.

        Examples:
            - Add a basemap under a scene (needs the ``3d`` extra and network tiles; not run in doctests):
                ```python
                >>> from digitalearth.three_d import Scene3D  # doctest: +SKIP
                >>> scene = Scene3D(off_screen=True, crs=3857)  # doctest: +SKIP
                >>> _ = scene.basemap(bbox=(-10.0, 35.0, 30.0, 60.0), zoom=4)  # doctest: +SKIP
                ```
        """
        box = self._basemap_bbox_4326(bbox, crs)
        return self._add_described_layer(
            kind="basemap",
            data=None,
            name=name if name is not None else "basemap",
            visible=visible,
            bbox=box,
            provider=provider,
            zoom=zoom,
            z=float(z),
            opacity=float(opacity),
        )

    @staticmethod
    def _basemap_bbox_4326(bbox: Any, crs: Any) -> tuple[float, float, float, float]:
        """Return `bbox` as a ``(west, south, east, north)`` box in EPSG:4326, reprojecting if needed.

        Args:
            bbox: Four numbers in `crs`.
            crs: What they are measured in.

        Returns:
            The box in EPSG:4326, which is the CRS cleopatra's tile math works in. When a reprojection is
            needed it is the envelope of the four reprojected corners — exact for the axis-aligned 4326↔3857
            case, approximate (slightly inside) for a curved projection over a wide extent.

        Raises:
            ValueError: when `bbox` is not four numbers.
        """
        try:
            values = [float(v) for v in bbox]
        except (TypeError, ValueError):
            raise ValueError(
                f"basemap() bbox must be (west, south, east, north); got {bbox!r}"
            ) from None
        if len(values) != 4:
            raise ValueError(
                f"basemap() bbox must be (west, south, east, north); got {bbox!r}"
            )
        west, south, east, north = values
        if crs is None or same_crs(crs, 4326):
            return west, south, east, north
        from pyramids.base.crs import reproject_coordinates

        xs, ys = reproject_coordinates(
            [west, east, west, east],
            [south, south, north, north],
            from_crs=crs,
            to_crs=4326,
        )
        return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

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

        Reference geography was reachable only inside :meth:`~digitalearth.three_d.builders.globe.GlobeMixin.globe`,
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

        The fill is ground cover and is banded ``underlay`` (bottom of the layer tree, like the static and
        interactive ``land``/``ocean``/``lakes`` kinds). That governs the **layer-tree order** — the scene's
        layer list and switcher — not what occludes what: the renderer composites actors by depth, so a flat
        data layer coincident on the ground plane (``z=0``) will z-fight an opaque fill regardless of band.
        Give the fill ``opacity`` below 1, or treat it as the scene's sole ground, when drawing flat data over
        it; a terrain or any elevated layer sits above the ``z=0`` fill with no conflict.

        Args:
            resolution: Natural-Earth resolution — ``"110m"`` (default), ``"50m"`` or ``"10m"``.
            name: Layer id; ``None`` numbers it ``land``.
            color: Fill colour.
            opacity: Fill opacity in ``[0, 1]``. Non-finite values are refused.
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
            ValueError: for an unknown resolution, or a non-finite opacity.
        """
        if resolution not in _NATURAL_EARTH_RESOLUTIONS:
            raise ValueError(
                f"{prefix}() resolution={resolution!r} must be one of "
                f"{sorted(_NATURAL_EARTH_RESOLUTIONS)} — the resolutions Natural Earth publishes"
            )
        # Refuse a non-finite opacity at the call, as the web tier's `as_finite` does: a figure holding NaN
        # or infinity could not be written down, and the symptom would otherwise surface far from the cause.
        if not math.isfinite(opacity):
            raise ValueError(f"{prefix}() opacity={opacity!r} must be a finite number")
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
