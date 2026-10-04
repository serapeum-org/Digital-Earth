"""VectorMixin — vector data and vector-field renders.

Points (scatter/grid_points/grid_cells), polygon products (choropleth/shapes/voronoi/cartogram/quadtree),
unstructured triangulations (tricontour/tricontourf/tripcolor), kernel density, flow/Sankey, and the u/v
vector field (quiver/barbs/streamplot/quiverkey) — all wired onto the matching cleopatra glyphs.
"""

import os
from functools import wraps
from math import isfinite
from numbers import Integral
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from cleopatra.glyphs.gridded.mesh_glyph import MeshGlyph
from cleopatra.glyphs.gridded.vector_glyph import VectorGlyph
from cleopatra.glyphs.primitives.flow_glyph import FlowGlyph
from cleopatra.glyphs.primitives.polygon_glyph import PolygonGlyph
from cleopatra.glyphs.primitives.scatter_glyph import ScatterGlyph
from cleopatra.glyphs.stats.hexbin_glyph import HexbinGlyph
from cleopatra.glyphs.stats.kde_glyph import KDEGlyph
from matplotlib.colors import is_color_like, to_hex
from matplotlib.path import Path as MplPath
from matplotlib.patheffects import withStroke
from pyramids.dataset import Dataset
from shapely import MultiPoint, box, voronoi_polygons
from shapely.affinity import scale as affine_scale

from digitalearth.base.arrays import NAN_REDUCERS, read_masked_band
from digitalearth.base.crs import reproject
from digitalearth.base.points import PointArrays
from digitalearth.base.sources import get_source, require_drawable
from digitalearth.base.spec import DEFAULT_BAND, DataRef, LayerSpec, Symbology
from digitalearth.base.spec._serial import crs_to_json
from digitalearth.base.spec.bounds import same_crs
from digitalearth.base.symbology import (
    DEFAULT_LABEL_COLOR,
    DEFAULT_LABEL_HALO_COLOR,
    DEFAULT_LABEL_HALO_WIDTH,
    DEFAULT_LABEL_TEXT_SIZE,
    MISSING_COLOR,
    is_null,
    nulls_to_none,
    resolve_categorical_cmap,
)
from digitalearth.static.capabilities import CAPABILITIES
from digitalearth.static.guides import (
    COUNT_FIELD,
    DENSITY_FIELD,
    MAGNITUDE_FIELD,
    source_field,
)
from digitalearth.static.maps.base import OffLimbError
from digitalearth.static.render_compat import relocate_flat_style
from digitalearth.static.renderer import DrawnLayer
from digitalearth.static.scene import LayerRecord, drawing_style

#: Per-cell reducers accepted by ``Map.quadtree``'s ``agg`` — the shared NaN-aware registry plus a special
#: ``"count"`` (``len`` over the per-cell index array, ignoring the column).
_QUADTREE_AGG = {**NAN_REDUCERS, "count": len}

#: The reducer a quadtree uses when the caller names none, and the one a figure falls back to when it carries
#: no name because the caller passed a callable — which no reader but the scene that held it can resolve.
DEFAULT_QUADTREE_AGG = "mean"

#: The registered kind each u/v render draws (#303). ``quiver`` and ``barbs`` are two glyphs for one thing —
#: a vector field — while a streamplot integrates that field into flow lines, which is a different layer.
_VECTOR_KINDS = {"quiver": "vectors", "barbs": "vectors", "streamplot": "streamlines"}

#: How a point layer names itself in a refusal. The builder and the drawer that replays it speak for the
#: same public call, so they share the one spelling.
_POINTS_CALLER = "Map.points()"

#: The same, for the label layer. Its refusals come from both sides of the seam — the finite checks at the
#: call, the column and the CRS when the features are opened — and all of them name the one public method.
_LABELS_CALLER = "Map.labels()"

#: The same, for the line layer. Its ``width=`` and ``color=`` are checked at the call and its ``column=``
#: when the features are read, so builder and drawer need the one spelling between them.
_LINES_CALLER = "Map.lines()"

#: Keywords :meth:`VectorMixin.labels` refuses in ``**opts``, each with what to write instead. Two of them are
#: matplotlib's own spelling of something this builder already declares, and forwarding either would have the
#: declared parameter silently win; the third is a capability this tier does not have. A keyword accepted and
#: then ignored is the defect this package has now fixed twice, so each is refused by name at the call.
#:
#: ``allow_overlap`` takes its reason from the tier's own declaration rather than restating it, so the refusal
#: a caller reads and the ``Capabilities`` entry a dispatcher reads cannot drift apart.
_LABEL_REFUSED_OPTS = {
    "allow_overlap": (
        f"label collision is not a capability of this tier: {CAPABILITIES.reason('label_collision')}"
    ),
    "fontsize": "a label's text size is text_size= here, which is the spelling every tier uses",
    "size": (
        "a label's text size is text_size=; size= is the visual size of a *marker* on every other builder, "
        "and one keyword cannot mean both"
    ),
}


def _label_offset(offset: Any) -> Optional[List[float]]:
    """Return a label offset in the spelling a description carries, refusing one it could not.

    Args:
        offset: What the caller gave — ``None``, or a pair of numbers in ems.

    Returns:
        ``None`` unchanged, else the pair as a two-element list of floats. A **list**, because a figure is
        written to JSON and read back: a tuple returns as a list, and two symbologies that differ only in
        that stop comparing equal, which is what makes a redraw think the layer changed.

    Raises:
        ValueError: for anything that is not two finite numbers — the wrong count, a string (which is iterable
            and would otherwise be read character by character), or a bound a figure could not be written with.
    """
    if offset is None:
        return None
    if isinstance(offset, (str, bytes)):
        raise ValueError(
            f"{_LABELS_CALLER} needs offset= as two numbers in ems, (x, y); got the string {offset!r}"
        )
    try:
        pair = list(offset)
    except TypeError:
        raise ValueError(
            f"{_LABELS_CALLER} needs offset= as two numbers in ems, (x, y); got {offset!r}"
        ) from None
    if len(pair) != 2:
        raise ValueError(
            f"{_LABELS_CALLER} needs offset= as two numbers in ems, (x, y); got {len(pair)} of them "
            f"({offset!r})"
        )
    return [_as_finite(value, "offset", _LABELS_CALLER) for value in pair]


def _as_finite(value: Any, argument: str, caller: str) -> float:
    """Return a builder's numeric keyword as a plain float, refusing one a figure could not be written with.

    A label records its text size and halo width in the layer's description, and a description holding NaN or
    infinity cannot be written at all — ``FigureSpec.to_dict`` refuses the whole figure. Left to be caught
    there, the refusal names a property path long after the call, for an argument the caller wrote as
    ``text_size=``.

    Args:
        value: The keyword's value, as given.
        argument: The keyword's name, as the caller spells it.
        caller: The public call, for the message.

    Returns:
        The value as a `float`.

    Raises:
        ValueError: when `value` is NaN, infinite, or not a number at all — the conversion's own failure
            re-raised in the builder's words, since an argument named in the signature should not surface as a
            bare ``float()`` error several frames down.

    Note:
        ``digitalearth.web.base.as_finite`` is the same helper for the same reason. The two are not shared
        because ``base/`` has no such home yet and a tier may not import another tier; giving them one is the
        obvious follow-up, and until then this docstring is the pointer between them.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{caller} needs {argument}= as a finite number; got {value!r}"
        ) from None
    if not isfinite(number):
        raise ValueError(
            f"{caller} needs {argument}= as a finite number; got {value!r}. A figure holding it could not be "
            "written down: JSON has no spelling for NaN or infinity"
        )
    return number


def _as_count(value: Any, argument: str, caller: str, *, minimum: int = 1) -> int:
    """Return a builder's whole-number keyword as a plain int, refusing one the engine would fail on.

    The counterpart of :func:`_as_finite` for an argument that counts things rather than measures them — a
    lattice's cells across, a floor on the points in a cell. Left unchecked these reach matplotlib's
    ``hexbin`` and come back as arithmetic: ``gridsize=0`` as ``ZeroDivisionError: float division by
    zero``, ``gridsize=None`` as ``unsupported operand type(s) for /: 'NoneType' and 'float'``,
    ``gridsize=-5`` as ``could not broadcast input array from shape (0,) into shape (4,)`` — none of which
    names the method or the argument the caller actually wrote.

    Args:
        value: The keyword's value, as given.
        argument: The keyword's name, as the caller spells it.
        caller: The public call, for the message.
        minimum: The smallest value the argument admits.

    Returns:
        The value as an `int`.

    Raises:
        ValueError: when `value` is not a whole number, or is below `minimum`. ``bool`` is excluded
            although it is an `Integral`: ``gridsize=True`` is a mistake, not a lattice one cell across.
    """
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < minimum
    ):
        raise ValueError(
            f"{caller} needs {argument}= as a whole number >= {minimum}; got {value!r}"
        )
    return int(value)


def _hexbin_lattice(gridsize: Any, min_count: Any) -> Tuple[Any, Optional[int]]:
    """Refuse a lattice ``HexbinGlyph`` could not bin onto, before the figure is touched.

    Args:
        gridsize: Hexagons across the window — one count, or an ``(nx, ny)`` pair.
        min_count: The floor on a cell's points, or ``None`` for the builder's own default.

    Returns:
        ``(gridsize, min_count)`` as whole numbers — ``gridsize`` an int for the one-count form and a
        pair of ints for the other, so the layer's description carries the shape it was given.

    Raises:
        ValueError: naming ``hexbin()`` and the argument that is wrong, with the value passed.
    """
    sides = list(gridsize) if isinstance(gridsize, (tuple, list)) else [gridsize]
    if len(sides) not in (1, 2):
        raise ValueError(
            "hexbin() needs gridsize= as one whole number, or a pair of them (nx, ny); "
            f"got {gridsize!r}"
        )
    counts = [_as_count(side, "gridsize", "hexbin()") for side in sides]
    floor = (
        None
        if min_count is None
        else _as_count(min_count, "min_count", "hexbin()", minimum=0)
    )
    return (counts[0] if len(counts) == 1 else tuple(counts)), floor


def _numeric_column(gdf: Any, column: str, argument: str, caller: str) -> np.ndarray:
    """Read one feature column as floats, refusing one that holds something a ramp cannot read.

    A colour ramp and a width scale both need numbers. Read untyped — ``gdf[column].to_numpy()``, which
    ``sankey`` does and ``lines`` inherited — a text column travels as ``object`` and fails several frames
    down in matplotlib as ``TypeError: ufunc 'isfinite' not supported for the input types``, naming neither
    the method, the keyword, nor the column.

    Args:
        gdf: The features, already in the display CRS.
        column: The column named by the caller.
        argument: The keyword it was named under — ``column`` or ``width``.
        caller: The public call, for the message.

    Returns:
        The column's values as a float array.

    Raises:
        ValueError: when the column holds values that are not numbers, naming all three.
    """
    try:
        return gdf[column].to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"{caller} needs {argument}={column!r} to name a column of numbers, and it holds "
            f"{gdf[column].dtype} values that are not numbers"
        ) from None


def _polygon_kind(fill: Any) -> str:
    """Return the kind a polygon layer is recorded under, given what colours it.

    Args:
        fill: Whatever decides the polygons' colour — the column a builder was handed, the band a raster's
            cells are read from, the per-polygon values themselves. `None` means nothing colours them and
            only their outlines are drawn.

    Returns:
        `"choropleth"` when the polygons are coloured by a value, `"polygons"` when only their outlines are
        drawn. Both come out of the same :meth:`VectorMixin._polygon_layer` recipe and the same cleopatra
        glyph, so the drawing cannot tell them apart — the fill is what makes one a choropleth, which is
        why the builder answers this from its arguments rather than from what it drew.

    Examples:
        - A filled and an outline-only layer are different kinds:
            ```python
            >>> from digitalearth.static.maps.vector import _polygon_kind
            >>> _polygon_kind([1.0, 2.0]), _polygon_kind(None)
            ('choropleth', 'polygons')

            ```
    """
    return "polygons" if fill is None else "choropleth"


def _draw_missing_neutral(artist: Any) -> None:
    """Colour a classified layer's missing values neutral instead of invisible.

    cleopatra maps a missing category to ``NaN`` in the class codes, and a colormap's default "bad" colour is
    fully transparent — so a feature whose attribute is missing is drawn as *nothing*, making it
    indistinguishable from a feature that was never in the collection. The same is true of a **graduated**
    scheme, whose ``NaN`` never falls in a class either: the 3-D tier lifts it onto
    :data:`~digitalearth.base.symbology.MISSING_COLOR` (``classified_scalars``), and this does the same here,
    so one classified column reads the same way whichever tier draws it. A *continuous* ramp is left alone —
    it has no classes, and nothing on any tier repaints it.

    ``with_extremes`` returns a *new* colormap rather than mutating in place, which matters: the glyph's may be
    a colormap registered globally under a name, and setting the bad colour on that instance would leak this
    policy into every other plot in the process.

    Args:
        artist: The rendered mappable (a ``PolyCollection``) whose colormap gets the neutral "bad" colour.
    """
    artist.set_cmap(artist.get_cmap().with_extremes(bad=MISSING_COLOR))


def _skips_off_limb(builder: Callable) -> Callable:
    """Wrap a vector builder so data the display CRS cannot place skips the layer instead of raising (#257).

    The builder itself no longer reprojects: it records the layer, and the drawer that replays the record
    does the warp (in :meth:`VectorMixin._vector_input`) and the drawing. So the `OffLimbError` comes up
    out of the drawer, through :meth:`~digitalearth.static.scene.Scene._draw` — which has already dropped
    the layer from the description — and is answered here, one wrapper for the whole call. The raster
    builders answer the same error inline (``except OffLimbError: self._skipped_off_limb(kind)``), because
    each of them reaches its drawer at its own point in its own body.

    Args:
        builder: A vector layer method whose drawer reprojects through
            :func:`~digitalearth.base.crs.reproject`.

    Returns:
        The same method, with an off-limb reprojection turned into a skipped layer (``None``) — or, under
        ``strict=True``, into an :class:`~digitalearth.base.crs.OffLimbError` naming the layer.
    """

    @wraps(builder)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the builder, answering an off-limb reprojection with a skipped layer.

        Args:
            self: The composed ``Map`` the builder is bound to.
            *args: The builder's positional arguments.
            **kwargs: The builder's keyword arguments.

        Returns:
            Whatever the builder returns, or ``None`` when the layer was skipped.

        Raises:
            OffLimbError: when the map was built with ``strict=True``.
        """
        try:
            return builder(self, *args, **kwargs)
        except OffLimbError:
            self._skipped_off_limb(builder.__name__)
            return None

    return guarded


def draw_scatter(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Draw the point layer a described scatter asks for, through ``cleopatra.ScatterGlyph``.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the scatter ``PathCollection``.

    Raises:
        OffLimbError: when the warp places none of the geometry in the display CRS.
        ValueError: when the collection is empty.
    """
    opts = drawing_style(scene, layer)
    # empty-guard; any geometry (centroid fallback) OK
    fc = scene._vector_input(data, name="points")
    src = get_source(fc)
    size_column = layer.symbology.props.get("size_column")
    sizes = (
        np.asarray(fc[size_column], dtype=float) if size_column is not None else None
    )
    opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
    # scheme/k -> plot() classify group; the `size` channel -> the glyph's own `point_size`.
    plot_style = relocate_flat_style(opts, folds_marker_size=True)
    glyph = ScatterGlyph(
        src.x.values,
        src.y.values,
        values=src.z.values if src.z is not None else None,
        sizes=sizes,
        ax=scene.ax,
        fig=scene.fig,
        **opts,
    )
    drawn = scene._render_glyph(glyph, artist="plot", **plot_style)
    # The numeric column `get_source` read the values from, when it found one. A collection with none draws
    # uniform markers, and the glyph then carries no value array, so no colour encoding is published for it.
    return drawn.colored_by(source_field(src))


def _labelled_features(features: Any, crs: Any) -> Any:
    """Return the features with a CRS to reproject from, settling ``labels(crs=)`` against their own.

    ``crs=`` is the one Core keyword for ``labels`` whose meaning differs per tier, because what it could
    possibly name differs: the web tier places labels in EPSG:4326 and so has no use for it at all, while here
    the coordinates come out of the geometry and pyramids reprojects them from whatever the collection
    declares. So it has exactly one job — to say what a collection declaring **nothing** is measured in, which
    is the case ``to_crs`` refuses outright ("Cannot transform naive geometries"). A naive frame is an ordinary
    thing to hold: coordinates read out of a CSV, or a frame built inline.

    It is never quietly ignored. Against a collection that declares its own CRS it either agrees — the same
    system, however spelled — or it is refused, because two different answers to "where is this" cannot both
    be acted on.

    Args:
        features: The collection the labels are read from, as the layer recorded it.
        crs: What the call named, in the description's spelling, or `None`.

    Returns:
        `features` itself when it declares a CRS, and a copy carrying `crs` when it does not. ``set_crs``
        returns a new frame rather than mutating, so the caller's own object is never re-labelled underneath
        them.

    Raises:
        ValueError: when the collection declares no CRS and the call named none, or when the two name
            different systems.
    """
    own = getattr(features, "crs", None)
    if own is None:
        if crs is None:
            raise ValueError(
                f"{_LABELS_CALLER} cannot place features that declare no CRS: name the one their coordinates "
                "are measured in with crs=, or set one on the collection"
            )
        return features.set_crs(crs)
    if crs is not None and not same_crs(own, crs):
        # In the description's spelling rather than `repr`: a pyproj CRS reprs as its whole multi-line WKT
        # block, which buries the one fact the message is about.
        declared = crs_to_json(own, "the features' CRS")
        raise ValueError(
            f"{_LABELS_CALLER} was given crs={crs!r} for features that already declare a different CRS "
            f"({declared!r}); drop crs= to use theirs, or reproject the collection first"
        )
    return features


def _label_text(features: Any, column: str) -> np.ndarray:
    """Return one label string per feature, refusing a column the features do not carry.

    Args:
        features: The collection, already placed.
        column: The attribute the labels are read from.

    Returns:
        An object array of strings, index-aligned with the features so
        :meth:`~digitalearth.base.points.PointArrays.finite` can filter it alongside the coordinates. A
        **missing** value becomes the empty string, which the drawer then draws no artist for: a null is
        absent data rather than the text ``"None"``, and this is the same reading a MapLibre expression gives
        a missing property.

    Raises:
        KeyError: naming the column and listing the ones the features do have — the message the web tier's
            `labels` already gives, because an expression over a column nobody has renders an empty map with
            nothing to explain it.
    """
    columns = list(getattr(features, "columns", []))
    if column not in columns:
        geometry = getattr(getattr(features, "geometry", None), "name", "geometry")
        raise KeyError(
            f"{_LABELS_CALLER}: labels(column={column!r}) is not a property of these features; available: "
            f"{sorted(name for name in columns if name != geometry)}"
        )
    return np.asarray(
        ["" if is_null(value) else str(value) for value in features[column]],
        dtype=object,
    )


def _label_style(scene: Any, layer: LayerSpec, props: dict) -> dict:
    """Return the ``Axes.annotate`` keywords one label is drawn with.

    Args:
        scene: The map being drawn on, for the caller's held keywords.
        layer: The layer's description.
        props: Its recorded properties, holding the four style values and the offset.

    Returns:
        The style, with the caller's own ``**opts`` underneath it. ``text_size`` and ``color`` are declared
        parameters, so they are *set* rather than defaulted — matplotlib's own spellings of the same two are
        refused at the call, so nothing of the caller's is overwritten here. Alignment is centred on the
        feature unless the caller says otherwise, and the halo is a ``path_effects`` stroke — matplotlib's
        equivalent of MapLibre's text halo, and the reason the two tiers can share the keyword names. A
        caller's own ``path_effects`` wins, and ``halo_width=0`` asks for none.
    """
    style = drawing_style(scene, layer)
    style.setdefault("ha", "center")
    style.setdefault("va", "center")
    style["fontsize"] = props["text_size"]
    style["color"] = props["color"]
    if props["halo_width"] > 0:
        style.setdefault(
            "path_effects",
            [withStroke(linewidth=props["halo_width"], foreground=props["halo_color"])],
        )
    return style


def _label_offset_points(props: dict) -> Tuple[float, float]:
    """Return the recorded offset as the ``(dx, dy)`` in **points** matplotlib places text by.

    MapLibre's ``text-offset`` is in **ems** — multiples of the text size — and its y axis points *down* the
    screen, which is why ``(0, -1.2)`` lifts a label off its point there. matplotlib's ``"offset points"``
    are absolute and its y axis points up. Both differences are resolved here, so one ``offset=(0, -1.2)``
    lifts the label on either tier by the same fraction of its own size.

    Args:
        props: The layer's recorded properties, holding ``offset`` and ``text_size``.

    Returns:
        ``(dx, dy)`` in typographic points; ``(0.0, 0.0)`` when the call named no offset.
    """
    offset = props["offset"]
    if offset is None:
        return 0.0, 0.0
    size = float(props["text_size"])
    return float(offset[0]) * size, -float(offset[1]) * size


def draw_labels(scene: Any, data: Any, layer: LayerSpec) -> Optional[DrawnLayer]:
    """Draw one text label per feature, from the column a described layer names.

    The third member of this tier's text family: :func:`~digitalearth.static.maps.decoration.draw_text` places
    one string at one coordinate and ``draw_annotate`` points at one, while this reads a string per feature out
    of an attribute. It draws through ``Axes.annotate`` rather than ``Axes.text`` because the offset is a
    text-relative one, which only ``annotate``'s ``"offset points"`` expresses.

    Unlike those two, the layer's **name is not read as a font family**. They keep that reading because the
    artist constructor they call takes ``name=`` as an alias of the family, so a builder had to decide whether
    a layer's name shadowed the font; nothing here passes a name to an artist, so there is nothing to
    reconcile and the font is chosen with ``fontname=``/``fontfamily=`` like any other style.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``Annotation`` of every label drawn —
        one layer however many there are, as a limb-split coastline is one layer — or ``None`` when no label
        survived: every feature on the far side of a globe, or every value in the column missing. That is a
        layer that was not drawn rather than an empty one, which is what :meth:`Scene._draw` drops again.

    Raises:
        KeyError: when the features carry no such column (see :func:`_label_text`).
        OffLimbError: when the warp places none of the geometry in the display CRS. The builder is wrapped
            with :func:`_skips_off_limb`, so a caller sees a skipped layer — or, under ``strict``, the error.
        ValueError: when the features declare no CRS and the call named none, or named a different one (see
            :func:`_labelled_features`); or when the collection is empty.
    """
    props = dict(layer.symbology.props)
    placed = _labelled_features(data, props["crs"])
    features = scene._vector_input(placed, name="labels")
    points = PointArrays.from_features(features, centroids=True)
    # `finite` drops the points a clipped display CRS placed nowhere and takes the aligned text with them, so
    # a label never ends up on another feature's coordinates. It types its aligned arrays as optional — a
    # `None` in, a `None` out — and this one is never `None`, so the name it comes back under is its own.
    kept, (labelled,) = points.finite(_label_text(features, props["column"]))
    style = _label_style(scene, layer, props)
    dx, dy = _label_offset_points(props)
    drawn = [
        scene.ax.annotate(
            label,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            **style,
        )
        # Spelled `is None` rather than `or`: a numpy array has no truth value, and `labelled or ()` would
        # raise about an ambiguous one instead of passing the array through.
        for x, y, label in zip(kept.x, kept.y, () if labelled is None else labelled)
        if label != ""
    ]
    return DrawnLayer(artist=drawn, artists=tuple(drawn)) if drawn else None


def draw_grid_points(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Draw a raster's cell centres as points coloured by value (pyramids ``to_xyz`` -> ``ScatterGlyph``).

    Args:
        scene: The map being drawn on.
        data: The pyramids ``Dataset`` the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the scatter ``PathCollection``.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows.
    """
    opts = drawing_style(scene, layer)
    xyz = scene._reproject(data).to_xyz()
    opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
    # This folds the `size` channel onto the constructor keyword cleopatra takes.
    plot_style = relocate_flat_style(opts, folds_marker_size=True)
    glyph = ScatterGlyph(
        xyz.iloc[:, 0].to_numpy(),
        xyz.iloc[:, 1].to_numpy(),
        values=xyz.iloc[:, 2].to_numpy(),
        ax=scene.ax,
        fig=scene.fig,
        **opts,
    )
    drawn = scene._render_glyph(glyph, artist="plot", **plot_style)
    # `to_xyz` reads the first band, so that is the band the points are coloured by.
    return drawn.colored_by(source_field(data))


def draw_grid_cells(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Draw a raster's cells as value-coloured polygons (pyramids ``get_cell_polygons`` -> ``PolygonGlyph``).

    Args:
        scene: The map being drawn on.
        data: The pyramids ``Dataset`` the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``PolyCollection``.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows.
    """
    opts = drawing_style(scene, layer)
    ds = scene._reproject(data)
    if ds.epsg is None:
        # Work around pyramids#979: get_cell_polygons labels the returned frame with `ds.epsg` and raises
        # on `None` (pyramids >=0.47 no longer fabricates EPSG:4326 for a CRS with no authority — e.g. an
        # orthographic globe). We read only the cell geometry, in display-CRS coordinates computed from
        # the geotransform, so the authority code is cosmetic; give the transient reprojected copy a
        # placeholder EPSG (coordinates are untouched) so the call runs. Drop this once pyramids#979 ships.
        ds.epsg = 4326
    polygons = [np.asarray(g.exterior.coords) for g in ds.get_cell_polygons().geometry]
    # 1-based band, nodata -> NaN (shared helper)
    values = read_masked_band(ds, layer.symbology.props["band"]).ravel()
    # drop far-side cells on a globe
    polygons, values = scene._finite_polygons(polygons, values)
    drawn = scene._polygon_layer(polygons, values, **opts)
    return drawn.colored_by(source_field(data, layer.symbology.props["band"]))


def draw_uv_field(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Draw the u/v field a described arrow, barb or streamline layer asks for (``cleopatra.VectorGlyph``).

    Args:
        scene: The map being drawn on.
        data: The ``(u_dataset, v_dataset)`` pair the layer draws — neither component draws the field on
            its own, which is why the pair is the layer's one source.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the vector mappable.

    Raises:
        OffLimbError: when the data lies entirely outside what the display CRS shows.
    """
    props = dict(layer.symbology.props)
    kind = props["via"]
    opts = drawing_style(scene, layer)
    u_dataset, v_dataset = data
    su = scene._prepare(u_dataset, props["band"])
    sv = scene._prepare(v_dataset, props["band"])
    xs, ys = su.x.values, su.y.values
    u, v = su.z.values, sv.z.values
    # streamplot (and a tidy grid generally) needs strictly increasing axes; raster y runs north->south,
    # so flip any descending axis and its data columns/rows to match.
    if xs[0] > xs[-1]:
        xs, u, v = xs[::-1], u[:, ::-1], v[:, ::-1]
    if ys[0] > ys[-1]:
        ys, u, v = ys[::-1], u[::-1, :], v[::-1, :]
    x_grid, y_grid = np.meshgrid(xs, ys)
    opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
    plot_style = relocate_flat_style(opts)
    glyph = VectorGlyph(x_grid, y_grid, u, v, ax=scene.ax, fig=scene.fig, **opts)
    drawn = scene._render_glyph(glyph, artist="plot", kind=kind, **plot_style)
    scene._last_vector = (drawn.glyph, drawn.artist, kind)  # remembered for quiverkey()
    # A u/v field has no column: the arrows are coloured by the vector's magnitude, computed from the pair.
    return drawn.colored_by(MAGNITUDE_FIELD)


def draw_tri(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Triangulate scattered points and draw the render a described unstructured layer asks for.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``Dataset`` (its cells become points) or ``FeatureCollection`` the layer draws.
        layer: The layer's description, whose ``via`` names the triangulated render.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the mappable.

    Raises:
        OffLimbError: when the data lies outside the display CRS, or when enough points were supplied but
            too few of them survive the reprojection to triangulate — a vector warp sends off-view points
            to infinity rather than raising, and losing them all means the same thing as an off-limb
            raster: nothing on the view to draw.
        ValueError: when fewer than three points were supplied in the first place, which no projection can
            fix.
    """
    from matplotlib.tri import Triangulation

    kind = layer.symbology.props["via"]
    opts = drawing_style(scene, layer)
    x, y, z, field = scene._scattered(data)
    # drop far-side points on a globe (Triangulation needs finite)
    finite = np.isfinite(x) & np.isfinite(y)
    supplied = np.asarray(x).size
    x, y, z = np.asarray(x)[finite], np.asarray(y)[finite], np.asarray(z)[finite]
    if x.size < 3:
        if supplied < 3:
            # Never enough points to triangulate, whatever the projection — a caller error, and the
            # accurate complaint is the one matplotlib would give.
            raise ValueError(
                f"{kind}() needs at least three points to triangulate, got {supplied}"
            )
        raise OffLimbError(
            f"{kind}: only {x.size} of {supplied} points survive the warp into "
            f"{scene.crs!r}, which is too few to triangulate"
        )
    tri = Triangulation(x, y)
    glyph = MeshGlyph(x, y, tri.triangles, ax=scene.ax, fig=scene.fig)
    # cleopatra 0.11.0 exposes the tripcolor/tricontour(f) artist on glyph.im (issue #2).
    if kind == "tripcolor":
        face_values = z[tri.triangles].mean(axis=1)
        drawn = scene._render_glyph(
            glyph, face_values, location="face", colorbar=False, **opts
        )
    else:
        drawn = scene._render_glyph(
            glyph,
            z,
            location="node",
            filled=(kind == "tricontourf"),
            colorbar=False,
            **opts,
        )
    return drawn.colored_by(field)


def draw_choropleth(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Fill the polygons a described choropleth asks for, coloured by the column it names.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` of polygons the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``PolyCollection``.

    Raises:
        OffLimbError: when the warp places none of the geometry in the display CRS.
        ValueError: when the collection is empty or holds non-polygon geometry.
    """
    props = dict(layer.symbology.props)
    opts = drawing_style(scene, layer)
    gdf = scene._vector_input(
        data,
        geom_types=("Polygon", "MultiPolygon"),
        name="choropleth",
        geom_label="polygon",
    )
    polygons, repeats = scene._polygon_vertices(gdf.geometry)
    values = np.repeat(gdf[props["column"]].to_numpy(), repeats)
    # drop far-side polygons on a globe
    polygons, values = scene._finite_polygons(polygons, values)
    scheme = props["scheme"]
    if scheme is not None:  # None means a continuous ramp: classify nothing
        opts["scheme"], opts["k"] = scheme, props["k"]
    return scene._polygon_layer(polygons, values, **opts)


def draw_shapes(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Draw the polygon outlines a described shapes layer asks for, without fill.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` of polygons the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``PolyCollection``.

    Raises:
        OffLimbError: when the warp places none of the geometry in the display CRS.
        ValueError: when the collection is empty or holds non-polygon geometry.
    """
    gdf = scene._vector_input(
        data,
        geom_types=("Polygon", "MultiPolygon"),
        name="polygons",
        geom_label="polygon",
    )
    polygons, _ = scene._polygon_vertices(gdf.geometry)
    # drop far-side polygons on a globe
    polygons, _ = scene._finite_polygons(polygons)
    return scene._polygon_layer(polygons, **drawing_style(scene, layer))


def draw_voronoi(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Tessellate the points a described Voronoi layer names and draw the cells.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` of points the layer draws.
        layer: The layer's description; the clip boundary, which a figure cannot carry, is held on the
            scene under the layer's id.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``PolyCollection``.

    Raises:
        OffLimbError: when the warp places none of the geometry in the display CRS.
        ValueError: when the collection is empty, holds non-point geometry, or leaves no finite point.
    """
    column = layer.symbology.props["column"]
    gdf = scene._vector_input(
        data, geom_types=("Point",), name="voronoi", geom_label="point"
    )
    geom = gdf.geometry
    col_vals = gdf[column].to_numpy() if column is not None else None
    # Drop points with non-finite reprojected coords (far side of a clipped/globe CRS); ordered=True
    # then keeps cell i aligned with input point i, so values map by position.
    xs, ys, col_vals = scene._finite_point_xy(geom, col_vals)
    if xs.size == 0:
        raise ValueError("voronoi: no finite points in the display CRS")
    cells = voronoi_polygons(MultiPoint(list(zip(xs, ys))), ordered=True)
    boundary = scene._clip_geometry(scene._layer_keys.get(layer.id))
    polygons, values = _clipped_cell_rings(cells, boundary, col_vals, column)
    values_arr = np.asarray(values) if values is not None else None
    # drop far-side cells on a globe
    polygons, values_arr = scene._finite_polygons(polygons, values_arr)
    return scene._polygon_layer(polygons, values_arr, **drawing_style(scene, layer))


def draw_cartogram(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Scale each polygon about its centroid by the column a described cartogram names, and draw it.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` of polygons the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``PolyCollection``.

    Raises:
        OffLimbError: when the warp places none of the geometry in the display CRS.
        ValueError: when the collection is empty or holds non-polygon geometry.
    """
    props = dict(layer.symbology.props)
    opts = drawing_style(scene, layer)
    gdf = scene._vector_input(
        data,
        geom_types=("Polygon", "MultiPolygon"),
        name="cartogram",
        geom_label="polygon",
    )
    factors = scene._scale_factors(
        gdf[props["scale"]].to_numpy(dtype=float), tuple(props["limits"])
    )
    scaled = [
        affine_scale(g, xfact=f, yfact=f, origin="centroid")
        for g, f in zip(gdf.geometry, factors)
    ]
    polygons, repeats = scene._polygon_vertices(scaled)
    column = props["column"]
    if column is None:
        polygons, _ = scene._finite_polygons(polygons)
        return scene._polygon_layer(polygons, **opts)
    values = np.repeat(gdf[column].to_numpy(), repeats)
    polygons, values = scene._finite_polygons(polygons, values)
    return scene._polygon_layer(polygons, values, **opts)


def draw_quadtree(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Aggregate the points a described quadtree names into adaptive cells and draw them.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` of points the layer draws.
        layer: The layer's description, which names the per-cell reducer when it has a name. The clip
            boundary — a geometry, which a figure cannot carry — and a reducer the caller wrote themselves
            are held on the scene under the layer's id.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``PolyCollection``.

    Raises:
        KeyError: when the description carries no ``agg`` property — a layer built by hand, or a figure
            written before the named reducer was recorded. The key is read outright rather than defaulted,
            so a description that cannot say how it aggregates is refused instead of aggregating some
            other way.
        OffLimbError: when the warp places none of the geometry in the display CRS.
        ValueError: when the collection is empty, holds non-point geometry, or leaves no finite point.
    """
    props = dict(layer.symbology.props)
    column = props["column"]
    held_agg, clip = scene._layer_keys.get(layer.id, (None, None))
    # The named reducer travels in the figure; only a caller's own callable is held beside the layer, and a
    # scene that holds none — a figure read back from JSON — reads the name the description carries. A
    # figure built with a callable carries no name, and falls back to the default the builder documents.
    described_agg = props["agg"]
    agg = held_agg if held_agg is not None else described_agg or DEFAULT_QUADTREE_AGG
    gdf = scene._vector_input(
        data, geom_types=("Point",), name="quadtree", geom_label="point"
    )
    # Drop points with non-finite reprojected coords (far side of a clipped/globe CRS) before binning.
    col_vals_full = gdf[column].to_numpy(dtype=float) if column is not None else None
    xs, ys, col_vals = scene._finite_point_xy(gdf.geometry, col_vals_full)
    if xs.size == 0:
        raise ValueError("quadtree: no finite points in the display CRS")
    agg_fn = _quadtree_reducer(agg, column, col_vals)
    cells = scene._quadtree_cells(xs, ys, agg_fn, props["nmax"], props["nmin"])
    polygons, values = _clipped_cell_boxes(cells, scene._clip_geometry(clip))
    values_arr = np.asarray(values, dtype=float)
    polygons, values_arr = scene._finite_polygons(polygons, values_arr)
    drawn = scene._polygon_layer(polygons, values_arr, **drawing_style(scene, layer))
    # Without a column the reducer counts the points in each cell, whatever `agg` names, so the cells are
    # coloured by a count rather than by any attribute (see :func:`_quadtree_reducer`).
    return drawn.colored_by(column or COUNT_FIELD)


#: The per-cell reducer ``hexbin`` aggregates a column with when none is named — cleopatra's own default.
DEFAULT_HEXBIN_REDUCE = "mean"


def draw_hexbin(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Bin the points a described hexbin layer names onto a hexagonal lattice and draw the cells.

    The lattice is laid in the display CRS, after pyramids has reprojected the points, so the cells are
    regular hexagons on the map — not equal-area on the ground, which is a property of the projection.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` of points the layer draws.
        layer: The layer's description. A reducer the caller wrote themselves — a function, which a figure
            cannot carry — is held on the scene under the layer's id; a named one is described.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the cells' ``PolyCollection``.

    Raises:
        OffLimbError: when the warp places none of the geometry in the display CRS.
        ValueError: when the collection is empty, holds non-point geometry, or leaves no finite point.
        KeyError: when ``column`` names no feature attribute.
    """
    props = dict(layer.symbology.props)
    column = props.get("column")
    held_reduce = scene._layer_keys.get(layer.id)
    reduce = (
        held_reduce
        if held_reduce is not None
        else props.get("reduce") or DEFAULT_HEXBIN_REDUCE
    )
    gdf = scene._vector_input(
        data, geom_types=("Point",), name="hexbin", geom_label="point"
    )
    values_full = gdf[column].to_numpy(dtype=float) if column is not None else None
    # Drop points with non-finite reprojected coords (far side of a clipped/globe CRS) before binning.
    xs, ys, values = scene._finite_point_xy(gdf.geometry, values_full)
    if xs.size == 0:
        raise ValueError("hexbin: no finite points in the display CRS")
    opts = drawing_style(scene, layer)
    opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
    min_count = props.get("min_count")
    if min_count is None and column is None:
        # cleopatra's counts mode draws every lattice cell in the window, empty ones as 0, which tints the
        # whole map; a map of counts shows only the cells a point fell in.
        min_count = 1
    plot_style = relocate_flat_style(opts)  # scheme/k -> plot() classify group
    glyph = HexbinGlyph(
        xs,
        ys,
        values,
        ax=scene.ax,
        fig=scene.fig,
        gridsize=props.get("gridsize", 50),
        reduce=reduce,
        min_count=min_count,
        **opts,
    )
    drawn = scene._render_glyph(glyph, artist="plot", **plot_style)
    counted = column is None or reduce == "count"
    return drawn.colored_by(COUNT_FIELD if counted else column)


def draw_kde(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Estimate the point density a described heatmap asks for and draw it (``cleopatra.KDEGlyph``).

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` of points the layer draws.
        layer: The layer's description; the clip boundary, which a figure cannot carry, is held on the
            scene under the layer's id.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the contour set.

    Raises:
        OffLimbError: when the warp places none of the geometry in the display CRS.
        ValueError: when the collection is empty, holds non-point geometry, or leaves no finite point.
    """
    opts = drawing_style(scene, layer)
    gdf = scene._vector_input(
        data, geom_types=("Point",), name="kde", geom_label="point"
    )
    # Drop points with non-finite reprojected coords (far side of a clipped/globe CRS) before the KDE.
    xs, ys, _ = scene._finite_point_xy(gdf.geometry)
    if xs.size == 0:
        raise ValueError("kde: no finite points in the display CRS")
    opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
    # levels/… -> plot() contour/data_style groups
    plot_style = relocate_flat_style(opts)
    glyph = KDEGlyph(
        xs,
        ys,
        clip_path=scene._clip_path(scene._layer_keys.get(layer.id)),
        ax=scene.ax,
        fig=scene.fig,
        **opts,
    )
    drawn = scene._render_glyph(glyph, artist="plot", **plot_style)
    # A KDE has no column either: the bands are the estimated density of the points themselves.
    return drawn.colored_by(DENSITY_FIELD)


def _line_parts(gdf: Any) -> Tuple[List[np.ndarray], np.ndarray]:
    """Split a line GeoDataFrame into one vertex array per drawn path.

    Args:
        gdf: Line geometries, already in the display CRS.

    Returns:
        ``(paths, repeats)``: one ``(n, 2)`` array per ``LineString`` part, and how many parts each feature
        contributed — so a per-feature column is repeated onto its parts with ``numpy.repeat``.
    """
    paths: List[np.ndarray] = []
    repeats: List[int] = []
    for geometry in gdf.geometry:
        parts = (
            list(geometry.geoms)
            if geometry.geom_type == "MultiLineString"
            else [geometry]
        )
        paths.extend(np.asarray(part.coords) for part in parts)
        repeats.append(len(parts))
    return paths, np.asarray(repeats)


def draw_lines(scene: Any, data: Any, layer: LayerSpec) -> Optional[DrawnLayer]:
    """Draw the line features a described ``lines`` layer asks for (``cleopatra.FlowGlyph``).

    The same glyph ``sankey`` draws with, held to line defaults: one width for every path unless a column
    sizes them, one colour unless a column colours them. The description's props are read with defaults,
    because a ``lines`` layer another tier described carries its own styling rather than these keys.

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` of lines the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``LineCollection``.

    Raises:
        OffLimbError: when the warp places none of the geometry in the display CRS.
        ValueError: when the collection is empty or holds non-line geometry.
        KeyError: when ``column`` or a column ``width`` names no feature attribute.
    """
    props = dict(layer.symbology.props)
    opts = drawing_style(scene, layer)
    gdf = scene._vector_input(
        data,
        geom_types=("LineString", "MultiLineString"),
        name="lines",
        geom_label="line",
    )
    paths, repeats = _line_parts(gdf)
    column, width = props.get("column"), props.get("width")
    values = (
        np.repeat(_numeric_column(gdf, column, "column", _LINES_CALLER), repeats)
        if column is not None
        else None
    )
    widths = None
    if isinstance(width, str):
        widths = np.repeat(_numeric_column(gdf, width, "width", _LINES_CALLER), repeats)
    elif width is not None:
        opts["line_width"] = float(width)
    if props.get("color") is not None:
        opts["color_1"] = props["color"]
    opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
    plot_style = relocate_flat_style(opts)  # scheme/k -> plot() classify group
    glyph = FlowGlyph(
        paths, values=values, widths=widths, ax=scene.ax, fig=scene.fig, **opts
    )
    drawn: Optional[DrawnLayer] = scene._render_glyph(
        glyph, artist="plot", **plot_style
    )
    opacity = props.get("opacity")
    if drawn is not None and opacity is not None:
        # FlowGlyph takes no opacity of its own; the collection it drew does.
        drawn.artist.set_alpha(float(opacity))
    return drawn


def draw_sankey(scene: Any, data: Any, layer: LayerSpec) -> DrawnLayer:
    """Draw the flow paths a described Sankey layer asks for (``cleopatra.FlowGlyph``).

    Args:
        scene: The map being drawn on.
        data: The pyramids ``FeatureCollection`` of lines the layer draws.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.static.renderer.DrawnLayer` holding the ``LineCollection``.

    Raises:
        OffLimbError: when the warp places none of the geometry in the display CRS.
        ValueError: when the collection is empty or holds non-line geometry.
    """
    props = dict(layer.symbology.props)
    opts = drawing_style(scene, layer)
    gdf = scene._vector_input(
        data,
        geom_types=("LineString", "MultiLineString"),
        name="sankey",
        geom_label="line",
    )
    paths, rep = _line_parts(gdf)
    column, scale = props["column"], props["scale"]
    values = np.repeat(gdf[column].to_numpy(), rep) if column is not None else None
    widths = np.repeat(gdf[scale].to_numpy(), rep) if scale is not None else None
    opts.setdefault("add_colorbar", False)  # the Scene owns the aggregated colorbar
    plot_style = relocate_flat_style(opts)  # scheme/k -> plot() classify group
    glyph = FlowGlyph(
        paths, values=values, widths=widths, ax=scene.ax, fig=scene.fig, **opts
    )
    return scene._render_glyph(glyph, artist="plot", **plot_style)


if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.static.maps.base import GeoLayerBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def _opened_component(component: Any) -> Any:
    """Open one half of a u/v pair when the caller named it by path or URL.

    Every other data builder hands its argument straight to the figure's source table, which records a
    path as a path and opens it for the drawer. A u/v field cannot: a layer names **one** source, and a
    field is not drawable from either component alone, so the pair is recorded as one in-memory object and
    the table never sees the two paths. The components are therefore opened here, with the same resolver
    the table uses, rather than reaching the drawer as strings — which is where they met ``to_crs`` and
    raised ``AttributeError: 'str' object has no attribute 'to_crs'`` from three frames inside pyramids.

    Args:
        component: A pyramids ``Dataset``, or a path or URL to one.

    Returns:
        The dataset. A component that is already an object is handed back untouched, so nothing is
        registered for it and nothing has to be forgotten.

    Raises:
        ValueError: when `component` is an empty string, which names nothing to open.
        FileNotFoundError: when the path names nothing, which is the resolver's message naming it.
        KeyError: when no resolver is registered for the URL's scheme, which is the resolver's message
            listing the schemes there are.

    Note:
        A u/v figure is still ``object:``-backed and so cannot be written down. That needs a layer to be
        able to name two sources, which is the shared spec's to settle, not this tier's.
    """
    if isinstance(component, (str, os.PathLike)):
        return DataRef.of(component).open()
    return component


def _described_agg(agg: Any) -> Tuple[Any, Any]:
    """Split a quadtree reducer into the name a description records and the callable held beside the layer.

    The sibling of :func:`~digitalearth.static.maps.raster._described_cmap`, and it exists for the same
    reason: ``agg`` decides the **values** a cell carries, so a figure that cannot say which reducer it was
    built with redraws different data rather than the same data styled differently.

    Args:
        agg: What the caller asked for — one of the registered names, or a callable of their own.

    Returns:
        ``(recorded, held)``. A name is recorded and nothing is held: it resolves to the same reducer
        wherever the figure is read. A callable resolves nowhere, so it is recorded as ``None`` and held
        beside the layer, which is what aggregates this drawing; a figure read back elsewhere then falls
        back to :data:`DEFAULT_QUADTREE_AGG`.
    """
    return (agg, None) if isinstance(agg, str) else (None, agg)


def _quadtree_reducer(agg: Any, column: Optional[str], col_vals: Any) -> Any:
    """Return the function that gives one quadtree cell its value.

    Args:
        agg: A named reducer, a callable, or ignored when `column` is `None`.
        column: The column being aggregated, or `None` to count points instead.
        col_vals: The per-point values, positionally aligned with the points.

    Returns:
        A callable taking a cell's point indices and returning its scalar value.

    Raises:
        ValueError: for a named reducer nobody registered, listing the ones that are.
    """
    if column is None:
        return lambda idx: float(len(idx))
    if callable(agg):
        reducer = agg
    elif agg in _QUADTREE_AGG:
        reducer = _QUADTREE_AGG[agg]
    else:
        raise ValueError(
            f"unknown agg {agg!r}; choose one of {sorted(_QUADTREE_AGG)} or a callable"
        )
    return lambda idx: float(reducer(col_vals[idx]))


def _clipped_cell_boxes(cells: Any, boundary: Any) -> tuple:
    """Return each quadtree cell as a ring, clipped to `boundary`, with the value it carries.

    Unclipped, a cell is its own four corners. Clipped, it can vanish, or break into several parts that
    are not all polygons — an intersection against a globe's limb produces both — so a cell contributes
    zero, one or several rings, each carrying the cell's value.

    Args:
        cells: `(xmin, ymin, xmax, ymax, value)` per cell.
        boundary: The clip geometry, or `None` to keep every cell whole.

    Returns:
        `(rings, values)`, positionally aligned.
    """
    rings: List[np.ndarray] = []
    values: List[float] = []
    for xmin, ymin, xmax, ymax, value in cells:
        if boundary is None:
            rings.append(
                np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
            )
            values.append(value)
            continue
        clipped = box(xmin, ymin, xmax, ymax).intersection(boundary)
        for ring in _polygon_rings(clipped):
            rings.append(ring)
            values.append(value)
    return rings, values


def _quadrants(xs: np.ndarray, ys: np.ndarray, idx: np.ndarray, box: tuple) -> list:
    """Split one quadtree cell into its four children, partitioning the points that fall in it.

    Args:
        xs: All point x-coordinates.
        ys: All point y-coordinates, aligned with `xs`.
        idx: The indices of the points inside this cell.
        box: The cell as `(xmin, ymin, xmax, ymax)`.

    Returns:
        Four `(xmin, ymin, xmax, ymax, indices)` tuples, south-west first then clockwise. A child may hold
        no points; that is the caller's cue, not an error.
    """
    xmin, ymin, xmax, ymax = box
    xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    cx, cy = xs[idx], ys[idx]
    return [
        (xmin, ymin, xmid, ymid, idx[(cx <= xmid) & (cy <= ymid)]),
        (xmid, ymin, xmax, ymid, idx[(cx > xmid) & (cy <= ymid)]),
        (xmin, ymid, xmid, ymax, idx[(cx <= xmid) & (cy > ymid)]),
        (xmid, ymid, xmax, ymax, idx[(cx > xmid) & (cy > ymid)]),
    ]


def _split_made_no_progress(quads: list, held: int) -> bool:
    """Whether splitting a cell moved nothing — every point landed in one child.

    Args:
        quads: The four children from :func:`_quadrants`.
        held: How many points the parent held.

    Returns:
        `True` when one child holds them all, which happens for coincident points and would otherwise
        recurse until the depth cap.
    """
    nonempty = [quad for quad in quads if len(quad[4]) > 0]
    return len(nonempty) == 1 and len(nonempty[0][4]) == held


def _polygon_rings(geometry: Any) -> list:
    """Return the exterior ring of every polygon part of a geometry.

    A clip can hand back nothing, one polygon, or a multipart geometry whose parts are not all polygons —
    an intersection against a globe's limb produces both. A part with no area has no exterior ring to read,
    so it contributes nothing rather than raising from `part.exterior`.

    Note:
        Only a `Multi*` geometry is unpacked, which is what both callers did inline before sharing this.
        A `GeometryCollection` is therefore read as a single non-polygon and contributes nothing; that is
        existing behaviour, preserved deliberately rather than quietly widened here.

    Args:
        geometry: Any shapely geometry, typically the result of an intersection.

    Returns:
        One coordinate array per polygon part, in the order the parts are held.
    """
    if geometry.is_empty:
        return []
    parts = geometry.geoms if geometry.geom_type.startswith("Multi") else [geometry]
    return [
        np.asarray(part.exterior.coords)
        for part in parts
        if part.geom_type == "Polygon" and not part.is_empty
    ]


def _clipped_cell_rings(
    cells: Any, boundary: Any, col_vals: Any, column: Optional[str]
) -> tuple:
    """Return each Voronoi cell's exterior ring, clipped to `boundary`, with the value it carries.

    A clipped cell can come back empty, or as a multipart geometry whose parts are not all polygons — an
    intersection against a globe's limb produces both — so a cell contributes zero, one or several rings.
    `ordered=True` keeps cell *i* aligned with input point *i*, which is what lets the value follow the
    ring however many parts the cell broke into.

    Args:
        cells: The `voronoi_polygons` result.
        boundary: The clip geometry, or `None` to keep every cell whole.
        col_vals: The per-point values, positionally aligned with `cells`, or `None`.
        column: The column the values came from, or `None` when the draw is outline-only.

    Returns:
        `(rings, values)` — the exterior coordinate arrays, and the values beside them or `None`.
    """
    rings: List[np.ndarray] = []
    values: Optional[list] = [] if column is not None else None
    for index, cell in enumerate(cells.geoms):
        clipped = cell if boundary is None else cell.intersection(boundary)
        for ring in _polygon_rings(clipped):
            rings.append(ring)
            if values is not None:
                values.append(col_vals[index])
    return rings, values


class VectorMixin(_MixinBase):
    """Vector-data and vector-field renders for :class:`~digitalearth.static.map.Map`.

    A capability mixin of :class:`~digitalearth.static.map.Map`: it is only ever composed into that map class, never
    instantiated or subclassed on its own. Its methods reach the shared figure/axes, the layer registry and the
    display CRS — and the sibling mixins' methods — through ``self``, and only the composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.static.maps.base.GeoLayerBase`, the state ``Map``
    inherits. At runtime that base is plain ``object``, so composing this mixin leaves the ``Map`` MRO exactly what
    it was before the annotation.

    Every builder here takes the caller's styling as ``**opts``/``**kwargs`` and hands it to the cleopatra
    glyph as passed. Those keywords **go two ways**
    (see :attr:`~digitalearth.static.scene.LayerRecord.opts`). The plain ones — a string, a boolean, a
    finite number, ``None`` — are written into the layer's description as well, so the same figure drawn on
    another scene comes back with the ``color``, ``alpha`` or ``linewidth`` the caller asked for.
    Everything else is held on the scene and nowhere else: a ``Normalize``, a per-point ``alpha`` array or
    a dash tuple is either unwritable or comes back as something matplotlib refuses, and a figure read
    elsewhere draws those with the engine's defaults. The description records the call the caller made
    besides them — the band, the column, the scheme, the recipe.

    See Also:
        digitalearth.static.map.Map: the composition that supplies the state these methods use.
        digitalearth.static.maps.base.GeoLayerBase: the typing-only base declared above the class.
    """

    def _vector_input(
        self,
        features: Any,
        *,
        geom_types: Optional[Sequence[str]] = None,
        name: str = "layer",
        geom_label: Optional[str] = None,
    ) -> Any:
        """Reproject a ``FeatureCollection`` to the display CRS, reject empty, and validate its geometry type.

        Consolidates the preamble shared by the validating vector **drawers** — the builders record their
        layer and this runs when it is drawn. Reprojects ``features`` to :attr:`crs`, raises on an empty
        collection, and — when ``geom_types`` is given — requires every geometry to be one of those types.

        Reprojection goes through the shared :func:`~digitalearth.base.crs.reproject` helper rather than
        calling ``to_crs`` directly: a vector warp does not raise when it places nothing, it hands back
        ``inf`` coordinates, and only that helper reads the result and reports it. Calling ``to_crs`` here
        is what kept the off-limb contract off this tier — the one tier with a clipped globe to be behind,
        and so the one where a layer really can land nowhere.

        Args:
            features: A pyramids ``FeatureCollection`` (reprojected to the display CRS).
            geom_types: Allowed shapely ``geom_type`` names (e.g. ``("Point",)``); ``None`` skips the check.
            name: The calling method's name, used in error messages.
            geom_label: Human label for the allowed geometry in the error (defaults to the joined types).

        Returns:
            The reprojected GeoDataFrame.

        Raises:
            OffLimbError: when the warp places none of the geometry in the display CRS. It travels out
                through the drawer and `Scene._draw`, which drops the layer from the description; the
                public builder is wrapped with :func:`_skips_off_limb`, so the caller sees a skipped layer
                (or, under ``strict``, the error).
            TypeError: when `features` is not a vector layer — refused by name here rather than leaking
                ``AttributeError: ... has no attribute 'to_crs'`` out of the warp below, which is the same
                defect ``Map.field`` had for a raster (#343). This runs in the drawer rather than in the
                builder, which is where every vector kind meets, so `name` is what makes the message name
                the public call the caller wrote.
            ValueError: if the collection is empty, or a geometry is not one of ``geom_types``.
        """
        require_drawable(features, caller=f"Map.{name}()", accepts=("vector",))
        gdf = reproject(features, self.crs)
        if len(gdf) == 0:
            raise ValueError(f"{name} got an empty FeatureCollection (nothing to draw)")
        if (
            geom_types is not None
            and not gdf.geometry.geom_type.isin(list(geom_types)).all()
        ):
            label = geom_label or " / ".join(geom_types)
            raise ValueError(
                f"{name} requires a FeatureCollection of {label} geometries"
            )
        return gdf

    def _polygon_layer(
        self,
        polygons: List[np.ndarray],
        values: Optional[np.ndarray] = None,
        **opts,
    ) -> DrawnLayer:
        """Draw polygons as a value-filled (``values`` given) or outline-only ``PolygonGlyph`` layer.

        Consolidates the fill-vs-outline branch shared by :meth:`grid_cells`, :meth:`choropleth`,
        :meth:`polygons`, :meth:`voronoi`, :meth:`cartogram` and :meth:`quadtree`. The Scene owns the
        aggregated colorbar, so the glyph's own colorbar is suppressed by default — except under
        ``scheme="categorical"``, where the value key is a per-class swatch legend the glyph builds from its
        own mapping (``PolygonGlyph.category_legend``). The Scene's colorbar cannot stand in for it: a
        categorical fill feeds the mappable opaque integer class codes, so a colorbar over them would read
        ``0, 1, 2 …`` instead of the category labels. Passing ``add_colorbar=False`` suppresses that swatch
        legend — for a caller keying the map some other way, e.g. drawing one shared legend across several
        layers via :meth:`~digitalearth.static.scene.Scene.legend`, which derives its rows from the layer's
        own :class:`~digitalearth.base.spec.scale.Scale` — the swatches this glyph drew, read back off it by
        :func:`~digitalearth.static.guides.color_encoding` — rather than being handed colours and labels.

        Args:
            polygons: Polygon rings as ``(N, 2)`` vertex arrays.
            values: Optional per-polygon scalar values; when ``None`` only the outlines are drawn.
            **opts: Styling kwargs forwarded to ``PolygonGlyph``.

        Returns:
            What the render produced, as a :class:`~digitalearth.static.renderer.DrawnLayer` — the drawers
            that call this hand it straight back to the renderer, which is what records the layer.
        """
        scheme = opts.get("scheme")
        # `isinstance` states the intent: cleopatra's `classify` also accepts a list/ndarray of explicit bin
        # edges as `scheme`, which must never be stringified into this comparison.
        categorical = isinstance(scheme, str) and scheme.lower() == "categorical"
        # Any scheme — categorical or graduated — puts a missing value outside every class, so it is the
        # classified layers (not the continuous ramp) that need the neutral "bad" colour.
        classified = scheme is not None
        opts.setdefault(
            "add_colorbar", categorical
        )  # the Scene owns the colorbar; the glyph owns the legend
        if categorical:
            # Normalize the spelling: cleopatra dispatches on an exact, case-sensitive `== "categorical"`, so a
            # case variant would set up a categorical render here and then fall through to the continuous path
            # there — dying inside `np.isfinite` on a string column. The web tier accepts any case, so
            # normalizing (rather than matching cleopatra's exactness) keeps a `scheme` portable across tiers.
            opts["scheme"] = "categorical"
            # Resolve the colormap here so all three tiers key off ONE sentinel. cleopatra applies the same
            # "swap the continuous default for a qualitative one" rule against a *different* sentinel (its own
            # default, "coolwarm_r"), so left to itself it would honour an explicit cmap="viridis" that the
            # web/interactive tiers swap for tab10 — one cmap, two maps.
            opts["cmap"] = resolve_categorical_cmap(opts.get("cmap"))
            if values is not None:
                # cleopatra's null test is `is None` plus a float-NaN check, so a `pd.NA` from a pandas
                # nullable dtype (`string`, `Int64`, …) would survive it and become a coloured `<NA>` class,
                # shifting every other category's colour — i.e. the same data would render differently
                # depending only on the column's dtype.
                values = nulls_to_none(values)
        # scheme/k moved onto the plot() `classify` group; pull them off the constructor kwargs.
        plot_style = relocate_flat_style(opts)
        if values is not None:
            glyph = PolygonGlyph(
                polygons, values=values, ax=self.ax, fig=self.fig, **opts
            )
            drawn = self._render_glyph(glyph, artist="plot", **plot_style)
            if classified:
                _draw_missing_neutral(drawn.artist)
            return drawn
        glyph = PolygonGlyph(polygons, ax=self.ax, fig=self.fig, **opts)
        return self._render_glyph(glyph, artist="plot", outline_only=True, **plot_style)

    @_skips_off_limb
    def points(
        self,
        features: Any,
        *,
        size_column: Optional[str] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Any:
        """Plot a pyramids ``FeatureCollection`` of points, sized by a column (``ScatterGlyph``).

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries, or a path or URL to one;
                reprojected to the display CRS. Only a path-backed layer can be written down.
            size_column: Optional column name whose values set the per-point marker size. Pair it with
                ``size_legend=True`` (and optionally ``size_limits`` / ``size_scale``) to draw a size
                legend. ``None`` (default) uses a single uniform marker size — set that size with
                ``size`` (which every backend spells the same way).
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Styling kwargs forwarded to ``ScatterGlyph`` (``cmap``, ``scheme``, ``k``, ``size``,
                ``size_limits``, ``size_scale``, ``size_legend``, ``size_legend_values``, …).
                ``size`` is the marker's visual size, spelled the same way on every backend, and
                :func:`~digitalearth.static.render_compat._fold_marker_size` is the one place that knows
                cleopatra's constructor calls it ``point_size``.

        Returns:
            The scatter ``PathCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        """
        return self._draw(
            LayerRecord(
                "points",
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "scatter",
                        "size_column": size_column,
                    }
                ),
                opts=opts,
            )
        )

    @_skips_off_limb
    def labels(
        self,
        features: Any,
        column: str,
        *,
        text_size: float = DEFAULT_LABEL_TEXT_SIZE,
        color: str = DEFAULT_LABEL_COLOR,
        halo_color: str = DEFAULT_LABEL_HALO_COLOR,
        halo_width: float = DEFAULT_LABEL_HALO_WIDTH,
        offset: Optional[Any] = None,
        crs: Any = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Any:
        """Label every feature with the text in ``column`` — one string per feature, from an attribute.

        The static third of order 27's text layers (ST-16, #345). ``WebMap.labels`` shipped first and the
        ``labels`` kind was registered with it, so a caller could label a collection on a web map and not on a
        static one; :meth:`~digitalearth.static.maps.decoration.DecorationMixin.text` is no substitute, since
        it places **one** string at **one** coordinate.

        The keywords are ``WebMap.labels``'s wherever the two tiers can mean the same thing, which is the point
        of the Core contract — one spelling per concept:

        * ``text_size``, ``color`` — the same, with the same defaults, so one call gives one picture. It is
          ``text_size`` and never ``size``: ``size`` is the visual size of a *marker* on every other builder,
          and a keyword cannot mean both. matplotlib's own ``fontsize``/``size`` are refused here rather than
          quietly overwritten.
        * ``halo_color``, ``halo_width`` — MapLibre's text halo is matplotlib's ``path_effects`` stroke: the
          same idea, a different spelling, so the keywords carry across and this builder builds the stroke.
          ``halo_width=0`` draws no halo, and a caller's own ``path_effects=`` replaces it.
        * ``offset`` — carried, and converted. MapLibre measures it in **ems** with y pointing *down*;
          matplotlib in points with y pointing up. Both are resolved in the drawer, so ``offset=(0, -1.2)``
          lifts a label off its point by 1.2 of its own text size on either tier.
        * ``allow_overlap`` — **not** carried, and refused rather than accepted. It asks MapLibre's symbol
          layer to keep colliding labels; matplotlib has no collision index to ask, and draws every label
          wherever it lands. The tier declares that under ``label_collision`` in
          :data:`~digitalearth.static.capabilities.CAPABILITIES`, and the refusal quotes that declaration, so
          the answer a caller gets and the answer a dispatcher reads are one sentence. A crowded column is
          thinned by filtering the features before drawing them.
        * ``crs`` — the Core keyword for this name, and here it has a job the web tier has no use for: it says
          what a collection declaring **no** CRS is measured in, which is the one case pyramids cannot
          reproject. It is never ignored — against a collection with its own CRS it either agrees or is
          refused (see :func:`_labelled_features`).

        Points are labelled at the point; a line or a polygon at its centroid, which is the fallback every
        other reader of vector geometry in this package uses. A feature whose value is missing, and one on the
        far side of a globe, are simply not labelled.

        Args:
            features: A pyramids ``FeatureCollection`` (or a path or URL to one), of any geometry. Only a
                path-backed layer can be written down.
            column: The attribute the label text is read from. Checked when the features are opened, because
                a path-referenced layer has none to check at the call.
            text_size: Text size in points. Named for the text rather than ``size``, which is a marker's size
                everywhere else.
            color: Text colour.
            halo_color: Colour of the outline drawn behind the glyphs, which is what keeps a label legible
                over a raster field or a basemap.
            halo_width: Halo width in points; ``0`` draws none.
            offset: ``(x, y)`` offset in ems — multiples of ``text_size`` — with y **down**, as MapLibre
                measures it; e.g. ``(0, -1.2)`` to lift the label off its point. ``None`` centres it.
            crs: What the features' coordinates are measured in, for a collection that declares none.
                ``None`` (default) reads the collection's own.
            name: The caller's own name for the layer, used as its id and its label; ``None`` (default)
                generates one from the kind, and a name already on the figure is suffixed ``-2``, ``-3``, …
                (#321). Unlike :meth:`~digitalearth.static.maps.decoration.DecorationMixin.text`, it is
                **not** also read as a font family — see :func:`draw_labels`.
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it hidden, so a
                switcher reading the figure agrees with the axes (#327).
            **opts: Forwarded to ``Axes.annotate`` (``fontname``, ``fontweight``, ``rotation``, ``ha``,
                ``va``, ``zorder``, …), the plain ones described beside the layer as on every other builder.

        Returns:
            The list of :class:`matplotlib.text.Annotation` drawn, one per labelled feature (registered as one
            Scene layer). ``None`` instead when nothing was labelled — every feature outside what the display
            CRS shows, or every value in the column missing.

        Raises:
            KeyError: when ``column`` is not one of the features' attributes, naming the ones that are.
            TypeError: when ``features`` is not a vector layer, naming this call and the type.
            ValueError: for a ``column`` that is not a non-empty name; for a ``text_size``, ``halo_width`` or
                ``offset`` a figure could not be written with; for a negative ``halo_width``; for one of the
                keywords in :data:`_LABEL_REFUSED_OPTS`; for an empty collection; or when the CRS question
                above cannot be answered.

        Examples:
            - Label each feature with its ``fid``, and count the annotations drawn:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> places = FeatureCollection.read_file("tests/data/points.geojson")
                >>> with Map(crs=places.epsg) as canvas:
                ...     drawn = canvas.labels(places, "fid", text_size=9.0)
                ...     len(drawn) == len(places)
                True

                ```

        See Also:
            digitalearth.static.maps.decoration.DecorationMixin.text: one string at one coordinate.
            digitalearth.static.maps.decoration.DecorationMixin.annotate: one string, with an arrow to a point.
        """
        # At the call as well as in the drawer, because the drawer settles the CRS question before it reaches
        # `_vector_input`'s guard — and an array would be refused for declaring no CRS rather than for being
        # an array, which is the wrong answer to the right question.
        require_drawable(features, caller=_LABELS_CALLER, accepts=("vector",))
        refused = sorted(set(opts).intersection(_LABEL_REFUSED_OPTS))
        if refused:
            reasons = "; ".join(
                f"{key}=: {_LABEL_REFUSED_OPTS[key]}" for key in refused
            )
            raise ValueError(f"{_LABELS_CALLER} does not take {refused} — {reasons}")
        if not isinstance(column, str) or not column.strip():
            raise ValueError(
                f"{_LABELS_CALLER} needs column= as the name of an attribute to read the text from; "
                f"got {column!r}"
            )
        text_size = _as_finite(text_size, "text_size", _LABELS_CALLER)
        halo_width = _as_finite(halo_width, "halo_width", _LABELS_CALLER)
        if halo_width < 0:
            raise ValueError(
                f"{_LABELS_CALLER} needs halo_width= as a width, so zero or more; got {halo_width!r}"
            )
        return self._draw(
            LayerRecord(
                "labels",
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "labels",
                        "column": column,
                        "text_size": text_size,
                        "color": color,
                        "halo_color": halo_color,
                        "halo_width": halo_width,
                        # A list, not the caller's tuple: a figure is written to JSON, which has no tuple, so
                        # a described tuple comes back as a list and the two symbologies stop comparing equal
                        # (`travels_in_a_figure`).
                        "offset": _label_offset(offset),
                        # In the shared CRS spelling, as `Map.text(crs=)` records one: a caller may hand in a
                        # CRS object, which a figure written to JSON has no form for.
                        "crs": crs_to_json(crs, "Map.labels(crs=)"),
                    }
                ),
                opts=opts,
            )
        )

    def grid_points(
        self,
        dataset: Any,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Any:
        """Plot raster cell centres as points coloured by value (pyramids ``to_xyz`` → ``ScatterGlyph``).

        Args:
            dataset: A pyramids ``Dataset``, or a path or URL to one (reprojected to the display CRS
                first). Only a path-backed layer can be written down.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Styling kwargs, filtered to ``ScatterGlyph``'s accepted options. ``size``
                sets the marker size — the cross-backend spelling, folded onto cleopatra's own
                ``point_size`` constructor keyword by the one place that knows about it.

        Returns:
            The scatter ``PathCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Examples:
            - Scatter the raster's valid cell centres and count the resulting layer:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> m = Map(crs=ds.epsg)
                >>> _ = m.grid_points(ds)
                >>> len(m.layers)
                1

                ```

        Raises:
            TypeError: when `dataset` is not a raster or a reference to one. A bare numpy array is refused
                here where :meth:`~digitalearth.static.maps.raster.RasterMixin.field` takes one: the cell
                centres come from pyramids' ``to_xyz``, which reads a geo-transform an array does not carry.
        """
        require_drawable(dataset, caller="Map.grid_points()", accepts=("raster",))
        record = LayerRecord(
            "points",
            source=dataset,
            name=name,
            visible=visible,
            symbology=Symbology(props={"via": "grid_points"}),
            opts=opts,
        )
        try:
            return self._draw(record)
        except OffLimbError:
            self._skipped_off_limb("grid_points")
            return None

    def point_cloud(
        self, dataset: Any, *, name: Optional[str] = None, visible: bool = True, **opts
    ) -> Any:
        """Alias of :meth:`grid_points` — scatter raster cell centres coloured by value.

        Args:
            dataset: A pyramids ``Dataset``, or a path or URL to one (reprojected to the display CRS
                first). Only a path-backed layer can be written down.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **opts: Styling kwargs, forwarded to :meth:`grid_points` unchanged.

        Returns:
            The scatter ``PathCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self.grid_points(dataset, name=name, visible=visible, **opts)

    def grid_cells(
        self,
        dataset: Any,
        band: int = DEFAULT_BAND,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Any:
        """Draw raster cells as value-coloured polygons (pyramids ``get_cell_polygons`` → ``PolygonGlyph``).

        Args:
            dataset: A pyramids ``Dataset``, or a path or URL to one (reprojected to the display CRS
                first). Only a path-backed layer can be written down.
            band: 1-based band whose values colour the cells.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: The caller's own engine keywords, handed to ``PolygonGlyph`` as passed and held beside
                the layer rather than recorded in it (see the class docstring). ``PolygonGlyph`` refuses a
                name it does not know rather than dropping it. A ``scheme`` (including
                ``scheme="categorical"``, keyed by a swatch legend) is honoured the same way :meth:`choropleth`
                describes — see ``_polygon_layer``.

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            TypeError: when `dataset` is not a raster or a reference to one. A bare numpy array is refused
                here where :meth:`~digitalearth.static.maps.raster.RasterMixin.field` takes one: the cell
                polygons come from pyramids' ``get_cell_polygons``, which reads a geo-transform an array does
                not carry.
            ValueError: from ``PolygonGlyph`` for a keyword in ``**opts`` it does not accept, naming the
                ones it does. The layer is dropped from the description again before it propagates.

        Examples:
            - Draw one polygon per raster cell and confirm the count equals rows*columns:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.dataset import Dataset
                >>> from digitalearth.static import Map
                >>> ds = Dataset.read_file("examples/data/acc4000.tif")
                >>> m = Map(crs=ds.epsg)
                >>> pc = m.grid_cells(ds)
                >>> len(pc.get_paths()) == ds.rows * ds.columns
                True

                ```
        """
        require_drawable(dataset, caller="Map.grid_cells()", accepts=("raster",))
        record = LayerRecord(
            # A raster's cells always carry their band's values, so this layer is always a choropleth —
            # `_polygon_kind` is named here for the same reason the outline builders name it: what makes a
            # polygon layer one kind or the other is whether its polygons are filled by a value.
            _polygon_kind(band),
            source=dataset,
            name=name,
            visible=visible,
            symbology=Symbology(props={"via": "grid_cells", "band": band}),
            opts=opts,
        )
        try:
            return self._draw(record)
        except OffLimbError:
            self._skipped_off_limb("grid_cells")
            return None

    def _vector(
        self,
        u_dataset: Any,
        v_dataset: Any,
        *,
        kind: str,
        band: int = DEFAULT_BAND,
        name: Optional[str] = None,
        visible: bool = True,
        **opts,
    ) -> Any:
        """Render a vector field from two rasters (u, v) on a shared grid via ``cleopatra.VectorGlyph``.

        Args:
            u_dataset: pyramids ``Dataset`` of the u (eastward) component, or a path or URL to one.
                Only a path-backed layer can be written down — and a u/v pair is not one of those yet,
                because a layer names one source and a field needs two.
            v_dataset: pyramids ``Dataset`` of the v (northward) component, or a path or URL to one.
            kind: ``"quiver"``, ``"barbs"`` or ``"streamplot"``.
            band: 1-based band index read from each dataset.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: The caller's own engine keywords, handed to ``VectorGlyph`` as passed and held beside
                the layer rather than recorded in it (see the class docstring). ``VectorGlyph`` refuses a
                name it does not know rather than dropping it.

        Returns:
            The vector mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            TypeError: when either component is not a raster or a reference to one. A bare numpy array is
                refused here where :meth:`~digitalearth.static.maps.raster.RasterMixin.field` takes one: the
                two components are read band-by-band through the display-CRS choke point, which places them
                against each other by their geo-transforms.
            ValueError: from ``VectorGlyph`` for a keyword in ``**opts`` it does not accept, naming the
                ones it does. The layer is dropped from the description again before it propagates.
            FileNotFoundError: when a component's path names nothing, or KeyError when no resolver is
                registered for its URL scheme (and ValueError for an empty one). Those come from
                :func:`_opened_component`, which runs **before** the layer is described, so there is
                nothing to drop.
        """
        for component in (u_dataset, v_dataset):
            require_drawable(component, caller=f"Map.{kind}()", accepts=("raster",))
        record = LayerRecord(
            _VECTOR_KINDS[kind],
            # The pair is the source: a field is not drawable from either component alone. It is opened
            # here rather than left to the figure's source table, because that table names **one** source
            # per layer and so has no spelling for a pair — see `_opened_component` (round 2, L7).
            source=(_opened_component(u_dataset), _opened_component(v_dataset)),
            name=name,
            visible=visible,
            symbology=Symbology(props={"via": kind, "band": band}),
            opts=opts,
        )
        try:
            return self._draw(record)
        except OffLimbError:
            self._skipped_off_limb(kind)
            return None

    def quiver(
        self,
        u_dataset: Any,
        v_dataset: Any,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Draw a vector field as arrows (``VectorGlyph`` ``kind="quiver"``).

        Args:
            u_dataset: pyramids ``Dataset`` of the u (eastward) component, or a path or URL to one.
                Only a path-backed layer can be written down — and a u/v pair is not one of those yet,
                because a layer names one source and a field needs two.
            v_dataset: pyramids ``Dataset`` of the v (northward) component, or a path or URL to one.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Forwarded to :meth:`_vector` — ``band`` is the one named argument; the rest is the
                caller's own ``VectorGlyph`` styling, split between the layer's description and the scene
                (see the class docstring).

        Returns:
            The ``Quiver`` mappable (registered as a Scene layer; carries the key for :meth:`quiverkey`).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: from ``VectorGlyph`` for a styling keyword it does not accept, or when a component
                is named by an empty string.
            FileNotFoundError: when a component's path names nothing.
            KeyError: when no resolver is registered for a component's URL scheme.
        """
        return self._vector(
            u_dataset, v_dataset, kind="quiver", name=name, visible=visible, **kwargs
        )

    def barbs(
        self,
        u_dataset: Any,
        v_dataset: Any,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs,
    ) -> Any:
        """Draw a vector field as wind barbs (``VectorGlyph`` ``kind="barbs"``).

        Args:
            u_dataset: pyramids ``Dataset`` of the u (eastward) component, or a path or URL to one.
                Only a path-backed layer can be written down — and a u/v pair is not one of those yet,
                because a layer names one source and a field needs two.
            v_dataset: pyramids ``Dataset`` of the v (northward) component, or a path or URL to one.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Forwarded to :meth:`_vector` — ``band`` is the one named argument; the rest is the
                caller's own ``VectorGlyph`` styling, split between the layer's description and the scene
                (see the class docstring).

        Returns:
            The ``Barbs`` mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: from ``VectorGlyph`` for a styling keyword it does not accept, or when a component
                is named by an empty string.
            FileNotFoundError: when a component's path names nothing.
            KeyError: when no resolver is registered for a component's URL scheme.
        """
        return self._vector(
            u_dataset, v_dataset, kind="barbs", name=name, visible=visible, **kwargs
        )

    def streamplot(
        self,
        u_dataset: Any,
        v_dataset: Any,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Draw a vector field as streamlines (``VectorGlyph`` ``kind="streamplot"``).

        Args:
            u_dataset: pyramids ``Dataset`` of the u (eastward) component, or a path or URL to one.
                Only a path-backed layer can be written down — and a u/v pair is not one of those yet,
                because a layer names one source and a field needs two.
            v_dataset: pyramids ``Dataset`` of the v (northward) component, or a path or URL to one.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: Forwarded to :meth:`_vector` — ``band`` is the one named argument; the rest is the
                caller's own ``VectorGlyph`` styling, split between the layer's description and the scene
                (see the class docstring).

        Returns:
            The streamplot mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: from ``VectorGlyph`` for a styling keyword it does not accept, or when a component
                is named by an empty string.
            FileNotFoundError: when a component's path names nothing.
            KeyError: when no resolver is registered for a component's URL scheme.
        """
        return self._vector(
            u_dataset,
            v_dataset,
            kind="streamplot",
            name=name,
            visible=visible,
            **kwargs,
        )

    def quiverkey(
        self,
        value: float,
        text: str,
        *,
        x: float = 0.9,
        y: float = 0.95,
        labelpos: str = "E",
        **kwargs,
    ) -> Any:
        """Draw the labelled reference arrow for the most recent :meth:`quiver` layer.

        Places a sample arrow of known magnitude with a label (via ``Axes.quiverkey`` on the stored quiver
        artist), so readers can scale the field. Only ``quiver`` arrows carry a key; ``barbs``/``streamplot``
        do not.

        Args:
            value: The reference magnitude the sample arrow represents (data units, e.g. ``10`` for 10 m/s).
            text: The label drawn next to the arrow (e.g. ``"10 m/s"``).
            x: Arrow x position in axes fraction (0–1). Default ``0.9``.
            y: Arrow y position in axes fraction (0–1). Default ``0.95``.
            labelpos: Side of the arrow for the label (``"N"``/``"S"``/``"E"``/``"W"``). Default ``"E"``.
            **kwargs: Forwarded to ``Axes.quiverkey`` (e.g. ``coordinates``, ``color``, ``fontproperties``).

        Returns:
            The :class:`matplotlib.quiver.QuiverKey` added to the axes.

        Raises:
            ValueError: if no :meth:`quiver` layer has been drawn yet (``barbs``/``streamplot`` have no key).
        """
        if self._last_vector is None or self._last_vector[2] != "quiver":
            raise ValueError(
                "quiverkey() needs a prior quiver(...) layer (barbs/streamplot have no key)"
            )
        _, artist, _ = self._last_vector
        return self.ax.quiverkey(artist, x, y, value, text, labelpos=labelpos, **kwargs)

    def _scattered(self, data: Any) -> tuple:
        """Return ``(x, y, z, field)`` for unstructured/point input (Dataset cells or a FeatureCollection).

        Args:
            data: A pyramids ``Dataset``, whose cells become points through ``to_xyz``, or a
                ``FeatureCollection`` of points. Either is reprojected to the display CRS first.

        Returns:
            Three parallel 1-D arrays — the x coordinates, the y coordinates and the value at each point — in
            the display CRS, and the **name** those values are known by: the raster's band or the
            collection's numeric column, as :func:`~digitalearth.static.guides.source_field` spells it. The
            name is returned here rather than resolved again by the drawer because this is where the source is
            opened, and it is what the layer publishes its colour encoding with (#261, order 24).

        Raises:
            ValueError: when a ``FeatureCollection`` carries no numeric column to take the value from,
                since there is nothing to contour.
        """
        if isinstance(data, Dataset):
            xyz = self._reproject(data).to_xyz()
            return (
                xyz.iloc[:, 0].to_numpy(),
                xyz.iloc[:, 1].to_numpy(),
                xyz.iloc[:, 2].to_numpy(),
                source_field(data),
            )
        src = get_source(reproject(data, self.crs))
        if src.z is None:
            raise ValueError("FeatureCollection has no numeric value column to contour")
        return src.x.values, src.y.values, src.z.values, source_field(src)

    def _tri(
        self,
        data: Any,
        *,
        kind: str,
        name: Optional[str] = None,
        visible: bool = True,
        **opts,
    ) -> Any:
        """Triangulate scattered points and render via ``cleopatra.MeshGlyph``.

        Args:
            data: A pyramids ``Dataset`` (its cells become points) or a ``FeatureCollection``, or a path
                or URL to either. It is recorded as the layer's source as given, so only a path-backed
                layer can be written down.
            kind: The triangulated render to draw (``tricontourf`` / ``tricontour`` /
                ``tripcolor``).
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: The caller's own engine keywords, forwarded to the glyph as passed and held beside the
                layer rather than recorded in it (see the class docstring).

        Returns:
            The mappable (registered as a Scene layer), or ``None`` when three points were supplied
            but fewer than three survive the reprojection — a vector warp sends off-view points to
            infinity rather than raising, and losing them all means the same as an off-limb raster:
            nothing on the view to draw.

        Raises:
            ValueError: when fewer than three points were supplied in the first place, which no
                projection can fix.
            AttributeError: from matplotlib for a keyword in ``**opts`` no artist property answers to —
                these renders reach the artist rather than a cleopatra option table, so an unknown name
                is refused there and by its message. The layer is dropped from the description again
                before either propagates.
        """
        # All three triangulated renders describe one thing — a mesh built from scattered points — which is
        # what the registry calls `unstructured`; the render itself is the `via` property.
        record = LayerRecord(
            "unstructured",
            source=data,
            name=name,
            visible=visible,
            symbology=Symbology(props={"via": kind}),
            opts=opts,
        )
        try:
            return self._draw(record)
        except OffLimbError:
            self._skipped_off_limb(kind)
            return None

    def tricontourf(
        self, data: Any, *, name: Optional[str] = None, visible: bool = True, **kwargs
    ) -> Any:
        """Filled contours of unstructured/point data (``MeshGlyph`` node data, ``filled=True``).

        Args:
            data: A pyramids ``Dataset`` (its cells become points) or a ``FeatureCollection``, or a
                path or URL to either. Only a path-backed layer can be written down.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: The caller's own engine styling, forwarded through :meth:`_tri` to the glyph and
                held beside the layer rather than described.

        Returns:
            The tricontourf mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: when fewer than three points were supplied, or a ``FeatureCollection`` carries no
                numeric column to contour.
            AttributeError: from matplotlib for a styling keyword no artist property answers to.
        """
        return self._tri(data, kind="tricontourf", name=name, visible=visible, **kwargs)

    def tricontour(
        self, data: Any, *, name: Optional[str] = None, visible: bool = True, **kwargs
    ) -> Any:
        """Line contours of unstructured/point data (``MeshGlyph`` node data, ``filled=False``).

        Args:
            data: A pyramids ``Dataset`` (its cells become points) or a ``FeatureCollection``, or a
                path or URL to either. Only a path-backed layer can be written down.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: The caller's own engine styling, forwarded through :meth:`_tri` to the glyph and
                held beside the layer rather than described.

        Returns:
            The tricontour mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: when fewer than three points were supplied, or a ``FeatureCollection`` carries no
                numeric column to contour.
            AttributeError: from matplotlib for a styling keyword no artist property answers to.
        """
        return self._tri(data, kind="tricontour", name=name, visible=visible, **kwargs)

    def tripcolor(
        self, data: Any, *, name: Optional[str] = None, visible: bool = True, **kwargs
    ) -> Any:
        """Flat-shaded triangles of unstructured/point data (``MeshGlyph`` face data).

        Args:
            data: A pyramids ``Dataset`` (its cells become points) or a ``FeatureCollection``, or a
                path or URL to either. Only a path-backed layer can be written down.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the drawing (#327).
            **kwargs: The caller's own engine styling, forwarded through :meth:`_tri` to the glyph and
                held beside the layer rather than described.

        Returns:
            The tripcolor mappable (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: when fewer than three points were supplied, or a ``FeatureCollection`` carries no
                numeric column to contour.
            AttributeError: from matplotlib for a styling keyword no artist property answers to.
        """
        return self._tri(data, kind="tripcolor", name=name, visible=visible, **kwargs)

    @staticmethod
    def _polygon_vertices(geometry: Any) -> tuple:
        """Return (vertex-arrays, repeat-counts) for a geopandas geometry series.

        Polygons contribute their exterior ring; MultiPolygons contribute one ring per part (so a single
        feature can map to several drawn polygons). The repeat count per feature lets callers expand a
        per-feature value array to per-polygon.

        Args:
            geometry: An iterable of shapely ``Polygon``/``MultiPolygon`` geometries — a geopandas
                geometry series, already in the display CRS.

        Returns:
            ``(polygons, repeats)``: one ``(n, 2)`` array of exterior-ring coordinates per drawn polygon,
            and one count per input feature saying how many of those polygons it contributed — ``1`` for a
            ``Polygon``, the number of parts for a ``MultiPolygon``.
        """
        polygons: List[np.ndarray] = []
        repeats: List[int] = []
        for geom in geometry:
            parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            polygons.extend(np.asarray(p.exterior.coords) for p in parts)
            repeats.append(len(parts))
        return polygons, repeats

    @staticmethod
    def _finite_polygons(
        polygons: List[np.ndarray], values: Optional[np.ndarray] = None
    ) -> tuple:
        """Drop polygons with any non-finite vertex (and the matching values).

        On a projected/globe map the far hemisphere reprojects to non-finite coordinates; matplotlib's
        ``PolyCollection`` would otherwise receive ``inf``/``nan`` vertices and fail or draw garbage. Keeps
        only fully-finite rings, preserving order and the positional alignment between rings and ``values``.

        Args:
            polygons: Polygon rings as ``(N, 2)`` vertex arrays; any ring with an ``inf``/``nan`` vertex is
                dropped.
            values: Optional per-polygon scalar values, positionally aligned with ``polygons``. When given,
                the same rings are dropped from it so the two stay aligned.

        Returns:
            ``(kept_polygons, kept_values)`` — the surviving rings, and either the filtered ``values`` array
            or ``None`` when ``values`` was ``None``.

        Examples:
            - A ring with an ``inf`` vertex is dropped along with its value:
                ```python
                >>> import numpy as np
                >>> from digitalearth.static import Map
                >>> good = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
                >>> bad = np.array([[0.0, 0.0], [np.inf, 0.0], [1.0, 1.0]])
                >>> kept, vals = Map._finite_polygons([good, bad, good], np.array([10.0, 20.0, 30.0]))
                >>> len(kept)
                2
                >>> vals.tolist()
                [10.0, 30.0]

                ```
            - Without values, only the finite rings come back and the second slot is ``None``:
                ```python
                >>> import numpy as np
                >>> from digitalearth.static import Map
                >>> good = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
                >>> bad = np.array([[np.nan, 0.0], [1.0, 0.0], [1.0, 1.0]])
                >>> kept, vals = Map._finite_polygons([good, bad])
                >>> len(kept), vals
                (1, None)

                ```
        """
        keep = [i for i, p in enumerate(polygons) if np.isfinite(p).all()]
        kept = [polygons[i] for i in keep]
        if values is None:
            return kept, None
        return kept, np.asarray(values)[keep]

    @_skips_off_limb
    def choropleth(
        self,
        features: Any,
        column: str,
        *,
        scheme: Optional[Any] = None,
        k: int = 5,
        cmap: Optional[Any] = None,
        opacity: Optional[float] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Any:
        """Fill polygons coloured by a feature attribute (pyramids ``FeatureCollection`` → ``PolygonGlyph``).

        Args:
            features: A pyramids ``FeatureCollection`` of polygons, or a path or URL to one
                (reprojected to the display CRS). Only a path-backed layer can be written down.
            column: Name of the column whose values colour the polygons — numeric for a continuous or
                graduated scale, or any nominal labels (strings, region codes, …) under
                ``scheme="categorical"``.
            scheme: How the values are classified. ``None`` (default) is a **continuous** ramp; a named
                scheme (``"quantiles"``, ``"fisher_jenks"``, ``"equal_interval"``, …) or an explicit
                sequence of bin edges is graduated into ``k`` classes; ``"categorical"`` gives every
                distinct value its own colour (an unordered attribute such as a land-use class or region
                name — ``k`` does not apply, and ``vmin``/``vmax``/``levels``/``color_scale`` are ignored),
                keyed by a swatch legend rather than a colorbar. Spelled the same way, with the same
                default, on every backend: ``scheme=None`` is a continuous ramp everywhere, so the same
                call classifies identically on all four tiers. Either kind of classification leaves a
                missing value (``NaN``/``None``/``pd.NA``) outside every class, so those features are
                drawn a neutral grey — not dropped, and not painted as the lowest class. A continuous
                ramp has no classes and is left to matplotlib.
            k: Number of classes a named ``scheme`` is cut into (ignored when ``scheme`` is ``None`` or
                ``"categorical"``).
            opacity: How opaque the fill is, in ``[0, 1]``. This is the spelling
                :data:`~digitalearth.base.spec.encoding.CHANNELS` and the Core contract use, and the one the
                web and 3-D tiers take, so the same channel is written the same way on every tier (#332).
                matplotlib's own ``alpha`` is what it arrives as. ``None`` (default) sets no opacity at
                all, which is not the same as ``1.0``: it leaves the colormap's own alpha channel in force.
            cmap: The colormap the fill is drawn with, forwarded to ``PolygonGlyph`` exactly as a
                ``cmap=`` in ``**opts`` always was. Named in the signature because the Core declares it
                as a keyword of ``choropleth`` on every tier, and a keyword that works but is not
                written down is one a caller has to read the source to find (#324).
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Styling kwargs forwarded to ``PolygonGlyph``. For a categorical scheme, ``cmap`` should
                be a **qualitative** (``ListedColormap``) map — ``"tab10"`` (the default), ``"Set2"``,
                ``"Paired"``, … A continuous ``LinearSegmentedColormap`` (``"coolwarm"``, ``"RdBu"``) is
                sampled at evenly-spaced points so the categories stay distinct; a perceptual
                ``ListedColormap`` (``"viridis"``, ``"plasma"``) is accepted but reads poorly (its first
                *n* of 256 entries are near-identical shades). The colours are identical to the
                web/interactive tiers either way.

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Examples:
            - Colour buffered point features by their ``fid`` column and count the drawn polygons:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> fc["geometry"] = fc.geometry.buffer(500.0)
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.choropleth(fc, column="fid")
                >>> len(pc.get_paths()) >= len(fc)
                True

                ```
            - Colour by an unordered attribute — one colour per distinct class, keyed by a swatch legend:
                ```python
                >>> fc["zone"] = ["urban", "rural"] * (len(fc) // 2) + ["urban"] * (len(fc) % 2)
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.choropleth(fc, column="zone", scheme="categorical")
                >>> from matplotlib.colors import BoundaryNorm
                >>> isinstance(pc.norm, BoundaryNorm)  # discrete class codes, not a continuous scale
                True
                >>> [t.get_text() for t in m.layers[-1][0].category_legend.get_texts()]
                ['rural', 'urban']

                ```
        """
        if cmap is not None:
            # Straight back into the caller's keywords: this is where `cmap=` has always travelled, so
            # naming it in the signature documents the keyword without moving it.
            opts["cmap"] = cmap
        # The channel is named `opacity` here and `alpha` where it reaches matplotlib, so it travels on
        # under the engine's own spelling — exactly where a caller's `alpha=` always went.
        if opacity is not None:
            opts["alpha"] = opacity
        return self._draw(
            LayerRecord(
                # A column is required, so the polygons are always filled by a value: a choropleth.
                _polygon_kind(column),
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "choropleth",
                        "column": column,
                        "scheme": scheme,
                        "k": k,
                    }
                ),
                # The classification is folded into the glyph's keywords by the drawer, so what the figure
                # records is the call the caller made; their own engine keywords travel beside it.
                opts=opts,
            )
        )

    @_skips_off_limb
    def polygons(
        self,
        features: Any,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Any:
        """Draw polygon outlines without fill (pyramids ``FeatureCollection`` → ``PolygonGlyph`` outline mode).

        Args:
            features: A pyramids ``FeatureCollection`` of polygons, or a path or URL to one
                (reprojected to the display CRS). Only a path-backed layer can be written down.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Styling kwargs, filtered to ``PolygonGlyph``'s accepted options.

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.
        """
        return self._draw(
            LayerRecord(
                _polygon_kind(None),  # outlines only, whatever the collection carries
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(props={"via": "shapes"}),
                opts=opts,
            )
        )

    def _clip_geometry(self, clip: Any) -> Any:
        """Resolve a clip boundary to a single geometry in the display CRS, or ``None``.

        Accepts a pyramids ``FeatureCollection`` / geopandas ``GeoDataFrame``/``GeoSeries`` (reprojected to the
        display CRS and unioned) or a shapely geometry (assumed already in the display CRS). ``None`` → no clip.

        Args:
            clip: A ``FeatureCollection``/``GeoDataFrame``/``GeoSeries`` (reprojected + unioned), a shapely
                geometry already in the display CRS, or ``None``.

        Returns:
            A single shapely geometry in the display CRS, or ``None`` when ``clip`` is ``None``.
        """
        if clip is None:
            return None
        if hasattr(clip, "to_crs"):  # FeatureCollection / GeoDataFrame / GeoSeries
            geoms = clip.to_crs(self.crs)
            geoms = geoms.geometry if hasattr(geoms, "geometry") else geoms
            return geoms.union_all()
        return clip  # shapely geometry, assumed already in the display CRS

    @staticmethod
    def _finite_point_xy(geom: Any, values: Optional[np.ndarray] = None):
        """Return ``(xs, ys, values)`` for points with finite coordinates.

        Points that reproject to non-finite coordinates (the far side of a clipped/globe display CRS) are
        dropped — together with their matching ``values`` — so downstream tessellation / binning / KDE never
        receives ``inf`` / ``nan``.

        This is now a thin adapter over :class:`~digitalearth.base.points.PointArrays`, which is this helper
        lifted into `base/` so the other three tiers stop hand-rolling it. Kept as a method because the three
        call sites here want the flat ``(xs, ys, values)`` triple rather than the value object.

        Args:
            geom: A geopandas point ``GeoSeries`` (already in the display CRS).
            values: Optional per-point array aligned with ``geom``; filtered by the same finite mask.

        Returns:
            tuple: ``(xs, ys, values)`` numpy arrays of finite points; ``values`` is ``None`` when not given.
        """
        points, (filtered,) = PointArrays.of(geom.x, geom.y).finite(values)
        return points.x, points.y, filtered

    @_skips_off_limb
    def voronoi(
        self,
        features: Any,
        column: Optional[str] = None,
        *,
        clip: Any = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts,
    ) -> Any:
        """Voronoi diagram of a point ``FeatureCollection`` (pyramids points → cells → ``PolygonGlyph``).

        Tessellates the points into Voronoi cells (``shapely.voronoi_polygons`` with ``ordered=True``, so cell
        *i* belongs to point *i*) and renders them. With ``column`` the cells are filled and coloured by that
        point's value (like :meth:`choropleth`); without it only the cell outlines are drawn (like
        :meth:`polygons`). Points that reproject to non-finite coordinates (the far side of a clipped/globe
        display CRS), and duplicate points, produce no cell and are silently skipped.

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries, or a path or URL to one
                (reprojected to the display CRS). Only a path-backed layer can be written down.
            column: Name of the numeric column whose value colours each cell, or ``None`` for outlines only.
            clip: Optional boundary the cells are clipped to — a ``FeatureCollection``/``GeoDataFrame`` (reprojected
                to the display CRS) or a shapely geometry already in the display CRS. ``None`` leaves shapely's
                default bounded cells.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Styling kwargs forwarded to ``PolygonGlyph``. Pass ``scheme`` (e.g. ``"quantiles"`` /
                ``"fisher_jenks"``) + ``k`` to colour cells by discrete classes instead of a continuous scale,
                or ``scheme="categorical"`` for one colour per distinct value (keyed by a swatch legend the same
                way :meth:`choropleth` describes — see ``_polygon_layer``).

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: if ``features`` is not all single ``Point`` geometries.

        Examples:
            - Tessellate the point fixture and colour the cells by ``fid``:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.voronoi(fc, column="fid")
                >>> len(m.layers)
                1

                ```
        """
        return self._draw(
            LayerRecord(
                # With a column the cells are filled by its value, without one only their outlines are
                # drawn — the same recipe, two kinds, decided by the argument rather than by the drawing.
                _polygon_kind(column),
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(props={"via": "voronoi", "column": column}),
                opts=opts,
                # The boundary is a shapely geometry or a feature collection: a figure written to JSON has
                # no spelling for either, and two symbologies holding a GeoDataFrame cannot be compared at
                # all. It travels with the scene instead, and is forgotten with the layer.
                key=clip,
            )
        )

    @staticmethod
    def _scale_factors(values: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
        """Linearly map ``values`` onto ``limits`` (a constant input maps to the midpoint factor).

        Args:
            values: Per-feature magnitudes to normalise.
            limits: ``(min, max)`` output range the smallest / largest value map to.

        Returns:
            np.ndarray: Per-feature scale factors spanning ``limits``, the same shape as ``values``.
        """
        lo, hi = limits
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if vmax > vmin:
            return lo + (values - vmin) * (hi - lo) / (vmax - vmin)
        return np.full(values.shape, (lo + hi) / 2.0)

    @_skips_off_limb
    def cartogram(
        self,
        features: Any,
        scale: str,
        column: Optional[str] = None,
        *,
        limits: Tuple[float, float] = (0.2, 1.0),
        name: Optional[str] = None,
        visible: bool = True,
        **opts,
    ) -> Any:
        """Cartogram: scale each polygon about its centroid by a value column (pyramids → ``PolygonGlyph``).

        Each feature's geometry is affine-scaled about its own centroid by a factor derived from ``scale``
        (linearly normalised across the layer to ``limits``), distorting area to encode magnitude. With
        ``column`` the scaled polygons are filled and coloured by that column (like :meth:`choropleth`);
        without it only the outlines are drawn (like :meth:`polygons`).

        Args:
            features: A pyramids ``FeatureCollection`` of polygon geometries, or a path or URL to one
                (reprojected to the display CRS). Only a path-backed layer can be written down.
            scale: Name of the numeric column whose value sets each feature's size (normalised to ``limits``).
            column: Optional column whose value colours each scaled polygon, or ``None`` for outlines only.
            limits: ``(min, max)`` scale factors mapped to the smallest/largest ``scale`` value.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Styling kwargs forwarded to ``PolygonGlyph``. Pass ``scheme`` (e.g. ``"quantiles"`` /
                ``"fisher_jenks"``) + ``k`` to colour by discrete classes instead of a continuous scale, or
                ``scheme="categorical"`` for one colour per distinct value (keyed by a swatch legend the same
                way :meth:`choropleth` describes — see ``_polygon_layer``).

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: if ``features`` contains non-polygon geometry.

        Examples:
            - Scale buffered points by ``fid`` and colour them by the same column:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> fc["geometry"] = fc.geometry.buffer(500.0)
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.cartogram(fc, scale="fid", column="fid")
                >>> len(m.layers)
                1

                ```
        """
        return self._draw(
            LayerRecord(
                # As in `voronoi`: a column fills the scaled polygons, no column leaves their outlines.
                _polygon_kind(column),
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "cartogram",
                        "scale": scale,
                        "column": column,
                        "limits": tuple(limits),
                    }
                ),
                opts=opts,
            )
        )

    @staticmethod
    def _quadtree_cells(
        xs: np.ndarray,
        ys: np.ndarray,
        agg_fn: Any,
        nmax: int,
        nmin: int,
        max_depth: int = 20,
    ) -> List[Tuple[float, float, float, float, float]]:
        """Recursively split the points' bbox into quadrants until each cell holds ``<= nmax`` points.

        Returns ``(xmin, ymin, xmax, ymax, value)`` per kept cell, where ``value = agg_fn(point_indices)``.
        A cell with fewer than ``nmin`` points is dropped; splitting stops at ``max_depth`` and when a split
        makes no progress (all points fall in one child), so coincident points cannot recurse forever.

        Args:
            xs: Finite point x-coordinates.
            ys: Finite point y-coordinates, aligned with ``xs``.
            agg_fn: Callable mapping an index array of the points in a cell to that cell's scalar value.
            nmax: Maximum points in a cell before it is split.
            nmin: Cells with fewer than this many points are dropped.
            max_depth: Hard recursion-depth cap guarding against coincident points. Default 20.

        Returns:
            list[tuple]: ``(xmin, ymin, xmax, ymax, value)`` for each kept cell.
        """
        x0, x1 = float(np.min(xs)), float(np.max(xs))
        y0, y1 = float(np.min(ys)), float(np.max(ys))
        if x1 <= x0:
            x1 = x0 + 1.0
        if y1 <= y0:
            y1 = y0 + 1.0
        out: List[Tuple[float, float, float, float, float]] = []
        stack = [(x0, y0, x1, y1, np.arange(len(xs)), 0)]
        while stack:
            xmin, ymin, xmax, ymax, idx, depth = stack.pop()
            n = len(idx)
            if n == 0:
                continue
            if n <= nmax or depth >= max_depth:
                if n >= nmin:
                    out.append((xmin, ymin, xmax, ymax, float(agg_fn(idx))))
                continue
            quads = _quadrants(xs, ys, idx, (xmin, ymin, xmax, ymax))
            if _split_made_no_progress(quads, n):
                # Coincident points: every child holds the whole cell, so splitting again would recurse
                # forever. The cell is kept whole instead.
                if n >= nmin:
                    out.append((xmin, ymin, xmax, ymax, float(agg_fn(idx))))
                continue
            for qx0, qy0, qx1, qy1, qidx in quads:
                stack.append((qx0, qy0, qx1, qy1, qidx, depth + 1))
        return out

    @_skips_off_limb
    def quadtree(
        self,
        features: Any,
        column: Optional[str] = None,
        *,
        agg: Any = DEFAULT_QUADTREE_AGG,
        nmax: int = 100,
        nmin: int = 0,
        clip: Any = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts,
    ) -> Any:
        """Quadtree choropleth: aggregate points into adaptive cells (pyramids points → ``PolygonGlyph``).

        Recursively splits the points' bounding box into quadrants until each cell holds ``<= nmax`` points,
        then colours each cell by an aggregate of ``column`` (or the point **count** when ``column`` is
        ``None``). The cells are always filled (a quadtree is a choropleth).

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries, or a path or URL to one
                (reprojected to the display CRS). Only a path-backed layer can be written down.
            column: Numeric column aggregated per cell, or ``None`` to colour by point count (density).
            agg: Per-cell reducer — one of ``"mean"``/``"sum"``/``"median"``/``"min"``/``"max"``/``"std"``/
                ``"count"`` or a callable taking a 1-D array. Ignored when ``column`` is ``None`` (count).
                A **name** is recorded in the figure, because it decides which values the cells carry and
                every reader resolves it to the same reducer. A **callable** is held beside the layer like
                any other engine object, so a figure read back elsewhere aggregates by
                :data:`DEFAULT_QUADTREE_AGG` instead.
            nmax: Maximum points in a cell before it is split (smaller → finer grid).
            nmin: Cells with fewer than this many points are dropped.
            clip: Optional boundary the cells are clipped to (``FeatureCollection``/``GeoDataFrame`` reprojected,
                or a shapely geometry in the display CRS). ``None`` keeps the full rectangular cells.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Styling kwargs forwarded to ``PolygonGlyph``. Pass ``scheme`` (e.g. ``"quantiles"`` /
                ``"fisher_jenks"``) + ``k`` to colour cells by discrete classes instead of a continuous scale,
                or ``scheme="categorical"`` for one colour per distinct value (keyed by a swatch legend the same
                way :meth:`choropleth` describes — see ``_polygon_layer``).

        Returns:
            The ``PolyCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: if ``features`` is not all single ``Point`` geometries, or ``agg`` is an unknown name.

        Examples:
            - Aggregate the point fixture into a fine density grid:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> m = Map(crs=fc.epsg)
                >>> pc = m.quadtree(fc, nmax=1)
                >>> len(m.layers)
                1

                ```
        """
        recorded_agg, held_agg = _described_agg(agg)
        return self._draw(
            LayerRecord(
                # A quadtree is always filled — by the column's aggregate, or by the point count.
                _polygon_kind(column or "count"),
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "quadtree",
                        "column": column,
                        "agg": recorded_agg,
                        "nmax": nmax,
                        "nmin": nmin,
                    }
                ),
                opts=opts,
                # What a figure cannot carry: `clip` is a geometry (see `voronoi`), and a reducer the
                # caller wrote themselves is a function. A *named* reducer is a plain string that decides
                # which values the cells carry, so it is described rather than held (round 2, M2).
                key=(held_agg, clip),
            )
        )

    @staticmethod
    def _polygons_of(geom: Any) -> list:
        """Return the ``Polygon`` parts of a shapely geometry (``[]`` for non-polygonal input).

        Args:
            geom: Any shapely geometry, or ``None``.

        Returns:
            list: ``[geom]`` for a Polygon, each Polygon part of a MultiPolygon / GeometryCollection, or
            ``[]`` for ``None`` / non-polygonal geometry.
        """
        if geom is None:
            return []
        kind = geom.geom_type
        if kind == "Polygon":
            return [geom]
        if kind in ("MultiPolygon", "GeometryCollection"):
            return [g for g in geom.geoms if g.geom_type == "Polygon"]
        return []

    def _clip_path(self, clip: Any) -> Optional[MplPath]:
        """Resolve a clip boundary to a matplotlib ``Path`` (data coords) for contour clipping, or ``None``.

        Reuses :meth:`_clip_geometry` to reproject/union the boundary, then turns each polygon exterior ring
        into a sub-path so a ``MultiPolygon`` clips correctly.

        Args:
            clip: A clip boundary accepted by :meth:`_clip_geometry`, or ``None``.

        Returns:
            matplotlib.path.Path | None: A path (in data coords) covering the boundary polygons, or ``None``
            when there is no usable boundary.
        """
        polys = self._polygons_of(self._clip_geometry(clip))
        verts: List[list] = []
        codes: List[int] = []
        for poly in polys:
            ring = np.asarray(poly.exterior.coords)
            if len(ring) < 3:
                continue
            verts.extend(ring.tolist())
            codes.extend(
                [MplPath.MOVETO]
                + [MplPath.LINETO] * (len(ring) - 2)
                + [MplPath.CLOSEPOLY]
            )
        if not verts:
            return None
        return MplPath(np.asarray(verts), codes)

    @_skips_off_limb
    def hexbin(
        self,
        features: Any,
        column: Optional[str] = None,
        *,
        reduce: Any = DEFAULT_HEXBIN_REDUCE,
        gridsize: Any = 50,
        min_count: Optional[int] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Any:
        """Aggregate points onto a hexagonal lattice and colour each cell (pyramids points → ``HexbinGlyph``).

        The discrete counterpart of :meth:`kde`: with no ``column`` each cell is coloured by how many points
        fell in it, and with one by the ``reduce`` of their values — a number read straight off the colour
        key, with none of the over-plotting a dense scatter has. The points are reprojected to the display
        CRS first and the lattice is laid there, so the cells are regular on the map rather than equal-area
        on the ground. The layer is a ``choropleth`` of the aggregate, as :meth:`quadtree`'s is.

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries, or a path or URL to one
                (reprojected to the display CRS). Only a path-backed layer can be written down.
            column: Numeric column aggregated per cell, or ``None`` (default) to count the points.
            reduce: How a cell's values aggregate: ``"mean"`` (default), ``"sum"``, ``"min"``, ``"max"``,
                ``"std"``, ``"count"``, or a function of an array. A named reducer is written into the
                figure; a function is held beside the layer, and a figure read back elsewhere falls back to
                ``"mean"``. Ignored without a ``column``.
            gridsize: Hexagons across the binned window — a whole number, or an ``(nx, ny)`` pair of
                them. One cell across is the smallest lattice there is, so the counts must be positive.
            min_count: Leave out cells with fewer points than this, as a whole number of points.
                ``None`` (default) leaves out the empty cells of a count map — which cleopatra would
                otherwise draw as zeros across the whole window — and is cleopatra's own default with a
                ``column``.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Further ``HexbinGlyph`` options and the shared styling (``cmap``, ``vmin``, ``vmax``,
                ``scheme``, ``k``, ``color_scale``, ``edge_color``, ``line_width``, ``extent``, …).

        Returns:
            The cells' ``PolyCollection`` (registered as a Scene layer), or ``None`` when the data lies
            entirely outside what the display CRS shows.

        Raises:
            ValueError: if ``features`` is empty or holds non-point geometry; if ``gridsize`` is not a
                positive whole number (or a pair of them), or ``min_count`` is not a non-negative whole
                number — both are checked in the builder, since unchecked they reach matplotlib's
                ``hexbin`` and come back as arithmetic naming neither the method nor the argument.
            KeyError: if ``column`` names no feature attribute.

        Examples:
            - Count the points per cell; three coincident points are one cell holding 3:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth import Map
                >>> wells = FeatureCollection(gpd.GeoDataFrame(
                ...     {"depth": [2.0, 4.0, 6.0, 100.0]},
                ...     geometry=[Point(0, 0), Point(0, 0), Point(0, 0), Point(8, 8)],
                ...     crs="EPSG:4326",
                ... ))
                >>> m = Map(crs=4326)
                >>> sorted(m.hexbin(wells, gridsize=4).get_array().tolist())
                [1.0, 3.0]

                ```
            - Aggregate a column instead; the origin cell is the mean of 2, 4 and 6:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth import Map
                >>> wells = FeatureCollection(gpd.GeoDataFrame(
                ...     {"depth": [2.0, 4.0, 6.0, 100.0]},
                ...     geometry=[Point(0, 0), Point(0, 0), Point(0, 0), Point(8, 8)],
                ...     crs="EPSG:4326",
                ... ))
                >>> m = Map(crs=4326)
                >>> sorted(m.hexbin(wells, "depth", reduce="mean", gridsize=4).get_array().tolist())
                [4.0, 100.0]
                >>> m.figure_spec.layers.get(m.layer_ids[-1]).symbology.props["via"]
                'hexbin'

                ```

        See Also:
            kde: the smoothed density of the same points.
            quadtree: points aggregated into adaptive square cells.
        """
        gridsize, min_count = _hexbin_lattice(gridsize, min_count)
        held_reduce = None if isinstance(reduce, str) else reduce
        return self._draw(
            LayerRecord(
                "choropleth",
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "hexbin",
                        "column": column,
                        "reduce": reduce if isinstance(reduce, str) else None,
                        "gridsize": gridsize,
                        "min_count": min_count,
                    }
                ),
                opts=opts,
                key=held_reduce,
            )
        )

    @_skips_off_limb
    def kde(
        self,
        features: Any,
        *,
        clip: Any = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts,
    ) -> Any:
        """2-D kernel-density (isochrone) plot of a point ``FeatureCollection`` (pyramids points → ``KDEGlyph``).

        Estimates the point density on a grid and draws it as filled (``shade=True``) or line contours, coloured
        through the shared scalar-mapping pipeline. The KDE is numpy-only (cleopatra ``KDEGlyph``).

        Args:
            features: A pyramids ``FeatureCollection`` of point geometries, or a path or URL to one
                (reprojected to the display CRS). Only a path-backed layer can be written down.
            clip: Optional boundary the density is clipped to (``FeatureCollection``/``GeoDataFrame`` reprojected,
                or a shapely geometry in the display CRS). ``None`` draws the full grid.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Styling kwargs forwarded to ``KDEGlyph`` (``levels``, ``shade``, ``gridsize``,
                ``bw_method``, ``cmap``, …).

        Returns:
            The contour set (``QuadContourSet``) registered as a Scene layer.
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: if ``features`` is not all single ``Point`` geometries.

        Examples:
            - Draw a density surface for the point fixture:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> fc = FeatureCollection.read_file("tests/data/points.geojson")
                >>> m = Map(crs=fc.epsg)
                >>> cs = m.kde(fc)
                >>> len(m.layers)
                1

                ```
        """
        return self._draw(
            LayerRecord(
                "heatmap",
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(props={"via": "kde"}),
                opts=opts,
                key=clip,  # a geometry, which a figure cannot carry — see `voronoi`
            )
        )

    @_skips_off_limb
    def lines(
        self,
        features: Any,
        *,
        column: Optional[str] = None,
        scheme: Optional[Any] = None,
        k: int = 5,
        cmap: Optional[Any] = None,
        width: Optional[Union[float, str]] = None,
        color: Optional[Any] = None,
        opacity: Optional[float] = None,
        name: Optional[str] = None,
        visible: bool = True,
        **opts: Any,
    ) -> Any:
        """Draw line features — rivers, roads, tracks, reach networks (pyramids lines → ``FlowGlyph``).

        The Core contract's ``lines``, with the web tier's keywords: a constant colour, or a colour per
        feature from ``column`` (classified by ``scheme``/``k``), and a width that is either one number or a
        column. ``MultiLineString`` features draw one path per part, each carrying its feature's values.
        :meth:`sankey` draws the same geometry as a flow map, with flow defaults.

        Args:
            features: A pyramids ``FeatureCollection`` of ``LineString``/``MultiLineString`` geometries, or a
                path or URL to one (reprojected to the display CRS). Only a path-backed layer can be written
                down.
            column: Numeric column whose value colours each line; ``None`` (default) draws one colour.
            scheme: A classification scheme cutting ``column`` into ``k`` classes (``"quantiles"``,
                ``"equal_interval"``, …); ``None`` (default) is a continuous ramp.
            k: Number of classes for ``scheme``.
            cmap: The colormap ``column`` is coloured through; ``None`` leaves cleopatra's default.
            width: Line width in points — a positive number — or the name of a numeric column whose values
                scale each line's width (between ``width_limits``, a ``FlowGlyph`` option). ``None``
                leaves cleopatra's width.
            color: The colour of every line when ``column`` is ``None`` — any matplotlib colour. A built
                ``ColorScaling`` or ``Normalize`` is instead the colour *scaling* of ``column``, as it is on
                every other static builder. A string that is not a colour is refused rather than passed on,
                with ``column=`` named as what takes a column's name.
            opacity: Layer opacity, 0 transparent to 1 opaque.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Further ``FlowGlyph`` options (``width_limits``, ``width_scale``, ``draw_order``,
                ``glow``, …).

        Returns:
            The ``LineCollection`` (registered as a Scene layer), or ``None`` when the data lies entirely
            outside what the display CRS shows.

        Raises:
            ValueError: if ``features`` is empty or contains non-line geometry; if ``width`` is not a
                positive finite number (a line has no negative thickness); if ``color`` is a string that
                is not a matplotlib colour — the refusal names ``column=`` as the keyword that takes a
                column's name; or if ``column`` or a column ``width`` names a column that does not hold
                numbers, which a ramp and a width scale both need.
            KeyError: if ``column`` or a column ``width`` names no feature attribute.

        Examples:
            - Colour reaches by discharge, classified into three classes, with one width:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import geopandas as gpd
                >>> from shapely.geometry import LineString
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth import Map
                >>> reaches = FeatureCollection(gpd.GeoDataFrame(
                ...     {"discharge": [10.0, 40.0, 25.0]},
                ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 1)]),
                ...               LineString([(2, 1), (3, 2)])],
                ...     crs="EPSG:4326",
                ... ))
                >>> m = Map(crs=4326)
                >>> lc = m.lines(reaches, column="discharge", scheme="equal_interval", k=3, width=2.0)
                >>> lc.get_array().tolist(), sorted({float(w) for w in lc.get_linewidths()})
                ([10.0, 40.0, 25.0], [2.0])
                >>> m.figure_spec.layers.get(m.layer_ids[-1]).kind
                'lines'

                ```
            - One colour, widths scaled by a column:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import geopandas as gpd
                >>> from shapely.geometry import LineString
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth import Map
                >>> roads = FeatureCollection(gpd.GeoDataFrame(
                ...     {"lanes": [1, 4]},
                ...     geometry=[LineString([(0, 0), (1, 0)]), LineString([(0, 1), (1, 1)])],
                ...     crs="EPSG:4326",
                ... ))
                >>> lc = Map(crs=4326).lines(roads, width="lanes", color="dimgray")
                >>> widths = lc.get_linewidths()
                >>> bool(widths[1] > widths[0])
                True

                ```

        See Also:
            sankey: the same geometry drawn as a flow map.
        """
        if width is not None and not isinstance(width, str):
            # A scalar width reached the collection exactly as written, negatives included
            # (`get_linewidths() == [-2.0]`), and zero drew lines of no width at all (review L4).
            if _as_finite(width, "width", _LINES_CALLER) <= 0.0:
                raise ValueError(
                    f"{_LINES_CALLER} needs width= as a positive number of points, or the name of a "
                    f"column to scale the widths by; got {width!r}"
                )
        if isinstance(color, str) or (color is not None and is_color_like(color)):
            # `color=` and `column=` sit next to each other in the signature, so a column name under
            # `color=` is a plausible slip — and `to_hex` answered it with matplotlib's "Invalid RGBA
            # argument", which names neither this method nor either keyword (review L4).
            try:
                line_color: Optional[Any] = to_hex(color, keep_alpha=True)
            except (ValueError, TypeError):
                raise ValueError(
                    f"{_LINES_CALLER} needs color= as one matplotlib colour for every line, and "
                    f"{color!r} is not one. To colour the lines by a column's values instead, pass "
                    f"column={color!r}"
                ) from None
        else:
            line_color = None
            if color is not None:
                opts["color"] = color
        if cmap is not None:
            opts["cmap"] = cmap
        if scheme is not None:
            opts["scheme"] = scheme
            opts["k"] = k
        return self._draw(
            LayerRecord(
                "lines",
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "lines",
                        "column": column,
                        "width": width,
                        "color": line_color,
                        "opacity": None if opacity is None else float(opacity),
                    }
                ),
                opts=opts,
            )
        )

    @_skips_off_limb
    def sankey(
        self,
        features: Any,
        column: Optional[str] = None,
        scale: Optional[str] = None,
        *,
        name: Optional[str] = None,
        visible: bool = True,
        **opts,
    ) -> Any:
        """Spatial flow / Sankey map of a line ``FeatureCollection`` (pyramids lines → ``FlowGlyph``).

        Draws each line as a path whose **colour** encodes ``column`` and whose **width** encodes ``scale``
        (each optional). MultiLineStrings contribute one path per part.

        Args:
            features: A pyramids ``FeatureCollection`` of ``LineString``/``MultiLineString`` geometries,
                or a path or URL to one (reprojected to the display CRS). Only a path-backed layer can be written down.
            column: Numeric column whose value colours each path, or ``None`` for a single colour.
            scale: Numeric column whose value sets each path's line width, or ``None`` for a uniform width.
            name: The caller's own name for the layer, used as its id and its label; ``None``
                (default) generates one from the kind, and a name already on the figure is suffixed
                ``-2``, ``-3``, … (#321).
            visible: Whether the layer is drawn. ``False`` builds it hidden **and** describes it
                hidden, so a switcher reading the figure agrees with the axes (#327).
            **opts: Styling kwargs forwarded to ``FlowGlyph`` (``width_limits``, ``width_scale``, ``cmap``,
                ``size_legend``, …).

        Returns:
            The ``LineCollection`` (registered as a Scene layer).
            ``None`` instead when the data lies entirely outside what the display CRS shows:
            an off-limb draw renders an empty frame rather than raising.

        Raises:
            ValueError: if ``features`` contains non-line geometry.

        See Also:
            lines: the same geometry as a plain line layer, with line defaults and the Core keywords.

        Examples:
            - Draw flow lines coloured and width-scaled by columns:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> import geopandas as gpd
                >>> from shapely.geometry import LineString
                >>> from pyramids.feature import FeatureCollection
                >>> from digitalearth.static import Map
                >>> gdf = gpd.GeoDataFrame(
                ...     {"flow": [1.0, 2.0]},
                ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(0, 1), (1, 2)])],
                ...     crs="EPSG:4326",
                ... )
                >>> m = Map(crs=4326)
                >>> lc = m.sankey(FeatureCollection(gdf), column="flow", scale="flow")
                >>> len(m.layers)
                1

                ```
        """
        return self._draw(
            LayerRecord(
                "flow",
                source=features,
                name=name,
                visible=visible,
                symbology=Symbology(
                    props={
                        "via": "sankey",
                        "column": column,
                        "scale": scale,
                    }
                ),
                opts=opts,
            )
        )
