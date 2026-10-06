"""The colour key of one layer: what its colour varies with, and the key that explains it.

Order 24 moved a colour key off the figure and onto the layer it describes. Before it, this tier's
``colorbar`` was keyed **by position** — ``layer=-1`` into ``Scene.layers`` — and drew when called: a
coastline added after a raster took the key over, a removed layer left its bar behind, and a figure written
to a dict carried no record that a key had ever been asked for. The key now hangs on the layer's colour
:class:`~digitalearth.base.spec.encoding.Encoding`, as a
:class:`~digitalearth.base.spec.encoding.Guide`, so it moves, hides and disappears with the layer for free.

Two halves live here, and they are deliberately the same shape as the reading the tier already took:

* :func:`color_encoding` publishes **what drives a layer's colour**, read off the artist that was drawn
  rather than recomputed beside it — the reading
  :meth:`~digitalearth.static.scene.Scene._record_class_edges` already takes for ``last_breaks``, and for
  the same reason: cleopatra does the classifying, and the norm it leaves on the mappable is what
  matplotlib actually colours through. A layer whose colour is one flat constant publishes nothing, and a
  key on it is then refused rather than drawn empty.
* :func:`draw_guide` draws the recorded guide, as the **one** place a key is drawn — so the same key comes
  back whether the draw was reached by the caller's :meth:`~digitalearth.static.scene.Scene.colorbar` call,
  by a restyle that rebuilt the layer, by a rollback, or by a figure read back onto another scene.

This tier draws **two** kinds of key where the web tier draws one: a matplotlib ``Colorbar`` (a strip beside
the axes) and a swatch ``Legend`` (rows inside it). Which one a layer's guide asks for is therefore part of
the record, under :data:`GUIDE_KIND_KEY` — a `Guide` says whether, what it is called and where, and has no
field for *which*, because the tiers that draw one kind need none.

The content of a swatch key is derived through :class:`~digitalearth.base.spec.legend.LegendSpec` from the
same `Scale` the layer publishes, so a swatch equals the colour that was drawn by construction rather than
by maintenance (#185).
"""

import logging
from dataclasses import dataclass
from math import isfinite
from typing import Any, Optional

from cleopatra.styling.styles import colorbar_legend, disjoint_legend, hatch_legend
from matplotlib.colors import to_hex, to_rgba
from matplotlib.contour import ContourSet

from digitalearth.base.spec import (
    DEFAULT_BAND,
    Encoding,
    LayerSpec,
    LegendSpec,
    Scale,
)

__all__ = [
    "COUNT_FIELD",
    "DENSITY_FIELD",
    "GUIDE_KIND_KEY",
    "GUIDE_KINDS",
    "GUIDE_LABELS_KEY",
    "MAGNITUDE_FIELD",
    "GuidePlan",
    "bar_refusal",
    "color_encoding",
    "draw_guide",
    "drawn_scale",
    "guide_kind",
    "paint_guide",
    "plan_guide",
    "source_field",
]

#: The property the kind of key a layer's guide asks for is recorded under — one of :data:`GUIDE_KINDS`.
#:
#: In ``props`` rather than on the `Guide`, because the *guide* is the shared vocabulary and this is the one
#: tier that has a choice to record: a MapLibre panel, a Bokeh toggle and a PyVista scalar bar are each one
#: piece of furniture, while matplotlib has both a colorbar and a legend and they are not interchangeable.
GUIDE_KIND_KEY: str = "guide_kind"

#: The two kinds of colour key this tier draws: a colorbar strip, or a list of labelled swatches.
GUIDE_KINDS: tuple[str, ...] = ("colorbar", "legend")

logger = logging.getLogger(__name__)

#: The alpha at or below which :func:`_is_transparent` calls a swatch colour fully see-through.
#:
#: A tolerance rather than an exact ``== 0.0``: testing a float for equality is a defect in its own right
#: (SonarCloud ``python:S1244``), because the value is only ever as exact as the arithmetic that produced
#: it. Half of the 8-bit quantum — ``to_rgba`` divides an 8-digit hex channel by 255, so ``1/255`` is the
#: smallest alpha a colour the plan carries can hold above zero. The bound therefore admits exactly what
#: ``== 0.0`` admitted among hex spellings, and in addition a computed alpha that no 8-bit channel could
#: tell from zero; the faintest colour an 8-digit hex *can* spell, ``#00000001``, stays opaque enough to
#: colour its swatch.
_TRANSPARENT_ALPHA: float = 0.5 / 255.0

#: The property a caller's own row labels for one layer's key are recorded under.
#:
#: Recorded rather than kept as a call argument, because the key is redrawn from the description whenever the
#: layer is (a restyle, a rollback, a figure read back), and labels that lived only in the call would come
#: back as the derived ones.
GUIDE_LABELS_KEY: str = "guide_labels"

#: Which matplotlib legend corner each :attr:`~digitalearth.base.spec.encoding.Guide.anchor` names.
#:
#: The anchors are the four the whole package shares
#: (:data:`~digitalearth.base.registry.FURNITURE_ANCHORS`) — the same spellings the web tier's ``position``
#: takes — so a figure that says where its key goes says it once and every tier honours it. matplotlib's
#: ``loc`` vocabulary is what it is translated *into*, not what a caller writes.
_LEGEND_LOCATIONS = {
    "top-left": "upper left",
    "top-right": "upper right",
    "bottom-left": "lower left",
    "bottom-right": "lower right",
}

#: Which side of the axes a colorbar steals space from, for each anchor.
#:
#: A colorbar is a strip beside the axes rather than a box inside it, so it has a side and not a corner: the
#: two right-hand anchors put it on the right and the two left-hand ones on the left. Coarser than the
#: legend's translation, and stated rather than silently dropped.
_COLORBAR_SIDES = {
    "top-left": "left",
    "bottom-left": "left",
    "top-right": "right",
    "bottom-right": "right",
}


#: What a layer is coloured by when its kind **computes** the values and no column names them.
#:
#: Each is the word this tier already uses for the quantity: ``count`` is the reducer
#: :data:`~digitalearth.static.maps.vector._QUADTREE_AGG` adds to the shared NaN-aware registry, a KDE draws a
#: density, and a u/v field colours its arrows by magnitude. They are named here rather than spelled at each
#: drawer so the three cannot drift.
COUNT_FIELD: str = "count"
DENSITY_FIELD: str = "density"
MAGNITUDE_FIELD: str = "magnitude"


def source_field(source: Any, band: int = DEFAULT_BAND) -> str:
    """Return the name a source's values are known by, as the tier already spells it.

    The builders record a band **number** and never open their source, so the name the colour varies with is
    the drawer's to publish: this is the same ``variable`` that
    :func:`~digitalearth.base.autostyle.auto_style` matches a colormap and a colorbar's units on, which is
    why it is read from there rather than invented beside it.

    Args:
        source: A :class:`~digitalearth.base.sources.Source` view, or the pyramids object behind one.
        band: The 1-based band, for a raster that names its bands.

    Returns:
        The source's ``variable`` — for a plain GeoTIFF that is pyramids' own ``Band_1``, and for a vector
        collection the numeric column its values were read from. A raster that names no band, which a bare
        numpy array is, has no name to publish, so the band is named by its number: the only thing the layer
        records about it.
    """
    read = getattr(source, "metadata", None)
    variable = str(read("variable") or "").strip() if callable(read) else ""
    if not variable:
        names = list(getattr(source, "band_names", None) or [])
        if 1 <= band <= len(names):
            variable = str(names[band - 1]).strip()
    return variable or f"band {band}"


def _colours_by_value(artist: Any) -> bool:
    """Whether a drawn artist maps per-datum values onto colours at all.

    Asked because a `norm` is not the answer: matplotlib gives **every** collection one, so an outline-only
    polygon fill and a scatter of uniform markers both carry a `Normalize` with nothing behind it, and a bar
    over either would label a colour nothing varies.

    Args:
        artist: The mappable the layer's drawer produced.

    Returns:
        ``True`` when the artist holds the value array it colours through. An RGB image is the one that holds
        an array and drives no single channel, and it is excluded by naming no field rather than here.
    """
    if getattr(artist, "norm", None) is None:
        return False
    read = getattr(artist, "get_array", None)
    return callable(read) and read() is not None


def _swatches(legend: Any) -> tuple[list[Any], list[str]]:
    """Return the categories and colours a glyph's own swatch legend was drawn with.

    Read off the drawn legend rather than re-derived from the values: cleopatra assigns a categorical fill's
    colours, so anything computed a second time here agrees only while nobody edits one of them — the
    disagreement :class:`~digitalearth.base.spec.legend.LegendSpec` exists to remove.

    Args:
        legend: The ``matplotlib.legend.Legend`` a cleopatra glyph exposes as ``category_legend``.

    Returns:
        ``(categories, colors)`` — one label and one hex colour per swatch, in the order they are drawn. Two
        empty lists for a legend whose handles carry no face colour or whose counts disagree, which is not a
        categorical fill's swatch list and must not be published as one.
    """
    labels = [text.get_text() for text in legend.get_texts()]
    colors: list[str] = []
    for handle in legend.legend_handles:
        face = getattr(handle, "get_facecolor", None)
        if face is None:
            return [], []
        colors.append(to_hex(face()))
    if not labels or len(colors) != len(labels):
        return [], []
    return list(labels), colors


def drawn_scale(artist: Any) -> Scale | None:
    """Return the scale a drawn artist colours through, read off its own norm.

    Shared rather than private because a drawer needs the same answer: `FieldColors.applied_to` in the
    raster mixin reads it to restate the layer's extreme colours on the scale it publishes (ST-10) — only
    as the fallback, since a categorical field already states a scale of its own — and two spellings of
    "the scale this artist drew" would be two chances to disagree about what the picture shows.

    Args:
        artist: The mappable the layer's drawer produced.

    Returns:
        A classified :class:`~digitalearth.base.spec.scale.Scale` when the norm carries class edges, a
        continuous one over its limits otherwise, and ``None`` when the artist colours by nothing or the
        norm has no usable limits to publish — unmeasured limits are honest as no scale, while inventing a
        pair would label the key with numbers the layer never draws.

    Examples:
        - The limits a plain norm was built with come back as the scale's domain:
            ```python
            >>> from matplotlib.colors import Normalize
            >>> from digitalearth.static.guides import drawn_scale
            >>> mappable = type("A", (), {"norm": Normalize(vmin=-3.0, vmax=7.0)})()
            >>> drawn_scale(mappable).as_limits()
            (-3.0, 7.0)

            ```
        - An artist that colours by nothing publishes no scale:
            ```python
            >>> from digitalearth.static.guides import drawn_scale
            >>> drawn_scale(object()) is None
            True

            ```
    """
    norm = getattr(artist, "norm", None)
    if norm is None:
        return None
    edges = getattr(norm, "boundaries", None)
    if edges is not None:
        breaks = tuple(float(edge) for edge in edges)
        if len(breaks) >= 2 and breaks[-1] > breaks[0] and all(map(isfinite, breaks)):
            # `scheme` carries the edges themselves rather than a scheme name: the edges are what was read
            # off the picture, and the name that cut them — if any did — is the caller's `scheme=`, which
            # this reading deliberately does not consult.
            return Scale(breaks[0], breaks[-1], scheme=breaks, breaks=breaks)
        return None
    low, high = getattr(norm, "vmin", None), getattr(norm, "vmax", None)
    if low is None or high is None:
        return None
    low, high = float(low), float(high)
    if not (isfinite(low) and isfinite(high)):
        return None
    return Scale.from_limits(low, high)


def color_encoding(field: str | None, drawn: Any) -> Encoding | None:
    """Return what drives a drawn layer's colour, or ``None`` when nothing does.

    Args:
        field: The data field the colour varies with — a raster's band variable, a vector's value column.
            ``None`` or empty for a layer that colours by no field, which publishes nothing: an encoding has
            to name what drives it, and a guide over a colour nothing varies has no values to label.
        drawn: What the layer's drawer produced, as
            :class:`~digitalearth.static.renderer.DrawnLayer` describes it.

    Returns:
        A field-driven colour :class:`~digitalearth.base.spec.encoding.Encoding`, carrying the scale the
        picture was actually drawn through. ``None`` for a layer whose artist colours by value nowhere — an
        outline-only fill, a graticule, a text label, a coastline — and for a caller's own artist, which is
        the one thing this tier cannot name a field for.

    Examples:
        - A raster field publishes its band and the limits the render settled on. This is the one caller
          that reads the whole drawn record rather than `Scene.artist(layer_id)`, which hands back the
          artist alone: the encoding is built from the layer's stated scale and its glyph's category
          legend as well, so the artist on its own is not enough — measured,
          `color_encoding("band 1", m.artist("grid"))` raises
          `AttributeError: 'AxesImage' object has no attribute 'glyph'`:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from digitalearth.static import Map
            >>> from digitalearth.static.guides import color_encoding
            >>> m = Map(globe=False)
            >>> _ = m.field(np.array([[0.0, 1.0], [2.0, 3.0]]), name="grid")
            >>> encoding = color_encoding("band 1", m._renderer.drawn["grid"])
            >>> encoding.field, encoding.scale.as_limits()
            ('band 1', (0.0, 3.0))
            >>> m.close()

            ```
        - A layer that names no field publishes nothing, so a key on it is refused rather than drawn empty:
            ```python
            >>> from digitalearth.static.guides import color_encoding
            >>> from digitalearth.static.renderer import DrawnLayer
            >>> color_encoding(None, DrawnLayer()) is None
            True

            ```
    """
    if not field:
        return None
    stated = getattr(drawn, "scale", None)
    if stated is not None:
        # The drawer knew the scale better than the norm can say it — a categorical raster's codes, which
        # its `BoundaryNorm` only brackets — so publish that rather than reading the norm.
        return Encoding.by_field("color", field, scale=stated)
    legend = getattr(drawn.glyph, "category_legend", None)
    if legend is not None:
        categories, colors = _swatches(legend)
        if not categories:
            return None
        return Encoding.by_field(
            "color", field, scale=Scale.categorical(categories, colors)
        )
    if not _colours_by_value(drawn.artist):
        return None
    return Encoding.by_field("color", field, scale=drawn_scale(drawn.artist))


def bar_refusal(layer: LayerSpec) -> str | None:
    """Return why a colorbar cannot describe one layer's colour, or ``None`` when it can.

    Asked twice, in one spelling: :meth:`~digitalearth.static.scene.Scene.colorbar` asks it **before** it
    records a guide, so a refused call leaves the layer exactly as it found it, and :func:`draw_guide` asks
    it again for a figure that was built somewhere else and asks for a bar the scale cannot be drawn as.

    Args:
        layer: The layer's description.

    Returns:
        The refusal, naming what the bar would read instead and which method fits — or ``None``.
    """
    encoding = layer.symbology.encoding("color")
    scale = None if encoding is None else encoding.scale
    if scale is not None and scale.is_categorical:
        return (
            f"layer {layer.id!r} is coloured by category, so a colorbar over it would read the class codes "
            f"cleopatra assigned rather than {list(scale.categories)}; ask for legend() instead"
        )
    return None


def guide_kind(layer: LayerSpec) -> str:
    """Return which kind of key one layer's guide asks for.

    **The package's rule, stated once, because two docstrings stated two of them** (review M2): *a scale is
    explained by what it is made of* — a categorical one by its categories, a classified one by its
    classes, a continuous one by its range. What differs between the four tiers is the **furniture** each
    engine has for saying that, not which of the three a layer gets:

    ===============  ==========================  =============================================
    scale            this tier / interactive     web / 3-D
    ===============  ==========================  =============================================
    categorical      swatch list                 swatch list
    classified       banded bar, ticks on the    keyed list of class ranges
                     class edges
    continuous       ramp bar                    gradient bar (web); scalar bar (3-D)
    ===============  ==========================  =============================================

    The classified row is the one that reads as a divergence and is not. matplotlib paints a classified
    fill through a ``BoundaryNorm``, so the strip is **banded** and its ticks are exactly the edges the
    scale published — measured on a three-class quantile cut over 0–5: ``bar.get_ticks()`` is
    ``[0.0, 1.667, 3.333, 5.0]``, the same numbers the web tier's keyed list of ranges shows. It is the
    class list, drawn as a strip. A tier that paints classes by **index** cannot say that — PyVista's
    scalar bar would read ``0, 1, 2`` over the indices ``classified_scalars`` assigns — which is why the
    3-D tier answers with the list, and the web tier builds its one panel from the classification it
    recorded. Pinned by
    ``tests/static/test_static_guides.py::TestAGraduatedFillIsKeyedByItsClasses``, on the edges rather than
    on the word "colorbar", since the word alone is satisfied by a ramp over the same limits.

    A **categorical** scale is the one case where no tier can use a bar: matplotlib's would read the class
    codes cleopatra assigned and Bokeh draws no ``ColorBar`` from a ``CategoricalColorMapper`` at all
    (:func:`~digitalearth.interactive.style_fold.guide_kind`).

    Args:
        layer: The layer's description.

    Returns:
        The recorded kind, or — for a guide that arrived without one, which is a figure built where the
        distinction does not exist — ``"legend"`` for a categorical scale and ``"colorbar"`` for every
        other, which is the rule this tier already applied through ``add_colorbar=categorical`` in
        :meth:`~digitalearth.static.maps.vector.VectorMixin._polygon_layer`.
    """
    recorded = layer.symbology.props.get(GUIDE_KIND_KEY)
    if recorded in GUIDE_KINDS:
        return str(recorded)
    encoding = layer.symbology.encoding("color")
    scale = None if encoding is None else encoding.scale
    return "legend" if scale is not None and scale.is_categorical else "colorbar"


def _legend_spec(scale: Scale, artist: Any, title: str | None) -> LegendSpec:
    """Return the rows a swatch key is drawn from, derived from the scale the layer was drawn through.

    Args:
        scale: The layer's published colour scale.
        artist: The mappable, for the colormap the rows are sampled from — a classified or continuous scale
            carries its edges or limits but not the colours the renderer chose for them.
        title: The key's heading, or ``None``.

    Returns:
        The :class:`~digitalearth.base.spec.legend.LegendSpec` to draw.
    """
    if scale.is_categorical:
        return LegendSpec.from_scale(scale, title=title)
    cmap, norm = artist.cmap, artist.norm
    if scale.is_classified:
        ranges = scale.class_ranges()
        # The colour each class is painted, through the artist's own cmap and norm — so the swatch is the
        # colour on the map rather than a second sampling that happens to agree.
        colors = [to_hex(cmap(norm((low + high) / 2.0))) for low, high in ranges]
        return LegendSpec.from_scale(scale, colors=colors, title=title)
    low, high = scale.as_limits()
    stops = [low + (high - low) * index / 4.0 for index in range(5)]
    return LegendSpec.from_scale(
        scale,
        colors=[to_hex(cmap(norm(stop))) for stop in stops],
        title=title,
        values=stops,
    )


def _rows(layer: LayerSpec, spec: LegendSpec) -> list[str]:
    """Return the row labels a swatch key is drawn with: the caller's own, or the derived ones.

    Args:
        layer: The layer's description, which records the caller's labels if they gave any.
        spec: The derived legend.

    Returns:
        One label per row.

    Raises:
        ValueError: when the recorded labels do not number the rows — a short list leaves swatches
            unlabelled and a long one labels swatches that are not drawn, and either way the key stops
            matching the picture.

    Note:
        "The caller gave none" is ``None``, not falsiness. This asked ``if not labels``, so an **empty
        list** read as "derive them" — ``legend(labels=["a"])`` was refused on a five-row key while
        ``legend(labels=[])`` was accepted, and the refusal below could never say "got 0" (review L1).
        `Scene._record_key` stores the caller's list as a tuple and *removes* the property for ``None``, so
        the two are distinguishable here; the web and interactive tiers both refuse the empty list.
    """
    labels = layer.symbology.props.get(GUIDE_LABELS_KEY)
    if labels is None:
        return [entry.label for entry in spec.entries]
    rows = [str(label) for label in labels]
    if len(rows) != len(spec.entries):
        raise ValueError(
            f"the key of layer {layer.id!r} has {len(spec.entries)} rows, so labels= needs "
            f"{len(spec.entries)} labels; got {len(rows)}"
        )
    return rows


def _hatches_of(artist: Any) -> tuple[str, ...] | None:
    """Return the hatch pattern of each band a filled contour set draws, or ``None`` when it hatches none.

    Args:
        artist: The layer's drawn artist.

    Returns:
        One pattern per band between the set's levels, cycled from its own list the way matplotlib cycles
        it — ``""`` for a band drawn plain. ``None`` for anything but a filled ``ContourSet`` carrying at least
        one pattern, which is every layer the swatch key already describes by colour.
    """
    if not isinstance(artist, ContourSet) or not artist.filled:
        return None
    patterns = list(artist.hatches or ())
    if not any(patterns):
        return None
    bands = len(tuple(artist.levels)) - 1
    return tuple(patterns[index % len(patterns)] or "" for index in range(bands))


def _marked_bands(artist: Any, bands: int) -> list[bool]:
    """Say, band by band, whether a filled contour set put any geometry on the axes for it.

    Levels may run past the data — ``levels=[0, 0.05, 1, 5, 10]`` over a p-value field that stops at 0.95
    — and the bands beyond it are drawn as empty paths. They carry the set's colour and pattern all the
    same, so a key filtered on those two alone invents classes a reader will look for and not find
    (review M1). A filled ``ContourSet`` holds one compound path per band, in level order, and an empty
    band's path has no vertices.

    Args:
        artist: The filled ``ContourSet``.
        bands: How many bands the set's levels declare.

    Returns:
        One flag per band, ``True`` when the band holds at least one vertex. Every band is reported marked
        when the artist's paths do not number the bands, since dropping a row on a reading that no longer
        holds would be worse than keeping an empty one.

    Examples:
        - A p-value field that stops at 0.95, under levels that run to 10: the top two bands are drawn as
          paths with no vertices, and only the two that hold data are reported marked:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np
            >>> from digitalearth.static.guides import _marked_bands
            >>> fig, ax = plt.subplots()
            >>> contours = ax.contourf(
            ...     np.linspace(0.0, 0.95, 16).reshape(4, 4), levels=[0.0, 0.05, 1.0, 5.0, 10.0]
            ... )
            >>> [len(path.vertices) > 0 for path in contours.get_paths()]
            [True, True, False, False]
            >>> _marked_bands(contours, 4)
            [True, True, False, False]
            >>> plt.close(fig)

            ```
        - A band count the artist's paths do not match is read as "cannot tell", and every band is kept:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np
            >>> from digitalearth.static.guides import _marked_bands
            >>> fig, ax = plt.subplots()
            >>> contours = ax.contourf(
            ...     np.linspace(0.0, 9.0, 16).reshape(4, 4), levels=[0.0, 3.0, 6.0, 9.0]
            ... )
            >>> _marked_bands(contours, 7)
            [True, True, True, True, True, True, True]
            >>> plt.close(fig)

            ```
    """
    paths = list(artist.get_paths())
    if len(paths) != bands:
        return [True] * bands
    return [len(path.vertices) > 0 for path in paths]


@dataclass(frozen=True)
class GuidePlan:
    """One layer's colour key, derived and checked but not yet on the figure.

    The seam between deciding what a key *is* and putting it there, so that every refusal a description can
    earn is raised while the figure is still untouched. :meth:`~digitalearth.static.renderer.Renderer
    .draw_guide` takes the layer's previous key off between the two, and until this split that removal came
    **first**: a redraw whose key then refused left the layer describing a bar whose axes had already been
    taken off the figure, and nothing could put it back — a removed axes cannot be re-added, and
    ``_PartialDraw.undo`` watches the axes' legend slot, which a colorbar does not use (review M4).

    Attributes:
        kind: ``"colorbar"`` or ``"legend"``, as :func:`guide_kind` answered it.
        title: What the key is called, or ``None`` for an unlabelled one.
        anchor: The corner the guide asks for, or ``None`` to leave the placement to matplotlib.
        mappable: The artist a bar is drawn from. ``None`` for a swatch list, which needs none.
        colors: One colour per swatch, in row order — `None` for a row the scale gave none, which is
            what `LegendEntry.color` allows and what the engine is handed either way. Empty for a bar.
        rows: One label per swatch, the same length as `colors`. Empty for a bar.
        hatches: One hatch pattern per swatch, for a layer whose bands are told apart by pattern — a filled
            contour drawn with ``hatches=``. Empty for every other key.
        hatch_color: The colour of the layer's hatch strokes, so a swatch's pattern is drawn as the map's
            is. ``None`` when there are no hatches.
    """

    kind: str
    title: str | None = None
    anchor: str | None = None
    mappable: Any = None
    colors: tuple[str | None, ...] = ()
    rows: tuple[str, ...] = ()
    hatches: tuple[str, ...] = ()
    hatch_color: str | None = None


@dataclass(frozen=True, eq=False)
class _HatchedKey:
    """The rows of a hatched layer's key: the bands it declares, the ones the map marks, their labels.

    The layer's published scale cannot answer any of this. A filled contour set publishes the continuous
    range its norm spans, so the colour key reads five evenly spaced stops — and an overlay drawn with
    ``fill=False`` has no colour for a stop to show at all. The bands are the set's own ``levels``, read off
    the artist as a graduated scale so the row labels are written exactly as a classified layer's are.

    These five values are born together out of one artist, are read only ever as a set — which rows are
    kept decides which colours, which labels and which patterns the key carries — and nothing outside this
    class reads any of them. That is what makes them one object rather than a three-tuple and a pair of
    locals threaded through :func:`plan_guide`, which is where they used to live.

    Attributes:
        layer_id: The layer these rows belong to, for the refusal and the warning to name.
        artist: The filled ``ContourSet``, kept because the hatch stroke colour is read off it when the key
            is actually built and not before.
        spec: The legend over **every** declared band, so a kept band's colour and derived label are read
            at the band's own index.
        kept: The indices of the bands the key draws. A band is kept only when the map actually marks it,
            which takes both: something to mark it *with* — a pattern, or a face that is not fully
            transparent — and some geometry to mark, which :func:`_marked_bands` reads off the artist.
            Levels that run past the data satisfy the first and not the second, and keying them invented
            classes nothing on the figure showed (review M1).
        rows: One label per **kept** band, in row order — the caller's recorded labels where there are
            some, else the ones :attr:`spec` derived.
        hatches: One pattern per declared band, from :func:`_hatches_of`.

    Examples:
        - A hatched overlay whose levels run past the data: four bands are declared, the top two hold no
          geometry, and only the hatched band that does is keyed:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth import Map
            >>> from digitalearth.static.guides import _HatchedKey
            >>> p = Dataset.from_array(
            ...     arr=np.linspace(0.0, 0.95, 16).reshape(4, 4),
            ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326),
            ...     no_data_value=-9999.0,
            ... )
            >>> with Map(crs=4326) as m:
            ...     _ = m.contours(
            ...         p, levels=[0.0, 0.05, 1.0, 5.0, 10.0], filled=True,
            ...         hatches=["///", "", "xx", ".."], fill=False, name="sig",
            ...     )
            ...     sig = m.artist("sig")
            ...     key = _HatchedKey.of(m.figure_spec.layers.get("sig"), sig, "p")
            ...     key.hatches
            ...     key.kept
            ...     key.rows
            ('///', '', 'xx', '..')
            (0,)
            ('0.0 – 0.05',)

            ```
        - Recorded labels are counted against the bands ``levels=`` declares, and the labels of the bands
          that are not kept are dropped alongside their rows:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import numpy as np
            >>> from pyramids.dataset import Dataset, GeoReference
            >>> from digitalearth import Map
            >>> from digitalearth.static.guides import _HatchedKey
            >>> p = Dataset.from_array(
            ...     arr=np.linspace(0.0, 0.95, 16).reshape(4, 4),
            ...     geo_ref=GeoReference(geo=(0.0, 1.0, 0.0, 4.0, 0.0, -1.0), epsg=4326),
            ...     no_data_value=-9999.0,
            ... )
            >>> with Map(crs=4326) as m:
            ...     _ = m.contours(
            ...         p, levels=[0.0, 0.05, 1.0, 5.0, 10.0], filled=True,
            ...         hatches=["///", "", "xx", ".."], fill=False, name="sig",
            ...     )
            ...     sig = m.artist("sig")
            ...     _ = m.legend("sig", labels=["a", "b", "c", "d"])
            ...     _HatchedKey.of(m.figure_spec.layers.get("sig"), sig, "p").rows
            ...     try:
            ...         m.legend("sig", labels=["a", "b"])
            ...     except ValueError as error:
            ...         print(error)
            ('a',)
            layer 'sig' declares 4 bands between its levels, so labels= needs 4 labels, one per band; got 2

            ```
    """

    layer_id: str
    artist: Any
    spec: LegendSpec
    kept: tuple[int, ...]
    rows: tuple[str, ...]
    hatches: tuple[str, ...]

    @classmethod
    def of(
        cls, layer: LayerSpec, artist: Any, title: str | None
    ) -> Optional["_HatchedKey"]:
        """Read a drawn layer's hatched bands, or answer ``None`` when it has none.

        Args:
            layer: The layer's description, which records the caller's labels if they gave any.
            artist: The layer's drawn artist.
            title: The key's heading, or ``None``.

        Returns:
            The rows of the key, or ``None`` for an artist that is not a filled ``ContourSet`` carrying at
            least one pattern (:func:`_hatches_of`) — which is every layer the swatch key already describes
            by colour.

        Raises:
            ValueError: when the recorded labels do not number the bands the layer's ``levels=``
                **declares**.

        Note:
            ``labels=`` is counted against the declared bands, not the kept ones, and the labels of the
            bands that are not kept are dropped alongside their rows. Counting the kept rows made the
            accepted spelling a property of the *raster*: a four-band call was told "the key has 2 rows",
            and a ``labels=`` recorded against one reading of the field started raising as soon as the data
            reached a third band (review M3). One label per band between the levels — equivalently, one per
            ``hatches=`` entry — is derivable from the call alone, so it keeps working as the data moves.
        """
        hatches = _hatches_of(artist)
        if hatches is None:
            return None
        levels = tuple(float(level) for level in artist.levels)
        faces = [tuple(face) for face in artist.get_facecolor()]
        faces = [faces[index % len(faces)] for index in range(len(hatches))]
        colors = [to_hex(face, keep_alpha=True) for face in faces]
        scale = Scale(levels[0], levels[-1], scheme=levels, breaks=levels)
        spec = LegendSpec.from_scale(scale, colors=colors, title=title)
        marked = _marked_bands(artist, len(hatches))
        kept = tuple(
            index
            for index, (face, pattern) in enumerate(zip(faces, hatches))
            if marked[index] and (pattern or face[3] > 0.0)
        )
        labels = layer.symbology.props.get(GUIDE_LABELS_KEY)
        if labels is None:
            rows = tuple(spec.entries[index].label for index in kept)
        else:
            recorded = [str(label) for label in labels]
            if len(recorded) != len(hatches):
                raise ValueError(
                    f"layer {layer.id!r} declares {len(hatches)} bands between its levels, so labels= "
                    f"needs {len(hatches)} labels, one per band; got {len(recorded)}"
                )
            rows = tuple(recorded[index] for index in kept)
        return cls(
            layer_id=layer.id,
            artist=artist,
            spec=spec,
            kept=kept,
            rows=rows,
            hatches=hatches,
        )

    def plan(self, guide: Any) -> GuidePlan | None:
        """Build the swatch key these rows describe, or answer ``None`` when there are no rows.

        Args:
            guide: The layer's recorded guide, read for its ``anchor`` and its ``show`` flag.

        Returns:
            The plan :func:`paint_guide` draws, or ``None`` when the map marks none of the declared bands
            — logged at ``WARNING`` — or when the guide is switched off. The two are answered in that
            order, so the warning is emitted for a key that is described but not shown.
        """
        if not self.kept:
            # Every band is unmarked, so there is not one row to draw — and a swatch legend with no
            # swatches is an empty framed box carrying only its title, which is itself the class a
            # reader looks for and does not find that `_marked_bands` drops rows to avoid (review L5).
            # Logged rather than left silent, as `facet` logs the bar it leaves off an unmeasurable
            # stack: the layer is on the figure, only its key is not.
            logger.warning(
                "legend(): layer %r marks none of its %d hatched bands, so there are no rows to key "
                "it with; no legend is drawn rather than an empty box",
                self.layer_id,
                len(self.hatches),
            )
            return None
        if not guide.show:
            return None
        return GuidePlan(
            "legend",
            title=self.spec.title,
            anchor=guide.anchor,
            colors=tuple(self.spec.entries[index].color for index in self.kept),
            rows=self.rows,
            hatches=tuple(self.hatches[index] for index in self.kept),
            hatch_color=to_hex(self.artist.get_hatchcolor()[0], keep_alpha=True),
        )


def _legend_plan(
    layer: LayerSpec, artist: Any, scale: Scale | None, guide: Any
) -> GuidePlan | None:
    """Derive the swatch key a layer's recorded legend asks for.

    The two ways a swatch key's rows are derived, in the order they are tried: off the drawn artist for a
    hatched layer (:class:`_HatchedKey`), and off the scale the layer publishes for every other. ``scale``
    is read only here, which is why it is this function's argument rather than :func:`plan_guide`'s local.

    Args:
        layer: The layer's description, which carries the recorded labels.
        artist: What its drawer produced.
        scale: The colour encoding's scale, or ``None`` when it publishes none.
        guide: The layer's recorded guide.

    Returns:
        The plan :func:`paint_guide` draws, or ``None`` when the guide is switched off or the layer marks
        none of its hatched bands.

    Raises:
        ValueError: when the layer publishes no scale for the rows to be derived from, and from
            :func:`_rows` or :meth:`_HatchedKey.of` when recorded labels do not number those rows.
    """
    hatched = _HatchedKey.of(layer, artist, guide.title)
    if hatched is not None:
        return hatched.plan(guide)
    if scale is None:
        raise ValueError(
            f"layer {layer.id!r} publishes no colour scale, so there are no rows to key it with; "
            "colorbar() draws the bar its mappable can still carry"
        )
    spec = _legend_spec(scale, artist, guide.title)
    # Derived before the `show` gate below, not after it: these two lines are what refuses a key the
    # description asks for and this tier cannot draw, and behind the gate they were skipped by
    # `visible=False` — so one spelling of `legend(labels=…)` raised and the other recorded a
    # description that could never be drawn (review M1).
    rows = _rows(layer, spec)
    if not guide.show:
        return None
    return GuidePlan(
        "legend",
        title=spec.title,
        anchor=guide.anchor,
        colors=tuple(entry.color for entry in spec.entries),
        rows=tuple(rows),
    )


def plan_guide(layer: LayerSpec, drawn: Any) -> GuidePlan | None:
    """Derive the key one layer's recorded guide asks for, refusing one this tier cannot draw.

    The orchestrator of the two kinds this tier draws: a swatch key is :func:`_legend_plan`'s, read either
    off the drawn artist (:class:`_HatchedKey`) or off the scale the layer publishes, and a bar is the tail
    of this function, where there is no derived state to name — only the refusal the layer earns and the
    ``show`` flag.

    Args:
        layer: The layer's description, which carries the guide.
        drawn: What its drawer produced, as :class:`~digitalearth.static.renderer.DrawnLayer` describes it.

    Returns:
        The plan :func:`paint_guide` draws, or ``None`` when there is nothing to draw — the guide is
        switched off or absent, the layer carries no colour encoding, nothing was drawn for it, or it is a
        hatched layer whose bands the map marks none of (logged at ``WARNING``).

    Raises:
        ValueError: when a colorbar is asked for over a **categorical** scale, naming the swatch legend
            instead: a categorical fill's norm bins the class codes cleopatra assigned, so a bar over them
            reads ``0, 1, 2 …`` where the category labels belong. Also when the layer publishes no scale for
            a swatch key's rows to be derived from, and when recorded row labels do not number those rows —
            for a hatched layer, the bands its ``levels=`` declares rather than the rows the data leaves.

            **Every one of the three is raised whether or not the guide is shown.** A guide switched off is
            a key the caller asked not to be drawn *yet*, not one that has stopped being described — so a
            description this tier could never draw is refused when it is recorded rather than when it is
            switched on, which is what keeps one spelling of a call from being valid only half the time
            (review M1). The ``show`` flag is read only once the key has been derived.
    """
    guide = layer.symbology.guide()
    encoding = layer.symbology.encoding("color")
    if guide is None or encoding is None or drawn.artist is None:
        return None
    if guide_kind(layer) == "legend":
        return _legend_plan(layer, drawn.artist, encoding.scale, guide)
    refusal = bar_refusal(layer)
    if refusal is not None:
        raise ValueError(refusal)
    if not guide.show:
        return None
    return GuidePlan(
        "colorbar",
        title=guide.title,
        anchor=guide.anchor,
        mappable=drawn.artist,
    )


def paint_guide(scene: Any, plan: GuidePlan, **kwargs: Any) -> Any:
    """Put a derived key on the figure, and return the artist it is.

    Args:
        scene: The scene the layer is drawn on.
        plan: What :func:`plan_guide` derived. Nothing here can refuse it — every check the description can
            earn has already been made, which is what lets the caller take the layer's previous key off in
            between.
        **kwargs: Forwarded to ``cleopatra.styling.styles.colorbar_legend`` (a bar), ``disjoint_legend``
            (swatches) or ``hatch_legend`` (the unfilled swatches of a hatch-only overlay — see
            :func:`_paint_hatch_legend`, which picks between the last two). These are the call's own
            styling and are **not** part of the record, so a key
            redrawn from the description comes back with matplotlib's defaults in their place — the same
            trade :attr:`~digitalearth.static.scene.LayerRecord.opts` makes for a value no figure can carry.

    Returns:
        The one artist the key is — a ``Colorbar`` or a ``Legend``.

    Examples:
        - A swatch plan built by hand is painted as the rows it carries, titled by the plan:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Scene
            >>> from digitalearth.static.guides import GuidePlan, paint_guide
            >>> with Scene() as scene:
            ...     key = paint_guide(
            ...         scene,
            ...         GuidePlan("legend", title="class", colors=("#1f77b4", "#d62728"),
            ...                   rows=("low", "high")),
            ...     )
            ...     [text.get_text() for text in key.get_texts()]
            ...     key.get_title().get_text()
            ['low', 'high']
            'class'

            ```
        - Nothing here refuses a plan: a legend plan with no rows is painted as the empty framed box it
          describes, which is why :func:`plan_guide` returns ``None`` instead of handing one over:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Scene
            >>> from digitalearth.static.guides import GuidePlan, paint_guide
            >>> with Scene() as scene:
            ...     empty = paint_guide(scene, GuidePlan("legend", title="class"))
            ...     [text.get_text() for text in empty.get_texts()]
            ...     empty.get_title().get_text()
            []
            'class'

            ```
    """
    if plan.kind == "legend":
        # `setdefault` rather than a keyword beside `**kwargs`: the recorded title is this call's default,
        # not something the caller is forbidden to override. Passed positionally-by-name it made `title=`
        # — the one keyword a legend most obviously takes — the one keyword `Renderer.draw_guide` could not
        # be given, raising `got multiple values for keyword argument 'title'` (review L4).
        kwargs.setdefault("title", plan.title)
        if plan.anchor is not None:
            kwargs.setdefault("loc", _LEGEND_LOCATIONS[plan.anchor])
        if plan.hatches:
            return _paint_hatch_legend(scene, plan, **kwargs)
        return disjoint_legend(scene.ax, list(plan.colors), list(plan.rows), **kwargs)
    if plan.anchor is not None:
        kwargs.setdefault("location", _COLORBAR_SIDES[plan.anchor])
    bar = colorbar_legend(plan.mappable, ax=scene.ax, **kwargs)
    if plan.title is not None:
        bar.set_label(plan.title)
    return bar


def _is_transparent(color: str) -> bool:
    """Say whether a swatch colour is fully see-through, by its alpha rather than by its spelling.

    The reading was ``face.endswith("00")``, which is true of every 8-digit hex :func:`plan_guide`
    produces for an uncoloured band — and also of an *opaque* 6-digit one whose blue channel happens to be
    zero, so ``#aabb00`` was dropped as transparent (review L3). ``GuidePlan`` + :func:`paint_guide` is a
    public seam and a plan built by hand may spell its colours either way.

    Args:
        color: A matplotlib colour as the plan carries it, or the string ``"None"`` for a row the scale
            gave no colour — which matplotlib itself reads as fully transparent.

    Returns:
        ``True`` when the colour's alpha is zero to within :data:`_TRANSPARENT_ALPHA` — half an 8-bit
        step, so no alpha an 8-bit channel could show is admitted. **Fully** see-through, not merely
        faint: ``#00000001``, the smallest non-zero alpha an 8-digit hex can spell, is ``False``.
        ``False`` for anything matplotlib cannot parse too, so an unreadable colour still reaches the
        engine that will say so, rather than being taken for transparent here.

    Examples:
        - An uncoloured band's 8-digit hex and the scale's ``"None"`` both read as see-through:
            ```python
            >>> from digitalearth.static.guides import _is_transparent
            >>> (_is_transparent("#aabbcc00"), _is_transparent("None"))
            (True, True)

            ```
        - An opaque colour whose blue channel is zero does not, which is what reading the spelling got
          wrong; nor does a colour matplotlib cannot parse at all:
            ```python
            >>> from digitalearth.static.guides import _is_transparent
            >>> (_is_transparent("#aabb00"), _is_transparent("mauve-ish"))
            (False, False)
            >>> "#aabb00".endswith("00")  # the reading this replaced
            True

            ```
        - The tolerance admits what ``== 0.0`` admitted and nothing an 8-bit channel can show beyond it:
          ``#00000001``, the faintest alpha an 8-digit hex can spell, is a whole 8-bit step and stays
          opaque, while the bound itself is half a step:
            ```python
            >>> from matplotlib.colors import to_rgba
            >>> from digitalearth.static.guides import _TRANSPARENT_ALPHA, _is_transparent
            >>> round(to_rgba("#00000001")[3], 6), round(_TRANSPARENT_ALPHA, 6)
            (0.003922, 0.001961)
            >>> _is_transparent("#00000001"), _is_transparent("#00000000")
            (False, True)

            ```
    """
    try:
        return bool(to_rgba(color)[3] <= _TRANSPARENT_ALPHA)
    except ValueError:
        return False


def _paint_hatch_legend(scene: Any, plan: GuidePlan, **kwargs: Any) -> Any:
    """Draw a hatched layer's key: each swatch the band's colour, if it has one, under the band's pattern.

    An uncoloured overlay — every band drawn with ``fill=False`` — is cleopatra's ``hatch_legend``: unfilled
    swatches stroked in the hatch colour. A coloured one is the ordinary swatch list with each band's pattern
    laid over its colour, since ``hatch_legend`` gives every swatch one face.

    Args:
        scene: The scene the layer is drawn on.
        plan: The derived key, carrying :attr:`GuidePlan.hatches`.
        **kwargs: Forwarded to ``Axes.legend`` through cleopatra.

    Returns:
        The ``Legend``.
    """
    faces = [str(color) for color in plan.colors]
    if all(_is_transparent(face) for face in faces):
        return hatch_legend(
            scene.ax,
            list(plan.hatches),
            list(plan.rows),
            edgecolor=plan.hatch_color or "black",
            **kwargs,
        )
    legend = disjoint_legend(scene.ax, faces, list(plan.rows), **kwargs)
    for handle, pattern in zip(legend.legend_handles, plan.hatches):
        handle.set_hatch(pattern or None)
        if plan.hatch_color is not None:
            handle.set_hatchcolor(plan.hatch_color)
    return legend


def draw_guide(scene: Any, layer: LayerSpec, drawn: Any, **kwargs: Any) -> Any | None:
    """Draw the colour key one layer's recorded guide asks for, and return what was drawn.

    The two halves in one call, for a caller with nothing to do between them.
    :meth:`~digitalearth.static.renderer.Renderer.draw_guide` has something to do — it takes the layer's
    previous key off — so it calls :func:`plan_guide` and :func:`paint_guide` itself, with the removal
    between them and therefore after every refusal (review M4).

    Args:
        scene: The scene the layer is drawn on.
        layer: The layer's description, which carries the guide.
        drawn: What its drawer produced, as :class:`~digitalearth.static.renderer.DrawnLayer` describes it.
        **kwargs: Forwarded to the engine, as :func:`paint_guide` describes.

    Returns:
        The one artist the key is — a ``Colorbar`` or a ``Legend`` — or ``None`` when the guide is switched
        off, when the layer carries no colour encoding, and when nothing was drawn for it.

        One artist rather than a one-long tuple: a key *is* the single object matplotlib hands back, and
        :meth:`~digitalearth.static.renderer.Renderer.draw_guide` already answers its own callers that way.
        :attr:`~digitalearth.static.renderer.DrawnLayer.guides` stays a tuple, since that is the layer's
        *record* of what has to come off with it rather than one call's result.

    Raises:
        ValueError: whatever :func:`plan_guide` refuses — see there.
    """
    plan = plan_guide(layer, drawn)
    return None if plan is None else paint_guide(scene, plan, **kwargs)
