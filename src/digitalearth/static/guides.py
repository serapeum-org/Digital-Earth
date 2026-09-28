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

from math import isfinite
from typing import Any, List, Optional, Tuple

from cleopatra.styling.styles import colorbar_legend, disjoint_legend
from matplotlib.colors import to_hex

from digitalearth.base.spec import Encoding, LayerSpec, LegendSpec, Scale

__all__ = [
    "COUNT_FIELD",
    "DENSITY_FIELD",
    "GUIDE_KIND_KEY",
    "GUIDE_KINDS",
    "GUIDE_LABELS_KEY",
    "MAGNITUDE_FIELD",
    "bar_refusal",
    "color_encoding",
    "draw_guide",
    "guide_kind",
    "source_field",
]

#: The property the kind of key a layer's guide asks for is recorded under — one of :data:`GUIDE_KINDS`.
#:
#: In ``props`` rather than on the `Guide`, because the *guide* is the shared vocabulary and this is the one
#: tier that has a choice to record: a MapLibre panel, a Bokeh toggle and a PyVista scalar bar are each one
#: piece of furniture, while matplotlib has both a colorbar and a legend and they are not interchangeable.
GUIDE_KIND_KEY: str = "guide_kind"

#: The two kinds of colour key this tier draws: a colorbar strip, or a list of labelled swatches.
GUIDE_KINDS: Tuple[str, ...] = ("colorbar", "legend")

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
#: Each is the word this tier already uses for the quantity: ``count`` is one of
#: :data:`~digitalearth.static.maps.vector.AGG_REDUCERS`, a KDE draws a density, and a u/v field colours its
#: arrows by magnitude. They are named here rather than spelled at each drawer so the three cannot drift.
COUNT_FIELD: str = "count"
DENSITY_FIELD: str = "density"
MAGNITUDE_FIELD: str = "magnitude"


def source_field(source: Any, band: int = 1) -> str:
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


def _swatches(legend: Any) -> Tuple[List[Any], List[str]]:
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
    colors: List[str] = []
    for handle in legend.legend_handles:
        face = getattr(handle, "get_facecolor", None)
        if face is None:
            return [], []
        colors.append(to_hex(face()))
    if not labels or len(colors) != len(labels):
        return [], []
    return list(labels), colors


def _scale_of(artist: Any) -> Optional[Scale]:
    """Return the scale a drawn artist colours through, read off its own norm.

    Args:
        artist: The mappable the layer's drawer produced.

    Returns:
        A classified :class:`~digitalearth.base.spec.scale.Scale` when the norm carries class edges, a
        continuous one over its limits otherwise, and ``None`` when the artist colours by nothing or the
        norm has no usable limits to publish — unmeasured limits are honest as no scale, while inventing a
        pair would label the key with numbers the layer never draws.
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


def color_encoding(field: Optional[str], drawn: Any) -> Optional[Encoding]:
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
        - A raster field publishes its band and the limits the render settled on:
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
    return Encoding.by_field("color", field, scale=_scale_of(drawn.artist))


def bar_refusal(layer: LayerSpec) -> Optional[str]:
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


def _legend_spec(scale: Scale, artist: Any, title: Optional[str]) -> LegendSpec:
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


def _rows(layer: LayerSpec, spec: LegendSpec) -> List[str]:
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
    """
    labels = layer.symbology.props.get(GUIDE_LABELS_KEY)
    if not labels:
        return [entry.label for entry in spec.entries]
    rows = [str(label) for label in labels]
    if len(rows) != len(spec.entries):
        raise ValueError(
            f"the key of layer {layer.id!r} has {len(spec.entries)} rows, so labels= needs "
            f"{len(spec.entries)} labels; got {len(rows)}"
        )
    return rows


def draw_guide(
    scene: Any, layer: LayerSpec, drawn: Any, **kwargs: Any
) -> Optional[Any]:
    """Draw the colour key one layer's recorded guide asks for, and return what was drawn.

    Args:
        scene: The scene the layer is drawn on.
        layer: The layer's description, which carries the guide.
        drawn: What its drawer produced, as :class:`~digitalearth.static.renderer.DrawnLayer` describes it.
        **kwargs: Forwarded to ``cleopatra.styling.styles.colorbar_legend`` (a bar) or ``disjoint_legend``
            (swatches). These are the call's own styling and are **not** part of the record, so a key
            redrawn from the description comes back with matplotlib's defaults in their place — the same
            trade :attr:`~digitalearth.static.scene.LayerRecord.opts` makes for a value no figure can carry.

    Returns:
        The one artist the key is — a ``Colorbar`` or a ``Legend`` — or ``None`` when the guide is switched
        off, when the layer carries no colour encoding, and when nothing was drawn for it.

        One artist rather than a one-long tuple: a key *is* the single object matplotlib hands back, and
        :meth:`~digitalearth.static.renderer.Renderer.draw_guide` — the only caller — already answers its
        own callers that way. :attr:`~digitalearth.static.renderer.DrawnLayer.guides` stays a tuple, since
        that is the layer's *record* of what has to come off with it rather than one call's result.

    Raises:
        ValueError: when a colorbar is asked for over a **categorical** scale, naming the swatch legend
            instead: a categorical fill's norm bins the class codes cleopatra assigned, so a bar over them
            reads ``0, 1, 2 …`` where the category labels belong. Also when recorded row labels do not
            number the key's rows.
    """
    guide = layer.symbology.guide()
    encoding = layer.symbology.encoding("color")
    if guide is None or not guide.show or encoding is None or drawn.artist is None:
        return None
    scale = encoding.scale
    if guide_kind(layer) == "legend":
        if scale is None:
            raise ValueError(
                f"layer {layer.id!r} publishes no colour scale, so there are no rows to key it with; "
                "colorbar() draws the bar its mappable can still carry"
            )
        spec = _legend_spec(scale, drawn.artist, guide.title)
        if guide.anchor is not None:
            kwargs.setdefault("loc", _LEGEND_LOCATIONS[guide.anchor])
        return disjoint_legend(
            scene.ax,
            [entry.color for entry in spec.entries],
            _rows(layer, spec),
            title=guide.title,
            **kwargs,
        )
    refusal = bar_refusal(layer)
    if refusal is not None:
        raise ValueError(refusal)
    if guide.anchor is not None:
        kwargs.setdefault("location", _COLORBAR_SIDES[guide.anchor])
    bar = colorbar_legend(drawn.artist, ax=scene.ax, **kwargs)
    if guide.title is not None:
        bar.set_label(guide.title)
    return bar
