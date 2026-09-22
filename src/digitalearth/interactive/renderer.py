"""Draw a :class:`~digitalearth.base.spec.FigureSpec` as HoloViews elements (#300, the interactive seam).

The tier held its drawing and nothing else: `self.layers` is a list of built elements and `self._styles` a
dict keyed by `id(element)`. An element is what HoloViews composes, not something a figure can be written
to — and an identity-keyed style table cannot survive a redraw, which is why a dynamic layer's style had to
be filed twice and the table trimmed by hand.

This is the other half of that seam. A builder records what it drew — kind, source, visibility and
symbology as values — and the drawer here rebuilds the element from that description.

**Two things the description deliberately leaves out**, so "round-trips" is read for what it is. ``to_dict``
refuses an ``object:`` source, so only a figure whose builders were given paths or URLs can be written down;
one built from a `FeatureCollection` or a dataset already in memory is drawn from its description in this
process and handed to a renderer directly. And a caller's keyword that does not travel in a figure — a
colormap built on the spot, a callable hook, a container of any kind — is held on the map beside the layer
rather than in its description (:func:`~digitalearth.interactive.base.describe_opts`), so a figure read back
elsewhere draws that layer with the engine's default in its place. The static tier's module docstring states
the same two, for the same reasons; both tiers lose exactly what
:func:`~digitalearth.base.spec._serial.travels_in_a_figure` refuses, and nothing else.

**This tier composes rather than mutates.** PyVista hands out a live plotter whose actors are mutated in
place; HoloViews elements are immutable values composed into an overlay on every `render()`. So
:meth:`Renderer.apply` reconciles the renderer's own *record* of what is drawn — :attr:`Renderer.drawn` —
and the observable contract it signs is the one the shared renderer conformance suite states, which is why
all four tiers can sign it.

**On this tier, for this wave, `apply` is record-only** — with one exception a caller can see. It does not
change *which* elements `render()` overlays, which is the map's `layers`, and it does not change what
`figure_spec` reports, which is the map's own layer tree. The exception is visibility: a layer `diff`
reports as shown or hidden is toggled on the very element the map registered, because `.opts()` writes into
HoloViews' global `Store` against that object. So a figure applied with a hidden layer draws it hidden
while `figure_spec` still describes it visible.
Nothing in the tier calls it: every builder draws through :meth:`Renderer.draw_layer`, one layer at a time.
Wiring `apply` into the map's public state is Wave 7 (order 23); until then a caller who applies a figure
has moved the record and nothing a viewer sees.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from digitalearth.base.registry import band_of
from digitalearth.base.spec import FigureSpec, LayerSpec
from digitalearth.interactive.capabilities import CAPABILITIES


@dataclass(frozen=True)
class DrawnLayer:
    """What one drawer produced: the HoloViews element a layer composes into.

    Every drawer answers in this shape so the compose step can take whatever came back without knowing
    which drawer made it — the equivalent of the web tier's source-and-layer pair, and of the 3-D tier's
    `(mesh, actor)`.

    Attributes:
        element: The HoloViews/GeoViews element, or a `DynamicMap` for a layer redrawn per frame.
        style: The options applied to it, as values. Kept beside the element because `.opts()` writes into
            HoloViews' global `Store` and returns nothing a caller can read back.

    Examples:
        - One builder call, one element, with the style it was drawn with beside it:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.interactive import InteractiveMap
            >>> wells = gpd.GeoDataFrame(
            ...     {"depth": [12.0, 31.0]},
            ...     geometry=[Point(4.9, 52.4), Point(5.1, 52.1)],
            ...     crs=4326,
            ... )
            >>> drawn = InteractiveMap().points(wells)._renderer.drawn["points-1"]
            >>> type(drawn.element).__name__
            'Points'
            >>> sorted(drawn.style)
            ['size']

            ```
    """

    element: Any
    style: Mapping[str, Any] = None  # type: ignore[assignment]


#: The layer kinds this tier draws **from its description**. Names only, so what is drawable can be asked
#: without importing every builder module behind them; `drawer_for` resolves them and is checked against
#: this tuple, which keeps the two from drifting.
#:
#: Every kind this tier declares in :data:`~digitalearth.interactive.capabilities.CAPABILITIES` is here, so
#: a kind is built by its drawer and never by its builder. The one kind outside it is `custom:holoviews` —
#: an element a caller built and handed to `add_element`, which has no description to rebuild it from and
#: so is drawn by being kept.
DRAWN_KINDS: Tuple[str, ...] = (
    "graticule",
    "text",
    "coastlines",
    "points",
    "lines",
    "polygons",
    "choropleth",
    "raster",
    "rgb",
    "mesh",
    "contours",
    "filled_contours",
    "heatmap",
    "flow",
    "vectors",
    "streamlines",
    "unstructured",
    "basemap",
    "labels",
    "land",
    "ocean",
    "borders",
    "rivers",
    "lakes",
)


#: The Bokeh style keywords that mean "do not draw this". Nearly every element spells it `visible`; a
#: graph-shaped one (`TriMesh`, `Graph`) splits the question into its edges and its nodes and has no single
#: `visible` at all. Which of these an element actually takes is read from the engine's own option table
#: rather than listed per element here, the way :mod:`digitalearth.interactive.style_fold` reads everything
#: else it folds — so this cannot drift from what HoloViews accepts.
_VISIBILITY_KEYWORDS: Tuple[str, ...] = ("visible", "edge_visible", "node_visible")


def _visibility_keywords(element: Any) -> Tuple[str, ...]:
    """Return the keywords this element spells "do not draw me" with, as its backend takes them.

    Args:
        element: The element (or `DynamicMap`) a drawer produced.

    Returns:
        The accepted keywords of :data:`_VISIBILITY_KEYWORDS`, in that order; empty for an element the
        Bokeh backend gives no way to hide. Two shapes answer that way: an element whose type has no
        visibility option — `Tiles` and `WMTS`, whose image *is* the basemap — and any type the Bokeh
        option tree does not hold at all, which includes a `DynamicMap` or `HoloMap` that has not yet
        produced a frame, so every `dynamic=True` rasterize, datashade, trajectory or `large_image`
        layer.
    """
    from digitalearth.interactive.style_fold import allowed_options

    element_type = type(element)
    # A `DynamicMap` is a producer of elements rather than one itself, so the options are the element
    # type's. `type` is `None` until the map has produced a frame, in which case there is nothing to ask.
    if element_type.__name__ in ("DynamicMap", "HoloMap"):
        element_type = getattr(element, "type", None) or element_type
    try:
        style = allowed_options(element_type.__name__)["style"]
    except KeyError:
        return ()
    return tuple(keyword for keyword in _VISIBILITY_KEYWORDS if keyword in style)


def _show(element: Any, layer_id: str, visible: bool) -> None:
    """Draw or stop drawing one element, in place.

    HoloViews' ``.opts()`` writes into its global option `Store` against the element it is called on and
    returns that same element, so the object the map registered is the object this changes — no element has
    to be swapped out of `InteractiveMap.layers`, and the style filed against it stays filed.

    Args:
        element: The element to show or hide.
        layer_id: The layer it was drawn for, for the warning below.
        visible: Whether it is drawn.

    Warns:
        UserWarning: when the element has no keyword to say it with — a `Tiles`/`WMTS` basemap, or a
            `DynamicMap` that has not produced a frame, which is what every `dynamic=True` layer is at
            build time (see :func:`_visibility_keywords`). Saying nothing would leave a figure that
            describes a hidden layer drawing it — the silence review M4 reports on this tier. The warning
            is worded for the hide direction because that is the case it costs something; it is emitted
            before `visible` is read, so `set_visible(id, True)` on such an element warns too.
    """
    keywords = _visibility_keywords(element)
    if not keywords:
        warnings.warn(
            f"layer {layer_id!r} is described hidden, but the interactive tier draws it as a "
            f"{type(element).__name__}, which Bokeh gives no way to hide; it stays drawn",
            UserWarning,
            stacklevel=3,
        )
        return
    element.opts(**{keyword: bool(visible) for keyword in keywords})


def _is_shown(element: Any) -> bool:
    """Whether HoloViews would currently draw one element.

    Args:
        element: The element to ask.

    Returns:
        The element's own visibility options, as the Bokeh backend resolved them. `True` for an element
        with no such option, which is one that cannot be hidden and so is never hidden.
    """
    import holoviews as hv

    keywords = _visibility_keywords(element)
    if not keywords:
        return True
    # Asked of the frame, not of the producer: a `DynamicMap` carries no style of its own, so looking the
    # applied options up against it returns nothing and the defaults answer "drawn" for a layer that is
    # hidden. `_visibility_keywords` already resolves the producer to the type it makes; this resolves it to
    # the object that holds what was applied. Before a frame exists there is nothing to ask, and a layer
    # nobody has drawn yet is drawn.
    asked = element
    if type(element).__name__ in ("DynamicMap", "HoloMap"):
        asked = getattr(element, "last", None) or element
    applied = hv.Store.lookup_options("bokeh", asked, "style").kwargs
    return all(bool(applied.get(keyword, True)) for keyword in keywords)


def _recipes() -> Dict[str, Dict[str, Any]]:
    """Return the drawers of this tier, by kind and then by the recipe each was built with.

    A kind is drawn more than one way here — `points` is a frame of geometry or a datashaded aggregate,
    `lines` a path or a trajectory — and the kind cannot say which, because the kind vocabulary is shared
    with every other tier and names *what* a layer is. So each builder records *how* it built its layer
    under `via`, and that is the second key.

    Returns:
        Kind -> recipe -> ``draw(interactive_map, data, layer) -> DrawnLayer | None``.
    """
    # Imported here rather than at module level: every builder module imports the map, so a
    # module-level import would close a cycle.
    from digitalearth.interactive import (
        bigdata,
        decoration,
        projection,
        raster,
        temporal,
        vector,
    )

    return {
        "graticule": {"graticule": projection.draw_graticule},
        "text": {"text": decoration.draw_text},
        "coastlines": {"coastlines": decoration.draw_coastlines},
        "basemap": {"tiles": decoration.draw_tiles},
        "labels": {"labels": decoration.draw_labels},
        "land": {"natural_earth": decoration.draw_natural_earth},
        "ocean": {"natural_earth": decoration.draw_natural_earth},
        "borders": {"natural_earth": decoration.draw_natural_earth},
        "rivers": {"natural_earth": decoration.draw_natural_earth},
        "lakes": {"natural_earth": decoration.draw_natural_earth},
        "points": {
            "geometry": vector.draw_vector,
            "datashade": bigdata.draw_datashade,
        },
        "lines": {
            "geometry": vector.draw_vector,
            "trajectory": bigdata.draw_trajectory,
        },
        "polygons": {"geometry": vector.draw_vector},
        "choropleth": {"geometry": vector.draw_vector},
        "raster": {
            "image": raster.draw_image,
            "large_image": raster.draw_large_image,
            "rasterize": bigdata.draw_rasterize,
            "timecube": temporal.draw_timecube,
        },
        "rgb": {"rgb": raster.draw_rgb},
        "mesh": {"quadmesh": raster.draw_quadmesh},
        "contours": {"contours": raster.draw_contours},
        # Two ways to draw one density, told apart by the recipe rather than by two invented kinds.
        "heatmap": {"hexbin": vector.draw_hexbin, "kde": vector.draw_kde},
        "flow": {"graph": vector.draw_graph},
        # Arrows and wind barbs are one u/v field drawn with two glyphs, which is what the registry's
        # `vectors` entry already says ("static quiver/barbs, interactive vectorfield/barbs").
        "vectors": {"vectorfield": vector.draw_uv_field, "barbs": vector.draw_uv_field},
        "streamlines": {"streamlines": vector.draw_uv_field},
        "unstructured": {"trimesh": vector.draw_trimesh},
        "filled_contours": {"contours": raster.draw_contours},
    }


def _dispatch(kind: str, recipes: Dict[str, Any]) -> Any:
    """Return the drawer for one kind, which picks between the recipes that kind is drawn by.

    Args:
        kind: The kind being drawn, used only to name it in a refusal.
        recipes: Recipe name -> drawer.

    Returns:
        A callable ``draw(interactive_map, data, layer)`` that reads the layer's recorded `via` and calls
        the drawer for it.
    """

    def draw(interactive_map: Any, data: Any, layer: LayerSpec) -> Optional[DrawnLayer]:
        """Draw the layer with the drawer its recipe names.

        Args:
            interactive_map: The map being drawn.
            data: The layer's opened source, or `None`.
            layer: The layer's description.

        Returns:
            What the drawer produced.

        Raises:
            KeyError: when the layer records no recipe this tier knows — a builder that was routed to a
                kind without recording how it draws it.
        """
        via = layer.symbology.props.get("via")
        if via not in recipes:
            raise KeyError(
                f"a {kind!r} layer records {via!r} as how it was drawn; this tier draws one of "
                f"{sorted(recipes)}"
            )
        drawn: Optional[DrawnLayer] = recipes[via](interactive_map, data, layer)
        return drawn

    return draw


def drawer_for(kind: str) -> Any:
    """Return the function that draws one kind of layer in this tier.

    Args:
        kind: A registered layer kind.

    Returns:
        A callable ``draw(interactive_map, data, layer) -> DrawnLayer | None``, where `data` is the layer's
        opened source (or `None` for a layer drawn from no source) and `layer` is its `LayerSpec`.

    Raises:
        KeyError: when this tier does not draw `kind`, naming the kinds it does and, when the tier declared
            one, the reason it does not draw this one; or when the drawer table and `DRAWN_KINDS` disagree,
            which is a defect in this module rather than in the caller.

    Examples:
        - A kind this tier has no drawer for is refused with the kinds it does draw:
            ```python
            >>> from digitalearth.interactive.renderer import drawer_for
            >>> drawer_for("model")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            KeyError: "the interactive tier does not draw 'model' layers; it draws [...]"

            ```
        - A kind it does draw comes back as a dispatcher, which reads the layer's own recipe — the `via` a
          builder recorded — because one kind is drawn more than one way here:
            ```python
            >>> from digitalearth.base.spec import LayerSpec, Symbology
            >>> from digitalearth.interactive.renderer import drawer_for
            >>> invented = LayerSpec("wells", "points", symbology=Symbology(props={"via": "smoke"}))
            >>> drawer_for("points")(None, None, invented)
            Traceback (most recent call last):
                ...
            KeyError: "a 'points' layer records 'smoke' as how it was drawn; this tier draws one of [...]"

            ```
    """
    if kind not in DRAWN_KINDS:
        # The reason is the tier's own, read from the declaration rather than written again here (#294).
        # This tier declares no layer kind absent today, so the clause is usually empty — but the rule is
        # the same on all four tiers, and a kind it later decides against explains itself for free.
        reason = CAPABILITIES.reason(kind)
        raise KeyError(
            f"the interactive tier does not draw {kind!r} layers"
            + (f" — {reason}" if reason else "")
            + f"; it draws {sorted(DRAWN_KINDS)}"
        )
    recipes = _recipes()
    if set(recipes) != set(DRAWN_KINDS):
        raise KeyError(
            f"the interactive tier's drawer table and DRAWN_KINDS disagree: "
            f"{sorted(set(recipes).symmetric_difference(DRAWN_KINDS))}"
        )
    return _dispatch(kind, recipes[kind])


class Renderer:
    """Draws a figure's layers as HoloViews elements, and reconciles one figure with another.

    Attributes:
        drawn: Layer id to what was drawn for it, in draw order.

    Examples:
        - A map's renderer holds what it drew, keyed by the same ids
          :attr:`~digitalearth.interactive.base.InteractiveMapBase.layer_ids` lists, in the order the
          layers were drawn:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.interactive import InteractiveMap
            >>> wells = gpd.GeoDataFrame(geometry=[Point(4.9, 52.4)], crs=4326)
            >>> list(InteractiveMap().points(wells).graticule()._renderer.drawn)
            ['points-1', 'graticule-2']

            ```
    """

    def __init__(self, interactive_map: Any) -> None:
        """Bind the renderer to the map it draws for.

        Args:
            interactive_map: The `InteractiveMap` this renderer serves. Held rather than passed per call,
                because a drawer reads the map's own helpers — the display CRS, the styling recorder, the
                auto-colormap resolution.
        """
        self._map = interactive_map
        self._drawn: dict = {}

    @property
    def drawn(self) -> Mapping[str, DrawnLayer]:
        """What has been drawn, by layer id.

        Returns:
            Layer id to the `DrawnLayer` its drawer produced, in the order the layers were drawn.
        """
        return dict(self._drawn)

    def draw_layer(self, figure: FigureSpec, layer_id: str) -> Optional[DrawnLayer]:
        """Draw one of a figure's layers and record what it produced.

        Args:
            figure: The figure holding the layer and its source.
            layer_id: Which layer to draw.

        Returns:
            What the drawer produced, or `None` for a layer it declined to draw.

        Raises:
            KeyError: when no layer has that id, or the tier has no drawer for its kind.

        Warns:
            UserWarning: when the figure describes the layer as not drawn and the element has no keyword
                to say so with — a basemap, or a `dynamic=True` layer, whose `DynamicMap` has produced no
                frame yet (see :func:`_show`). The layer is drawn, visible, and the warning says so.

        Examples:
            - Drawing a layer again from a figure that describes it hidden draws it hidden:
                ```python
                >>> from dataclasses import replace
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from digitalearth.interactive import InteractiveMap
                >>> wells = gpd.GeoDataFrame(geometry=[Point(4.9, 52.4)], crs=4326)
                >>> m = InteractiveMap().points(wells)
                >>> figure = m.figure_spec
                >>> hidden = replace(figure, layers=figure.layers.set_visible("points-1", False))
                >>> _ = m._renderer.draw_layer(hidden, "points-1")
                >>> m._renderer.is_visible("points-1")
                False

                ```
        """
        layer = figure.layers.get(layer_id)
        data = self._source_object(figure, layer)
        drawn: Optional[DrawnLayer] = drawer_for(layer.kind)(self._map, data, layer)
        if drawn is not None:
            self._drawn[layer_id] = drawn
            if not figure.layers.is_visible(layer_id):
                # This tier read neither the layer's flag nor its group, so a figure describing a hidden
                # layer drew it visible — with the description carrying `visible: False` all the while
                # (review M4). Asked of the tree, which is the flag *and* the group, so all four tiers now
                # answer "is this layer drawn hidden?" the same way.
                self.set_visible(layer_id, False)
        return drawn

    @staticmethod
    def _source_object(figure: FigureSpec, layer: LayerSpec) -> Any:
        """Return the object a layer draws, or `None` when it draws from no source.

        Args:
            figure: The figure holding the sources.
            layer: The layer being drawn.

        Returns:
            The opened source, or `None` for a layer drawn from nothing, such as a graticule.
        """
        if layer.source_id is None:
            return None
        return figure.sources[layer.source_id].open()

    def apply(self, before: FigureSpec, after: FigureSpec) -> None:
        """Bring the renderer's record of what is drawn from one figure to another.

        **Record-only on this tier, for this wave** — with one exception. It reconciles :attr:`drawn`
        and nothing else: *which* elements the map's `render()` overlays, and the figure its `figure_spec`
        reports, are the map's own and neither follows. The exception is visibility: a layer `diff` reports
        as shown or hidden is toggled on the very element the map holds, so applying a figure that hides a
        layer draws it hidden while `figure_spec` still describes it visible. Nothing in the tier calls
        this yet; wiring it into the map is Wave 7 (order 23).

        **A restyle expressed only in a value the figure cannot carry is invisible to this path.** The
        difference between two figures is read off their descriptions, and a keyword that does not travel in
        one — a colormap built on the spot, a callable hook, a dash pattern — is held on the map beside the
        layer rather than described (:func:`~digitalearth.interactive.base.describe_opts`). Two layers
        differing only in one of those therefore compare **equal**: `diff` reports no restyle,
        :meth:`_reaches_holoviews` answers `False`, and nothing is redrawn, although the two draw different
        pictures. A plain scalar is described and does reach here, so this is exactly the set
        :func:`~digitalearth.base.spec._serial.travels_in_a_figure` refuses, and no wider (review M3, M5;
        #322).

        Args:
            before: The figure the record currently holds.
            after: The figure it should hold.

        Raises:
            KeyError: when a layer names a kind this tier does not draw. The record is rolled back to what it
                held before the call, so a refusal part-way through leaves no layer drawn that no figure owns.
        """
        # `apply` is not atomic: it draws layer by layer, so a refusal on the third has already drawn the
        # first two. Rolling back only the caller's description would leave this record holding layers no
        # figure owns — the defect the shared conformance contract states, and which the other tiers hit too.
        held = dict(self._drawn)
        try:
            self._reconcile(before, after)
        except BaseException:
            # `BaseException`, the same class the static tier catches: what the record must survive is a
            # change stopping part-way, and a `KeyboardInterrupt` stops it exactly as an error does. Three
            # tiers signing one contract with two answers to "what is a refusal" is the drift (review N2).
            self._drawn = held
            raise

    def _reconcile(self, before: FigureSpec, after: FigureSpec) -> None:
        """Draw the difference between two figures, layer by layer.

        Removed, rebuilt, restyled and added layers pass through this renderer's record; shown and hidden
        ones do not — they are toggled on the elements themselves, which is the one way this tier's
        `apply` is not record-only (see :meth:`apply`).

        Args:
            before: The figure the map currently draws.
            after: The figure it should draw.

        Raises:
            KeyError: when a layer names a kind this tier does not draw.

        Warns:
            UserWarning: for a shown or hidden layer whose element has no way to say it — see
                :func:`_show`.
        """
        change = before.diff(after)
        for layer_id in change.removed:
            self.remove(layer_id)
        # A rebuilt layer is one whose *data* changed — its kind, source or slice — and every one of those
        # means a new element. Only a restyle is worth asking about, because `diff` groups a change of
        # `label` with a change of colour and one of those never reaches HoloViews.
        for layer_id in change.rebuilt:
            self.remove(layer_id)
            self.draw_layer(after, layer_id)
        for layer_id in change.restyled:
            if not self._reaches_holoviews(before, after, layer_id):
                continue
            self.remove(layer_id)
            self.draw_layer(after, layer_id)
        for layer_id in change.added:
            self.draw_layer(after, layer_id)
        for layer_id in change.shown:
            self.set_visible(layer_id, True)
        for layer_id in change.hidden:
            self.set_visible(layer_id, False)

    @staticmethod
    def _reaches_holoviews(
        before: FigureSpec, after: FigureSpec, layer_id: str
    ) -> bool:
        """Whether a restyle changes anything HoloViews draws.

        Args:
            before: The figure drawn now.
            after: The figure to draw.
            layer_id: The layer whose restyle is in question.

        Returns:
            `True` when the layer's symbology, filter or group differs — the parts a drawer reads — and for
            a layer either figure does not hold. `False` for a change to `label` alone.
        """
        try:
            was = before.layers.get(layer_id)
            now = after.layers.get(layer_id)
        except KeyError:
            return True
        return (was.symbology, was.filter, was.group) != (
            now.symbology,
            now.filter,
            now.group,
        )

    def remove(self, layer_id: str) -> None:
        """Forget what was drawn for a layer.

        Args:
            layer_id: The layer to remove. An id nothing was drawn for is ignored, so a caller can remove
                a layer the tier declined to draw.
        """
        self._drawn.pop(layer_id, None)

    def set_visible(self, layer_id: str, visible: bool) -> None:
        """Draw or stop drawing the element recorded for a layer.

        The element is changed in place — ``.opts()`` writes into HoloViews' option `Store` against the
        object it is called on and hands that same object back — so what `InteractiveMap.layers` holds, and
        the style filed against it, are the ones this changes.

        Args:
            layer_id: The layer to toggle. An id nothing was drawn for is ignored, which is how the other
                three tiers answer one too.
            visible: Whether it is drawn.

        Warns:
            UserWarning: when the element has no keyword to say it with, in either direction — see
                :func:`_show`, which this delegates to.

        Examples:
            - Hiding a layer and reading it back, off the element rather than off the description:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from digitalearth.interactive import InteractiveMap
                >>> wells = gpd.GeoDataFrame(geometry=[Point(4.9, 52.4)], crs=4326)
                >>> renderer = InteractiveMap().points(wells)._renderer
                >>> renderer.is_visible("points-1")
                True
                >>> renderer.set_visible("points-1", False)
                >>> renderer.is_visible("points-1")
                False

                ```
        """
        drawn = self._drawn.get(layer_id)
        if drawn is not None:
            _show(drawn.element, layer_id, visible)

    def is_visible(self, layer_id: str) -> bool:
        """Whether HoloViews would currently draw the element recorded for a layer.

        The read-back of :meth:`set_visible`. Every tier's renderer answers this, in its own terms, so the
        question "is this layer drawn hidden?" can be asked of any of them — which is what the shared
        renderer conformance suite does (review M4).

        Args:
            layer_id: The layer to ask about.

        Returns:
            `True` when the element carries no visibility option turned off, or has none to carry.

        Raises:
            KeyError: when nothing was drawn for `layer_id`, naming it. A layer the overlay does not hold
                has no visibility to report, and :attr:`drawn` is what says which those are.

        Examples:
            - An id nothing was drawn for is refused by name, with what the tier does hold:
                ```python
                >>> import geopandas as gpd
                >>> from shapely.geometry import Point
                >>> from digitalearth.interactive import InteractiveMap
                >>> wells = gpd.GeoDataFrame(geometry=[Point(4.9, 52.4)], crs=4326)
                >>> InteractiveMap().points(wells)._renderer.is_visible("nope")  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                KeyError: "nothing is drawn for layer 'nope', so it has no visibility to report; ..."

                ```
        """
        drawn = self._drawn.get(layer_id)
        if drawn is None:
            raise KeyError(
                f"nothing is drawn for layer {layer_id!r}, so it has no visibility to report; the "
                f"interactive tier holds {sorted(self._drawn)}"
            )
        return _is_shown(drawn.element)

    def band_for(self, layer: LayerSpec) -> str:
        """Return the draw-order band a layer belongs to.

        Args:
            layer: The layer being placed.

        Returns:
            The layer's own band when it declares one — what a caller's `custom:holoviews` object needs,
            since the engine name says nothing about what it draws — else its kind's band.

        Examples:
            - A points layer that declares no band of its own is placed by its kind:
                ```python
                >>> from digitalearth.base.spec import LayerSpec
                >>> from digitalearth.interactive import InteractiveMap
                >>> renderer = InteractiveMap()._renderer
                >>> renderer.band_for(LayerSpec("wells", "points"))
                'data'
                >>> renderer.band_for(LayerSpec("wells", "points", band="overlay"))
                'overlay'

                ```
        """
        return layer.band or band_of(layer.kind)
