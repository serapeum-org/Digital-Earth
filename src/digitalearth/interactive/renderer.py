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
colormap built on the spot, a callable hook, a tuple such as a dash pattern, an array — is held on the map
beside the layer rather than in its description (:func:`~digitalearth.interactive.base.describe_opts`), so a
figure read back elsewhere draws that layer with the engine's default in its place. A ``list`` or ``dict``
of plain values is **not** in that half: JSON reads it back as itself, so it is described, which is what
keeps a reloaded figure's palette (#330). Measured: ``describe_opts`` describes
``color_levels=[0.0, 0.5, 1.0]`` and holds ``line_dash=(0, (5, 5))``. The static tier's module docstring
states the same two, for the same reasons; both tiers lose exactly what
:func:`~digitalearth.base.spec._serial.travels_in_a_figure` refuses, and nothing else.

**This tier composes rather than mutates.** PyVista hands out a live plotter whose actors are mutated in
place; HoloViews elements are immutable values composed into an overlay on every `render()`. So
:meth:`Renderer.apply` reconciles the renderer's own *record* of what is drawn — :attr:`Renderer.drawn` —
and the observable contract it signs is the one the shared renderer conformance suite states, which is why
all four tiers can sign it.

**`apply` reaches what the map draws.** It was record-only for a wave: it changed neither *which*
elements `render()` overlays — the map's `layers` — nor what `figure_spec` reports, so a caller who applied
a figure had moved a record and nothing a viewer sees. Two halves close that. :meth:`Renderer._arrange`
re-lays `InteractiveMap.layers` from the applied figure's draw order, so a removed layer leaves the overlay,
an added one enters it and a moved one changes what is on top; and
:meth:`~digitalearth.interactive.base.InteractiveMapBase._change` — the path `get_layer`, `remove_layer`,
`set_visible`, `move_layer` and `replace_layer` take — installs the description once this has returned, so a
figure `apply` refuses is never one the map describes. Visibility was the one thing that always reached the
element, because `.opts()` writes into HoloViews' global `Store` against the object the map holds.
"""

import warnings
from dataclasses import dataclass
from dataclasses import replace as with_fields
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple

from digitalearth.base.capabilities import CapabilityError
from digitalearth.base.custom import custom_kind
from digitalearth.base.registry import band_of
from digitalearth.base.spec import FigureDiff, FigureSpec, LayerSpec
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
#: an element a caller built and handed to `add_layer`, which has no description to rebuild it from and
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


#: The kinds this tier **declares and keeps** rather than draws, each with the reason — the other half of
#: :data:`DRAWN_KINDS`, and the same pair `INTERACTIVE_UNDRAWN_KINDS` in the shared renderer conformance
#: suite holds. `Capabilities.require` cannot answer for these: the tier does support the kind, which is how
#: `add_layer` accepts one, and what it cannot do is build a second element from a description that never
#: described the first. So a change that would need one built is refused against this table, before a drawer
#: is asked (review H7); `CAPABILITIES.absent` stays for the kinds the tier does not support at all.
KEPT_KINDS: Mapping[str, str] = MappingProxyType(
    {
        custom_kind("holoviews"): (
            "a caller's own element, drawn by being kept: add_layer() holds no description to rebuild it "
            "from"
        ),
    }
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
        UserWarning: when the element has no keyword to say it with — a `Tiles`/`WMTS` basemap, a
            `DynamicMap` that has not produced a frame, which is what every `dynamic=True` layer is at
            build time (see :func:`_visibility_keywords`), or an object a caller handed `add_layer` that
            HoloViews has no option tree for at all. Saying nothing would leave a figure that describes a
            hidden layer drawing it — the silence review M4 reports on this tier, and review H8 on the
            custom-layer half of it. The warning is worded for the hide direction because that is the case
            it costs something; it is emitted before `visible` is read, so `set_visible(id, True)` on such
            an element warns too.
    """
    keywords = _visibility_keywords(element)
    if not keywords:
        # "its element is a …" rather than "the tier draws it as a …": this path is reached for a caller's
        # own object too, which the tier keeps rather than draws (review H8).
        warnings.warn(
            f"layer {layer_id!r} is described hidden, but its element is a "
            f"{type(element).__name__}, which Bokeh gives no way to hide; it stays drawn",
            UserWarning,
            stacklevel=3,
        )
        return
    element.opts(**{keyword: bool(visible) for keyword in keywords})


def _draw_recorded_guide(layer: LayerSpec, drawn: DrawnLayer) -> DrawnLayer:
    """Put the colour key a layer's description asks for back onto the element just drawn for it.

    **A drawer builds from `props`, and a guide is not one.** `colorbar()`/`legend()` record the guide on the
    layer's colour encoding and then apply the options to the element the map holds *at that moment*; a
    redraw — a restyle, a rollback, a figure read back onto another map — hands back a new element on which
    nothing has ever been applied. Without this the description went on saying "hidden, titled Flow" while
    the picture drew a visible untitled bar (review H2), which is the divergence the static tier avoids by
    calling :meth:`~digitalearth.static.renderer.Renderer.draw_guide` from its own `draw_layer`.

    Reapplied **here** rather than by recording through
    :meth:`~digitalearth.interactive.base.InteractiveMapBase._change`: a guide is description and nothing
    else, and routing the record through a reconcile would restyle the layer, redraw it and hand back a new
    element — orphaning the styles :attr:`~digitalearth.interactive.base.InteractiveMapBase._styles` files by
    ``id()`` and any stream a `hover()` attached. So the record still goes straight into the tree, and the
    *draw* reads it back.

    Args:
        layer: The layer's description, which carries the guide and the kind of key it asks for.
        drawn: What its drawer just produced.

    Returns:
        `drawn` carrying the element with the key applied, or `drawn` itself when the layer describes no key
        — or describes one this element has no option to draw, which the recording call already warned
        about and which a redraw would only repeat once per rebuild.
    """
    from digitalearth.interactive.style_fold import fold_guide, split_guide_options

    opts = fold_guide(layer.symbology, layer.label)
    if not opts:
        return drawn
    taken, _refused = split_guide_options(drawn.element, opts)
    if not taken:
        return drawn
    return with_fields(drawn, element=drawn.element.opts(**taken, backend="bokeh"))


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
          layers were drawn. Each generated id counts **within its kind**, so the second layer of a
          second kind is ``-1`` and not ``-2``: this example said ``graticule-2`` until review R2-M10
          gave the tier one counter per prefix, and it is the drawn ids themselves that say otherwise
          (`R2-N3`):
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.interactive import InteractiveMap
            >>> wells = gpd.GeoDataFrame(geometry=[Point(4.9, 52.4)], crs=4326)
            >>> list(InteractiveMap().points(wells).graticule()._renderer.drawn)
            ['points-1', 'graticule-1']

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
            drawn = _draw_recorded_guide(layer, drawn)
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
        """Bring what the map overlays from one figure to another, or leave it as it was.

        **It reaches what the tier draws.** `render()` composes `InteractiveMap.layers`, and until order 23
        only the builders wrote that list: `apply` reconciled :attr:`drawn` and nothing a viewer would see,
        so after a remove and an add the overlay still showed what the builders had put down (review M1).
        Now :meth:`_arrange` re-arranges the list from `after`'s draw order — which is what makes a removed
        layer leave the picture, an added one enter it, and a moved one change what is on top.

        The figure the map *reports* is still the map's own:
        :meth:`~digitalearth.interactive.base.InteractiveMapBase._change` installs that once this has
        returned, so a figure this refuses is never one the map describes.

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
            CapabilityError: when the change would have to build a layer of a kind this tier **keeps**
                rather than draws — a caller's own element, which has no description to rebuild it from
                (:data:`KEPT_KINDS`). Refused before the first layer is drawn, so nothing is rolled back.
            KeyError: when a layer names a kind this tier does not draw at all. The record **and** the
                overlay are rolled back to what they held before the call, so a refusal part-way through
                leaves no layer drawn that no figure owns.
        """
        change = before.diff(after)
        asked = self._would_draw(before, after, change)
        # Before the first draw, not from the drawer table part-way through one: the gate on
        # `replace_layer` passes for `custom:holoviews`, because the tier *declares* that kind, and the
        # refusal then came out of `drawer_for` as a bare `KeyError` naming a kind the declaration lists
        # (review H7). Asked of the same list `_reconcile` draws from, so the two cannot disagree about
        # which layers a drawer is asked for.
        self._refuse_what_it_cannot_rebuild(after, asked)
        # `apply` is not atomic: it draws layer by layer, so a refusal on the third has already drawn the
        # first two. Rolling back only the caller's description would leave this record holding layers no
        # figure owns — the defect the shared conformance contract states, and which the other tiers hit too.
        held = dict(self._drawn)
        overlay = list(self._map.layers)
        try:
            self._reconcile(after, change, asked)
            # Inside the `try`, as the web tier's is, so the restore below is a live path rather than a
            # comment about one (review L4). The arrangement is where the record and the overlay are brought
            # back into agreement, so a failure in it leaves the record describing `after` while the overlay
            # still holds `before` — and `_change` installs no figure for a call that raised, so that record
            # is one no figure owns.
            self._arrange(before, after, held, overlay)
        except BaseException:
            # `BaseException`, the same class the static tier catches: what the record must survive is a
            # change stopping part-way, and a `KeyboardInterrupt` stops it exactly as an error does. Three
            # tiers signing one contract with two answers to "what is a refusal" is the drift (review N2).
            self._drawn = held
            self._map.layers[:] = overlay
            raise

    def _arrange(
        self,
        before: FigureSpec,
        after: FigureSpec,
        held: Mapping[str, DrawnLayer],
        overlay: List[Any],
    ) -> None:
        """Re-arrange what the map overlays so it is `after`'s layers, in `after`'s draw order.

        In place, because `colorbar`, `legend` and `add_tool` assign into
        :attr:`~digitalearth.interactive.base.InteractiveMapBase.layers` by index and a caller may hold the
        list: replacing the object would leave those writing into a list nothing composes.

        Which element each layer contributes is the part worth reading. A layer this reconcile **drew** — one
        added, rebuilt or restyled — contributes the element it has just built. A layer it did not touch
        contributes the object the map already holds, rather than an equal element built again: `.opts()`
        writes into HoloViews' option store against the object it is called on, so the colorbar and legend a
        caller switched are on *that* element and nowhere else. A layer neither has — a caller's own element
        the map let go of — contributes nothing, so nothing is overlaid for it.

        Args:
            before: The figure the map drew, which is the order `overlay` stood in: `add_layer` inserts each
                element at its layer's index in the tree, so the tree's order **is** the overlay's order —
                custom layers, which this record never holds, included.
            after: The figure the map now draws, whose layer order is the overlay order.
            held: What this renderer had drawn before the reconcile, by layer id.
            overlay: The elements the map overlaid before it, in that order.
        """
        was = dict(zip(before.layers.ids, overlay))
        arranged = []
        for layer_id in after.layers.ids:
            now = self._drawn.get(layer_id)
            if now is not None and now is not held.get(layer_id):
                arranged.append(now.element)
            elif layer_id in was:
                arranged.append(was[layer_id])
            elif (
                now is not None
            ):  # pragma: no cover - a drawn layer the map never overlaid
                arranged.append(now.element)
        self._map.layers[:] = arranged

    def _would_draw(
        self, before: FigureSpec, after: FigureSpec, change: FigureDiff
    ) -> Tuple[str, ...]:
        """Return the layers a reconcile of this difference asks a drawer for, in the order it asks.

        Read once and handed to both :meth:`_refuse_what_it_cannot_rebuild` and :meth:`_reconcile`, so the
        refusal cannot be asked of a different set of layers than the one that would be drawn.

        Args:
            before: The figure the map currently draws.
            after: The figure it should draw.
            change: The difference between them.

        Returns:
            The ids, rebuilt first, then the restyles that reach HoloViews, then the added layers.
        """
        # A rebuilt layer is one whose *data* changed — its kind, source or slice — and every one of those
        # means a new element. Only a restyle is worth asking about, because `diff` groups a change of
        # `label` with a change of colour and one of those never reaches HoloViews.
        return (
            *change.rebuilt,
            *(
                layer_id
                for layer_id in change.restyled
                if self._reaches_holoviews(before, after, layer_id)
            ),
            *change.added,
        )

    def _refuse_what_it_cannot_rebuild(
        self, after: FigureSpec, asked: Tuple[str, ...]
    ) -> None:
        """Refuse a change that would have to build a layer of a kind this tier only keeps.

        Args:
            after: The figure the map should draw, which holds the kinds being asked for.
            asked: The layers a drawer would be asked for — :meth:`_would_draw`'s answer.

        Raises:
            CapabilityError: for the first such layer, naming it, its kind and why the tier has nothing to
                rebuild it from. A `ValueError`, which is the class the tier refuses every other
                "this backend cannot do that" with.
        """
        for layer_id in asked:
            kind = after.layers.get(layer_id).kind
            reason = KEPT_KINDS.get(kind)
            if reason is None:
                continue
            raise CapabilityError(
                f"the interactive tier keeps {kind!r} layers rather than drawing them, so layer "
                f"{layer_id!r} cannot be built again from a description — {reason}. Re-describe only what "
                f"the figure says about it (its label, its band, whether it is drawn), or remove it and "
                f"hand the new object to add_layer()"
            )

    def _reconcile(
        self, after: FigureSpec, change: FigureDiff, asked: Tuple[str, ...]
    ) -> None:
        """Draw the difference between two figures, layer by layer.

        Removed, rebuilt, restyled and added layers pass through this renderer's record; shown and hidden
        ones do not — they are toggled on the elements themselves, which is the one way this tier's
        `apply` is not record-only (see :meth:`apply`).

        Args:
            after: The figure the map should draw.
            change: The difference between the figure it draws and that one.
            asked: The layers to draw again — :meth:`_would_draw`'s answer, read before the first draw so
                the refusal above and this loop agree on which layers a drawer is asked for.

        Raises:
            KeyError: when a layer names a kind this tier does not draw.

        Warns:
            UserWarning: for a shown or hidden layer whose element has no way to say it — see
                :func:`_show`.
        """
        for layer_id in change.removed:
            self.remove(layer_id)
        for layer_id in asked:
            # `remove` before the draw for an added layer too: nothing is recorded under its id, so the
            # pop is a no-op there and the two cases need no branch between them.
            self.remove(layer_id)
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
        """Draw or stop drawing the element a layer contributes to the overlay.

        The element is changed in place — ``.opts()`` writes into HoloViews' option `Store` against the
        object it is called on and hands that same object back — so what `InteractiveMap.layers` holds, and
        the style filed against it, are the ones this changes.

        **A layer of the caller's own is reached through the map**, because this record never holds one: the
        object is in :attr:`~digitalearth.interactive.base.InteractiveMapBase.layers` and nowhere else, and
        it is the same `.opts()` call that hides it. Before, an id with nothing recorded was ignored, so
        `set_visible` on a custom layer changed the description, reached no engine, and did not emit the
        warning its own contract promises for a layer it leaves drawn — review H8. `is_visible` still
        refuses such an id: it reports on what a *drawer* produced, and no drawer produced this.

        Args:
            layer_id: The layer to toggle. An id this map has no element for at all is ignored, which is how
                the other three tiers answer one too.
            visible: Whether it is drawn.

        Warns:
            UserWarning: when the element has no keyword to say it with, in either direction — see
                :func:`_show`, which this delegates to. A caller's own object reaches it whenever HoloViews
                has no option tree for what was handed in.

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
        element = (
            drawn.element if drawn is not None else self._map._kept_element(layer_id)
        )
        if element is not None:
            _show(element, layer_id, visible)

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
