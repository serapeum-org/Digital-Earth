"""Draw a :class:`~digitalearth.base.spec.FigureSpec` onto one matplotlib axes (#303, the static seam).

The first half of #303 gave this tier a description: every builder ends at ``Scene._describe_layer``, so a
figure can finally say what it draws. It still *drew* the old way — the builder computed the geometry, built
the cleopatra glyph and registered the artist, and the description was written beside it. Two code paths for
one layer is exactly the drift the other three tiers removed: a layer could be described one way and drawn
another, and nothing held them together.

This is the other half. A builder now records what it wants drawn and **this** module draws it, so the layer
is built once, from the description. What follows from that is the same everywhere: a figure round-trips
through ``to_dict``/``from_dict`` and renders the same picture, and a change to one layer is a change to one
layer rather than a rebuild of the scene around it.

**Two things the description deliberately leaves out**, so "round-trips" is read for what it is. ``to_dict``
refuses an ``object:`` source, so only a figure whose builders were given paths or URLs can be written down;
one built from data already in memory is handed to a renderer directly instead. And the caller's own engine
keywords are held on the scene beside the layer (:func:`drawing_opts`) rather than in its description, so a
figure read back elsewhere draws the same layers with the engine's defaults in their place.

**This tier mutates, like the 3-D one.** matplotlib hands out live artists on a live axes, so
:meth:`Renderer.apply` reconciles against them: a removed layer's artists come off the axes, a rebuilt one is
drawn again. That is why :meth:`Renderer.apply` has to roll the *engine* back as well as its own record when
a figure is refused half-way — the web and interactive tiers rebuild their engine object on every render and
can restore a dict; here, an artist already added to the axes stays on it until something takes it off.

**:meth:`Renderer.apply` is record-only in this wave.** It reaches the axes and this module's record of what
is on it; it does not reach the scene. Nothing in ``src/`` calls it — a builder draws through
:meth:`Renderer.draw_layer` — so a map's description is written by its builders and never by a reconcile.
Wave 7 (order 23) routes a map-level change through it, and the scene's own state follows then. Until it
does, read every ``apply`` here as "the artists and the record", never as "what the map reports".

**The drawer table is keyed by kind and then by recipe.** Several builders draw one kind: ``imshow`` and
``block`` are both a field render, and ``choropleth``, ``voronoi``, ``cartogram``, ``quadtree`` and
``grid_cells`` are all a ``choropleth``. The kind vocabulary is shared with every other tier and names *what*
a layer is, so it cannot say which builder made it — each builder records that under ``via``, and that is the
second key (the shape :mod:`digitalearth.interactive.renderer` settled on).
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from matplotlib.artist import Artist

from digitalearth.base.custom import MissingObject, held_object
from digitalearth.base.registry import band_of
from digitalearth.base.spec import FigureSpec, LayerSpec
from digitalearth.static.capabilities import CAPABILITIES

__all__ = [
    "DRAWN_KINDS",
    "DrawnLayer",
    "Renderer",
    "artists_added",
    "drawer_for",
    "drawing_opts",
]

logger = logging.getLogger(__name__)


def drawing_opts(scene: Any, layer: LayerSpec) -> Dict[str, Any]:
    """Return the engine keywords a layer's caller passed, exactly as they passed them.

    They are held on the scene beside the layer, never in its description (see
    :attr:`~digitalearth.static.scene.LayerRecord.opts`). Round-tripping them through the description broke
    them both ways: a dash pattern came back a list matplotlib refuses, and an object with no JSON form — a
    ``Normalize``, a ``FontProperties``, a per-pixel ``alpha`` array — made the figure impossible to save.
    Held as passed, matplotlib gets the caller's own objects.

    Args:
        scene: The scene the layer is drawn on, which holds the keywords.
        layer: The layer being drawn.

    Returns:
        A fresh dict of the caller's keywords, so a drawer may pop and set keys the way a builder used to
        on the caller's own; each value is the very object passed. Empty for a layer the scene holds none
        for — one given none, or one described elsewhere and drawn here, such as a figure read back from
        JSON — which then draws with the engine's defaults.

    Examples:
        - A scene that holds nothing for a layer hands its drawer nothing, and the layer draws with defaults:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.base.spec import LayerSpec
            >>> from digitalearth.static import Scene
            >>> from digitalearth.static.renderer import drawing_opts
            >>> drawing_opts(Scene(), LayerSpec("a", "raster"))
            {}

            ```
    """
    return dict(scene._layer_opts.get(layer.id) or {})


@dataclass(frozen=True)
class DrawnLayer:
    """What one drawer produced: the artists a layer put on the axes, plus what the builder hands back.

    Every drawer answers in this shape so the renderer can record, hide and remove whatever came back
    without knowing which drawer made it — the equivalent of the 3-D tier's ``(mesh, actor)`` pair and of
    the web tier's source-and-layer.

    Attributes:
        artist: What the builder returns to its caller — the mappable of a field render, the ``Text`` of a
            label, the list of polylines a limb-split coastline became. It is whatever that builder has
            always returned, because the public return type is part of this tier's API and the seam must not
            change it.
        glyph: The cleopatra glyph that drew it, or ``None`` for a layer drawn straight onto the axes (a
            tile basemap, a Natural-Earth overlay, a text label). Kept because a categorical fill's swatch
            legend is read off the glyph rather than off the artist.
        artists: Every matplotlib artist the layer owns, which is what :meth:`Renderer.remove` takes off the
            axes and what :meth:`Renderer.set_visible` toggles. Empty for a layer that left nothing
            addressable behind — ``cleopatra.basemap.reference.add_features`` draws onto the axes and hands
            back the axes, so there is no per-layer artist to hold.
    """

    artist: Any = None
    glyph: Any = None
    artists: Tuple[Any, ...] = field(default=())


#: The layer kinds this tier draws **from its description**. Names only, so what is drawable can be asked —
#: by a caller, and by the tier's own capability test — without importing every builder module behind them.
#: :func:`drawer_for` resolves them to functions and is checked against this tuple, which is what keeps the
#: two from drifting: a kind with a drawer and no declaration is a refusal message that lies, and a kind
#: declared with no drawer is a bare `KeyError` where a message belongs.
#:
#: It is the whole of :data:`~digitalearth.static.capabilities.CAPABILITIES`'s ``kinds``: every builder this
#: tier has draws from its description, so there is no half-converted set to keep track of.
DRAWN_KINDS: Tuple[str, ...] = (
    "raster",
    "mesh",
    "contours",
    "filled_contours",
    "rgb",
    "points",
    "choropleth",
    "polygons",
    "vectors",
    "streamlines",
    "unstructured",
    "heatmap",
    "flow",
    "text",
    "graticule",
    "basemap",
    "coastlines",
    "borders",
    "land",
    "ocean",
    "lakes",
    "rivers",
    # The caller's own artist. Declared and drawn like any other kind, as each other tier declares
    # its engine's custom layers — what it cannot do is be *rebuilt*, which is why the object is held.
    "custom:matplotlib",
)


def _recipes() -> Dict[str, Dict[str, Any]]:
    """Return the drawers of this tier, by kind and then by the recipe each layer was built with.

    A kind is drawn more than one way here — a ``choropleth`` is a feature collection, a Voronoi
    tessellation, a cartogram, a quadtree or a raster's own cells — and the kind cannot say which, because
    the kind vocabulary is shared with every other tier and names *what* a layer is. So each builder records
    *how* it drew its layer under ``via``, and that is the second key.

    Returns:
        Kind -> recipe -> ``draw(scene, data, layer) -> DrawnLayer | None``.
    """
    # Imported here rather than at module level: every builder module imports the scene, and the scene
    # builds a renderer, so a module-level import would close a cycle — and a scene that draws nothing
    # should not pay for loading all of them.
    from digitalearth.static.maps import decoration, projection, raster, vector

    return {
        "raster": {"imshow": raster.draw_field},
        "mesh": {"pcolormesh": raster.draw_field},
        "contours": {"contour": raster.draw_field},
        "filled_contours": {"contourf": raster.draw_field},
        "rgb": {
            "rgb_composite": raster.draw_rgb_composite,
            "hsv_composite": raster.draw_hsv_composite,
        },
        "points": {
            "scatter": vector.draw_scatter,
            "grid_points": vector.draw_grid_points,
        },
        # One kind, five builders: what makes a polygon layer a choropleth is that its polygons carry
        # values, not where the polygons came from.
        "choropleth": {
            "grid_cells": vector.draw_grid_cells,
            "choropleth": vector.draw_choropleth,
            "voronoi": vector.draw_voronoi,
            "cartogram": vector.draw_cartogram,
            "quadtree": vector.draw_quadtree,
        },
        # The same three builders again, drawn without values: an outline-only layer is `polygons`.
        "polygons": {
            "shapes": vector.draw_shapes,
            "voronoi": vector.draw_voronoi,
            "cartogram": vector.draw_cartogram,
        },
        # Arrows and wind barbs are one u/v field drawn with two glyphs; a streamplot integrates that field
        # into flow lines, which is a layer of its own.
        "vectors": {
            "quiver": vector.draw_uv_field,
            "barbs": vector.draw_uv_field,
        },
        "streamlines": {"streamplot": vector.draw_uv_field},
        "unstructured": {
            "tricontour": vector.draw_tri,
            "tricontourf": vector.draw_tri,
            "tripcolor": vector.draw_tri,
        },
        "heatmap": {"kde": vector.draw_kde},
        "flow": {"sankey": vector.draw_sankey},
        "text": {
            "text": decoration.draw_text,
            "annotate": decoration.draw_annotate,
        },
        "graticule": {"graticule": projection.draw_graticule},
        # The recipe is cleopatra's own dataset name, which is the singular `coastline` for the plural kind.
        "coastlines": {"coastline": decoration.draw_natural_earth},
        "borders": {"borders": decoration.draw_natural_earth},
        "land": {"land": decoration.draw_natural_earth},
        "ocean": {"ocean": decoration.draw_natural_earth},
        "lakes": {"lakes": decoration.draw_natural_earth},
        "rivers": {"rivers": decoration.draw_natural_earth},
        "basemap": {"basemap": decoration.draw_basemap},
        "custom:matplotlib": {"custom": draw_custom},
    }


def draw_custom(scene: Any, _data: Any, layer: LayerSpec) -> Optional["DrawnLayer"]:
    """Put back an artist the caller built themselves and handed to the scene.

    A custom layer is the one kind a description cannot rebuild: the object is the caller's, and a
    figure holds no spelling for it. The scene keeps it under the layer's id instead, so the layer is
    still addressable — it can be hidden, taken off the axes and drawn again — which is exactly what
    the other three tiers do with theirs.

    Args:
        scene: The scene holding the object.
        _data: The source slot every drawer takes, unread here — a custom layer has no source.
        layer: The layer's description.

    Returns:
        What the layer holds, as a :class:`DrawnLayer` — or ``None`` when the scene does not hold the
        object, which is a layer that was not drawn rather than an error: a figure read back from
        elsewhere carries the description and not the artist.

    Raises:
        MissingObject: when the object is not here and the scene is ``strict``, which is the same
            skip-or-raise answer every other layer gives for data it cannot draw.
    """
    try:
        glyph, artist, label = held_object(
            layer.id,
            layer.kind,
            scene._held_objects,
            engine="matplotlib",
            backend="matplotlib",
        )
    except MissingObject as error:
        if scene.strict:
            raise
        logger.warning("%s; the layer is skipped", error)
        return None
    # Only an artist can go back on an axes, and only one that is not already there: `_add_layer`
    # describes an artist the caller has *just* drawn, and adding it again would file it twice.
    if isinstance(artist, Artist) and artist.axes is not scene.ax:
        scene.ax.add_artist(artist)
    scene._register_artist(glyph, artist, label)
    return DrawnLayer(
        artist=artist,
        glyph=glyph,
        artists=() if artist is None else (artist,),
    )


def _dispatch(kind: str, recipes: Dict[str, Any]) -> Any:
    """Return the drawer for one kind, which picks between the recipes that kind is drawn by.

    Args:
        kind: The kind being drawn, used only to name it in a refusal.
        recipes: Recipe name -> drawer.

    Returns:
        A callable ``draw(scene, data, layer)`` that reads the layer's recorded ``via`` and calls the drawer
        for it.
    """

    def draw(scene: Any, data: Any, layer: LayerSpec) -> Optional[DrawnLayer]:
        """Draw the layer with the drawer its recipe names.

        Args:
            scene: The scene being drawn on.
            data: The layer's opened source, or ``None``.
            layer: The layer's description.

        Returns:
            What the drawer produced, or ``None`` for a layer it declined to draw.

        Raises:
            KeyError: when the layer records no recipe this tier knows — a description built by hand, or
                loaded from a figure written before this tier recorded how it drew its layers.
        """
        via = layer.symbology.props.get("via")
        if via not in recipes:
            raise KeyError(
                f"a {kind!r} layer records {via!r} as how it was drawn; this tier draws one of "
                f"{sorted(recipes)}"
            )
        drawn: Optional[DrawnLayer] = recipes[via](scene, data, layer)
        return drawn

    return draw


def drawer_for(kind: str) -> Any:
    """Return the function that draws one kind of layer in this tier.

    Args:
        kind: A registered layer kind.

    Returns:
        A callable ``draw(scene, data, layer) -> DrawnLayer | None``, where `data` is the layer's opened
        source (or ``None`` for a layer drawn from no source, such as a graticule) and `layer` is its
        `LayerSpec`.

    Raises:
        KeyError: when this tier does not draw `kind`. The message names the kinds it does draw, and — for
            a kind this tier decided against rather than never had — appends the reason the declaration
            gives (:meth:`~digitalearth.base.capabilities.Capabilities.reason`), which is the refusal every
            tier's ``drawer_for`` makes (#294). Also when the drawer table and :data:`DRAWN_KINDS`
            disagree, which is a defect in this module rather than in the caller.

    Examples:
        - Every kind the tier declares has a drawer:
            ```python
            >>> from digitalearth.static.capabilities import CAPABILITIES
            >>> from digitalearth.static.renderer import drawer_for
            >>> sorted(kind for kind in CAPABILITIES.kinds if drawer_for(kind) is None)
            []

            ```
        - A kind from another tier is refused by name:
            ```python
            >>> from digitalearth.static.renderer import drawer_for
            >>> drawer_for("terrain")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            KeyError: "the static tier does not draw 'terrain' layers; it draws [...]"

            ```
    """
    # Before the imports: the point of naming the kinds separately is that what is drawable can be asked
    # without loading every builder behind them.
    if kind not in DRAWN_KINDS:
        # The reason is the tier's own, read from the declaration rather than written again here: a kind
        # this tier decided against says why in `absent`, and a caller who reaches the refusal needs that
        # sentence more than the list of kinds (#294). The other three tiers refuse the same way.
        reason = CAPABILITIES.reason(kind)
        raise KeyError(
            f"the static tier does not draw {kind!r} layers"
            + (f" — {reason}" if reason else "")
            + f"; it draws {sorted(DRAWN_KINDS)}"
        )
    recipes = _recipes()
    if set(recipes) != set(DRAWN_KINDS):
        raise KeyError(
            f"the static tier's drawer table and DRAWN_KINDS disagree: "
            f"{sorted(set(recipes).symmetric_difference(DRAWN_KINDS))}"
        )
    return _dispatch(kind, recipes[kind])


def _set_visible(artist: Any, visible: bool) -> None:
    """Show or hide one matplotlib artist.

    Args:
        artist: The artist to toggle.
        visible: Whether it is drawn.

    Note:
        The ``AttributeError`` is caught rather than asked about with ``hasattr``: a property getter can
        have side effects — PyVista's ``Plotter.camera`` reframes the scene when nothing has set one — and
        the rule that came out of that is to *call* and answer the failure, never to probe.
    """
    try:
        artist.set_visible(bool(visible))
    except AttributeError:  # pragma: no cover - every matplotlib artist has it
        logger.debug("%r cannot be shown or hidden; leaving it as it is", artist)


def _is_visible(artist: Any) -> bool:
    """Whether one matplotlib artist is currently drawn.

    Args:
        artist: The artist to ask.

    Returns:
        Its own flag, or ``True`` for something that has none — one that cannot be hidden is never hidden.

    Note:
        The ``AttributeError`` is caught rather than asked about with ``hasattr``, for the reason
        :func:`_set_visible` gives: a property getter can have side effects, so call and answer the failure.
    """
    try:
        return bool(artist.get_visible())
    except AttributeError:  # pragma: no cover - every matplotlib artist has it
        logger.debug("%r cannot say whether it is drawn; reporting it as drawn", artist)
        return True


def _detach(artist: Any, axes: Any) -> None:
    """Take one artist off the axes, and out of cleopatra's record of what it last rendered.

    Args:
        artist: The artist to remove. One matplotlib has already detached — or one that was never attached,
            such as a mappable a glyph built but did not add — is left alone, so removing a layer twice is
            harmless.
        axes: The axes the layer was drawn on, whose render record is kept in step.
    """
    try:
        artist.remove()
    except (AttributeError, NotImplementedError, ValueError):
        # ValueError is matplotlib's own answer to removing an artist that is not on an axes; the other two
        # are an object that was never one. None of the three is a reason to fail a reconcile.
        logger.debug("%r was not on the axes to remove", artist)
    _forget_render_artist(axes, artist)


def _forget_render_artist(axes: Any, artist: Any) -> None:
    """Drop one artist from cleopatra's per-axes record of what it last rendered.

    cleopatra keeps that record so a second ``plot()`` on the same axes can take the first one's artists
    off before drawing (``_clear_prior_render_artists``). It removes them by calling ``Artist.remove()``
    and swallows ``KeyError``/``NotImplementedError``/``AttributeError`` — but matplotlib answers a
    **second** removal with ``ValueError``, which is not on that list. So an artist this renderer has
    already detached blows up the next render on the same axes, which is exactly what a rebuild, a restyle
    and a rollback all do.

    Args:
        axes: The axes whose record is trimmed. One that has never been rendered onto by cleopatra has
            none, and is left alone.
        artist: The artist to forget.

    Note:
        This reaches for a private attribute of an upstream object, which is a workaround rather than a
        design: the clean fix is for cleopatra to catch ``ValueError`` there, or to expose "forget these
        artists". Reported rather than patched — cleopatra is an upstream package, not ours to change.

        The ``getattr`` is safe to read, unlike the probes the renderer contract forbids: it is a plain
        data attribute cleopatra assigns to the axes, not a property whose getter does work.
    """
    registry = getattr(axes, "_cleo_render_artists", None)
    if not isinstance(registry, dict):
        return
    # Built as a new table rather than edited in place: the old one is being read while it is decided what
    # stays, and a dict cannot lose an entry mid-iteration. cleopatra reads the attribute afresh on every
    # render, so replacing it is the same to it as editing it.
    pruned = {}
    for token, artists in registry.items():
        kept = [held for held in artists if held is not artist]
        if kept:
            pruned[token] = kept
    axes._cleo_render_artists = pruned or None


def _painted(axes: Any) -> Optional[List[Any]]:
    """Return the axes' own list of the artists added to it, in the order they were added.

    matplotlib draws artists sorted by z-order, and artists of one z-order in the order they were added —
    so that order is part of what the axes shows, and a layer put back last is painted over layers it was
    under. The list is the one matplotlib draws from, so reordering it reorders the painting.

    Args:
        axes: The axes being drawn on.

    Returns:
        The live list, or ``None`` when the axes keeps none — in which case a restored layer is left where
        matplotlib put it rather than guessed into place.

    Note:
        ``Axes._children`` is private to matplotlib (it has held every artist an axes owns since 3.5), and
        there is no public way to insert an artist at a position. Read with ``getattr`` rather than probed:
        it is a plain list attribute, not a property whose getter does work.
    """
    children = getattr(axes, "_children", None)
    return children if isinstance(children, list) else None


@contextmanager
def artists_added(axes: Any) -> Iterator[List[Any]]:
    """Collect the artists a block puts on the axes, for a drawer that cannot name them itself.

    ``cleopatra.basemap.tiles.add_tiles`` and ``cleopatra.basemap.reference.add_features`` both draw onto
    the axes and hand the *axes* back, so a drawer calling them has no handle on what it drew. Recording
    the axes as the layer's artist made :meth:`Renderer.set_visible` hide the whole map and
    :meth:`Renderer.remove` detach it from the figure; recording nothing left the layer undrawable-off.
    What they added is the difference between the axes before and after.

    Args:
        axes: The axes being drawn on.

    Yields:
        The list the artists are collected into — empty inside the block, filled on the way out. It stays
        empty when the axes keeps no list of its own, which leaves the layer with no artists to toggle
        rather than with the wrong ones.
    """
    painted = _painted(axes)
    existing = {id(artist) for artist in painted or ()}
    added: List[Any] = []
    yield added
    added.extend(
        artist for artist in _painted(axes) or () if id(artist) not in existing
    )


def _reinsert(
    painted: List[Any], block: List[Any], was: Tuple[Any, ...], order: Tuple[Any, ...]
) -> None:
    """Move the artists a restored layer added back to where that layer's old artists stood, in place.

    Args:
        painted: The axes' live artist list (see :func:`_painted`).
        block: The artists the restored layer just added, which matplotlib appended at the end.
        was: The artists the layer had before it was taken off, which fix where it belongs.
        order: The axes' artists as they stood before the refused change.
    """
    old = {id(artist) for artist in was}
    positions = [index for index, artist in enumerate(order) if id(artist) in old]
    if not block or not positions:
        return
    moving = {id(artist) for artist in block}
    rest = [artist for artist in painted if id(artist) not in moving]
    present = {id(artist) for artist in rest}
    # The first artist that stood after the layer and is still on the axes is what it goes back in front
    # of. When none is left, the layer was the last thing painted, and appended it still is.
    successor = next(
        (artist for artist in order[positions[-1] + 1 :] if id(artist) in present),
        None,
    )
    if successor is None:
        return
    at = next(index for index, artist in enumerate(rest) if artist is successor)
    painted[:] = [*rest[:at], *block, *rest[at:]]


class _PartialDraw:
    """What the axes and the scene held before one drawer ran, so a drawer that fails part-way is undone.

    A drawer adds artists as it goes — a glyph's image, then its colorbar; a limb-split coastline, one
    polyline at a time — and registers its mappable for the colorbar only once the glyph has drawn. When it
    raises after any of that, the layer is never recorded, so nothing would take those artists off again.
    """

    def __init__(self, scene: Any) -> None:
        """Note what is already there.

        Args:
            scene: The scene about to be drawn on.
        """
        self._scene = scene
        self._existing = {id(artist) for artist in _painted(scene.ax) or ()}
        self._registered = len(scene.layers)
        self._composing = scene._drew_on_axes

    def undo(self) -> None:
        """Take off everything the drawer added since, and forget what it registered.

        Only artists that were not there before are touched, so the layers under this one are left alone.
        The compose flag goes back too: a layer that was never drawn is not the first render on the axes.
        """
        axes = self._scene.ax
        added = [
            artist
            for artist in _painted(axes) or ()
            if id(artist) not in self._existing
        ]
        for artist in added:
            _detach(artist, axes)
        del self._scene.layers[self._registered :]
        del self._scene._layer_labels[self._registered :]
        self._scene._drew_on_axes = self._composing


@dataclass(frozen=True)
class _Held:
    """What a refused :meth:`Renderer.apply` has to put back, captured before it starts.

    A rollback that re-draws a removed layer re-appends it everywhere matplotlib and the scene append:
    last in the renderer's record, last in the colorbar registry — so ``colorbar()``'s default ``-1`` keys
    a different layer — and last on the axes, painted over layers it was under. Each of the three is put
    back from here.

    Attributes:
        drawn: Layer id -> what was drawn for it, in draw order.
        registered: The scene's ``(glyph, mappable)`` pairs with their default colorbar labels, in the order
            ``colorbar(layer=...)`` indexes them.
        painted: The axes' artists in the order they were added (see :func:`_painted`).
    """

    drawn: Dict[str, DrawnLayer]
    registered: Tuple[Tuple[Tuple[Any, Any], Optional[str]], ...]
    painted: Tuple[Any, ...]


class Renderer:
    """Draws a figure's layers onto one matplotlib axes, and reconciles one figure with another.

    Attributes:
        drawn: Layer id to the :class:`DrawnLayer` its drawer produced, in draw order.
    """

    def __init__(self, scene: Any) -> None:
        """Bind the renderer to the scene whose axes it draws on.

        Args:
            scene: The :class:`~digitalearth.static.scene.Scene` (or subclass) this renderer serves. Held
                rather than passed per call, because a drawer reads the scene's own helpers — the display
                CRS, the reprojection, the off-limb policy, the artist registry a colorbar is keyed to.
        """
        self._scene = scene
        self._drawn: Dict[str, DrawnLayer] = {}

    @property
    def drawn(self) -> Mapping[str, DrawnLayer]:
        """What has been drawn, by layer id.

        Returns:
            A read-only copy, in the order the layers were drawn. A layer that was skipped — an off-limb
            raster, a label on the far side of a globe — has no entry, which is how a caller tells "drawn"
            from "described".
        """
        return dict(self._drawn)

    def draw_layer(self, figure: FigureSpec, layer_id: str) -> Optional[DrawnLayer]:
        """Draw one of a figure's layers and record what it produced.

        Args:
            figure: The figure holding the layer and its source.
            layer_id: Which layer to draw.

        Returns:
            What the drawer produced, or ``None`` for a layer it declined to draw.

        Raises:
            KeyError: when no layer has that id, or the tier has no drawer for its kind.
            OffLimbError: when the layer's data cannot be placed and the scene is ``strict``.

        Note:
            A drawer that raises — or declines — after it has already put something on the axes leaves
            artists no layer owns: the layer is not recorded, so nothing would ever take them off. Whatever
            it added to the axes and to the colorbar registry is taken back before the error carries on.
        """
        layer = figure.layers.get(layer_id)
        data = self._source_object(figure, layer)
        draw = drawer_for(layer.kind)
        partial = _PartialDraw(self._scene)
        try:
            drawn: Optional[DrawnLayer] = draw(self._scene, data, layer)
        except BaseException:
            partial.undo()
            raise
        if drawn is None:
            partial.undo()
            return None
        self._drawn[layer_id] = drawn
        if not figure.layers.is_visible(layer_id):
            self.set_visible(layer_id, False)
        return drawn

    @staticmethod
    def _source_object(figure: FigureSpec, layer: LayerSpec) -> Any:
        """Return the object a layer draws, or ``None`` when it draws from no source.

        Args:
            figure: The figure holding the sources.
            layer: The layer being drawn.

        Returns:
            The opened source — a pyramids ``Dataset``, a ``FeatureCollection``, the ``(u, v)`` pair a
            vector field is drawn from — or ``None`` for a layer drawn from nothing, such as a graticule.
        """
        if layer.source_id is None:
            return None
        return figure.sources[layer.source_id].open()

    def apply(self, before: FigureSpec, after: FigureSpec) -> None:
        """Bring what the axes shows from one figure to another.

        **Record-only, in this wave.** It reconciles the artists on the axes and this renderer's own
        record of them, and nothing else: the scene's layer tree, its sources and what
        :attr:`~digitalearth.static.scene.Scene.figure_spec` reports are untouched, so a map that has
        applied a figure still describes the one its builders made. Nothing in ``src/`` calls this yet —
        routing a map-level change through it is Wave 7 (order 23), and the description follows then.

        Args:
            before: The figure the axes currently shows.
            after: The figure it should show.

        Raises:
            KeyError: when a layer names a kind this tier does not draw, or a recipe it does not know.
            ValueError: when a layer's description does not carry what its drawer needs.
            OffLimbError: when a layer's data cannot be placed and the scene is ``strict`` — one of the
                drawers' own refusals, re-raised by the rollback rather than swallowed.

        Note:
            ``apply`` is **not atomic**: it draws layer by layer, so a refusal on the third layer has
            already drawn the first two. Rolling back only this record would leave artists on the axes that
            no layer owns, and rolling back only the caller's description would leave the record holding
            layers no figure owns — the defect the shared conformance contract states, and which every
            earlier tier hit. Both go back together, which for this tier means taking the new artists off
            the axes and drawing the old ones again.
        """
        held = _Held(
            drawn=dict(self._drawn),
            registered=tuple(zip(self._scene.layers, self._scene._layer_labels)),
            painted=tuple(_painted(self._scene.ax) or ()),
        )
        try:
            self._reconcile(before, after)
        except BaseException:
            self._rollback(held, before)
            raise
        # A removed layer's data is deliberately *not* forgotten here: `before` — and any figure captured
        # before the change — still names it, and forgetting it made such a figure dangle and made undoing
        # the change (`apply(after, before)`) fail. `close()` is where a map lets its data go, which is the
        # caller saying they are finished with every figure it produced; the web tier settled on the same.

    def _reconcile(self, before: FigureSpec, after: FigureSpec) -> None:
        """Draw the difference between two figures, layer by layer.

        Args:
            before: The figure the axes currently shows.
            after: The figure it should show.

        Raises:
            KeyError: when a layer names a kind this tier does not draw, or a recipe it does not know.
            ValueError: when a layer's description does not carry what its drawer needs.
        """
        change = before.diff(after)
        for layer_id in change.removed:
            self.remove(layer_id)
        # A rebuilt layer is one whose *data* changed — its kind, source, slice, or the reference behind its
        # source id — and every one of those means new artists. Only a restyle is worth asking about,
        # because `diff` groups a change of `label` with a change of colormap and one of those never reaches
        # matplotlib.
        for layer_id in change.rebuilt:
            self.remove(layer_id)
            self.draw_layer(after, layer_id)
        for layer_id in change.restyled:
            if not self._reaches_matplotlib(before, after, layer_id):
                continue
            # A cleopatra glyph bakes its colormap and its class breaks into the artist it built, so a
            # restyle is the layer drawn again from its description rather than a property set on it.
            self.remove(layer_id)
            self.draw_layer(after, layer_id)
        for layer_id in change.added:
            self.draw_layer(after, layer_id)
        for layer_id in change.shown:
            self.set_visible(layer_id, True)
        for layer_id in change.hidden:
            self.set_visible(layer_id, False)

    def _rollback(self, held: _Held, before: FigureSpec) -> None:
        """Put the axes and this record back the way a refused :meth:`apply` found them.

        Args:
            held: What was drawn, registered and painted before the refused change.
            before: The figure those layers were drawn from, which is what a restored layer is drawn from
                again.

        Note:
            Restoring is a re-draw rather than a re-attach: matplotlib's ``Artist.remove`` unlinks an artist
            from its axes and from the transform stack, and adding the same object back is not supported.
            Drawing it again from `before` is, and it is the same call that drew it the first time. A
            re-draw appends, so the restored layer is then moved back to the place it held — in this
            record, in the colorbar registry and in the order the axes paints.
        """
        # Decided first, removed after: `remove` takes each layer out of `_drawn`, which cannot shrink while
        # it is being read.
        partial = [
            layer_id
            for layer_id, drawn in self._drawn.items()
            if drawn is not held.drawn.get(layer_id)
        ]
        for layer_id in partial:
            self.remove(layer_id)
        for layer_id in [key for key in held.drawn if key not in self._drawn]:
            self._restore(layer_id, before, held)
        self._drawn = {
            layer_id: self._drawn[layer_id]
            for layer_id in held.drawn
            if layer_id in self._drawn
        }
        self._restore_registry(held)

    def _restore(self, layer_id: str, before: FigureSpec, held: _Held) -> None:
        """Draw one removed layer again and move what it drew back to where the layer stood on the axes.

        Args:
            layer_id: The layer to restore.
            before: The figure it is drawn from.
            held: What the axes held before the refused change.
        """
        painted = _painted(self._scene.ax)
        existing = {id(artist) for artist in painted or ()}
        try:
            self.draw_layer(before, layer_id)
        except (
            Exception
        ) as error:  # pragma: no cover - a drawer that drew once draws again
            # Never mask the refusal that started the rollback: a layer that cannot be restored is reported
            # and the original exception carries on out of `apply`.
            logger.warning(
                "rolling back a refused change could not re-draw layer %r: %s",
                layer_id,
                error,
            )
            return
        if painted is None:
            return
        # Everything the re-draw added, not only the artists the drawer reported: a glyph can leave more on
        # the axes than its mappable, and all of it belongs where the layer was.
        added = [artist for artist in painted if id(artist) not in existing]
        _reinsert(painted, added, held.drawn[layer_id].artists, held.painted)

    def _restore_registry(self, held: _Held) -> None:
        """Put the scene's colorbar registry back in the order it had, with restored layers in place.

        ``Scene.layers`` is what ``colorbar(layer=-1)`` indexes, so a restored layer re-registered last would
        silently re-key the default colorbar. The registry is rebuilt from what it held, each restored
        layer's old pair swapped for the one its re-draw registered, and a layer that could not be restored
        dropped.

        Args:
            held: What the scene had registered before the refused change.
        """
        swapped: Dict[Tuple[int, int], Optional[Tuple[Any, Any]]] = {}
        for layer_id, was in held.drawn.items():
            now = self._drawn.get(layer_id)
            if now is was or was.artist is None:
                continue
            swapped[(id(was.glyph), id(was.artist))] = (
                None if now is None else (now.glyph, now.artist)
            )
        pairs: List[Tuple[Any, Any]] = []
        labels: List[Optional[str]] = []
        for (glyph, mappable), label in held.registered:
            key = (id(glyph), id(mappable))
            pair = swapped[key] if key in swapped else (glyph, mappable)
            if pair is None:
                continue
            pairs.append(pair)
            labels.append(label)
        self._scene.layers = pairs
        self._scene._layer_labels = labels

    @staticmethod
    def _reaches_matplotlib(
        before: FigureSpec, after: FigureSpec, layer_id: str
    ) -> bool:
        """Whether a restyle changes anything matplotlib draws.

        ``diff`` groups a change to ``label`` — what a legend calls the layer — with a change to its
        colormap, because both are "the description changed". Only one of them reaches the engine, and
        answering both with a redraw meant renaming a layer re-opened its source, re-ran the reprojection
        and built the whole glyph again.

        Args:
            before: The figure drawn now.
            after: The figure to draw.
            layer_id: The layer whose restyle is in question.

        Returns:
            ``True`` when the layer's symbology, filter or group differs — the parts a drawer reads — and
            for a layer either figure does not hold, which is a question about the layer rather than about
            its style. ``False`` for a change to ``label`` alone.
        """
        try:
            was = before.layers.get(layer_id)
            now = after.layers.get(layer_id)
        except KeyError:
            # `restyled` only ever names ids both figures hold, so neither lookup raises on the path that
            # calls this. Guarding both rather than the first is what makes that true of a direct call too,
            # and the answer for a layer that is not in both is to draw it.
            return True
        return (was.symbology, was.filter, was.group) != (
            now.symbology,
            now.filter,
            now.group,
        )

    def remove(self, layer_id: str) -> None:
        """Take a layer off the axes and forget what was drawn for it.

        Only the *drawing* goes. The in-memory data the layer registered, and the key its drawer reads, stay
        with the scene: a figure captured before the removal still names them, and drawing that figure again
        — an undo, a rollback, a rebuild — needs them. ``Scene.close()`` is where they are let go.

        Args:
            layer_id: The layer to remove. One that was never drawn — skipped, or described but not yet
                rendered — is ignored, so removing a layer twice is harmless.
        """
        drawn = self._drawn.pop(layer_id, None)
        if drawn is None:
            return
        self._scene._unregister_artist(drawn)
        for artist in drawn.artists:
            _detach(artist, self._scene.ax)

    def set_visible(self, layer_id: str, visible: bool) -> None:
        """Show or hide every artist drawn for a layer.

        Args:
            layer_id: The layer to toggle. An id nothing was drawn for is ignored.
            visible: Whether it is drawn.
        """
        drawn = self._drawn.get(layer_id)
        if drawn is None:
            return
        for artist in drawn.artists:
            _set_visible(artist, visible)

    def is_visible(self, layer_id: str) -> bool:
        """Whether the axes is currently drawing what this renderer holds for a layer.

        The read-back of :meth:`set_visible`. Every tier's renderer answers this, in its own terms, so the
        question "is this layer drawn hidden?" can be asked of any of them — which is what the shared
        renderer conformance suite does, and what no tier could be asked before (review M4).

        Args:
            layer_id: The layer to ask about.

        Returns:
            ``True`` when every artist the layer owns is on. A layer that left no addressable artist
            behind — a graticule, a Natural-Earth overlay — answers ``True``: there is nothing that could
            have been switched off.

        Raises:
            KeyError: when nothing was drawn for `layer_id`, naming it. A layer the axes does not hold has
                no visibility to report, and :attr:`drawn` is what says which those are.
        """
        drawn = self._drawn.get(layer_id)
        if drawn is None:
            raise KeyError(
                f"nothing is drawn for layer {layer_id!r}, so it has no visibility to report; the static "
                f"tier holds {sorted(self._drawn)}"
            )
        return all(_is_visible(artist) for artist in drawn.artists)

    def band_for(self, layer: LayerSpec) -> str:
        """Return the draw-order band a layer belongs to.

        Args:
            layer: The layer being placed.

        Returns:
            The layer's own band when it declares one — what a backdrop raster and a caller's own artist
            need, since ``custom:matplotlib`` names the engine and says nothing about what it draws — else
            its kind's band.
        """
        return layer.band or band_of(layer.kind)
