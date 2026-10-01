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
one built from data already in memory is handed to a renderer directly instead. And of the caller's own
engine keywords only the half a figure carries is described — a string, a boolean, a finite number or
``None``, and a ``list`` or ``dict`` built out of those, which is what
:func:`~digitalearth.base.spec._serial.travels_in_a_figure` accepts. A tuple (which JSON reads back as a
list), an array or an engine object stays on the scene beside the layer (:func:`drawing_opts`) and nowhere
else, so a figure read back elsewhere draws *those* with the engine's defaults in their place. Measured:
``described_opts`` describes ``levels=[0.0, 0.5, 1.0]`` and ``ticks={'a': 1}`` and holds
``figsize=(8, 6)``.

**This tier mutates, like the 3-D one.** matplotlib hands out live artists on a live axes, so
:meth:`Renderer.apply` reconciles against them: a removed layer's artists come off the axes, a rebuilt one is
drawn again. That is why :meth:`Renderer.apply` has to roll the *engine* back as well as its own record when
a figure is refused half-way — the web and interactive tiers rebuild their engine object on every render and
can restore a dict; here, an artist already added to the axes stays on it until something takes it off.

**:meth:`Renderer.apply` has a caller.** It reaches the axes and this module's record of what is on it,
and it reached nothing else for a wave: no code in ``src/`` called it, so a map's description was written by
its builders and never by a reconcile. :meth:`~digitalearth.static.scene.Scene._change` is that caller now —
the path ``add_layer``, ``get_layer``, ``remove_layer``, ``set_visible``, ``move_layer`` and
``replace_layer`` take — and it installs the scene's layer tree and its sources once this has returned, so a
figure this refuses is never one the map describes. ``apply`` itself still writes only the artists, this
record and the scene's colorbar registry (a drawer registers its mappable there, and a rollback has to put
that back): read it as "the picture", and the description as what ``_change`` moves with it.

**Draw order reaches the picture.** ``FigureDiff`` reports ``order`` and no renderer acted on it, so a
reorder changed every tier's description and none of their figures. :meth:`Renderer._repaint` is this tier's
half: matplotlib paints the artists of one z-order in the order they were added, so re-arranging that list
re-arranges the picture.

**The drawer table is keyed by kind and then by recipe.** Several builders draw one kind: ``imshow`` and
``block`` are both a field render, and ``choropleth``, ``voronoi``, ``cartogram``, ``quadtree`` and
``grid_cells`` are all a ``choropleth``. The kind vocabulary is shared with every other tier and names *what*
a layer is, so it cannot say which builder made it — each builder records that under ``via``, and that is the
second key (the shape :mod:`digitalearth.interactive.renderer` settled on).
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from dataclasses import replace as with_fields
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, cast

from matplotlib.artist import Artist

from digitalearth.base.custom import MissingObject, held_object
from digitalearth.base.registry import band_of
from digitalearth.base.spec import FigureSpec, LayerSpec, retarget_recipes
from digitalearth.static.capabilities import CAPABILITIES
from digitalearth.static.guides import paint_guide, plan_guide

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

    This is the *held* half of the pair: every keyword the scene was given, as the very object it was given
    (see :attr:`~digitalearth.static.scene.LayerRecord.opts`), so matplotlib gets the caller's own objects
    and never a frozen copy. The layer's description carries the plain ones as well
    (:func:`~digitalearth.base.spec._serial.travels_in_a_figure`), and a drawer asks
    :func:`~digitalearth.static.scene.drawing_style` for the two composed rather than calling this directly.
    Round-tripping the rest broke them both ways: a dash pattern came back a list matplotlib refuses, and an
    object with no JSON form — a ``Normalize``, a ``FontProperties``, a per-pixel ``alpha`` array — made the
    figure impossible to save.

    Args:
        scene: The scene the layer is drawn on, which holds the keywords.
        layer: The layer being drawn.

    Returns:
        A fresh dict of the caller's keywords, so a drawer may pop and set keys the way a builder used to
        on the caller's own; each value is the very object passed. Empty for a layer the scene holds none
        for — one given none, or one described elsewhere and drawn here, such as a figure read back from
        JSON — which then draws from the plain keywords its description carries, and with the engine's
        defaults wherever it carries none.

    Examples:
        - A scene that holds nothing for a layer hands its drawer nothing of its own:
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
            axes and what :meth:`Renderer.set_visible` toggles. A drawer that hands its caller something
            else back still fills this: ``cleopatra.basemap.reference.add_features`` draws onto the axes and
            returns the axes, so :func:`artists_added` watches the axes and collects whatever appeared while
            it ran. Empty only for a layer with nothing on the axes yet — a graticule, whose lines are
            computed when it is drawn and put on the axes afterwards, by the globe frame.
        color_field: The data field this layer's colour varies with, when only its drawer can say — the band
            variable a raster render resolved through
            :func:`~digitalearth.base.autostyle.auto_style`, which is not in the description because the
            builder never opened the source. ``None`` for every other layer, whose colour field is the
            ``column`` its own description already names, and for one that colours by no field at all.
        guides: The colour key drawn for this layer — one ``Colorbar`` or one ``Legend``, or nothing. Held
            **beside** `artists` rather than in it, deliberately: `artists` is the layer's drawing, which
            :meth:`Renderer._repaint` arranges in draw order and :func:`_rank_zorders` gives z-orders to,
            and a colorbar lives on its own axes where neither applies. What the two share is the layer's
            *lifetime* — :meth:`Renderer.remove` takes the key off with the drawing and
            :meth:`Renderer.set_visible` hides it with it (#261, order 24).

    Examples:
        - A text label owns the one ``Text`` it drew, and was drawn straight onto the axes rather than
          through a glyph:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> m = Map()
            >>> _ = m.text(4.9, 52.4, "Amsterdam")
            >>> drawn = m._renderer.drawn["text-1"]
            >>> len(drawn.artists), drawn.glyph
            (1, None)
            >>> drawn.artist.get_text()
            'Amsterdam'
            >>> m.close()

            ```
    """

    artist: Any = None
    glyph: Any = None
    artists: Tuple[Any, ...] = field(default=())
    color_field: Optional[str] = None
    guides: Tuple[Any, ...] = field(default=())

    def colored_by(self, color_field: Optional[str]) -> "DrawnLayer":
        """Return this drawing again, carrying the data field its colour varies with.

        The one thing a drawer knows that its builder does not, and therefore the one field every drawer
        fills in on its way out (see :attr:`color_field`). It is a method rather than a bare
        ``dataclasses.replace`` at each of those return statements because that is the whole of what they
        do with it, and because ``replace`` is typed as "some dataclass" rather than as this one — a
        drawer declaring ``-> DrawnLayer`` then hands back a value nothing can check against the
        declaration. The ``cast`` is where that narrowing happens, once, instead of at eight call sites.

        Inside, it **is** ``replace``. It was a field-by-field copy, which was correct on the day it was
        written and would have gone on being correct only until `DrawnLayer` grew a sixth field: a copy
        that names the fields drops the one nobody remembered to add to it, silently, while ``replace``
        cannot forget one (review L2). Proved by adding a sixth field and running
        ``test_colored_by_carries_every_other_field_of_the_drawing`` against both.

        Args:
            color_field: The field name, or ``None`` for a layer that colours by no field at all.

        Returns:
            A new ``DrawnLayer`` with everything else as it was; this one is frozen and unchanged.

        Examples:
            - The drawing is untouched apart from the field name:
                ```python
                >>> from digitalearth.static.renderer import DrawnLayer
                >>> drawn = DrawnLayer(artist="im", glyph="g", artists=("im",), guides=("bar",))
                >>> drawn.colored_by("elev")
                DrawnLayer(artist='im', glyph='g', artists=('im',), color_field='elev', guides=('bar',))
                >>> drawn.color_field is None
                True

                ```
        """
        return cast("DrawnLayer", with_fields(self, color_field=color_field))


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
    "labels",
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
        # One string at one coordinate is `text`; one string per feature, read from a column, is its own kind
        # — and its own drawer, in the vector module, because the coordinates come from the geometry.
        "labels": {"labels": vector.draw_labels},
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
        MissingObject: when the object is not here and the scene is ``strict`` — the same skip-or-raise
            answer every other layer gives for data it cannot draw, and since #325 the same *type* as
            well. `MissingObject` derives from :class:`~digitalearth.base.crs.OffLimbError`, so the one
            ``except OffLimbError`` that catches a strict off-limb raster on this tier catches this too.
            It did not before: this was a bare `LookupError` while the 3-D and web tiers raised
            `OffLimbError` for the same case, so a caller who handled one was not handling the other.
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


#: The plain recipe of each kind this tier draws more than one way — the one a bare
#: ``field``/``points``/``choropleth`` records. Used only to retarget a ``via`` this tier does not know, when
#: a figure another tier described is drawn here (:func:`retarget_via`): a kind drawn one way needs no entry
#: (its sole recipe is unambiguous), and a kind whose plain recipe is a genuine judgement (none here) is left
#: out so its foreign ``via`` is refused rather than guessed. Held against :func:`_recipes` by a test.
_CANONICAL_VIA: Dict[str, str] = {
    "rgb": "rgb_composite",
    "points": "scatter",
    "choropleth": "choropleth",
    "polygons": "shapes",
    "vectors": "quiver",
    "unstructured": "tricontour",
    "text": "text",
}


def retarget_via(figure: FigureSpec) -> FigureSpec:
    """Rewrite each layer's recorded ``via`` to this tier's recipe for the same kind.

    A ``via`` names *how* a layer was drawn, and the name is the drawing tier's own: a ``choropleth`` this tier
    records as ``"choropleth"`` the interactive tier records as ``"geometry"``. This translates a foreign
    ``via`` (or none) to the recipe this tier draws the kind with, so the dispatch step
    (:func:`_dispatch`) finds a recipe instead of refusing an unknown one.

    This is the shared cross-tier mechanism. For a same-tier figure every ``via`` is already known, so this
    returns the figure unchanged; a figure another tier described has its foreign recipe retargeted here, and
    the field drawer fills the style a portable figure does not carry (``cmap``/``default_cmap``/``levels``)
    from the variable's own style or the tier default — so a figure the web or interactive tier described
    draws here too (the cross-tier contract :func:`digitalearth.api.to_backend` states, across the three 2-D
    tiers).

    A kind with one recipe retargets to it unambiguously; a kind drawn several ways uses :data:`_CANONICAL_VIA`,
    and one absent from that table keeps its foreign ``via`` and is refused rather than drawn as a guess.

    Args:
        figure: The figure to draw, as another tier (or this one) described it.

    Returns:
        The figure with foreign recipes retargeted; the same object when nothing needed changing.
    """
    return retarget_recipes(figure, _recipes(), _CANONICAL_VIA)


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


def _guide_artist(guide: Any) -> Any:
    """Return the artist that carries a colour key's visibility flag.

    Args:
        guide: A drawn key — a ``matplotlib.colorbar.Colorbar`` or a ``matplotlib.legend.Legend``.

    Returns:
        The key itself for a ``Legend``, which is an artist; the **axes** for a ``Colorbar``, which is not
        one and has no ``set_visible`` of its own — hiding a bar means hiding the axes it was drawn on.
        Asked by attribute rather than by type so a key this tier does not yet draw still answers.
    """
    return getattr(guide, "ax", guide)


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


def _detach_guide(guide: Any, axes: Any) -> None:
    """Take one colour key off the axes — unless it is a legend the axes has already replaced.

    Args:
        guide: The drawn key to remove, a ``Colorbar`` or a ``Legend``.
        axes: The axes it was drawn on.

    Note:
        **A displaced legend is forgotten, never removed.** An axes holds one legend, in a slot of its own,
        and matplotlib gives every legend it makes for that axes the same removal hook —
        ``Axes._remove_legend``, which sets ``legend_`` to ``None`` whichever legend is sitting there. So
        calling ``remove()`` on a legend the axes has since replaced takes the **surviving** layer's key off
        the figure, and does it silently: a second removal of an artist normally answers ``ValueError``, and
        this one does not (review H1).

        :meth:`Renderer._displaced` keeps the *record* clear of those orphans, which is the other half; this
        is the half that holds wherever a record slipped through, because the renderer is not the only
        writer of that slot — a categorical fill's cleopatra glyph writes it on its way in. Stated here, at
        the one call that does the damage, rather than trusted to every caller having swept first.

        A ``Colorbar`` is removed normally: it lives on an axes of its own, any number coexist, and
        :func:`_guide_artist` is what tells the two kinds apart without asking about types.
    """
    if _guide_artist(guide) is guide and axes.get_legend() is not guide:
        logger.debug(
            "%r is not the legend the axes holds; forgetting it instead", guide
        )
        _forget_render_artist(axes, guide)
        return
    _detach(guide, axes)


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
    under. The list is the one matplotlib draws from, so reordering it reorders the painting **within one
    z-order**; reaching across two takes :func:`_rank_zorders` as well.

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


def _zorder_of(artist: Any) -> float:
    """Return one matplotlib artist's z-order.

    Args:
        artist: The artist to ask.

    Returns:
        Its z-order, or ``0.0`` for something that carries none — matplotlib's own default.

    Note:
        The ``AttributeError`` is caught rather than asked about with ``hasattr``, for the reason
        :func:`_set_visible` gives: a property getter can have side effects, so call and answer the failure.
    """
    try:
        return float(artist.get_zorder())
    except AttributeError:  # pragma: no cover - every matplotlib artist has one
        return 0.0


def _band_of_layer(layers: Any, layer_id: str) -> str:
    """Return the band a layer is drawn in, as the key the z-order ranking groups by.

    Args:
        layers: The figure's `LayerTree`, or `None` when the caller has none to offer.
        layer_id: The layer to place.

    Returns:
        The layer's own `band` when it declares one, else the band its kind declares. `""` when there is no
        tree to ask, which puts every layer in one band — the answer that changes nothing, for the rollback
        path that is restoring an arrangement rather than making a move.
    """
    if layers is None:
        return ""
    layer = layers.get(layer_id)
    if layer is None:
        return ""
    return layer.band or band_of(layer.kind)


def _rank_zorders(bands: List[List[List[Any]]]) -> None:
    """Give each layer the z-order its place in its **own band** earns, within that band's own span.

    matplotlib draws an axes' artists ``sorted(children, key=attrgetter("zorder"))`` — a **stable** sort, so
    rearranging the list reaches the picture only between artists that already share a z-order. Carrying a move
    across two of them means writing the z-orders, and the question review H1 answered the hard way is *whose*.

    **Per band, never across them.** `LayerTree.move` already confines a move to one band — the docstring on
    :meth:`~digitalearth.static.Scene.move_layer` says so, and an index outside the band is an `IndexError` —
    so a move cannot change any other band's order and must not change any other band's values. The first
    attempt at this dealt the layers' values back out across *every* layer at once, which is precisely how a
    move between two data rasters handed a text label's ``3`` to a coastline at ``2.5`` and painted one over
    the other: two layers in bands the caller never touched, swapped.

    Within one band each layer's value comes from its own index there, spread across the span that band's
    layers already occupied. Two properties follow:

    - **A band the move did not reach is not written at all**, so the package's own stacking survives — a
      basemap underlay at ``-3``, coastlines and borders at ``2.5``, rivers at ``2.4``.
    - **The picture follows the description inside the band**, which is what a layer switcher is for: a point
      layer at matplotlib's ``1`` really does go under a raster at its ``0``.

    Where a band's layers all sit on one z-order — two rasters, two point layers — nothing is written: the
    list order already decides. A layer whose own artists carried different z-orders keeps its internal
    stacking through the list, which is sorted by those values and which matplotlib's stable sort preserves.

    Where a band's described order disagrees with its values, a ``zorder=`` a caller gave one of *that band's*
    layers is superseded: both it and the move say where the layer paints, and the move is the later word.

    Args:
        bands: One list per band, each holding that band's layers' artists in the order the figure now
            describes them, bottom first; each layer's own artists already in paint order.
    """
    for blocks in bands:
        held = [_zorder_of(artist) for block in blocks for artist in block]
        if not held:
            continue
        low, high = min(held), max(held)
        last = len(blocks) - 1
        if last < 1 or low == high:
            # One layer in the band, or all of them on one z-order: the arrangement already decides, and
            # writing a value would move this band's layers against artists that are not layers for no gain.
            continue
        span = high - low
        for index, block in enumerate(blocks):
            zorder = low + span * index / last
            for artist in block:
                try:
                    artist.set_zorder(zorder)
                except (
                    AttributeError
                ):  # pragma: no cover - every matplotlib artist has one
                    logger.debug(
                        "%r cannot carry a z-order; leaving it where matplotlib paints it",
                        artist,
                    )


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
        The list the artists are collected into — empty inside the block, filled on the way out **even
        when the block raises**, which is the case a bare ``yield`` skipped: a drawer that fails part-way
        has still put artists on the axes, and this list is the only record naming them. It stays empty
        when the axes keeps no list of its own, which leaves the layer with no artists to toggle rather
        than with the wrong ones.
    """
    painted = _painted(axes)
    existing = {id(artist) for artist in painted or ()}
    added: List[Any] = []
    try:
        yield added
    finally:
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
        #: The axes' legend before the draw. An axes holds it in its own slot rather than among the artists
        #: :func:`_painted` lists, so a drawer that makes one — a categorical fill's swatch legend is made
        #: by the cleopatra glyph itself — puts something on the figure that watching that list cannot see.
        self._legend = self._scene.ax.get_legend()

    def undo(self) -> None:
        """Take off everything the drawer added since, and forget what it registered.

        Only artists that were not there before are touched, so the layers under this one are left alone.
        The compose flag goes back too: a layer that was never drawn is not the first render on the axes.

        The axes' single legend slot goes back as well, and by assignment rather than by ``remove()``:
        matplotlib's own removal hook for a legend clears whatever is in that slot, so removing the one this
        draw made would take the previous one with it (see :meth:`Renderer._displaced`).
        """
        axes = self._scene.ax
        added = [
            artist
            for artist in _painted(axes) or ()
            if id(artist) not in self._existing
        ]
        for artist in added:
            _detach(artist, axes)
        if axes.get_legend() is not self._legend:
            axes.legend_ = self._legend
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
            they were drawn — which is the order :meth:`~digitalearth.static.scene.Scene._registered_index`
            matches a layer against to find the units its key is titled with.
        painted: The axes' artists in the order they were added (see :func:`_painted`).
    """

    drawn: Dict[str, DrawnLayer]
    registered: Tuple[Tuple[Tuple[Any, Any], Optional[str]], ...]
    painted: Tuple[Any, ...]


class Renderer:
    """Draws a figure's layers onto one matplotlib axes, and reconciles one figure with another.

    Attributes:
        drawn: Layer id to the :class:`DrawnLayer` its drawer produced, in draw order.

    Examples:
        - A map's renderer holds what it drew, keyed by the same ids
          :attr:`~digitalearth.static.scene.Scene.layer_ids` lists, in the order the layers were drawn:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> from digitalearth.static import Map
            >>> m = Map()
            >>> _ = m.text(4.9, 52.4, "Amsterdam")
            >>> _ = m.text(2.35, 48.86, "Paris")
            >>> list(m._renderer.drawn)
            ['text-1', 'text-2']
            >>> m.layer_ids
            ['text-1', 'text-2']
            >>> m.close()

            ```
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
        #: The visibility asked of a layer that owns **no** artist to carry it, by layer id. A graticule is
        #: the one that gets there (see :meth:`is_visible`); every other layer's flag lives on its artists,
        #: where matplotlib reads it, and is not duplicated here.
        self._asked: Dict[str, bool] = {}

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
            ValueError: from :meth:`draw_guide`, for a key the layer's description asks for and this tier
                cannot draw — a swatch list over a layer publishing no scale, row labels that do not number
                the rows, a bar over a categorical scale.

        Note:
            A drawer that raises — or declines — after it has already put something on the axes leaves
            artists no layer owns: the layer is not recorded, so nothing would ever take them off. Whatever
            it added to the axes and to the colorbar registry is taken back before the error carries on.

            **The key's draw is inside that guard too.** It was added after it, and after the record was
            written, so a refusal from the key left the drawer's artists on the axes *and* the layer in
            :attr:`drawn` while the error carried on — the one shape this note promises cannot happen
            (review M5). A refusal now puts the record back as it was, empty for a layer that was not drawn
            before, and takes the new artists off with it.
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
        # Before the record, so a layer drawn again does not inherit what was asked of the one that held
        # this id before it: the ask is only consulted for a layer with no artist, where nothing else could
        # correct it.
        previous = self._drawn.get(layer_id)
        asked = self._asked.pop(layer_id, None)
        # The new drawing carries the previous one's key across, so `draw_guide`'s first act — taking the
        # layer's previous key off — has something to take off. Overwriting with a bare `guides=()` left
        # that loop iterating an empty tuple, so a layer drawn twice orphaned its own bar on the figure
        # instead of replacing it (review M5).
        self._drawn[layer_id] = (
            drawn if previous is None else with_fields(drawn, guides=previous.guides)
        )
        try:
            # The colour key comes back with the layer. A guide is recorded on the layer's own encoding, so
            # a rebuild, a restyle, a rollback and a figure read back onto another scene each draw the key
            # the description asks for — there is no second writer of it, which is what keeps one recorded
            # guide from becoming two bars. Inside the guard, because it refuses a description this tier
            # cannot draw and a refusal must not leave the layer half-drawn.
            self.draw_guide(layer)
        except BaseException:
            if previous is None:
                self._drawn.pop(layer_id, None)
            else:
                self._drawn[layer_id] = previous
            if asked is not None:
                self._asked[layer_id] = asked
            partial.undo()
            raise
        # The records agree with the axes' single legend slot now, whoever wrote it during this draw — and
        # a **drawer** writes it too: a categorical fill's cleopatra glyph draws its own swatch legend on
        # its way in. `draw_guide` only reconciles when it made the new legend itself, so on a layer nobody
        # keyed the hook never fired and the previously keyed layer went on holding a legend matplotlib had
        # already dropped (review H1). Here rather than before `draw_guide`, so a refused key leaves every
        # record exactly as `partial.undo()` leaves the axes: the sweep only ever drops, and nothing could
        # put a swept record back.
        self._displaced(self._scene.ax.get_legend())
        if not figure.layers.is_visible(layer_id):
            self.set_visible(layer_id, False)
        return drawn

    def draw_guide(self, layer: LayerSpec, **kwargs: Any) -> Optional[Any]:
        """Draw the colour key one layer's guide asks for, replacing whatever it already had.

        The one call that puts a key on the figure, whoever asked: :meth:`draw_layer` for a layer whose
        description already carries a guide, and :meth:`~digitalearth.static.scene.Scene.colorbar` /
        :meth:`~digitalearth.static.scene.Scene.legend` for a caller who has just recorded one. It takes the
        layer's previous key off, which is what makes a second call **replace** rather than stack a second
        bar in the same figure — the same answer the web tier gives for its one legend panel.

        **The new key is derived before the old one comes off.** That removal used to be the first act, so a
        refusal from the derivation left the layer describing a key that was no longer on the figure and
        nothing could put back: a ``Colorbar``'s axes is *removed*, which :meth:`_PartialDraw.undo` does not
        watch — it restores the axes' single legend slot by assignment, and a bar does not use that slot —
        and a removed axes cannot be re-added (review M4). :func:`~digitalearth.static.guides.plan_guide`
        answers every refusal a description can earn, so nothing between the removal and the draw can fail
        except the engine itself.

        Its last act is to give the new key the layer's **current visibility**, so a key asked for on a
        hidden layer is drawn hidden. :meth:`set_visible` already carried the key with the layer in the
        other direction; without this, hiding-then-keying and keying-then-hiding left two different figures
        and the first of them showed a bar explaining a picture the reader cannot see.

        Args:
            layer: The layer's description, which carries the guide (see
                :func:`~digitalearth.static.guides.plan_guide`).
            **kwargs: Styling forwarded to the matplotlib colorbar or legend.

        Returns:
            The artist drawn, or ``None`` when the guide asks for nothing — switched off, absent, or on a
            layer this renderer has drawn nothing for.

        Raises:
            ValueError: from :func:`~digitalearth.static.guides.plan_guide`, for a key the layer's scale
                cannot be drawn as — raised before the layer's previous key is taken off, so a refusal
                leaves the figure exactly as it was.
        """
        drawn = self._drawn.get(layer.id)
        if drawn is None:
            return None
        plan = plan_guide(layer, drawn)
        for held in drawn.guides:
            _detach_guide(held, self._scene.ax)
        made = None if plan is None else paint_guide(self._scene, plan, **kwargs)
        # The layer records what has to come off with it, which is nothing when no key was drawn.
        guides = () if made is None else (made,)
        self._drawn[layer.id] = with_fields(drawn, guides=guides)
        if made is not None and self._scene.ax.get_legend() is made:
            self._displaced(made, keeper=layer.id)
        if made is not None and not self.is_visible(layer.id):
            # A key is drawn as hidden as the layer it explains. `set_visible` already hid the key it
            # *found*, so hiding a layer and then keying it used to give a different figure from keying it
            # and then hiding it — a bar labelling a picture the reader cannot see, which is the symptom
            # order 24 names as its motivation. Applied here rather than in
            # :func:`~digitalearth.static.guides.draw_guide` because visibility is the renderer's reading:
            # it lives on the artists, not in the description, and `is_visible` is what knows that a layer
            # owning no artist carries it here instead.
            _set_visible(_guide_artist(made), False)
        return made

    def _displaced(self, legend: Any, keeper: Optional[str] = None) -> None:
        """Forget every swatch legend but the one the axes now holds, wherever it was recorded.

        An axes holds **one** legend, so drawing a second layer's swatches takes the first layer's off — and
        the first layer went on holding it in :attr:`DrawnLayer.guides`, a key that is not on the figure and
        can never be drawn again. The record then disagreed with the axes about which layer the reader is
        looking at, which is the disagreement order 24 moved the guide onto the layer to remove (review M4).

        Stated as "every legend but this one" rather than "the one I just replaced", because this renderer
        is not the only thing that writes that slot: a **drawer** can, and one does — a categorical fill's
        cleopatra glyph draws its own swatch legend on its way in, silently displacing whatever another
        layer's guide had put there before :meth:`draw_guide` is reached at all. Measured on a figure read
        back with a shown guide on two fills: the second layer's rebuild displaced the first layer's key
        through the glyph, and a hook that only forgot what *it* had replaced left the first layer holding
        an orphan. The stronger rule costs one pass over a record that is already in hand.

        Colorbars are left alone: they live on their own axes, any number of them coexist, and
        :func:`_guide_artist` is what tells the two kinds apart without asking about types — it answers with
        the key itself for a ``Legend`` and with the axes for a ``Colorbar``.

        The orphan is **dropped, never detached.** matplotlib gives every legend it makes for an axes the
        same removal hook — ``Axes._remove_legend``, which sets ``legend_`` to ``None`` whichever legend is
        sitting there — so calling ``remove()`` on a displaced one takes the *surviving* layer's key off the
        figure. That is what `Renderer.remove` did to it, and dropping it here is what stops that: there is
        nothing to take off, because matplotlib has already taken it off. :func:`_detach_guide` is the same
        rule at the call that does the damage, for a record this sweep has not reached yet.

        **Called from both writers**, which is what review H1 closed: :meth:`draw_guide` when it made the
        new legend, and :meth:`draw_layer` after every drawer, since a drawer writes that slot too and the
        layer it drew for need not carry a guide at all. Reaching it only from the first left the ordinary
        flow — key one categorical fill, add a second — with an orphan in the keyed layer's record.

        Args:
            legend: The ``Legend`` the axes is showing now, which every other record gives up. ``None`` when
                the axes holds none, in which case every recorded legend is an orphan.
            keeper: The layer whose record is left alone, or ``None`` for "no layer is exempt" — which is
                what :meth:`draw_layer` asks for, because the drawer may have displaced the redrawn layer's
                *own* previous legend. Passing the layer :meth:`draw_guide` has just keyed changes nothing
                on that path (its one recorded guide **is** `legend`, so the sweep keeps it either way) and
                is named there for the reader.
        """
        # Walked live rather than over a copy: every write below lands on a key this walk is already
        # holding, so the table never gains or loses an entry mid-iteration — the one thing a dict
        # refuses — and the copy the first version took was insurance against a mutation that cannot
        # happen here.
        for layer_id, drawn in self._drawn.items():
            if layer_id == keeper:
                continue
            kept = tuple(
                held
                for held in drawn.guides
                if held is legend or _guide_artist(held) is not held
            )
            if len(kept) != len(drawn.guides):
                for gone in drawn.guides:
                    if not any(gone is stays for stays in kept):
                        # Out of cleopatra's record of what it last rendered as well, for the same reason:
                        # it removes what it finds there by calling `remove()`, and these are the artists
                        # whose `remove()` reaches another layer's key. Unlike `_detach`, this forgets
                        # without removing.
                        _forget_render_artist(self._scene.ax, gone)
                self._drawn[layer_id] = with_fields(drawn, guides=kept)

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

        **This is the picture, not the description.** It reconciles the artists on the axes and this
        renderer's own record of them: the scene's layer tree and its sources are untouched, so a caller who
        applies a figure through *this* method still has a map describing the one its builders made.
        :meth:`~digitalearth.static.scene.Scene._change` is what pairs the two — it calls this and then
        installs the description, so the figure the map reports is only ever one the axes was brought to.

        **A restyle expressed only in a value the description does not carry is invisible to this path.**
        The difference between two figures is read off their descriptions, and of a caller's engine
        keywords only the plain half is described — a string, a boolean, a finite number or ``None``
        (:func:`~digitalearth.base.spec._serial.travels_in_a_figure`). A colormap built on the spot, a
        ``Normalize``, a dash tuple, a per-pixel ``alpha`` array: each is held on the scene beside the
        layer instead (:func:`drawing_opts`), and two layers differing only in one of them compare
        **equal** — ``diff`` reports no restyle and nothing is redrawn, although the two draw different
        pictures. The plain half is described and does reach here, which is the narrowing review M3 made.
        What is left is still wider than what the writer itself refuses: ``to_json_value`` accepts a tuple
        (as a list) and an array (as nested lists), and both are kept out of the description deliberately —
        the first because matplotlib refuses what comes back, the second because it is the layer's data
        (review M5).

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
        if change.order is not None:
            # Last, so the layers a redraw or an add has just appended are in the arrangement too.
            self._repaint(change.order, after.layers, before.layers.ids)

    def _repaint(
        self, order: Tuple[str, ...], layers: Any = None, previous: Tuple[str, ...] = ()
    ) -> None:
        """Paint the layers this renderer holds in `order`, leaving every other artist where it is.

        ``FigureDiff`` has reported ``order`` since the seam landed and no renderer acted on it, so a
        reorder reached every tier's *description* and none of their pictures. On this tier it reaches both,
        in two steps: the layers' artists are re-arranged in :func:`_painted`, which matplotlib honours
        between artists of one z-order, and then :func:`_rank_zorders` gives each layer the z-order its own
        place in that arrangement earns, which is what carries the move across two of them. Review H5: the
        first step alone left a raster moved above a point layer still painted underneath it.

        The layers' own slots are refilled and nothing else moves. The axes' patch, its spines, a colorbar's
        axes and anything a caller drew straight onto :attr:`Scene.ax` are not layers, and a reorder is not
        licence to move them, so :func:`_rank_zorders` writes only within the span the layers already
        occupied — see there for why each layer's value comes from its own place in `order` rather than from
        the multiset the layers held between them.

        Args:
            order: The layer ids in the draw order the figure now describes, bottom first.
            layers: The figure's `LayerTree`, read only for each layer's band so the z-orders are ranked
                within a band and never across them. When it is `None` every layer is treated as one band,
                which is what the rollback path wants: it is restoring an arrangement, not making a move.
            previous: The order being moved away from. A band whose own sequence is unchanged between
                `previous` and `order` is not written at all — `LayerTree.move` confines a move to one band,
                so every other band's values are none of a move's business. Empty means "compare against
                nothing", which ranks every band; that is the rollback path, restoring an arrangement.
        """
        painted = _painted(self._scene.ax)
        if painted is None:
            return
        owner = {
            id(artist): layer_id
            for layer_id, drawn in self._drawn.items()
            for artist in drawn.artists
        }
        slots = [index for index, artist in enumerate(painted) if id(artist) in owner]
        blocks: Dict[str, List[Any]] = {}
        for index in slots:
            blocks.setdefault(owner[id(painted[index])], []).append(painted[index])
        for artists in blocks.values():
            # One layer's artists in the order matplotlib paints them, so a layer whose pieces carry
            # different z-orders — a limb-split coastline is one, a glyph that drew its own overlay another —
            # keeps its own stacking through the list, which `_rank_zorders` leaves ordered.
            artists.sort(key=_zorder_of)
        ordered = [list(blocks.get(layer_id, ())) for layer_id in order]
        arranged = [artist for block in ordered for artist in block]
        bands: Dict[str, List[List[Any]]] = {}
        for layer_id, block in zip(order, ordered):
            bands.setdefault(_band_of_layer(layers, layer_id), []).append(block)
        if previous:
            was: Dict[str, List[str]] = {}
            for layer_id in previous:
                was.setdefault(_band_of_layer(layers, layer_id), []).append(layer_id)
            now: Dict[str, List[str]] = {}
            for layer_id in order:
                now.setdefault(_band_of_layer(layers, layer_id), []).append(layer_id)
            bands = {
                band: blocks
                for band, blocks in bands.items()
                if now.get(band) != was.get(band)
            }
        if len(arranged) != len(slots):
            # An artist owned by a layer the new order does not name would be dropped from the axes by the
            # assignment below, and one named twice would be painted twice. Neither is reachable from
            # `apply` — `order` is the new figure's full draw order — and leaving the axes alone is the
            # answer that cannot lose a drawing.
            logger.debug(
                "not repainting: %d artists to place in %d slots",
                len(arranged),
                len(slots),
            )
            return
        for slot, artist in zip(slots, arranged):
            painted[slot] = artist
        _rank_zorders(list(bands.values()))

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
        # The arrangement goes back with the artists. `_restore` puts a *re-drawn* layer back where it stood,
        # but a refused change that reordered the layers it kept moved artists nothing re-drew — so the
        # rollback has to say what the order was, which is what `before` is.
        self._repaint(before.layers.ids)

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

        ``Scene.layers`` is where each layer's default key title is filed, matched by the identity of its
        ``(glyph, mappable)`` pair, so a restored layer whose pair is not put back where it was loses the
        units its colorbar labels itself with. (Until order 24 it was worse: the registry was what
        ``colorbar(layer=-1)`` indexed, so a re-registration silently re-keyed the default colorbar.) The
        registry is rebuilt from what it held, each restored layer's old pair swapped for the one its
        re-draw registered, and a layer that could not be restored dropped.

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
        # First, and unconditionally: a layer that never drew has no entry in `_drawn` to reach the body
        # below, and its ask would outlive it and answer for whatever took the id next.
        self._asked.pop(layer_id, None)
        drawn = self._drawn.pop(layer_id, None)
        if drawn is None:
            return
        self._scene._unregister_artist(drawn)
        # The key goes with the layer it explains. It is taken off first, because a colorbar holds a
        # reference to the mappable it was built from and `Colorbar.remove` is the call that lets it go.
        for guide in drawn.guides:
            _detach_guide(guide, self._scene.ax)
        for artist in drawn.artists:
            _detach(artist, self._scene.ax)

    def set_visible(self, layer_id: str, visible: bool) -> None:
        """Show or hide every artist drawn for a layer.

        **Every** artist, not the first: a layer can own several — a limb-split coastline is one polyline per
        piece — and they are one layer to a viewer, so a half-applied hide leaves part of the layer on the
        figure while :meth:`is_visible` already answers ``False`` over the aggregate.

        Args:
            layer_id: The layer to toggle. An id nothing was drawn for is ignored.
            visible: Whether it is drawn. For a layer that owns **no** artist the request is *remembered*
                rather than written anywhere, because there is nothing to write it to — see
                :meth:`is_visible`.
        """
        drawn = self._drawn.get(layer_id)
        if drawn is None:
            return
        # The key is hidden with the layer, before the early return below: a hidden layer whose colorbar
        # stayed on the figure is a bar labelling something the reader cannot see. `_guide_artist` is what
        # knows a `Colorbar` carries the flag on its axes rather than on itself.
        for guide in drawn.guides:
            _set_visible(_guide_artist(guide), visible)
        if not drawn.artists:
            # Nothing on the axes carries the flag, so the renderer carries it instead. Without this a
            # layer asked to hide answered `True` when asked back, because `all(())` is `True` — and
            # `Scene.set_visible`, whose whole contract is that this answer can be trusted, inherited that.
            self._asked[layer_id] = bool(visible)
            return
        self._asked.pop(layer_id, None)
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
            ``True`` when every artist the layer owns is on.

            **A layer that owns no artist answers what was last asked of it**, and ``True`` when nothing has
            been. A graticule is the one that gets there: its lines reach the axes with the globe frame, so it
            owns none until
            :meth:`~digitalearth.static.maps.projection.ProjectionMixin._apply_frame` has run, and none at all
            on a flat map. The answer used to be ``all(())`` — unconditionally ``True`` — so a flat graticule
            described ``visible=False`` read back as drawn, and ``set_visible(id, False)`` followed by this
            could answer ``True`` for the layer it had just hidden. Nothing was drawn either way, so no pixel
            was wrong; the renderer's *answer* was, and that answer is what the behavioural conformance suite
            and a layer switcher read. An unframed globe's grid still reads ``True``, because nobody asked for
            it to be off — which is the case the empty aggregate got right.

        Raises:
            KeyError: when nothing was drawn for `layer_id`, naming it. A layer the axes does not hold has
                no visibility to report, and :attr:`drawn` is what says which those are.

        Examples:
            - A layer the caller has just hidden reads back hidden, and one nothing was drawn for is
              refused by name:
                ```python
                >>> import matplotlib
                >>> matplotlib.use("Agg")
                >>> from digitalearth.static import Map
                >>> m = Map()
                >>> _ = m.text(4.9, 52.4, "Amsterdam")
                >>> m._renderer.is_visible("text-1")
                True
                >>> m._renderer.set_visible("text-1", False)
                >>> m._renderer.is_visible("text-1")
                False
                >>> m._renderer.is_visible("nope")  # doctest: +ELLIPSIS
                Traceback (most recent call last):
                    ...
                KeyError: "nothing is drawn for layer 'nope', so it has no visibility to report; ..."
                >>> m.close()

                ```
        """
        drawn = self._drawn.get(layer_id)
        if drawn is None:
            raise KeyError(
                f"nothing is drawn for layer {layer_id!r}, so it has no visibility to report; the static "
                f"tier holds {sorted(self._drawn)}"
            )
        if not drawn.artists:
            return self._asked.get(layer_id, True)
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
