"""GuideMixin — the 3-D tier's colour key, recorded on the layer it explains (order 24, #203).

The last two Core names this tier was short of. `contract.PENDING["3d"]` scheduled both against order 24 —
*"the scalar bar is PyVista's, and becomes a guide on the encoding"*, *"a keyed list beside a scene"* — and this
is that order: :meth:`GuideMixin.colorbar` and :meth:`GuideMixin.legend`, keyed to a layer's
:class:`~digitalearth.base.spec.encoding.Encoding` rather than to whatever was drawn last.

**Recorded first, drawn from the record.** Both methods write a
:class:`~digitalearth.base.spec.encoding.Guide` onto the layer's colour encoding through
:meth:`~digitalearth.base.spec.style.Symbology.with_guide`, and only then draw. That single ordering is what the
order buys: the key moves with its layer, goes away with it, hides with it, and survives
`figure_spec.to_dict()` → `FigureSpec.from_dict()` — because the layer's `Symbology` is already what travels.
A key drawn when asked and then forgotten could do none of the four.

**Two widgets, one channel.** A colour encoding whose `Scale` cuts classes is explained by PyVista's keyed
legend (`add_legend`); one that runs a continuous ramp is explained by PyVista's scalar bar
(`add_scalar_bar`). So the *form* is read off the description rather than stored beside it, and the two methods
refuse each other's case by name: a scalar bar over class **indices** would be labelled `0, 1, 2 …`, and a
keyed list over a ramp has no keys to list. Both are read off the layer's *description*, which is the same
place :func:`digitalearth.api._has_a_key_to_draw` asks the static tier which of its layers publish a colour a
bar can describe.

**Where the key is drawn, so it survives a re-render.** Not in
:meth:`~digitalearth.three_d.base.Scene3DBase._dress_plotter`: that re-applies *scene* state — the view scale,
the camera, the title — onto a plotter that carries none of the layers' actors, and a scalar bar needs the
mapper of a drawn actor. It is drawn at the end of :meth:`~digitalearth.three_d.renderer.Renderer3D.apply`,
which is the one place every figure change funnels through — a layer added, removed, hidden, shown, moved,
restyled, or a whole figure brought in by `draw_figure`/`from_figure` — and the only place where the actors a
bar reads are current. :func:`redraw_guides` therefore takes the figure it is reconciling against as an
argument rather than reading `scene._figure`, which during `apply` is still the figure being left behind.

**What PyVista's own slots mean, and what this does about them.** Measured on PyVista 0.48.4:

* A plotter holds **one scalar bar per title**. A second `add_scalar_bar(title=...)` under a title already
  there does not raise and does not make a second bar — it binds the new mapper to the *existing* bar, which
  goes on showing the first layer's range. So a title that would collide with another layer's is **refused**
  by name here, with `label=` named as the way out.
* `add_mesh`/`add_points`/`add_volume` draw a scalar bar **by default**, titled by the scalars array's own
  name. That bar is this tier's colour key today, so a guide has to know which bar is its layer's before it can
  retitle or remove one — and the array's name is read back off the engine (:func:`_engine_title`) rather than
  assumed to be the encoding's field, because the two differ wherever a column has a better name than the array
  it was bound under (a cloud coloured by `value_column="depth"` binds `scalar`). Where they do agree, asking
  a key a caller already has changes nothing at all.
* `Plotter.remove_actor` takes the scalar bar bound to that actor with it. So a layer's bar goes when the layer
  does without this module doing anything — what :func:`redraw_guides` drops for a removed layer is the
  **record**, which would otherwise be inherited by the next layer to answer to that id.
* A plotter holds exactly **one** legend actor: a second `add_legend` replaces the first. So the keyed list is
  built as one box from every layer that asks for one, in layer order, rather than letting the last caller win.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as with_fields
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from digitalearth.base.spec import Encoding, FigureSpec, Guide, LegendSpec, Scale

__all__ = ["ColourGuide", "GuideMixin", "color_scale", "guide_field", "redraw_guides"]

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.three_d.base import Scene3DBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object

#: Which corner of the render window a keyed list is drawn in, per
#: :data:`~digitalearth.base.registry.FURNITURE_ANCHORS`. PyVista's `add_legend(loc=...)` spells the same four
#: places its own way, so the shared anchor vocabulary is translated here rather than leaked into a caller's
#: call — a figure written on another tier carries `anchor="bottom-right"` and has to mean the same thing.
LEGEND_LOCATIONS: dict[str, str] = {
    "top-left": "upper left",
    "top-right": "upper right",
    "bottom-left": "lower left",
    "bottom-right": "lower right",
}

#: Where a scalar bar sits for each shared anchor: `(position_x, position_y)` in viewport fractions.
#: PyVista places its own bar along the bottom of the window and takes no anchor, so an anchor that arrived on
#: a `Guide` would otherwise be recorded and silently dropped. The x offsets leave the default bar width room
#: inside the frame.
BAR_POSITIONS: dict[str, tuple[float, float]] = {
    "top-left": (0.05, 0.90),
    "top-right": (0.55, 0.90),
    "bottom-left": (0.05, 0.05),
    "bottom-right": (0.55, 0.05),
}

#: The corner a keyed list falls back to when no anchor was asked for — PyVista's own default for
#: `add_legend`, so a guide carrying no anchor draws where the engine would have drawn it.
DEFAULT_LEGEND_LOCATION: str = "upper right"


def guide_field(layer: Any) -> str | None:
    """Return the field a layer's colour is driven by, or `None` when nothing drives it.

    Args:
        layer: The layer's :class:`~digitalearth.base.spec.LayerSpec`.

    Returns:
        The band, variable or column name the layer's `color` encoding names — what a reader wants the key
        titled after. It is **not** always the name of the array the drawer binds on the mesh, and so not
        always the title PyVista's own bar carries: a cloud given `value_column="depth"` names `depth` here and
        binds its values under `scalar`. Where the two differ, the engine's title is read back off the drawn
        actor by :func:`_engine_title` rather than assumed from this. `None` for a layer whose colour is a
        flat constant, or which publishes no colour encoding at all: a key over a colour nothing varies would
        have no values to label.

    Examples:
        - A colour-driven layer names the array its key would describe:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import get_source
            >>> from digitalearth.three_d import Scene3D
            >>> from digitalearth.three_d.guides import guide_field
            >>> scene = Scene3D(off_screen=True)
            >>> _ = scene.terrain(get_source(np.add.outer(np.arange(4.0), np.arange(5.0))))
            >>> guide_field(scene.get_layer("terrain-1"))
            'elevation'
            >>> scene.close()

            ```
        - A flat-coloured layer publishes none, so there is nothing to key:
            ```python
            >>> import numpy as np
            >>> from digitalearth.base.sources import get_source
            >>> from digitalearth.three_d import Scene3D
            >>> from digitalearth.three_d.guides import guide_field
            >>> dem = get_source(np.add.outer(np.arange(4.0), np.arange(5.0)))
            >>> scene = Scene3D(off_screen=True)
            >>> _ = scene.terrain(dem, scalars=None, color="red")
            >>> guide_field(scene.get_layer("terrain-1")) is None
            True
            >>> scene.close()

            ```
        - A named column is the field, and the array the drawer binds is not — the case the engine's own
          title has to be read back for:
            ```python
            >>> import geopandas as gpd
            >>> from shapely.geometry import Point
            >>> from digitalearth.three_d import Scene3D
            >>> from digitalearth.three_d.guides import guide_field
            >>> features = gpd.GeoDataFrame(
            ...     {"depth": [1.0, 5.0, 9.0]},
            ...     geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            ...     crs=4326,
            ... )
            >>> scene = Scene3D(off_screen=True)
            >>> _ = scene.point_cloud(features, value_column="depth")
            >>> guide_field(scene.get_layer("point_cloud-1"))
            'depth'
            >>> list(scene.plotter.scalar_bars.keys())
            ['scalar']
            >>> scene.close()

            ```
    """
    encoding = layer.symbology.encoding("color")
    return None if encoding is None else encoding.field


def color_scale(
    values: Any, *, scheme: Any, k: int, cmap: Any, missing: str | None = None
) -> Scale | None:
    """Return the :class:`~digitalearth.base.spec.scale.Scale` a colour column is drawn through.

    The builder's half of the classification the drawer performs. It is the **same** computation, not a second
    one: a graduated scheme's edges come from :meth:`~digitalearth.base.spec.scale.Scale.breaks_of` — which is
    what :func:`~digitalearth.three_d.base.classified_scalars` calls — and a categorical scheme's categories
    and colours from :func:`~digitalearth.base.symbology.categorical_colors` with the same
    :func:`~digitalearth.base.symbology.resolve_categorical_cmap` sentinel. That is what lets
    :meth:`~digitalearth.base.spec.legend.LegendSpec.from_scale` derive a legend whose rows cannot disagree
    with the classes the picture was painted with.

    Args:
        values: The colour column, or `None` when the layer colours by nothing.
        scheme: `None` for a continuous ramp, `"categorical"` for one colour per distinct value, or any
            graduated scheme name (or explicit edge sequence) the shared classifier takes.
        k: How many classes a graduated scheme cuts.
        cmap: The colormap the classes are drawn from.
        missing: Colour for a value the scale cannot place.

    Returns:
        The scale, or `None` when there is nothing to measure — no values at all, or values the scheme cannot
        classify. **A refusal is answered with `None` rather than raised**: the drawer classifies the same
        column a moment later and raises the tier's own message for it, so raising here would move a caller's
        error to a different call with different words for the same cause.

    Examples:
        - A graduated scheme records the class edges, so a legend can label each class:
            ```python
            >>> from digitalearth.three_d.guides import color_scale
            >>> scale = color_scale([1.0, 2.0, 3.0, 40.0], scheme="quantiles", k=2, cmap="viridis")
            >>> scale.is_classified, len(scale.class_ranges())
            (True, 2)

            ```
        - A categorical scheme records the categories and the colours they were given:
            ```python
            >>> from digitalearth.three_d.guides import color_scale
            >>> scale = color_scale(["a", "b", "a"], scheme="categorical", k=5, cmap="tab10")
            >>> scale.categories, scale.color_for("a"), scale.color_for("b")
            (('a', 'b'), '#1f77b4', '#ff7f0e')

            ```
        - A column the scheme cannot cut answers `None`, leaving the refusal to the drawer:
            ```python
            >>> from digitalearth.three_d.guides import color_scale
            >>> color_scale([1.0, 1.0, 1.0], scheme="quantiles", k=4, cmap="viridis") is None
            True

            ```
    """
    if values is None:
        return None
    from digitalearth.base.symbology import categorical_colors, resolve_categorical_cmap

    try:
        if isinstance(scheme, str) and scheme.lower() == "categorical":
            categories, colours = categorical_colors(
                values, cmap=resolve_categorical_cmap(cmap)
            )
            return Scale.categorical(
                [_native(category) for category in categories],
                list(colours),
                missing=missing,
            )
        return Scale.from_values(values, scheme=scheme, k=k, missing=missing)
    # Anything the classifier, the colormap or the column itself refuses. The drawer runs the same
    # computation on the same values and raises the tier's message for it, so this must not pre-empt that
    # with a message of its own — and a layer described with no scale is under-described rather than
    # mis-described, which is the trade `asked_constants` makes for the same reason.
    except Exception:  # noqa: BLE001
        return None


def _native(value: Any) -> Any:
    """Return a category as a plain Python value, so a figure can be written with it.

    Args:
        value: One category, as the shared categoriser produced it — often a numpy scalar, since the
            categories are read off a numpy column.

    Returns:
        `value.item()` for a numpy scalar, `value` otherwise. A `numpy.str_` category stored as it came back
        left `Symbology.to_dict()` writing a type no JSON form covers.
    """
    return value.item() if isinstance(value, np.generic) else value


def _keyed(scale: Scale | None) -> bool:
    """Whether a scale cuts keys a list can name, rather than running a continuous ramp.

    Args:
        scale: The colour encoding's scale, or `None` for a layer whose colours the engine maps itself.

    Returns:
        `True` for a classified or categorical scale — the two shapes with discrete classes to list.
    """
    return scale is not None and (scale.is_classified or scale.is_categorical)


def _keyed_row_count(scale: Scale) -> int:
    """Return how many rows a keyed layer's classes contribute to the legend box.

    Args:
        scale: The layer's colour scale, of one of the two shapes :func:`_keyed` answers `True` for.

    Returns:
        One row per category for a categorical scale, one per class for a graduated one — the two arms
        :meth:`~digitalearth.base.spec.legend.LegendSpec.from_scale` keys.

        The rows *themselves* are derived at draw time from that method, which needs the colours read back
        off the drawn lookup table and so cannot be built before the layer is on the window. The **count**
        is a property of the scale alone, and :meth:`GuideMixin.legend` has to know it before it records
        anything: refusing from the reconcile instead would roll the figure back and leave the caller's
        label list on the record, where every later draw would trip over it (review R2-M5).
    """
    return len(scale.categories) if scale.is_categorical else len(scale.class_ranges())


def _colour_encoding(figure: FigureSpec, layer_id: str) -> Encoding:
    """Return one layer's colour encoding, refusing a layer that has none.

    Args:
        figure: The figure the layer belongs to.
        layer_id: The layer to read.

    Returns:
        Its `color` :class:`~digitalearth.base.spec.encoding.Encoding`.

    Raises:
        KeyError: when no layer on the scene has that id, from the layer tree.
        ValueError: when the layer's colour is not driven by anything, naming the layers that are. A key over
            a flat colour would have no values to label and no scale to sample.
    """
    layer = figure.layers.get(layer_id)
    encoding = layer.symbology.encoding("color")
    if encoding is None or encoding.field is None:
        keyed = [
            held for held in figure.layers.ids if guide_field(figure.layers.get(held))
        ]
        raise ValueError(
            f"layer {layer_id!r} is not coloured by any data, so it has no colour key to show; the layers "
            f"that are coloured by data are {keyed}"
        )
    return encoding


def _resolve_layer(figure: FigureSpec, layer_id: str | None, caller: str) -> str:
    """Return the layer a key was asked for, defaulting to the most recent colour-driven one.

    Args:
        figure: The figure the scene shows.
        layer_id: The caller's layer, or `None` for "whichever last published a colour encoding".
        caller: The method that was called, for the message.

    Returns:
        The layer's id.

    Raises:
        ValueError: when `layer_id` is `None` and no layer is coloured by data. **The most recent
        colour-driven layer, not simply the last one added**: a label or a flat-coloured mesh added after a
        terrain is the last layer and has nothing to key, so taking it would refuse a call that has an
        obvious answer.
    """
    if layer_id is not None:
        return layer_id
    for held in reversed(figure.layers.ids):
        if guide_field(figure.layers.get(held)):
            return held
    raise ValueError(
        f"{caller} has nothing to describe: no layer on this scene is coloured by data. Draw a terrain, a "
        "volume, a point cloud given values=/value_column=, or extruded polygons given column= first, or "
        "name the layer with layer_id=."
    )


def _plotter_of(scene: Any) -> Any:
    """Return the plotter a key can be drawn on, or `None` when there is none to draw on.

    Args:
        scene: The scene whose plotter is wanted.

    Returns:
        The scene's plotter when it has one that answers to the scalar-bar API — a described scene builds no
        render window, and a stand-in plotter need not implement the whole annotation surface, which is the
        rule :func:`~digitalearth.three_d.decoration.redraw_decoration` already applies. `None` otherwise.
    """
    plotter = getattr(scene, "_plotter", None)
    if plotter is None or not hasattr(plotter, "add_scalar_bar"):
        return None
    return plotter


def _drawn_actor(scene: Any, layer_id: str) -> Any:
    """Return the actor drawn for one layer, or `None` when nothing was.

    Args:
        scene: The scene whose renderer holds the drawn pairs.
        layer_id: The layer to look up.

    Returns:
        The PyVista actor, or `None` for a layer the plotter does not hold — which is what a scene handed a
        fresh plotter looks like, since a swap carries the scene's view state over and not its actors.
    """
    drawn = scene.renderer.drawn.get(layer_id)
    return None if drawn is None else drawn[1]


def _engine_title(scene: Any, layer_id: str, field: str) -> str:
    """Return the title PyVista's **own** scalar bar for one layer carries.

    `add_mesh`/`add_points`/`add_volume` draw a scalar bar by default and title it after the scalars array —
    so every colour-driven layer already has a key, and the reconcile has to know which bar is that layer's
    before it can retitle or remove it. The answer is read off the engine rather than kept in a per-kind table
    beside the builders: the array a drawer binds (`elevation`, `scalar`, `value`, `field`) is not always the
    **field** the encoding names, since a point cloud coloured by `value_column="depth"` binds its values under
    `scalar` and a key over it should read `depth`.

    Args:
        scene: The scene holding what was drawn.
        layer_id: The layer to ask about.
        field: The encoding's field, used when nothing was drawn to read a name off.

    Returns:
        The array's name — `mapper.array_name` where the mapper has one (a volume mapper does not) and the
        mesh's own active scalars otherwise; `field` for a layer the plotter does not hold.
    """
    drawn = scene.renderer.drawn.get(layer_id)
    if drawn is None:
        return field
    mesh, actor = drawn
    return (
        getattr(actor.mapper, "array_name", "")
        or getattr(mesh, "active_scalars_name", "")
        or field
    )


def _sharing_bar(
    scene: Any, figure: FigureSpec, layer_id: str, title: str
) -> list[str]:
    """Return the other layers whose own engine-drawn bar is the same one.

    Args:
        scene: The scene holding what was drawn.
        figure: The figure the scene shows.
        layer_id: The layer asking for a key.
        title: The engine's title for that layer's bar, from :func:`_engine_title`.

    Returns:
        The ids of the other colour-driven layers PyVista titled the same way. They matter because a plotter
        holds one bar per title: two volumes, or a volume and an isosurface, both bind `field`, so the engine
        has already bound both mappers to one bar — and that bar is not either layer's to retitle or take
        away on its own.
    """
    return [
        held
        for held in figure.layers.ids
        if held != layer_id
        and (field := guide_field(figure.layers.get(held)))
        and _engine_title(scene, held, field) == title
    ]


def _remove_bar(plotter: Any, title: str) -> None:
    """Take one scalar bar off a plotter, by title.

    Args:
        plotter: The plotter holding it.
        title: The bar's title. One the plotter does not hold is ignored — `remove_scalar_bar` raises a bare
            `KeyError` for it, and a reconcile that has already established the bar is gone should not.
    """
    if title in plotter.scalar_bars:
        plotter.remove_scalar_bar(title, render=False)


def _release_bar(
    scene: Any, figure: FigureSpec, plotter: Any, layer_id: str, title: str
) -> None:
    """Give up one layer's claim on a bar, taking it off the window only if nothing else binds it.

    Args:
        scene: The scene holding what was drawn.
        figure: The figure being drawn.
        plotter: The plotter holding the bar.
        layer_id: The layer letting it go.
        title: The bar's title.

    Note:
        **Whether it comes off is asked of the other layers, not of how the title was arrived at.** A title
        no other layer binds is this layer's alone and comes off with the claim; a shared one does not,
        because a layer that adopted it and then asked for a title of its own would otherwise take the key
        away from the layer still reading it. That bar comes off in :func:`_drop_orphaned_bars`, once every
        layer binding it has settled.

        A *custom* title is not exempt from that question, although a second layer **asking** for one
        already on the window is refused (:func:`_add_bar`). A second layer's **engine** title is not asked
        for, and it can equal a custom one: measured, two point clouds given `value_column="depth"` both
        bind their values under `scalar`, so `colorbar("one", label="scalar")` is accepted — it adopts the
        bar already there — and `_sharing_bar(scene, figure, "one", "scalar")` then answers `['two']`, so
        releasing it leaves the bar on the window. :func:`_sharing_bar` is the whole of the rule here; the
        spelling of the title decides nothing.
    """
    if not _sharing_bar(scene, figure, layer_id, title):
        _remove_bar(plotter, title)


def _bar_placement(anchor: str | None) -> dict[str, Any]:
    """Return the `add_scalar_bar` keywords that put a bar where a guide asked for it.

    Args:
        anchor: The guide's anchor, or `None` for PyVista's own placement.

    Returns:
        `position_x`/`position_y` for one of the four shared anchors, or an empty dict — which leaves
        PyVista's default, the placement :class:`~digitalearth.base.spec.encoding.Guide` documents `None` as
        meaning.
    """
    placed = BAR_POSITIONS.get(anchor or "")
    if placed is None:
        return {}
    return {"position_x": placed[0], "position_y": placed[1]}


def _drawn_colors(actor: Any, scale: Scale) -> list[str] | None:
    """Return the colours a classified layer was actually drawn in, read off its lookup table.

    This is the reason a swatch cannot disagree with the picture. The colours are not recomputed from the
    colormap for the legend — they are read back from the `vtkLookupTable` the layer is being rendered
    through, which is the table :func:`~digitalearth.three_d.base.classified_scalars` built.

    Args:
        actor: The layer's drawn actor.
        scale: Its colour scale. A categorical scale carries its own colours, so none are read — passing them
            would reintroduce the second source of truth
            :meth:`~digitalearth.base.spec.legend.LegendSpec.from_scale` removes.

    Returns:
        One `#rrggbb` per class for a classified scale, or `None` for a categorical one.
    """
    if scale.is_categorical:
        return None
    from matplotlib.colors import to_hex

    values = np.asarray(actor.mapper.lookup_table.values, dtype="float64") / 255.0
    return [to_hex(row[:3]) for row in values]


def _legend_rows(
    spec: LegendSpec, labels: Sequence[str] | None, prefix: str | None
) -> list[tuple[str, str]]:
    """Return one layer's contribution to the keyed list, as the `(text, colour)` pairs PyVista takes.

    Args:
        spec: The legend derived from the layer's scale.
        labels: The caller's own row labels, replacing the derived ones. :meth:`GuideMixin.legend` refuses a
            list that does not number the rows, so in the ordinary case there is exactly one label per
            entry. The per-row fallback below is for the residual case that check cannot reach: the labels
            are drawn state on the scene rather than part of the figure, so a layer reclassified by
            `replace_layer` after they were recorded can arrive here with more classes than the caller
            named. Those rows keep their derived text rather than the box losing them.
        prefix: The layer's title, prepended to every row when more than one layer is keyed — a render window
            holds one legend box, and an unprefixed box of two layers' classes says nothing about which layer
            a swatch belongs to. `None` leaves the labels bare, which is the single-layer case.

    Returns:
        The rows, in the order the legend derived them.
    """
    chosen = list(labels) if labels is not None else []
    rows = []
    for position, entry in enumerate(spec.entries):
        text = chosen[position] if position < len(chosen) else entry.label
        rows.append((f"{prefix}: {text}" if prefix else text, entry.color or "#cccccc"))
    return rows


@dataclass(frozen=True)
class ColourGuide:
    """One layer's ask for a colour key: what it drives, what it wants said, and whether it is drawn.

    A value rather than a tuple because the reconcile asks the same four questions of every row — is this a
    keyed list or a bar, is it shown, what is it called, what title should its bar be on — and answering them
    at each call site is where the two widgets would drift apart.

    Attributes:
        layer_id: The layer being keyed.
        field: The column, band or array its colour is driven by, from the encoding.
        guide: What the layer says about the key — shown, titled, anchored.
        visible: Whether the layer itself is drawn. A hidden layer explains nothing.
        scale: How its values map onto colour, or `None` where the engine holds the mapping itself.
    """

    layer_id: str
    field: str
    guide: Guide
    visible: bool
    scale: Scale | None = None

    @property
    def keyed(self) -> bool:
        """Whether this layer's colour has classes a list can name rather than a ramp.

        Returns:
            `True` for a classified or categorical scale.
        """
        return _keyed(self.scale)

    @property
    def shown(self) -> bool:
        """Whether the key is to be drawn at all.

        Returns:
            `True` when the guide asks for it **and** the layer is visible.
        """
        return self.guide.show and self.visible

    @property
    def title(self) -> str:
        """What the key is called.

        Returns:
            The guide's own title, or the field the colour is driven by — which in the common case is the name
            PyVista has already titled its own bar with.
        """
        return self.guide.title or self.field

    @property
    def wanted_bar(self) -> str | None:
        """The title this layer's scalar bar should be on.

        Returns:
            :attr:`title` when a bar is wanted, or `None` — for a key switched off, a hidden layer, or a
            classified layer, whose key is the legend box instead.
        """
        return self.title if self.shown and not self.keyed else None


def _guide_plan(figure: FigureSpec) -> list[ColourGuide]:
    """Return what every layer carrying a colour guide asks for, in draw order.

    Args:
        figure: The figure to read.

    Returns:
        One :class:`ColourGuide` per layer whose colour encoding carries a
        :class:`~digitalearth.base.spec.encoding.Guide`. A layer with a colour encoding and **no** guide is
        left out deliberately: its key is whatever PyVista drew of its own accord, and the tier does not own
        it until a caller asks for one.
    """
    plan = []
    for layer_id in figure.layers.ids:
        layer = figure.layers.get(layer_id)
        encoding = layer.symbology.encoding("color")
        if encoding is None or encoding.field is None or encoding.guide is None:
            continue
        plan.append(
            ColourGuide(
                layer_id,
                encoding.field,
                encoding.guide,
                figure.layers.is_visible(layer_id),
                encoding.scale,
            )
        )
    return plan


def redraw_guides(scene: Any, figure: FigureSpec) -> None:
    """Bring the scene's drawn colour keys into line with what its layers ask for.

    Called at the end of :meth:`~digitalearth.three_d.renderer.Renderer3D.apply` — see the module docstring
    for why there and not in `_dress_plotter`. Everything it draws is derived from `figure`, so a key follows
    its layer for free: a removed layer has no row, a hidden one asks for nothing, and a moved one asks for
    exactly what it asked for before.

    Args:
        scene: The scene being reconciled. Nothing is drawn when it has no plotter yet.
        figure: The figure the plotter is being brought **to** — passed in rather than read off the scene,
            because during `apply` the scene still holds the figure it is leaving.

    Raises:
        ValueError: when two layers' guides would want one scalar-bar title. :meth:`GuideMixin.colorbar`
            refuses that at the call, so this is reachable only from a figure built elsewhere — and drawing
            it would bind two layers' mappers to one bar showing the first layer's range.
    """
    plotter = _plotter_of(scene)
    if plotter is None:
        return
    held = scene._guides
    live = set(figure.layers.ids)
    # Read before the prune: a layer that has just been removed took its record with it, and the keyed list
    # has to come off the window even though nothing is left to say it was ever there.
    was_keyed = any(record.get("keyed") for record in held.values())
    for layer_id in [stale for stale in held if stale not in live]:
        # The **record** of a gone layer's key, which is what this drops. The bar itself is already off the
        # window: measured on PyVista 0.48.4, `Plotter.remove_actor` takes the scalar bar bound to that actor
        # with it, and `Renderer3D.remove` calls it. What would outlive the layer without this is the entry —
        # and an id goes back to the pool the moment it addresses nothing on this tier
        # (`free_layer_id`), so a new layer answering to a removed one's name would inherit its bar title and
        # its row labels. `_remove_bar` is still called for the case the engine did not cover: a layer whose
        # actor was gone before its record was, which leaves a bar nothing is bound to.
        drawn = held.pop(layer_id)
        if drawn.get("bar") is not None:
            _remove_bar(plotter, drawn["bar"])
    _redraw_bars(scene, figure, plotter, held)
    _redraw_legend(scene, figure, plotter, held, was_keyed)
    # After the legend rebuild, not inside the bar pass, because the sweep asks of every layer binding a
    # shared bar whether it has a key of its own — and `keyed` is written by `_redraw_legend`. Run before
    # it, the sweep read the *previous* round's answer and so ran one change late: two classified clouds
    # share one `0, 1, 2` index bar, and after the second `legend()` keyed them both that bar was still
    # beside the box until some later, unrelated change swept it (review M7).
    _drop_orphaned_bars(scene, figure, plotter, held)


def _redraw_bars(
    scene: Any, figure: FigureSpec, plotter: Any, held: dict[str, dict[str, Any]]
) -> None:
    """Reconcile the scalar bars the scene's continuous colour guides ask for.

    Args:
        scene: The scene whose actors the bars read their mappers from.
        figure: The figure being drawn.
        plotter: The plotter to draw on.
        held: The scene's record of what it has drawn, keyed by layer id.

    Raises:
        ValueError: when a guide wants a title another layer's bar already holds — see :func:`redraw_guides`.

    Note:
        **Two passes, not one reconcile per layer.** Every changing layer gives its old bar up before any
        layer asks for a new one, because a plotter holds one bar per title and the rule is about the state
        this pass arrives at, not about every step of getting there. Bringing each layer all the way through
        in turn refused a figure that merely **swaps** two titles: the second layer was reached while the
        first's old bar was still on the window, and an end state holding one bar per title was refused
        partway (review L9).
    """
    plans = _guide_plan(figure)
    current = {
        plan.layer_id: _current_bar(scene, figure, plotter, held, plan)
        for plan in plans
    }
    for plan in plans:
        title = current[plan.layer_id]
        if title is not None and title != plan.wanted_bar:
            _release_bar(scene, figure, plotter, plan.layer_id, title)
            held[plan.layer_id].pop("bar", None)
    # Which layer each title has been given to, so a layer asking for one already held can be told whose it
    # is. Seeded with the layers **keeping** the title they already hold, then filled after each add.
    # Filling it only as the pass walked named the holder when it came earlier in draw order and said "for
    # another layer" when it came later, although the refusal is the same one either way (review R2-N2) —
    # and a layer that keeps its title is not reached by the release pass above, so it is exactly the case
    # the walk cannot see. A layer still never reads its own entry: `_add_bar` returns on
    # `current == wanted`, which is the condition the seed is built from, before it looks anything up.
    taken: dict[str, str] = {
        plan.wanted_bar: plan.layer_id
        for plan in plans
        if plan.wanted_bar is not None and current[plan.layer_id] == plan.wanted_bar
    }
    for plan in plans:
        _add_bar(
            scene, plotter, held[plan.layer_id], plan, current[plan.layer_id], taken
        )
        if plan.wanted_bar is not None:
            taken[plan.wanted_bar] = plan.layer_id


def _current_bar(
    scene: Any,
    figure: FigureSpec,
    plotter: Any,
    held: dict[str, dict[str, Any]],
    plan: ColourGuide,
) -> str | None:
    """Return the title of the bar one layer's key is on right now, or `None` when it has none.

    Args:
        scene: The scene whose actors name the arrays.
        figure: The figure being drawn.
        plotter: The plotter to read.
        held: The scene's record of what it has drawn, keyed by layer id. The layer's record is created here
            if it has none, since every layer in the plan is about to be reconciled.
        plan: What this layer asks for.

    Returns:
        The title this layer's key is drawn under: the one it was recorded with, else PyVista's own.
    """
    record = held.setdefault(plan.layer_id, {})
    # Annotated rather than inferred: the record is a `Dict[str, Any]` bag of drawn state, so an unannotated
    # read of it is `Any` and this function would hand one back under a `str | None` signature.
    current: str | None = record.get("bar")
    if current is not None:
        return current
    # PyVista titles the bar it draws of its own accord after the scalars array, so that bar *is* this
    # layer's key. Where another layer binds an array of the same name the engine has bound both mappers to
    # one bar — which is still this layer's key, so asking for it **by its own title** is the no-op
    # `colorbar()` is documented to be, and is adopted. Refusing it instead made `colorbar()` with no label
    # an error on every scene holding two layers over one array (review M8). It is not this layer's alone to
    # *retitle or take away*, though, which is why adoption stops at that one title and why
    # :func:`_release_bar` gives it up without removing it.
    engine = _engine_title(scene, plan.layer_id, plan.field)
    if engine in plotter.scalar_bars and (
        plan.wanted_bar == engine
        or not _sharing_bar(scene, figure, plan.layer_id, engine)
    ):
        return engine
    return None


def _add_bar(
    scene: Any,
    plotter: Any,
    record: dict[str, Any],
    plan: ColourGuide,
    current: str | None,
    taken: dict[str, str],
) -> None:
    """Draw one layer's scalar bar, every changing layer having already given its old one up.

    Args:
        scene: The scene whose actors the bar reads its mapper from.
        plotter: The plotter to draw on.
        record: This layer's entry in the scene's record of what it has drawn.
        plan: What this layer asks for.
        current: The title its key was on before this pass, from :func:`_current_bar`.
        taken: Which layer holds each title, for the message — the layers keeping the one they already have
            plus the layers this pass has given one to, so the holder is named whether it comes earlier or
            later in draw order.

    Raises:
        ValueError: when the wanted title is one the window already carries for another layer. **This is the
            only place that refusal is made** — see the note under :meth:`GuideMixin._record_guide` for why
            there is no second guard over the guides before the record is written.
    """
    wanted = plan.wanted_bar
    if wanted is None:
        record.pop("bar", None)
        return
    if current == wanted:
        record["bar"] = wanted
        return
    if wanted in plotter.scalar_bars:
        holder = taken.get(wanted)
        raise ValueError(
            f"a scalar bar titled {wanted!r} is already on this scene"
            + (f" for layer {holder!r}" if holder else " for another layer")
            + ", and a render window holds one bar per title — a second would show that layer's range "
            f"rather than {plan.layer_id!r}'s. Title this one with colorbar(label=...)"
        )
    actor = _drawn_actor(scene, plan.layer_id)
    if actor is None:
        return
    plotter.add_scalar_bar(
        title=wanted,
        mapper=actor.mapper,
        render=False,
        **_bar_placement(plan.guide.anchor),
    )
    record["bar"] = wanted


def _drop_orphaned_bars(
    scene: Any, figure: FigureSpec, plotter: Any, held: dict[str, dict[str, Any]]
) -> None:
    """Take off an engine-drawn bar that no layer keys off any more.

    The one case :func:`_redraw_bars`' two passes cannot settle on their own. Two terrains both bind
    `elevation`, so PyVista drew **one** bar for both and neither layer may take it away; once each of them
    has a key of its own, that shared bar explains nothing and would otherwise sit in the corner of the
    window for good.

    Called from :func:`redraw_guides` **after** :func:`_redraw_legend`, because the question it asks of each
    binding layer — has it a key of its own — is answered by `record["keyed"]`, which the legend rebuild
    writes.

    Args:
        scene: The scene whose actors name the arrays.
        figure: The figure being drawn.
        plotter: The plotter to draw on.
        held: The scene's record of what it has drawn, keyed by layer id.
    """
    keeping = {record["bar"] for record in held.values() if record.get("bar")}
    # A layer whose guide asks for **no** bar has settled its claim on this one as surely as a layer holding
    # a key of its own: `colorbar(visible=False)` is an ask, not silence. It has to be read off the plan
    # because asking for no bar leaves no record to read — which is why switching the key off on every
    # terrain sharing one bar left that bar on the window, although one terrain on its own loses it
    # (review M8). A layer carrying no guide at all is in neither set: it is still reading the engine's bar.
    silent = {plan.layer_id for plan in _guide_plan(figure) if plan.wanted_bar is None}
    for layer_id in figure.layers.ids:
        field = guide_field(figure.layers.get(layer_id))
        if not field:
            continue
        engine = _engine_title(scene, layer_id, field)
        if engine not in plotter.scalar_bars or engine in keeping:
            continue
        binding = [layer_id, *_sharing_bar(scene, figure, layer_id, engine)]
        if all(
            held.get(other, {}).get("bar")
            or held.get(other, {}).get("keyed")
            or other in silent
            for other in binding
        ):
            _remove_bar(plotter, engine)


def _redraw_legend(
    scene: Any,
    figure: FigureSpec,
    plotter: Any,
    held: dict[str, dict[str, Any]],
    was_keyed: bool,
) -> None:
    """Rebuild the one keyed list from every layer that asks for one.

    A render window holds a single legend actor, so this is a rebuild rather than an append: every keyed
    layer's rows are derived again and the box is replaced. That is what makes the box follow the layers —
    removing one, hiding one or renaming one's key redraws it without a second mechanism to keep in step.

    Args:
        scene: The scene whose actors the drawn colours are read from.
        figure: The figure being drawn.
        plotter: The plotter to draw on.
        held: The scene's record of what it has drawn, keyed by layer id.
        was_keyed: Whether a keyed list was on the window before this reconcile — read by
            :func:`redraw_guides` **before** it pruned the records of layers that have gone, since a removed
            layer is exactly the case where the box has to come off and nothing is left to say it was there.
    """
    keyed = [
        plan
        for plan in _guide_plan(figure)
        if plan.keyed and plan.shown and plan.scale is not None
    ]
    drawn: list[tuple[str, str]] = []
    location = DEFAULT_LEGEND_LOCATION
    for position, plan in enumerate(keyed):
        actor = _drawn_actor(scene, plan.layer_id)
        scale = plan.scale
        if actor is None or scale is None:
            continue
        if position == 0:
            location = LEGEND_LOCATIONS.get(plan.guide.anchor or "", location)
        spec = LegendSpec.from_scale(
            scale, colors=_drawn_colors(actor, scale), title=plan.title
        )
        drawn.extend(
            _legend_rows(
                spec,
                held.get(plan.layer_id, {}).get("labels"),
                plan.title if len(keyed) > 1 else None,
            )
        )
        # A classified layer's automatic scalar bar reads its class *indices* — `0, 1, 2 …` — so the keyed
        # list replaces it rather than sitting beside it. Only where the engine bound that bar to this layer
        # alone: two layers that bind an array of one name share it.
        engine = _engine_title(scene, plan.layer_id, plan.field)
        if not _sharing_bar(scene, figure, plan.layer_id, engine):
            _remove_bar(plotter, engine)
    contributing = {plan.layer_id for plan in keyed}
    for layer_id, record in held.items():
        record["keyed"] = layer_id in contributing
    if drawn:
        # Replace rather than append: `add_legend` overwrites the plotter's one legend actor, so building the
        # box from every keyed layer at once is what stops the last caller winning.
        plotter.remove_legend(render=False)
        plotter.add_legend(labels=drawn, loc=location)
    elif was_keyed:
        plotter.remove_legend(render=False)


class GuideMixin(_MixinBase):
    """:meth:`colorbar` and :meth:`legend` for a :class:`~digitalearth.three_d.scene3d.Scene3D`.

    Composed into `Scene3D` beside the other capability mixins, so the composed class stays a thin composition
    and the methods live here. Both return `self`, so a key reads as part of the same expression the layer was
    drawn in: `scene.colorbar(label="Elevation (m)").legend("classes")`. The builders themselves return
    the PyVista actor rather than the scene, so a key is a call of its own after the layer.

    Note:
        The class is declared against :class:`~digitalearth.three_d.base.Scene3DBase` under `TYPE_CHECKING`
        only — at runtime the base is `object`, so the composed MRO is what it was before the annotation.

    See Also:
        digitalearth.three_d.decoration.DecorationMixin: the scene's other decoration, which explains no
            encoding and so is not a guide.
    """

    def _record_guide(
        self,
        layer_id: str | None,
        *,
        title: str | None,
        visible: bool,
        caller: str,
        keyed: bool,
        labels: Sequence[str] | None = None,
    ) -> str:
        """Put a guide on one layer's colour encoding, then draw from it.

        The shared half of :meth:`colorbar` and :meth:`legend`: both resolve the layer, check it can carry
        the key they draw, record the guide and let the reconcile draw it. Written once because "record,
        then draw" is the ordering order 24 is about, and two copies of it is two places for the order to
        come apart.

        Args:
            layer_id: The layer to key, or `None` for the most recent colour-driven one.
            title: What to call the key; `None` names it after the field the colour is driven by.
            visible: Whether the key is drawn. Recorded as `Guide(show=...)`, so switching a key off keeps
                what it would have said.
            caller: The method that was called, for the messages.
            keyed: Whether the caller asked for the keyed list (`True`) or the scalar bar (`False`).
            labels: Row labels for the keyed list, recorded as drawn state **after** every check and before
                the reconcile runs — recorded afterwards they would be missing from the box the reconcile
                just drew, and recorded before the checks they would outlive a refused call. `None`
                **clears** labels an earlier call recorded rather than leaving them in place, so one call's
                override does not outlive it; `colorbar()` therefore drops them too, exactly as the static
                tier's `colorbar` does. A list that does not number the layer's rows is refused here, which
                is why "after every check" matters: from the reconcile the figure would roll back and the
                list would stay on the record, refusing every later draw.

        Returns:
            The id of the layer that was keyed.

        Raises:
            KeyError: when `layer_id` names no layer on this scene.
            ValueError: when the layer is not coloured by data, when no layer is and none was named, when
                the layer's colour has the wrong shape for the key asked for, or when `labels` does not
                number the rows the layer contributes. A title that collides with another layer's is
                refused too, from the reconcile the record kicks off rather than from here — see the note
                below this method.
        """
        # Validated before the flag is read, all of it: `colorbar(visible=False)` on a layer with no colour
        # key was accepted while `colorbar()` on the same layer raised, so one spelling was checked only
        # half the time. The web tier's `legend` carries the same fix for its corner keyword (review L7).
        resolved = _resolve_layer(self._figure, layer_id, caller)
        encoding = _colour_encoding(self._figure, resolved)
        if keyed != _keyed(encoding.scale):
            raise ValueError(
                f"{caller}: layer {resolved!r} is coloured by "
                + (
                    "a continuous ramp, which has no classes to list; show it with colorbar()"
                    if keyed
                    else "classes, and a scalar bar over class indices reads '0, 1, 2 …'; "
                    "show them with legend()"
                )
            )
        scale = encoding.scale
        if labels is not None and scale is not None and keyed:
            # Refused before anything is recorded, and before the reconcile runs: a list that does not
            # number the rows leaves swatches unlabelled or labels swatches that are not drawn, and either
            # way the key stops matching the picture. This tier alone took it — a short list kept the
            # derived text for the rows it did not cover and a long one was silently dropped — while
            # static, web and interactive all refuse, which is the asymmetry round 1 removed between the
            # other three (review R2-M5).
            rows = _keyed_row_count(scale)
            given = len(list(labels))
            if given != rows:
                raise ValueError(
                    f"{caller}: labels= has {given} entries and layer {resolved!r} contributes {rows} "
                    f"rows; a short list leaves swatches unlabelled and a long one labels swatches that "
                    f"are not drawn, so either way the key stops matching the picture"
                )
        # Replaced on every call, not only written on the calls that name rows: `None` *clears* an override
        # an earlier call recorded, so one call's rows do not outlive it. Written only when given, a layer
        # whose rows had once been named could never get its derived ranges back (review M9). This is the
        # rule the static tier documents and implements in `Scene._record_key`, and one order cannot mean
        # two things on two tiers.
        if labels is None:
            self._guides.get(resolved, {}).pop("labels", None)
        else:
            self._guides.setdefault(resolved, {})["labels"] = tuple(
                str(label) for label in labels
            )
        symbology = self._figure.layers.get(resolved).symbology.with_guide(
            Guide(show=visible, title=title)
        )
        self.replace_layer(
            with_fields(self._figure.layers.get(resolved), symbology=symbology)
        )
        return resolved

    # There is deliberately **no** guard here walking the other layers' guides to refuse a title one of them
    # already holds. The reconcile refuses the same thing — `_add_bar`'s `wanted in plotter.scalar_bars`,
    # which is the whole of that refusal and the only place it is made — and refuses strictly more: a figure
    # built elsewhere and drawn with `from_figure` never goes through this method at all. (`_redraw_bars`'
    # `taken` map refuses nothing; it is read only to name the layer holding the title, and saying otherwise
    # here contradicted `_add_bar`'s own note — review R2-N5.) Two guards for one
    # rule is two messages for one cause and two places to keep in step —
    # and the net effect is the same, because a refusal inside the reconcile rolls the figure *and* the
    # plotter back to what the scene was already showing (`Scene3DBase._change`).

    def colorbar(
        self,
        layer_id: str | None = None,
        *,
        label: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Show the continuous colour key of a layer — PyVista's scalar bar, as a guide on its encoding.

        The Core contract's `colorbar`, in the shape the web tier already answers to (#299, #261): the same
        name, the same three keywords and the same meaning. What order 24 changes is where the key lives. It
        is recorded on the layer's colour :class:`~digitalearth.base.spec.encoding.Encoding` and **then**
        drawn from that record, so it moves with the layer, hides with it, goes away with it, and comes back
        when the layer's description is drawn again.

        Args:
            layer_id: Which layer's key to show. `None` takes the most recent layer that is **coloured by
                data** — not simply the last layer added, since a label or a flat-coloured mesh drawn after a
                terrain has nothing to key.
            label: What to call the bar — the variable and its units, usually. `None` leaves it titled after
                the array the colour is driven by, which is the title PyVista's own bar already carries.
            visible: `False` draws no bar, so a caller passing a flag through does not have to branch. The
                layer is still resolved and checked first, so one spelling is not valid only half the time.

        Returns:
            This scene, so the call chains.

        Raises:
            KeyError: when `layer_id` names no layer on this scene.
            ValueError: when the named layer is not coloured by data; when none is and none was named; when
                the layer is coloured by **classes**, whose key is :meth:`legend` (a scalar bar over class
                indices reads `0, 1, 2 …`); or when `label` is a title another layer's bar already holds.

        Examples:
            - A terrain's bar, titled, and recorded on the layer that carries it:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> dem = get_source(np.add.outer(np.arange(8.0), np.arange(8.0)))
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(dem)
                >>> _ = scene.colorbar(label="Elevation (m)")
                >>> scene.get_layer("terrain-1").symbology.guide().title
                'Elevation (m)'
                >>> "Elevation (m)" in scene.plotter.scalar_bars
                True
                >>> scene.close()

                ```
            - The key goes away with its layer:
                ```python
                >>> import numpy as np
                >>> from digitalearth.base.sources import get_source
                >>> from digitalearth.three_d import Scene3D
                >>> dem = get_source(np.add.outer(np.arange(8.0), np.arange(8.0)))
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.terrain(dem)
                >>> _ = scene.colorbar(label="Elevation (m)")
                >>> _ = scene.remove_layer("terrain-1")
                >>> list(scene.plotter.scalar_bars.keys())
                []
                >>> scene.close()

                ```

        See Also:
            legend: the keyed list, for a layer coloured by classes.
            digitalearth.web.base.WebMapBase.colorbar: the same name on the web tier.
        """
        self._record_guide(
            layer_id,
            title=label,
            visible=visible,
            caller="colorbar()",
            keyed=False,
        )
        return self

    def legend(
        self,
        layer_id: str | None = None,
        *,
        title: str | None = None,
        labels: Sequence[str] | None = None,
        visible: bool = True,
    ) -> Self:
        """Show the keyed colour list of a layer — PyVista's legend box, as a guide on its encoding.

        The Core contract's `legend`, and `PENDING`'s *"a keyed list beside a scene"*. The rows are derived
        from the layer's own :class:`~digitalearth.base.spec.scale.Scale` through
        :meth:`~digitalearth.base.spec.legend.LegendSpec.from_scale`, and their colours are read back off the
        lookup table the layer is being **drawn** through — so a swatch cannot disagree with the picture,
        which is the whole reason that type exists.

        A render window holds one legend box, so the box is built from **every** layer that asks for one, in
        draw order, rather than the last caller replacing the previous one. Where more than one layer
        contributes, each row is prefixed with that layer's title, since an unprefixed box says nothing about
        which layer a swatch belongs to.

        Args:
            layer_id: Which layer's classes to list. `None` takes the most recent layer coloured by data.
            title: Heading for this layer's rows. `None` uses the column or array the colour is driven by.
            labels: The caller's own row labels, replacing the derived ones — for units, or for renaming
                categories. One per row: a list that does not number the classes this layer contributes is
                refused, as it is on the other three tiers, because a short one leaves swatches unlabelled
                and a long one labels swatches that are not drawn. Recorded as **drawn state** on the scene
                rather than in the figure, exactly as
                :meth:`~digitalearth.three_d.decoration.DecorationMixin.set_title`'s font size and subtitle
                are: the guide is what travels, and the labels are how this tier draws it. `None` clears
                labels an earlier call recorded, bringing the derived ranges or categories back, so one
                call's override does not outlive it.
            visible: `False` draws none of this layer's rows. The layer is resolved and checked first.

        Returns:
            This scene, so the call chains.

        Raises:
            KeyError: when `layer_id` names no layer on this scene.
            ValueError: when the named layer is not coloured by data; when none is and none was named; when
                the layer is coloured by a **continuous ramp**, which has no classes to list — its key is
                :meth:`colorbar`; or when `labels` does not number the rows the layer contributes.

        Examples:
            - A classified point cloud's classes, listed with the colours they were drawn in:
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> points = np.column_stack([np.arange(20.0), np.arange(20.0), np.zeros(20)])
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=4)
                >>> _ = scene.legend(title="Distance")
                >>> scene.plotter.legend.GetNumberOfEntries()
                4
                >>> scene.close()

                ```
            - The caller's own labels replace the derived ranges:
                ```python
                >>> import numpy as np
                >>> from digitalearth.three_d import Scene3D
                >>> points = np.column_stack([np.arange(20.0), np.arange(20.0), np.zeros(20)])
                >>> scene = Scene3D(off_screen=True)
                >>> _ = scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=2)
                >>> _ = scene.legend(labels=["low", "high"])
                >>> scene.plotter.legend.GetEntryString(0)
                'low'
                >>> scene.close()

                ```

        See Also:
            colorbar: the scalar bar, for a layer coloured by a continuous ramp.
            digitalearth.web.decoration.DecorationMixin.legend: the same name on the web tier.
        """
        self._record_guide(
            layer_id,
            title=title,
            visible=visible,
            caller="legend()",
            keyed=True,
            labels=labels,
        )
        return self
