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
  name. It is applied through :func:`redraw_decoration`, which puts the scene's own furniture onto whichever
  plotter it is drawing on, exactly as :meth:`~digitalearth.three_d.base.Scene3DBase._dress_plotter` does for
  the vertical exaggeration and the camera.

**What this module deliberately does not build.** A scalar bar and a legend. Both are Core names, and
`contract.PENDING["3d"]` schedules both against order 24 — "the scalar bar is PyVista's, and becomes a guide on
the encoding", "a keyed list beside a scene" — so `colorbar()` and `legend()` are that order's work, under those
spellings, keyed to a layer's `Encoding`. Adding a differently shaped `scalar_bar()` here would put two
spellings on one concept, which is the single thing the Core contract exists to prevent. A north arrow is not
built either: the tier declares it absent, with its reason (a scene can be looked at from any direction, so
there is no fixed north on screen), and PyVista has no `add_north_arrow` to wire to anyway.
"""

import math
from dataclasses import replace as with_fields
from typing import TYPE_CHECKING, Any, Dict, Optional, Self, Tuple

import numpy as np

from digitalearth.base.spec import LayerSpec
from digitalearth.base.spec._serial import crs_to_json
from digitalearth.base.spec.bounds import same_crs
from digitalearth.three_d.layer import drawing_props

__all__ = ["DecorationMixin", "draw_text", "redraw_decoration"]

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
) -> Optional[Tuple[float, float]]:
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


def _draw_title(plotter: Any, heading: str, style: Dict[str, Any]) -> None:
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


def draw_text(scene: Any, _data: Any, layer: LayerSpec) -> Optional[Tuple[Any, Any]]:
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
        subtitle: Optional[str] = None,
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

    def text(
        self,
        lon: Any,
        lat: Any,
        s: Optional[str] = None,
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
