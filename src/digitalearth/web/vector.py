"""VectorMixin — web-tier vector builders (DW.2, recipe W2).

``points`` / ``lines`` / ``polygons`` turn a pyramids ``FeatureCollection`` — or a path or URL naming one,
which is what lets the figure be written down — (reprojected to lon/lat through pyramids) into MapLibre
circle / line / fill layers over a GeoJSON source; ``choropleth`` is the thematic
polygon map. Colour-by-value compiles into a MapLibre **data-driven paint expression**:

* graduated (``scheme`` set) → a ``["step", ["get", col], …]`` expression whose breaks come from
  **cleopatra.styling.styles.classify** (pure-numpy quantiles / equal-interval / Fisher-Jenks — no mapclassify), the
  same classifier the static tier uses, so the classes match across tiers;
* continuous (``scheme=None``) → an ``["interpolate", ["linear"], ["get", col], …]`` colour ramp.

cleopatra / matplotlib / numpy are imported lazily inside the methods; importing the tier needs none of them.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from loguru import logger

from digitalearth.base.ask import UNSET, Ask, Maybe, asked_record
from digitalearth.base.spec import (
    DEFAULT_BAND,
    DEFAULT_RAMP_STOPS,
    Encoding,
    LayerSpec,
    LegendSpec,
    Scale,
    Symbology,
)
from digitalearth.base.symbology import (
    DEFAULT_LABEL_COLOR,
    DEFAULT_LABEL_HALO_COLOR,
    DEFAULT_LABEL_HALO_WIDTH,
    DEFAULT_LABEL_TEXT_SIZE,
)
from digitalearth.web.base import _require_layer_api, as_finite, placed_features


@dataclass(frozen=True)
class ContourInterval:
    """Regularly spaced contour levels: one every `spacing`, anchored at `base`.

    `base` is only meaningful for a regular interval — it is the value the spacing counts from, so
    `ContourInterval(100, base=50)` traces 50, 150, 250 rather than 0, 100, 200. Pairing the two in one type
    is what keeps that from being said where it cannot be honoured: an explicit `levels=[...]` list has no
    spacing to anchor, and a loose `base=` alongside it used to be accepted and silently dropped.

    Attributes:
        spacing: Distance between successive levels, in the band's own units. Must be finite and positive.
        base: The value the spacing is anchored to. Defaults to 0.

    Examples:
        - Trace a level every 100 units, starting from zero:
            ```python
            >>> from digitalearth.web.vector import ContourInterval
            >>> every_hundred = ContourInterval(100)
            >>> every_hundred.spacing, every_hundred.base
            (100, 0.0)

            ```
        - Anchor the same spacing at 50, so the levels land on 50, 150, 250:
            ```python
            >>> from digitalearth.web.vector import ContourInterval
            >>> offset = ContourInterval(100, base=50)
            >>> offset.base
            50

            ```
        - A spacing of zero has no levels to give and is refused:
            ```python
            >>> from digitalearth.web.vector import ContourInterval
            >>> ContourInterval(0)
            Traceback (most recent call last):
                ...
            ValueError: contour spacing must be finite and greater than zero; got 0

            ```
    """

    spacing: float
    base: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a spacing that cannot produce levels.

        Raises:
            ValueError: when `spacing` is zero, negative, or not finite — each traces nothing, and pyramids
                reports it as an empty result rather than as the bad argument it was.
        """
        if not math.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError(
                f"contour spacing must be finite and greater than zero; got {self.spacing!r}"
            )


@dataclass(frozen=True)
class VectorTileSource:
    """A MapLibre ``vector`` source — the tile set a :meth:`VectorMixin.vector_tiles` layer draws from.

    The five fields that name the source — the tile template, the TileJSON URL and the service's coverage —
    are born together, travel together into one source dict and die together, so they are one value rather
    than five parameters threaded through the builder. The tiles-xor-url rule is theirs too: a vector source
    is reached by **exactly one** of ``tiles`` or ``url``, and MapLibre reads a template and a TileJSON as the
    same thing said twice — so the pairing is refused at construction, before a builder is ever called.

    Attributes:
        tiles: An MVT tile URL template carrying ``{z}/{x}/{y}``, or a sequence of them, or ``None`` when the
            set is named by `url` instead.
        url: A TileJSON URL the service publishes, or ``None`` when `tiles` names the set instead.
        min_zoom: The shallowest zoom the service serves, or ``None`` to leave the source unbounded.
        max_zoom: The deepest zoom the service serves — past it MapLibre over-zooms the last real tiles rather
            than requesting levels that do not exist — or ``None``.
        attribution: Attribution text shown in the map's attribution control; empty adds none.

    Raises:
        ValueError: when neither `tiles` nor `url` is given, or both are — a vector source is reached by
            exactly one of them.

    Examples:
        - A ``{z}/{x}/{y}`` template becomes a ``vector`` source listing that one template:
            ```python
            >>> from digitalearth.web.vector import VectorTileSource
            >>> VectorTileSource(tiles="https://tiles.example.org/{z}/{x}/{y}.pbf").to_source()
            {'type': 'vector', 'tiles': ['https://tiles.example.org/{z}/{x}/{y}.pbf']}

            ```
        - A TileJSON ``url`` and a zoom range fold into the same dict:
            ```python
            >>> from digitalearth.web.vector import VectorTileSource
            >>> VectorTileSource(url="https://tiles.example.org/roads.json", max_zoom=14).to_source()
            {'type': 'vector', 'url': 'https://tiles.example.org/roads.json', 'maxzoom': 14}

            ```
        - Naming neither tile set is refused, at construction rather than at the builder:
            ```python
            >>> from digitalearth.web.vector import VectorTileSource
            >>> VectorTileSource()
            Traceback (most recent call last):
                ...
            ValueError: vector_tiles() takes exactly one of tiles= (a {z}/{x}/{y} template)...

            ```
    """

    tiles: Any = None
    url: str | None = None
    min_zoom: int | None = None
    max_zoom: int | None = None
    attribution: str = ""

    def __post_init__(self) -> None:
        """Refuse a source named by neither tile set or by both.

        Raises:
            ValueError: when neither `tiles` nor `url` is given, or both are — a vector source is reached by
                exactly one of them, and MapLibre reads a template and a TileJSON as the same thing said
                twice.
        """
        if (self.tiles is None) == (self.url is None):
            raise ValueError(
                "vector_tiles() takes exactly one of tiles= (a {z}/{x}/{y} template) or url= (a TileJSON "
                f"URL); got tiles={self.tiles!r} and url={self.url!r}"
            )

    def to_source(self) -> dict:
        """Build the MapLibre ``vector`` source dict, naming the tile set exactly one way.

        Returns:
            The source dict :func:`draw_vector_tiles` builds the MapLibre source from — plain JSON, so it is
            written straight into a saved figure.
        """
        source: dict = {"type": "vector"}
        if self.url is not None:
            source["url"] = self.url
        else:
            source["tiles"] = (
                [self.tiles]
                if isinstance(self.tiles, str)
                else [str(tile) for tile in self.tiles]
            )
        if self.min_zoom is not None:
            source["minzoom"] = int(self.min_zoom)
        if self.max_zoom is not None:
            source["maxzoom"] = int(self.max_zoom)
        if self.attribution:
            source["attribution"] = self.attribution
        return source


#: How many levels `contours` traces when the caller names none and the band's variable carries none. Ten,
#: which is what the interactive tier has always fallen back to and what matplotlib's own `levels=10` means,
#: so the same call describes the same number of levels on either tier (#262).
DEFAULT_CONTOUR_LEVELS = 10


def _even_levels(source: Any, count: int) -> list:
    """Return ``count`` levels evenly spaced strictly inside a band's finite range.

    **Strictly inside** on purpose: a level sitting exactly on the minimum or the maximum traces the frame's
    edge or nothing at all, so `count` levels between the two extremes is `count` contours a reader can see.
    That is also what matplotlib's own `levels=<int>` produces in effect, and what the interactive tier's
    fallback hands HoloViews.

    Args:
        source: The display-CRS :class:`~digitalearth.base.sources.source.Source` for the band being traced.
        count: How many levels to cut the range into.

    Returns:
        The levels, as plain floats so a figure can be written down with them.

    Raises:
        ValueError: when the band has no finite values, or they are all the same — there is genuinely
            nothing to contour, and guessing a level would draw a line the data does not support.
    """
    import numpy as np

    values = np.asarray(source.z.values, dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(finite.min()) == float(finite.max()):
        raise ValueError(
            "contours() has no range to cut into levels: the band is empty or constant. Pass interval= or "
            "levels= if the values are meaningful, or contour a band that varies."
        )
    edges = np.linspace(float(finite.min()), float(finite.max()), count + 2)
    return [float(level) for level in edges[1:-1]]


def _as_interval(
    interval: float | ContourInterval | None,
) -> ContourInterval | None:
    """Read the `interval=` argument in either spelling, so a plain number keeps meaning "every N".

    Args:
        interval: A bare spacing, a :class:`ContourInterval` carrying its own `base`, or `None`.

    Returns:
        The interval as one type, or `None` when the caller gave none.

    Raises:
        TypeError: when `interval` is neither a real number nor a :class:`ContourInterval`. `bool` is
            rejected with it: `interval=True` is a mis-typed flag, not a spacing of one.
    """
    if interval is None or isinstance(interval, ContourInterval):
        return interval
    if isinstance(interval, bool) or not isinstance(interval, (int, float)):
        raise TypeError(
            f"contours() takes interval= as a number or a ContourInterval; got {interval!r}"
        )
    return ContourInterval(interval)


@dataclass(frozen=True)
class _ContourLevels:
    """The iso-values one `contours()` call traces, held with the spelling its caller asked in.

    `interval=` and `levels=` are two spellings of a single ask, and the pair used to travel the whole
    length of :meth:`~digitalearth.web.vector.VectorMixin.contours` as loose locals: normalised, handed
    to pyramids as three separate keywords, and — when the trace comes back empty — quoted back to the
    caller in whichever spelling they wrote. Every one of those readings was another conditional on the
    same two variables, in a method that is about drawing contours rather than about pyramids' argument
    protocol. Here they are one value that already knows which arm it is, and the two readings that are
    only ever taken together with it are its own.

    Attributes:
        spacing: The regular interval the levels are cut at, or `None` when they are named explicitly.
        fixed: The settled explicit levels, or `None` when `spacing` decides them instead. Exactly one of
            the two is set: :meth:`~digitalearth.web.vector.VectorMixin._contour_levels` refuses the pair,
            and falls back on evenly spaced levels rather than settling on neither.
        asked: The caller's own `interval=` argument, kept as they passed it — a message quotes
            `interval=100` as it was written, not as the :class:`ContourInterval` it normalises to.
    """

    spacing: ContourInterval | None
    fixed: Any | None
    asked: float | ContourInterval | None = None

    @property
    def asked_as(self) -> str:
        """Return this ask in the caller's own words, for a message that names what they wrote.

        Returns:
            `interval=<the argument they passed>`, or `levels=<the levels settled on>` — the settled ones
            rather than the argument, because the argument is `None` for every auto-resolved trace and the
            reader needs to see which levels were actually looked for.
        """
        if self.spacing is None:
            return f"levels={self.fixed!r}"
        return f"interval={self.asked!r}"

    def trace(self, dataset: Any, *, band: int, polygonize: bool) -> Any:
        """Trace these levels out of one band of `dataset` (pyramids' `Dataset.contour`).

        Args:
            dataset: The pyramids `Dataset` to contour.
            band: 1-based band to contour, as this tier counts bands.
            polygonize: Trace the bands between successive levels as polygons rather than the levels as
                lines.

        Returns:
            The `FeatureCollection` pyramids traced — carrying `level` for lines, and
            `level_min`/`level_max` for bands. Empty when no level falls inside the band's range.
        """
        if self.spacing is None:
            every, base = None, 0.0
        else:
            every, base = self.spacing.spacing, self.spacing.base
        return dataset.contour(
            interval=every,
            fixed_levels=None if self.fixed is None else list(self.fixed),
            base=base,
            band=int(band) - 1,  # pyramids counts bands from 0; this tier counts from 1
            attribute="level",
            polygonize=polygonize,
        )


#: Flat colour a vector layer falls back on when the caller pins neither `color=` nor a value column.
#: It used to be repeated as a parameter default as well, so that it showed in a rendered signature; a style
#: keyword's default is now :data:`~digitalearth.base.ask.UNSET`, so the builder can tell an ask from its own
#: default (#334), and this is the one place the value itself lives.
VECTOR_COLOR = "#3388ff"

#: What an unstyled layer is drawn at, one name per builder and channel.
#:
#: These are the values that used to sit in the signatures. They moved here rather than disappearing, because
#: a signature reading ``size: Maybe[float] = UNSET`` no longer shows the number and the number is still this
#: tier's answer: each builder's docstring cites the constant by name, and a default and the value something
#: else claims for it can no longer drift, because there is no second copy left to drift from.
POINT_SIZE = 5.0
POINT_OPACITY = 0.9
LINE_WIDTH = 2.0
LINE_OPACITY = 1.0
CONTOUR_WIDTH = 1.5
CONTOUR_OPACITY = 1.0
FILL_OPACITY = 0.6
CHOROPLETH_OPACITY = 0.85

#: Outline of a filled contour band — `polygons()`'s own default outline, which is what filled contours were
#: drawn with while they were drawn through it.
CONTOUR_OUTLINE_COLOR = "#ffffff"

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def draw_vector(web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the MapLibre source and typed layer for any of the vector kinds.

    All seven — points, lines, polygons, choropleth, labels, and the two contour kinds — are a GeoJSON
    source plus one typed layer, differing only in the MapLibre type and the paint the builder resolved. They
    share a drawer for the same reason they share a registration funnel.

    Args:
        web_map: The map being drawn, whose display CRS the geometry is placed in.
        data: The layer's source, served as GeoJSON — the display-CRS frame the builder already holds, or
            whatever the figure's reference opened to when the layer is being drawn back from a
            description.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`. It never declines: any of these kinds either
        draws or refuses by raising, which is why their builders do not ask whether the layer survived.

    Raises:
        ValueError: when the description records no MapLibre type or no paint for the layer, naming the
            layer rather than the missing MapLibre key.
        TypeError: when the figure's source is not a vector layer, and OffLimbError when the warp
            places none of its geometry and the map is `strict` — both from :func:`placed_features`,
            which places the data when the drawer is handed nothing already placed.
    """
    from digitalearth.web.renderer import DrawnLayer, required_props

    layer_cls, _ = _require_layer_api()
    props = required_props(layer, "maplibre_type", "paint")
    spec_layout = dict(props.get("layout") or {})
    if not layer.visible:
        spec_layout["visibility"] = "none"
    source_id = f"{layer.id}-src"
    return DrawnLayer(
        source_id=source_id,
        source_spec=placed_features(web_map, data, layer),
        layer=layer_cls(
            id=layer.id,
            type=props[
                "maplibre_type"
            ],  # MapLibre coerces the string back to its own enum
            source=source_id,
            paint=dict(props["paint"]),
            layout=spec_layout or None,
        ),
    )


def draw_vector_tiles(_web_map: Any, _data: Any, layer: LayerSpec) -> Any:
    """Build the MapLibre vector source and the typed layer for an MVT tile layer (WB-6).

    A vector-tile layer draws from no data in the figure: its source is the tile URL the caller passed, and
    the one named layer to read out of the pyramid, its geometry's MapLibre type and the paint are all values
    recorded on its symbology — which is why `_data` is unused. They are plain JSON, so a map whose layer is
    a tile set describes itself in a figure that can be written down and read back, exactly as a raster
    basemap does.

    Args:
        _web_map: Unused — every drawer takes the map, and this one draws without it.
        _data: Unused — a tile layer has no feature source in the figure to place.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer` holding the ``vector`` source and the circle / line
        / fill layer reading one of its source-layers. It is an ordinary style layer, so it takes the
        ordinary route.

    Raises:
        ValueError: when the description carries none of the MapLibre source, the source-layer, the layer
            type or the paint the drawer reads — naming the layer, its kind and what is missing.
    """
    from digitalearth.web.renderer import DrawnLayer, required_props

    layer_cls, _ = _require_layer_api()
    props = required_props(layer, "source", "source_layer", "maplibre_type", "paint")
    spec_layout = dict(props.get("layout") or {})
    if not layer.visible:
        spec_layout["visibility"] = "none"
    source_id = f"{layer.id}-src"
    return DrawnLayer(
        source_id=source_id,
        source_spec=dict(props["source"]),
        layer=layer_cls(
            id=layer.id,
            type=props[
                "maplibre_type"
            ],  # MapLibre coerces the string back to its own enum
            source=source_id,
            # The vector source holds many named layers; this is the one to draw. MapLibre takes it under
            # `source-layer`, which the `maplibre` Layer serialises `source_layer` to.
            source_layer=props["source_layer"],
            paint=dict(props["paint"]),
            layout=spec_layout or None,
        ),
    )


class VectorMixin(_MixinBase):
    """Point / line / polygon / choropleth builders for :class:`~digitalearth.web.map.WebMap`.

    A capability mixin of :class:`~digitalearth.web.map.WebMap`: it is only ever composed into that map class, never
    instantiated or subclassed on its own. Its methods reach the layer registry, the display CRS and the render/save
    lifecycle — and the sibling mixins' methods — through ``self``, and only the composition supplies those.

    The ``if TYPE_CHECKING`` base declared above the class is what records that contract for a type checker: it
    resolves each ``self.<attr>`` against :class:`~digitalearth.web.base.WebMapBase`, the state ``WebMap`` inherits.
    At runtime that base is plain ``object``, so composing this mixin leaves the ``WebMap`` MRO exactly what it was
    before the annotation.

    See Also:
        digitalearth.web.map.WebMap: the composition that supplies the state these methods use.
        digitalearth.web.base.WebMapBase: the typing-only base declared above the class.
    """

    def _color_expr(
        self,
        values: Any,
        column: str,
        scheme: Any | None,
        k: int,
        cmap: str,
    ) -> tuple[list, Encoding]:
        """Compile a MapLibre data-driven colour expression for ``column`` and record the breaks.

        Args:
            values: The 1-D value array driving the colour (used to compute breaks / limits).
            column: The GeoJSON property name the expression reads with ``["get", column]``.
            scheme: A cleopatra classification scheme (``"quantiles"``/``"fisher_jenks"``/… or an explicit
                edge sequence) for a graduated ``step`` expression; ``"categorical"`` for a distinct-value
                ``match`` expression (DC.8); ``None`` for a continuous ramp.
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap name sampled for the class / ramp colours (a qualitative map such as
                ``"tab10"`` is used for ``scheme="categorical"`` when ``cmap`` is left at the default).

        Returns:
            ``(expression, encoding)``. The expression is MapLibre's own (``["step", …]``, ``["match", …]``
            or ``["interpolate", …]``) and is what the drawer paints with. The encoding is the same colour
            said portably —
            :meth:`~digitalearth.base.spec.encoding.Encoding.by_field` on the ``color`` channel, carrying
            the very :class:`~digitalearth.base.spec.scale.Scale` the arm classified with — and is what a
            :class:`~digitalearth.base.spec.encoding.Guide` is later hung on, so a colour key belongs to
            its layer instead of to the map (DE-48, order 24).

            Both are **returned** rather than one of them left on ``self``: a caller that never reaches
            `_vector_layer` would otherwise leave an encoding behind for the next layer to pick up, which
            is the class of bug the ``_filed_legend`` identity dance exists to work around.

            Also sets ``self.last_breaks`` to the breaks (graduated edges), the categories, or the ramp
            stops, and ``self.last_legend`` to those values *plus the colours they were drawn with*, which
            is what :meth:`~digitalearth.web.decoration.DecorationMixin.legend` renders.

        Raises:
            ValueError: from :class:`~digitalearth.base.spec.scale.Scale` when the scheme cannot classify
                the column (unknown scheme, no spread, …), naming the scheme and ``k``; or from the
                categorical helper (no non-null values).
        """
        if isinstance(scheme, str) and scheme.lower() == "categorical":
            return self._categorical_color_expr(values, column, cmap)
        if scheme is not None:
            return self._graduated_color_expr(values, column, scheme, k, cmap)
        return self._ramp_color_expr(values, column, cmap)

    def _categorical_color_expr(
        self, values: Any, column: str, cmap: str
    ) -> tuple[list, Encoding]:
        """Compile a MapLibre ``match`` expression over the column's distinct values (DC.8).

        Args:
            values: The 1-D value array driving the colour.
            column: The GeoJSON property the expression reads.
            cmap: The colormap sampled for the class colours; a qualitative map is resolved for it.

        Returns:
            ``(expression, encoding)`` — the ``["match", …]`` expression, and the colour channel bound to
            ``column`` through the same categorical `Scale` the swatches come from — with `last_breaks` and
            `last_legend` recorded.

        Raises:
            ValueError: for a non-integral float category, which cannot key a ``match``; or from the
                categorical helper when the column holds no non-null values.
        """
        import numpy as np

        from digitalearth.base.symbology import MISSING_COLOR

        def _native(value: Any) -> Any:
            """Coerce a category to a JSON-native, MapLibre-legal ``match`` label.

            MapLibre rejects non-integer numeric ``match`` labels ("Numeric branch labels must be integer
            values"), so a whole-valued float (integer class codes stored as ``float64`` — the common shape
            for codes read from GeoJSON) is narrowed to ``int``; a genuinely non-integral float cannot key a
            ``match`` and is rejected with a clear error. Integers and strings pass through unchanged.
            """
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, (np.floating, float)):
                as_float = float(value)
                if not as_float.is_integer():
                    raise ValueError(
                        f"categorical column {column!r} has a non-integer category {as_float!r}; a "
                        f"categorical scheme needs discrete integer codes or string labels"
                    )
                return int(as_float)
            return value

        from digitalearth.base.symbology import (
            categorical_colors,
            resolve_categorical_cmap,
        )

        categories, colors = categorical_colors(values, resolve_categorical_cmap(cmap))
        expr = ["match", ["get", column]]
        for category, color in zip(categories, colors):
            expr.extend([_native(category), color])
        expr.append(
            MISSING_COLOR
        )  # fallback for values outside the known categories (shared by all tiers)
        # Derived from the scale that produced the colours, not assembled a second time beside it:
        # that is what makes the swatches equal what was drawn (#185, DE-19).
        scale = Scale.categorical([_native(c) for c in categories], list(colors))
        legend = LegendSpec.from_scale(scale, title=column)
        self.last_breaks = [_native(c) for c in categories]
        self.last_legend = self._legend_dict(legend, column)
        # The same scale a third time would be a third chance to disagree, so the portable encoding is
        # bound to the object the swatches and the `match` arms were both built from. No `output_range`:
        # `Encoding` refuses one on a categorical scale, and a fill's colour has no numeric range anyway.
        return expr, Encoding.by_field("color", column, scale=scale)

    def _graduated_color_expr(
        self, values: Any, column: str, scheme: Any, k: int, cmap: str
    ) -> tuple[list, Encoding]:
        """Compile a MapLibre ``step`` expression over class edges.

        Args:
            values: The 1-D value array driving the colour.
            column: The GeoJSON property the expression reads.
            scheme: A cleopatra classification scheme, or an explicit edge sequence.
            k: The number of classes.
            cmap: The colormap sampled for the class colours.

        Returns:
            ``(expression, encoding)`` — the guarded ``["case", …, ["step", …], MISSING_COLOR]`` expression,
            and the colour channel bound to ``column`` through the same `Scale` the class ranges come from —
            with `last_breaks` and `last_legend` recorded.

        Raises:
            ValueError: when the scheme cannot classify the column, naming the column as well as the
                scheme and `k` that `Scale` already names.
        """
        from digitalearth.base.symbology import MISSING_COLOR

        try:
            edges = Scale.breaks_of(values, scheme, k)
        except (
            ValueError
        ) as err:  # constant / single-feature column, unknown scheme, k<1, …
            # Scale's own message already names the scheme and `k`; this adds the column, which it
            # cannot know. Repeating scheme/k here printed both twice in a row.
            raise ValueError(f"cannot classify column {column!r}: {err}") from err
        colors = self._cmap_hex(cmap, len(edges) - 1)
        step: list = ["step", ["get", column], colors[0]]
        for edge, color in zip(edges[1:-1], colors[1:]):
            step.extend([float(edge), color])
        # A feature with no value must read as missing, not as the lowest class: MapLibre's `step`
        # has no null arm, so it is guarded by a type test, exactly as the categorical `match` above
        # falls back to MISSING_COLOR. Every tier draws an unclassifiable feature this same grey.
        expr: list = [
            "case",
            ["==", ["typeof", ["get", column]], "number"],
            step,
            MISSING_COLOR,
        ]
        self.last_breaks = [float(e) for e in edges]
        # Through from_scale, like the categorical arm: building the rows by hand here was the same
        # assembly this wave set out to remove, routed through an extra type for no new guarantee — and
        # it re-spelled the range label a fourth time. Going through the Scale exercises class_ranges()'s
        # k-classes-from-k+1-edges arithmetic in production rather than only in its own test.
        scale = Scale(
            self.last_breaks[0],
            self.last_breaks[-1],
            scheme=scheme,
            breaks=tuple(self.last_breaks),
        )
        self.last_legend = self._legend_dict(
            LegendSpec.from_scale(scale, colors=list(colors), title=column),
            column,
            values=self.last_breaks,
        )
        # The portable reading of the same fill: the class edges the `step` arms were built from, carried on
        # the channel they drive rather than only in `last_breaks`.
        return expr, Encoding.by_field("color", column, scale=scale)

    def _ramp_color_expr(
        self, values: Any, column: str, cmap: str
    ) -> tuple[list, Encoding]:
        """Compile a MapLibre ``interpolate`` expression over a continuous ramp.

        Args:
            values: The 1-D value array driving the colour.
            column: The GeoJSON property the expression reads.
            cmap: The colormap sampled for the ramp stops.

        Returns:
            ``(expression, encoding)`` — the ``["interpolate", …]`` expression, and the colour channel bound
            to ``column`` over the same limits the ramp was sampled across — with `last_breaks` and
            `last_legend` recorded.

        Raises:
            ValueError: when the column holds no finite values to colour.
        """
        import numpy as np

        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"column {column!r} has no finite values to colour")
        lo, hi = Scale.from_finite(finite).as_limits()
        # The shared count, not a literal: the raster band's ramp
        # (:meth:`~digitalearth.web.raster.RasterMixin._band_colour`) samples at the same one, so a
        # continuous key reads the same whichever the values came from. Both used to spell `5` separately,
        # which made the invariant a coincidence either could break in silence (review N6).
        stops = np.linspace(lo, hi, DEFAULT_RAMP_STOPS)
        colors = self._cmap_hex(cmap, len(stops))
        expr = ["interpolate", ["linear"], ["get", column]]
        for stop, color in zip(stops, colors):
            expr.extend([float(stop), color])
        self.last_breaks = [float(s) for s in stops]
        # The stops are handed over rather than recomputed from the limits: np.linspace pins its last
        # element to `hi` exactly and the arithmetic in `from_scale` does not, so for lo=-3.7, hi=12.9 the
        # top swatch was labelled 12.900000000000002 while the ramp drew 12.9. The legend must be the stops
        # that were drawn, not a second computation that usually agrees with them.
        scale = Scale.from_limits(lo, hi)
        self.last_legend = self._legend_dict(
            LegendSpec.from_scale(
                scale,
                colors=list(colors),
                title=column,
                values=self.last_breaks,
            ),
            column,
        )
        # The limits, not the five sampled stops: the stops are this tier's own sampling of the ramp — the
        # gradient bar is drawn from them — while the encoding says what the colour varies *with*, which
        # another engine ramps at whatever resolution it likes.
        return expr, Encoding.by_field("color", column, scale=scale)

    def labels(
        self,
        features: Any,
        column: str,
        *,
        text_size: float = DEFAULT_LABEL_TEXT_SIZE,
        color: Maybe[str] = UNSET,
        halo_color: str = DEFAULT_LABEL_HALO_COLOR,
        halo_width: float = DEFAULT_LABEL_HALO_WIDTH,
        offset: Any | None = None,
        allow_overlap: bool = False,
        name: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Label features with the text in ``column`` (recipe W2).

        Labels are how a map says what is on it, and MapLibre's symbol layer does the work — data-driven
        text, collision detection, halos and placement. None of it was reachable: the only symbol layer the
        tier built was the count inside ``cluster``.

        Args:
            features: A pyramids ``FeatureCollection`` or GeoDataFrame; points label at the point, lines
                and polygons at a placement MapLibre picks. A path or URL to one is taken too, and is the
                only input this layer can be written down with — a pyramids object does not know where it
                came from. The reference is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what the
                figure records.
            column: The property to read the text from.
            text_size: Text size in pixels. Named for the text rather than ``size``, which means the
                visual size of a marker everywhere else.
            color: Text colour; not passed leaves
                :data:`~digitalearth.base.symbology.DEFAULT_LABEL_COLOR`, the same constant the static
                tier's ``labels`` reads (#345).
            halo_color: Colour of the outline drawn behind the glyphs, which is what keeps a label legible
                over imagery.
            halo_width: Halo width in pixels; ``0`` disables it.
            offset: ``(x, y)`` offset in ems, e.g. ``(0, -1.2)`` to lift a label off its point.
            allow_overlap: Whether labels may overlap. ``False`` (the default) lets MapLibre drop labels
                that collide, which is what keeps a dense layer readable.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: when ``features`` is not a vector layer.
            KeyError: when ``column`` is not one of its properties — a MapLibre expression reading a
                missing property renders nothing at all, with no error to explain the empty map — or when
                ``features`` is a URL with no resolver registered for its scheme.
            ValueError: when ``text_size`` or ``halo_width`` is not a finite number, refused at this call
                because a figure holding NaN or infinity could not be written down.
            FileNotFoundError: when ``features`` is a path that names nothing.

        Examples:
            - Name each feature:
                ```python
                >>> from digitalearth.web import WebMap                          # doctest: +SKIP
                >>> WebMap().basemap().polygons(gdf).labels(gdf, "name")         # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.decoration.DecorationMixin.text: a single annotation at a coordinate.
        """
        #: This builder's own name, for the refusals below to quote back at the caller.
        call = "WebMap.labels()"
        _, layer_types = _require_layer_api()
        ask = Ask()
        text_size = as_finite(text_size, "text_size", call)
        halo_width = as_finite(halo_width, "halo_width", call)
        gdf = self._display_gdf(features, method="labels")
        if column not in getattr(gdf, "columns", []):
            raise KeyError(
                f"labels(column={column!r}) is not a property of these features; available: "
                f"{sorted(c for c in getattr(gdf, 'columns', []) if c != 'geometry')}"
            )
        layout: dict = {
            "text-field": ["get", column],
            "text-size": float(text_size),
            "text-allow-overlap": bool(allow_overlap),
        }
        if offset is not None:
            layout["text-offset"] = [float(value) for value in offset]
        paint = {
            "text-color": ask("text-color", color, DEFAULT_LABEL_COLOR),
            "text-halo-color": halo_color,
            "text-halo-width": float(halo_width),
        }
        return self._vector_layer(
            gdf,
            "label",
            layer_types.SYMBOL,
            paint,
            kind="labels",
            name=name,
            visible=visible,
            layout=layout,
            source=features,
            asked=ask.named,
        )

    def _contour_levels(
        self, source: Any, *, interval: ContourInterval | None, levels: Any | None
    ) -> Any | None:
        """Settle which iso-values `contours` traces, refusing the two ways of asking that cannot combine.

        Args:
            source: The display-CRS `Source` for the band being traced, consulted for an auto-style default.
            interval: The caller's spacing, already resolved to a `ContourInterval`, or `None`.
            levels: The caller's explicit level list, or `None` to fall back on the variable's own levels.

        Returns:
            The levels to trace, or `None` when `interval` decides the spacing instead.

        Raises:
            ValueError: when both `interval` and `levels` were given, or when neither was and the band's
                variable is not one `auto_style` carries levels for.
        """
        if interval is not None:
            if levels is not None:
                raise ValueError(
                    "contours() takes at most one of interval= or levels=; "
                    f"got interval={interval!r} and levels={levels!r}"
                )
            return None
        resolved = self._auto_levels(source, levels)
        if resolved is None:
            # The tier used to refuse here, while the interactive tier fell back to ten from the same call —
            # one call, two answers, from tiers a caller is told are interchangeable (#262). Ten evenly
            # spaced levels is what that tier has always done and what matplotlib's own `levels=10` means,
            # so falling back is the tiers agreeing rather than a third behaviour.
            resolved = _even_levels(source, DEFAULT_CONTOUR_LEVELS)
        return resolved

    def _draw_contour_features(
        self,
        features: Any,
        *,
        filled: bool,
        column: str | None,
        cmap: str,
        color: str | None,
        width: float,
        opacity: float,
        name: str | None,
        visible: bool,
        asked: Sequence[str] = (),
    ) -> None:
        """Describe the traced features as one contour layer: the levels as lines, or the bands between as fills.

        The layer is recorded under its own kind — `contours` or `filled_contours` — when it is drawn, and
        :func:`draw_vector` draws it back from that description like any other vector kind (review L3). It
        used to be drawn through `lines`/`polygons` and relabelled afterwards, which reached for whatever
        `_last_layer_id` named: a flat fill over the big-data threshold routed to a deck.gl overlay that
        records no layer, so the relabelling found none, or renamed an earlier layer. A contour layer is
        therefore always the MapLibre layer it describes, whatever the threshold.

        Args:
            features: The `FeatureCollection` pyramids traced.
            filled: Draw the bands between levels as polygons rather than the levels as lines.
            column: Attribute to colour by, or `None` when an explicit `color` was given.
            cmap: Colormap for the continuous ramp over the attribute, already resolved by `_auto_cmap`.
            color: A single flat colour; `None` takes the tier's default contour blue.
            width: Line width, used only by the line branch.
            opacity: Layer opacity in `[0, 1]`.
            name: What a layer switcher calls the layer.
            visible: Whether the layer starts visible.
            asked: The paint keys :meth:`contours` recorded the caller as having named, in MapLibre's own
                spelling. Threaded rather than derived here, because `contours` is where the keyword was
                seen and only the caller's own call can say whether it was passed.

        Raises:
            KeyError: naming ``column`` when the traced features carry no such attribute (see
                :meth:`_require_column`). The layer is dropped from the description again before it
                propagates, so a refused trace leaves the map with no layer and no source.
            ImportError: when the `web` extra is not installed, so there is no MapLibre layer API to
                describe the layer against.
        """
        _, layer_types = _require_layer_api()
        gdf = self._display_gdf(features, method="contours")
        # The level colours each contour along a continuous ramp, as `lines`/`polygons` do with no scheme;
        # a caller's `color=` pins every contour to one colour instead — and a flat colour drives nothing, so
        # there is no field-driven encoding to publish and no colour key to be asked for on it.
        colour: Any = color or VECTOR_COLOR
        color_encoding: Encoding | None = None
        if column is not None:
            colour, color_encoding = self._ramp_color_expr(
                self._require_column(gdf, column), column, cmap
            )
        if filled:
            paint = self._fill_paint(opacity, CONTOUR_OUTLINE_COLOR, colour)
            prefix, layer_type, kind = "fill", layer_types.FILL, "filled_contours"
        else:
            paint = self._line_paint(width, opacity, colour)
            prefix, layer_type, kind = "line", layer_types.LINE, "contours"
        self._vector_layer(
            gdf,
            prefix,
            layer_type,
            paint,
            kind=kind,
            name=name,
            visible=visible,
            asked=asked,
            color_encoding=color_encoding,
        )

    @staticmethod
    def _line_paint(width: float, opacity: float, colour: Any) -> dict:
        """Return a MapLibre line layer's paint.

        Args:
            width: Line width in pixels.
            opacity: Line opacity in `[0, 1]`.
            colour: A colour, or a data-driven colour expression.

        Returns:
            The paint dict `lines` and `contours` both draw with.
        """
        return {
            "line-width": float(width),
            "line-opacity": float(opacity),
            "line-color": colour,
        }

    @staticmethod
    def _fill_paint(opacity: float, outline_color: str, colour: Any) -> dict:
        """Return a MapLibre fill layer's paint.

        Args:
            opacity: Fill opacity in `[0, 1]`.
            outline_color: The polygon outline colour.
            colour: A colour, or a data-driven colour expression.

        Returns:
            The paint dict `polygons` and filled `contours` both draw with.
        """
        return {
            "fill-opacity": float(opacity),
            "fill-outline-color": outline_color,
            "fill-color": colour,
        }

    def contours(
        self,
        dataset: Any,
        *,
        interval: float | ContourInterval | None = None,
        levels: Any | None = None,
        band: int = DEFAULT_BAND,
        filled: bool = False,
        cmap: str | None = None,
        units: str | None = None,
        color: str | None = None,
        width: Maybe[float] = UNSET,
        opacity: Maybe[float] = UNSET,
        labels: bool = False,
        name: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Trace iso-value contours from a raster band and draw them as vectors.

        The GIS is pyramids' (``Dataset.contour``, the ``gdal_contour`` equivalent); this only draws what
        it returns. Contours are the one field type that maps cleanly onto MapLibre — the result is a
        ``FeatureCollection``, which the tier already knows how to render — so it is the field renderer
        this tier can honestly offer. Vector fields and meshes have no native primitive here; see the
        static and interactive tiers for those.

        Args:
            dataset: A pyramids ``Dataset``.
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            interval: Spacing between levels — a number for "one every N", or a
                :class:`ContourInterval` when the spacing has to be anchored somewhere other than zero.
                Give at most one of this or ``levels``.
            levels: Explicit levels to contour. Give at most one of this or ``interval``. With neither,
                the levels come from :func:`~digitalearth.base.autostyle.auto_style` when the band's
                variable is one it recognises (mean sea-level pressure, 2-m temperature, …) — the levels
                that field is conventionally drawn with.
            band: 1-based band to contour, matching
                :meth:`~digitalearth.web.raster.RasterMixin.field` — pyramids counts bands from 0, and
                this converts, so the same number means the same band everywhere in this tier.
            filled: Draw filled bands between successive levels instead of lines.
            cmap: Colormap for colouring by level; ``None`` resolves the autostyle default for the
                band's variable.
            units: What the contoured values are measured in, which the key names in parentheses after the
                column — when this layer makes a key at all. An explicit ``color=`` colours every level
                alike, so there is no classification to key and the units are recorded nowhere the reader
                sees (review #314). ``None`` (the default) takes the variable's units from
                :func:`~digitalearth.base.autostyle.auto_style`, and leaves the heading bare when it
                carries none — a unit is never guessed. Pass one to correct a band the library
                mis-identifies, or to name the units of a variable it does not know.
            color: A single colour for every contour, overriding ``cmap``. Use it when the levels are
                labelled rather than colour-coded.
            width: Line width in pixels; ignored when ``filled``.
            opacity: Layer opacity in ``[0, 1]``.
            labels: Whether to label each contour with its value — the level for a line, the band's lower
                edge when ``filled``.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when both ``interval`` and ``levels`` are given — pyramids takes exactly one, and
                saying so here names the argument the caller actually wrote — when neither is given and
                the variable is not one ``auto_style`` knows levels for, or when ``width`` or ``opacity``
                is not a finite number, refused at this call because a figure holding NaN or infinity
                could not be written down.
            TypeError: when ``interval`` is neither a number nor a :class:`ContourInterval`.
            FileNotFoundError: when `dataset` is a path that names nothing, or KeyError when no resolver
                is registered for its URL scheme — from :meth:`~digitalearth.web.base.WebMapBase._opened`.

        Examples:
            - Contour a DEM every 100 m:
                ```python
                >>> from digitalearth.web import WebMap                        # doctest: +SKIP
                >>> WebMap().basemap().contours(dem, interval=100)             # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.vector.VectorMixin.lines: what the traced contours are drawn as.
            digitalearth.base.autostyle.auto_style: supplies the levels, colormap and units.
        """
        spacing = _as_interval(interval)
        ask = Ask()
        # Which key the opacity lands on is what `filled` decides, so the record is built here rather than
        # in the sub-builder. `width=` is recorded under the line key either way: a filled band has no line,
        # so a `width=` passed to one is an ask this layer cannot express, and the lift drops a recorded key
        # the paint holds no value under rather than inventing one for it.
        width = as_finite(
            ask("line-width", width, CONTOUR_WIDTH), "width", "WebMap.contours()"
        )
        opacity = as_finite(
            ask("fill-opacity" if filled else "line-opacity", opacity, CONTOUR_OPACITY),
            "opacity",
            "WebMap.contours()",
        )
        if color is not None:
            # `color=None` already spells "not passed" on this builder, so a colour here is the caller's
            # whichever it is — an unstyled trace is coloured by level, as an expression.
            ask.always("fill-color" if filled else "line-color")
        data = self._display_raster_or_skip(dataset, layer="contours")
        if data is None:
            return self
        source = self._to_display_source(data, band=band)
        cmap = self._auto_cmap(source, cmap)
        asked_for = _ContourLevels(
            spacing,
            self._contour_levels(source, interval=spacing, levels=levels),
            asked=interval,
        )
        # Recorded before the sub-builder runs, so the key it sets can say what the values are measured in.
        self.last_units = self._auto_units(source, units)
        features = asked_for.trace(data, band=band, polygonize=filled)
        if len(features) == 0:
            # No level fell inside the band's range. Passing this on raises "column 'level' not found",
            # because pyramids only writes the attribute when it writes a feature — which points at the
            # wrong thing entirely.
            self._skipped(
                "contours",
                "no level lies within the data, so nothing was traced — check that "
                f"{asked_for.asked_as} suits band {band}'s range",
            )
            return self
        # Lines carry `level`; filled bands carry `level_min`/`level_max` for the band's two edges, so
        # colour and label the lower edge — it is what orders the bands.
        attribute = "level_min" if filled else "level"
        column = None if color else attribute
        self._draw_contour_features(
            features,
            filled=filled,
            column=column,
            cmap=cmap,
            color=color,
            width=width,
            opacity=opacity,
            name=name,
            visible=visible,
            asked=ask.named,
        )
        # The units describe *this* layer's values, so they are written on *this* layer's key — looked up
        # by the id the sub-builder just drew under, not on `last_legend`. `last_legend` answers "the most
        # recent classification", so with an explicit `color=` (where `column` is `None` and this layer
        # classifies nothing) it still pointed at whichever layer classified before, and that layer's
        # classes ended up labelled in this raster's units (#314). Asking the layer also means a draw the
        # tier declined — which leaves no key behind — writes nothing at all.
        drawn_id = self._last_layer_id
        keyed = self._legends.get(drawn_id) if drawn_id is not None else None
        if self.last_units and keyed is not None:
            # Never guessed: `last_units` is only set when auto_style supplied one.
            keyed["units"] = self.last_units
        if labels:
            # Otherwise a hidden contour layer leaves its level numbers floating with nothing to annotate.
            return self.labels(features, attribute, visible=visible)
        return self

    def _vector_layer(
        self,
        features: Any,
        prefix: str,
        layer_type: Any,
        paint: dict,
        *,
        kind: str,
        name: str | None = None,
        visible: bool = True,
        layout: dict | None = None,
        source: Any = None,
        asked: Sequence[str] = (),
        color_encoding: Encoding | None = None,
    ) -> Self:
        """Describe a GeoJSON source + a typed layer with `paint`, and record it as the last data layer.

        The single funnel the vector builders and `contours` register through. It does not build the MapLibre
        objects: it records what they should be — the type, the paint, the layout — as values on the
        layer's symbology, and :func:`draw_vector` rebuilds the source and the layer from exactly that.

        Args:
            features: The display-CRS GeoDataFrame to serve as the GeoJSON source.
            prefix: The layer id prefix, from the MapLibre type (`"circle"`/`"line"`/`"fill"`/`"label"`).
                It shapes the public `layer_ids`, so it is kept apart from `kind`.
            layer_type: The `maplibre` `LayerType` member for the layer. Recorded by its value, not the
                member itself, so a figure written to disk carries a string MapLibre reads back.
            paint: The MapLibre paint dict for the layer.
            kind: The registered, engine-neutral kind the layer is recorded as — `"points"`, `"polygons"`,
                `"choropleth"` — which the MapLibre type cannot tell apart.
            name: What a layer switcher calls this layer; `None` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.
            layout: MapLibre layout properties for the layer (a symbol layer's `text-field` and its
                placement live here rather than in `paint`). Merged with the visibility flag.
            source: What the *figure* records the layer as drawing — the caller's own path or URL, which is
                the only reference that survives leaving the process. `features` is what the first draw is
                handed, so nothing is warped twice; `None` records `features` itself, which a builder that
                derived its geometry (`contours` traces its own) has nothing better than (review H1).
            asked: The paint keys this layer's caller actually named, in MapLibre's own spelling. Recorded
                on the symbology so the lift can publish the caller's ask instead of reconstructing it from
                the resolved paint, which cannot tell an ask from a default (#334). Empty — the default —
                records no key at all, so an unstyled layer's description is exactly what it was.
            color_encoding: What drives the layer's colour when a `column` does, as
                :meth:`~digitalearth.base.spec.encoding.Encoding.by_field` built it in `_color_expr`. Filed
                on the symbology's ``color`` channel, which is where
                :meth:`~digitalearth.web.decoration.DecorationMixin.legend` hangs its
                :class:`~digitalearth.base.spec.encoding.Guide` — so the colour key belongs to the layer and
                moves, hides and disappears with it. `None` — a flat colour, or no colour at all — files
                nothing, which is what makes a key on an unclassified layer refusable.

        Returns:
            The same map instance, so builder calls chain.
        """
        _require_layer_api()
        layer_id = self._layer_id(prefix, name)
        # `paint` and `layout` are MapLibre's own spelling, so they are held under a single prop rather
        # than spread across the symbology as if they were engine-neutral channels. `draw_vector` reads
        # them straight back; what makes this a description rather than a closure is that they are
        # *values in the figure* — a saved figure carries them, and nothing is captured in a lambda.
        # Unconditional: `draw_vector` either draws or raises — only the raster drawers decline, and
        # only they need to ask whether the layer survived.
        self._index_layer(
            layer_id,
            name,
            kind=kind,
            visible=visible,
            source=features if source is None else source,
            placed=features,
            symbology=Symbology(
                # Built in rather than lifted in `portable_encodings`: the lift sees only the finished
                # description, and a `Scale` is not a value `props` can carry — `Symbology.to_dict` writes
                # properties as JSON, so the scale would have to be flattened here and rebuilt there, two
                # spellings of one thing. `Encoding` carries a `Scale` natively, so the honest reading of a
                # data-driven fill is recorded once, where the scale is still in hand.
                encodings={} if color_encoding is None else {"color": color_encoding},
                props={
                    # The enum's value, not the member: a description holds plain values, so a figure
                    # written to disk carries a string MapLibre reads back. The spec refuses the member,
                    # which is how this was caught rather than shipped as an unserialisable figure.
                    "maplibre_type": getattr(layer_type, "value", layer_type),
                    "paint": dict(paint),
                    "layout": dict(layout) if layout else {},
                    **asked_record(asked),
                },
            ),
        )
        self._last_layer_id = layer_id
        return self

    @staticmethod
    def _require_column(gdf: Any, column: str) -> Any:
        """Return ``gdf[column]`` as a numpy array, raising a clear error when the column is absent.

        Args:
            gdf: The feature attributes the column is read from — the display-CRS GeoDataFrame.
            column: The attribute name the caller asked to colour or label by.

        Returns:
            The column's values as a numpy array, in feature order.

        Raises:
            KeyError: naming the column that is not there, rather than letting pandas raise from inside a
                builder several frames deeper.
        """
        if column not in getattr(gdf, "columns", []):
            raise KeyError(f"column {column!r} not found in the feature attributes")
        return gdf[column].to_numpy()

    def points(
        self,
        features: Any,
        *,
        column: str | None = None,
        scheme: Any | None = None,
        k: int = 5,
        cmap: str = "viridis",
        size: Maybe[float] = UNSET,
        color: Maybe[str] = UNSET,
        opacity: Maybe[float] = UNSET,
        big: bool | None = None,
        big_data_threshold: int | None = None,
        name: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Draw a point ``FeatureCollection`` as a MapLibre circle layer (recipe W2).

        Args:
            features: A pyramids point ``FeatureCollection`` / GeoDataFrame.
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            column: Optional value column; when given, circles are coloured by it (graduated if ``scheme``
                is set, else a continuous ramp).
            scheme: A cleopatra classification scheme for graduated colouring (with ``column``).
                ``None`` (the default) is a continuous ramp; a scheme means ``k`` graduated classes.
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            size: Circle radius in pixels. The same ``size`` that means marker size on every tier; not
                passed leaves :data:`POINT_SIZE`.
            color: Fixed circle colour used when ``column`` is ``None``; not passed leaves
                :data:`VECTOR_COLOR`.
            opacity: Circle fill opacity in ``[0, 1]``; not passed leaves :data:`POINT_OPACITY`.
            big: Big-data routing — ``None`` (default) auto-routes to a GPU deck.gl layer above
                ``big_data_threshold`` (logged); ``False`` forces per-feature circles; ``True`` forces deck.gl.
            big_data_threshold: Feature count above which this one call auto-routes to deck.gl. ``None``
                uses the map's :attr:`~digitalearth.web.base.WebMapBase.big_data_threshold`, which is the
                way to change it for every layer at once.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: when ``features`` is a raster rather than a vector layer.
            KeyError: when ``column`` names no feature attribute — a MapLibre expression
                reading a property that is not there colours nothing, with no error to explain
                the blank layer — or when ``features`` is a URL with no resolver registered for
                its scheme.
            FileNotFoundError: when ``features`` is a path that names nothing.
            ValueError: when ``size`` or ``opacity`` is not a finite number, refused at this call
                because a figure holding NaN or infinity could not be written down; and when the
                layer routes to a GPU deck.gl overlay — through ``big=True``,
                or by crossing ``big_data_threshold`` — and ``name`` or ``visible`` was passed
                as well. A deck overlay is not a MapLibre style layer, so the registry cannot
                address it to rename or hide it, and honouring those arguments silently would
                be a promise the saved page does not keep. Pass ``big=False`` to force
                per-feature circles instead.

        Examples:
            - Fixed-colour circles, addressable afterwards by the name they were given. Every
              block below needs the ``web`` extra, so all of them are skipped where MapLibre is
              absent; the values shown are what the ``web`` environment returns:
                ```python
                >>> import geopandas as gpd                          # doctest: +SKIP
                >>> from shapely.geometry import Point               # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> gdf = gpd.GeoDataFrame(                          # doctest: +SKIP
                ...     {"pop": [10.0, 20.0, 30.0]},
                ...     geometry=[Point(0, 0), Point(1, 1), Point(2, 2)], crs=4326,
                ... )
                >>> WebMap().points(gdf, size=8.0, name="s").layer_ids  # doctest: +SKIP
                ['s']

                ```
            - Naming a ``column`` colours by it. With no ``scheme`` that is a continuous ramp, and
              the key the call recorded says which of the two it was:
                ```python
                >>> m = WebMap().points(gdf, column="pop")           # doctest: +SKIP
                >>> m.last_legend["kind"], m.last_legend["column"]   # doctest: +SKIP
                ('continuous', 'pop')
                >>> m.last_breaks                                    # doctest: +SKIP
                [10.0, 15.0, 20.0, 25.0, 30.0]

                ```
            - A ``scheme`` classifies that same column into ``k`` graduated classes instead,
              and the key then carries one colour per class:
                ```python
                >>> m = WebMap().points(                             # doctest: +SKIP
                ...     gdf, column="pop", scheme="quantiles", k=3,
                ... )
                >>> m.last_legend["kind"], len(m.last_legend["colors"])  # doctest: +SKIP
                ('graduated', 3)

                ```

        See Also:
            digitalearth.web.bigdata.BigDataMixin.deck_scatter: the GPU path for large tables.
            digitalearth.web.vector.VectorMixin.choropleth: the thematic polygon counterpart.
        """
        #: This builder's own name, for the refusals below to quote back at the caller.
        call = "WebMap.points()"
        _, layer_types = _require_layer_api()
        ask = Ask()
        size = as_finite(ask("circle-radius", size, POINT_SIZE), "size", call)
        opacity = as_finite(
            ask("circle-opacity", opacity, POINT_OPACITY), "opacity", call
        )
        gdf = self._display_gdf(features, method="points")
        # Auto-route to a GPU deck.gl layer only when there is no per-feature symbology to preserve; a forced
        # big=True with a column still routes but warns that the deck path drops the colouring (M1).
        if big or (
            big is None
            and column is None
            and self._route_big(gdf, "points", threshold=big_data_threshold)
        ):
            if column is not None:
                logger.warning(
                    "points: big=True routes {} features to a flat deck.gl layer; column={!r} styling is dropped",
                    len(gdf),
                    column,
                )
            if name is not None or not visible:
                # A deck.gl overlay is not a MapLibre style layer: the switcher toggles layers with
                # setLayoutProperty, which cannot reach it, so it has no registry entry to name or hide.
                # Dropping these silently would change the API contract at 50 000 features.
                raise ValueError(
                    f"points() is rendering {len(gdf)} features as a deck.gl overlay"
                    f"{'' if big else ' (over the ' + str(self._threshold(big_data_threshold)) + '-feature threshold)'}"
                    ", which the layer registry cannot address — so name= and visible= cannot be "
                    "honoured. Pass big=False to force a MapLibre layer, or drop those arguments."
                )
            return self.deck_scatter(gdf, size=size)
        paint: dict = {"circle-radius": float(size), "circle-opacity": float(opacity)}
        color_encoding: Encoding | None = None
        if column is not None:
            paint["circle-color"], color_encoding = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        else:
            paint["circle-color"] = ask("circle-color", color, VECTOR_COLOR)
        return self._vector_layer(
            gdf,
            "circle",
            layer_types.CIRCLE,
            paint,
            kind="points",
            name=name,
            visible=visible,
            source=features,
            asked=ask.named,
            color_encoding=color_encoding,
        )

    def lines(
        self,
        features: Any,
        *,
        column: str | None = None,
        scheme: Any | None = None,
        k: int = 5,
        cmap: str = "viridis",
        width: Maybe[float] = UNSET,
        color: Maybe[str] = UNSET,
        opacity: Maybe[float] = UNSET,
        name: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Draw a line ``FeatureCollection`` as a MapLibre line layer (recipe W2).

        Args:
            features: A pyramids line ``FeatureCollection`` / GeoDataFrame.
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            column: Optional value column to colour the lines by.
            scheme: A cleopatra classification scheme for graduated colouring (with ``column``).
                ``None`` (the default) is a continuous ramp; a scheme means ``k`` graduated classes.
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            width: Line width in pixels.
            color: Fixed line colour used when ``column`` is ``None``.
            opacity: Line opacity in ``[0, 1]``.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: when ``features`` is a raster rather than a vector layer.
            KeyError: when ``column`` names no feature attribute, or when ``features`` is a URL with no
                resolver registered for its scheme.
            ValueError: when ``width`` or ``opacity`` is not a finite number, refused at this call because
                a figure holding NaN or infinity could not be written down.
            FileNotFoundError: when ``features`` is a path that names nothing.

        Examples:
            - A fixed-colour network (needs the ``web`` extra, so the block is skipped without it):
                ```python
                >>> import geopandas as gpd                          # doctest: +SKIP
                >>> from shapely.geometry import LineString          # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> gdf = gpd.GeoDataFrame(                          # doctest: +SKIP
                ...     {"flow": [1.0, 5.0]},
                ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 0)])],
                ...     crs=4326,
                ... )
                >>> m = WebMap().lines(gdf, color="#ffcc00", width=3.0)  # doctest: +SKIP
                >>> m.layer_ids                                      # doctest: +SKIP
                ['line-1']

                ```
            - Colouring by a column is a continuous ramp unless a ``scheme`` is named, and the
              sample points the ramp was built from are readable back off the map:
                ```python
                >>> m = WebMap().lines(gdf, column="flow", cmap="plasma")  # doctest: +SKIP
                >>> m.last_legend["kind"], m.last_breaks             # doctest: +SKIP
                ('continuous', [1.0, 2.0, 3.0, 4.0, 5.0])

                ```
            - With no ``column`` there is nothing to key, so no classification is recorded and a
              legend built afterwards has nothing to draw from:
                ```python
                >>> m = WebMap().lines(gdf)                          # doctest: +SKIP
                >>> m.last_legend, m.last_breaks                     # doctest: +SKIP
                (None, None)

                ```

        See Also:
            digitalearth.web.vector.VectorMixin.contours: traces a raster into these lines.
        """
        _, layer_types = _require_layer_api()
        ask = Ask()
        width = as_finite(
            ask("line-width", width, LINE_WIDTH), "width", "WebMap.lines()"
        )
        opacity = as_finite(
            ask("line-opacity", opacity, LINE_OPACITY), "opacity", "WebMap.lines()"
        )
        gdf = self._display_gdf(features, method="lines")
        colour: Any
        color_encoding: Encoding | None = None
        if column is None:
            # Only on this arm, as before: `ask` *records* the key as named, so reaching it on the
            # classified arm too would add `line-color` to what the caller is said to have asked for.
            colour = ask("line-color", color, VECTOR_COLOR)
        else:
            colour, color_encoding = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        return self._vector_layer(
            gdf,
            "line",
            layer_types.LINE,
            self._line_paint(width, opacity, colour),
            kind="lines",
            name=name,
            visible=visible,
            source=features,
            asked=ask.named,
            color_encoding=color_encoding,
        )

    def polygons(
        self,
        features: Any,
        *,
        column: str | None = None,
        scheme: Any | None = None,
        k: int = 5,
        cmap: str = "viridis",
        color: Maybe[str] = UNSET,
        opacity: Maybe[float] = UNSET,
        outline_color: str = "#ffffff",
        big: bool | None = None,
        big_data_threshold: int | None = None,
        name: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Draw a polygon ``FeatureCollection`` as a MapLibre fill layer (recipe W2).

        Args:
            features: A pyramids polygon ``FeatureCollection`` / GeoDataFrame.
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            column: Optional value column to colour the polygons by (graduated if ``scheme`` is set, else a
                continuous ramp). For full thematic symbology prefer :meth:`choropleth`.
            scheme: A cleopatra classification scheme for graduated colouring (with ``column``).
                ``None`` (the default) is a continuous ramp; a scheme means ``k`` graduated classes.
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            color: Fixed fill colour used when ``column`` is ``None``.
            opacity: Fill opacity in ``[0, 1]``.
            outline_color: Polygon outline colour.
            big: Big-data routing — ``None`` (default) auto-routes to a GPU deck.gl layer above
                ``big_data_threshold`` (logged); ``False`` forces per-feature fills; ``True`` forces deck.gl.
            big_data_threshold: Feature count above which this one call auto-routes to deck.gl. ``None``
                uses the map's :attr:`~digitalearth.web.base.WebMapBase.big_data_threshold`, which is the
                way to change it for every layer at once.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: when ``features`` is a raster rather than a vector layer.
            KeyError: when ``column`` names no feature attribute, or when ``features`` is a URL with
                no resolver registered for its scheme.
            FileNotFoundError: when ``features`` is a path that names nothing.
            ValueError: when ``opacity`` is not a finite number, refused at this call because a figure
                holding NaN or infinity could not be written down; and when the
                layer routes to a GPU deck.gl overlay — through ``big=True``,
                or by crossing ``big_data_threshold`` — and ``name`` or ``visible`` was passed
                as well. A deck overlay is not a MapLibre style layer, so the registry cannot
                address it to rename or hide it, and honouring those arguments silently would
                be a promise the saved page does not keep. Pass ``big=False`` to force
                per-feature fills instead.

        Examples:
            - A flat fill in one colour, named so the switcher can list it (needs the ``web``
              extra, so the block is skipped without it):
                ```python
                >>> import geopandas as gpd                          # doctest: +SKIP
                >>> from shapely.geometry import Polygon             # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> gdf = gpd.GeoDataFrame(                          # doctest: +SKIP
                ...     {"pop": [10.0, 30.0]},
                ...     geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                ...               Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])],
                ...     crs=4326,
                ... )
                >>> m = WebMap().polygons(gdf, color="#cc4444", name="p")  # doctest: +SKIP
                >>> m.layer_ids                                      # doctest: +SKIP
                ['p']

                ```
            - Naming a ``column`` colours by it — continuously, unless a ``scheme`` is given. For a
              thematic map prefer :meth:`choropleth`, which is this same fill with the
              classification spelled out in its signature:
                ```python
                >>> m = WebMap().polygons(gdf, column="pop")         # doctest: +SKIP
                >>> m.last_legend["kind"], m.last_breaks             # doctest: +SKIP
                ('continuous', [10.0, 15.0, 20.0, 25.0, 30.0])

                ```
            - Forcing the GPU path while also asking for ``name``/``visible`` is refused,
              rather than accepted and then quietly dropped from the saved page:
                ```python
                >>> try:                                             # doctest: +SKIP
                ...     WebMap().polygons(gdf, big=True, name="p")
                ... except ValueError as error:
                ...     print(str(error).split(",")[0])
                polygons() is rendering 2 features as a deck.gl overlay

                ```

        See Also:
            digitalearth.web.vector.VectorMixin.choropleth: the thematic build of this fill.
            digitalearth.web.bigdata.BigDataMixin.deck_polygons: the GPU path for large tables.
        """
        _, layer_types = _require_layer_api()
        ask = Ask()
        opacity = as_finite(
            ask("fill-opacity", opacity, FILL_OPACITY), "opacity", "WebMap.polygons()"
        )
        gdf = self._display_gdf(features, method="polygons")
        # Auto-route to deck.gl only when no column styling would be lost; a forced big=True with a column
        # still routes but warns that the deck path drops the colouring (M1).
        if big or (
            big is None
            and column is None
            and self._route_big(gdf, "polygons", threshold=big_data_threshold)
        ):
            if column is not None:
                logger.warning(
                    "polygons: big=True routes {} features to a flat deck.gl layer; column={!r} styling is dropped",
                    len(gdf),
                    column,
                )
            if name is not None or not visible:
                # A deck.gl overlay is not a MapLibre style layer: the switcher toggles layers with
                # setLayoutProperty, which cannot reach it, so it has no registry entry to name or hide.
                # Dropping these silently would change the API contract at 50 000 features.
                raise ValueError(
                    f"polygons() is rendering {len(gdf)} features as a deck.gl overlay"
                    f"{'' if big else ' (over the ' + str(self._threshold(big_data_threshold)) + '-feature threshold)'}"
                    ", which the layer registry cannot address — so name= and visible= cannot be "
                    "honoured. Pass big=False to force a MapLibre layer, or drop those arguments."
                )
            return self.deck_polygons(gdf)
        colour: Any
        color_encoding: Encoding | None = None
        if column is None:
            # Only on this arm, as before: `ask` *records* the key as named, so reaching it on the
            # classified arm too would add `fill-color` to what the caller is said to have asked for.
            colour = ask("fill-color", color, VECTOR_COLOR)
        else:
            colour, color_encoding = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        return self._vector_layer(
            gdf,
            "fill",
            layer_types.FILL,
            self._fill_paint(opacity, outline_color, colour),
            kind="polygons",
            name=name,
            visible=visible,
            source=features,
            asked=ask.named,
            color_encoding=color_encoding,
        )

    def choropleth(
        self,
        features: Any,
        column: str,
        *,
        scheme: Any | None = None,
        k: int = 5,
        cmap: str = "viridis",
        opacity: Maybe[float] = UNSET,
        outline_color: str = "#ffffff",
        name: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Draw a thematic polygon choropleth coloured by ``column`` (recipe W2).

        A **continuous ramp** by default (``scheme=None``), the same default the static and interactive
        ``choropleth`` carry — one call classifies the same way whichever tier draws it. Pass a scheme
        (``"quantiles"``, ``"equal_interval"``, ``"fisher_jenks"``, …) for a graduated map of ``k``
        classes, whose breaks come from ``cleopatra.styling.styles.classify`` and compile into a MapLibre
        ``step`` paint expression, or ``scheme="categorical"`` to colour an unordered attribute by distinct
        value (a MapLibre ``match`` expression, DC.8). The breaks / categories are exposed on
        ``WebMap.last_breaks`` for building a legend.

        Args:
            features: A pyramids polygon ``FeatureCollection`` / GeoDataFrame.
                A path or URL to one is taken too, and is the only input this layer can be
                written down with — a pyramids object does not know where it came from. The reference
                is opened at the display choke point
                (:meth:`~digitalearth.web.base.WebMapBase._opened`) and the caller's own path is what
                the figure records.
            column: The attribute that colours the polygons (numeric for graduated/continuous; any hashable
                value for ``scheme="categorical"``). Required.
            scheme: A cleopatra classification scheme (``"quantiles"``, ``"equal_interval"``,
                ``"fisher_jenks"``, …) or an explicit edge sequence for graduated colouring;
                ``"categorical"`` for distinct-value colouring; ``None`` (the default) for a continuous ramp.
                Under either classification a feature whose ``column`` value is missing falls outside every
                class, and is painted the shared neutral
                :data:`~digitalearth.base.symbology.MISSING_COLOR` rather than the lowest class — the same
                colour every other tier gives an unclassifiable feature.
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap name.
            opacity: Fill opacity in ``[0, 1]``.
            outline_color: Polygon outline colour.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            KeyError: when ``column`` is not a feature attribute, when ``cmap`` names no registered
                colormap, or when ``features`` is a URL with no resolver registered for its scheme.
            ValueError: when ``opacity`` is not a finite number — refused before the classifier runs,
                because a figure holding NaN or infinity could not be written down — and propagated from
                the classifier itself (unknown scheme, constant data, …).
            FileNotFoundError: when ``features`` is a path that names nothing.

        Examples:
            - The default is a **continuous** ramp: no ``scheme``, no classes, and ``last_breaks``
              carries the sample points the ramp was built from rather than class edges (needs the
              ``web`` extra, so the block is skipped without it):
                ```python
                >>> import geopandas as gpd                          # doctest: +SKIP
                >>> from shapely.geometry import Polygon             # doctest: +SKIP
                >>> from digitalearth.web import WebMap              # doctest: +SKIP
                >>> gdf = gpd.GeoDataFrame(                          # doctest: +SKIP
                ...     {"pop": [10.0, 30.0], "zone": ["urban", "rural"]},
                ...     geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                ...               Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])],
                ...     crs=4326,
                ... )
                >>> m = WebMap().choropleth(gdf, "pop")              # doctest: +SKIP
                >>> m.last_legend["kind"], m.last_breaks             # doctest: +SKIP
                ('continuous', [10.0, 15.0, 20.0, 25.0, 30.0])

                ```
            - Naming a scheme classifies instead, and the breaks become the ``k`` class edges the
              MapLibre ``step`` expression was compiled from:
                ```python
                >>> m = WebMap().choropleth(                         # doctest: +SKIP
                ...     gdf, "pop", scheme="equal_interval", k=2,
                ... )
                >>> m.last_legend["kind"], m.last_breaks             # doctest: +SKIP
                ('graduated', [10.0, 20.0, 30.0])

                ```
            - ``scheme="categorical"`` colours an unordered attribute by distinct value, and
              the key then lists the values it matched rather than numeric edges:
                ```python
                >>> m = WebMap().choropleth(                         # doctest: +SKIP
                ...     gdf, "zone", scheme="categorical", name="zones",
                ... )
                >>> m.last_legend["values"], m.layer_ids             # doctest: +SKIP
                (['rural', 'urban'], ['zones'])

                ```

        See Also:
            digitalearth.web.decoration.DecorationMixin.legend: draws the key from ``last_legend``.
            digitalearth.web.vector.VectorMixin.polygons: the same fill layer without the thematic
                classification.
        """
        _, layer_types = _require_layer_api()
        ask = Ask()
        opacity = as_finite(
            ask("fill-opacity", opacity, CHOROPLETH_OPACITY),
            "opacity",
            "WebMap.choropleth()",
        )
        gdf = self._display_gdf(features, method="choropleth")
        values = self._require_column(gdf, column)
        fill, color_encoding = self._color_expr(values, column, scheme, k, cmap)
        paint = {
            "fill-color": fill,
            "fill-opacity": float(opacity),
            "fill-outline-color": outline_color,
        }
        return self._vector_layer(
            gdf,
            "fill",
            layer_types.FILL,
            paint,
            kind="choropleth",
            name=name,
            visible=visible,
            source=features,
            asked=ask.named,
            color_encoding=color_encoding,
        )

    def vector_tiles(
        self,
        source: VectorTileSource,
        *,
        source_layer: str,
        geometry: str = "line",
        color: str = VECTOR_COLOR,
        width: Maybe[float] = UNSET,
        size: Maybe[float] = UNSET,
        opacity: Maybe[float] = UNSET,
        outline_color: str = "#ffffff",
        name: str | None = None,
        visible: bool = True,
    ) -> Self:
        """Draw a Mapbox Vector Tile (MVT) set as a circle / line / fill layer (WB-6).

        MapLibre serves a vector tile pyramid — an ``.mvt``/``.pbf`` tile set, or a TileJSON describing one —
        as a ``vector`` source, and draws one of its named layers with an ordinary style layer. This is the
        vector counterpart of :meth:`~digitalearth.web.decoration.DecorationMixin.tiles`, which drapes a
        *raster* pyramid under the data: the features here are drawn **among** the data from the tile URL,
        never read into memory, so a continent of roads costs the page a URL rather than a GeoJSON blob.

        The source is the tile URL, not a :class:`~pyramids.feature.FeatureCollection`, so there is no
        reprojection and no pyramids involvement — a vector tile set is already Web-Mercator tiles. The tile
        set and its coverage travel as one :class:`VectorTileSource`, which names the set exactly one way — a
        ``{z}/{x}/{y}`` template as ``tiles`` or a TileJSON ``url`` — and refuses an ambiguous pairing when it
        is constructed, before this builder is reached.

        Args:
            source: The tile set to draw, as a :class:`VectorTileSource` carrying its template-or-``url`` and
                the service's zoom range and attribution.
            source_layer: The name of the layer to draw *inside* the tile set — a vector tile holds many
                named layers (``roads``, ``buildings``, ``water``, …), and MapLibre draws one at a time.
            geometry: How to draw the features — ``"line"`` (the default, for roads and boundaries),
                ``"fill"`` (for buildings and land use) or ``"circle"`` (for points of interest). Each maps
                to the MapLibre layer type of the same shape.
            color: The colour the features are drawn in; the line colour, the fill colour or the circle
                colour, by ``geometry``. Not passed leaves :data:`VECTOR_COLOR`.
            width: Line width in pixels, used by ``geometry="line"``; not passed leaves :data:`LINE_WIDTH`.
            size: Circle radius in pixels, used by ``geometry="circle"``; not passed leaves
                :data:`POINT_SIZE`.
            opacity: Layer opacity in ``[0, 1]``; not passed leaves the geometry's own default
                (:data:`LINE_OPACITY`, :data:`FILL_OPACITY` or :data:`POINT_OPACITY`).
            outline_color: Polygon outline colour, used by ``geometry="fill"``.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            ValueError: when ``geometry`` is not ``"line"``, ``"fill"`` or ``"circle"``; or when ``width``,
                ``size`` or ``opacity`` is not a finite number, refused at this call because a figure holding
                NaN or infinity could not be written down. The tiles-xor-``url`` rule is refused earlier, when
                the :class:`VectorTileSource` is constructed.
            ImportError: when the ``web`` extra is not installed, so there is no MapLibre layer API.

        Examples:
            - Draw a roads tile set as lines, addressable by the name it was given (needs the ``web``
              extra, so the block is skipped without it):
                ```python
                >>> from digitalearth.web import VectorTileSource, WebMap      # doctest: +SKIP
                >>> m = WebMap().basemap().vector_tiles(                       # doctest: +SKIP
                ...     VectorTileSource(tiles="https://tiles.example.org/{z}/{x}/{y}.pbf"),
                ...     source_layer="roads", name="roads",
                ... )
                >>> m.layer_ids                                              # doctest: +SKIP
                ['tiles-1', 'roads']

                ```

        See Also:
            digitalearth.web.decoration.DecorationMixin.tiles: the raster-pyramid counterpart, drawn under
                the data.
            digitalearth.web.vector.VectorMixin.lines: the same line layer drawn from an in-memory
                collection instead of a tile set.
        """
        #: This builder's own name, for the refusals below to quote back at the caller.
        call = "WebMap.vector_tiles()"
        _, layer_types = _require_layer_api()
        source_spec = source.to_source()
        ask = Ask()
        if geometry == "line":
            width = as_finite(ask("line-width", width, LINE_WIDTH), "width", call)
            opacity = as_finite(
                ask("line-opacity", opacity, LINE_OPACITY), "opacity", call
            )
            prefix, layer_type = "line", layer_types.LINE
            paint = self._line_paint(width, opacity, color)
        elif geometry == "fill":
            opacity = as_finite(
                ask("fill-opacity", opacity, FILL_OPACITY), "opacity", call
            )
            prefix, layer_type = "fill", layer_types.FILL
            paint = self._fill_paint(opacity, outline_color, color)
        elif geometry == "circle":
            size = as_finite(ask("circle-radius", size, POINT_SIZE), "size", call)
            opacity = as_finite(
                ask("circle-opacity", opacity, POINT_OPACITY), "opacity", call
            )
            prefix, layer_type = "circle", layer_types.CIRCLE
            paint = {
                "circle-radius": float(size),
                "circle-opacity": float(opacity),
                "circle-color": color,
            }
        else:
            raise ValueError(
                f"vector_tiles() takes geometry= as 'line', 'fill' or 'circle'; got {geometry!r}"
            )
        layer_id = self._layer_id(prefix, name)
        # Recorded as values — the source dict, the source-layer, the MapLibre type and the paint — never a
        # closure: :func:`draw_vector_tiles` rebuilds the layer from exactly this, and no feature source is
        # recorded (there is none of), so the layer draws from its description alone, as a raster basemap does.
        self._index_layer(
            layer_id,
            name,
            kind="vector_tiles",
            visible=visible,
            symbology=Symbology(
                props={
                    "source": source_spec,
                    "source_layer": str(source_layer),
                    # The enum's value, not the member: a description holds plain values a saved figure can
                    # carry, and the spec refuses the member.
                    "maplibre_type": getattr(layer_type, "value", layer_type),
                    "paint": dict(paint),
                    "layout": {},
                    **asked_record(ask.named),
                }
            ),
        )
        self._last_layer_id = layer_id
        return self
