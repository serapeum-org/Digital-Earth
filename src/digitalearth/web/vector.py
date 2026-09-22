"""VectorMixin — web-tier vector builders (DW.2, recipe W2).

``points`` / ``lines`` / ``polygons`` turn a pyramids ``FeatureCollection`` (reprojected to lon/lat through
pyramids) into MapLibre circle / line / fill layers over a GeoJSON source; ``choropleth`` is the thematic
polygon map. Colour-by-value compiles into a MapLibre **data-driven paint expression**:

* graduated (``scheme`` set) → a ``["step", ["get", col], …]`` expression whose breaks come from
  **cleopatra.styling.styles.classify** (pure-numpy quantiles / equal-interval / Fisher-Jenks — no mapclassify), the
  same classifier the static tier uses, so the classes match across tiers;
* continuous (``scheme=None``) → an ``["interpolate", ["linear"], ["get", col], …]`` colour ramp.

cleopatra / matplotlib / numpy are imported lazily inside the methods; importing the tier needs none of them.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Self, Union

from loguru import logger

from digitalearth.base.deprecation import renamed_parameter
from digitalearth.base.spec import LayerSpec, LegendSpec, Scale, Symbology
from digitalearth.web.base import _require_layer_api


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


def _as_interval(
    interval: Union[float, ContourInterval, None],
) -> Optional[ContourInterval]:
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


#: Flat colour a vector layer falls back on when the caller pins neither `color=` nor a value column.
#: The builders repeat it as a parameter default so it shows in a rendered signature; this is the name
#: the internals use when they have to supply it themselves.
CONTOUR_COLOR = "#3388ff"

#: Outline of a filled contour band — `polygons()`'s own default outline, which is what filled contours were
#: drawn with while they were drawn through it.
CONTOUR_OUTLINE_COLOR = "#ffffff"

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


def draw_vector(_web_map: Any, data: Any, layer: LayerSpec) -> Any:
    """Build the MapLibre source and typed layer for any of the vector kinds.

    All seven — points, lines, polygons, choropleth, labels, and the two contour kinds — are a GeoJSON
    source plus one typed layer, differing only in the MapLibre type and the paint the builder resolved. They
    share a drawer for the same reason they share a registration funnel.

    Args:
        _web_map: Unused — every drawer takes the map, and this one draws without it.
        data: The layer's source — the display-CRS GeoDataFrame served as GeoJSON.
        layer: The layer's description.

    Returns:
        A :class:`~digitalearth.web.renderer.DrawnLayer`. It never declines: any of these kinds either
        draws or refuses by raising, which is why their builders do not ask whether the layer survived.

    Raises:
        ValueError: when the description records no MapLibre type or no paint for the layer, naming the
            layer rather than the missing MapLibre key.
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
        source_spec=data,
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

    @staticmethod
    def _legend_dict(legend: LegendSpec, column: str, values: Any = None) -> dict:
        """Return the ``last_legend`` shape this tier has always stored, derived from a `LegendSpec`.

        The dict is unchanged — it is public-ish, asserted by tests, and read by the legend control — but it
        is now *derived* from the spec rather than assembled a fourth time. Replacing the attribute itself is
        a follow-up; producing it from one place is what DE-19 is for.

        Args:
            legend: The spec the colours and labels come from.
            column: The column the layer was coloured by.
            values: Override for the ``values`` list, where the stored shape carries the class *edges*
                rather than one value per entry.

        Returns:
            The legend dict, with ``kind``/``column``/``values``/``colors`` exactly as before.
        """
        payload = legend.to_dict()
        return {
            "kind": payload["kind"],
            "column": column,
            "values": payload["values"] if values is None else values,
            "colors": payload["colors"],
        }

    def _color_expr(
        self,
        values: Any,
        column: str,
        scheme: Optional[Any],
        k: int,
        cmap: str,
    ) -> list:
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
            A MapLibre expression list (``["step", …]``, ``["match", …]`` or ``["interpolate", …]``). Also
            sets ``self.last_breaks`` to the breaks (graduated edges), the categories, or the ramp stops,
            and ``self.last_legend`` to those values *plus the colours they were drawn with*, which is what
            :meth:`~digitalearth.web.decoration.DecorationMixin.legend` renders.

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

    def _categorical_color_expr(self, values: Any, column: str, cmap: str) -> list:
        """Compile a MapLibre ``match`` expression over the column's distinct values (DC.8).

        Args:
            values: The 1-D value array driving the colour.
            column: The GeoJSON property the expression reads.
            cmap: The colormap sampled for the class colours; a qualitative map is resolved for it.

        Returns:
            The ``["match", …]`` expression, with `last_breaks` and `last_legend` recorded.

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
        legend = LegendSpec.from_scale(
            Scale.categorical([_native(c) for c in categories], list(colors)),
            title=column,
        )
        self.last_breaks = [_native(c) for c in categories]
        self.last_legend = self._legend_dict(legend, column)
        return expr

    def _graduated_color_expr(
        self, values: Any, column: str, scheme: Any, k: int, cmap: str
    ) -> list:
        """Compile a MapLibre ``step`` expression over class edges.

        Args:
            values: The 1-D value array driving the colour.
            column: The GeoJSON property the expression reads.
            scheme: A cleopatra classification scheme, or an explicit edge sequence.
            k: The number of classes.
            cmap: The colormap sampled for the class colours.

        Returns:
            The guarded ``["case", …, ["step", …], MISSING_COLOR]`` expression, with `last_breaks`
            and `last_legend` recorded.

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
        self.last_legend = self._legend_dict(
            LegendSpec.from_scale(
                Scale(
                    self.last_breaks[0],
                    self.last_breaks[-1],
                    scheme=scheme,
                    breaks=tuple(self.last_breaks),
                ),
                colors=list(colors),
                title=column,
            ),
            column,
            values=self.last_breaks,
        )
        return expr

    def _ramp_color_expr(self, values: Any, column: str, cmap: str) -> list:
        """Compile a MapLibre ``interpolate`` expression over a continuous ramp.

        Args:
            values: The 1-D value array driving the colour.
            column: The GeoJSON property the expression reads.
            cmap: The colormap sampled for the ramp stops.

        Returns:
            The ``["interpolate", …]`` expression, with `last_breaks` and `last_legend` recorded.

        Raises:
            ValueError: when the column holds no finite values to colour.
        """
        import numpy as np

        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"column {column!r} has no finite values to colour")
        lo, hi = Scale.from_finite(finite).as_limits()
        stops = np.linspace(lo, hi, 5)
        colors = self._cmap_hex(cmap, len(stops))
        expr = ["interpolate", ["linear"], ["get", column]]
        for stop, color in zip(stops, colors):
            expr.extend([float(stop), color])
        self.last_breaks = [float(s) for s in stops]
        # The stops are handed over rather than recomputed from the limits: np.linspace pins its last
        # element to `hi` exactly and the arithmetic in `from_scale` does not, so for lo=-3.7, hi=12.9 the
        # top swatch was labelled 12.900000000000002 while the ramp drew 12.9. The legend must be the stops
        # that were drawn, not a second computation that usually agrees with them.
        self.last_legend = self._legend_dict(
            LegendSpec.from_scale(
                Scale.from_limits(lo, hi),
                colors=list(colors),
                title=column,
                values=self.last_breaks,
            ),
            column,
        )
        return expr

    def labels(
        self,
        features: Any,
        column: str,
        *,
        text_size: Optional[float] = None,
        color: str = "#ffffff",
        halo_color: str = "#000000",
        halo_width: float = 1.0,
        offset: Optional[Any] = None,
        allow_overlap: bool = False,
        name: Optional[str] = None,
        visible: bool = True,
        size: Optional[float] = None,
    ) -> Self:
        """Label features with the text in ``column`` (recipe W2).

        Labels are how a map says what is on it, and MapLibre's symbol layer does the work — data-driven
        text, collision detection, halos and placement. None of it was reachable: the only symbol layer the
        tier built was the count inside ``cluster``.

        Args:
            features: A pyramids ``FeatureCollection`` or GeoDataFrame; points label at the point, lines
                and polygons at a placement MapLibre picks.
            column: The property to read the text from.
            text_size: Text size in pixels (``12.0`` when omitted — the signature's ``None`` is the
                "not passed" sentinel the deprecated spelling is resolved against). Named for the text
                rather than ``size``, which means the visual size of a marker everywhere else.
            color: Text colour.
            halo_color: Colour of the outline drawn behind the glyphs, which is what keeps a label legible
                over imagery.
            halo_width: Halo width in pixels; ``0`` disables it.
            offset: ``(x, y)`` offset in ems, e.g. ``(0, -1.2)`` to lift a label off its point.
            allow_overlap: Whether labels may overlap. ``False`` (the default) lets MapLibre drop labels
                that collide, which is what keeps a dense layer readable.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.
            size: **Deprecated** spelling of ``text_size``; forwarded unchanged, after a
                ``DeprecationWarning`` that ``size=`` will be removed in a future release.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: when ``features`` is not a vector layer, or when both ``text_size`` and the
                deprecated ``size`` are passed — they name one parameter.
            KeyError: when ``column`` is not one of its properties — a MapLibre expression reading a
                missing property renders nothing at all, with no error to explain the empty map.

        Examples:
            - Name each feature:
                ```python
                >>> from digitalearth.web import WebMap                          # doctest: +SKIP
                >>> WebMap().basemap().polygons(gdf).labels(gdf, "name")         # doctest: +SKIP

                ```

        See Also:
            digitalearth.web.decoration.DecorationMixin.text: a single annotation at a coordinate.
        """
        _, layer_types = _require_layer_api()
        text_size = renamed_parameter(
            new="text_size",
            value=text_size,
            old="size",
            alias=size,
            caller="WebMap.labels()",
            default=12.0,
        )
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
            "text-color": color,
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
        )

    def _contour_levels(
        self, source: Any, *, interval: Optional[ContourInterval], levels: Optional[Any]
    ) -> Optional[Any]:
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
            raise ValueError(
                "contours() needs interval= or levels=: neither was given, and the band's variable "
                f"({source.metadata('variable')!r}) is not one auto_style carries levels for."
            )
        return resolved

    def _draw_contour_features(
        self,
        features: Any,
        *,
        filled: bool,
        column: Optional[str],
        cmap: str,
        color: Optional[str],
        width: float,
        opacity: float,
        name: Optional[str],
        visible: bool,
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
        # a caller's `color=` pins every contour to one colour instead.
        colour = (
            color or CONTOUR_COLOR
            if column is None
            else self._ramp_color_expr(self._require_column(gdf, column), column, cmap)
        )
        if filled:
            paint = self._fill_paint(opacity, CONTOUR_OUTLINE_COLOR, colour)
            prefix, layer_type, kind = "fill", layer_types.FILL, "filled_contours"
        else:
            paint = self._line_paint(width, opacity, colour)
            prefix, layer_type, kind = "line", layer_types.LINE, "contours"
        self._vector_layer(
            gdf, prefix, layer_type, paint, kind=kind, name=name, visible=visible
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
        interval: Union[float, ContourInterval, None] = None,
        levels: Optional[Any] = None,
        band: int = 1,
        filled: bool = False,
        cmap: Optional[str] = None,
        units: Optional[str] = None,
        color: Optional[str] = None,
        width: float = 1.5,
        opacity: float = 1.0,
        labels: bool = False,
        name: Optional[str] = None,
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
            interval: Spacing between levels — a number for "one every N", or a
                :class:`ContourInterval` when the spacing has to be anchored somewhere other than zero.
                Give at most one of this or ``levels``.
            levels: Explicit levels to contour. Give at most one of this or ``interval``. With neither,
                the levels come from :func:`~digitalearth.base.autostyle.auto_style` when the band's
                variable is one it recognises (mean sea-level pressure, 2-m temperature, …) — the levels
                that field is conventionally drawn with.
            band: 1-based band to contour, matching
                :meth:`~digitalearth.web.raster.RasterMixin.add_raster` — pyramids counts bands from 0, and
                this converts, so the same number means the same band everywhere in this tier.
            filled: Draw filled bands between successive levels instead of lines.
            cmap: Colormap for colouring by level; ``None`` resolves the autostyle default for the
                band's variable.
            units: What the contoured values are measured in, which the key names in parentheses after the
                column. ``None`` (the default) takes the variable's units from
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
                saying so here names the argument the caller actually wrote — or when neither is given
                and the variable is not one ``auto_style`` knows levels for.
            TypeError: when ``interval`` is neither a number nor a :class:`ContourInterval`.

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
        data = self._display_raster_or_skip(dataset, layer="contours")
        if data is None:
            return self
        source = self._to_display_source(data, band=band)
        cmap = self._auto_cmap(source, cmap)
        levels = self._contour_levels(source, interval=spacing, levels=levels)
        # Recorded before the sub-builder runs, so the key it sets can say what the values are measured in.
        self.last_units = self._auto_units(source, units)
        features = data.contour(
            interval=spacing.spacing if spacing is not None else None,
            fixed_levels=list(levels) if levels is not None else None,
            base=spacing.base if spacing is not None else 0.0,
            band=int(band) - 1,  # pyramids counts bands from 0; this tier counts from 1
            attribute="level",
            polygonize=filled,
        )
        if len(features) == 0:
            # No level fell inside the band's range. Passing this on raises "column 'level' not found",
            # because pyramids only writes the attribute when it writes a feature — which points at the
            # wrong thing entirely.
            self._skipped(
                "contours",
                "no level lies within the data, so nothing was traced — check that "
                f"{'levels=' + repr(levels) if spacing is None else 'interval=' + repr(interval)} "
                f"suits band {band}'s range",
            )
            return self
        # Lines carry `level`; filled bands carry `level_min`/`level_max` for the band's two edges, so
        # colour and label the lower edge — it is what orders the bands.
        attribute = "level_min" if filled else "level"
        # Held by identity, to tell a key this call produces from the one already sitting in `last_legend`.
        keyed_before = self.last_legend
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
        )
        if (
            self.last_units
            and self.last_legend is not None
            and self.last_legend is not keyed_before
        ):
            # The classification the sub-builder just recorded describes this raster's values, so the key
            # can name their unit. Never guessed: `last_units` is only set when auto_style supplied one.
            # Compared by identity, because with an explicit `color=` this layer classifies nothing —
            # `column` is `None` above — and `last_legend` then still belongs to whichever layer classified
            # before it. Stamping it there labelled one layer's classes in another layer's units (#314).
            self.last_legend["units"] = self.last_units
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
        name: Optional[str] = None,
        visible: bool = True,
        layout: Optional[dict] = None,
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
            source=features,
            symbology=Symbology(
                props={
                    # The enum's value, not the member: a description holds plain values, so a figure
                    # written to disk carries a string MapLibre reads back. The spec refuses the member,
                    # which is how this was caught rather than shipped as an unserialisable figure.
                    "maplibre_type": getattr(layer_type, "value", layer_type),
                    "paint": dict(paint),
                    "layout": dict(layout) if layout else {},
                }
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
        column: Optional[str] = None,
        scheme: Optional[Any] = None,
        k: int = 5,
        cmap: str = "viridis",
        size: Optional[float] = None,
        color: str = "#3388ff",
        opacity: float = 0.9,
        big: Optional[bool] = None,
        big_data_threshold: Optional[int] = None,
        name: Optional[str] = None,
        visible: bool = True,
        radius: Optional[float] = None,
    ) -> Self:
        """Draw a point ``FeatureCollection`` as a MapLibre circle layer (recipe W2).

        Args:
            features: A pyramids point ``FeatureCollection`` / GeoDataFrame.
            column: Optional value column; when given, circles are coloured by it (graduated if ``scheme``
                is set, else a continuous ramp).
            scheme: A cleopatra classification scheme for graduated colouring (with ``column``).
                ``None`` (the default) is a continuous ramp; a scheme means ``k`` graduated classes.
            k: Number of classes for the graduated schemes.
            cmap: matplotlib colormap for the value colouring.
            size: Circle radius in pixels (``5.0`` when omitted — the signature's ``None`` is the
                "not passed" sentinel the deprecated spelling is resolved against). The same ``size``
                that means marker size on every tier.
            color: Fixed circle colour used when ``column`` is ``None``.
            opacity: Circle fill opacity in ``[0, 1]``.
            big: Big-data routing — ``None`` (default) auto-routes to a GPU deck.gl layer above
                ``big_data_threshold`` (logged); ``False`` forces per-feature circles; ``True`` forces deck.gl.
            big_data_threshold: Feature count above which this one call auto-routes to deck.gl. ``None``
                uses the map's :attr:`~digitalearth.web.base.WebMapBase.big_data_threshold`, which is the
                way to change it for every layer at once.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.
            radius: **Deprecated** spelling of ``size``; forwarded unchanged, after a
                ``DeprecationWarning`` that ``radius=`` will be removed in a future release.

        Returns:
            The same map instance, so builder calls chain.

        Raises:
            TypeError: when ``features`` is a raster rather than a vector layer, or when both ``size``
                and the deprecated ``radius`` are passed — they name one parameter.
            KeyError: when ``column`` names no feature attribute — a MapLibre expression
                reading a property that is not there colours nothing, with no error to explain
                the blank layer.
            ValueError: when the layer routes to a GPU deck.gl overlay — through ``big=True``,
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
        _, layer_types = _require_layer_api()
        size = renamed_parameter(
            new="size",
            value=size,
            old="radius",
            alias=radius,
            caller="WebMap.points()",
            default=5.0,
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
        if column is not None:
            paint["circle-color"] = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        else:
            paint["circle-color"] = color
        return self._vector_layer(
            gdf,
            "circle",
            layer_types.CIRCLE,
            paint,
            kind="points",
            name=name,
            visible=visible,
        )

    def lines(
        self,
        features: Any,
        *,
        column: Optional[str] = None,
        scheme: Optional[Any] = None,
        k: int = 5,
        cmap: str = "viridis",
        width: float = 2.0,
        color: str = "#3388ff",
        opacity: float = 1.0,
        name: Optional[str] = None,
        visible: bool = True,
    ) -> Self:
        """Draw a line ``FeatureCollection`` as a MapLibre line layer (recipe W2).

        Args:
            features: A pyramids line ``FeatureCollection`` / GeoDataFrame.
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
            KeyError: when ``column`` names no feature attribute.

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
        gdf = self._display_gdf(features, method="lines")
        colour = (
            color
            if column is None
            else self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        )
        return self._vector_layer(
            gdf,
            "line",
            layer_types.LINE,
            self._line_paint(width, opacity, colour),
            kind="lines",
            name=name,
            visible=visible,
        )

    def polygons(
        self,
        features: Any,
        *,
        column: Optional[str] = None,
        scheme: Optional[Any] = None,
        k: int = 5,
        cmap: str = "viridis",
        color: str = "#3388ff",
        opacity: float = 0.6,
        outline_color: str = "#ffffff",
        big: Optional[bool] = None,
        big_data_threshold: Optional[int] = None,
        name: Optional[str] = None,
        visible: bool = True,
    ) -> Self:
        """Draw a polygon ``FeatureCollection`` as a MapLibre fill layer (recipe W2).

        Args:
            features: A pyramids polygon ``FeatureCollection`` / GeoDataFrame.
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
            KeyError: when ``column`` names no feature attribute.
            ValueError: when the layer routes to a GPU deck.gl overlay — through ``big=True``,
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
        colour = (
            color
            if column is None
            else self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        )
        return self._vector_layer(
            gdf,
            "fill",
            layer_types.FILL,
            self._fill_paint(opacity, outline_color, colour),
            kind="polygons",
            name=name,
            visible=visible,
        )

    def choropleth(
        self,
        features: Any,
        column: str,
        *,
        scheme: Optional[Any] = None,
        k: int = 5,
        cmap: str = "viridis",
        opacity: float = 0.85,
        outline_color: str = "#ffffff",
        name: Optional[str] = None,
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
            KeyError: when ``column`` is not a feature attribute.
            ValueError: propagated from the classifier (unknown scheme, constant data, …).

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
        gdf = self._display_gdf(features, method="choropleth")
        values = self._require_column(gdf, column)
        paint = {
            "fill-color": self._color_expr(values, column, scheme, k, cmap),
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
        )
