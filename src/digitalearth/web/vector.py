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

from typing import TYPE_CHECKING, Any, List, Optional, Self

from loguru import logger

from digitalearth.web.base import _require_layer_api, deprecated_alias

if TYPE_CHECKING:  # pragma: no cover - resolved by the type checker, never at runtime
    from digitalearth.web.base import WebMapBase as _MixinBase
else:  # at runtime the mixin stays a plain class, so the composed MRO is unchanged
    _MixinBase = object


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
            ValueError: propagated from ``cleopatra.styling.styles.classify`` (unknown scheme, no spread, …) or from
                the categorical helper (no non-null values).
        """
        import numpy as np

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

        if isinstance(scheme, str) and scheme.lower() == "categorical":
            from digitalearth.base.symbology import (
                MISSING_COLOR,
                categorical_colors,
                resolve_categorical_cmap,
            )

            categories, colors = categorical_colors(
                values, resolve_categorical_cmap(cmap)
            )
            expr = ["match", ["get", column]]
            for category, color in zip(categories, colors):
                expr.extend([_native(category), color])
            expr.append(
                MISSING_COLOR
            )  # fallback for values outside the known categories (shared by all tiers)
            self.last_breaks = [_native(c) for c in categories]
            self.last_legend = {
                "kind": "categorical",
                "column": column,
                "values": [_native(c) for c in categories],
                "colors": list(colors),
            }
            return expr

        if scheme is not None:
            from cleopatra.styling.styles import classify

            try:
                edges, _ = classify(values, scheme, k)
            except (
                ValueError
            ) as err:  # constant / single-feature column, unknown scheme, k<1, …
                raise ValueError(
                    f"cannot classify column {column!r} (scheme={scheme!r}, k={k}): {err}"
                ) from err
            colors = self._cmap_hex(cmap, len(edges) - 1)
            expr: list = ["step", ["get", column], colors[0]]
            for edge, color in zip(edges[1:-1], colors[1:]):
                expr.extend([float(edge), color])
            self.last_breaks = [float(e) for e in edges]
            self.last_legend = {
                "kind": "graduated",
                "column": column,
                "values": [float(e) for e in edges],
                "colors": list(colors),
            }
            return expr

        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"column {column!r} has no finite values to colour")
        lo, hi = float(finite.min()), float(finite.max())
        if hi <= lo:
            hi = lo + 1.0
        stops = np.linspace(lo, hi, 5)
        colors = self._cmap_hex(cmap, len(stops))
        expr = ["interpolate", ["linear"], ["get", column]]
        for stop, color in zip(stops, colors):
            expr.extend([float(stop), color])
        self.last_breaks = [float(s) for s in stops]
        self.last_legend = {
            "kind": "continuous",
            "column": column,
            "values": [float(s) for s in stops],
            "colors": list(colors),
        }
        return expr

    def labels(
        self,
        features: Any,
        column: str,
        *,
        text_size: float = 12.0,
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
            text_size: Text size in pixels. Named for the text rather than ``size``, which means the
                visual size of a marker everywhere else in the package.
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
            TypeError: when ``features`` is not a vector layer.
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
        _, LayerType = _require_layer_api()
        if size is not None:
            text_size = deprecated_alias("text_size", "size", size)
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
            LayerType.SYMBOL,
            paint,
            name=name,
            visible=visible,
            layout=layout,
        )

    def contours(
        self,
        dataset: Any,
        *,
        interval: Optional[float] = None,
        levels: Optional[Any] = None,
        base: float = 0.0,
        band: int = 1,
        filled: bool = False,
        cmap: Optional[str] = None,
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
            interval: Spacing between levels, anchored at ``base``. Give at most one of this or ``levels``.
            levels: Explicit levels to contour. Give at most one of this or ``interval``. With neither,
                the levels come from :func:`~digitalearth.base.autostyle.auto_style` when the band's
                variable is one it recognises (mean sea-level pressure, 2-m temperature, …) — the levels
                that field is conventionally drawn with.
            base: The value a regular ``interval`` is anchored to.
            band: 1-based band to contour, matching
                :meth:`~digitalearth.web.raster.RasterMixin.add_raster` — pyramids counts bands from 0, and
                this converts, so the same number means the same band everywhere in this tier.
            filled: Draw filled bands between successive levels instead of lines.
            cmap: Colormap for colouring by level; ``None`` resolves the autostyle default for the
                band's variable.
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
        if interval is not None and levels is not None:
            raise ValueError(
                "contours() takes at most one of interval= or levels=; "
                f"got interval={interval!r} and levels={levels!r}"
            )
        data = self._display_raster_or_skip(dataset, layer="contours")
        if data is None:
            return self
        source = self._to_display_source(data, band=band)
        cmap = self._auto_cmap(source, cmap)
        if interval is None:
            levels = self._auto_levels(source, levels)
            if levels is None:
                raise ValueError(
                    "contours() needs interval= or levels=: neither was given, and the band's variable "
                    f"({source.metadata('variable')!r}) is not one auto_style carries levels for."
                )
        # Recorded before the sub-builder runs, so the key it sets can say what the values are measured in.
        self.last_units = self._auto_units(source, None)
        features = data.contour(
            interval=interval,
            fixed_levels=list(levels) if levels is not None else None,
            base=base,
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
                f"{'levels=' + repr(levels) if interval is None else 'interval=' + repr(interval)} "
                f"suits band {band}'s range",
            )
            return self
        # Lines carry `level`; filled bands carry `level_min`/`level_max` for the band's two edges, so
        # colour and label the lower edge — it is what orders the bands.
        attribute = "level_min" if filled else "level"
        column = None if color else attribute
        if filled:
            self.polygons(
                features,
                column=column,
                scheme=None,
                cmap=cmap,
                color=color or "#3388ff",
                opacity=opacity,
                name=name,
                visible=visible,
            )
        else:
            self.lines(
                features,
                column=column,
                scheme=None,
                cmap=cmap,
                color=color or "#3388ff",
                width=width,
                opacity=opacity,
                name=name,
                visible=visible,
            )
        if self.last_units and self.last_legend is not None:
            # The classification the sub-builder just recorded describes this raster's values, so the key
            # can name their unit. Never guessed: `last_units` is only set when auto_style supplied one.
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
        name: Optional[str] = None,
        visible: bool = True,
        layout: Optional[dict] = None,
    ) -> Self:
        """Register a GeoJSON source + a typed layer with ``paint`` and record it as the last data layer.

        Args:
            features: The display-CRS GeoDataFrame to serve as the GeoJSON source.
            prefix: The id prefix / kind tag (``"circle"``/``"line"``/``"fill"``).
            layer_type: The ``maplibre`` ``LayerType`` member for the layer.
            paint: The MapLibre paint dict for the layer.
            name: What a layer switcher calls this layer; ``None`` uses its generated id.
            visible: Whether the layer starts visible, which is what a layer switcher toggles.
            layout: MapLibre layout properties for the layer (a symbol layer's ``text-field`` and its
                placement live here rather than in ``paint``). Merged with the visibility flag.

        Returns:
            The same map instance, so builder calls chain.
        """
        Layer, _ = _require_layer_api()
        src_id = self._uid(f"{prefix}-src")
        layer_id = self._layer_id(prefix, name)
        spec_layout = dict(layout) if layout else {}
        if not visible:
            spec_layout["visibility"] = "none"
        layer = Layer(
            id=layer_id,
            type=layer_type,
            source=src_id,
            paint=paint,
            layout=spec_layout or None,
        )

        def apply(widget: Any) -> None:
            widget.add_source(src_id, features)
            widget.add_layer(layer)

        apply._digitalearth_layer_id = layer_id  # type: ignore[attr-defined]
        self._last_layer_id = layer_id
        self._index_layer(layer_id, name)
        return self.add_layer(apply)

    @staticmethod
    def _require_column(gdf: Any, column: str) -> Any:
        """Return ``gdf[column]`` as a numpy array, raising a clear error when the column is absent."""
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
        size: float = 5.0,
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
            size: Circle radius in pixels — the same ``size`` that means marker size on every tier.
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
            TypeError: when ``features`` is a raster rather than a vector layer.
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
        Layer, LayerType = _require_layer_api()
        if radius is not None:
            size = deprecated_alias("size", "radius", radius)
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
            gdf, "circle", LayerType.CIRCLE, paint, name=name, visible=visible
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
                ['line-2']

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
        Layer, LayerType = _require_layer_api()
        gdf = self._display_gdf(features, method="lines")
        paint: dict = {"line-width": float(width), "line-opacity": float(opacity)}
        if column is not None:
            paint["line-color"] = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        else:
            paint["line-color"] = color
        return self._vector_layer(
            gdf, "line", LayerType.LINE, paint, name=name, visible=visible
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
        Layer, LayerType = _require_layer_api()
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
        paint: dict = {
            "fill-opacity": float(opacity),
            "fill-outline-color": outline_color,
        }
        if column is not None:
            paint["fill-color"] = self._color_expr(
                self._require_column(gdf, column), column, scheme, k, cmap
            )
        else:
            paint["fill-color"] = color
        return self._vector_layer(
            gdf, "fill", LayerType.FILL, paint, name=name, visible=visible
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
        Layer, LayerType = _require_layer_api()
        gdf = self._display_gdf(features, method="choropleth")
        values = self._require_column(gdf, column)
        paint = {
            "fill-color": self._color_expr(values, column, scheme, k, cmap),
            "fill-opacity": float(opacity),
            "fill-outline-color": outline_color,
        }
        return self._vector_layer(
            gdf, "fill", LayerType.FILL, paint, name=name, visible=visible
        )
