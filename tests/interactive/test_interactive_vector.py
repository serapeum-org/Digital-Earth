"""DI.1b — interactive vector builders (points / path / polygons / choropleth).

Element-type, CRS-declaration and styling assertions plus matplotlib-backend render smokes — no
browser, no network. Runs in the ``interactive`` pixi env; every test ``importorskip``s geoviews.
"""

import numpy as np
import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


@pytest.fixture
def point_fc():
    """The point fixture as a pyramids FeatureCollection (EPSG:32618, numeric 'fid')."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def polygon_fc(point_fc):
    """A polygon FeatureCollection built by buffering the point fixture (same CRS, 'fid' column)."""
    fc = point_fc.copy()
    fc["geometry"] = fc.geometry.buffer(500.0)
    return fc


class TestPoints:
    """``points`` — point layers from a FeatureCollection."""

    def test_registers_gv_points_and_chains(self, m, point_fc):
        out = m.points(point_fc)
        assert out is m, "points() must return the map for chaining"
        assert isinstance(m.layers[0], gv.Points), (
            f"expected gv.Points, got {type(m.layers[0])}"
        )

    def test_crs_is_declared_as_display_crs(self, m, point_fc):
        """The element's crs must be the display CRS GeoViews built internally (no re-projection)."""
        m.points(point_fc)
        crs = m.layers[0].crs
        assert "3857" in str(crs) or "Mercator" in type(crs).__name__, (
            f"element CRS should declare the 3857 display CRS, got {crs!r}"
        )

    def test_coordinates_are_reprojected_via_pyramids(self, m, point_fc):
        """Geometry must be in Web-Mercator metres (pyramids to_crs), not the source UTM range."""
        m.points(point_fc)
        x_values = m.layers[0].dimension_values(0)
        expected = point_fc.to_crs(3857).geometry.x.to_numpy()
        assert np.allclose(np.sort(x_values), np.sort(expected)), (
            "point x coordinates must equal the pyramids-reprojected geometry"
        )

    def test_value_column_drives_colour_and_hover(self, m, point_fc):
        m.points(point_fc, value_column="fid", cmap="magma")
        element = m.layers[0]
        assert "fid" in [d.name for d in element.vdims], (
            "value column must be a vdim for hover"
        )
        style = hv.Store.lookup_options("bokeh", element, "style").kwargs
        assert style["color"] == "fid"
        assert style["cmap"] == "magma"

    def test_size_is_recorded(self, m, point_fc):
        m.points(point_fc, size=12.0)
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert style["size"] == 12.0, f"size not honoured: {style.get('size')}"

    def test_already_display_crs_passes_through(self, m, point_fc, monkeypatch):
        """Features already in 3857 must not be reprojected again (pyramids to_crs untouched)."""
        mercator = point_fc.to_crs(3857)

        def _boom(*a, **k):  # pragma: no cover - only fires on regression
            raise AssertionError(
                "to_crs must not run for features already in the display CRS"
            )

        monkeypatch.setattr(type(mercator), "to_crs", _boom)
        m.points(mercator)
        assert isinstance(m.layers[0], gv.Points)

    def test_mpl_render_smoke(self, m, point_fc, tmp_path):
        out = tmp_path / "points.png"
        m.points(point_fc, value_column="fid").save(str(out))
        assert out.exists() and out.stat().st_size > 0


class TestPath:
    """``path`` — line layers."""

    def test_registers_gv_path(self, m, point_fc):
        lines = point_fc.copy()
        lines["geometry"] = point_fc.geometry.shortest_line(
            point_fc.geometry.shift(1).fillna(point_fc.geometry.iloc[0])
        )
        m.path(lines)
        assert isinstance(m.layers[0], gv.Path), (
            f"expected gv.Path, got {type(m.layers[0])}"
        )


class TestPolygonsAndChoropleth:
    """``polygons`` / ``choropleth`` — polygon layers and colour-by-attribute fills."""

    def test_registers_gv_polygons(self, m, polygon_fc):
        m.polygons(polygon_fc)
        assert isinstance(m.layers[0], gv.Polygons), f"got {type(m.layers[0])}"

    def test_outline_only_when_no_column(self, m, polygon_fc):
        m.polygons(polygon_fc)
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert style["fill_alpha"] == 0.0, "no-column polygons must draw outlines only"

    @pytest.mark.parametrize("scheme", [None, "categorical", "quantiles"])
    def test_every_choropleth_scheme_is_described_as_a_choropleth(
        self, polygon_fc, scheme
    ):
        """What a figure says a layer is must not depend on which colouring path drew it.

        Args:
            polygon_fc: Buffered points with a numeric `fid`.
            scheme: The colouring path — the continuous ramp, distinct values, or classes.

        Test scenario:
            The continuous ramp delegated to `polygons()`, which records `polygons`, so the one
            `choropleth()` call a caller makes most often was described as a plain polygon layer while the
            categorical and graduated paths said `choropleth` (review L6).
        """
        interactive_map = InteractiveMap()
        try:
            interactive_map.choropleth(polygon_fc, "fid", scheme=scheme)
            layer = interactive_map.figure_spec.layers.get(interactive_map.layer_ids[0])
            assert layer.kind == "choropleth", layer.kind
        finally:
            interactive_map.close()

    def test_polygons_with_a_column_is_still_described_as_polygons(self, polygon_fc):
        """The other half: `polygons(column=)` fills by an attribute and stays a polygon layer.

        Args:
            polygon_fc: Buffered points with a numeric `fid`.
        """
        interactive_map = InteractiveMap()
        try:
            interactive_map.polygons(polygon_fc, column="fid")
            layer = interactive_map.figure_spec.layers.get(interactive_map.layer_ids[0])
            assert layer.kind == "polygons", layer.kind
        finally:
            interactive_map.close()

    def test_choropleth_colours_by_column(self, m, polygon_fc):
        m.choropleth(polygon_fc, "fid", cmap="plasma", clim=(0.0, 10.0))
        element = m.layers[0]
        assert isinstance(element, gv.Polygons)
        assert "fid" in [d.name for d in element.vdims], (
            "choropleth column must be a vdim"
        )
        style = hv.Store.lookup_options("bokeh", element, "style").kwargs
        assert style["color"] == "fid"
        assert style["cmap"] == "plasma"
        plot = hv.Store.lookup_options("bokeh", element, "plot").kwargs
        assert plot["clim"] == (0.0, 10.0), f"clim not honoured: {plot.get('clim')}"

    def test_choropleth_categorical_scheme(self, m, polygon_fc):
        """scheme='categorical' gives one *discrete* colour per distinct value, recording the categories (DC.8).

        The colour column (``fid``, numeric) is rendered as discrete string labels with a ``{label: colour}``
        dict cmap, so Bokeh colours it categorically rather than interpolating a continuous palette (M1).
        """
        m.choropleth(polygon_fc, "fid", scheme="categorical")
        element = m.layers[0]
        assert isinstance(element, gv.Polygons)
        style = hv.Store.lookup_options("bokeh", element, "style").kwargs
        assert style["color"] == "fid"
        cmap = style["cmap"]
        assert isinstance(cmap, dict), (
            f"categorical cmap should be a label->colour dict, got {cmap!r}"
        )
        assert all(isinstance(c, str) and c.startswith("#") for c in cmap.values()), (
            f"hex colours: {cmap}"
        )
        # one discrete colour per distinct value, and the labels are the (stringified) categories
        n = len(m.last_breaks)
        assert n >= 1, "categories should be recorded"
        assert len(cmap) == n, f"expected {n} discrete colours, got {len(cmap)}"
        assert set(cmap) == {str(c) for c in m.last_breaks}, (
            "cmap keys must be the category labels"
        )
        # the rendered colour column is discrete (string), not the original numeric dtype
        assert element.dimension_values("fid").dtype.kind in ("U", "O"), (
            "colour column must be string-typed"
        )

    def test_choropleth_categorical_missing_values_get_fallback(self, m, polygon_fc):
        """A NaN/None category gets the neutral '#cccccc' fallback in the dict cmap (web parity, L1)."""
        fc = polygon_fc.copy()
        n = len(fc)
        kinds = [("a", "b")[i % 2] for i in range(n)]
        kinds[0] = None
        fc["kind"] = kinds
        m.choropleth(fc, "kind", scheme="categorical")
        cmap = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs["cmap"]
        assert cmap.get("n/a") == "#cccccc", (
            f"missing values must map to the neutral fallback: {cmap}"
        )
        assert {"a", "b"} <= set(cmap), (
            f"real categories must still be coloured: {cmap}"
        )

    def test_choropleth_categorical_real_na_not_clobbered(self, m, polygon_fc):
        """A genuine 'n/a' category keeps its colour; missing rows use a collision-free sentinel (L3)."""
        fc = polygon_fc.copy()
        n = len(fc)
        kinds = [("n/a", "b")[i % 2] for i in range(n)]
        kinds[0] = None
        fc["kind"] = kinds
        m.choropleth(fc, "kind", scheme="categorical")
        cmap = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs["cmap"]
        assert cmap.get("n/a") not in (None, "#cccccc"), (
            f"the real 'n/a' category must keep its colour: {cmap}"
        )
        assert cmap.get("n/a_") == "#cccccc", (
            f"missing rows must use a distinct sentinel: {cmap}"
        )

    def test_choropleth_missing_column_raises(self, m, polygon_fc):
        with pytest.raises(KeyError, match="nope"):
            m.choropleth(polygon_fc, "nope")

    def test_choropleth_mpl_render_smoke(self, m, polygon_fc, tmp_path):
        out = tmp_path / "choropleth.png"
        m.choropleth(polygon_fc, "fid").save(str(out))
        assert out.exists() and out.stat().st_size > 0


#: Each vector builder call, by the name a failure reports it under, as `(builder, geometry, kwargs)`.
_ONE_WARP_CALLS = {
    "points": ("points", "point", {}),
    "points-graduated": (
        "points",
        "point",
        {"value_column": "fid", "scheme": "quantiles", "k": 3},
    ),
    "points-categorical": (
        "points",
        "point",
        {"value_column": "fid", "scheme": "categorical"},
    ),
    "polygons": ("polygons", "polygon", {}),
    "polygons-column": ("polygons", "polygon", {"column": "fid"}),
    "choropleth": ("choropleth", "polygon", {"column": "fid"}),
    "choropleth-categorical": (
        "choropleth",
        "polygon",
        {"column": "fid", "scheme": "categorical"},
    ),
    "choropleth-graduated": (
        "choropleth",
        "polygon",
        {"column": "fid", "scheme": "quantiles", "k": 3},
    ),
}


class TestAVectorLayerIsReprojectedOnce:
    """The drawer reprojects what it draws, so the builder must not warp the same frame first."""

    @pytest.mark.parametrize("call", sorted(_ONE_WARP_CALLS))
    def test_one_builder_call_warps_its_features_once(
        self, call, point_fc, polygon_fc, monkeypatch
    ):
        """A builder that describes its layer needs the attribute values and the row count, not a warp.

        Args:
            call: Which builder call to count, a key of `_ONE_WARP_CALLS`.
            point_fc: The point fixture, in EPSG:32618 so the Web-Mercator map has to reproject it.
            polygon_fc: The same points buffered into polygons.
            monkeypatch: Used to count calls to the tier's vector reprojection.

        Test scenario:
            Since the seam, the drawer reprojects what it draws. The builders kept their own
            `_display_gdf` too — for a row count and a column's values, which a warp does not change — and
            discarded the result, so every one of these warped the frame twice (review M8).
        """
        from digitalearth.interactive import vector

        warps = []
        real = vector.reproject

        def counting(dataset, crs):
            """Count the call and warp as the tier would.

            Args:
                dataset: What is being warped.
                crs: Where to.

            Returns:
                The warped dataset.
            """
            warps.append(crs)
            return real(dataset, crs)

        monkeypatch.setattr(vector, "reproject", counting)
        builder, geometry, kwargs = _ONE_WARP_CALLS[call]
        features = point_fc if geometry == "point" else polygon_fc
        interactive_map = InteractiveMap()
        try:
            getattr(interactive_map, builder)(features, **kwargs)
            assert len(interactive_map.layers) == 1, "the layer was not drawn"
            assert len(warps) == 1, f"{call} warped its features {len(warps)} times"
        finally:
            interactive_map.close()


class TestRasterVectorCompose:
    """Raster + vector layers compose into one overlay (the DI.1 headline)."""

    def test_image_plus_choropleth_overlay(self, m, dataset, polygon_fc):
        m.image(dataset).choropleth(polygon_fc, "fid")
        overlay = m.render()
        assert isinstance(overlay, hv.Overlay)
        assert len(overlay) == 2, (
            f"expected 2 layers in the overlay, got {len(overlay)}"
        )


class TestGraduatedPoints:
    """``points(scheme=...)`` classifies through the same path a classified polygon layer uses."""

    def test_a_classified_point_layer_records_its_breaks(self, m, point_fc):
        """``points(value_column=, scheme=)`` colours by class and records the edges on ``last_breaks``.

        Args:
            m: The map under test.
            point_fc: A point ``FeatureCollection`` fixture.

        Test scenario:
            The tier took neither ``scheme`` nor ``k`` while the web tier took both, so a classified point
            layer was impossible here and routine there. It now goes through the one classifier this tier
            uses for polygons, which is what makes the edges — and so the legend — agree between the two.
        """
        m.points(point_fc, value_column="fid", scheme="quantiles", k=3)
        assert m.last_breaks is not None, (
            f"a classified layer must record its edges, got {m.last_breaks!r}"
        )
        assert len(m.last_breaks) == 4, (
            f"3 classes must record 4 edges, got {m.last_breaks!r}"
        )
        assert all(isinstance(edge, float) for edge in m.last_breaks), (
            f"the edges must be plain floats for the legend to render, got {m.last_breaks!r}"
        )

    def test_no_scheme_leaves_the_points_on_a_continuous_ramp(self, m, point_fc):
        """``scheme=None`` keeps the previous behaviour: one colour dimension, no class edges.

        Args:
            m: The map under test.
            point_fc: A point ``FeatureCollection`` fixture.

        Test scenario:
            The classification is opt-in on every tier, so the default must not start binning a column that
            a caller expects to read as a continuous ramp.
        """
        m.points(point_fc, value_column="fid")
        assert not m.last_breaks, (
            f"an unclassified layer must record no breaks, got {m.last_breaks!r}"
        )


class TestGraduatedChoropleth:
    """#245 — graduated schemes classify here too, matching the web tier's classifier."""

    @pytest.fixture
    def m(self) -> InteractiveMap:
        return InteractiveMap()

    @pytest.fixture
    def polygon_fc(self):
        from pyramids.feature import FeatureCollection

        fc = FeatureCollection.read_file("tests/data/points.geojson").copy()
        fc["geometry"] = fc.geometry.buffer(500.0)
        return fc

    def test_quantiles_renders_instead_of_raising(self, m, polygon_fc):
        """The call that raised NotImplementedError now builds a graduated polygon layer."""
        out = m.choropleth(polygon_fc, "fid", scheme="quantiles", k=5)
        assert out is m, "choropleth() must return the map for chaining"
        assert isinstance(m.layers[0], gv.Polygons), (
            f"expected gv.Polygons, got {type(m.layers[0])}"
        )
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        plot = hv.Store.lookup_options("bokeh", m.layers[0], "plot").kwargs
        assert isinstance(style["cmap"], list), (
            f"a graduated layer needs a list of flat colours: {style['cmap']}"
        )
        assert len(style["cmap"]) == 5, (
            f"one flat colour per class expected: {style['cmap']}"
        )
        assert len(plot["color_levels"]) == 6, (
            f"k+1 class edges expected: {plot['color_levels']}"
        )
        assert type(hv.renderer("bokeh").get_plot(m.layers[0])).__name__.endswith(
            "PolygonPlot"
        ), "the graduated layer must build a real polygon plot"

    def test_breaks_match_the_shared_classifier(self, m, polygon_fc):
        """``last_breaks`` holds the edges cleopatra's classify returns — the web tier's contract."""
        from cleopatra.styling.styles import classify

        m.choropleth(polygon_fc, "fid", scheme="quantiles", k=5)
        edges, _ = classify(polygon_fc.to_crs(3857)["fid"].to_numpy(), "quantiles", 5)
        assert m.last_breaks == [float(edge) for edge in edges], (
            f"breaks must match the shared classifier: {m.last_breaks}"
        )

    @pytest.mark.parametrize("scheme", ["equal_interval", "fisher_jenks"])
    def test_other_schemes_classify_too(self, m, polygon_fc, scheme):
        m.choropleth(polygon_fc, "fid", scheme=scheme, k=4)
        assert m.last_breaks is not None, (
            f"{scheme} must record its class edges: {m.last_breaks}"
        )
        assert len(m.last_breaks) == 5, (
            f"{scheme} must produce k+1 edges: {m.last_breaks}"
        )

    def test_k_controls_the_class_count(self, m, polygon_fc):
        m.choropleth(polygon_fc, "fid", scheme="quantiles", k=3)
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert len(style["cmap"]) == 3, f"k=3 must give 3 colours: {style['cmap']}"
        assert len(m.last_breaks) == 4, f"k=3 must give 4 edges: {m.last_breaks}"

    def test_explicit_clim_reaches_a_graduated_layer(self, m, polygon_fc):
        """``clim=`` is forwarded on the graduated path, and a caller's own opts still win over it.

        Args:
            m: The map under test.
            polygon_fc: The buffered point fixture, classified on ``fid``.

        Test scenario:
            A caller pinning the colour limits — to hold one scale across several maps — passes ``clim``
            the same way on every path. The graduated branch builds its own style dict, so the limits
            have to be merged into it rather than dropped on the way to the classifier.
        """
        m.choropleth(polygon_fc, "fid", scheme="quantiles", k=3, clim=(0.0, 10.0))
        options = {
            **hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs,
            **hv.Store.lookup_options("bokeh", m.layers[0], "plot").kwargs,
        }
        assert options.get("clim") == (0.0, 10.0), (
            f"clim must reach the graduated layer, got {options.get('clim')!r}"
        )

    def test_unknown_scheme_errors_with_context(self, m, polygon_fc):
        """An unclassifiable request names the column, the scheme and k (web-tier parity)."""
        with pytest.raises(ValueError, match=r"cannot classify column 'fid'"):
            m.choropleth(polygon_fc, "fid", scheme="not_a_scheme")

    def test_categorical_and_continuous_are_unchanged(self, m, polygon_fc):
        """The two schemes that already worked keep their exact behaviour."""
        categorical = InteractiveMap().choropleth(
            polygon_fc, "fid", scheme="categorical"
        )
        cmap = hv.Store.lookup_options("bokeh", categorical.layers[0], "style").kwargs[
            "cmap"
        ]
        assert isinstance(cmap, dict), "categorical still maps label -> colour"
        m.choropleth(polygon_fc, "fid")
        assert m.last_breaks is None, "the continuous ramp still records no breaks"
        assert (
            hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs["cmap"]
            == "viridis"
        ), "the continuous ramp still takes the cmap name straight through"


class TestTheClassifiedPointLayerRefusesWhatItCannotHonour:
    """Round-2 review M2/M3/M4/M5 — the newly shared classification, on the tier that added it."""

    def test_a_scheme_without_a_value_column_is_refused(self, m, point_fc):
        """M2 — ``points(scheme=...)`` with nothing to classify was accepted and silently dropped.

        Args:
            m: The map under test.
            point_fc: A point ``FeatureCollection`` fixture.

        Test scenario:
            A dropped styling request, on the same branch that added ``_reject_unsupported`` specifically so
            requests are refused rather than dropped. The message has to name ``value_column``, because that
            is the half the caller has to add.
        """
        with pytest.raises(ValueError) as excinfo:
            m.points(point_fc, scheme="quantiles", k=2)
        message = str(excinfo.value)
        assert "value_column=" in message, (
            f"the message must name the half the caller has to add: {message}"
        )
        assert "scheme=" in message, (
            f"the message must name the request that was refused: {message}"
        )
        assert not m.layers, "nothing may be drawn when the request was refused"

    def test_a_categorical_scheme_colours_points_by_distinct_value(self, m, point_fc):
        """M3 — ``points(scheme="categorical")`` died in the float coercion every other tier avoids.

        Args:
            m: The map under test.
            point_fc: A point ``FeatureCollection`` fixture.

        Test scenario:
            The same defect class as the ``extruded_polygons`` crash fixed earlier: classification ran after
            a float coercion, so a label column raised "could not convert string to float". It now takes the
            same distinct-value path ``choropleth(scheme="categorical")`` takes, so both key a column alike.
        """
        fc = point_fc.copy()
        fc["label"] = [("a", "b", "c")[index % 3] for index in range(len(fc))]
        m.points(fc, value_column="label", scheme="categorical")
        style = m.style_of(m.layers[0])["common"]
        assert isinstance(style["cmap"], dict), (
            f"a categorical point layer maps label -> colour, got {style['cmap']!r}"
        )
        assert style["colorbar"] is False, (
            "a categorical layer is keyed by swatches, not a colorbar"
        )
        assert m.last_breaks == sorted({"a", "b", "c"}), m.last_breaks

    def test_a_categorical_point_layer_matches_the_polygon_one(self, m, point_fc):
        """M3 — the point and polygon categorical paths must agree on the colours they assign.

        Args:
            m: The map under test.
            point_fc: A point ``FeatureCollection`` fixture.

        Test scenario:
            One classifier is the point of the shared helper; two implementations would let a point layer
            and a polygon layer over the same column disagree on which colour a category gets.
        """
        fc = point_fc.copy()
        fc["label"] = [str(value % 3) for value in range(len(fc))]
        polygons = fc.copy()
        polygons["geometry"] = polygons.geometry.buffer(500.0)
        m.points(fc, value_column="label", scheme="categorical")
        other = InteractiveMap().choropleth(polygons, "label", scheme="categorical")
        point_cmap = m.style_of(m.layers[0])["common"]["cmap"]
        polygon_cmap = other.style_of(other.layers[0])["common"]["cmap"]
        assert point_cmap == polygon_cmap, (point_cmap, polygon_cmap)

    @pytest.mark.parametrize("builder", ["points", "choropleth"])
    def test_an_explicit_option_outranks_the_classifier_on_both_builders(
        self, m, point_fc, polygon_fc, builder
    ):
        """M4 — the shared classifier had opposite kwarg precedence on its two callers.

        Args:
            m: The map under test.
            point_fc: A point ``FeatureCollection`` fixture.
            polygon_fc: A polygon ``FeatureCollection`` fixture.
            builder: Which classified builder to exercise.

        Test scenario:
            ``points()`` let the classifier win over ``**opts`` while ``_graduated_polygons()`` let the
            caller win, so ``colorbar=False`` was honoured on a classified choropleth and ignored on a
            classified point layer. The caller's explicit value wins on both.
        """
        if builder == "points":
            m.points(
                point_fc, value_column="fid", scheme="quantiles", k=2, colorbar=False
            )
        else:
            m.choropleth(polygon_fc, "fid", scheme="quantiles", k=2, colorbar=False)
        style = m.style_of(m.layers[0])["common"]
        assert style["colorbar"] is False, (
            f"an explicit colorbar=False must survive classification, got {style!r}"
        )

    @pytest.mark.parametrize("colours", [["#ff0000", "#00ff00"], ["#ff0000"] * 9])
    def test_a_colour_list_that_does_not_match_the_class_count_is_refused(
        self, m, polygon_fc, colours
    ):
        """M5 — a short colour list collapsed classes onto one colour with no warning.

        Args:
            m: The map under test.
            polygon_fc: A polygon ``FeatureCollection`` fixture.
            colours: An explicit colour sequence that does not carry one colour per class.

        Test scenario:
            ``sample_cmap`` takes a sequence as given, so the class edges and the colour list could disagree
            silently — three of five classes rendering identically. One colour per class, or a colormap
            name to sample; anything else is refused by name.
        """
        with pytest.raises(ValueError) as excinfo:
            m.choropleth(polygon_fc, "fid", scheme="quantiles", k=5, cmap=colours)
        message = str(excinfo.value)
        assert "one colour per class" in message, message
        assert f"{len(colours)} colours for 5 classes" in message, message

    def test_a_colour_list_of_exactly_k_is_honoured(self, m, polygon_fc):
        """The guard must not disturb the deliberate case it protects.

        Args:
            m: The map under test.
            polygon_fc: A polygon ``FeatureCollection`` fixture.
        """
        colours = ["#ff0000", "#00ff00", "#0000ff"]
        m.choropleth(polygon_fc, "fid", scheme="quantiles", k=3, cmap=colours)
        style = m.style_of(m.layers[0])["common"]
        assert style["cmap"] == colours, style["cmap"]
