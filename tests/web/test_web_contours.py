"""Contours on the web tier (#193).

pyramids does the tracing (``Dataset.contour``, the ``gdal_contour`` equivalent); this tier only draws what
comes back. Contours are the one field type with a native MapLibre representation — the result is a
``FeatureCollection``, which the vector builders already render.

Lives under ``tests/web/``, which is what the ``test-web`` pixi task runs in the ``web`` env.
"""

import pytest

from digitalearth.base.crs import OffLimbError
from digitalearth.web import ContourInterval, WebMap


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the web extra is absent."""
    pytest.importorskip("maplibre")


def _payload(html):
    """Return the page's call payload — what this map does, not what the library contains.

    Args:
        html: A page from ``to_html``.

    Returns:
        The substring from ``var data =`` to the end of the page.
    """
    marker = html.rfind("var data = ")
    assert marker != -1, "the exported page carries no call payload"
    return html[marker:]


class TestTracingIsPyramids:
    """The GIS belongs upstream; this tier chooses levels and draws the result."""

    def test_lines_are_drawn_as_a_line_layer(self, dataset):
        """A traced contour is a LineString, so it renders like any other line layer.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = WebMap().basemap().contours(dataset, interval=10)
        payload = _payload(m.to_html())
        assert '"line"' in payload
        assert len(m.layer_ids) == 1

    def test_filled_contours_are_polygons(self, dataset):
        """Bands between levels are areas, not lines.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = WebMap().basemap().contours(dataset, levels=[5.0, 20.0], filled=True)
        payload = _payload(m.to_html())
        assert '"fill"' in payload

    def test_the_band_argument_is_one_based(self, dataset):
        """pyramids counts bands from 0 and this tier from 1; a silent off-by-one would draw the wrong band.

        Test scenario:
            ``band=1`` must reach pyramids as band 0 — the same band ``add_raster(band=1)`` draws.
        """
        seen = {}
        original = type(dataset).contour

        def spy(self, **kwargs):
            """Record the band pyramids was asked for, then trace normally."""
            seen["band"] = kwargs.get("band")
            return original(self, **kwargs)

        type(dataset).contour = spy
        try:
            WebMap().basemap().contours(dataset, interval=10, band=1)
        finally:
            type(dataset).contour = original
        assert seen["band"] == 0, seen

    def test_labels_are_an_extra_layer(self, dataset):
        """Labelling adds a symbol layer over the same features rather than restyling the lines.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = WebMap().basemap().contours(dataset, interval=10, labels=True)
        assert len(m.layer_ids) == 2, m.layer_ids
        assert any("label" in layer_id for layer_id in m.layer_ids)

    def test_a_single_colour_skips_the_classification(self, dataset):
        """With one colour there is nothing to classify, and the level column need not be read.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        m = WebMap().basemap().contours(dataset, interval=10, color="#ff0000")
        assert '"#ff0000"' in _payload(m.to_html())


class TestTheUnitsNameTheLayerThatWasClassified:
    """`units=` describes the contoured values, so it may only reach a key this call produced (#314)."""

    @staticmethod
    def _classifiable_points():
        """Return points whose `value` column a scheme can cut into classes.

        Returns:
            A point `GeoDataFrame` with four distinct values, so `k=2` quantiles have something to split.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        return gpd.GeoDataFrame(
            {"value": [1.0, 2.0, 3.0, 4.0]},
            geometry=[Point(4.9 + index / 10, 52.4) for index in range(4)],
            crs=4326,
        )

    def test_an_explicit_colour_leaves_an_earlier_layer_s_key_alone(self, dataset):
        """A contour layer that classifies nothing must not annotate someone else's colour key.

        Test scenario:
            `contours` set `last_units` and then stamped it onto `last_legend` whenever one existed. With an
            explicit `color=` the layer classifies nothing — `column` is `None` — so `last_legend` still
            belonged to whichever layer classified before it, and that layer's key gained this raster's
            units. The visible effect is one layer's classes labelled in another layer's units.
        """
        web_map = WebMap()
        web_map.points(
            self._classifiable_points(), column="value", scheme="quantiles", k=2
        )
        keyed = web_map.last_legend
        web_map.contours(dataset, interval=10, color="#ff0000", units="m")
        assert "units" not in keyed, (
            f"the points key must keep its own units, got {keyed.get('units')!r}"
        )

    def test_the_units_land_on_the_contour_layer_s_own_key(self, dataset):
        """The units are attributed by layer id, not by whichever key was most recent.

        Test scenario:
            Asking the layer is what makes the attribution right rather than merely non-wrong: the key is
            read back through `_legend_of(layer_id)`, which is the same per-layer store `legend(layer_id=)`
            reads. A fix that only checked whether `last_legend` had moved would pass this too — but it
            would also write the units onto a key whose layer the tier declined to draw, because that key
            is still the most recent one. Reading the layer's own entry cannot.
        """
        web_map = WebMap()
        web_map.points(
            self._classifiable_points(), column="value", scheme="quantiles", k=2
        )
        points_id = web_map.layer_ids[-1]
        web_map.contours(dataset, interval=10, units="m")
        contour_id = web_map.layer_ids[-1]
        assert web_map._legend_of(contour_id).get("units") == "m", (
            "the contour layer's own key names the units"
        )
        assert "units" not in web_map._legend_of(points_id), (
            "the points layer keeps its own key, and it is not measured in the raster's units"
        )

    def test_a_classified_contour_layer_still_names_its_units(self, dataset):
        """The guard must not cost the case it protects: a key this call made does take the units."""
        web_map = WebMap()
        web_map.contours(dataset, interval=10, units="m")
        assert web_map.last_legend is not None, (
            "a classified contour layer records a key"
        )
        assert web_map.last_legend.get("units") == "m", (
            f"its own key names the units, got {web_map.last_legend.get('units')!r}"
        )


class TestTheIntervalCarriesItsAnchor:
    """``interval=`` reads as a plain spacing or as a :class:`ContourInterval` that anchors it."""

    @staticmethod
    def _traced_with(dataset, **kwargs):
        """Contour ``dataset`` and return the arguments pyramids was handed.

        Args:
            dataset: The raster fixture to contour.
            **kwargs: Forwarded to ``contours``.

        Returns:
            The keyword arguments ``Dataset.contour`` received.
        """
        seen = {}
        original = type(dataset).contour

        def spy(self, **passed):
            """Record what pyramids was asked for, then trace normally."""
            seen.update(passed)
            return original(self, **passed)

        type(dataset).contour = spy
        try:
            WebMap().basemap().contours(dataset, **kwargs)
        finally:
            type(dataset).contour = original
        return seen

    def test_a_plain_number_still_means_one_level_every_n(self, dataset):
        """A bare number is the common spelling and reaches pyramids as spacing anchored at zero.

        Args:
            dataset: The raster fixture to contour.

        Test scenario:
            Folding ``base`` into the interval must not make the ordinary call more verbose — the spacing
            is the argument most callers write, and a number can only mean spacing here.
        """
        seen = self._traced_with(dataset, interval=50)
        assert seen["interval"] == 50, (
            f"the spacing must reach pyramids, got {seen['interval']!r}"
        )
        assert seen["base"] == 0.0, (
            f"a bare spacing anchors at zero, got {seen['base']!r}"
        )

    def test_an_anchored_interval_offsets_the_levels(self, dataset):
        """``ContourInterval(spacing, base=)`` carries the anchor through to pyramids.

        Args:
            dataset: The raster fixture to contour.

        Test scenario:
            ``base`` is the value the spacing counts from, so it decides which iso-values are traced at
            all. This type is now the only way to say it, so it is the only thing that can carry it down.
        """
        seen = self._traced_with(dataset, interval=ContourInterval(50, base=25))
        assert seen["interval"] == 50, (
            f"the spacing must survive the type, got {seen['interval']!r}"
        )
        assert seen["base"] == 25, (
            f"the anchor must reach pyramids, got {seen['base']!r}"
        )

    def test_base_can_no_longer_be_passed_where_it_would_be_dropped(self, dataset):
        """A loose ``base=`` is refused outright rather than accepted and ignored.

        Args:
            dataset: The raster fixture to contour.

        Test scenario:
            ``contours(levels=[...], base=50)`` used to be accepted, and pyramids ignored the anchor because
            explicit levels have no spacing to anchor. Silently dropping an argument the caller wrote is the
            failure this signature change removes: the only way to say ``base`` now is beside a spacing.
        """
        scene = WebMap()
        with pytest.raises(TypeError) as excinfo:
            scene.contours(dataset, levels=[100, 200], base=50)
        assert "base" in str(excinfo.value), (
            f"the error must name the argument that no longer exists, got {excinfo.value}"
        )

    @pytest.mark.parametrize("bad", [0, -10, float("nan"), float("inf")])
    def test_a_spacing_that_traces_nothing_is_refused(self, bad):
        """A zero, negative or non-finite spacing raises instead of tracing an empty result.

        Args:
            bad: A spacing that cannot produce levels.

        Test scenario:
            Each of these reaches pyramids as "no features found", which reports the symptom at the wrong
            layer. The type refuses it at the point the caller wrote it.
        """
        with pytest.raises(ValueError, match="finite and greater than zero"):
            ContourInterval(bad)

    def test_a_boolean_is_not_a_spacing_of_one(self, dataset):
        """``interval=True`` is a mis-typed flag, and ``bool`` being an ``int`` must not hide that.

        Args:
            dataset: The raster fixture to contour.

        Test scenario:
            Python makes ``True`` an ``int``, so a plain number check would quietly contour every 1 unit --
            an enormous trace from what was obviously meant as a switch.
        """
        scene = WebMap()
        with pytest.raises(TypeError, match="number or a ContourInterval"):
            scene.contours(dataset, interval=True)


class TestWhatItRefuses:
    """Both failures produce an error that names what the caller actually wrote."""

    def test_both_interval_and_levels_is_an_error(self, dataset):
        """pyramids takes one of the two; saying so here names the argument, not its internals.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="at most one"):
            web_map.contours(dataset, interval=10, levels=[1.0])

    def test_neither_and_no_autostyle_levels_names_both_arguments(self, dataset):
        """With neither given, `auto_style` is consulted — and an unknown variable has no levels (C6).

        Args:
            dataset: The shared pyramids raster fixture.

        Test scenario:
            The fixture's variable is not one the style library knows, so nothing can be resolved and the
            caller is asked for `interval=`/`levels=` rather than handed a guessed set.
        """
        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match=r"needs interval= or levels="):
            web_map.contours(dataset)

    def test_an_interval_coarser_than_the_data_skips_and_warns(
        self, dataset, warning_log
    ):
        """The fixture tops out near 88, so a 1000-unit interval crosses no level at all.

        Args:
            dataset: The shared pyramids raster fixture.
            warning_log: The tier's loguru warnings.

        Test scenario:
            pyramids writes the level attribute only when it writes a feature, so an empty trace reached
            the vector builder as "column 'level' not found" — an error about the wrong thing entirely.
            Under C7 an empty trace is a skip: no layer, a warning that names the interval, and the rest
            of the chain still draws.
        """
        web_map = WebMap().basemap()
        before = len(web_map.layers)
        assert web_map.contours(dataset, interval=1000) is web_map
        assert len(web_map.layers) == before, "an empty trace still added a layer"
        assert any("nothing was traced" in line for line in warning_log), warning_log

    def test_strict_raises_on_an_empty_trace(self, dataset):
        """`strict=True` turns the skip back into the error it used to be (C7).

        It is an ``OffLimbError``, the one type every tier raises under ``strict`` (M6): "no level lies
        within the data" is the same "there is nothing renderable here" the other tiers signal that way,
        and a pipeline written to ``except OffLimbError`` used to miss this tier's bare ``ValueError``.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        web_map = WebMap(strict=True).basemap()
        with pytest.raises(OffLimbError, match="nothing was traced"):
            web_map.contours(dataset, interval=1000)


class TestHiddenContoursHideTheirLabels:
    """Round-1 M1 asked for this on both builders; only the graticule half was done."""

    def test_hiding_the_contours_hides_the_level_labels(self, dataset):
        """Level numbers with no contour under them is not a map.

        Args:
            dataset: The shared pyramids raster fixture.
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .contours(dataset, interval=10, labels=True, visible=False)
        )
        payload = _payload(m.to_html())
        assert payload.count('"visibility": "none"') == 2, (
            "the contours are hidden but their labels are not"
        )


class TestAContourLayerIsDrawnFromItsDescription:
    """Review L3: `contours` and `filled_contours` are drawn from what the figure records."""

    @pytest.mark.parametrize(
        ("filled", "kind", "maplibre_type"),
        [(False, "contours", "line"), (True, "filled_contours", "fill")],
        ids=["contours", "filled_contours"],
    )
    def test_a_contour_layer_redraws_from_the_figure(
        self, dataset, filled, kind, maplibre_type
    ):
        """A figure holding a contour layer is drawable: the renderer has a drawer for its kind.

        Args:
            dataset: The shared pyramids raster fixture.
            filled: Whether the bands between levels are drawn rather than the levels.
            kind: The kind the builder records.
            maplibre_type: The MapLibre layer type the description draws.

        Test scenario:
            Both kinds were declared, but `drawer_for("contours")` refused them, so a figure written with a
            contour layer in it could be read back and never drawn. The redraw here goes through the figure
            alone, and draws what the build drew.
        """
        m = WebMap().contours(dataset, interval=10, filled=filled, name="iso")
        assert m.get_layer("iso").kind == kind, m.get_layer("iso")
        redrawn = m._renderer.draw_layer(m.figure_spec, "iso")
        assert redrawn.layer.type == maplibre_type, redrawn.layer.type
        assert redrawn.layer.paint == m._renderer.drawn["iso"].layer.paint, (
            "the redraw did not draw the recorded paint"
        )

    def test_a_filled_contour_layer_is_drawn_whatever_the_big_data_threshold(
        self, dataset
    ):
        """A contour layer is recorded under its own kind when it is drawn, not relabelled afterwards.

        Args:
            dataset: The shared pyramids raster fixture.

        Test scenario:
            `contours` drew through `polygons` and relabelled whatever `_last_layer_id` named. A flat-colour
            fill over the big-data threshold routed to a deck.gl overlay that records no layer, so the
            relabelling reached for a layer that was not there — or, on a map with layers already, renamed
            the wrong one. Drawn under its own kind, a contour layer is always the MapLibre layer it
            describes.
        """
        m = WebMap().points(_one_point(), name="obs")
        m.big_data_threshold = 1
        m.contours(dataset, interval=10, filled=True, color="#cc4444", name="bands")
        kinds = {layer.id: layer.kind for layer in m.figure_spec.layers}
        assert kinds == {"obs": "points", "bands": "filled_contours"}, kinds


def _one_point():
    """Return one point in EPSG:4326.

    Returns:
        A one-row GeoDataFrame.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame({"value": [1.0]}, geometry=[Point(4.9, 52.4)], crs=4326)
