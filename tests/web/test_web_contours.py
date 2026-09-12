"""Contours on the web tier (#193).

pyramids does the tracing (``Dataset.contour``, the ``gdal_contour`` equivalent); this tier only draws what
comes back. Contours are the one field type with a native MapLibre representation — the result is a
``FeatureCollection``, which the vector builders already render.

Lives under ``tests/web/``, which is what the ``test-web`` pixi task runs in the ``web`` env.
"""

import pytest

from digitalearth.web import WebMap


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


class TestWhatItRefuses:
    """Both failures produce an error that names what the caller actually wrote."""

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"interval": 10, "levels": [1.0]}],
    )
    def test_exactly_one_of_interval_or_levels(self, dataset, kwargs):
        """pyramids requires one of the two; saying so here names the argument, not its internals.

        Args:
            dataset: The shared pyramids raster fixture.
            kwargs: Neither given, or both.
        """
        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="exactly one"):
            web_map.contours(dataset, **kwargs)

    def test_an_interval_coarser_than_the_data_says_so(self, dataset):
        """The fixture tops out near 88, so a 1000-unit interval crosses no level at all.

        Test scenario:
            pyramids writes the level attribute only when it writes a feature, so an empty trace reached
            the vector builder as "column 'level' not found" — an error about the wrong thing entirely.
        """
        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="traced nothing"):
            web_map.contours(dataset, interval=1000)
