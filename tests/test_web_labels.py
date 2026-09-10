"""Text on a web map (#191) — labels from a column, a single annotation, and the map's own title.

Named ``test_web_*`` so the ``test-web`` pixi task collects it in the ``web`` env. Assertions read the
exported page's call payload, not the page as a whole, which also contains the MapLibre library.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the web extra is absent."""
    pytest.importorskip("maplibre")


@pytest.fixture
def places():
    """Two named points.

    Returns:
        A GeoDataFrame in EPSG:4326 with a ``name`` column.
    """
    return gpd.GeoDataFrame(
        {"name": ["Alpha", "Beta"], "v": [1, 2]},
        geometry=[Point(4.0, 52.0), Point(5.0, 53.0)],
        crs="EPSG:4326",
    )


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


class TestLabelsFromAColumn:
    """MapLibre's symbol layer was reachable only through the cluster count."""

    def test_the_text_is_bound_to_the_column(self, places):
        """A data-driven label reads the property per feature, rather than repeating one string."""
        from digitalearth.web import WebMap

        payload = _payload(WebMap().basemap().labels(places, "name").to_html())
        assert '"symbol"' in payload, "no symbol layer was built"
        assert '"text-field": ["get", "name"]' in payload, payload[-500:]

    def test_a_missing_column_is_refused(self, places):
        """A MapLibre expression on a missing property renders nothing, with nothing to explain it.

        Test scenario:
            The empty map is the whole problem — the failure has to happen in Python, where the caller
            can see which properties exist.
        """
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        with pytest.raises(KeyError, match="not a property"):
            web_map.labels(places, "nope")

    def test_the_styling_reaches_the_layer(self, places):
        """A label over imagery is unreadable without its halo, so none of it may be dropped."""
        from digitalearth.web import WebMap

        payload = _payload(
            WebMap()
            .basemap()
            .labels(
                places,
                "name",
                size=18.0,
                color="#ff0000",
                halo_color="#00ff00",
                halo_width=2.0,
            )
            .to_html()
        )
        assert '"text-size": 18.0' in payload
        assert '"text-color": "#ff0000"' in payload
        assert '"text-halo-color": "#00ff00"' in payload
        assert '"text-halo-width": 2.0' in payload

    def test_an_offset_lifts_the_label_off_its_point(self, places):
        """Without an offset a point label sits on top of the marker it describes."""
        from digitalearth.web import WebMap

        payload = _payload(
            WebMap().basemap().labels(places, "name", offset=(0.0, -1.2)).to_html()
        )
        assert '"text-offset": [0.0, -1.2]' in payload

    def test_overlap_is_refused_by_default(self, places):
        """Letting labels pile up is what makes a dense layer unreadable."""
        from digitalearth.web import WebMap

        payload = _payload(WebMap().basemap().labels(places, "name").to_html())
        assert '"text-allow-overlap": false' in payload

    def test_a_label_layer_joins_the_registry(self, places):
        """Labels are a layer a viewer would want to switch off, so they must be addressable."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().points(places).labels(places, "name", name="Names")
        assert len(m.layer_ids) == 2, m.layer_ids
        assert any("label" in layer_id for layer_id in m.layer_ids), m.layer_ids


class TestASingleAnnotation:
    """The "label this spot" case, which needed a whole GeoDataFrame before."""

    def test_the_string_is_placed_at_the_coordinate(self):
        """One point, one string — no frame to build."""
        from digitalearth.web import WebMap

        payload = _payload(WebMap().basemap().text(4.9, 52.4, "Amsterdam").to_html())
        assert "Amsterdam" in payload
        assert "[4.9, 52.4]" in payload, payload[-400:]

    def test_an_annotation_may_overlap(self):
        """A caller placing one label by hand means it to appear, whatever else is there."""
        from digitalearth.web import WebMap

        payload = _payload(WebMap().basemap().text(4.9, 52.4, "Amsterdam").to_html())
        assert '"text-allow-overlap": true' in payload

    def test_an_annotation_contributes_to_the_framing(self):
        """A map whose only content is an annotation should open on it."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().text(4.9, 52.4, "Amsterdam")
        assert m._data_bounds == [4.9, 52.4, 4.9, 52.4], m._data_bounds


class TestTheMapCarriesItsTitle:
    """An exported page travels alone; the notebook's context does not go with it."""

    def test_the_heading_is_in_the_saved_page(self):
        """Otherwise the title lives only in the surrounding notebook."""
        from digitalearth.web import WebMap

        payload = _payload(WebMap().basemap().title("Population, 2024").to_html())
        assert "InfoBoxControl" in payload
        assert "Population, 2024" in payload

    def test_a_subtitle_is_a_second_line(self):
        """A date or a source belongs under the heading, not inside it."""
        from digitalearth.web import WebMap

        payload = _payload(
            WebMap().basemap().title("Population", subtitle="Source: CBS").to_html()
        )
        assert "Source: CBS" in payload

    def test_a_bad_position_is_refused(self):
        """The four corners are MapLibre's; a typo is silently ignored by the browser."""
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        with pytest.raises(ValueError):
            web_map.title("x", position="middle")
