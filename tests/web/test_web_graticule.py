"""The lat/lon grid a web map can draw for itself (#192).

Lives under ``tests/web/``, which is what the ``test-web`` pixi task runs in the ``web`` env. A graticule is
arithmetic rather than data, so none of this touches the network or a data source.
"""

import pytest


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


class TestTheGridItself:
    """No tile service supplies a graticule, so the lines have to be generated."""

    @pytest.mark.parametrize(
        "spacing, meridians, parallels",
        [(10.0, 36, 17), (30.0, 12, 5), (90.0, 4, 1)],
    )
    def test_the_line_count_follows_the_spacing(self, spacing, meridians, parallels):
        """The grid is arithmetic; a wrong count means the ranges are off by one somewhere.

        Args:
            spacing: Degrees between lines.
            meridians: Expected meridian count — 360/spacing, with the antimeridian drawn once.
            parallels: Expected parallel count over the -80..80 band.
        """
        from digitalearth.web.decoration import _graticule_features

        features = _graticule_features(spacing)["features"]
        lines = [f["geometry"]["coordinates"] for f in features]
        vertical = [c for c in lines if c[0][0] == c[-1][0]]
        horizontal = [c for c in lines if c[0][1] == c[-1][1]]
        assert len(vertical) == meridians, f"{len(vertical)} meridians at {spacing}°"
        assert len(horizontal) == parallels, (
            f"{len(horizontal)} parallels at {spacing}°"
        )

    def test_the_antimeridian_is_drawn_once(self):
        """-180 and +180 are the same line; drawing both doubles it."""
        from digitalearth.web.decoration import _graticule_features

        features = _graticule_features(10.0)["features"]
        longitudes = [
            f["geometry"]["coordinates"][0][0]
            for f in features
            if f["geometry"]["coordinates"][0][0] == f["geometry"]["coordinates"][-1][0]
        ]
        assert longitudes.count(-180.0) == 1
        assert 180.0 not in longitudes

    def test_lines_carry_a_hemisphere_label(self):
        """ "10" alone does not say which side of the meridian it is."""
        from digitalearth.web.decoration import _graticule_features

        labels = {
            f["properties"]["label"] for f in _graticule_features(10.0)["features"]
        }
        assert "10°E" in labels
        assert "10°W" in labels
        assert "10°N" in labels
        assert "10°S" in labels
        assert "0°" in labels, "the prime meridian and equator take no hemisphere"

    def test_meridians_are_sampled_not_straight(self):
        """A projection curves a meridian, which it cannot do with only two vertices."""
        from digitalearth.web.decoration import _graticule_features

        features = _graticule_features(30.0)["features"]
        meridian = next(
            f
            for f in features
            if f["geometry"]["coordinates"][0][0] == f["geometry"]["coordinates"][-1][0]
        )
        assert len(meridian["geometry"]["coordinates"]) > 10


class TestTheGridOnTheMap:
    """Reference geography that only exists in Python is not reference geography."""

    def test_it_reaches_the_saved_page(self):
        """Embedding as GeoJSON is what lets an offline page keep its grid."""
        from digitalearth.web import WebMap

        payload = _payload(WebMap().basemap().graticule(spacing=30.0).to_html())
        assert "graticule" in payload
        assert '"line-color"' in payload

    def test_labels_can_be_turned_off(self):
        """At a tight spacing the labels crowd the map out."""
        from digitalearth.web import WebMap

        with_labels = _payload(WebMap().basemap().graticule().to_html())
        without = _payload(WebMap().basemap().graticule(labels=False).to_html())
        assert "symbol-placement" in with_labels
        assert "symbol-placement" not in without

    def test_it_is_drawn_over_the_basemap_and_under_the_data(self, tmp_path):
        """The band matters in both directions, and only the emitted order proves it.

        Args:
            tmp_path: pytest's per-test directory, unused but keeps the signature uniform.

        Test scenario:
            An underlay puts the grid *beneath* the opaque basemap tiles, where it cannot be seen — the
            method's own docstring example, ``basemap().graticule()``, was exactly that case. Appending it
            instead would draw it over the data. So the assertion is the order of the emitted addLayer
            calls: basemap, then graticule, then data.
        """
        import json
        import re

        import geopandas as gpd
        from shapely.geometry import Point

        from digitalearth.web import WebMap

        points = gpd.GeoDataFrame(
            {"v": [1]}, geometry=[Point(0.0, 0.0)], crs="EPSG:4326"
        )
        web_map = WebMap().basemap().points(points).graticule(name="Grid")
        payload = _payload(web_map.to_html())
        ids = re.findall(r'\["addLayer", \[\{"id": "([^"]+)"', payload)
        assert ids, payload[-400:]
        # By id, not by prefix: the grid also emits a "graticule-label-N" layer, and a prefix match finds
        # that one first — which would leave the line layer's own position unasserted.
        assert "Grid" in ids, f"the grid line layer is missing: {ids}"
        tiles = next(
            i for i, layer_id in enumerate(ids) if layer_id.startswith("tiles")
        )
        circle = next(
            i for i, layer_id in enumerate(ids) if layer_id.startswith("circle")
        )
        assert tiles < ids.index("Grid"), (
            f"the graticule is drawn beneath the basemap: {ids}"
        )
        assert ids.index("Grid") < circle, (
            f"the graticule is drawn over the data: {ids}"
        )

    def test_it_does_not_decide_where_the_map_looks(self):
        """A global grid would frame every map on the whole world."""
        from digitalearth.web import WebMap

        assert WebMap().basemap().graticule()._data_bounds is None

    def test_it_is_a_switchable_layer(self):
        """A viewer who wants the grid off should be able to turn it off."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().graticule()
        assert m.layer_ids == ["Graticule"], m.layer_ids
        payload = _payload(m.layer_control().to_html())
        assert '"layerIds": ["Graticule"]' in payload, payload[-300:]

    @pytest.mark.parametrize("bad", [0.0, -5.0, 200.0])
    def test_an_impossible_spacing_is_refused(self, bad):
        """A grid with no lines in it looks like a bug in the map, not in the call.

        Args:
            bad: A spacing that cannot divide the globe.
        """
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="greater than 0"):
            web_map.graticule(spacing=bad)


class TestTheHiddenGraticuleHidesItsLabels:
    """Round-1 M1 shipped this fix untested; the degree numbers are the visible half."""

    def test_hiding_the_grid_hides_the_degree_labels(self):
        """Otherwise the numbers float over the map with nothing to annotate.

        Test scenario:
            The line layer and the symbol layer are separate MapLibre layers, so `visible=False` has to
            reach both — the switcher and `remove_layer` only ever see the line layer.
        """
        from digitalearth.web import WebMap

        payload = _payload(WebMap().basemap().graticule(visible=False).to_html())
        assert payload.count('"visibility": "none"') == 2, (
            "the grid is hidden but its labels are not"
        )

    def test_a_visible_grid_hides_nothing(self):
        """The default must not write a visibility property that was never asked for."""
        from digitalearth.web import WebMap

        assert '"visibility": "none"' not in _payload(
            WebMap().basemap().graticule().to_html()
        )

    def test_two_graticules_keep_their_call_order(self):
        """Both land in the reference band, and the second should draw over the first.

        Test scenario:
            Inserting each at the band's start reversed them, so a fine grid added after a coarse one
            ended up underneath it.
        """
        import re

        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .graticule(spacing=30.0, name="coarse")
            .graticule(spacing=10.0, name="fine", labels=False)
        )
        ids = re.findall(r'\["addLayer", \[\{"id": "([^"]+)"', _payload(m.to_html()))
        assert ids.index("coarse") < ids.index("fine"), ids
