"""The lat/lon grid a web map can draw for itself (#192).

Named ``test_web_*`` so the ``test-web`` pixi task collects it in the ``web`` env. A graticule is arithmetic
rather than data, so none of this touches the network or a data source.
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
        assert "10°E" in labels and "10°W" in labels
        assert "10°N" in labels and "10°S" in labels
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

    def test_it_sits_under_the_data(self):
        """A grid drawn over a choropleth reads as part of the data."""
        from digitalearth.web import WebMap

        m = WebMap().basemap()
        before = len(m.layers)
        m.graticule()
        assert len(m.layers) == before + 1
        assert m.layers[0] is not None, (
            "the graticule was appended rather than underlaid"
        )

    def test_it_does_not_decide_where_the_map_looks(self):
        """A global grid would frame every map on the whole world."""
        from digitalearth.web import WebMap

        assert WebMap().basemap().graticule()._data_bounds is None

    def test_it_is_a_switchable_layer(self):
        """A viewer who wants the grid off should be able to turn it off."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().graticule()
        assert len(m.layer_ids) == 1
        assert m._layer_index[0][1] == "Graticule", m._layer_index

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
