"""The legend a classified web map carries (#185).

Lives under ``tests/web/``, which is what the ``test-web`` pixi task runs in the ``web`` env. The assertions are about
what reaches the exported page's call payload, not about the bundled MapLibre library — searching the whole
page would match the library's own source and pass either way.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

#: Four unit squares in a row, so a classification has something to split.
CELLS = [Polygon([(i, 0.0), (i + 1, 0.0), (i + 1, 1.0), (i, 1.0)]) for i in range(4)]


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the web extra is absent."""
    pytest.importorskip("maplibre")


@pytest.fixture
def cells():
    """Four polygons carrying a numeric and a categorical column.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    return gpd.GeoDataFrame(
        {"pop": [1, 5, 9, 14], "kind": ["a", "b", "a", "c"]},
        geometry=CELLS,
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


class TestTheClassificationIsRecorded:
    """A key must show the colours that were actually drawn, not a second guess at them."""

    @pytest.mark.parametrize(
        "kwargs, kind",
        [
            ({"scheme": "quantiles", "k": 3}, "graduated"),
            ({"scheme": None}, "continuous"),
            ({"scheme": "categorical", "column": "kind"}, "categorical"),
        ],
    )
    def test_every_classification_shape_records_what_it_drew(self, cells, kwargs, kind):
        """The three expression shapes each need a different key, so each must record its own kind.

        Args:
            cells: The fixture frame.
            kwargs: How to classify.
            kind: The recorded kind expected.
        """
        from digitalearth.web import WebMap

        column = kwargs.pop("column", "pop")
        m = WebMap().basemap().choropleth(cells, column=column, **kwargs)
        assert m.last_legend["kind"] == kind, m.last_legend
        assert m.last_legend["column"] == column
        assert m.last_legend["colors"], "no colours were recorded"

    def test_last_breaks_still_holds_the_raw_numbers(self, cells):
        """`last_breaks` is the documented accessor; adding a richer one must not disturb it."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().choropleth(cells, column="pop", scheme="quantiles", k=3)
        assert m.last_breaks == m.last_legend["values"], (m.last_breaks, m.last_legend)


class TestTheLegendReachesTheSavedPage:
    """The whole point is a key someone else can read, so export is where it counts."""

    @pytest.mark.parametrize(
        "kwargs", [{"scheme": "quantiles", "k": 3}, {"scheme": None}]
    )
    def test_the_rendered_colours_are_in_the_key(self, cells, kwargs):
        """A key drawn from different colours than the map would be worse than none.

        Args:
            cells: The fixture frame.
            kwargs: How to classify.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().choropleth(cells, column="pop", **kwargs).legend()
        payload = _payload(m.to_html())
        assert "InfoBoxControl" in payload, "no legend control was added"
        for color in m.last_legend["colors"]:
            assert color.lower() in payload.lower(), (
                f"{color} is drawn but not in the key"
            )

    def test_the_column_names_the_key_by_default(self, cells):
        """A key with no heading leaves the reader guessing which variable it describes."""
        from digitalearth.web import WebMap

        payload = _payload(
            WebMap().basemap().choropleth(cells, column="pop").legend().to_html()
        )
        assert "pop" in payload

    def test_an_explicit_title_wins(self, cells):
        """The column name is rarely what a reader should see — units belong here."""
        from digitalearth.web import WebMap

        payload = _payload(
            WebMap()
            .basemap()
            .choropleth(cells, column="pop")
            .legend(title="People per km2")
            .to_html()
        )
        assert "People per km2" in payload

    def test_explicit_labels_replace_the_derived_rows(self, cells):
        """Renaming categories is the reason this exists; derived ranges are only a default."""
        from digitalearth.web import WebMap

        payload = _payload(
            WebMap()
            .basemap()
            .choropleth(cells, column="kind", scheme="categorical")
            .legend(labels=["Arable", "Built", "Water"])
            .to_html()
        )
        assert "Arable" in payload
        assert "Built" in payload
        assert "Water" in payload

    def test_a_graduated_key_shows_class_ranges(self, cells):
        """A swatch with no numbers beside it says nothing about where the classes fall.

        Test scenario:
            The row separator is an en dash, which the payload carries JSON-escaped as ``\\u2013`` — the
            page is JSON inside a script tag, not markup, so that is the form to assert on.
        """
        from digitalearth.web import WebMap

        m = WebMap().basemap().choropleth(cells, column="pop", scheme="quantiles", k=3)
        low, high = m.last_breaks[0], m.last_breaks[1]
        payload = _payload(m.legend().to_html())
        assert "\\u2013" in payload, "no class ranges in the key"
        assert f"{low:g} \\u2013 {high:g}" in payload, "the first class range is wrong"

    def test_a_continuous_key_is_a_gradient(self, cells):
        """A ramp has no classes, so rows would misrepresent it."""
        from digitalearth.web import WebMap

        payload = _payload(
            WebMap()
            .basemap()
            .choropleth(cells, column="pop", scheme=None)
            .legend()
            .to_html()
        )
        assert "linear-gradient" in payload, (
            "a continuous ramp was drawn as discrete rows"
        )


class TestTheLegendRefusesWhatItCannotDescribe:
    """An empty box in the corner is worse than being told why there is nothing to show."""

    def test_no_classification_is_an_error(self):
        """`legend()` on an unclassified map has nothing to read."""
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        with pytest.raises(ValueError, match="nothing to describe"):
            web_map.legend()

    def test_a_bad_position_is_refused(self, cells):
        """The four corners are MapLibre's, and a typo would be silently ignored by the browser."""
        from digitalearth.web import WebMap

        web_map = WebMap().basemap().choropleth(cells, column="pop")
        with pytest.raises(ValueError):
            web_map.legend(position="middle")
