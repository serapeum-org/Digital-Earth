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


class TestNothingInterpolatedIsMarkup:
    """H3: `InfoBoxControl` assigns its content to `innerHTML`, so every value is markup until escaped."""

    @staticmethod
    def _control_contents(html):
        """Return the `content` string of every InfoBoxControl in the page.

        Args:
            html: A page from ``to_html``.

        Returns:
            The control contents, as they appear in the call payload.
        """
        import re

        return re.findall(
            r'"InfoBoxControl", \{"content": "(.*?)", "cssText"', _payload(html), re.S
        )

    def test_a_hostile_category_value_is_escaped(self):
        """Class values come out of the caller's data — a downloaded shapefile is a realistic source.

        Test scenario:
            The exported page is meant to be emailed or hosted, so an unescaped column value is stored
            XSS in the artifact this tier exists to produce.
        """
        import geopandas as gpd
        from shapely.geometry import Polygon

        from digitalearth.web import WebMap

        hostile = gpd.GeoDataFrame(
            {"kind": ['<img src=x onerror="alert(1)">', "safe"]},
            geometry=[
                Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
                Polygon([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]),
            ],
            crs="EPSG:4326",
        )
        m = (
            WebMap()
            .basemap()
            .choropleth(hostile, column="kind", scheme="categorical")
            .legend()
        )
        contents = self._control_contents(m.to_html())
        assert contents, "no legend control was rendered"
        assert "<img src=x onerror" not in contents[0], contents[0][:200]
        assert "&lt;img" in contents[0], contents[0][:200]

    @pytest.mark.parametrize(
        "kwargs, needle",
        [
            ({"title": "<script>alert(1)</script>"}, "&lt;script&gt;"),
            ({"labels": ["<b>one</b>", "two", "three"]}, "&lt;b&gt;"),
        ],
    )
    def test_caller_strings_are_escaped(self, cells, kwargs, needle):
        """A title and explicit labels are caller input, and reach the same innerHTML sink.

        Args:
            cells: The fixture frame.
            kwargs: The legend argument under test.
            needle: The escaped form that must appear.
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .choropleth(cells, column="kind", scheme="categorical")
            .legend(**kwargs)
        )
        contents = self._control_contents(m.to_html())
        assert needle in contents[0], contents[0][:200]
        assert "<script>" not in contents[0]

    def test_a_title_and_subtitle_are_escaped(self):
        """`title()` shares the sink, so it shares the rule."""
        from digitalearth.web import WebMap

        m = WebMap().basemap().title("<b>Head</b>", subtitle="<i>Sub</i>")
        contents = self._control_contents(m.to_html())
        assert "&lt;b&gt;" in contents[0] and "&lt;i&gt;" in contents[0], contents[0]
        assert "<b>Head</b>" not in contents[0]

    def test_a_short_labels_list_is_refused(self, cells):
        """Zipping a short list against the classes silently omits the rest from the key.

        Args:
            cells: The fixture frame.
        """
        from digitalearth.web import WebMap

        web_map = (
            WebMap().basemap().choropleth(cells, column="kind", scheme="categorical")
        )
        with pytest.raises(ValueError, match="entries but the classification"):
            web_map.legend(labels=["only one"])


class TestTheSwatchColourCannotCarryCss:
    """L4: escaping stops an attribute break-out, but `;` starts a new declaration inside `style=""`."""

    @pytest.mark.parametrize(
        "color", ["#1f77b4", "#abc", "rgb(1, 2, 3)", "rgba(1,2,3,0.5)", "red"]
    )
    def test_real_colours_pass_through(self, color):
        """The classifier's own hex values and CSS keywords must render unchanged.

        Args:
            color: A colour the tier actually produces.
        """
        from digitalearth.web.decoration import _css_color

        assert _css_color(color) == color

    @pytest.mark.parametrize(
        "color", ["red;background:url(x)", 'x"onload=alert(1)', "expression(alert(1))"]
    )
    def test_anything_else_becomes_transparent(self, color):
        """A swatch that renders wrong beats one that smuggles a declaration into the page.

        Args:
            color: A value that is not a plain colour.
        """
        from digitalearth.web.decoration import _css_color

        assert _css_color(color) == "transparent"


class TestTheThemeIsValidated:
    """L8: round-1's theme validation shipped without a test."""

    def test_an_unknown_theme_is_refused(self, cells):
        """A typo should read like `position`'s error, not a pydantic traceback.

        Args:
            cells: The fixture frame.
        """
        from digitalearth.web import WebMap

        web_map = WebMap().basemap().choropleth(cells, column="pop")
        with pytest.raises(ValueError, match="must be one of"):
            web_map.layer_control(theme="fancy")

    @pytest.mark.parametrize("theme", ["default", "simple"])
    def test_both_real_themes_are_accepted(self, cells, theme):
        """The guard must not reject the two styles py-maplibregl ships.

        Args:
            cells: The fixture frame.
            theme: A supported switcher style.
        """
        from digitalearth.web import WebMap

        m = (
            WebMap()
            .basemap()
            .choropleth(cells, column="pop")
            .layer_control(theme=theme)
        )
        assert f'"theme": "{theme}"' in _payload(m.to_html())
