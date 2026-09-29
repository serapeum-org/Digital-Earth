"""The legend a classified web map carries (#185).

Lives under ``tests/web/``, which is what the ``test-web`` pixi task runs in the ``web`` env. The assertions are about
what reaches the exported page's call payload, not about the bundled MapLibre library — searching the whole
page would match the library's own source and pass either way.
"""

import ast
import inspect
import textwrap

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

    @pytest.mark.parametrize(
        ("call", "named", "other"),
        [("legend", "legend()", "colorbar()"), ("colorbar", "colorbar()", "legend()")],
    )
    def test_the_refusal_names_the_method_that_was_called(self, call, named, other):
        """A caller who wrote `colorbar()` must not be told to fix their `legend()` call.

        Args:
            call: The public method under test.
            named: The spelling its refusal must carry.
            other: The spelling it must not carry.

        Test scenario:
            `colorbar()` is a thin call onto `legend()`, and the "nothing to describe" refusal was written
            by the method underneath — so the one refusal a caller on the `colorbar` spelling actually hits
            named a method they never called and an argument (`labels=`) it does not take. This tier names
            the caller in every other refusal it gives (`WebMap.field()`, `WebMap.extrusion()`); this one
            named whichever of the two wrote the message.
        """
        from digitalearth.web import WebMap

        web_map = WebMap().basemap()
        method = getattr(web_map, call)
        with pytest.raises(ValueError, match="nothing to describe") as refusal:
            method()
        assert named in str(refusal.value), (
            f"{call}()'s refusal does not name it: {refusal.value}"
        )
        assert other not in str(refusal.value), (
            f"{call}()'s refusal points at a method the caller never called: {refusal.value}"
        )

    def test_a_bad_position_is_refused(self, cells):
        """The four corners are MapLibre's, and a typo would be silently ignored by the browser."""
        from digitalearth.web import WebMap

        web_map = WebMap().basemap().choropleth(cells, column="pop")
        with pytest.raises(ValueError):
            web_map.legend(position="middle")

    def test_a_refusal_leaves_the_map_exactly_as_it_was(self, cells):
        """The atomicity `legend()`'s own comment claims, asserted rather than asserted about.

        Args:
            cells: The fixture frame.

        Test scenario:
            `legend()` builds a panel it throws away, so a `labels` list that does not match the
            classification is refused *before* anything is recorded. Only the refusal itself was tested
            (`test_a_short_labels_list_is_refused` checks the `ValueError` and nothing else), so hoisting
            `_attach_guide` above the validating `_legend_panel` call — one plausible way to hoist the
            `labels` check out from under `visible` — would leave the suite green while leaving a guide
            behind whose key every later rebuild would fail to draw (review M13). The state a caller can
            read is what this asserts: the panel on screen, and the layer's own description.
        """
        from digitalearth.web import WebMap

        web_map = (
            WebMap()
            .basemap()
            .choropleth(cells, column="kind", scheme="categorical", name="A")
        )
        web_map.legend(layer_id="A", title="Land cover")
        before_panels = dict(web_map._panels)
        before_symbology = web_map.get_layer("A").symbology
        with pytest.raises(ValueError, match="entries but the classification"):
            web_map.legend(layer_id="A", labels=["only one"])
        assert dict(web_map._panels) == before_panels, (
            f"the refused call changed the key on screen: {web_map._panels.get('legend')}"
        )
        assert web_map.get_layer("A").symbology == before_symbology, (
            f"the refused call was recorded on the layer: {web_map.get_layer('A').symbology}"
        )


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

        m = WebMap().basemap().set_title("<b>Head</b>", subtitle="<i>Sub</i>")
        contents = self._control_contents(m.to_html())
        assert "&lt;b&gt;" in contents[0], f"the heading was not escaped: {contents[0]}"
        assert "&lt;i&gt;" in contents[0], (
            f"the subtitle was not escaped: {contents[0]}"
        )
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


#: Two squares apart, the shape every layer in the guide tests below is drawn from.
_SQUARES = [
    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
]


def _layer(column, values):
    """Return a two-polygon frame carrying one numeric column.

    Args:
        column: The attribute name to classify by.
        values: Its two values, chosen so one layer's range cannot be mistaken for another's.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    return gpd.GeoDataFrame({column: values}, geometry=_SQUARES, crs=4326)


def _whose_key(web_map):
    """Say which layer's classification the map's key is showing.

    Args:
        web_map: The map to read.

    Returns:
        ``"A"`` for the ``pop`` layer's range, ``"B"`` for the ``rain`` layer's, ``None`` when no key is
        drawn. Read off the panel's markup rather than off the guide, so the assertion is about what a viewer
        sees rather than about the state the fix writes.
    """
    panel = web_map._panels.get("legend")
    if panel is None:
        return None
    drawn = panel[0]
    if ">1<" in drawn or ">100<" in drawn:
        return "A"
    if ">500<" in drawn or ">900<" in drawn:
        return "B"
    return "?"


def _two_classified():
    """Return a map with two classified layers, ``A`` below ``B``.

    Returns:
        The map. ``A`` is coloured by ``pop`` over 1..100, ``B`` by ``rain`` over 500..900, so the two keys
        are told apart by the numbers in the panel.
    """
    from digitalearth.web import WebMap

    web_map = WebMap().choropleth(_layer("pop", [1, 100]), column="pop", name="A")
    return web_map.choropleth(_layer("rain", [500, 900]), column="rain", name="B")


class TestTheKeyIsRecordedOnTheLayerItDescribes:
    """Order 24: a colour key is a `Guide` on the layer's `color` encoding, not figure decoration."""

    def test_asking_for_a_key_records_a_guide_on_that_layer(self):
        """`colorbar()` writes the guide the other three tiers write, on the same channel.

        Test scenario:
            The tier drew a panel and recorded nothing, so a figure written out said nothing about the key
            and the map's own description could not be asked which layer was explained. The guide is what
            four tiers now share, so it has to be *on* the layer.
        """
        web_map = _two_classified().colorbar("A", label="People")
        guide = web_map.get_layer("A").symbology.guide()
        assert guide is not None, "asking for a key must record one on the layer"
        assert (guide.show, guide.title, guide.anchor) == (
            True,
            "People",
            "bottom-right",
        ), guide
        assert web_map.get_layer("B").symbology.guide() is None, (
            "a layer nobody keyed must carry no guide"
        )

    def test_position_is_recorded_as_the_guides_anchor(self):
        """`position` and `Guide.anchor` are the same four spellings, so neither is translated.

        Test scenario:
            `FURNITURE_ANCHORS` and `CONTROL_POSITIONS` are **equal** tuples — two objects holding one
            vocabulary, which executed is `==` and not `is` (review N3: the docstrings claimed "the same
            tuple", and `FURNITURE_ANCHORS is CONTROL_POSITIONS` is `False`). Equality is the property the
            method relies on and the one asserted here, so a spelling added to one list and not the other
            fails this test rather than silently needing a translation table between the two.
        """
        from digitalearth.base.controls import CONTROL_POSITIONS
        from digitalearth.base.registry import FURNITURE_ANCHORS

        assert FURNITURE_ANCHORS == CONTROL_POSITIONS, (
            f"the two anchor vocabularies have drifted: {FURNITURE_ANCHORS} vs {CONTROL_POSITIONS}"
        )
        web_map = _two_classified()
        for corner in FURNITURE_ANCHORS:
            web_map.legend(layer_id="A", position=corner)
            assert web_map.get_layer("A").symbology.guide().anchor == corner
            assert web_map._panels["legend"][1] == corner, (
                f"the panel must sit where the guide says: {web_map._panels['legend'][1]!r}"
            )

    def test_visible_false_records_a_guide_that_says_so(self):
        """`visible=False` is a statement about the layer, not a silent no-op.

        Test scenario:
            The flag used to return before anything was recorded, so `legend(visible=False)` left a key
            asked for earlier still on screen. Recording `show=False` says outright that this layer's colour
            is explained by nothing, and the derived panel honours it.
        """
        web_map = _two_classified().colorbar("A", label="People")
        assert _whose_key(web_map) == "A"
        web_map.colorbar("A", visible=False)
        guide = web_map.get_layer("A").symbology.guide()
        assert guide is not None, "visible=False recorded no guide on the layer"
        assert guide.show is False, f"the recorded guide does not say so: {guide}"
        assert _whose_key(web_map) is None, (
            "a guide that says show=False must leave no key on screen"
        )

    def test_hiding_with_no_id_takes_off_the_key_that_is_drawn(self):
        """`colorbar(visible=False)` means "the key on screen, off" — the one on screen, not another.

        Test scenario:
            With no id the target was `keyable[-1]`, the topmost *classified* layer, whatever layer the
            drawn key actually belongs to. So on a map where A is keyed and B (topmost) is classified but
            was never keyed, `colorbar(visible=False)` wrote `Guide(show=False)` onto **B** and left A's key
            exactly where it was: the call did nothing a viewer could see, and silently described a layer
            nobody had asked anything about (review M3). A key is asked for and taken off through the same
            resolution or the two cannot agree.
        """
        web_map = _two_classified().colorbar("A", label="People")
        assert _whose_key(web_map) == "A", "A's key is the one on screen"
        web_map.colorbar(visible=False)
        assert _whose_key(web_map) is None, (
            "the key the call took off is still drawn; a different layer was switched off"
        )
        assert web_map.get_layer("A").symbology.guide().show is False, (
            "the layer whose key was drawn must be the one recorded as unexplained"
        )
        assert web_map.get_layer("B").symbology.guide() is None, (
            "a layer nobody keyed must not gain a guide from a call that named no layer"
        )

    def test_asking_with_no_id_keys_a_layer_the_viewer_can_see(self):
        """An unnamed `colorbar()` means "key what is on the map", so it must draw something.

        Test scenario:
            With no id the target was `keyable[-1]` with visibility never consulted, so on a map whose
            topmost classified layer is hidden the guide went to the hidden layer and the call drew nothing
            at all — a `colorbar()` that returns the map and leaves it exactly as it found it, with the
            visible classified layer below still unexplained (review L6). Naming a hidden layer is a
            different request and still works; this is only about the one the map picks for the caller.
        """
        web_map = _two_classified()
        web_map.set_visible("B", False)
        web_map.colorbar(label="People")
        assert _whose_key(web_map) == "A", (
            "an unnamed colorbar() drew no key; the hidden topmost layer took the guide"
        )
        assert web_map.get_layer("B").symbology.guide() is None, (
            "the hidden layer must not take a key the caller cannot see"
        )

    def test_a_key_asked_for_with_every_layer_hidden_still_lands_somewhere(self):
        """Preferring a visible layer is a preference, not a refusal: a hidden map is not an error.

        Test scenario:
            The other half of the same choice. Refusing when nothing keyable is visible would turn
            `quickmap(..., colorbar=True)` on a map built hidden into a crash, and there is a layer to
            explain — it is simply not on screen yet. The guide is recorded and the key appears with the
            layer, which is the rule a named hidden layer already follows.
        """
        web_map = _two_classified()
        web_map.set_visible("A", False)
        web_map.set_visible("B", False)
        web_map.colorbar(label="Rain")
        assert _whose_key(web_map) is None, "nothing is visible, so no key is drawn"
        web_map.set_visible("B", True)
        assert _whose_key(web_map) == "B", (
            "showing the topmost classified layer must bring the recorded key with it"
        )

    def test_hiding_with_no_id_reaches_a_key_recorded_on_a_hidden_layer(self):
        """Nothing is on screen, but a key was asked for — and asking for none must undo asking for one.

        Test scenario:
            The same defect from the other side, and why "the layer whose key is drawn" cannot be the whole
            rule: a guide on a hidden layer draws nothing yet is still a key the caller asked for, and
            showing the layer brings it back. Targeting only the drawn key would leave that one
            unreachable, so `colorbar(visible=False)` would be a no-op that the next `set_visible` undid.
        """
        web_map = _two_classified().colorbar("B", label="Rain")
        web_map.set_visible("B", False)
        assert _whose_key(web_map) is None, "a hidden layer's key is not drawn"
        web_map.colorbar(visible=False)
        web_map.set_visible("B", True)
        assert _whose_key(web_map) is None, (
            "showing the layer brought back a key the caller had taken off"
        )

    @pytest.mark.parametrize("call", ["legend", "colorbar"])
    @pytest.mark.parametrize("built", ["empty", "classified"])
    def test_taking_off_a_key_nobody_asked_for_is_answered_not_refused(
        self, call, built
    ):
        """`visible=False` on a map carrying no key comes back with the map, under either spelling.

        Args:
            call: The public method under test — one mechanism, two names.
            built: Whether the map holds nothing at all or a classified layer nobody keyed.

        Test scenario:
            The flag decides which question a `None` `layer_id` asks, and only the `True` arm refuses: a
            map with nothing classified cannot be *given* a key, but taking one off a map that has none is
            already true. Both shapes of "no key" go through that arm — an empty map, which is the case
            `WebMap().colorbar(visible=False)` must not crash on, and a classified layer nobody has keyed,
            which is where `colorbar()` succeeds and `colorbar(visible=False)` has nothing to reach. Neither
            may leave a guide behind either, since the resolution is what decides there is nothing to write.
        """
        from digitalearth.web import WebMap

        web_map = WebMap() if built == "empty" else _two_classified()
        answered = getattr(web_map, call)(visible=False)
        assert answered is web_map, (
            f"{call}(visible=False) on a {built} map handed back {answered!r} rather than the map"
        )
        described = [
            layer_id
            for layer_id in web_map.layer_ids
            if web_map.get_layer(layer_id).symbology.guide() is not None
        ]
        assert described == [], (
            f"{call}(visible=False) wrote a guide onto {described} although no layer was keyed"
        )


class TestTheKeyFollowsItsLayer:
    """The deliverable: the panel is derived from the live layers' guides, so it cannot describe a ghost."""

    def test_removing_the_keyed_layer_takes_its_key_with_it(self):
        """The measured defect: the panel went on showing the removed layer's classes.

        Test scenario:
            `remove_layer`'s own `Note:` admitted it — the classification a legend described was cleared only
            when the last layer went, so after removing one layer of several the key still described the
            removed one. A viewer read a ramp for data that is not on the map.
        """
        web_map = _two_classified().colorbar("B", label="Rain")
        assert _whose_key(web_map) == "B"
        web_map.remove_layer("B")
        assert web_map.layer_ids == ["A"], web_map.layer_ids
        assert _whose_key(web_map) is None, (
            "the removed layer's key must not stay on screen; A was never keyed"
        )

    def test_a_surviving_keyed_layer_shows_its_own_key(self):
        """Both keyed, one removed: what is left on screen is the survivor's, not the removed one's.

        Test scenario:
            The other half of the same defect. Clearing the panel would satisfy "not the removed one's"
            without satisfying "the survivor's", so the two are separate assertions on separate maps.
        """
        web_map = _two_classified()
        web_map.colorbar("A", label="People").colorbar("B", label="Rain")
        assert _whose_key(web_map) == "B", "the topmost guided layer is drawn"
        web_map.remove_layer("B")
        assert _whose_key(web_map) == "A", (
            "A carries its own guide and is now topmost, so A's key is drawn"
        )

    def test_hiding_the_keyed_layer_takes_its_key_off_screen(self):
        """A key for a layer that is not drawn labels nothing.

        Test scenario:
            `set_visible(False)` leaves the layer described so a viewer can switch it back on, and used to
            leave its key drawn as well — a panel of classes with no pixels under them.
        """
        web_map = _two_classified().colorbar("B", label="Rain")
        web_map.set_visible("B", False)
        assert _whose_key(web_map) is None, "a hidden layer's key must come off"

    def test_showing_it_again_brings_the_key_back(self):
        """The guide stayed on the layer, so nothing has to be asked for twice.

        Test scenario:
            This is what separates "derived" from "cleared": a panel that was merely deleted could not come
            back, and the caller would have to call `legend()` again after every toggle.
        """
        web_map = _two_classified().colorbar("B", label="Rain")
        web_map.set_visible("B", False)
        web_map.set_visible("B", True)
        assert _whose_key(web_map) == "B", (
            "showing the layer must bring its own key back"
        )

    def test_restyling_a_layer_flat_takes_its_classification_with_it(self):
        """A layer restyled to one flat colour has no classes left, so nothing may still file some for it.

        Test scenario:
            `_legends` is the dict beside the tree that order 24 set out to stop trusting, and
            `replace_layer` never touched it: after swapping a classified layer's description for a
            flat-coloured one, `list(m._legends)` still held its id. The leak is readable from the outside
            in the refusal it produces — `legend(layer_id=...)` resolved through `_legend_of`, found the
            stale entry, accepted the layer, and then failed a level down in `Symbology.with_guide` with
            the base-level "nothing drives the 'color' channel" rather than this tier's own "was not drawn
            with a classification" (review L8). Same refusal for the same reason as a layer that was never
            classified, so the same message and the same place.
        """
        from dataclasses import replace

        from digitalearth.base.spec import Symbology

        web_map = _two_classified().colorbar("B", label="Rain")
        assert _whose_key(web_map) == "B"
        classified = web_map.get_layer("B")
        props = dict(classified.symbology.props)
        props["paint"] = {**props["paint"], "fill-color": "#ff0000"}
        web_map.replace_layer(replace(classified, symbology=Symbology(props=props)))
        assert list(web_map._legends) == ["A"], (
            f"a flat layer is still filed as classified: {list(web_map._legends)}"
        )
        assert _whose_key(web_map) is None, "a flat layer has no classes to key"
        with pytest.raises(ValueError, match="was not drawn with a classification"):
            web_map.legend(layer_id="B")

    def test_the_two_most_recent_accessors_agree_after_a_removal(self):
        """`last_breaks` and `last_legend` are set together by every builder, so they must move together.

        Test scenario:
            `_forget_legend` handed `last_legend` to the most recent *surviving* classification — the whole
            point being that it must never describe a removed layer — and cleared `last_breaks` only when
            the map went empty. So after removing one classified layer of two, `last_legend` described the
            survivor while `last_breaks` still held the removed layer's numbers, and the documented
            raw-numbers accessor answered for data that is not on the map (review L7). The two are the same
            classification read two ways — `last_breaks == last_legend["values"]` holds for all three
            classification shapes — so a reader combining them got a key with one layer's colours and
            another's edges.
        """
        web_map = _two_classified()
        assert web_map.last_legend["column"] == "rain", web_map.last_legend
        assert web_map.last_breaks == web_map.last_legend["values"]
        web_map.remove_layer("B")
        assert web_map.last_legend["column"] == "pop", (
            "last_legend must describe the survivor"
        )
        assert web_map.last_breaks == web_map.last_legend["values"], (
            f"last_breaks {web_map.last_breaks} still describes the removed layer; last_legend has moved "
            f"on to {web_map.last_legend['values']}"
        )

    def test_explicit_labels_survive_a_rebuild(self):
        """The rows a caller named are recorded on the layer, not only in the panel that was built.

        Test scenario:
            The panel is rebuilt whenever the layers change, so anything it is drawn from has to live on the
            layer. Labels held only in the built markup were lost the first time an unrelated layer was
            removed, and the key silently reverted to the derived numbers.
        """
        from digitalearth.web import WebMap

        web_map = WebMap().choropleth(
            _layer("pop", [1, 100]), column="pop", name="A", scheme="quantiles", k=2
        )
        web_map.choropleth(_layer("rain", [500, 900]), column="rain", name="B")
        web_map.legend(layer_id="A", labels=["few", "many"])
        assert ">few<" in web_map._panels["legend"][0]
        web_map.remove_layer("B")
        assert ">few<" in web_map._panels["legend"][0], (
            f"the caller's labels must survive: {web_map._panels['legend'][0][:160]}"
        )


class TestANamedLayerIsCheckedWhateverTheFlagSays:
    """Review L7's rule, applied to the layer as well as to the corner."""

    @pytest.mark.parametrize("visible", [True, False])
    def test_an_id_nobody_drew_is_refused(self, visible):
        """`legend(layer_id="nope", visible=False)` used to be accepted.

        Args:
            visible: Both spellings of the flag.

        Test scenario:
            The corner was checked before the flag was read precisely so one spelling of it was not valid
            half the time. The layer is a caller argument on the same terms, and it was checked half the
            time: a typo'd id under `visible=False` was silently accepted.
        """
        web_map = _two_classified()
        with pytest.raises(KeyError, match="no layer 'nope'"):
            web_map.legend(layer_id="nope", visible=visible)

    @pytest.mark.parametrize("visible", [True, False])
    def test_a_layer_with_a_flat_colour_is_refused(self, visible):
        """A key over a colour nothing varies has no values to label.

        Args:
            visible: Both spellings of the flag.

        Test scenario:
            The same asymmetry one step in: naming an unclassified layer raised under the default and was
            accepted under `visible=False`. And this refusal is what the field-driven encoding buys — a flat
            colour publishes no `color` binding, so there is nothing for a guide to explain.
        """
        from digitalearth.web import WebMap

        web_map = WebMap().choropleth(_layer("pop", [1, 100]), column="pop", name="A")
        web_map.polygons(_layer("pop", [1, 100]), name="plain")
        with pytest.raises(ValueError, match="was not drawn with a classification"):
            web_map.legend(layer_id="plain", visible=visible)

    @pytest.mark.parametrize("visible", [True, False])
    def test_a_labels_list_that_does_not_fit_the_classes_is_refused(self, visible):
        """The third caller argument, checked on the same terms as the corner and the layer.

        Args:
            visible: Both spellings of the flag.

        Test scenario:
            A regression of the rule the two tests above exist for, in the very method whose comment
            explains it: `position` and `layer_id` were hoisted above the flag, and `labels` was left under
            `if visible` — so `legend(labels=["only one"], visible=False)` recorded a one-entry override
            against a two-class key while `legend(labels=["only one"])` raised (review M2). The docstring's
            `Raises:` promises the refusal with no `visible` qualifier, and a caller threading a flag
            through must not be checked half the time.
        """
        from digitalearth.web import WebMap

        web_map = WebMap().choropleth(
            _layer("pop", [1, 100]), column="pop", name="A", scheme="quantiles", k=2
        )
        with pytest.raises(ValueError, match="entries but the classification"):
            web_map.legend(layer_id="A", labels=["only one"], visible=visible)
        assert web_map.get_layer("A").symbology.guide() is None, (
            "a refused call must record nothing, whatever the flag says"
        )


#: Every malformed key description the two public spellings can be handed, with the refusal it must draw.
#: One row per caller argument that has a legal range, spelled the way the method that takes it spells it —
#: ``colorbar()`` has no ``position``/``labels`` of its own and names the heading ``label``.
_MALFORMED = [
    pytest.param(
        "legend", {"title": 123}, "Guide title must be", id="legend-title-number"
    ),
    pytest.param(
        "legend", {"title": "   "}, "Guide title must be", id="legend-title-blank"
    ),
    pytest.param(
        "legend", {"position": "middle"}, "unknown control position", id="legend-corner"
    ),
    pytest.param(
        "legend",
        {"labels": ["only one"]},
        "entries but the classification",
        id="legend-labels",
    ),
    pytest.param(
        "colorbar", {"label": 123}, "Guide title must be", id="colorbar-label-number"
    ),
    pytest.param(
        "colorbar", {"label": "   "}, "Guide title must be", id="colorbar-label-blank"
    ),
]


def _one_classified(name="A"):
    """Return a map holding a single two-class layer and nothing else.

    Args:
        name: The layer's id.

    Returns:
        The map. Two classes is the smallest classification a one-entry ``labels`` list can mismatch, and
        nothing on the map is keyed — the state the flag bypass needed.
    """
    from digitalearth.web import WebMap

    return WebMap().choropleth(
        _layer("pop", [1, 100]), column="pop", name=name, scheme="quantiles", k=2
    )


def _method_source(method):
    """Parse one method and return its signature and its statements, docstring dropped.

    Args:
        method: The function to read. Read from source rather than called, because the property under test
            is *where* in the body a check sits — which no call can observe.

    Returns:
        The ``(ast.arguments, list[ast.stmt])`` pair.
    """
    parsed = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
    body = [
        statement
        for statement in parsed.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]
    return parsed.args, body


def _parameters(args):
    """Return every parameter name of a parsed signature, ``self`` excluded.

    Args:
        args: The ``ast.arguments`` from :func:`_method_source`.

    Returns:
        The names, in declaration order.
    """
    declared = args.posonlyargs + args.args + args.kwonlyargs
    return [arg.arg for arg in declared if arg.arg != "self"]


def _names_in(node):
    """Return every bare name read anywhere under one AST node.

    Args:
        node: Any AST node.

    Returns:
        The set of `ast.Name` ids.
    """
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def _call_to(body, attribute):
    """Find the first statement holding a call to ``self.<attribute>``.

    Args:
        body: The statements from :func:`_method_source`.
        attribute: The method name to look for.

    Returns:
        The ``(index, ast.Call)`` pair, or ``(None, None)`` when no statement holds one.
    """
    for index, statement in enumerate(body):
        for child in ast.walk(statement):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == attribute
            ):
                return index, child
    return None, None


def _first_read_of(body, name):
    """Return the index of the first statement reading ``name``.

    Args:
        body: The statements from :func:`_method_source`.
        name: The parameter to look for.

    Returns:
        The index, or ``len(body)`` when nothing reads it.
    """
    for index, statement in enumerate(body):
        if name in _names_in(statement):
            return index
    return len(body)


def _handed_to(call):
    """Return the bare names handed to one call, positionally or by keyword.

    Args:
        call: The `ast.Call` to read.

    Returns:
        The set of names. A literal argument contributes nothing, which is the point — a parameter
        forwarded as ``None`` is not forwarded.
    """
    handed = set()
    for argument in list(call.args) + [keyword.value for keyword in call.keywords]:
        handed |= _names_in(argument)
    return handed


class TestEveryArgumentIsCheckedWhateverTheFlagSays:
    """Review H2: the same call must be valid or refused on its own merits, never on call order.

    Test scenario:
        This is review L7 / M1 / M2 / H2 — **one** defect, found four times, in four different keywords.
        Each recurrence was fixed by hoisting that one keyword above the flag, and the next one arrived in
        the keyword nobody had hoisted yet. The matrix below is the behavioural half: every argument that
        has a legal range, under both spellings of the flag, on a map where a key had been asked for and on
        one where none ever was — which is the axis all three previous fixes missed, since resolving
        ``visible=False`` against "the key the map already has" answers `None` on a map nobody keyed and
        returns before any argument is looked at.
    """

    @pytest.mark.parametrize("spelling, kwargs, refusal", _MALFORMED)
    @pytest.mark.parametrize("visible", [True, False], ids=["asked", "taken-off"])
    @pytest.mark.parametrize(
        "keyed", [False, True], ids=["never-keyed", "already-keyed"]
    )
    def test_a_malformed_argument_is_refused(
        self, spelling, kwargs, refusal, visible, keyed
    ):
        """One bad argument, both flag values, keyed and unkeyed — twenty-four calls, one rule.

        Args:
            spelling: The public method under test — one mechanism, two names.
            kwargs: The single malformed argument.
            refusal: Part of the message it must carry.
            visible: Both spellings of the flag.
            keyed: Whether some earlier call already keyed the layer.
        """
        web_map = _one_classified()
        if keyed:
            web_map.colorbar("A", label="People")
        before = web_map.get_layer("A").symbology.guide()
        call = getattr(web_map, spelling)
        with pytest.raises(ValueError, match=refusal):
            call(visible=visible, **kwargs)
        assert web_map.get_layer("A").symbology.guide() == before, (
            f"{spelling}({kwargs}, visible={visible}) was refused but still changed the record"
        )

    @pytest.mark.parametrize("visible", [True, False], ids=["asked", "taken-off"])
    def test_a_malformed_title_is_refused_on_a_map_holding_nothing(self, visible):
        """The floor of the rule: a title is a caller argument even where there is no layer to key.

        Args:
            visible: Both spellings of the flag.

        Test scenario:
            ``WebMap().colorbar(visible=False)`` must come back with the map rather than raise — that is the
            documented, doctested point of the flag — and the bypass took cover behind exactly that. The two
            are separable: "this map has nothing to describe" is a fact about the map, while "123 is not a
            heading" is a fact about the call, and only the first is allowed to depend on the flag.
        """
        from digitalearth.web import WebMap

        web_map = WebMap()
        with pytest.raises(ValueError, match="Guide title must be"):
            web_map.colorbar(label=123, visible=visible)

    def test_an_empty_map_still_answers_a_well_formed_take_off(self):
        """The other side of the same line, so the fix above cannot be "refuse everything"."""
        from digitalearth.web import WebMap

        web_map = WebMap()
        assert web_map.colorbar(visible=False) is web_map, (
            "a well-formed take-off on an empty map must still be answered, not refused"
        )


class TestNoArgumentCanBeAddedBelowTheFlag:
    """Review H2's structural half: pin the *shape*, because three hand-written fixes did not hold.

    Test scenario:
        Every earlier fix pinned one keyword's behaviour, and the class of defect — "a caller argument read
        below the flag" — stayed open for the next keyword. These three tests read the source instead: the
        two public spellings must forward every argument they take to the one shared body and do nothing
        else on the way; the shared body must hand every argument to a single check **above** its first read
        of the flag; and the check must actually read each one. A keyword added anywhere in that chain and
        left under the flag reddens here without anybody having thought to write a case for it.
    """

    #: The two ``_record_key`` parameters that describe nothing about the key, so nothing validates them.
    #: ``visible`` is the flag itself, and ``caller`` only spells the public method's name in a refusal.
    NOT_A_DESCRIPTION = frozenset({"visible", "caller"})

    #: The flag whose read the checks must precede.
    FLAG = "visible"

    @pytest.mark.parametrize("spelling", ["legend", "colorbar"])
    def test_a_public_spelling_only_forwards(self, spelling):
        """Each entry point is a pure forward, so the shared body is the only place a check can live.

        Args:
            spelling: The public method under test.
        """
        from digitalearth.web import WebMap

        args, body = _method_source(getattr(WebMap, spelling))
        assert len(body) == 1, (
            f"{spelling}() does {len(body)} things; it must only forward to _record_key, or an argument "
            "could be handled — or missed — here instead"
        )
        _, call = _call_to(body, "_record_key")
        assert call is not None and isinstance(body[0], ast.Return), (
            f"{spelling}() no longer returns a _record_key call"
        )
        assert body[0].value is call, f"{spelling}() does work around the forward"
        missing = set(_parameters(args)) - _handed_to(call)
        assert missing == set(), (
            f"{spelling}() takes {sorted(missing)} and does not forward them to the shared body, so "
            "nothing downstream can check them"
        )

    def test_the_shared_body_checks_every_argument_above_the_flag(self):
        """The defect class as a property of the source: no argument is read after the flag first is.

        Test scenario:
            ``_check_position`` survived three rounds because it sits above the resolution; ``title`` and
            ``labels`` did not, because they sat below an early return the flag reaches. Rather than assert
            that about the three keywords known today, assert it about every parameter the body takes.
        """
        from digitalearth.web.decoration import DecorationMixin

        args, body = _method_source(DecorationMixin._record_key)
        described = set(_parameters(args)) - self.NOT_A_DESCRIPTION
        check_at, check = _call_to(body, "_check_key_arguments")
        assert check is not None, (
            "_record_key no longer runs one check over its arguments"
        )
        assert check_at < _first_read_of(body, self.FLAG), (
            "the argument check is reached after the flag is first read, so a flag value can skip it — "
            "which is review L7/M1/M2/H2, every time"
        )
        missing = described - _handed_to(check)
        assert missing == set(), (
            f"_record_key takes {sorted(missing)} and does not hand them to the check above the flag"
        )

    def test_the_check_reads_every_argument_it_is_handed(self):
        """Threading an argument into the check is not checking it, so the last link is pinned too."""
        from digitalearth.web.decoration import DecorationMixin

        args, body = _method_source(DecorationMixin._check_key_arguments)
        read = set()
        for statement in body:
            read |= _names_in(statement)
        missing = set(_parameters(args)) - read
        assert missing == set(), (
            f"_check_key_arguments is handed {sorted(missing)} and never reads them"
        )
