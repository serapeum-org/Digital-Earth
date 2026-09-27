"""Naming and hiding an interactive layer as it is built (#321, #327).

`InteractiveMap.add_layer` has taken `name=` and `visible=` since the tier had a layer tree; **no builder
passed either**. So `points(name="wells")` put the string into `**opts`, where HoloViews took it as a style
option, and `points(visible=False)` did the same with the flag — which HoloViews *does* honour, so the element
was drawn hidden while the figure went on describing the layer `visible=True`. A layer switcher reading that
figure offered a ticked box for a layer nobody could see (#327).

These hold the builder side of both on this tier, and the half the description alone cannot show: that the
element really is hidden. The cross-tier probes are in `tests/base/test_map_conformance.py`, which is also
where the interactive tier came off `CANNOT_HIDE_AT_BUILD`.
"""

import pytest

from digitalearth.interactive import InteractiveMap

pytest.importorskip("holoviews")
pytest.importorskip("geoviews")

#: The name the probes ask for, and the id a second layer asking for it again must be given.
ASKED = "wells"
SUFFIXED = "wells-2"

#: The column the fixture carries, so a classified builder has something to colour by.
COLUMN = "fid"


@pytest.fixture
def drawn():
    """Yield a fresh Web-Mercator map, closed on the way out.

    Yields:
        An empty `InteractiveMap`.
    """
    built = InteractiveMap()
    yield built
    built.close()


@pytest.fixture
def point_fc():
    """Return the committed point fixture as a pyramids `FeatureCollection`.

    Returns:
        Points in EPSG:32618 with a numeric `fid` column.
    """
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def polygon_fc(point_fc):
    """Return polygons built by buffering the point fixture.

    Args:
        point_fc: The point fixture.

    Returns:
        Polygons in the same CRS, carrying the same `fid` column.
    """
    buffered = point_fc.copy()
    buffered["geometry"] = buffered.geometry.buffer(500.0)
    return buffered


class TestTheNameACallerGives:
    """`name=` reaches the layer id rather than HoloViews' styling (#321)."""

    def test_a_point_layer_is_filed_under_the_name(self, drawn, point_fc):
        """`points(name=)` is the id `layer_ids` reports.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
        """
        drawn.points(point_fc, name=ASKED)
        assert drawn.layer_ids == [ASKED], drawn.layer_ids

    def test_a_decoration_layer_is_too(self, drawn):
        """`graticule(name=)` names a reference layer, which is a layer like any other.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(spacing=30.0, name=ASKED)
        assert drawn.layer_ids == [ASKED], drawn.layer_ids

    def test_an_unnamed_layer_is_still_numbered_by_its_kind(self, drawn, point_fc):
        """The default is unchanged: no name asked for, an id generated from the kind.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
        """
        drawn.points(point_fc)
        assert drawn.layer_ids == ["points-1"], drawn.layer_ids

    def test_the_suffix_counts_the_name_and_not_the_map(self, drawn, point_fc):
        """The second `wells` is `wells-2` however many other layers were drawn first.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.

        Test scenario:
            This tier minted a colliding name from the map-wide id counter, so three unnamed layers ahead of
            the reuse made it `wells-4` — while the web and 3-D tiers answered `wells-2` to the same script.
            The three unnamed layers are what make the two rules give different answers.
        """
        for _ in range(3):
            drawn.points(point_fc)
        drawn.points(point_fc, name=ASKED)
        drawn.points(point_fc, name=ASKED)
        assert drawn.layer_ids[-2:] == [ASKED, SUFFIXED], drawn.layer_ids


class TestANameThatIsWhatTheGeneratorWouldMint:
    """The collision the suffix rule cannot answer: the caller asks for the id first.

    `points(name="points-1")` takes the very id the next unnamed `points` would generate, so the generator
    has to step over what is already issued rather than count blindly. What it costs when it does not is
    not a cosmetic clash: `Renderer._drawn` maps layer id to the element drawn for it, so a re-minted id
    replaces the entry and the first element is left on the overlay with nothing able to reach it —
    `set_visible`, `is_visible` and `remove_layer` all key on that id.
    """

    def test_an_unnamed_layer_steps_over_the_id_a_caller_already_took(
        self, drawn, point_fc
    ):
        """The generator skips `points-1` because the caller holds it, and mints `points-2`.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.

        Test scenario:
            This tier counts a point layer under `points`, so the first generated id is `points-1` — asked
            for by name here before any unnamed layer exists. Without the skip the second call re-mints it
            and the figure describes two layers under one id.
        """
        drawn.points(point_fc, name="points-1")
        drawn.points(point_fc)
        assert drawn.layer_ids == ["points-1", "points-2"], drawn.layer_ids

    def test_each_layer_is_still_the_only_element_its_id_toggles(self, drawn, point_fc):
        """The half the id list cannot show: two ids, two separately switchable elements.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.

        Test scenario:
            Asked of the renderer rather than the description, because that is where a re-minted id does
            its damage — one entry where two elements were drawn. Hiding the caller's own layer and reading
            the generated one back off the element is what separates the two.
        """
        drawn.points(point_fc, name="points-1")
        drawn.points(point_fc)
        drawn._renderer.set_visible("points-1", False)
        assert drawn._renderer.is_visible("points-2") is True, drawn.layer_ids


class TestALayerBuiltHidden:
    """#327 — `visible=False` hid the element and described the layer visible."""

    def test_points_describes_the_layer_hidden(self, drawn, point_fc):
        """The reported half of the bug: the figure said `visible=True`.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
        """
        drawn.points(point_fc, name=ASKED, visible=False)
        assert drawn.figure_spec.layers.get(ASKED).visible is False

    def test_points_hides_the_element_too(self, drawn, point_fc):
        """The half that always worked, asserted so a fix cannot trade one for the other.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.

        Test scenario:
            The flag reached HoloViews through `**opts` before, which is exactly why the element was hidden
            while the description was not. Moving it out of `**opts` could have hidden nothing at all, so
            the drawing is checked as well as the description.
        """
        drawn.points(point_fc, name=ASKED, visible=False)
        assert drawn._renderer.is_visible(ASKED) is False

    def test_polygons_describes_the_layer_hidden(self, drawn, polygon_fc):
        """`choropleth` and `polygons` were the same bug, so they carry the same probe.

        Args:
            drawn: The map under test.
            polygon_fc: The polygon fixture.
        """
        drawn.polygons(polygon_fc, name=ASKED, visible=False)
        assert drawn.figure_spec.layers.get(ASKED).visible is False

    def test_choropleth_describes_the_layer_hidden(self, drawn, polygon_fc):
        """The classified builder reaches the funnel by its own path, so it is asked separately.

        Args:
            drawn: The map under test.
            polygon_fc: The polygon fixture.
        """
        drawn.choropleth(polygon_fc, COLUMN, name=ASKED, visible=False)
        assert drawn.figure_spec.layers.get(ASKED).visible is False

    def test_a_layer_nobody_hid_stays_visible(self, drawn, point_fc):
        """The complement, so the probes above cannot pass by hiding everything.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
        """
        drawn.points(point_fc, name=ASKED)
        assert drawn._renderer.is_visible(ASKED) is True


class TestTheSquareGraticuleShorthand:
    """`graticule(spacing=)` — the web tier's shorthand, now spelled the same here (#324)."""

    def test_one_spacing_sets_both_steps(self, drawn):
        """`spacing=10` is `lon_step=10, lat_step=10`, which is what the contract declares it means.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(spacing=10.0, name=ASKED)
        assert drawn.figure_spec.layers.get(ASKED).symbology.props["step"] == 10

    def test_the_two_steps_still_decide_when_no_spacing_is_given(self, drawn):
        """The shorthand is an override, not a replacement.

        Args:
            drawn: The map under test.
        """
        drawn.graticule(lon_step=20.0, lat_step=20.0, name=ASKED)
        assert drawn.figure_spec.layers.get(ASKED).symbology.props["step"] == 20
