"""Tests for T9.1 — quickplot/quickmap one-call entry points and module functions."""

import pytest

import digitalearth
from digitalearth import api as qp
from digitalearth.base.spec import Encoding, Guide, LayerSpec, Scale, Symbology
from digitalearth.static import Map


def test_quickplot_returns_finished_map(dataset):
    """quickplot(dataset) returns a Map with one drawn layer and a colorbar."""
    m = digitalearth.quickplot(dataset, crs=dataset.epsg)
    assert isinstance(m, Map)
    assert len(m.layers) == 1
    assert len(m.fig.axes) == 2  # main axes + colorbar


def test_quickmap_kind_dispatch(dataset):
    """quickmap honours an explicit raster kind."""
    m = qp.quickmap(dataset, crs=dataset.epsg, kind="contourf")
    assert m.layers


def test_quickmap_saves_png(dataset, tmp_path):
    """The finished map can be saved to a PNG file."""
    m = qp.quickmap(dataset, crs=dataset.epsg)
    out = tmp_path / "quick.png"
    m.save(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_quickmap_point_features():
    """A FeatureCollection of points is drawn as a marker map."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    m = qp.quickmap(fc, crs=fc.epsg)
    assert m.ax.collections


def test_quickmap_choropleth_polygons():
    """A polygon FeatureCollection with a column is drawn as a choropleth."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    m = qp.quickmap(fc, crs=fc.epsg, column="fid")
    assert m.ax.collections


def _zoned_polygons():
    """Buffered-point polygons carrying a nominal 'zone' column (three unordered classes)."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    fc["zone"] = [["urban", "rural", "park"][i % 3] for i in range(len(fc))]
    return fc


def test_quickmap_categorical_has_no_spurious_colorbar():
    """A categorical quickmap keys itself with a swatch legend, so no meaningless numeric colorbar is added.

    The mappable carries integer class codes; a colorbar over them would read -0.5, 0.5, 1.5 next to the swatch
    legend — the one-call api must skip it (M2), leaving a single axes.
    """
    fc = _zoned_polygons()
    m = qp.quickmap(fc, crs=fc.epsg, column="zone", scheme="categorical")
    assert len(m.fig.axes) == 1, (
        "categorical map must not gain a colorbar axes on top of its swatch legend"
    )
    assert m.layers[-1][0].category_legend is not None, (
        "the swatch legend is still the key"
    )


def test_module_choropleth_categorical_has_no_spurious_colorbar():
    """The module-level choropleth() helper (the _finish path) also skips the colorbar for a categorical fill."""
    fc = _zoned_polygons()
    m = qp.choropleth(fc, crs=fc.epsg, column="zone", scheme="categorical")
    assert len(m.fig.axes) == 1, (
        "the _finish path must skip the colorbar for a categorical fill too"
    )


def test_quickmap_records_the_categorical_key_on_the_layer():
    """The key a categorical fill gets is the layer's, on this backend as on the other three.

    Test scenario:
        The swatch legend the glyph draws for `scheme="categorical"` is pre-order-24 figure decoration: it
        cannot move with its layer, cannot go away with it, and `to_dict()` carries no note that a key was
        ever asked for. `quickmap` recorded a `Guide` for the same fill on the web and 3-D backends and
        nothing at all here, because `_add_static_key` called only `colorbar()` and swallowed the refusal a
        categorical scale answers with. It asks for the key the layer's scale calls for now, so all four
        agree about what `quickmap(colorbar=True)` leaves behind.
    """
    fc = _zoned_polygons()
    m = qp.quickmap(fc, crs=fc.epsg, column="zone", scheme="categorical")
    guide = m.get_layer(m.layer_ids[-1]).symbology.guide()
    assert guide is not None, "the categorical fill records no key at all"
    assert guide.show is True, guide
    drawn = m.ax.get_legend()
    assert drawn is not None, "the recorded key is not on the figure"
    assert {text.get_text() for text in drawn.get_texts()} == {
        "urban",
        "rural",
        "park",
    }, [text.get_text() for text in drawn.get_texts()]


def _legend_artists(canvas):
    """Return the matplotlib `Legend` artists on a map's axes.

    Args:
        canvas: The :class:`~digitalearth.static.Map`.

    Returns:
        Every `Legend` child of the axes — asked of the children rather than of ``get_legend()``, so a second
        one stacked beside the first would show up here instead of hiding behind the one it replaced.
    """
    return [
        child for child in canvas.ax.get_children() if type(child).__name__ == "Legend"
    ]


def test_quickmap_colorbar_false_takes_the_categorical_key_off():
    """`colorbar=False` asks for a map with no colour key, and must get one on this backend too.

    Test scenario:
        The other half of the M6 disagreement, one flag-value over. The glyph draws a swatch legend for
        `scheme="categorical"` whatever the flag says, and the matplotlib path only ever *added* a key — so
        `quickmap(colorbar=False)` returned a map with a legend on it and nothing in the description to say
        the caller had asked against it, while the interactive and 3-D paths both recorded
        `Guide(show=False)` and drew nothing.
    """
    fc = _zoned_polygons()
    m = qp.quickmap(
        fc, crs=fc.epsg, column="zone", scheme="categorical", colorbar=False
    )
    assert m.ax.get_legend() is None, "colorbar=False left a key on the picture"
    assert _legend_artists(m) == [], "colorbar=False left a legend artist behind"
    assert m.get_layer(m.layer_ids[-1]).symbology.guide() == Guide(show=False), (
        m.get_layer(m.layer_ids[-1]).symbology.guide()
    )
    assert len(m.fig.axes) == 1, "switching a key off must not add an axes"


def test_a_key_quickmap_switched_off_can_be_switched_back_on():
    """Taking the key off is a recorded decision, not a one-way removal.

    Test scenario:
        `Guide(show=False)` is how the shared vocabulary spells "there is a key here and it is not drawn",
        which is exactly what lets a switcher offer it back. A removal that could not be undone would be a
        new defect in place of the one it fixed.
    """
    fc = _zoned_polygons()
    m = qp.quickmap(
        fc, crs=fc.epsg, column="zone", scheme="categorical", colorbar=False
    )
    assert _legend_artists(m) == [], "the key should start off"
    m.legend(title="Zones")
    guide = m.get_layer(m.layer_ids[-1]).symbology.guide()
    assert guide == Guide(show=True, title="Zones"), guide
    back = m.ax.get_legend()
    assert back is not None, "the key did not come back"
    assert {text.get_text() for text in back.get_texts()} == {
        "urban",
        "rural",
        "park",
    }, [text.get_text() for text in back.get_texts()]
    assert len(_legend_artists(m)) == 1, "the key came back twice"
    assert len(m.fig.axes) == 1, "bringing the key back must not add an axes"


def test_quickmap_colorbar_false_records_the_bar_switched_off_too():
    """A continuous layer's key is recorded switched off as well, so the two kinds agree.

    Test scenario:
        The picture was already right here — this tier's raster builder draws no bar of its own, so
        `colorbar=False` simply never added one. What was missing is the *record*: the description said
        nothing about a key at all, where the 3-D path recorded `Guide(show=False)` for the same call.
    """
    from pyramids.dataset import Dataset

    dem = Dataset.read_file("examples/data/acc4000.tif")
    m = qp.quickmap(dem, crs=dem.epsg, colorbar=False)
    assert m.get_layer(m.layer_ids[-1]).symbology.guide() == Guide(show=False), (
        m.get_layer(m.layer_ids[-1]).symbology.guide()
    )
    assert len(m.fig.axes) == 1, "colorbar=False must still add no colorbar axes"


def test_quickmap_graduated_still_gets_its_colorbar():
    """A non-categorical fill must still receive its aggregated colorbar (the M2 guard must not over-fire)."""
    fc = _zoned_polygons()
    m = qp.quickmap(fc, crs=fc.epsg, column="fid", scheme="quantiles", k=3)
    assert len(m.fig.axes) == 2, "a graduated numeric fill still carries a colorbar"


def test_quickmap_rejects_unsupported_type():
    """quickmap raises on an input type it cannot draw."""
    with pytest.raises(TypeError, match="cannot draw"):
        qp.quickmap("not data")


def test_quickmap_rejects_empty_features():
    """quickmap raises a clear error on an empty FeatureCollection (not silently treated as polygons)."""
    from pyramids.feature import FeatureCollection

    empty = FeatureCollection.read_file("tests/data/points.geojson").iloc[0:0]
    with pytest.raises(ValueError, match="empty FeatureCollection"):
        qp.quickmap(empty, crs=4326)


def test_module_function_filled_contours(dataset):
    """The module-level contours builds a finished Map via the contourf kind."""
    m = qp.contours(dataset, crs=dataset.epsg, filled=True)
    assert m.layers


def test_module_function_grid_cells(dataset):
    """The module-level grid_cells builds a finished Map with a polygon layer."""
    m = qp.grid_cells(dataset, crs=dataset.epsg)
    assert m.layers


def test_quickmap_with_domain(dataset):
    """quickmap(domain=...) sets the axes extent from the named region."""
    m = qp.quickmap(dataset, crs=3857, domain="europe")
    assert m.ax.get_xlim()[1] > 1e6  # reprojected Europe bbox in metres


@pytest.mark.parametrize(
    ("basemap", "expected_args"),
    [(True, ()), ("CartoDark", ("CartoDark",))],
)
def test_interactive_basemap_branches_reach_the_tier(
    dataset, mocker, basemap, expected_args
):
    """``basemap=True`` asks for the tier's default; a named source is forwarded unchanged.

    Args:
        dataset: The raster to draw.
        mocker: Patches the tile builder so no engine or network is needed.
        basemap: The spelling under test.
        expected_args: What the tile builder must be called with.

    Test scenario:
        The two spellings take different branches — ``True`` means "the backend's default", a name means that
        source — and only one was covered. Reducing a named preset such as ``"Planet.NICFI"`` to the default
        would silently draw the wrong map, which is exactly the kind of failure that leaves no trace.
    """
    interactive = pytest.importorskip("digitalearth.interactive")
    tiles = mocker.patch.object(interactive.InteractiveMap, "tiles")
    mocker.patch.object(interactive.InteractiveMap, "field")
    qp.quickmap(dataset, backend="interactive", basemap=basemap)
    assert tiles.call_args.args == expected_args, (
        f"basemap={basemap!r} must call tiles{expected_args}, got {tiles.call_args!r}"
    )


@pytest.mark.parametrize(
    ("basemap", "expected_args"),
    [(True, ()), ("cartodark", ("cartodark",))],
)
def test_web_basemap_branches_reach_the_tier(dataset, mocker, basemap, expected_args):
    """``basemap=True`` asks for the web tier's default; a named provider is forwarded unchanged.

    Args:
        dataset: The raster to draw.
        mocker: Patches the basemap and raster builders so no engine or network is needed.
        basemap: The spelling under test.
        expected_args: What ``WebMap.basemap`` must be called with.

    Test scenario:
        The web dispatcher's two spellings used to share one line, so running either counted as covering
        both. They are different calls: ``True`` means "whatever the tier defaults to", a name means that
        provider, and collapsing a named provider to the default draws a different map with nothing to show
        for it. This is the web counterpart of the interactive check above.
    """
    web = pytest.importorskip("digitalearth.web")
    tiles = mocker.patch.object(web.WebMap, "basemap")
    mocker.patch.object(web.WebMap, "field")
    qp.quickplot(dataset, backend="web", basemap=basemap)
    assert tiles.call_args.args == expected_args, (
        f"basemap={basemap!r} must call basemap{expected_args}, got {tiles.call_args!r}"
    )


def test_quickmap_decorations_best_effort(dataset):
    """coastlines/basemap flags run without raising even if data/tiles are unreachable."""
    m = qp.quickmap(dataset, crs=3857, coastlines=True, basemap=True)
    assert m.layers  # the raster layer is drawn regardless of decoration availability


def test_quickmap_polygons_without_column_skips_colorbar():
    """A polygon FeatureCollection with no column draws outlines and skips the colorbar gracefully."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    m = qp.quickmap(fc, crs=fc.epsg)  # no column -> polygons (outline only)
    assert m.ax.collections


def test_an_empty_map_has_no_key_to_draw(dataset):
    """`_has_a_key_to_draw` is the question the four-clause colorbar condition was asking.

    Args:
        dataset: A raster to draw.

    Test scenario:
        Nothing drawn is the first of its three answers, and the one a bare `quickmap` reaches when every
        builder skipped. A drawn field is the other, so both arms are read here.
    """
    from digitalearth.api import _has_a_key_to_draw
    from digitalearth.static import Map

    with Map() as canvas:
        assert _has_a_key_to_draw(canvas) is False, "an empty map has nothing to key"
        canvas.field(dataset)
        assert _has_a_key_to_draw(canvas) is True, "a drawn field has a key"


def test_module_choropleth(dataset):
    """The module-level choropleth colours polygons by a column."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(500.0)
    m = qp.choropleth(fc, column="fid", crs=fc.epsg)
    assert m.ax.collections


def test_module_points():
    """The module-level points draws a point FeatureCollection."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    m = qp.points(fc, crs=fc.epsg)
    assert m.ax.collections


def test_quickmap_colorbar_false(dataset):
    """colorbar=False leaves the figure with a single axes (no colorbar axes)."""
    m = qp.quickmap(dataset, crs=dataset.epsg, colorbar=False)
    assert len(m.fig.axes) == 1


def test_quickmap_warns_on_decoration_failures(dataset, mocker):
    """When coastlines/basemap/colorbar fail for an expected reason, quickmap warns and still returns."""
    mocker.patch.object(Map, "coastlines", side_effect=ConnectionError("no net"))
    mocker.patch.object(Map, "basemap", side_effect=ConnectionError("no tiles"))
    mocker.patch.object(Map, "colorbar", side_effect=ValueError("bad mappable"))
    m = qp.quickmap(
        dataset, crs=dataset.epsg, coastlines=True, basemap=True, colorbar=True
    )
    assert m.layers


def test_module_grid_cells_warns_on_colorbar_failure(dataset, mocker):
    """grid_cells still returns a map when its colorbar fails for an expected reason."""
    mocker.patch.object(Map, "colorbar", side_effect=ValueError("bad mappable"))
    m = qp.grid_cells(dataset, crs=dataset.epsg)
    assert m.layers


class _FakeScene:
    """Minimal scene stand-in for _finish: the layers it keys, and a recording/raising ``colorbar()``.

    ``_has_a_key_to_draw`` reads the **description** since order 24 — which layers publish a colour encoding
    — rather than the last registered artist, so the stand-in answers that question instead of holding a
    ``layers`` list alone.
    """

    def __init__(
        self, keyed=(), raises=False, categorical=False, raises_when_drawn=False
    ):
        """Build the stand-in.

        Args:
            keyed: The ids of the layers that publish a colour encoding.
            raises: Whether ``colorbar()`` raises, to mimic an unmappable layer.
            categorical: Whether those layers are coloured by category, which is the case keyed by a swatch
                legend rather than by a bar.
            raises_when_drawn: Whether the key call raises only when ``visible=True`` — the engine's own
                failure, which the ``visible=False`` call never reaches because it draws nothing.
        """
        self._keyed = list(keyed)
        self.layers = [f"artist-{held}" for held in self._keyed]
        self._raises = raises
        self._raises_when_drawn = raises_when_drawn
        self._categorical = categorical
        self.colorbar_calls = 0
        self.legend_calls = 0
        #: Every key call this scene was given, as ``(method, visible)`` in order — so a test can say what
        #: the sequence was, not only how many calls it held.
        self.asked = []

    def _color_keyed(self):
        """Return the ids of the layers a key could explain."""
        return list(self._keyed)

    def get_layer(self, layer_id):
        """Return a description carrying the colour encoding this stand-in claims for ``layer_id``.

        Args:
            layer_id: The layer to describe.

        Returns:
            A `LayerSpec` whose colour is driven by a field, through a categorical or a continuous scale.
        """
        scale = (
            Scale.categorical(["a", "b"], ["#f00", "#00f"])
            if self._categorical
            else Scale.from_limits(0.0, 1.0)
        )
        return LayerSpec(
            layer_id,
            "raster",
            symbology=Symbology(
                encodings={"color": Encoding.by_field("color", "v", scale=scale)}
            ),
        )

    def colorbar(self, *, visible=True):
        """Record the call and its flag (and optionally raise to mimic an unmappable layer).

        Args:
            visible: Whether the key is drawn, which `quickmap` passes through from its own ``colorbar=``.
        """
        self.colorbar_calls += 1
        self.asked.append(("colorbar", visible))
        if self._raises or (self._raises_when_drawn and visible):
            raise ValueError("nothing mappable to colorbar")

    def legend(self, *, visible=True):
        """Record the call and its flag — the key a categorical fill is explained by on every tier.

        Args:
            visible: Whether the key is drawn.
        """
        self.legend_calls += 1
        self.asked.append(("legend", visible))
        if self._raises or (self._raises_when_drawn and visible):
            raise ValueError("nothing mappable to legend")


class TestTheKeyIsAskedForEitherWay:
    """`_add_static_key` carries the caller's flag through, as `_add_3d_key` does.

    The flag decides whether the key is *drawn*, never whether the layer is resolved or the decision is
    recorded — the "validate before honouring visible=False" rule the four tiers share. Reading it as "skip
    the call entirely" is what left this backend's categorical fill keyed against the caller's wish.

    Switching a key off is two calls rather than one, and that is the point of asserting the sequence
    instead of a count. `Renderer.draw_guide` takes off the key **it** drew; the swatch legend a categorical
    glyph draws for itself is not one of those, so the key is taken over first and switched off second.
    """

    def test_the_bar_is_asked_for_with_the_flag_it_was_given(self):
        """A continuous layer's key goes through `colorbar()`, ending on the caller's flag.

        Test scenario:
            Both values, because a helper that only ever passed `True` would look correct against a test
            that checked the `True` case alone.
        """
        shown = _FakeScene(keyed=["layer"])
        qp._add_static_key(shown, visible=True)
        assert shown.asked == [("colorbar", True)], shown.asked

        hidden = _FakeScene(keyed=["layer"])
        qp._add_static_key(hidden, visible=False)
        assert hidden.asked == [("colorbar", True), ("colorbar", False)], hidden.asked

    def test_the_swatch_legend_is_asked_for_the_same_way(self):
        """And a categorical layer's goes through `legend()`, by the same two steps.

        Test scenario:
            The branch and the flag are independent: which key it is, is the layer's property; whether it is
            drawn is the caller's. A fix to one arm must not leave the other spelling `visible=True`.
        """
        shown = _FakeScene(keyed=["zone"], categorical=True)
        qp._add_static_key(shown, visible=True)
        assert shown.asked == [("legend", True)], shown.asked

        hidden = _FakeScene(keyed=["zone"], categorical=True)
        qp._add_static_key(hidden, visible=False)
        assert hidden.asked == [("legend", True), ("legend", False)], hidden.asked

    @pytest.mark.parametrize(
        ("categorical", "kind"), [(False, "colorbar"), (True, "legend")]
    )
    def test_a_refusal_from_the_take_over_still_records_the_decision(
        self, categorical, kind
    ):
        """The take-over draw failing must not cost `quickmap(colorbar=False)` its record.

        Args:
            categorical: Whether the keyed layer is coloured by category.
            kind: The method that layer's key goes through.

        Test scenario:
            The take-over call is the **only** one that reaches `colorbar_legend`/`disjoint_legend`, since
            the second draws nothing — so it is the only one an engine failure can come out of, and one
            `except UNMAPPABLE` around both meant that failure took the `Guide(show=False)` record with it.
            The decision is the thing `colorbar=False` exists to record, and the layer resolves and checks
            identically on both calls, so the second has its own tolerance now (review N1).

            Both kinds are parametrised because the two arms are independent: which key it is, is the
            layer's property, and a fix to one spelling must not leave the other under a shared `try`.
        """
        scene = _FakeScene(
            keyed=["zone"], categorical=categorical, raises_when_drawn=True
        )
        qp._add_static_key(scene, visible=False)
        assert scene.asked == [(kind, True), (kind, False)], (
            f"the refusal from the take-over draw stopped the call that records the decision: {scene.asked}"
        )


class _FakeInteractiveScene:
    """Minimal ``InteractiveMap`` stand-in for `_add_interactive_key`.

    Answers the three questions the helper asks — which layers can be explained, which one it keys, and
    what that layer's colour looks like — and records the key call it is given. A stand-in rather than a
    real map because the arm under test is what happens when the tier's own `colorbar`/`legend` *fails*,
    which a working tier does not do; the same trade ``tests/test_add_web_legend.py`` makes, and the reason
    both run in the default ``dev`` environment with no HoloViz extra installed.

    Args:
        categorical: Whether the keyed layer is coloured by category, which is keyed by a legend.
        raises: What the key call raises instead of recording, or ``None`` to record.
    """

    def __init__(self, *, categorical=False, raises=None):
        self._categorical = categorical
        self._raises = raises
        #: Every key call this scene was given, as ``(method, layer_id, visible)`` in order.
        self.asked = []

    def _guidable(self):
        """Return the ids of the layers whose colour varies with their data."""
        return ["flow"]

    def _guided_layer(self, method, layer_id):
        """Return the layer the key describes, as the tier resolves it.

        Args:
            method: The caller's name, for the refusal this stand-in never gives.
            layer_id: The caller's id, always ``None`` from this helper.

        Returns:
            The one layer this stand-in holds.
        """
        return "flow"

    def get_layer(self, layer_id):
        """Return a description whose colour is driven through a categorical or a continuous scale.

        Args:
            layer_id: The layer to describe.

        Returns:
            A `LayerSpec` carrying that colour encoding.
        """
        scale = (
            Scale.categorical(["a", "b"], ["#f00", "#00f"])
            if self._categorical
            else Scale.from_limits(0.0, 1.0)
        )
        return LayerSpec(
            layer_id,
            "raster",
            symbology=Symbology(
                encodings={"color": Encoding.by_field("color", "v", scale=scale)}
            ),
        )

    def colorbar(self, layer_id, *, visible=True):
        """Record the bar call, or raise what this stand-in was built with.

        Args:
            layer_id: The layer being keyed.
            visible: Whether the key is drawn.
        """
        self.asked.append(("colorbar", layer_id, visible))
        if self._raises is not None:
            raise self._raises

    def legend(self, layer_id, *, visible=True):
        """Record the swatch call, or raise what this stand-in was built with.

        Args:
            layer_id: The layer being keyed.
            visible: Whether the key is drawn.
        """
        self.asked.append(("legend", layer_id, visible))
        if self._raises is not None:
            raise self._raises


class TestAnInteractiveKeyThatFailsIsSkippedNotRaised:
    """`_add_interactive_key` tolerates a layer it cannot key, and only that.

    ``quickplot``'s ``colorbar=`` default asks for a key *if there is one to draw*, so a tier that refuses
    the one layer it resolved must not take the whole one-call map down with it — the same line
    :func:`~digitalearth.api._add_static_key` and :func:`~digitalearth.api._add_web_legend` hold. The
    tolerance is bounded by :data:`~digitalearth.api.UNMAPPABLE`: anything else is a defect, not an
    unkeyable layer, and has to reach the caller.
    """

    def test_a_refused_key_is_warned_about_and_the_call_still_returns(self, caplog):
        """The map comes back and the warning names the key that was skipped.

        Args:
            caplog: pytest's capture of the stdlib logger `api` warns through.
        """
        import logging

        scene = _FakeInteractiveScene(raises=ValueError("nothing mappable to colorbar"))
        with caplog.at_level(logging.WARNING, logger="digitalearth.api"):
            assert qp._add_interactive_key(scene, visible=True) is None, (
                "the helper returns nothing; the map is the caller's own"
            )
        assert scene.asked == [("colorbar", "flow", True)], scene.asked
        assert "quickplot: colorbar skipped" in caplog.text, (
            f"the skip was not announced: {caplog.text!r}"
        )

    def test_a_failure_that_is_not_an_unkeyable_layer_reaches_the_caller(self):
        """A `KeyError` is a defect in the tier, not a layer with no key, so it propagates.

        Test scenario:
            The counterpart of the tolerance above, and why the `except` names three types rather than
            `Exception`: swallowing everything would turn a renamed method or a broken install into a map
            that silently carries no key, which is the failure mode `quickmap` is least able to explain.
        """
        scene = _FakeInteractiveScene(raises=KeyError("flow"))
        with pytest.raises(KeyError):
            qp._add_interactive_key(scene, visible=True)


class TestFinish:
    """Tests for api._finish (PA-5)."""

    def test_adds_colorbar_when_requested_and_layers_present(self):
        """_finish draws a colorbar when colorbar=True and a layer exists.

        Test scenario:
            A scene with one layer and colorbar=True gets exactly one colorbar() call.
        """
        scene = _FakeScene(keyed=["layer"])
        out = qp._finish(scene, colorbar=True)
        assert scene.colorbar_calls == 1, (
            f"expected one colorbar call, got {scene.colorbar_calls}"
        )
        assert out is scene, "the same scene must be returned"

    def test_skips_colorbar_when_disabled(self):
        """_finish never draws a colorbar when colorbar=False.

        Test scenario:
            Even with layers present, colorbar=False suppresses the colorbar() call.
        """
        scene = _FakeScene(keyed=["layer"])
        out = qp._finish(scene, colorbar=False)
        assert scene.colorbar_calls == 0, "colorbar must not be drawn when disabled"
        assert out is scene, "the same scene must be returned"

    def test_skips_the_key_when_nothing_is_coloured_by_a_value(self):
        """_finish keys nothing when no layer publishes a colour at all.

        Test scenario:
            An empty scene with colorbar=True draws neither kind of key — the one case where "add a key if
            there is one to draw" has nothing to add.
        """
        scene = _FakeScene(keyed=[])
        out = qp._finish(scene, colorbar=True)
        assert (scene.colorbar_calls, scene.legend_calls) == (0, 0), (
            "no key must be drawn without a keyed layer"
        )
        assert out is scene, "the same scene must be returned"

    def test_a_categorical_fill_is_keyed_by_a_legend_rather_than_by_a_bar(self):
        """The branch the 3-D path already had: which key is the layer's property, not the flag's.

        Test scenario:
            A bar over a categorical fill would read the class codes cleopatra assigned rather than the
            class names, which is why `Map.colorbar` refuses it. `_add_static_key` used to call only
            `colorbar()` and swallow that refusal, so the one backend `quickmap` reaches without an extra
            recorded no key at all while the web and 3-D paths recorded one.
        """
        categorical = _FakeScene(keyed=["zone"], categorical=True)
        qp._finish(categorical, colorbar=True)
        assert (categorical.colorbar_calls, categorical.legend_calls) == (0, 1), (
            "a categorical fill is keyed by a legend, not by a bar"
        )

    def test_swallows_colorbar_exception(self):
        """_finish swallows an exception from colorbar() (outline-only/unmappable layer).

        Test scenario:
            colorbar() raising must not propagate; the scene is still returned.
        """
        scene = _FakeScene(keyed=["layer"], raises=True)
        out = qp._finish(scene, colorbar=True)
        assert scene.colorbar_calls == 1, "colorbar() should have been attempted once"
        assert out is scene, "the scene must be returned despite the swallowed error"


class TestBackendCapabilityRefusal:
    """C11 — ``quickmap`` refuses a parameter the chosen backend cannot honour, instead of dropping it.

    The four backends are not the same map. Only the matplotlib one has a fixed extent to set, only the 2-D
    ones have a display CRS or a coastline layer, and the web tier keys itself with a legend rather than a
    colorbar. ``quickmap`` used to take every keyword for every backend and quietly discard the ones that did
    not apply, so ``quickmap(ds, backend="web", domain="europe")`` returned a world map with no hint that the
    domain had gone nowhere. Each such parameter is now checked against
    :data:`digitalearth.api.BACKEND_CAPABILITIES` and refused by name.

    These tests deliberately do not build a scene where they can avoid it: the refusal happens before the
    backend is imported, which is what lets them run without the ``3d``/``web``/``interactive`` extras.
    """

    @pytest.mark.parametrize(
        ("backend", "parameter", "value"),
        [
            ("3d", "domain", "europe"),
            ("3d", "coastlines", True),
            ("3d", "basemap", True),
            ("3d", "kind", "contourf"),
            ("web", "domain", "europe"),
            ("web", "coastlines", True),
            ("web", "kind", "contourf"),
            ("interactive", "domain", "europe"),
        ],
    )
    def test_an_unsupported_parameter_names_itself_and_the_backend(
        self, dataset, backend, parameter, value
    ):
        """Every unsupported (backend, parameter) pair raises a ValueError naming both.

        Args:
            dataset: The raster to draw (never actually drawn — the refusal comes first).
            backend: The backend that cannot honour the parameter.
            parameter: The parameter it cannot honour.
            value: A value that genuinely requests something.

        Test scenario:
            A caller who gets a plain map back has no way to tell a dropped argument from one that had no
            visible effect. The message has to carry both halves so they know which one to change.
        """
        with pytest.raises(ValueError) as raised:
            qp.quickmap(dataset, backend=backend, **{parameter: value})
        message = str(raised.value)
        assert f"{parameter}= is not supported" in message, (
            f"the error must name the parameter, got {message!r}"
        )
        assert f"backend={backend!r}" in message, (
            f"the error must name the backend, got {message!r}"
        )

    @pytest.mark.parametrize(
        ("backend", "parameter"),
        [
            ("3d", "domain"),
            ("3d", "coastlines"),
            ("3d", "basemap"),
            ("3d", "kind"),
            ("web", "coastlines"),
            ("web", "kind"),
            ("web", "colorbar"),
        ],
    )
    def test_a_parameter_that_asks_for_nothing_is_not_a_dropped_request(
        self, dataset, backend, parameter, mocker
    ):
        """``domain=None`` / ``coastlines=False`` pass anywhere: nothing was requested, so nothing was lost.

        Args:
            dataset: The raster to draw.
            backend: The backend that cannot honour the parameter.
            parameter: The parameter, passed at its inert value.
            mocker: Stubs the backend builder so no optional extra is needed.

        Test scenario:
            The rule is about *dropped requests*, not about the presence of a keyword. Refusing an explicit
            ``coastlines=False`` on the web tier would be pedantry — there were no coastlines to lose — and
            would break callers who pass one kwargs dict through to whichever backend they picked.

            ``colorbar=False`` is the same shape and was refused anyway until review L9. It is no longer a
            special case at all: since #254 the web tier honours ``colorbar=`` in both directions, building
            its key through ``WebMap.legend``, so neither value is a dropped request there.
        """
        builder = mocker.patch.object(
            qp, "_quickmap_3d" if backend == "3d" else "_quickmap_web"
        )
        inert = {
            "domain": None,
            "coastlines": False,
            "basemap": False,
            "kind": "auto",
            "colorbar": False,
        }[parameter]
        qp.quickmap(dataset, backend=backend, **{parameter: inert})
        assert builder.called, (
            "an inert value must not stop the call reaching the backend"
        )

    def test_an_unset_parameter_is_never_refused(self, dataset, mocker):
        """A backend is not refused a parameter its caller never named.

        Args:
            dataset: The raster to draw.
            mocker: Stubs the 3-D builder so pyvista is not needed.

        Test scenario:
            ``crs`` and ``colorbar`` carry real defaults (``3857`` and ``True``), so the default value cannot
            double as "not passed". Without a separate sentinel, the plainest call there is —
            ``quickmap(ds, backend="3d")`` — would be refused for a CRS nobody asked for. ``colorbar``
            still needs the sentinel after L9: its inert value is ``False``, and its default is ``True``.
        """
        builder = mocker.patch.object(qp, "_quickmap_3d")
        qp.quickmap(dataset, backend="3d")
        assert builder.called, "a bare backend='3d' call must not be refused"

    def test_the_refusal_precedes_building_anything(self, dataset, mocker):
        """Nothing is constructed before the check, so a refused call leaks no figure or plotter.

        Args:
            dataset: The raster to draw.
            mocker: Watches the 3-D backend builder.

        Test scenario:
            A ``Scene3D`` opens a VTK render window in its constructor and a ``Map`` opens a matplotlib
            figure. Checking first is what keeps every rejected call free of a resource nobody will close.
        """
        three_d = mocker.patch.object(qp, "_quickmap_3d")
        with pytest.raises(ValueError, match="domain="):
            qp.quickmap(dataset, backend="3d", domain="europe")
        assert not three_d.called, "a refused call must not reach the backend builder"

    def test_a_crs_is_forwarded_to_the_3d_backend(self, dataset, mocker):
        """The 3-D tier has a display CRS, so ``crs=`` reaches its builder instead of being refused (#291).

        Args:
            dataset: The raster to draw.
            mocker: Replaces the 3-D backend builder with a recorder.
        """
        three_d = mocker.patch.object(qp, "_quickmap_3d")
        qp.quickmap(dataset, backend="3d", crs=4326)
        assert three_d.call_args.kwargs["crs"] == 4326, three_d.call_args

    def test_every_backend_has_a_capability_entry(self):
        """The dispatcher and the capability table name the same backends.

        Test scenario:
            The table is what the unknown-backend check now reads, so a backend added to one and not the
            other would either be unreachable or escape the parameter check entirely.
        """
        assert set(qp.BACKEND_CAPABILITIES) == {
            "matplotlib",
            "interactive",
            "3d",
            "web",
        }, f"unexpected backend set: {sorted(qp.BACKEND_CAPABILITIES)}"

    def test_an_unknown_backend_still_names_the_real_ones(self, dataset):
        """An unrecognised backend raises before the capability lookup, listing the four that exist.

        Args:
            dataset: The raster to draw.

        Test scenario:
            Routing the unknown-backend check through the capability table must not turn a clear error into
            a ``KeyError`` from inside the refusal helper.
        """
        with pytest.raises(ValueError, match="unknown backend"):
            qp.quickmap(dataset, backend="opengl")

    def test_the_not_passed_sentinel_reads_as_unset(self):
        """The "not passed" sentinel spells itself ``<unset>`` wherever a default is rendered.

        Test scenario:
            ``help(quickmap)``, the rendered signature and every IDE hint print a default with ``repr``.
            Without a spelling of its own the sentinel shows as ``<digitalearth.api._Unset object at
            0x...>``, which reads as an implementation leak rather than as "leave this one alone".
        """
        import inspect

        rendered = str(inspect.signature(qp.quickmap))
        assert repr(qp._UNSET) == "<unset>", (
            f"the sentinel must spell itself <unset>, got {qp._UNSET!r}"
        )
        assert "<unset>" in rendered, (
            f"the rendered signature must show <unset> for a sentinel default, got {rendered!r}"
        )

    def test_matplotlib_honours_all_four(self, dataset):
        """The default backend takes every checked parameter, exactly as it did before.

        Args:
            dataset: The raster to draw.

        Test scenario:
            C11 is about the tiers that *cannot* honour a parameter. The static tier can honour all four, so
            the guard must be invisible to it — this is the regression that would catch an over-fire.
        """
        m = qp.quickmap(
            dataset, crs=3857, domain="europe", coastlines=False, colorbar=False
        )
        assert len(m.fig.axes) == 1, (
            "colorbar=False must still suppress the colorbar axes"
        )
        assert m.layers, "the data layer must still be drawn"

    def test_web_receives_the_crs_it_was_given(self, dataset, mocker):
        """``crs=`` is forwarded to the web tier rather than dropped — it has one, and validates it.

        Args:
            dataset: The raster to draw.
            mocker: Stubs ``_quickmap_web`` so the ``web`` extra is not needed.

        Test scenario:
            The web tier carries its own ``crs`` and refuses anything but 4326, explaining why inline data
            can only be placed in lon/lat. Forwarding lets that tier answer; dropping the argument here, as
            the old code did, was exactly the silent failure C11 exists to remove.
        """
        forward = mocker.patch.object(qp, "_quickmap_web")
        qp.quickmap(dataset, backend="web", crs=4326)
        assert forward.call_args.kwargs["crs"] == 4326, (
            f"crs must reach the web builder, got {forward.call_args!r}"
        )


class TestTheRefusalNamesWhatTheCallerWrote:
    """Round-2 review L3/L4/L5 and M19 — a refusal must name a keyword the caller can actually act on."""

    def test_an_array_domain_is_refused_by_name_not_by_numpy(self):
        """A bbox handed over as an ndarray gets this module's message, not numpy's ambiguity error.

        Test scenario:
            ``_INERT`` was consulted with ``==``, so ``value == None`` on an array returned an *array* and
            the ``if`` over it raised "The truth value of an array with more than one element is
            ambiguous" — from inside the guard whose whole job is to name the offending parameter (L3).
        """
        import numpy as np

        domain = np.array([0.0, 0.0, 1.0, 1.0])
        with pytest.raises(
            ValueError, match=r"domain= is not supported by backend='web'"
        ):
            qp._reject_unsupported("web", domain=domain, crs=qp._UNSET)

    @pytest.mark.parametrize(
        ("backend", "kwargs"),
        [("3d", {"basemap": 0}), ("web", {"coastlines": 0})],
    )
    def test_a_numeric_zero_is_not_the_inert_false(self, backend, kwargs):
        """``0 == False`` is ``True`` in Python, but ``basemap=0`` is a value the caller typed.

        Args:
            backend: The tier that cannot honour the keyword.
            kwargs: The keyword, written as the number zero.

        Test scenario:
            The inert check means *identity* with the "asks for nothing" value, not equality with it, or a
            caller who passes a number through a shared kwargs dict is told nothing was dropped when it
            was (L4).
        """
        name = next(iter(kwargs))
        with pytest.raises(ValueError, match=rf"{name}= is not supported"):
            qp._reject_unsupported(backend, crs=qp._UNSET, **kwargs)

    @pytest.mark.parametrize("backend", ["web", "3d"])
    def test_a_module_wrapper_names_itself_not_the_kind_it_injected(
        self, dataset, backend
    ):
        """``field(ds, backend="web")`` must not tell the caller to drop a keyword they never wrote.

        Args:
            dataset: The raster to draw.
            backend: A tier with no renderer selector.

        Test scenario:
            The wrapper *is* the ``kind``: it injects ``kind="imshow"`` itself. The generic refusal then
            said "drop the argument", naming a parameter that does not appear in the caller's source (L5).
        """
        with pytest.raises(ValueError) as excinfo:
            qp.field(dataset, backend=backend)
        message = str(excinfo.value)
        assert message.startswith("field()"), message
        assert f"backend={backend!r}" in message, message
        assert "kind=" not in message, (
            f"the message must not name the injected keyword: {message}"
        )

    def test_a_kind_and_the_keyword_it_implies_cannot_both_be_given(self, dataset):
        """``kind="contourf"`` *is* ``filled=True``, so naming both is a contradiction, not a style option.

        Args:
            dataset: The raster to draw.

        Test scenario:
            Both kind tables inject `filled=`, and the two drawers splatted the injection beside the caller's
            keywords — so the collision was Python's, naming `RasterMixin.contours()`, a private mixin the
            caller never wrote, and never saying that the kind already meant it (R2-L4). The refusal names the
            kind, what it draws with, and the kind that means what was asked for instead.
        """
        # `dataset.epsg` is a pyramids property, read above the block: a raise from *it* would satisfy
        # the block and leave the refusal under test unreached.
        epsg = dataset.epsg
        with pytest.raises(ValueError) as excinfo:
            qp.quickmap(dataset, crs=epsg, kind="contourf", filled=False)
        message = str(excinfo.value)
        assert "kind='contourf' is itself filled=True" in message, message
        assert "RasterMixin" not in message, (
            f"the refusal must not name a private mixin: {message}"
        )
        assert "kind='contour'" in message, (
            f"the refusal should name the kind that draws filled=False: {message}"
        )

    @pytest.mark.parametrize(
        "table", [qp._STATIC_RASTER_KINDS, qp._INTERACTIVE_RASTER_KINDS]
    )
    def test_both_kind_tables_refuse_the_contradiction_the_same_way(self, table):
        """The two tiers that take a ``kind`` answer one contradiction with one sentence.

        Args:
            table: The tier's kind table.

        Test scenario:
            Asked of the resolver rather than through a map, because the interactive tier's engine is in
            another environment and this refusal happens before anything is drawn — which is the point of
            resolving the kind in one shared place instead of once per drawer.
        """
        with pytest.raises(ValueError) as excinfo:
            qp._renderer_for(table, "contourf", {"filled": False}, "quickmap")
        assert "is itself filled=True" in str(excinfo.value), excinfo.value

    def test_a_kind_that_implies_nothing_still_takes_the_keyword(self, dataset):
        """The positive control: only a keyword the kind *settles* is a contradiction.

        Args:
            dataset: The raster to draw.

        Test scenario:
            `imshow` carries no keywords in either table, so `filled=` written beside it is a style option
            for the renderer to answer for — and a guard that refused any keyword named in any table would
            have taken this call away too. The refusal here comes from cleopatra, naming `filled`, which is
            the tier's own answer rather than this module's.
        """
        # Read above the block for the reason the test two above gives: `epsg` is a property.
        epsg = dataset.epsg
        with pytest.raises(ValueError) as excinfo:
            qp.quickmap(dataset, crs=epsg, kind="imshow", filled=False)
        message = str(excinfo.value)
        assert "contradicts it" not in message, (
            f"imshow implies no filled=, so this must not be refused as a contradiction: {message}"
        )
        assert "filled" in message, message

    def test_a_column_on_point_input_is_refused_by_name(self):
        """``quickmap(points, column=...)`` names the keyword instead of leaking cleopatra's error.

        Test scenario:
            The polygon branch pops ``column`` and draws a choropleth; the point branch forwarded it into
            ``Map.points``'s ``**opts``, where cleopatra answered with its own accepted-keyword list and
            never mentioned ``column`` (M19). ``Map.points`` has no fill column — it sizes markers by
            ``size_column`` — so the honest answer is a refusal that says so.
        """
        from pyramids.feature import FeatureCollection

        fc = FeatureCollection.read_file("tests/data/points.geojson")
        # Read above the block: `epsg` is a pyramids property, so a raise from it would pass this test
        # with the `column=` refusal never reached.
        epsg = fc.epsg
        with pytest.raises(ValueError) as excinfo:
            qp.quickmap(fc, crs=epsg, column="fid")
        message = str(excinfo.value)
        assert "column='fid'" in message, message
        assert "size_column=" in message, message
