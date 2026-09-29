"""``Map.labels`` — label every feature from a column, on the static tier (ST-16, #345).

The kind was registered and the web tier shipped the builder, so a caller could label a collection on a web map
and not on a static one; ``Map.text`` places one string at one coordinate and is no substitute. What is covered
here is the builder, its drawer, and — the part the issue asked to be settled rather than guessed — which of
``WebMap.labels``' keywords carry across and which are refused because this tier cannot honour them.

The keyword *names* are held here, in :class:`TestTheKeywordsAreTheWebTiers`. The four shared **defaults** are
held one level lower, in ``tests/base/test_shared_tier_defaults.py``, beside the constants they come from.
"""

import inspect

import geopandas as gpd
import numpy as np
import pytest
from pyramids.feature import FeatureCollection
from shapely.geometry import Point, Polygon

from digitalearth.base.crs import OffLimbError
from digitalearth.static import Map, projections
from digitalearth.static.capabilities import CAPABILITIES
from digitalearth.static.maps.vector import _LABEL_REFUSED_OPTS


@pytest.fixture
def places():
    """Return the committed point fixture as a ``FeatureCollection``.

    Returns:
        Ten points in EPSG:32618 with a numeric ``fid`` column.
    """
    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def named():
    """Return three lon/lat points carrying a string column, one of whose values is missing.

    Returns:
        A ``FeatureCollection`` in EPSG:4326 with a ``name`` column of ``["A", None, "C"]``.
    """
    frame = gpd.GeoDataFrame(
        {"name": ["A", None, "C"]},
        geometry=[Point(4.0, 52.0), Point(4.1, 52.1), Point(4.2, 52.2)],
        crs=4326,
    )
    return FeatureCollection(frame)


def _texts(drawn) -> list:
    """Return the strings a label layer put on the axes.

    Args:
        drawn: What the builder handed back — one ``Annotation`` per labelled feature.

    Returns:
        The label strings, in the order they were drawn.
    """
    return [artist.get_text() for artist in drawn]


class TestOneLabelPerFeature:
    """The capability: a string per feature, read from an attribute rather than written per call."""

    def test_every_feature_is_labelled(self, places):
        """Args:
        places: The ten-point fixture.
        """
        with Map(crs=places.epsg) as canvas:
            drawn = canvas.labels(places, "fid")
        assert len(drawn) == len(places), (
            f"{len(places)} features should give {len(places)} labels; got {len(drawn)}"
        )

    def test_the_text_is_the_column_and_not_the_index(self, named):
        """A label is the *value*, which a row counter would coincidentally match on an ordered fixture.

        Args:
            named: Points whose ``name`` column is letters, so a positional label could not pass.
        """
        with Map(crs=4326) as canvas:
            drawn = canvas.labels(named, "name")
        assert _texts(drawn) == ["A", "C"], f"the labels read {_texts(drawn)}"

    def test_a_missing_value_is_not_labelled(self, named):
        """A null is absent data, not the text ``"None"`` — the reading a MapLibre expression gives it too.

        Args:
            named: Points whose middle value is `None`.
        """
        with Map(crs=4326) as canvas:
            drawn = canvas.labels(named, "name")
        assert len(drawn) == 2, (
            f"the null should draw no artist; got {len(drawn)} labels"
        )

    def test_a_label_sits_at_its_feature(self, named):
        """The point it annotates is the geometry, reprojected into the display CRS.

        Args:
            named: Points in EPSG:4326, drawn on a map in the same CRS so the coordinates are comparable.
        """
        with Map(crs=4326) as canvas:
            drawn = canvas.labels(named, "name")
            anchored = [
                tuple(round(value, 6) for value in artist.xy) for artist in drawn
            ]
        assert anchored == [(4.0, 52.0), (4.2, 52.2)], (
            f"the labels were anchored at {anchored}"
        )

    def test_a_polygon_is_labelled_at_its_centroid(self):
        """Any geometry is labellable, as it is on the web tier; a non-point falls back to its centroid."""
        square = Polygon([(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        frame = gpd.GeoDataFrame({"name": ["block"]}, geometry=[square], crs=3857)
        with Map(crs=3857) as canvas:
            drawn = canvas.labels(FeatureCollection(frame), "name")
            anchor = tuple(float(value) for value in drawn[0].xy)
        assert anchor == (1.0, 1.0), (
            f"a square's centroid is (1, 1); the label sat at {anchor}"
        )

    def test_the_layer_is_described_under_the_registered_kind(self, places):
        """The kind was registered before any tier drew it; this is the tier claiming it.

        Args:
            places: The features labelled.
        """
        with Map(crs=places.epsg) as canvas:
            canvas.labels(places, "fid", name="tags")
            described = canvas.figure_spec.layers.get("tags")
        assert described.kind == "labels", f"the layer recorded {described.kind!r}"

    def test_the_layer_is_drawn_over_the_data(self, places, dataset):
        """``labels`` is registered in the ``overlay`` band, so it stays above a field however it is ordered.

        Args:
            places: The features labelled.
            dataset: A raster drawn *after* the labels, which must still land beneath them.
        """
        with Map(crs=dataset.epsg) as canvas:
            canvas.labels(places, "fid", name="tags")
            canvas.field(dataset, name="dem")
            order = list(canvas.layer_ids)
        assert order == ["dem", "tags"], f"the bands put the layers in {order}"

    def test_the_tier_declares_the_kind_it_now_draws(self):
        """A builder drawing an undeclared kind would make the declaration a lie."""
        assert "labels" in CAPABILITIES.kinds, sorted(CAPABILITIES.kinds)


class TestTheKeywordsAreTheWebTiers:
    """One spelling per concept — which of ``WebMap.labels``' keywords carry across, and which cannot.

    ``base/contract.py`` declares the *name*; what this asserts is the argument list underneath it, which is
    where two tiers most easily drift. The four shared **defaults** are held to one value a level lower, in
    ``tests/base/test_shared_tier_defaults.py``, where the constants they come from live.
    """

    def test_the_portable_keywords_are_spelled_the_same(self):
        """The four the two engines can both mean, plus the two every builder has.

        Test scenario:
            ``halo_color``/``halo_width`` are MapLibre's text halo and matplotlib's ``path_effects`` stroke —
            the same idea under two engine spellings, so the *keyword* carries. ``offset`` is MapLibre's ems
            with y down, converted here rather than renamed.
        """
        portable = {
            "text_size",
            "color",
            "halo_color",
            "halo_width",
            "offset",
            "name",
            "visible",
        }
        taken = set(inspect.signature(Map.labels).parameters)
        assert portable <= taken, f"not carried across: {sorted(portable - taken)}"

    def test_the_web_only_keyword_is_not_quietly_taken(self):
        """``allow_overlap`` is MapLibre's collision index, which matplotlib has nothing to answer with."""
        taken = set(inspect.signature(Map.labels).parameters)
        assert "allow_overlap" not in taken, (
            "a keyword this tier cannot honour must not be in its signature"
        )

    def test_the_web_only_keyword_is_declared_absent_with_a_reason(self):
        """Declared, not merely missing: "no collision detection" and "not yet" are different answers."""
        assert CAPABILITIES.reason("label_collision"), (
            "the tier must say why label collision is absent, not just leave it out"
        )

    def test_the_refusal_quotes_the_declaration(self, places):
        """The sentence a caller reads and the one a dispatcher reads have to be one sentence.

        Args:
            places: The features the refused call would have labelled.
        """
        reason = CAPABILITIES.reason("label_collision")
        canvas = Map(crs=places.epsg)
        with pytest.raises(ValueError, match="allow_overlap"):
            canvas.labels(places, "fid", allow_overlap=True)
        canvas.close()
        assert reason in _LABEL_REFUSED_OPTS["allow_overlap"], (
            "the refusal restates the declaration instead of quoting it, so the two can drift"
        )

    def test_the_core_keyword_for_this_name_is_taken(self):
        """``base/contract.py`` declares ``labels(column=, crs=, name=)``; all three have to be there."""
        taken = set(inspect.signature(Map.labels).parameters)
        assert {"column", "crs", "name"} <= taken, (
            f"short of the Core keywords: {sorted(taken)}"
        )


class TestTextSizeIsNotMarkerSize:
    """The renamed-keyword settlement, enforced rather than documented."""

    def test_the_declared_size_reaches_the_artist(self, places):
        """Args:
        places: The features labelled.
        """
        with Map(crs=places.epsg) as canvas:
            drawn = canvas.labels(places, "fid", text_size=17.0)
            sizes = {artist.get_fontsize() for artist in drawn}
        assert sizes == {17.0}, f"the labels were drawn at {sizes}"

    @pytest.mark.parametrize("alias", ["size", "fontsize"])
    def test_matplotlibs_own_spelling_is_refused_rather_than_overwritten(
        self, places, alias
    ):
        """Accepted and then overwritten by ``text_size`` is the inert-keyword defect, so it is refused.

        Args:
            places: The features the refused call would have labelled.
            alias: matplotlib's spelling of the same thing.
        """
        canvas = Map(crs=places.epsg)
        with pytest.raises(ValueError, match=f"does not take \\['{alias}'\\]"):
            canvas.labels(places, "fid", **{alias: 9.0})
        canvas.close()


class TestTheHaloIsAPathEffectsStroke:
    """MapLibre's halo and matplotlib's stroke are one idea, which is why the keywords carry across."""

    def test_a_halo_is_drawn_by_default(self, places):
        """Args:
        places: The features labelled.
        """
        with Map(crs=places.epsg) as canvas:
            drawn = canvas.labels(places, "fid")
            effects = drawn[0].get_path_effects()
        assert len(effects) == 1, f"one stroke should be applied; got {effects}"

    def test_a_zero_width_asks_for_none(self, places):
        """``halo_width=0`` is MapLibre's "no halo" too, so it means the same thing here.

        Args:
            places: The features labelled.
        """
        with Map(crs=places.epsg) as canvas:
            drawn = canvas.labels(places, "fid", halo_width=0.0)
            effects = drawn[0].get_path_effects()
        assert effects == [], f"no halo was asked for, yet {effects} was applied"

    def test_the_callers_own_path_effects_win(self, places):
        """Replaced, not ignored — which is the difference between a wart and a defect.

        Args:
            places: The features labelled.
        """
        from matplotlib.patheffects import Normal

        own = [Normal(), Normal()]
        with Map(crs=places.epsg) as canvas:
            drawn = canvas.labels(places, "fid", path_effects=own)
            effects = drawn[0].get_path_effects()
        assert len(effects) == 2, (
            f"the caller's own two effects should be drawn; got {effects}"
        )

    def test_a_negative_width_is_refused(self, places):
        """A width is zero or more; a negative one is a typo matplotlib would draw something for.

        Args:
            places: The features the refused call would have labelled.
        """
        canvas = Map(crs=places.epsg)
        with pytest.raises(ValueError, match="halo_width= as a width"):
            canvas.labels(places, "fid", halo_width=-2.0)
        canvas.close()


class TestTheOffsetIsConvertedNotRenamed:
    """MapLibre measures it in ems with y down; matplotlib in points with y up. Both are resolved here."""

    def test_no_offset_leaves_the_label_on_its_feature(self, places):
        """Args:
        places: The features labelled.
        """
        with Map(crs=places.epsg) as canvas:
            drawn = canvas.labels(places, "fid")
            shift = tuple(float(value) for value in drawn[0].get_position())
        assert shift == (0.0, 0.0), (
            f"an unoffset label should sit on its point; got {shift}"
        )

    def test_an_em_offset_becomes_points_of_the_text_size(self, places):
        """One em is one text size, so the same ``offset`` moves a bigger label further.

        Args:
            places: The features labelled.
        """
        with Map(crs=places.epsg) as canvas:
            small = canvas.labels(
                places, "fid", text_size=10.0, offset=(2.0, 0.0), name="a"
            )
            large = canvas.labels(
                places, "fid", text_size=20.0, offset=(2.0, 0.0), name="b"
            )
            shifts = (small[0].get_position()[0], large[0].get_position()[0])
        assert shifts == (20.0, 40.0), (
            f"two ems of 10 and of 20 points are 20 and 40; got {shifts}"
        )

    def test_a_negative_y_lifts_the_label_as_it_does_on_the_web_tier(self, places):
        """MapLibre's ``(0, -1.2)`` lifts a label off its point, and so must this one.

        Test scenario:
            The axes' y points *up* and MapLibre's screen y points *down*, so carrying the number across
            unchanged would have pushed the label down — the same call, two pictures.

        Args:
            places: The features labelled.
        """
        with Map(crs=places.epsg) as canvas:
            drawn = canvas.labels(places, "fid", text_size=10.0, offset=(0.0, -1.2))
            lift = float(drawn[0].get_position()[1])
        assert lift > 0.0, (
            f"a negative y offset must lift the label above its point; it moved {lift}"
        )

    @pytest.mark.parametrize(
        "offset",
        [(1.0,), (1.0, 2.0, 3.0), "up", 4.0, (1.0, float("inf"))],
        ids=["one", "three", "string", "scalar", "infinite"],
    )
    def test_an_offset_a_figure_could_not_carry_is_refused_at_the_call(
        self, places, offset
    ):
        """Args:
        places: The features the refused call would have labelled.
        offset: Something that is not two finite numbers.
        """
        canvas = Map(crs=places.epsg)
        with pytest.raises(ValueError, match="offset="):
            canvas.labels(places, "fid", offset=offset)
        canvas.close()

    def test_the_described_offset_is_a_list_a_figure_can_carry(self, places):
        """A tuple returns from JSON as a list, and two symbologies differing only in that stop comparing.

        Args:
            places: The features labelled.
        """
        with Map(crs=places.epsg) as canvas:
            canvas.labels(places, "fid", offset=(0.0, -1.2), name="tags")
            recorded = canvas.figure_spec.layers.get("tags").symbology.props["offset"]
        assert list(recorded) == [0.0, -1.2], f"the offset was recorded as {recorded!r}"


class TestTheCrsKeywordHasAJobHere:
    """The Core declares ``labels(crs=)``, and on this tier it answers a question the web tier cannot have."""

    def test_a_collection_that_declares_no_crs_is_placed_by_it(self):
        """A naive frame is ordinary — coordinates read out of a CSV — and ``to_crs`` refuses one outright."""
        naive = gpd.GeoDataFrame(
            {"name": ["A", "B"]}, geometry=[Point(4.0, 52.0), Point(5.0, 53.0)]
        )
        with Map(crs=3857) as canvas:
            drawn = canvas.labels(FeatureCollection(naive), "name", crs=4326)
            eastings = [float(artist.xy[0]) for artist in drawn]
        assert eastings[0] > 100_000.0, (
            f"the labels should be in Web Mercator metres, not degrees; got {eastings}"
        )

    def test_a_collection_that_declares_none_and_no_crs_given_is_refused(self):
        """Naming nothing leaves the coordinates unplaceable, which is a refusal rather than a guess."""
        naive = FeatureCollection(
            gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(4.0, 52.0)])
        )
        canvas = Map(crs=3857)
        with pytest.raises(ValueError, match="features that declare no CRS"):
            canvas.labels(naive, "name")
        canvas.close()

    def test_a_crs_that_agrees_with_the_features_is_accepted(self, named):
        """Portable code passes ``crs=4326`` habitually; agreeing with the data is not an error.

        Args:
            named: Points that declare EPSG:4326.
        """
        with Map(crs=4326) as canvas:
            drawn = canvas.labels(named, "name", crs=4326)
        assert len(drawn) == 2, (
            f"a restated CRS should change nothing; got {len(drawn)} labels"
        )

    def test_a_crs_that_contradicts_the_features_is_refused(self, places):
        """Two answers to "where is this" cannot both be acted on, so neither is guessed at.

        Args:
            places: Features in EPSG:32618.
        """
        canvas = Map(crs=places.epsg)
        with pytest.raises(ValueError, match="already declare a different CRS"):
            canvas.labels(places, "fid", crs=4326)
        canvas.close()

    def test_the_refusal_names_the_crs_by_its_code_and_not_its_wkt(self, places):
        """A pyproj CRS reprs as a multi-line WKT block, which buries the fact the message is about.

        Args:
            places: Features in EPSG:32618.
        """
        canvas = Map(crs=places.epsg)
        with pytest.raises(ValueError, match="EPSG:32618") as raised:
            canvas.labels(places, "fid", crs=4326)
        canvas.close()
        assert "\n" not in str(raised.value), f"the refusal spans lines: {raised.value}"

    def test_the_callers_collection_is_not_relabelled_underneath_them(self):
        """``set_crs`` returns a new frame; mutating the caller's object would be a side effect."""
        naive = gpd.GeoDataFrame({"name": ["A"]}, geometry=[Point(4.0, 52.0)])
        features = FeatureCollection(naive)
        with Map(crs=3857) as canvas:
            canvas.labels(features, "name", crs=4326)
        assert features.crs is None, (
            f"the caller's collection now declares {features.crs}"
        )


class TestWhatTheBuilderRefuses:
    """Each in the register ``quickmap`` set: the call, what was wrong, and what to write instead."""

    def test_a_column_the_features_do_not_carry_is_named(self, places):
        """An unchecked column renders an empty layer with nothing to explain it.

        Args:
            places: Features carrying only ``fid``.
        """
        canvas = Map(crs=places.epsg)
        with pytest.raises(KeyError, match="is not a property of these features"):
            canvas.labels(places, "nope")
        canvas.close()

    def test_the_refusal_lists_the_columns_there_are(self, places):
        """Args:
        places: Features carrying only ``fid``.
        """
        canvas = Map(crs=places.epsg)
        with pytest.raises(KeyError, match=r"available: \['fid'\]"):
            canvas.labels(places, "nope")
        canvas.close()

    @pytest.mark.parametrize(
        "column", ["", "   ", None, 3], ids=["empty", "blank", "none", "int"]
    )
    def test_a_column_that_is_not_a_name_is_refused_at_the_call(self, places, column):
        """Args:
        places: The features the refused call would have labelled.
        column: Something that is not the name of an attribute.
        """
        canvas = Map(crs=places.epsg)
        with pytest.raises(
            ValueError, match="needs column= as the name of an attribute"
        ):
            canvas.labels(places, column)
        canvas.close()

    @pytest.mark.parametrize(
        "value", [float("nan"), float("inf"), "big"], ids=["nan", "inf", "string"]
    )
    def test_a_text_size_a_figure_could_not_carry_is_refused_at_the_call(
        self, places, value
    ):
        """JSON has no spelling for NaN, so the whole figure would refuse to be written later.

        Args:
            places: The features the refused call would have labelled.
            value: A text size that is not a finite number.
        """
        canvas = Map(crs=places.epsg)
        with pytest.raises(ValueError, match="text_size= as a finite number"):
            canvas.labels(places, "fid", text_size=value)
        canvas.close()

    def test_a_non_vector_input_is_refused_by_name(self):
        """The shared guard, reached through this builder — it used to leak a ``to_crs`` attribute error."""
        canvas = Map()
        grid = np.zeros((2, 2))
        with pytest.raises(TypeError, match=r"Map\.labels\(\) cannot draw a ndarray"):
            canvas.labels(grid, "fid")
        canvas.close()

    def test_an_empty_collection_is_refused(self):
        """There is nothing to label, which is a call that cannot have meant what it said."""
        empty = FeatureCollection(gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326))
        canvas = Map(crs=4326)
        with pytest.raises(ValueError, match="empty FeatureCollection"):
            canvas.labels(empty, "name")
        canvas.close()

    def test_a_refused_call_leaves_no_layer_behind(self, places):
        """A description must never name a layer that was not drawn, refusals included.

        Args:
            places: The features the refused call would have labelled.
        """
        canvas = Map(crs=places.epsg)
        with pytest.raises(KeyError, match="is not a property"):
            canvas.labels(places, "nope", name="tags")
        remaining = list(canvas.layer_ids)
        canvas.close()
        assert remaining == [], f"the refused layer was left on the figure: {remaining}"


class TestWhatTheDrawerDeclines:
    """A layer with nothing to draw is dropped, which is not the same as a layer that raised."""

    def test_labels_entirely_behind_a_globes_limb_are_skipped(self, places):
        """Args:
        places: Features placed on the far side of an orthographic globe.
        """
        away = projections.orthographic(lon=-175.0, lat=15.0)
        with Map(crs=away, globe=True) as canvas:
            drawn = canvas.labels(places, "fid")
            remaining = list(canvas.layer_ids)
        assert drawn is None, f"an off-limb label layer draws nothing; got {drawn!r}"
        assert remaining == [], (
            f"a layer that drew nothing must not be described: {remaining}"
        )

    def test_only_the_labels_on_the_near_side_of_a_globe_are_drawn(self):
        """The half a whole-layer check cannot reach, and the one the coordinate filter exists for.

        Test scenario:
            A layer that lands nowhere is caught by the warp itself, which reports it before any label is
            placed. A **partly** hidden layer is not: the far-side features come back with non-finite
            coordinates, and an annotation at infinity is drawn into nowhere while still counting as a label
            the figure claims. So the coordinates are filtered, and the text is filtered with them — which is
            also what keeps a surviving label off another feature's position.
        """
        frame = gpd.GeoDataFrame(
            {"name": ["near", "far"]},
            geometry=[Point(4.0, 52.0), Point(-176.0, -52.0)],
            crs=4326,
        )
        globe = projections.orthographic(lon=4.0, lat=52.0)
        with Map(crs=globe, globe=True) as canvas:
            drawn = canvas.labels(FeatureCollection(frame), "name")
        assert _texts(drawn) == ["near"], (
            f"only the near-side feature is placeable, so only it is labelled; got {_texts(drawn)}"
        )

    def test_the_same_layer_raises_under_strict(self, places):
        """``strict`` is for a pipeline that must not publish a map with a layer silently missing.

        Args:
            places: Features placed on the far side of an orthographic globe.
        """
        away = projections.orthographic(lon=-175.0, lat=15.0)
        canvas = Map(crs=away, globe=True, strict=True)
        with pytest.raises(OffLimbError):
            canvas.labels(places, "fid")
        canvas.close()

    def test_a_column_of_nothing_but_nulls_draws_no_layer(self):
        """Every label dropped is a layer that was not drawn, as an off-limb one is."""
        frame = gpd.GeoDataFrame(
            {"name": [None, None]},
            geometry=[Point(4.0, 52.0), Point(4.1, 52.1)],
            crs=4326,
        )
        with Map(crs=4326) as canvas:
            drawn = canvas.labels(FeatureCollection(frame), "name")
            remaining = list(canvas.layer_ids)
        assert drawn is None, (
            f"nothing was labellable, so nothing should be drawn; got {drawn!r}"
        )
        assert remaining == [], (
            f"a layer that drew nothing must not be described: {remaining}"
        )


class TestTheLayerIsManagedLikeAnyOther:
    """Order 23's layer management is the reason the kind was registered; a new builder inherits it."""

    def test_the_layer_can_be_hidden_at_build_time(self, places):
        """``visible=False`` builds it hidden *and* describes it hidden (#327).

        Args:
            places: The features labelled.
        """
        with Map(crs=places.epsg) as canvas:
            drawn = canvas.labels(places, "fid", name="tags", visible=False)
            described = canvas.figure_spec.layers.get("tags")
            shown = {artist.get_visible() for artist in drawn}
        assert described.visible is False, "the figure describes the layer as visible"
        assert shown == {False}, f"the annotations were drawn with visibility {shown}"

    def test_the_layer_can_be_taken_off_again(self, places):
        """Args:
        places: The features labelled.
        """
        with Map(crs=places.epsg) as canvas:
            canvas.labels(places, "fid", name="tags")
            canvas.remove_layer("tags")
            remaining = list(canvas.layer_ids)
        assert remaining == [], f"the layer survived its removal: {remaining}"

    def test_a_second_label_layer_gets_its_own_id(self, places):
        """Two layers sharing an id makes the second unaddressable (#321).

        Args:
            places: The features labelled twice.
        """
        with Map(crs=places.epsg) as canvas:
            canvas.labels(places, "fid")
            canvas.labels(places, "fid")
            ids = list(canvas.layer_ids)
        assert ids == ["labels-1", "labels-2"], f"the ids came out as {ids}"
