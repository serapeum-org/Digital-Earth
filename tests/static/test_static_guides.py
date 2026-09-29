"""The static tier's colour key: a guide on the layer's encoding rather than figure furniture (#261, order 24).

What the tier had before this: ``colorbar(layer=-1)`` keyed **by position** into ``Scene.layers`` and drew when
called. Four things followed, and each has a test here:

* a layer added after the data took the key over, because "the last registered mappable" is not "the layer the
  reader needs explained";
* nothing recorded that a key had been asked for, so a removed layer left its bar behind, a hidden layer kept
  it, and a figure written to a dict came back with no key;
* ``legend`` was handed the colours and labels themselves, so the key agreed with the picture only while the
  caller did that right;
* both returned matplotlib objects, while the Core declares ``returns="self"`` and the other tiers answer
  that way.

The guide now hangs on the layer's colour :class:`~digitalearth.base.spec.encoding.Encoding`, which is what
travels — so the key moves, hides and disappears with the layer for free.
"""

import numpy as np
import pytest

from digitalearth.base.spec import Symbology
from digitalearth.static import Map
from digitalearth.static.guides import color_encoding, source_field
from digitalearth.static.renderer import DrawnLayer


@pytest.fixture
def keyed(dataset):
    """A map with one raster field on it, `acc`, and a text label added **after** it.

    The label is the whole point of the fixture: it is the layer that used to take the key, because it is the
    most recent one and `colorbar` counted positions.

    Args:
        dataset: The committed ``acc4000`` raster.

    Yields:
        The map. Closed afterwards.
    """
    canvas = Map(crs=dataset.epsg)
    canvas.field(dataset, name="acc")
    canvas.text(0.0, 0.0, "caption", name="caption")
    yield canvas
    canvas.close()


class TestWhatDrivesTheColourIsPublished:
    """Step one of order 24: a guide needs an encoding to hang on, and no static builder published one."""

    def test_a_field_publishes_the_band_it_is_coloured_by(self, dataset):
        """The layer records the band name and the limits the render settled on.

        Args:
            dataset: The committed ``acc4000`` raster.

        Test scenario:
            `Encoding.__post_init__` takes exactly one of a constant or a field, so an *unbound* colour is no
            binding at all and a guide has nothing to attach to — which is why publishing this is the
            prerequisite rather than a nicety. The scale is read off the artist's own norm, so it is the range
            matplotlib coloured through rather than a second computation beside it.
        """
        with Map(crs=dataset.epsg) as canvas:
            image = canvas.field(dataset, name="acc")
            encoding = canvas.get_layer("acc").symbology.encoding("color")
            assert encoding is not None, "the field published no colour encoding"
            assert encoding.field == "Band_1", encoding.field
            assert encoding.scale.as_limits() == (
                float(image.norm.vmin),
                float(image.norm.vmax),
            ), "the published scale is not the range the picture was drawn through"

    def test_a_classified_fill_publishes_its_column_and_its_class_edges(self, polygons):
        """A graduated choropleth publishes the column, and the edges cleopatra actually cut.

        Args:
            polygons: Buffered point features with a numeric ``fid`` column.

        Test scenario:
            The web tier records a classification in a side table; here the classification the *picture*
            carries is the ``BoundaryNorm`` on the mappable, so that is what the scale is built from — the same
            reading `Scene._record_class_edges` takes for ``last_breaks``.
        """
        with Map(crs=polygons.epsg) as canvas:
            fill = canvas.choropleth(
                polygons, column="fid", scheme="quantiles", k=3, name="grad"
            )
            scale = canvas.get_layer("grad").symbology.encoding("color").scale
            assert scale.is_classified, "a classified fill published a continuous scale"
            assert list(scale.breaks) == [float(edge) for edge in fill.norm.boundaries]

    def test_an_outline_publishes_nothing_and_a_key_on_it_is_refused(self, polygons):
        """A flat-coloured layer has no colour key, and says so instead of drawing an empty one.

        Args:
            polygons: Buffered point features.

        Test scenario:
            The refusal the web tier already gives ("was not drawn with a classification, so it has no colour
            key to show"). An outline-only `PolyCollection` still carries a `Normalize` — matplotlib gives
            every collection one — so the test that decides this cannot be "does it have a norm".
        """
        with Map(crs=polygons.epsg) as canvas:
            canvas.polygons(polygons, name="edges")
            assert canvas.get_layer("edges").symbology.encoding("color") is None
            with pytest.raises(ValueError, match="has no colour key to show"):
                canvas.colorbar("edges")

    def test_a_layer_whose_artist_holds_no_values_publishes_nothing(self, points_fc):
        """Naming a field is not enough: the artist has to be colouring *through* values.

        Args:
            points_fc: A point collection, stripped of every attribute.

        Test scenario:
            The gate that a `norm` cannot be. A collection with no numeric column draws uniform markers, and
            the `PathCollection` still carries a `Normalize` — so "it has a norm" would publish a colour
            encoding for a layer whose colour is one flat constant, and a bar over it would label nothing. The
            reading that decides it is whether the artist holds the value array it colours through.
        """
        bare = points_fc[["geometry"]]
        with Map(crs=points_fc.epsg) as canvas:
            markers = canvas.points(bare, name="plain")
            assert markers.get_array() is None, (
                "the fixture is colouring by value after all, so it proves nothing"
            )
            assert canvas.get_layer("plain").symbology.encoding("color") is None, (
                "a flat-coloured scatter published a colour encoding"
            )

    def test_a_layer_that_names_no_field_publishes_nothing(self):
        """`color_encoding` needs a field name, whatever the artist looks like.

        Test scenario:
            The unit at the bottom of the rule: a caller's own artist reaches it with no field, and the answer
            is no encoding rather than one bound to a guessed name.
        """
        assert color_encoding(None, DrawnLayer()) is None
        assert color_encoding("", DrawnLayer()) is None

    def test_a_nameless_band_is_named_by_its_number(self):
        """A bare numpy array names no band, so the band is named by the only thing the layer records.

        Test scenario:
            `source_field` is the one place the tier spells this, and a raster that names its bands must win
            over the fallback — otherwise every GeoTIFF would be keyed ``band 1``.
        """
        assert source_field(object(), 2) == "band 2"

        class _Named:
            band_names = ["elevation", "slope"]

        assert source_field(_Named(), 2) == "slope"


class TestTheDefaultKeyIsTheLayerTheReaderNeeds:
    """The divergence #261 names: "the most recent layer" is not "the last layer added"."""

    def test_a_label_added_after_the_data_does_not_take_the_key(self, keyed):
        """`colorbar()` keys the field, although a text layer was added after it.

        Args:
            keyed: A map with a raster and then a text label.

        Test scenario:
            The bug the positional default had: the text layer is the most recent one, so ``layer=-1`` keyed
            whatever had registered last. Asserted on the layer the key landed on rather than on the number of
            axes, since a bar drawn for the wrong layer also adds an axes.
        """
        assert keyed.colorbar() is keyed
        assert keyed.get_layer("acc").symbology.guide() is not None, (
            "the raster, which is the only layer coloured by a value, was not keyed"
        )
        assert keyed._renderer.drawn["acc"].guides != (), "no bar was drawn for it"
        assert keyed._renderer.drawn["caption"].guides == (), "the text label was keyed"

    def test_the_most_recent_of_several_keyed_layers_wins(self, dataset):
        """Among layers that *are* coloured by a value, the default is the latest.

        Args:
            dataset: The committed ``acc4000`` raster.

        Test scenario:
            The other half of the rule: "skip the layers with no colour" must not become "always the first
            one".
        """
        with Map(crs=dataset.epsg) as canvas:
            canvas.field(dataset, name="first")
            canvas.field(dataset, name="second")
            canvas.colorbar()
            assert canvas.get_layer("second").symbology.guide() is not None
            assert canvas.get_layer("first").symbology.guide() is None


class TestTheKeyIsRecordedBeforeItIsDrawn:
    """The record is the point of the order: it is what makes the key follow the layer."""

    def test_the_guide_survives_a_round_trip_through_the_dict_form(self, keyed):
        """A figure written down and read back still asks for the key it was given.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            Figure furniture drawn on the axes could not be written down at all. A guide on the encoding is
            part of `Symbology`, so it round-trips with everything else the layer records.
        """
        keyed.colorbar(label="flow (m3/s)")
        stored = keyed.get_layer("acc").symbology.to_dict()
        assert Symbology.from_dict(stored).guide().title == "flow (m3/s)"

    def test_the_units_title_the_key_when_the_caller_names_none(self, keyed, mocker):
        """`label=None` resolves the layer's own units **into the record**, and `""` records no title.

        Args:
            keyed: A map with a raster and a text label.
            mocker: Used to give the layer units to resolve.

        Test scenario:
            A `Guide` refuses a blank title, so "no label at all" cannot be spelled as one — the resolution
            happens where the caller's intent is still known, which is at the record.
        """
        mocker.patch.object(keyed, "_default_label", return_value="m3/s")
        keyed.colorbar()
        assert keyed.get_layer("acc").symbology.guide().title == "m3/s"
        keyed.colorbar(label="")
        assert keyed.get_layer("acc").symbology.guide().title is None

    def test_asking_twice_replaces_the_key_rather_than_stacking_one(self, keyed):
        """One recorded guide is one drawn key, however many times the draw is reached.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            The trap of recording *and* drawing while a later re-render also draws from the record. The bar is
            taken off before it is drawn again, so the figure never grows a second identical strip.
        """
        keyed.colorbar()
        first = len(keyed.fig.axes)
        keyed.colorbar(label="again")
        assert len(keyed.fig.axes) == first, "a second bar was stacked beside the first"
        assert len(keyed._renderer.drawn["acc"].guides) == 1


class TestTheKeyFollowsItsLayer:
    """The three things a key keyed by position could not do."""

    def test_removing_the_layer_takes_the_key_with_it(self, keyed):
        """A removed layer leaves no bar behind.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            The bar was figure decoration, so `remove_layer` took the image off the axes and left its
            colorbar labelling nothing.
        """
        keyed.colorbar()
        assert len(keyed.fig.axes) == 2
        keyed.remove_layer("acc")
        assert len(keyed.fig.axes) == 1, "the key outlived the layer it explained"

    def test_hiding_the_layer_hides_the_key(self, keyed):
        """`set_visible(False)` hides the key, and switching the layer back on brings it back.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            A `Colorbar` is not an artist and has no ``set_visible`` of its own, so hiding one means hiding the
            axes it was drawn on — which is exactly the kind of thing a shared toggle gets wrong silently.
        """
        keyed.colorbar()
        bar = keyed._renderer.drawn["acc"].guides[0]
        keyed.set_visible("acc", False)
        assert bar.ax.get_visible() is False, "the key stayed on with the layer hidden"
        keyed.set_visible("acc", True)
        assert bar.ax.get_visible() is True, "the key did not come back with the layer"

    def test_a_key_asked_for_on_a_hidden_layer_is_drawn_hidden(self, keyed):
        """A bar created on a layer that is already hidden starts hidden too.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            The other half of `test_hiding_the_layer_hides_the_key`, and the symptom order 24 names as its
            motivation: `set_visible` hid the key it found, but the *draw* never asked whether the layer it
            explains is on, so keying a hidden layer put a bar on the figure labelling a picture the reader
            cannot see. Hiding then keying and keying then hiding must leave the same figure.
        """
        keyed.set_visible("acc", False)
        keyed.colorbar("acc")
        bar = keyed._renderer.drawn["acc"].guides[0]
        assert bar.ax.get_visible() is False, (
            "a bar drawn onto a hidden layer is on the figure explaining a picture nobody can see"
        )
        keyed.set_visible("acc", True)
        assert bar.ax.get_visible() is True, (
            "the key did not come back when its layer did"
        )

    def test_colorbars_honours_visibility_too(self, keyed):
        """The plural draws each bar as hidden as the layer it explains.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            `colorbars()` reads the layer tree and keys everything coloured by a value; it never consulted
            visibility either, so "key everything" un-hid what `set_visible` had put away.
        """
        keyed.set_visible("acc", False)
        keyed.colorbars()
        assert keyed._renderer.drawn["acc"].guides[0].ax.get_visible() is False, (
            "colorbars() drew a visible bar for a hidden layer"
        )

    def test_moving_the_layer_does_not_orphan_the_key(self, keyed):
        """A reorder leaves the key attached to the layer it explains.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            `move_layer` re-arranges the artists the renderer holds and deals their z-orders back out. A
            colorbar lives on its own axes, so it must be outside that arrangement and still survive it.
        """
        keyed.colorbar()
        bar = keyed._renderer.drawn["acc"].guides[0]
        keyed.move_layer("acc", 0)
        assert keyed.layer_ids == ["acc", "caption"]
        assert keyed._renderer.drawn["acc"].guides == (bar,)
        assert len(keyed.fig.axes) == 2

    def test_a_rebuilt_layer_draws_its_key_again(self, keyed):
        """A restyle takes the layer off the axes and draws it again — key and all.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            The reason `Renderer.draw_layer` draws the recorded guide rather than leaving it to the caller's
            call: a layer redrawn from its description would otherwise come back without the key the
            description still asks for.
        """
        keyed.colorbar(label="flow")
        described = keyed.get_layer("acc")
        keyed.replace_layer(
            described.__class__(
                described.id,
                described.kind,
                source_id=described.source_id,
                symbology=described.symbology.with_props(cmap="magma"),
                label=described.label,
                visible=described.visible,
                band=described.band,
            )
        )
        drawn = keyed._renderer.drawn["acc"]
        assert len(drawn.guides) == 1, "the rebuilt layer lost its key"
        assert drawn.guides[0].mappable is drawn.artist, (
            "the key still points at the artist the rebuild replaced"
        )


class TestTheLayerIsResolvedBeforeVisibleIsHonoured:
    """Review L7 on the web tier, one tier over: one spelling must not be valid only half the time."""

    def test_an_unknown_id_is_refused_even_with_visible_false(self, keyed):
        """`colorbar("nope", visible=False)` is as wrong as `colorbar("nope")`.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            The bug shape: an early return on the flag, taken before the argument that names the layer is
            checked, so a typo is an error in one call and silence in the other.
        """
        with pytest.raises(KeyError):
            keyed.colorbar("nope", visible=False)
        with pytest.raises(ValueError, match="has no colour key to show"):
            keyed.colorbar("caption", visible=False)

    def test_visible_false_records_the_guide_and_draws_nothing(self, keyed):
        """The flag reaches the record, so switching the key on later needs no second description.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            ``visible=False`` is "no key, please" rather than "no guide" — which is what lets it also take an
            already-drawn key off again.
        """
        keyed.colorbar(visible=False)
        assert keyed.get_layer("acc").symbology.guide().show is False
        assert len(keyed.fig.axes) == 1, "a bar was drawn although none was asked for"
        keyed.colorbar()
        assert len(keyed.fig.axes) == 2
        keyed.colorbar(visible=False)
        assert len(keyed.fig.axes) == 1, "the bar was not taken off again"


class TestACategoricalFillIsKeyedBySwatches:
    """A bar over the class codes cleopatra assigned reads ``0, 1, 2 …`` where the labels belong."""

    @pytest.fixture
    def zoned(self, polygons):
        """A map with one categorical fill on it, `zone`.

        Args:
            polygons: Buffered point features.

        Yields:
            The map. Closed afterwards.
        """
        polygons["zone"] = ["urban", "rural"] * (len(polygons) // 2) + ["urban"] * (
            len(polygons) % 2
        )
        canvas = Map(crs=polygons.epsg)
        canvas.choropleth(polygons, column="zone", scheme="categorical", name="zone")
        yield canvas
        canvas.close()

    def test_the_scale_carries_the_colours_the_glyph_drew(self, zoned):
        """The categories and colours are read off the swatch legend cleopatra built.

        Args:
            zoned: A map with a categorical fill.

        Test scenario:
            Deriving them a second time from the values would agree only while nobody edits one of the two —
            the disagreement `LegendSpec` exists to remove.
        """
        scale = zoned.get_layer("zone").symbology.encoding("color").scale
        drawn = zoned._renderer.drawn["zone"].glyph.category_legend
        assert list(scale.categories) == [text.get_text() for text in drawn.get_texts()]

    def test_a_bar_is_refused_and_names_the_legend_instead(self, zoned):
        """`colorbar()` on a categorical fill refuses, and leaves the layer as it found it.

        Args:
            zoned: A map with a categorical fill.

        Test scenario:
            Refused **before** the guide is recorded, so a call this tier cannot answer leaves both halves of
            the layer as it found them: no description that can never be drawn, and — the half the rollback
            inside `_record_key` cannot give back — the key the layer already had, still on the axes. The
            drawing path takes the previous key off before it draws the next one, so a refusal reached from
            there would have removed the legend on its way out.
        """
        zoned.legend(title="Zone")
        with pytest.raises(ValueError, match="ask for legend\\(\\) instead"):
            zoned.colorbar()
        assert zoned.get_layer("zone").symbology.guide() is not None, (
            "the refused bar took the legend's guide off the layer"
        )
        assert zoned.get_layer("zone").symbology.props["guide_kind"] == "legend", (
            "the refused bar was recorded as the layer's key anyway"
        )
        assert zoned.ax.get_legend() is not None, (
            "the refused bar took the drawn legend off the axes"
        )

    def test_the_legend_rows_are_the_categories(self, zoned):
        """`legend()` draws one swatch per category, in the order the colours were assigned.

        Args:
            zoned: A map with a categorical fill.

        Test scenario:
            The rows come from the layer's scale rather than from arguments, which is the whole of what
            `legend(colors, labels)` is replaced by.
        """
        assert zoned.legend(title="Zone") is zoned
        rows = [text.get_text() for text in zoned.ax.get_legend().get_texts()]
        assert rows == ["rural", "urban"]

    def test_a_swatch_key_on_a_hidden_fill_is_drawn_hidden(self, zoned):
        """The swatch half of the same rule: a `Legend` drawn onto a hidden layer starts off.

        Args:
            zoned: A map with a categorical fill.

        Test scenario:
            A `Legend` *is* an artist, unlike a `Colorbar`, so the two kinds of key carry their flag in
            different places — which is exactly the sort of difference a rule applied at one of them and not
            the other slips through.
        """
        zoned.set_visible("zone", False)
        zoned.legend("zone", title="Zone")
        assert zoned._renderer.drawn["zone"].guides[0].get_visible() is False, (
            "a swatch key drawn onto a hidden fill is on the figure"
        )

    def test_colorbars_skips_it_rather_than_raising_on_it(self, zoned, dataset):
        """The plural keys what it can and passes over what it cannot.

        Args:
            zoned: A map with a categorical fill.
            dataset: A raster added beside it, which *can* carry a bar.

        Test scenario:
            "Key everything" must not stop at the first layer whose key is not a bar — the singular is the
            call that refuses.
        """
        zoned.field(dataset, name="acc")
        assert zoned.colorbars() is zoned
        assert len(zoned.fig.axes) == 2, (
            "one data axes and one bar, for the raster only"
        )
        assert zoned.get_layer("zone").symbology.guide() is None
        assert zoned.get_layer("acc").symbology.guide() is not None


class TestDerivedValuesAreNamedByWhatTheyAre:
    """A layer whose kind computes its own values has no column to name, and still needs one."""

    @pytest.mark.parametrize(
        ("builder", "expected"),
        [("quadtree", "count"), ("kde", "density")],
    )
    def test_a_computed_quantity_publishes_the_tier_s_own_word_for_it(
        self, points_fc, builder, expected
    ):
        """A quadtree counts and a KDE estimates a density, so those are the fields they publish.

        Args:
            points_fc: A point collection.
            builder: The builder under test.
            expected: The field name it must publish.

        Test scenario:
            Without this the one-call `quadtree()` and `kde()` wrappers silently stopped drawing the bar they
            document, because a layer that names no field publishes no encoding.
        """
        with Map(crs=points_fc.epsg) as canvas:
            kwargs = {"nmax": 1} if builder == "quadtree" else {}
            getattr(canvas, builder)(points_fc, name="cells", **kwargs)
            encoding = canvas.get_layer("cells").symbology.encoding("color")
            assert encoding is not None, f"{builder}() published no colour encoding"
            assert encoding.field == expected, (
                f"{builder}() publishes {encoding.field!r}, not {expected!r}"
            )

    def test_a_uv_field_is_coloured_by_magnitude(self, dataset):
        """Arrows carry no column either: they are coloured by the vector's magnitude.

        Args:
            dataset: The raster used as both components.

        Test scenario:
            The pair is the layer's one source, so neither component's band names what the colour varies with.
        """
        with Map(crs=dataset.epsg) as canvas:
            canvas.quiver(dataset, dataset, name="wind")
            encoding = canvas.get_layer("wind").symbology.encoding("color")
            assert encoding is not None, "quiver() published no colour encoding"
            assert encoding.field == "magnitude", (
                f"the arrows are coloured by {encoding.field!r}, not by their magnitude"
            )


@pytest.fixture
def polygons(dataset):
    """Buffered point features in the raster's CRS, with a numeric ``fid`` column.

    Args:
        dataset: Only for its CRS, so the fixtures share one display CRS.

    Returns:
        The collection.
    """
    from pyramids.feature import FeatureCollection

    collection = FeatureCollection.read_file("tests/data/points.geojson")
    collection["geometry"] = collection.geometry.buffer(500.0)
    return collection


@pytest.fixture
def points_fc():
    """The committed point collection, unbuffered.

    Returns:
        The collection.
    """
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


def test_a_key_is_not_in_the_layer_s_artists(keyed):
    """The drawn key is held beside the layer's artists, never among them.

    Args:
        keyed: A map with a raster and a text label.

    Test scenario:
        `artists` is what `Renderer._repaint` arranges in draw order and what `_rank_zorders` writes z-orders
        into; a colorbar lives on its own axes where neither applies, so putting it there would feed the
        z-order ranking an artist that is not painted on the data axes. What the two share is the layer's
        lifetime, not its stacking.
    """
    keyed.colorbar()
    drawn = keyed._renderer.drawn["acc"]
    assert drawn.artists == (drawn.artist,), drawn.artists
    assert drawn.guides != (), "the drawn key was not held beside the layer"
    assert drawn.guides[0] not in drawn.artists, (
        f"the key is among the layer's artists: {drawn.artists}"
    )
    assert np.isclose(drawn.artist.get_zorder(), keyed.ax.images[0].get_zorder())
