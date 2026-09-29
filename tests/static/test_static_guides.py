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

    def test_moving_the_layer_does_not_orphan_the_key(self, dataset):
        """A reorder leaves each key attached to the layer it explains.

        Args:
            dataset: The committed ``acc4000`` raster, drawn twice so there is something to swap.

        Test scenario:
            `move_layer` re-arranges the artists the renderer holds and deals their z-orders back out. A
            colorbar lives on its own axes, so it must be outside that arrangement and still survive it.

            Built with **two** keyed rasters rather than a raster and a text label, because the two sit in
            different draw-order bands and `LayerTree.move` confines a move to one band: with one layer in
            the data band there is nowhere for it to go, so the move this test is named for could not
            happen and `Renderer._repaint` was never called (review M11). Reading `ax.images` on both sides
            of the move is what says the reorder reached the *picture* — the ids alone would still swap if
            the renderer ignored the move entirely, which is how the no-op version of this test passed with
            `move_layer` replaced by `pass`. The z-order half of `_repaint` is deliberately not asserted
            here: `_rank_zorders` deals back out the values the layers already held, and two rasters both
            hold matplotlib's default, so it has nothing to distinguish them with.

            The closing line reads `DrawnLayer.artists`, which is the list `_repaint` arranges, rather than
            `canvas.ax.get_children()`. It asserted the latter until review L6: a `Colorbar`'s axes is
            created figure-level by `make_axes` and is never a child of the data axes, so that spelling
            could only have failed for an inset axes — which nothing here produces — and dealing the key
            into the layers' own arrangement, the defect the message names, left it green.
        """
        with Map(crs=dataset.epsg) as canvas:
            canvas.field(dataset, name="lower")
            canvas.field(dataset, name="upper")
            canvas.colorbar("lower", label="L")
            canvas.colorbar("upper", label="U")
            drawn = canvas._renderer.drawn
            lower_bar, upper_bar = drawn["lower"].guides[0], drawn["upper"].guides[0]
            lower, upper = drawn["lower"].artist, drawn["upper"].artist
            painted = [id(image) for image in canvas.ax.images]
            assert canvas.layer_ids == ["lower", "upper"]
            assert painted.index(id(lower)) < painted.index(id(upper))
            canvas.move_layer("lower", 1)
            assert canvas.layer_ids == ["upper", "lower"], "the move did not happen"
            painted = [id(image) for image in canvas.ax.images]
            assert painted.index(id(lower)) > painted.index(id(upper)), (
                "the reorder reached the description but not the artists on the axes"
            )
            drawn = canvas._renderer.drawn
            assert drawn["lower"].guides == (lower_bar,), (
                "the moved layer lost the key that explains it"
            )
            assert drawn["upper"].guides == (upper_bar,)
            assert len(canvas.fig.axes) == 3, (
                "one data axes and one bar per keyed layer; a reorder created or dropped one"
            )
            assert lower_bar not in drawn["lower"].artists, (
                "the key was dealt into the layers' own arrangement"
            )

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


class TestAKeyIsTitledByWhatTheCallerWrote:
    """`label=`/`title=` is the caller's word, and a blank one is the caller's word for "none"."""

    def test_a_blank_label_names_no_key_rather_than_raising(self, keyed):
        """`colorbar(label="   ")` draws an unlabelled bar instead of refusing.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            `_title_for` returns `asked or None`, so `""` correctly became "no title" — but a whitespace-only
            label is truthy, so it travelled on to `Guide`, which refuses a blank title. The caller wrote
            `label=` and got back `ValueError: Guide title must be a non-empty string or None; got '   '`,
            naming a type they have never heard of for an argument they did not pass (review L5). Two
            spellings of "no label" must not part company at the space bar.
        """
        assert keyed.colorbar("acc", label="   ") is keyed
        assert keyed.get_layer("acc").symbology.guide().title is None, (
            "a blank label was recorded as the key's title"
        )
        keyed.legend("acc", title="\t\n ")
        assert keyed.get_layer("acc").symbology.guide().title is None, (
            "title= and label= disagree about what a blank one means"
        )

    def test_a_padded_label_reaches_the_record_as_it_was_written(self, keyed):
        """A label that is not blank is recorded exactly as the caller wrote it.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            The other side of the blank-label fix, and the reason it tests `strip()` rather than assigning
            it: trimming the caller's padding would be a second, silent change to what the figure records,
            and `Guide` itself does not trim — it only refuses a title with nothing in it.
        """
        keyed.colorbar("acc", label="  Flow  ")
        assert keyed.get_layer("acc").symbology.guide().title == "  Flow  ", (
            "the caller's label was trimmed on its way into the record"
        )

    @pytest.mark.parametrize("asked", [123, 1.5, ("Flow",)])
    def test_a_label_that_is_not_text_is_refused_by_name(self, keyed, asked):
        """A non-string `label=`/`title=` is answered by `Guide`, not by `str.strip`.

        Args:
            keyed: A map with a raster and a text label.
            asked: A label of a type a title cannot be.

        Test scenario:
            `_title_for` reads a blank label as "no label", and the reading is `asked.strip()` — which a
            number has no method for, so the caller got `AttributeError: 'int' object has no attribute
            'strip'` where every other tier answers `ValueError: Guide title must be a non-empty string or
            None; got 123` (review M3). `Scene.colorbar` and `Scene.legend` document `KeyError` and
            `ValueError` only, and `api.UNMAPPABLE` holds `AttributeError`, so a `quickmap` path swallowed
            it as "this layer has no key to draw".

            Parametrised over three types rather than one so the fix cannot be a check for `int`, and
            asserted on both spellings because `label=` and `title=` resolve through the same helper — the
            reason review L5's blank-label fix had to cover both.
        """
        with pytest.raises(ValueError, match="Guide title must be a non-empty string"):
            keyed.colorbar("acc", label=asked)
        with pytest.raises(ValueError, match="Guide title must be a non-empty string"):
            keyed.legend("acc", title=asked)
        assert keyed.get_layer("acc").symbology.guide() is None, (
            "a refused label left a key described on the layer"
        )


class TestTheLayerIsResolvedBeforeVisibleIsHonoured:
    """Review L7 on the web tier, one tier over: one spelling must not be valid only half the time.

    Both halves of the call are checked here, deliberately. The **layer** is resolved by `Scene._keyed_layer`
    before anything is recorded, and the two tests below cover that. The **key** is checked inside
    `guides.draw_guide`, which used to sit behind the `guide.show` early return — so the refusals that
    describe the key, rather than the layer, were skipped by exactly the flag this class exists to police
    (review M1). A class that covered only the half that already held reported the property as pinned while
    it did not hold, which is worse than not testing it at all.
    """

    @staticmethod
    def _strip_the_scale(canvas, layer_id: str) -> None:
        """Leave a layer publishing a colour field with no scale behind it.

        Args:
            canvas: The map holding the layer.
            layer_id: The layer to strip.

        Test scenario:
            The state a figure built elsewhere arrives in: `Encoding` takes a scale or no scale, and a tier
            that measured no limits publishes none rather than inventing a pair. There is no keyword on this
            tier that produces it, so it is written onto the tree the way `from_dict` would.
        """
        from dataclasses import replace as with_fields

        from digitalearth.base.spec import Encoding

        layer = canvas.get_layer(layer_id)
        encodings = dict(layer.symbology.encodings)
        encodings["color"] = Encoding.by_field("color", "Band_1", scale=None)
        canvas._layer_tree = canvas._layer_tree.replace(
            with_fields(
                layer,
                symbology=with_fields(layer.symbology, encodings=encodings),
            )
        )

    def test_labels_that_do_not_number_the_rows_are_refused_either_way(self, keyed):
        """`legend(labels=["one"], visible=False)` is as wrong as `legend(labels=["one"])`.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            The refusal this class is named for, on the half it never reached. A continuous scale keys as
            five swatches, so one label is one label for five rows whichever way the flag is set — and a
            description that can never be drawn must not be recordable just because nothing is drawn from it
            yet. `visible=False` is "no key, please", not "no checking, please": switching it back on later
            would then raise from a call the caller has long finished making.
        """
        with pytest.raises(ValueError, match="needs 5 labels; got 1") as shown:
            keyed.legend(labels=["one"])
        with pytest.raises(ValueError, match="needs 5 labels; got 1") as hidden:
            keyed.legend(labels=["one"], visible=False)
        assert str(shown.value) == str(hidden.value), (
            "the two spellings refuse with different messages"
        )
        assert keyed.get_layer("acc").symbology.guide() is None, (
            "the refused call left a guide on the layer"
        )
        assert "guide_labels" not in keyed.get_layer("acc").symbology.props, (
            "the refused labels were recorded anyway"
        )

    def test_an_empty_label_list_is_refused_like_any_other_wrong_count(self, keyed):
        """`legend(labels=[])` is a count that does not number the rows, so it is refused too.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            `_rows` asked `if not labels`, so the empty list read as "the caller gave none" and the rows
            were derived instead — `legend(labels=["a"])` raised on a five-row key while `legend(labels=[])`
            was accepted, and the refusal could never say "got 0" (review L1). Web and interactive both
            refuse it. `None` is the spelling for "derive them"; an empty list is a list, and the argument
            must not be valid only for one of its values.
        """
        with pytest.raises(ValueError, match="needs 5 labels; got 0"):
            keyed.legend(labels=[])
        assert keyed.get_layer("acc").symbology.guide() is None, (
            "the refused call left a guide on the layer"
        )
        # The neighbouring spelling still derives, so the fix is "an empty list is a count", not "any
        # falsy labels argument is refused".
        keyed.legend(labels=None)
        assert keyed.get_layer("acc").symbology.guide().show is True
        assert "guide_labels" not in keyed.get_layer("acc").symbology.props, (
            "labels=None must clear the record rather than store an empty override"
        )

    def test_a_layer_with_no_scale_is_refused_either_way(self, keyed):
        """The other refusal raised inside the draw: no scale, so no rows to key it with.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            The second of the two checks `visible=False` skipped. It is a different shape from the labels
            one — that one is about the caller's argument, this one about what the layer publishes — and a
            fix that hoists only the argument check would leave this one behind, which is how the web tier's
            L7 fix came to miss `labels` in the first place.
        """
        self._strip_the_scale(keyed, "acc")
        with pytest.raises(ValueError, match="publishes no colour scale"):
            keyed.legend("acc")
        with pytest.raises(ValueError, match="publishes no colour scale"):
            keyed.legend("acc", visible=False)
        assert keyed.get_layer("acc").symbology.guide() is None, (
            "the refused call left a guide on the layer"
        )

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
        """The flag reaches the record, so the figure says the key was decided rather than never asked for.

        Args:
            keyed: A map with a raster and a text label.

        Test scenario:
            ``visible=False`` is "no key, please" rather than "no guide" — which is what lets it also take an
            already-drawn key off again.

            The summary line said "so switching the key on later needs no second description", which is the
            claim review L2 found false one level up: the record keeps the title, but a later
            ``colorbar()`` with no ``label=`` re-derives it rather than reading the switched-off guide's
            own. What the flag buys is the *record* of the decision, not a description the next call reuses.
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

    def test_the_renderer_accepts_a_title_of_its_own(self, zoned):
        """`Renderer.draw_guide(layer, title=...)` styles the call rather than raising.

        Args:
            zoned: A map with a categorical fill.

        Test scenario:
            `Renderer.draw_guide` is public and documents `**kwargs` as "styling forwarded to the matplotlib
            colorbar or legend" — but the swatch arm passed `title=guide.title` *beside* those kwargs, so
            the one keyword a legend most obviously takes was the one that could not be given: `TypeError:
            disjoint_legend() got multiple values for keyword argument 'title'`. `Scene.legend` cannot reach
            it, because `title` is a named parameter there, which is why it survived untested (review L4).
        """
        zoned.legend("zone", title="Zone")
        made = zoned._renderer.draw_guide(zoned.get_layer("zone"), title="mine")
        assert made.get_title().get_text() == "mine", (
            "the caller's own title did not reach the legend"
        )
        zoned._renderer.draw_guide(zoned.get_layer("zone"))
        assert zoned.ax.get_legend().get_title().get_text() == "Zone", (
            "a call that names no title lost the recorded one"
        )

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


class TestARefusedKeyTakesTheRebuildWithIt:
    """`Renderer.draw_layer` is all-or-nothing, and the key it draws is part of the "all".

    `draw_guide` was added to `draw_layer` *after* the guarded region and after the record was written, so a
    key the layer's description asks for and this tier cannot draw left the drawer's artists on the axes and
    the layer in `_drawn` while the refusal carried on — against the method's own note, which promises that
    whatever it added is taken back before the error does (review M5).
    """

    @pytest.fixture
    def undrawable(self, polygons):
        """A map with one categorical fill whose recorded key asks for more rows than it has.

        Args:
            polygons: Buffered point features.

        Yields:
            The map, with the fill removed from the axes again so the next draw is a rebuild. Closed
            afterwards.
        """
        from dataclasses import replace as with_fields

        from digitalearth.static.guides import GUIDE_LABELS_KEY

        polygons["zone"] = ["urban", "rural"] * (len(polygons) // 2) + ["urban"] * (
            len(polygons) % 2
        )
        canvas = Map(crs=polygons.epsg)
        canvas.choropleth(polygons, column="zone", scheme="categorical", name="zone")
        canvas.legend("zone")
        # One label for a two-row key, written onto the description the way a figure built elsewhere
        # carries it — the refusal `_rows` raises is the one reached from inside the draw.
        layer = canvas.get_layer("zone")
        props = dict(layer.symbology.props)
        props[GUIDE_LABELS_KEY] = ("only-one",)
        canvas._layer_tree = canvas._layer_tree.replace(
            with_fields(layer, symbology=with_fields(layer.symbology, props=props))
        )
        canvas._renderer.remove("zone")
        yield canvas
        canvas.close()

    def test_the_drawing_comes_off_and_the_record_stays_empty(self, undrawable):
        """A refusal from the key leaves neither artists nor a record behind.

        Args:
            undrawable: A map whose fill describes a key that cannot be drawn.

        Test scenario:
            This is the shape `_reconcile` takes for a restyle and a rebuild — remove, then draw again — so
            a half-applied draw here is artists no layer owns and a `_drawn` entry for a layer the axes is
            not showing. The two halves have to go back together, which is the tier's own rollback contract.

            `figure_spec` is read above the block, not inside it: it is a property, so a raise from *it*
            would satisfy the block and leave `draw_layer` unrun (R2-M12, measured — with the property made
            to raise this very message the test passed).
        """
        children = len(undrawable.ax.get_children())
        spec = undrawable.figure_spec
        with pytest.raises(ValueError, match="needs 2 labels; got 1"):
            undrawable._renderer.draw_layer(spec, "zone")
        assert "zone" not in undrawable._renderer.drawn, (
            "the refused rebuild left the layer in the renderer's record"
        )
        assert len(undrawable.ax.get_children()) == children, (
            "the refused rebuild left its artists on the axes"
        )

    def test_a_second_draw_replaces_the_previous_key_rather_than_orphaning_it(
        self, dataset
    ):
        """Drawing a layer again takes the bar it already had off, rather than stacking a second.

        Args:
            dataset: The committed ``acc4000`` raster.

        Test scenario:
            The record was overwritten with a fresh `DrawnLayer` — `guides=()` — *before* `draw_guide` ran,
            so `draw_guide`'s first act, "take the layer's previous key off", iterated an empty tuple and
            the previous key was orphaned rather than detached. A colorbar is what makes that visible: it
            lives on its own axes, so an orphan is a second strip stealing width from the map. Latent on the
            reconcile path, which always removes first, and reachable by anyone calling `draw_layer` on a
            layer that is already drawn.
        """
        with Map(crs=dataset.epsg) as canvas:
            canvas.field(dataset, name="acc")
            canvas.colorbar("acc", label="flow")
            first = canvas._renderer.drawn["acc"].guides[0]
            axes = len(canvas.fig.axes)
            canvas._renderer.draw_layer(canvas.figure_spec, "acc")
            second = canvas._renderer.drawn["acc"].guides[0]
            assert second is not first, "the layer was not keyed again"
            assert len(canvas.fig.axes) == axes, (
                "the previous bar was orphaned rather than taken off, so the figure carries two"
            )
            assert first.ax not in canvas.fig.axes

    def test_a_refused_redraw_puts_back_the_drawing_and_the_ask_it_replaced(self):
        """A layer that was already drawn goes back to the drawing it had, visibility ask and all.

        Test scenario:
            The other rollback test removes the layer first, so `previous` is `None` and the record is
            simply popped. The half nobody exercised is the one the arm was written for: a layer that **is**
            drawn, whose second draw refuses. `draw_layer` overwrites `_drawn` before it draws the key, so
            without the restore the layer would be left describing a drawing whose artists `undo()` has just
            taken off the axes.

            A graticule is the layer that reaches the second half too. Its drawer hands back a `DrawnLayer`
            with an `artist` and **no** `artists`, so `set_visible` has nothing on the axes to write the flag
            to and remembers it in `_asked` instead — the one state the `asked is not None` arm restores.
            The key is refused by asking for a bar over a categorical scale, which is the refusal
            `draw_guide` raises from inside the guarded region.
        """
        from dataclasses import replace as with_fields

        from digitalearth.base.spec import Encoding, Guide, Scale
        from digitalearth.static.guides import GUIDE_KIND_KEY

        with Map(crs=4326) as canvas:
            canvas.graticule(30, 30, name="grid")
            renderer = canvas._renderer
            renderer.set_visible("grid", False)
            drawn_before = renderer.drawn["grid"]
            layer = canvas.get_layer("grid")
            symbology = with_fields(
                layer.symbology,
                encodings={
                    **layer.symbology.encodings,
                    "color": Encoding.by_field(
                        "color",
                        "zone",
                        scale=Scale.categorical(["a", "b"], ["#f00", "#00f"]),
                    ),
                },
                props={**layer.symbology.props, GUIDE_KIND_KEY: "colorbar"},
            ).with_guide(Guide(show=True, title="zones"))
            canvas._layer_tree = canvas._layer_tree.replace(
                with_fields(layer, symbology=symbology)
            )
            # Read outside the block for the reason above: `figure_spec` is a property, so a raise from it
            # would satisfy the refusal and the draw would never run.
            spec = canvas.figure_spec
            with pytest.raises(ValueError, match="coloured by category"):
                renderer.draw_layer(spec, "grid")
            restored = renderer.drawn["grid"]
            assert restored is drawn_before, (
                "the refused redraw left the layer describing the drawing it rolled back, not the one it "
                f"had: {restored}"
            )
            assert restored.artist is not canvas._graticule_lines, (
                "the record still points at the lines the refused draw computed, so nothing was put back"
            )
            assert renderer.is_visible("grid") is False, (
                "the remembered visibility ask was dropped by the rollback, so a layer hidden before the "
                f"refusal reports {renderer.is_visible('grid')!r} after it"
            )

    def test_a_refused_redraw_leaves_the_bar_it_had_on_the_figure(self, dataset):
        """A layer that was keyed by a bar keeps the bar, not just the record of it.

        Args:
            dataset: The committed ``acc4000`` raster.

        Test scenario:
            The half of the rollback the record-side tests cannot see. `draw_guide`'s first act is to take
            the layer's previous key off, so when it then refuses, `_drawn[layer_id] = previous` put back a
            record naming a bar that had just been detached: the layer described a key the reader could not
            see, and `set_visible`/`remove` then toggled an artist that is not on the figure. It self-healed
            on the next successful `colorbar()`, which is why nothing noticed (review M4).

            The legend half looked symmetric and is not: `_PartialDraw.undo` restores the axes' single
            legend slot by assignment, and a *colorbar* has no slot — its axes was **removed**, which
            `undo` does not watch, and a removed axes cannot be re-added. So the fix is not in the
            rollback: every refusal the description can be checked for is now raised **before** the
            previous key comes off, and there is nothing to put back.

            The refusal used is a bar over a categorical scale, which is the one `bar_refusal` gives from
            inside the draw. `figure_spec` is read above the block because it is a property (a raise from
            it would satisfy the block and leave `draw_layer` unrun), as is `dataset.epsg`.
        """
        from dataclasses import replace as with_fields

        from digitalearth.base.spec import Encoding, Guide, Scale
        from digitalearth.static.guides import GUIDE_KIND_KEY

        epsg = dataset.epsg
        with Map(crs=epsg) as canvas:
            canvas.field(dataset, name="acc")
            canvas.colorbar("acc", label="flow")
            bar = canvas._renderer.drawn["acc"].guides[0]
            axes = len(canvas.fig.axes)
            layer = canvas.get_layer("acc")
            symbology = with_fields(
                layer.symbology,
                encodings={
                    **layer.symbology.encodings,
                    "color": Encoding.by_field(
                        "color",
                        "zone",
                        scale=Scale.categorical(["a", "b"], ["#f00", "#00f"]),
                    ),
                },
                props={**layer.symbology.props, GUIDE_KIND_KEY: "colorbar"},
            ).with_guide(Guide(show=True, title="zones"))
            canvas._layer_tree = canvas._layer_tree.replace(
                with_fields(layer, symbology=symbology)
            )
            spec = canvas.figure_spec
            with pytest.raises(ValueError, match="coloured by category"):
                canvas._renderer.draw_layer(spec, "acc")
            assert canvas._renderer.drawn["acc"].guides == (bar,), (
                "the refused redraw did not put the layer's own key back in the record"
            )
            assert bar.ax in canvas.fig.axes, (
                "the record names a bar that is no longer on the figure, so the layer describes a key the "
                "reader cannot see"
            )
            assert len(canvas.fig.axes) == axes, (
                f"the figure should still hold its {axes} axes; it holds {len(canvas.fig.axes)}"
            )


class TestOneAxesHoldsOneSwatchLegend:
    """matplotlib keeps one legend per axes, so keying a second layer takes the first layer's key off.

    The *drawing* has always worked that way and `Scene.legend` documents it. What did not was the
    **record**: the displaced layer kept `Guide(show=True)` on its encoding and kept the detached `Legend`
    in `DrawnLayer.guides`, so a figure written out claimed two swatch keys of which only one can ever be
    drawn, `set_visible` toggled an artist that is not on the axes, and which of the two the reader sees was
    decided by draw order. Keeping the record in step with the one legend slot is the whole of what order 24
    moved the guide onto the layer to do (review M4).
    """

    @pytest.fixture
    def two_fills(self, polygons):
        """A map with two categorical fills on it, `a` below and `b` above.

        Args:
            polygons: Buffered point features.

        Yields:
            The map. Closed afterwards.
        """
        polygons["zone"] = ["urban", "rural"] * (len(polygons) // 2) + ["urban"] * (
            len(polygons) % 2
        )
        canvas = Map(crs=polygons.epsg)
        canvas.choropleth(polygons, column="zone", scheme="categorical", name="a")
        canvas.choropleth(polygons, column="zone", scheme="categorical", name="b")
        yield canvas
        canvas.close()

    def test_the_displaced_layer_stops_claiming_a_key(self, two_fills):
        """Keying `b` takes `a`'s key off both halves of the record, not just off the axes.

        Args:
            two_fills: A map with two categorical fills.

        Test scenario:
            The axes' single legend slot is the fact; the two halves of the record have to agree with it. A
            `Guide(show=True)` on a layer whose key is not drawn is the disagreement the guide-on-the-layer
            model exists to remove, and it survives `to_dict()`.
        """
        two_fills.legend("a", title="A")
        first = two_fills._renderer.drawn["a"].guides[0]
        two_fills.legend("b", title="B")
        second = two_fills._renderer.drawn["b"].guides[0]
        assert two_fills.ax.get_legend() is second, (
            "the axes is not showing the key that was asked for last"
        )
        assert first not in list(two_fills.ax.get_children()), (
            "matplotlib is expected to have displaced the first legend"
        )
        assert two_fills._renderer.drawn["a"].guides == (), (
            "the displaced layer still holds a legend that is not on the axes"
        )
        assert two_fills.get_layer("a").symbology.guide().show is False, (
            "the displaced layer still describes a key the figure cannot draw"
        )
        assert two_fills.get_layer("b").symbology.guide().show is True

    @pytest.mark.parametrize("second_writer", ["legend()", "the fill's own glyph"])
    def test_removing_the_displaced_layer_leaves_the_drawn_key_alone(
        self, polygons, second_writer
    ):
        """Removing the layer whose key was displaced must not take the surviving key off the axes.

        Args:
            polygons: Buffered point features.
            second_writer: What put the surviving legend in the axes' slot.

        Test scenario:
            The sharpest consequence of the stale record, and a visible one. matplotlib gives every legend
            it makes for an axes the *same* removal hook — `Axes._remove_legend`, which sets `ax.legend_` to
            `None` whichever legend is sitting there — so `Renderer.remove` walking the displaced layer's
            recorded guides called `remove()` on an orphan and wiped **the other layer's** key off the
            figure. Before this fix: `remove_layer("a")` left `ax.get_legend()` as `None` while `b` still
            described and still held a key. This is also why the fix drops the displaced legend from the
            record rather than detaching it — detaching is the very call that does the damage.

            **Parametrised over who wrote that slot, because the renderer is not the only writer** — which
            is what `Renderer._displaced`'s own docstring says and what this test could not see until
            review R2-H1. It keyed **both** layers, so `draw_guide` always made the surviving legend and
            the reconcile hook always ran. A categorical fill's cleopatra glyph writes the slot on its way
            in, for a layer nobody has keyed; on that arm the hook never fired, `a` went on holding a
            legend matplotlib had already dropped, and `remove_layer("a")` wiped `b`'s swatches off the
            figure. Measured on the most ordinary flow there is — key one fill, add a second, remove the
            first.
        """
        polygons["zone"] = ["urban", "rural"] * (len(polygons) // 2) + ["urban"] * (
            len(polygons) % 2
        )
        with Map(crs=polygons.epsg) as canvas:
            canvas.choropleth(polygons, column="zone", scheme="categorical", name="a")
            canvas.legend("a", title="A")
            canvas.choropleth(polygons, column="zone", scheme="categorical", name="b")
            if second_writer == "legend()":
                canvas.legend("b", title="B")
            winner = canvas.ax.get_legend()
            assert winner is not None, "no legend is on the axes to survive anything"
            assert canvas._renderer.drawn["a"].guides == (), (
                f"with the slot written by {second_writer}, the keyed layer still holds a legend "
                "matplotlib has already dropped"
            )
            canvas.remove_layer("a")
            assert canvas.ax.get_legend() is winner, (
                "removing the displaced layer took the surviving layer's key off the axes"
            )

    def test_removing_a_record_that_still_holds_an_orphan_leaves_the_slot_alone(
        self, two_fills
    ):
        """Even with a stale legend in the record, removing its layer must not clear the axes' slot.

        Args:
            two_fills: A map with two categorical fills.

        Test scenario:
            The other half of the fix, and the half that holds for a writer the reconcile sweep has not
            been taught about. `Renderer._displaced` keeps the *record* clear of orphans; this pins that
            the damage is impossible even when one slips through, because `_detach_guide` refuses to call
            `remove()` on a legend the axes does not hold. Without it the guarantee is only as good as the
            list of places the sweep is called from — which is exactly how review H1 happened, the sweep
            having been reached from one of the slot's two writers.

            The orphan is put back by hand, on the renderer's own record, because with the sweep in place
            no supported call can produce one any more. matplotlib's removal hook is what makes this
            destructive: `Axes._remove_legend` sets `legend_` to `None` whichever legend is sitting there,
            and a *second* `remove()` of an artist does not raise, so nothing else would notice.
        """
        from dataclasses import replace as with_fields

        two_fills.legend("a", title="A")
        stale = two_fills._renderer.drawn["a"].guides[0]
        two_fills.legend("b", title="B")
        winner = two_fills.ax.get_legend()
        assert winner is not stale, "the second key did not displace the first"
        two_fills._renderer._drawn["a"] = with_fields(
            two_fills._renderer.drawn["a"], guides=(stale,)
        )
        two_fills.remove_layer("a")
        assert two_fills.ax.get_legend() is winner, (
            "removing a layer whose record held a displaced legend called remove() on it, and matplotlib's "
            "hook took the surviving layer's key off the figure with it"
        )

    def test_keying_the_first_layer_again_takes_the_second_s_key_off(self, two_fills):
        """The rule is symmetric: whoever asks last owns the slot, and the other stops claiming it.

        Args:
            two_fills: A map with two categorical fills.

        Test scenario:
            Written as the reverse of the first test so the rule cannot be satisfied by "the lower layer
            always loses" — the loser is whichever one did not ask last, not whichever is further down.
        """
        two_fills.legend("a", title="A")
        two_fills.legend("b", title="B")
        two_fills.legend("a", title="A again")
        assert two_fills.get_layer("b").symbology.guide().show is False, (
            "the layer whose key was displaced still describes one"
        )
        assert two_fills._renderer.drawn["b"].guides == ()
        assert two_fills.get_layer("a").symbology.guide().show is True
        assert two_fills._renderer.drawn["a"].guides[0] is two_fills.ax.get_legend()

    def test_a_figure_arriving_with_two_shown_legends_only_records_the_drawn_one(
        self, two_fills
    ):
        """Two shown legend guides read back from elsewhere leave one drawn key, recorded once.

        Args:
            two_fills: A map with two categorical fills.

        Test scenario:
            `Scene.legend` cannot produce this state any more, but a figure built where the one-legend rule
            does not exist can: `from_dict` hands both layers a shown guide and `Renderer.draw_layer` draws
            each in turn. This drives the renderer directly, which is the path that rebuild takes, so the
            picture-side record is kept honest even where no scene call is involved.
        """
        from dataclasses import replace as with_fields

        from digitalearth.base.spec import Guide
        from digitalearth.static.guides import GUIDE_KIND_KEY

        for layer_id in ("a", "b"):
            layer = two_fills.get_layer(layer_id)
            props = dict(layer.symbology.props)
            props[GUIDE_KIND_KEY] = "legend"
            two_fills._layer_tree = two_fills._layer_tree.replace(
                with_fields(
                    layer,
                    symbology=with_fields(layer.symbology, props=props).with_guide(
                        Guide(show=True, title=layer_id)
                    ),
                )
            )
        for layer_id in ("a", "b"):
            two_fills._renderer.draw_guide(two_fills._layer_tree.get(layer_id))
        assert two_fills._renderer.drawn["a"].guides == (), (
            "the layer whose legend the axes displaced still holds it"
        )
        assert two_fills._renderer.drawn["b"].guides[0] is two_fills.ax.get_legend()

    @staticmethod
    def _both_keyed(canvas):
        """Return the figure `canvas` describes, with a shown swatch guide on **both** its fills.

        Args:
            canvas: The map to read the figure off.

        Returns:
            The `FigureSpec` a figure built where the one-legend rule does not exist would arrive as.
        """
        from dataclasses import replace as with_fields

        from digitalearth.base.spec import Guide
        from digitalearth.static.guides import GUIDE_KIND_KEY

        arriving = canvas.figure_spec
        for layer_id in ("a", "b"):
            layer = arriving.layers.get(layer_id)
            props = dict(layer.symbology.props)
            props[GUIDE_KIND_KEY] = "legend"
            symbology = with_fields(layer.symbology, props=props).with_guide(
                Guide(show=True, title=layer_id)
            )
            arriving = with_fields(
                arriving,
                layers=arriving.layers.replace(with_fields(layer, symbology=symbology)),
            )
        return arriving

    @staticmethod
    def _described_as_shown(canvas):
        """Return the ids of the layers whose description claims a drawn swatch key, bottom first.

        Args:
            canvas: The map to read.

        Returns:
            One id per layer carrying a shown legend guide.
        """
        from digitalearth.static.guides import guide_kind

        shown = []
        for layer_id in canvas.layer_ids:
            layer = canvas.get_layer(layer_id)
            guide = layer.symbology.guide()
            if guide is not None and guide.show and guide_kind(layer) == "legend":
                shown.append(layer_id)
        return shown

    def test_a_figure_read_back_settles_on_one_described_key(self, two_fills):
        """A description arriving with two shown swatch keys is installed claiming one.

        Args:
            two_fills: A map with two categorical fills.

        Test scenario:
            The half of the one-legend rule that `Scene.legend` cannot reach. A figure built where the rule
            does not exist hands both layers a shown guide; `Renderer._displaced` keeps the *drawn* record
            honest, but the descriptions both went on saying `show=True` — and the description is what
            travels to another tier and what a reader trusts when the figure is not in front of them, so it
            is the worse half to leave wrong.

            The layer that keeps it is the **topmost in draw order**, which is the same layer `legend()`
            with no id would key, so the rule a reader predicts from the API is the rule a rebuild applies.
        """
        two_fills._change(self._both_keyed(two_fills))
        assert two_fills.layer_ids == ["a", "b"]
        assert self._described_as_shown(two_fills) == ["b"], (
            "the installed description claims a swatch key on a layer the axes is not showing one for"
        )
        assert two_fills._renderer.drawn["b"].guides[0] is two_fills.ax.get_legend(), (
            "the layer the description keeps is not the one the axes draws"
        )
        assert two_fills._renderer.drawn["a"].guides == ()
        assert two_fills.get_layer("a").symbology.guide().title == "a", (
            "the displaced layer lost the title it arrived with, so its key cannot come back as it was"
        )

    def test_the_displaced_layer_can_still_take_the_key_back(self, two_fills):
        """Settling a read-back figure is not one-way: `legend()` on the loser still works.

        Args:
            two_fills: A map with two categorical fills.

        Test scenario:
            Switching the guide *off* rather than taking it off is what makes this possible, and it has to
            keep holding for the rebuild path as well as the API one — a figure read back that quietly made
            one of its layers unkeyable, or that dropped what the loser's key was called, would be a worse
            answer than the stale record it replaced.

            The assertion that carries this is `b`'s *own description surviving the handover*: it is
            switched off, keeps the title it arrived with, and is therefore a key that can come back as it
            was rather than one that has to be described again. Clearing the loser's guide instead of
            switching it off passes every other line here and fails that one.
        """
        two_fills._change(self._both_keyed(two_fills))
        two_fills.legend("a", title="A again")
        assert self._described_as_shown(two_fills) == ["a"], (
            "the layer that lost the slot on the rebuild could not take it back"
        )
        assert two_fills._renderer.drawn["a"].guides[0] is two_fills.ax.get_legend()
        assert two_fills._renderer.drawn["b"].guides == ()
        handed_over = two_fills.get_layer("b").symbology.guide()
        assert handed_over is not None, (
            "the layer that gave the slot up lost its guide rather than having it switched off, so its "
            "key cannot come back as it was"
        )
        assert (handed_over.show, handed_over.title) == (False, "b"), (
            f"the handed-over key should be switched off and still called 'b'; it is {handed_over}"
        )

    def test_taking_the_key_back_re_derives_what_it_is_called(self, two_fills):
        """A bare `legend(layer_id)` brings the key back, but not the title and rows it was given.

        Args:
            two_fills: A map with two categorical fills.

        Test scenario:
            The record keeps the displaced layer's title and row labels — the test above pins that — and
            both `Scene.legend`'s Note and `_displace_other_legends`' docstring read that as
            "`legend(layer_id)` brings that key back, title and labels and all". It does not: the call
            resolves `title` through `_title_for`, which answers the layer's own units for a `None`, and
            passes `labels=None`, which *clears* a recorded override so one call's rename does not outlive
            it. So what comes back is a key with a re-derived title and derived rows (review L2).

            Pinned rather than only reworded, because it is the kind of claim that goes stale silently:
            either the prose or this test has to change if the call ever learns to reuse a switched-off
            guide's own description.
        """
        from digitalearth.static.guides import GUIDE_LABELS_KEY

        two_fills.legend("a", title="Alpha", labels=["x", "y"])
        two_fills.legend("b", title="Beta")
        switched_off = two_fills.get_layer("a")
        assert (
            switched_off.symbology.guide().show,
            switched_off.symbology.guide().title,
        ) == (
            False,
            "Alpha",
        ), (
            "the record did not keep the displaced layer's own title while switching it off"
        )
        assert switched_off.symbology.props.get(GUIDE_LABELS_KEY) == ("x", "y"), (
            "the record did not keep the displaced layer's own row labels"
        )
        two_fills.legend("a")
        back = two_fills.get_layer("a")
        assert back.symbology.guide().show is True, "the key did not come back at all"
        assert back.symbology.guide().title is None, (
            "a bare legend(layer_id) is documented as re-deriving the title; it kept "
            f"{back.symbology.guide().title!r}, so the prose that says otherwise is now the right one"
        )
        assert GUIDE_LABELS_KEY not in back.symbology.props, (
            "a bare legend(layer_id) is documented as clearing the recorded row labels; it kept "
            f"{back.symbology.props.get(GUIDE_LABELS_KEY)!r}"
        )


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


def test_colored_by_carries_every_other_field_of_the_drawing():
    """`DrawnLayer.colored_by` changes the colour field and nothing else, whatever the fields are.

    Test scenario:
        Written against `dataclasses.fields` rather than against a list of names, because the hazard is a
        field added to `DrawnLayer` **later** (review L2): a hand-written copy is correct on the day it is
        written and silently lossy on the day the dataclass grows, and a test that names the five fields
        would grow the same blind spot at the same moment. This one covers a sixth field the day it is
        added, without being touched.

        Proved not vacuous by adding a sixth field and running it against both implementations: the
        hand-written copy drops it and this test reddens; `replace` carries it and it passes.
    """
    from dataclasses import fields

    carried = {
        held.name: object() for held in fields(DrawnLayer) if held.name != "color_field"
    }
    again = DrawnLayer(**carried).colored_by("elev")
    assert again.color_field == "elev"
    for name, mark in carried.items():
        assert getattr(again, name) is mark, (
            f"colored_by() dropped {name!r}; it copies the drawing field by field, so every field "
            f"`DrawnLayer` gains has to be added there by hand"
        )


def test_a_key_is_not_in_the_layer_s_artists(keyed):
    """The drawn key is held beside the layer's artists, never among them.

    Args:
        keyed: A map with a raster and a text label.

    Test scenario:
        `artists` is what `Renderer._repaint` arranges in draw order and what `_rank_zorders` writes z-orders
        into; a colorbar lives on its own axes where neither applies, so putting it there would feed the
        z-order ranking an artist that is not painted on the data axes. What the two share is the layer's
        lifetime, not its stacking.

        The membership of `artists` is the whole of what can be asserted here, and it is asserted twice —
        by what the tuple *is* and by what it does not hold. A fourth line compared
        `drawn.artist.get_zorder()` with `keyed.ax.images[0].get_zorder()` and sold itself as the guard on
        `_rank_zorders`: `drawn.artist` **is** `ax.images[0]`, so both sides read one object, and forcing
        every drawn artist to `zorder = -4321.0` — feeding the ranking nonsense, which is the defect the
        line named — left this test passing. Dealing the key into `DrawnLayer.artists` is the mutation that
        reddens what is left (review M8).
    """
    keyed.colorbar()
    drawn = keyed._renderer.drawn["acc"]
    assert drawn.artists == (drawn.artist,), drawn.artists
    assert drawn.guides != (), "the drawn key was not held beside the layer"
    assert drawn.guides[0] not in drawn.artists, (
        f"the key is among the layer's artists: {drawn.artists}"
    )
