"""What an interactive layer's id is, and how it is numbered (review round 2).

The sibling of `tests/static/test_static_layer_identity.py`, for the two findings that reach this tier:

* **R2-H4** — `points(fc, name="wells ")` raised a `KeyError` naming neither the layer nor the name, because
  the mint kept the padding and `DataRef` stripped the uri it was registered under.
* **R2-M10 / R2-L2** — an unnamed layer was numbered within the *map* (`points-1, text-2, points-3`) while
  this tier's own `layer_ids` docstring said it was numbered within its kind.

R2-M11 is not here: this tier already rolled a failed description back, and is the tier the static one was
brought into line with. The pin that it still does is the last class below, so the two tiers cannot drift
apart again in the other direction.
"""

import pytest

from digitalearth.interactive import InteractiveMap

pytest.importorskip("holoviews")
pytest.importorskip("geoviews")

#: The name the probes ask for, padded in every spelling that used to crash.
ASKED = "wells"


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


class TestANamePaddedWithWhitespace:
    """R2-H4: the crash a one-keystroke mistake used to be, and the id it produces instead."""

    @pytest.mark.parametrize(
        "padded", ["wells ", " wells", "  wells  ", "wells\t", "\nwells"]
    )
    def test_the_layer_is_filed_under_the_name_without_its_padding(
        self, drawn, point_fc, padded
    ):
        """Every padding spelling reaches one id, and none of them raises.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
            padded: The name as the caller mistyped it.

        Test scenario:
            Measured at the reviewed HEAD, `points(name="wells ")` raised
            `KeyError: "no object is registered as '<ns>:wells'..."` — the *stripped* key, against the
            padded one the registry held — while `name=" wells"` worked, because leading padding survives a
            `str.strip()` comparison on the way in. Both spellings are probed for that reason.
        """
        drawn.points(point_fc, name=padded)
        assert drawn.layer_ids == [ASKED], (padded, drawn.layer_ids)

    def test_the_source_the_layer_registered_still_resolves(self, drawn, point_fc):
        """The half the id list cannot show: the reference reaches the data it was registered for.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
        """
        drawn.points(point_fc, name="wells ")
        reopened = drawn.figure_spec.sources[ASKED].open()
        assert reopened is not None, reopened

    def test_two_spellings_of_one_name_are_one_name(self, drawn, point_fc):
        """`"wells "` and `"wells"` collide, so the second is suffixed rather than shadowing the first.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
        """
        drawn.points(point_fc, name="wells ")
        drawn.points(point_fc, name=ASKED)
        assert drawn.layer_ids == [ASKED, "wells-2"], drawn.layer_ids

    @pytest.mark.parametrize("blank", ["   ", "\t", ""])
    def test_a_blank_name_is_no_name_rather_than_a_crash(self, drawn, point_fc, blank):
        """A name with nothing in it means the layer was not named, as `name=None` already did.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
            blank: The blank name.
        """
        drawn.points(point_fc, name=blank)
        assert drawn.layer_ids == ["points-1"], (blank, drawn.layer_ids)

    @pytest.mark.parametrize("wrong", [123, 0, True, 2.5])
    def test_a_name_that_is_not_a_string_says_so_by_type(self, drawn, point_fc, wrong):
        """Refused at the mint and named after the keyword, not by `LayerSpec` about an "id".

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
            wrong: The non-string the caller passed.
        """
        with pytest.raises(TypeError, match=r"a layer name must be a string"):
            drawn.points(point_fc, name=wrong)


class TestAnUnnamedLayerCountsItsOwnKind:
    """R2-M10 / R2-L2: the number in a generated id counts the kind, on this tier as on the 3-D one."""

    def test_two_kinds_interleaved_each_start_at_one(self, drawn, point_fc):
        """The change: `points-1, points-2, text-1, text-2`, not `points-1, points-3, text-2, text-4`.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.

        Test scenario:
            Interleaving the two kinds is the whole point — four layers of one kind number the same either
            way. The ids are listed in draw order, and a text label sits in the overlay band, so the two
            text layers follow the two point layers however the calls were ordered.
        """
        drawn.points(point_fc)
        drawn.text(0.5, 0.5, "a")
        drawn.points(point_fc)
        drawn.text(0.6, 0.6, "b")
        assert drawn.layer_ids == [
            "points-1",
            "points-2",
            "text-1",
            "text-2",
        ], drawn.layer_ids

    def test_the_generator_still_steps_over_an_id_a_caller_holds(self, drawn, point_fc):
        """Per-kind counting must not cost the skip: two layers under one id is the real damage.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
        """
        drawn.points(point_fc)
        drawn.text(0.5, 0.5, "a", name="text-1")
        drawn.text(0.6, 0.6, "b")
        assert drawn.layer_ids == ["points-1", "text-1", "text-2"], drawn.layer_ids


class TestADescriptionThatRefuses:
    """R2-M11, from the other side: this tier already rolls back, and must keep doing so."""

    def test_the_minted_id_is_given_back(self, drawn, point_fc, monkeypatch):
        """Nothing is left reserved when describing the layer raises.

        Args:
            drawn: The map under test.
            point_fc: The point fixture.
            monkeypatch: Makes `_index_layer` refuse — the step that runs `LayerSpec` validation,
                `DataRef.of` and `LayerTree.add`.

        Test scenario:
            The static tier stranded the id here and this one did not, which is the divergence R2-M11
            reports. Pinning the behaviour on *both* tiers is what stops the pair drifting apart again
            from whichever side moves next.
        """

        def refuse(*args, **kwargs):
            """Stand in for a description step that refuses.

            Args:
                *args: Ignored.
                **kwargs: Ignored.

            Raises:
                ValueError: always.
            """
            raise ValueError("refused")

        monkeypatch.setattr(InteractiveMap, "_index_layer", refuse)
        with pytest.raises(ValueError, match="refused"):
            drawn.points(point_fc, name=ASKED)
        assert drawn._issued_ids == set(), drawn._issued_ids
