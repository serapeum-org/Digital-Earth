"""What a web layer's id is, how long it lasts, and how it is numbered (review round 2).

The sibling of `tests/static/test_static_layer_identity.py`, for the three findings that reach this tier:

* **R2-M9** — `remove_layer` dropped the tree entry, the source, the legend and the slider frame, and never
  touched `_issued_ids`. So a removed name stayed reserved for the map's life: `remove_layer("wells")`
  followed by a new `wells` drew `wells-2` on a map with nothing called `wells`. The 3-D tier reads its ids
  off the live tree and answered `wells`. One rule now: **an id is reserved exactly as long as the layer is
  on the figure**, which is also what this tier's own `_forget_layer` rollback already did.
* **R2-H4** — this is the tier that *kept* a padded name, where the other three crashed on it, so a script
  moved between tiers got a different id either way.
* **R2-M10** — an unnamed layer was numbered within the *figure* rather than within its kind.

On this tier the id is not only an address: py-maplibregl's layer switcher captions each row with the layer
id itself, so a reserved-forever id is a caption nobody asked for.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from digitalearth.web import WebMap

pytest.importorskip("maplibre")

#: The name the probes ask for.
ASKED = "wells"

#: The column the fixture carries, so a classified builder has something to colour by.
COLUMN = "pop"


@pytest.fixture
def drawn():
    """Return an empty web map in EPSG:4326.

    Returns:
        A `WebMap` with no layers.
    """
    return WebMap()


def _points() -> gpd.GeoDataFrame:
    """Return two point features with a numeric column.

    Returns:
        A GeoDataFrame in EPSG:4326.
    """
    return gpd.GeoDataFrame(
        {COLUMN: [1.0, 9.0]},
        geometry=[Point(0.0, 0.0), Point(1.0, 1.0)],
        crs=4326,
    )


class TestHowLongAnIdIsReserved:
    """R2-M9: an id is reserved exactly as long as the layer is on the map, and no longer."""

    def test_a_removed_name_is_handed_straight_back(self, drawn):
        """Remove `wells`, ask for `wells`, get `wells` — as the 3-D tier already answered.

        Args:
            drawn: The map under test.

        Test scenario:
            Measured at the reviewed HEAD, the same three calls gave `['wells-2']` here and `['wells']` on
            the 3-D tier. Nothing called `wells` was on the map in either case, and on this tier the id is
            the caption the layer switcher shows.
        """
        drawn.points(_points(), name=ASKED)
        drawn.remove_layer(ASKED)
        drawn.points(_points(), name=ASKED)
        assert drawn.layer_ids == [ASKED], drawn.layer_ids

    def test_the_ids_its_drawer_derived_go_back_too(self, drawn):
        """A graticule reserves `grid` **and** `grid-label`; removing it must release both.

        Args:
            drawn: The map under test.

        Test scenario:
            `DERIVED_SUFFIXES` gives `graticule` a second MapLibre layer named after the first, and
            `_layer_id` passes over a candidate whose derived id is taken. Releasing only the layer's own
            id would leave `grid-label` reserved, so the re-added `grid` would be passed over for
            `grid-2` — a half-freed id, which is worse than either whole answer.
        """
        drawn.graticule(spacing=30.0, name="grid")
        drawn.remove_layer("grid")
        drawn.graticule(spacing=30.0, name="grid")
        assert drawn.layer_ids == ["grid"], sorted(drawn._issued_ids)

    def test_an_id_still_reserved_while_the_layer_is_there(self, drawn):
        """The other direction: releasing on removal must not release on *anything*.

        Args:
            drawn: The map under test.

        Test scenario:
            The rule is "as long as the layer is on the figure". A second `wells` while the first is still
            drawn is the case that rule has always answered, and a fix that freed too eagerly would make
            the two layers share an id — where MapLibre drops the second with only a console error.
        """
        drawn.points(_points(), name=ASKED)
        drawn.points(_points(), name=ASKED)
        assert drawn.layer_ids == [ASKED, "wells-2"], drawn.layer_ids

    def test_a_removed_id_addresses_nothing_until_it_is_re_used(self, drawn):
        """What makes releasing safe: a stale id is refused by name rather than reaching a new layer.

        Args:
            drawn: The map under test.

        Test scenario:
            Reserving an id forever was defended as protecting a caller who still holds the string. It does
            not: `get_layer` raises for an id the tree no longer holds either way, so the caller learns
            the layer is gone at the first call they make — which is the guarantee the shared rule rests on.
        """
        drawn.points(_points(), name=ASKED)
        drawn.remove_layer(ASKED)
        with pytest.raises(KeyError, match=r"no layer 'wells' on this map"):
            drawn.get_layer(ASKED)


class TestANamePaddedWithWhitespace:
    """R2-H4: the tier that kept the padding now strips it, like the three that crashed on it."""

    @pytest.mark.parametrize(
        "padded", ["wells ", " wells", "  wells  ", "wells\t", "\nwells"]
    )
    def test_the_layer_is_filed_under_the_name_without_its_padding(self, drawn, padded):
        """Every padding spelling reaches one id, here as everywhere else.

        Args:
            drawn: The map under test.
            padded: The name as the caller mistyped it.

        Test scenario:
            Measured at the reviewed HEAD this tier answered `['wells ']`, `['\\nwells']` and even
            `['  ']` — ids that differ from `wells` only by characters nothing renders, so `layer_ids`, a
            traceback and the switcher caption all read the same for two different layers.
        """
        drawn.points(_points(), name=padded)
        assert drawn.layer_ids == [ASKED], (padded, drawn.layer_ids)

    def test_two_spellings_of_one_name_are_one_name(self, drawn):
        """`"wells "` and `"wells"` collide, so the second is suffixed rather than shadowing the first.

        Args:
            drawn: The map under test.
        """
        drawn.points(_points(), name="wells ")
        drawn.points(_points(), name=ASKED)
        assert drawn.layer_ids == [ASKED, "wells-2"], drawn.layer_ids

    @pytest.mark.parametrize("blank", ["   ", "\t", ""])
    def test_a_blank_name_is_no_name(self, drawn, blank):
        """A name with nothing in it means the layer was not named, as `name=None` already did.

        Args:
            drawn: The map under test.
            blank: The blank name.

        Test scenario:
            `name="  "` drew a layer whose switcher row is captioned with two spaces — a row a viewer
            cannot read and cannot guess at. The other tiers raised a `KeyError` on the same call.
        """
        drawn.points(_points(), name=blank)
        assert drawn.layer_ids == ["circle-1"], (blank, drawn.layer_ids)

    @pytest.mark.parametrize("wrong", [123, 0, True, 2.5])
    def test_a_name_that_is_not_a_string_says_so_by_type(self, drawn, wrong):
        """Refused at the mint and named after the keyword, not by `LayerSpec` about an "id".

        Args:
            drawn: The map under test.
            wrong: The non-string the caller passed.
        """
        features = _points()
        with pytest.raises(TypeError, match=r"a layer name must be a string"):
            drawn.points(features, name=wrong)


class TestAnUnnamedLayerCountsItsOwnKind:
    """R2-M10: the number in a generated id counts the kind, on this tier as on the 3-D one."""

    def test_two_kinds_interleaved_each_start_at_one(self, drawn):
        """The change: `circle-1, circle-2, text-1, text-2`, not `circle-1, text-2, circle-3, text-4`.

        Args:
            drawn: The map under test.

        Test scenario:
            Interleaving the two kinds is the whole point — four layers of one kind number the same either
            way. The prefix is this tier's MapLibre type rather than the engine-neutral kind, which
            `tests/base/test_map_conformance.py` records as deliberately unsettled; the *number* is what
            R2-M10 is about, and it is settled here.
        """
        drawn.points(_points())
        drawn.text(4.9, 52.4, "a")
        drawn.points(_points())
        drawn.text(2.35, 48.86, "b")
        assert drawn.layer_ids == [
            "circle-1",
            "circle-2",
            "text-1",
            "text-2",
        ], drawn.layer_ids

    def test_the_generator_still_steps_over_an_id_a_caller_holds(self, drawn):
        """Per-kind counting must not cost the skip: two MapLibre layers under one id lose one.

        Args:
            drawn: The map under test.

        Test scenario:
            MapLibre's `addLayer` drops a second layer sharing an id with only a console error, so the
            generator consulting the issued set rather than trusting its own count is load bearing here in
            a way it is not on the tiers that keep their own artists.
        """
        drawn.points(_points(), name="circle-1")
        drawn.points(_points())
        assert drawn.layer_ids == ["circle-1", "circle-2"], drawn.layer_ids
