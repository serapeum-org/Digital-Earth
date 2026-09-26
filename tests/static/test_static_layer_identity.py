"""What a static layer's id is, how long it lasts, and what a caller's `name=` may be (review round 2).

`tests/static/test_layer_naming.py` covers the id a caller **asks** for. These cover the four things round 2
found underneath it, all of them about identity rather than about naming:

* **R2-H4** — `name="roads "` raised a `KeyError` naming neither the layer nor the name. The mint kept the
  padding, the id became the key the layer's source was registered under, and
  `DataRef.__post_init__` stripped the uri — so the reference no longer reached the object registered one
  line earlier. Only a layer carrying a source failed, which made it intermittent across builders.
* **R2-M10 / R2-L2** — an unnamed layer was numbered within the *figure* (`points-1, text-2, points-3`)
  while this tier's own `layer_ids` docstring said it was numbered within its kind.
* **R2-M11** — a description that refused left the minted id reserved on a scene that draws no such layer.
* **R2-L11** — `_forget_layer`'s docstring promised to ignore a layer already gone, and `LayerTree.remove`
  raised `KeyError` on one.

The cross-tier half — that the other three tiers answer identically — is in the sibling
`test_*_layer_identity.py` files under `tests/interactive`, `tests/web` and `tests/three_d`; the shared rule
itself is in `tests/base/test_layer.py`.
"""

import matplotlib
import pytest

matplotlib.use("Agg")

import geopandas as gpd
from pyramids.feature import FeatureCollection
from shapely.geometry import Point

from digitalearth.static import Map

#: The name the probes ask for, padded in every spelling that used to crash.
ASKED = "wells"

#: The column the fixture carries, so a builder that colours by one has something to colour by.
COLUMN = "pop"


@pytest.fixture
def drawn():
    """Yield an empty map in EPSG:4326, closed on the way out.

    Yields:
        A `Map` whose display CRS matches the fixture below, so no builder reprojects.
    """
    built = Map(crs=4326)
    yield built
    built.close()


def _points() -> FeatureCollection:
    """Return two point features with a numeric column.

    Returns:
        A pyramids `FeatureCollection` in EPSG:4326.
    """
    return FeatureCollection(
        gpd.GeoDataFrame(
            {COLUMN: [1.0, 9.0]},
            geometry=[Point(0.0, 0.0), Point(1.0, 1.0)],
            crs=4326,
        )
    )


class TestANamePaddedWithWhitespace:
    """R2-H4: the crash a one-keystroke mistake used to be, and the id it produces instead."""

    @pytest.mark.parametrize(
        "padded", ["wells ", " wells", "  wells  ", "wells\t", "\nwells"]
    )
    def test_the_layer_is_filed_under_the_name_without_its_padding(self, drawn, padded):
        """Every padding spelling reaches one id, and none of them raises.

        Args:
            drawn: The map under test.
            padded: The name as the caller mistyped it.

        Test scenario:
            Measured at the reviewed HEAD, `points(name="wells ")` raised
            `KeyError: "no object is registered as '<ns>:wells'..."` — the *stripped* key, against the
            padded one the registry held — while `name=" wells"` worked, because leading padding survives a
            `str.strip()` comparison on the way in. Both spellings are probed here for that reason.
        """
        drawn.points(_points(), name=padded)
        assert drawn.layer_ids == [ASKED], (padded, drawn.layer_ids)

    def test_the_source_the_layer_registered_still_resolves(self, drawn):
        """The half the id list cannot show: the reference reaches the data it was registered for.

        Args:
            drawn: The map under test.

        Test scenario:
            The id list would look right even if the uri and the registry key still disagreed, because the
            id is minted before anything is registered. Opening the reference is what asks the registry,
            which is where the `KeyError` came from.
        """
        drawn.points(_points(), name="wells ")
        reopened = drawn.figure_spec.sources[ASKED].open()
        assert isinstance(reopened, FeatureCollection), type(reopened).__name__

    def test_two_spellings_of_one_name_are_one_name(self, drawn):
        """`"wells "` and `"wells"` collide, so the second is suffixed rather than shadowing the first.

        Args:
            drawn: The map under test.

        Test scenario:
            Two ids that differ only by padding are two layers nobody can tell apart in `layer_ids`, in a
            traceback or in a layer switcher. Colliding them is what makes the suffix rule reach the case.
        """
        drawn.points(_points(), name="wells ")
        drawn.points(_points(), name=ASKED)
        assert drawn.layer_ids == [ASKED, "wells-2"], drawn.layer_ids

    @pytest.mark.parametrize("blank", ["   ", "\t", ""])
    def test_a_blank_name_is_no_name_rather_than_a_crash(self, drawn, blank):
        """A name with nothing in it means the layer was not named, as `name=None` already did.

        Args:
            drawn: The map under test.
            blank: The blank name.

        Test scenario:
            `name="  "` raised `KeyError: "no object is registered as '<ns>:'"` — the id had been stripped
            to nothing — while `name=""` generated an id. Two spellings of "no name", two outcomes.
        """
        drawn.points(_points(), name=blank)
        assert drawn.layer_ids == ["points-1"], (blank, drawn.layer_ids)

    @pytest.mark.parametrize("wrong", [123, 0, True, 2.5])
    def test_a_name_that_is_not_a_string_says_so_by_type(self, drawn, wrong):
        """Refused at the mint and named after the keyword, not by `LayerSpec` about an "id".

        Args:
            drawn: The map under test.
            wrong: The non-string the caller passed.

        Test scenario:
            `name=123` raised `ValueError: LayerSpec needs an id that is a non-empty string; got 123`,
            which names a concept the caller never wrote — and `name=0` raised nothing at all, silently
            drawing `points-1`.
        """
        features = _points()
        with pytest.raises(TypeError, match=r"a layer name must be a string"):
            drawn.points(features, name=wrong)

    def test_a_long_name_survives_whole(self, drawn):
        """No length cap: truncating an id silently would make two named layers one.

        Args:
            drawn: The map under test.
        """
        long_name = "w" * 300
        drawn.points(_points(), name=long_name)
        assert drawn.layer_ids == [long_name], len(drawn.layer_ids[0])


class TestAnUnnamedLayerCountsItsOwnKind:
    """R2-M10 / R2-L2: the number in a generated id counts the kind, on this tier as on the 3-D one."""

    def test_two_kinds_interleaved_each_start_at_one(self, drawn):
        """The change: `points-1, points-2, text-1, text-2`, not `points-1, points-3, text-2, text-4`.

        Args:
            drawn: The map under test.

        Test scenario:
            Interleaving the two kinds is the whole point — four layers of one kind number the same either
            way. Measured at the reviewed HEAD this call gave `['points-1', 'points-3', 'text-2',
            'text-4']`, which reads as if two layers had gone missing, and disagreed with the 3-D tier and
            with `layer_ids`' own docstring. The ids are listed in draw order, so the text layers follow.
        """
        drawn.points(_points())
        drawn.text(0.5, 0.5, "a")
        drawn.points(_points())
        drawn.text(0.6, 0.6, "b")
        assert drawn.layer_ids == ["points-1", "points-2", "text-1", "text-2"], (
            drawn.layer_ids
        )

    def test_the_generator_still_steps_over_an_id_a_caller_holds(self, drawn):
        """Per-kind counting must not cost the skip: two layers under one id is the real damage.

        Args:
            drawn: The map under test.

        Test scenario:
            A caller can name a layer `text-1` before any unnamed `text` exists. The counter is per kind
            now, so the check that it still consults the issued set rather than trusting its own count has
            to be made again.
        """
        drawn.points(_points())
        drawn.text(0.5, 0.5, "a", name="text-1")
        drawn.text(0.6, 0.6, "b")
        assert drawn.layer_ids == ["points-1", "text-1", "text-2"], drawn.layer_ids


class TestADescriptionThatRefuses:
    """R2-M11 / R2-L11: a layer that was minted but never described takes its id back with it."""

    def test_the_minted_id_is_given_back(self, drawn, monkeypatch):
        """Nothing is left reserved when recording the layer raises.

        Args:
            drawn: The map under test.
            monkeypatch: Makes `_index_layer` refuse — the step that runs `LayerSpec` validation,
                `DataRef.of` and `LayerTree.add`, each of which refuses for its own reasons.

        Test scenario:
            The mint sat outside every `try` on this tier, so a refusal left the id in `_issued_ids` with
            the scene describing no such layer; the interactive tier already rolled its own back. The
            refusal is injected rather than provoked because the one reachable refusal the review measured
            — `name=123` — was closed by the R2-H4 fix above and now happens *before* the mint; the
            finding is about the window between the mint and the description, not about any one way in.
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

        features = _points()
        monkeypatch.setattr(Map, "_index_layer", refuse)
        with pytest.raises(ValueError, match="refused"):
            drawn.points(features, name=ASKED)
        assert drawn._issued_ids == set(), drawn._issued_ids

    def test_the_next_layer_gets_the_name_unsuffixed(self, drawn, monkeypatch):
        """The cost of stranding it, read the way a caller would notice: a name that is suddenly taken.

        Args:
            drawn: The map under test.
            monkeypatch: Makes the first attempt refuse, and only the first.

        Test scenario:
            A stranded id is invisible until the caller retries. The second call is the one that used to
            come back as `wells-2` on a map with no `wells` on it.
        """
        calls = {"n": 0}
        indexed = Map._index_layer

        def refuse_once(self, *args, **kwargs):
            """Refuse the first layer and describe every one after it.

            Args:
                self: The scene describing the layer.
                *args: Passed through once the first call is past.
                **kwargs: Passed through once the first call is past.

            Raises:
                ValueError: on the first call only.
            """
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("refused")
            indexed(self, *args, **kwargs)

        refused = _points()
        monkeypatch.setattr(Map, "_index_layer", refuse_once)
        with pytest.raises(ValueError, match="refused"):
            drawn.points(refused, name=ASKED)
        drawn.points(_points(), name=ASKED)
        assert drawn.layer_ids == [ASKED], drawn.layer_ids

    def test_forgetting_a_layer_that_was_never_added_is_ignored(self, drawn):
        """R2-L11: the docstring promised this and `LayerTree.remove` raised `KeyError` instead.

        Args:
            drawn: The map under test.

        Test scenario:
            It matters now rather than later: the rollback above runs for a refusal that may have come from
            `LayerTree.add` itself, so an unguarded `remove` would raise over the exception it was
            unwinding and the caller would see the wrong one.
        """
        assert drawn._forget_layer("never-added") is None
