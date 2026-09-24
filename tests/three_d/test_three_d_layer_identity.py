"""What a 3-D layer's id is, and how long it lasts (review round 2).

The sibling of `tests/static/test_static_layer_identity.py`, for the three findings that reach this tier:

* **R2-H4** — the review measured the padded-name crash on the static and interactive tiers; it is here too.
  `terrain(source, name="wells ")` raised a `KeyError` naming neither the layer nor the name, because the mint
  kept the padding and `DataRef` stripped the uri the source was registered under.
* **R2-L1** — this tier **silently ignored** a non-string name and generated `terrain-1`, where the other
  three refused it. Ignoring is the worse half: a caller who passes the wrong thing gets a layer they cannot
  address by the name they think it has.
* **R2-M9** — this tier is the one the shared id lifetime was taken *from*: `next_layer_id` reads the ids off
  the live tree, so an id is reserved exactly as long as the layer is on the scene. The pin below is what
  stops that being traded away when a ledger is added here later.

R2-M10 needs nothing here: this tier has always numbered a generated id within its kind, and is the tier the
other three were brought into line with.
"""

# The package imports have to follow pytest.importorskip("pyvista") — importing digitalearth.three_d
# without pyvista is the very thing the skip exists to avoid — so E402 is expected throughout.
# ruff: noqa: E402
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D

#: The name the probes ask for.
ASKED = "wells"


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test so no window opens in CI."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


@pytest.fixture
def scene():
    """Yield an off-screen scene, closed on the way out.

    Yields:
        An empty `Scene3D`.
    """
    built = Scene3D(off_screen=True)
    yield built
    built.close()


def _dem():
    """Return a small ramped DEM as the uniform source a builder takes.

    Returns:
        A `Source` over a 4x5 float grid.
    """
    return get_source(np.add.outer(np.arange(4.0), np.arange(5.0)))


class TestANamePaddedWithWhitespace:
    """R2-H4: the crash reaches this tier too, and is closed at the same shared mint."""

    @pytest.mark.parametrize(
        "padded", ["wells ", " wells", "  wells  ", "wells\t", "\nwells"]
    )
    def test_the_layer_is_filed_under_the_name_without_its_padding(self, scene, padded):
        """Every padding spelling reaches one id, and none of them raises.

        Args:
            scene: The scene under test.
            padded: The name as the caller mistyped it.

        Test scenario:
            Measured at the reviewed HEAD, `terrain(dem, name="wells ")` raised
            `KeyError: "no object is registered as '<ns>:wells'..."` — the *stripped* key, against the
            padded one the registry held. The review listed the static and interactive tiers; probing all
            four is what found this one.
        """
        scene.terrain(_dem(), name=padded)
        assert scene.layer_ids == [ASKED], (padded, scene.layer_ids)

    def test_two_spellings_of_one_name_are_one_name(self, scene):
        """`"wells "` and `"wells"` collide, so the second is suffixed rather than shadowing the first.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem(), name="wells ")
        scene.terrain(_dem(), name=ASKED)
        assert scene.layer_ids == [ASKED, "wells-2"], scene.layer_ids

    @pytest.mark.parametrize("blank", ["   ", "\t", ""])
    def test_a_blank_name_is_no_name(self, scene, blank):
        """A name with nothing in it means the layer was not named, as `name=None` already did.

        Args:
            scene: The scene under test.
            blank: The blank name.
        """
        scene.terrain(_dem(), name=blank)
        assert scene.layer_ids == ["terrain-1"], (blank, scene.layer_ids)

    @pytest.mark.parametrize("wrong", [123, 0, True, 2.5])
    def test_a_name_that_is_not_a_string_says_so_by_type(self, scene, wrong):
        """R2-L1: ignored here and refused on the other three tiers; refused on all four now.

        Args:
            scene: The scene under test.
            wrong: The non-string the caller passed.

        Test scenario:
            Measured at the reviewed HEAD, `terrain(dem, name=123)` drew a layer called `terrain-1` and
            said nothing — so the caller's `scene.set_visible(123, False)` would then fail somewhere else
            entirely, about an id nobody wrote.
        """
        with pytest.raises(TypeError, match=r"a layer name must be a string"):
            scene.terrain(_dem(), name=wrong)


class TestHowLongAnIdIsReserved:
    """R2-M9: the lifetime the other three tiers were brought to — and it has to stay this one."""

    def test_a_removed_name_is_handed_straight_back(self, scene):
        """Remove `wells`, ask for `wells`, get `wells`.

        Args:
            scene: The scene under test.

        Test scenario:
            This is the answer the web tier now gives too; it used to give `wells-2`. Pinning it here as
            well is what stops the pair being unified in the other direction by whichever tier moves next.
        """
        scene.terrain(_dem(), name=ASKED)
        scene.remove_layer(ASKED)
        scene.terrain(_dem(), name=ASKED)
        assert scene.layer_ids == [ASKED], scene.layer_ids

    def test_an_id_is_still_reserved_while_the_layer_is_there(self, scene):
        """The other direction: two layers asking for one name while both are drawn get two ids.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem(), name=ASKED)
        scene.terrain(_dem(), name=ASKED)
        assert scene.layer_ids == [ASKED, "wells-2"], scene.layer_ids

    def test_a_removed_id_addresses_nothing_until_it_is_re_used(self, scene):
        """What makes releasing safe: a stale id is refused by name rather than reaching a new layer.

        Args:
            scene: The scene under test.
        """
        scene.terrain(_dem(), name=ASKED)
        scene.remove_layer(ASKED)
        with pytest.raises(KeyError, match=r"no layer 'wells' in this scene"):
            scene.get_layer(ASKED)


class TestAnUnnamedLayerCountsItsOwnKind:
    """R2-M10, from the tier that was already right: a generated id counts its own kind."""

    def test_two_kinds_interleaved_each_start_at_one(self, scene):
        """`terrain-1, terrain-2, point_cloud-1` — unchanged here, and now shared.

        Args:
            scene: The scene under test.

        Test scenario:
            The other three tiers counted the figure and produced numbers with gaps in them. This pin is
            what makes that a shared rule rather than a coincidence of which tier was read last.
        """
        scene.terrain(_dem())
        scene.point_cloud(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        scene.terrain(_dem())
        assert scene.layer_ids == [
            "terrain-1",
            "point_cloud-1",
            "terrain-2",
        ], scene.layer_ids
