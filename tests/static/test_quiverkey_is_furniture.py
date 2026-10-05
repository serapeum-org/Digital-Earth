"""`quiverkey` is furniture, not a layer — and the carve-out census has to say so (round 2's N2).

Every public builder on `Map` chains: it is annotated `-> Self`, registers a layer, and the artist it drew is
reached through `Scene.artist`. `quiverkey` does none of the three. It puts a labelled reference arrow on the
axes through matplotlib's `Axes.quiverkey`, hands the `QuiverKey` straight back, and registers nothing — so
the key is in no layer list, in no figure description and unreachable through `artist()`.

That is the right shape for it: the call is *about* the object it returns (you reposition it, restyle it, or
hand it to a caller that does), exactly as `save` and `stock_img` are, and a reference arrow describing the
layer beside it is not itself data anyone would switch off or re-colour. What was missing is that the decision
was nowhere written down — `quiverkey` appeared in no census of the `-> Any` carve-outs, so it read as a
builder someone had forgotten to convert. These tests pin both halves: the behaviour, and the census entry
that makes it a decision rather than an oversight.
"""

import numpy as np
import pytest
from matplotlib.quiver import Quiver, QuiverKey
from pyramids.dataset import Dataset, GeoReference

from digitalearth.static import Map

#: The id the one `quiver` layer below takes, generated from its kind.
QUIVER_LAYER = "vectors-1"


@pytest.fixture
def uv():
    """Two small synthetic u/v rasters on an increasing-y grid, as the quiver tests use.

    Returns:
        tuple[Dataset, Dataset]: the (u, v) single-band datasets, in EPSG:4326.
    """
    ny, nx = 6, 8
    u = np.ones((ny, nx), dtype="float32")
    v = np.linspace(-1.0, 1.0, ny, dtype="float32")[:, None] * np.ones(
        (1, nx), "float32"
    )
    geo = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return (
        Dataset.from_array(arr=u, geo_ref=GeoReference(geo=geo, epsg=4326)),
        Dataset.from_array(arr=v, geo_ref=GeoReference(geo=geo, epsg=4326)),
    )


@pytest.fixture
def keyed(uv):
    """A map carrying one quiver layer and the reference key drawn over it.

    Args:
        uv: The (u, v) pair.

    Yields:
        tuple[Map, Any]: the map, and whatever `quiverkey` handed back.
    """
    u_ds, v_ds = uv
    scene = Map(crs=4326)
    scene.quiver(u_ds, v_ds)
    key = scene.quiverkey(1.0, "1 m/s")
    yield scene, key
    scene.close()


class TestTheCarveOutIsRecorded:
    """A carve-out nobody wrote down is indistinguishable from a method someone forgot."""

    def test_the_census_names_quiverkey(self):
        """`field`'s Returns note is the canonical census of the `-> Any` exceptions.

        Test scenario:
            It named `stock_img` as "the one deliberate exception" while `quiverkey` had been annotated
            `-> Any` since before this branch — so the census was both incomplete and wrong about being
            complete.
        """
        assert "quiverkey" in Map.field.__doc__, (
            "the carve-out census in Map.field's Returns note must name quiverkey"
        )

    def test_the_census_still_names_stock_img(self):
        """And the entry it already had is still there, so the two read as one list."""
        assert "stock_img" in Map.field.__doc__, (
            "the census must keep naming stock_img beside quiverkey"
        )

    def test_quiverkey_says_it_registers_no_layer(self):
        """The method's own docstring carries the same decision, where a caller reads it."""
        assert "no layer" in Map.quiverkey.__doc__, (
            "quiverkey's docstring must say it registers no layer"
        )


class TestItIsFurnitureRatherThanALayer:
    """The behaviour the census describes, measured rather than asserted from the annotation."""

    def test_it_hands_back_matplotlibs_quiver_key(self, keyed):
        """The return *is* the point of the call, as with `save` and `stock_img`.

        Args:
            keyed: The map and the returned key.
        """
        _, key = keyed
        assert isinstance(key, QuiverKey), f"expected a QuiverKey, got {type(key)}"

    def test_it_does_not_chain(self, keyed):
        """So it cannot be mistaken for a builder: what comes back is not the map.

        Args:
            keyed: The map and the returned key.
        """
        _, key = keyed
        assert not isinstance(key, Map), (
            f"quiverkey must not look chainable; got {type(key)}"
        )

    def test_it_registers_no_layer(self, keyed):
        """The figure still holds exactly the one quiver layer the data drew.

        Args:
            keyed: The map and the returned key.
        """
        scene, _ = keyed
        assert scene.layer_ids == [QUIVER_LAYER], (
            f"the key must register no layer of its own; got {scene.layer_ids}"
        )

    def test_the_key_really_is_on_the_axes(self, keyed):
        """It draws — the point of the carve-out is where the artist goes, not that there is none.

        Args:
            keyed: The map and the returned key.
        """
        scene, key = keyed
        assert key in scene.ax.get_children(), (
            "the reference arrow must be on the shared axes"
        )

    def test_it_is_not_reachable_through_artist(self, keyed):
        """Registering no layer means `artist()` cannot name it, which is what the census records.

        Args:
            keyed: The map and the returned key.
        """
        scene, _ = keyed
        with pytest.raises(KeyError, match="no layer 'quiverkey'"):
            scene.artist("quiverkey")

    def test_the_layer_it_keys_is_still_reachable(self, keyed):
        """The quiver underneath is an ordinary layer, so its artist comes back by name.

        Args:
            keyed: The map and the returned key.
        """
        scene, _ = keyed
        assert isinstance(scene.artist(QUIVER_LAYER), Quiver), (
            f"the keyed layer's own artist must come back; got {type(scene.artist(QUIVER_LAYER))}"
        )
