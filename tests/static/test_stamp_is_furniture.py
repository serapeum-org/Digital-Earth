"""`Scene.stamp` is figure furniture, not a layer — and the carve-out has to say so (#391 item 2).

Every public builder on `Map` chains: it is annotated `-> Self`, registers a layer, and the artist it drew is
reached through `Scene.artist`. `stamp` does none of the three. It bakes a logo/watermark onto the figure through
cleopatra's `WatermarkMixin.stamp_mark`, hands the frameless inset `Axes` straight back, and registers nothing —
so the mark is in no layer list, in no figure description and unreachable through `artist()`.

That is the right shape for it, exactly as it is for `quiverkey` and `save`: the call is *about* the object it
returns (you reposition or restyle the mark), and a watermark fixed to the figure frame is not data anyone would
switch off, re-colour or key in a legend. What #391 found missing is that the decision was nowhere written down
at `stamp`'s own site — it read as a builder someone had forgotten to convert. These tests pin both halves: the
behaviour, and the docstring entry that makes it a decision rather than an oversight, mirroring
`test_quiverkey_is_furniture.py`.
"""

import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from pyramids.dataset import Dataset, GeoReference

from digitalearth.static import Map, Scene


@pytest.fixture
def mark():
    """A small opaque RGBA watermark, as the `stamp` doctest uses.

    Returns:
        numpy.ndarray: a ``(40, 80, 4)`` ``uint8`` mark, fully opaque white.
    """
    logo = np.zeros((40, 80, 4), dtype=np.uint8)
    logo[..., :3] = 255
    logo[..., 3] = 255
    return logo


@pytest.fixture
def dem():
    """A small synthetic single-band raster to carry one field layer.

    Returns:
        Dataset: a 6x8 increasing-value raster in EPSG:4326.
    """
    arr = np.arange(48, dtype="float32").reshape(6, 8)
    geo = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return Dataset.from_array(arr=arr, geo_ref=GeoReference(geo=geo, epsg=4326))


@pytest.fixture
def stamped(dem, mark):
    """A map carrying one field layer, with a watermark stamped over it.

    Args:
        dem: The single-band raster.
        mark: The watermark image.

    Yields:
        tuple[Map, list[str], Any]: the map, its layer ids before stamping, and whatever `stamp` handed back.
    """
    scene = Map(crs=4326)
    scene.field(dem)
    before = list(scene.layer_ids)
    drawn = scene.stamp(mark, frac=0.2, shadow=False)
    yield scene, before, drawn
    scene.close()


class TestTheCarveOutIsRecorded:
    """A carve-out nobody wrote down is indistinguishable from a method someone forgot."""

    def test_stamp_says_it_registers_no_layer(self):
        """The method's own docstring carries the decision, where a caller reads it."""
        assert "no layer" in Scene.stamp.__doc__, (
            "stamp's docstring must say it registers no layer"
        )

    def test_the_census_discussion_names_stamp(self):
        """`quiverkey`'s Returns census names `stamp` as furniture sitting beside the builder carve-outs."""
        assert "stamp" in Map.quiverkey.__doc__, (
            "the carve-out census in quiverkey's docstring must name stamp"
        )


class TestItIsFurnitureRatherThanALayer:
    """The behaviour the docstring describes, measured rather than asserted from the annotation."""

    def test_it_hands_back_the_mark_axes(self, stamped):
        """The return *is* the point of the call, as with `save` and `quiverkey`.

        Args:
            stamped: The map, its prior layer ids, and the returned object.
        """
        _, _, drawn = stamped
        assert isinstance(drawn, Axes), (
            f"expected the mark's inset Axes, got {type(drawn)}"
        )

    def test_it_does_not_chain(self, stamped):
        """So it cannot be mistaken for a builder: what comes back is not the scene.

        Args:
            stamped: The map, its prior layer ids, and the returned object.
        """
        _, _, drawn = stamped
        assert not isinstance(drawn, Scene), (
            f"stamp must not look chainable; got {type(drawn)}"
        )

    def test_it_registers_no_layer(self, stamped):
        """The figure still holds exactly the one field layer the data drew.

        Args:
            stamped: The map, its prior layer ids, and the returned object.
        """
        scene, before, _ = stamped
        assert scene.layer_ids == before, (
            f"the mark must register no layer of its own; got {scene.layer_ids}"
        )

    def test_the_mark_is_on_the_figure(self, stamped):
        """It draws — the point of the carve-out is where the artist goes, not that there is none.

        Args:
            stamped: The map, its prior layer ids, and the returned object.
        """
        scene, _, drawn = stamped
        assert drawn in scene.fig.axes, "the mark's inset axes must be on the figure"

    def test_it_is_not_reachable_through_artist(self, stamped):
        """Registering no layer means `artist()` cannot name it, which is what the carve-out records.

        Args:
            stamped: The map, its prior layer ids, and the returned object.
        """
        scene, _, _ = stamped
        with pytest.raises(KeyError, match="no layer 'stamp'"):
            scene.artist("stamp")

    def test_the_layer_beneath_is_still_reachable(self, stamped):
        """The field layer underneath is an ordinary layer, so its artist comes back by name.

        Args:
            stamped: The map, its prior layer ids, and the returned object.
        """
        scene, before, _ = stamped
        assert isinstance(scene.artist(before[0]), Artist), (
            f"the field layer's own artist must come back; got {type(scene.artist(before[0]))}"
        )
