"""``AnimationMixin._frame_values`` — the static tier's half of the shared stack scan (#174 / DE-9).

The colour-range rule moved to :mod:`digitalearth.base.clim`, which is pure arithmetic over arrays and knows
nothing about projections. What stayed here is the engine-specific half: warp each frame onto the display CRS
and read its band. The seam between them is this generator, and it carries two behaviours that the reduction
on the other side of it cannot express — a frame that warps to nothing is *skipped* rather than yielded as an
empty array, and nothing is warped at all until the values are asked for.
"""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference

from digitalearth.base.crs import OffLimbError
from digitalearth.static import Map
from digitalearth.static.maps import animation


def _frame(value: float) -> Dataset:
    """A small global lon/lat raster whose every cell holds ``value``.

    Args:
        value: The single value this frame contributes to a colour range.

    Returns:
        Dataset: a 6x12 global EPSG:4326 raster.
    """
    arr = np.full((6, 12), value, dtype="float32")
    return Dataset.from_array(
        arr=arr,
        geo_ref=GeoReference(geo=(-180.0, 30.0, 0.0, 90.0, 0.0, -30.0), epsg=4326),
    )


class TestFrameValues:
    """What the generator yields, when it yields it, and what it reads from each frame."""

    def test_a_frame_that_cannot_be_warped_is_skipped_and_the_rest_still_arrive(
        self, monkeypatch
    ):
        """An off-limb frame drops out of the scan while its measurable neighbours are still read.

        Args:
            monkeypatch: Makes the middle frame unwarpable, so no globe geometry is needed to stage it.

        Test scenario:
            The mixed stack, which the whole-stack off-limb tests never reach: on a globe view, one frame of
            a moving AOI can leave the visible hemisphere while the others stay on it. A skipped frame must
            contribute *nothing* — yielding an empty array would be equivalent here, but yielding a
            placeholder range would floor the animation's colour scale at zero and crush the real values
            into the top of the ramp.
        """
        scene = Map(crs=4326)
        frames = [_frame(10.0), _frame(20.0), _frame(30.0)]
        hidden = frames[1]

        def reproject(dataset, *args, **kwargs):
            """Warp every frame but ``hidden``, which is treated as entirely off the view."""
            if dataset is hidden:
                raise OffLimbError("this frame is on the far side of the globe")
            return dataset

        monkeypatch.setattr(type(scene), "_reproject", staticmethod(reproject))
        values = list(scene._frame_values(frames))
        assert len(values) == 2, (
            f"the unwarpable frame must be skipped, so 2 arrays should arrive, got {len(values)}"
        )
        assert [float(np.nanmax(arr)) for arr in values] == [10.0, 30.0], (
            "the surviving frames must arrive in series order, carrying their own values"
        )

    def test_the_skipped_frame_leaves_the_measured_range_alone(self, monkeypatch):
        """The range over a mixed stack is the range of the frames that could be drawn.

        Args:
            monkeypatch: Hides the frame holding the extreme value.

        Test scenario:
            The reason the skip matters, stated in the units the caller sees. The hidden frame carries the
            stack's maximum, so counting it would stretch the ramp to a value no frame ever renders, and
            every drawn frame would then occupy the bottom half of the colours.
        """
        scene = Map(crs=4326)
        frames = [_frame(10.0), _frame(900.0)]
        hidden = frames[1]

        def reproject(dataset, *args, **kwargs):
            """Warp only the visible frame."""
            if dataset is hidden:
                raise OffLimbError("this frame is on the far side of the globe")
            return dataset

        monkeypatch.setattr(type(scene), "_reproject", staticmethod(reproject))
        assert scene._measured_clim(frames) == (10.0, 10.0), (
            f"the hidden frame must not widen the range, got {scene._measured_clim(frames)}"
        )

    def test_a_stack_no_frame_of_which_warps_measures_nothing(self, monkeypatch):
        """Every frame off the view answers ``None``, which is not the same as a ``(0, 1)`` range.

        Args:
            monkeypatch: Makes every frame unwarpable.

        Test scenario:
            ``_measured_clim`` feeds :meth:`Map._clim_across_views`, which unions one result per swept
            projection. A view that shows none of the data has to contribute nothing; a placeholder there
            floors the union at zero, which is the bug the ``None`` return exists to prevent.
        """
        scene = Map(crs=4326)

        def reproject(dataset, *args, **kwargs):
            """Refuse every frame."""
            raise OffLimbError("nothing is on this view")

        monkeypatch.setattr(type(scene), "_reproject", staticmethod(reproject))
        assert scene._measured_clim([_frame(1.0), _frame(2.0)]) is None, (
            "a stack with no drawable frame must measure None, not a placeholder range"
        )

    def test_nothing_is_warped_until_the_values_are_consumed(self, monkeypatch):
        """Calling the generator warps no frame; the cost is paid by whoever iterates it.

        Args:
            monkeypatch: Counts the warps.

        Test scenario:
            The cap on how many frames a scan reads only bounds anything if the frames are warped lazily,
            one at a time, as the reduction pulls them. A method that built a list up front would pay for
            every sampled frame even when the caller stops early, and would hold every warped raster in
            memory at once.
        """
        scene = Map(crs=4326)
        warps = []

        def reproject(dataset, *args, **kwargs):
            """Record the warp, then hand the frame back unchanged."""
            warps.append(dataset)
            return dataset

        monkeypatch.setattr(type(scene), "_reproject", staticmethod(reproject))
        values = scene._frame_values([_frame(1.0), _frame(2.0), _frame(3.0)])
        assert warps == [], (
            f"no frame should be warped before iteration, warped {len(warps)}"
        )
        next(iter(values))
        assert len(warps) == 1, (
            f"pulling one value must warp exactly one frame, warped {len(warps)}"
        )

    @pytest.mark.parametrize("band", [1, 2])
    def test_the_requested_band_is_the_one_read(self, band, monkeypatch):
        """The band being animated is forwarded to every frame, not silently replaced by band 1.

        Args:
            band: The 1-based band the caller asked for.
            monkeypatch: Replaces the band read so the assertion is about the argument, not the raster.

        Test scenario:
            The colour scale has to describe the band on screen. Reading band 1 while drawing band 2 gives a
            scale derived from different data, which is invisible in a single still and wrong in every frame.
        """
        scene = Map(crs=4326)
        asked = []

        def read(warped, band=1):
            """Note which band was requested, then return a one-value array."""
            asked.append(band)
            return np.array([float(band)])

        monkeypatch.setattr(
            type(scene), "_reproject", staticmethod(lambda ds, *a, **k: ds)
        )
        monkeypatch.setattr(animation, "read_masked_band", read)
        list(scene._frame_values([_frame(1.0), _frame(2.0)], band=band))
        assert asked == [band, band], (
            f"every frame must be read at band={band}, got {asked}"
        )
