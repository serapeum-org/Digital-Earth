"""The web tier derives a stack colour range by the shared rule (#174 / DE-9).

This tier used to read ``collection.datasets[:50]`` — a head slice that measured only the beginning of a series
and clipped any later peak to solid top-of-ramp. It now strides across the whole stack, like the static and
interactive tiers.
"""

import numpy as np
import pytest

from digitalearth.base.clim import DEFAULT_CLIM_SCAN_CAP
from digitalearth.web import WebMap
from digitalearth.web.temporal import _CLIM_SCAN_CAP


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the web extra is absent."""
    pytest.importorskip("maplibre")


class _Band:
    """The ``.z`` of a stubbed source: just the values a frame would contribute."""

    def __init__(self, values):
        self.values = values


class _Source:
    """A stand-in for a display-CRS source, carrying one frame's values."""

    def __init__(self, values):
        self.z = _Band(values)


class _Stack:
    """A stand-in ``DatasetCollection`` whose members are their own index."""

    def __init__(self, count):
        self.datasets = list(range(count))


class TestTheStackScanSpansTheSeries:
    """The colour range describes the whole stack, not its first 50 frames."""

    def test_a_late_peak_sets_the_upper_limit(self, monkeypatch):
        """A value near the end of a 60-frame stack reaches the colour range.

        Args:
            monkeypatch: Replaces the display-CRS read, so the test needs no rasters on disk.

        Test scenario:
            Each frame contributes its own index as its only value, so ``vmax`` is exactly the highest frame
            the scan looked at. Under the old ``datasets[:50]`` head slice that was 49 and every later frame
            was clipped; under the shared index-selected sample it reaches 59, the final frame.
        """
        scene = WebMap()
        monkeypatch.setattr(
            scene,
            "_to_display_source",
            lambda member, band=1: _Source(np.array([float(member)])),
        )
        vmin, vmax = scene._global_clim(_Stack(60), 1)
        assert vmin == 0.0, f"the range must start at the first frame, got {vmin}"
        assert vmax == 59.0, (
            f"vmax={vmax}: the old head slice stopped at 49, and the index-selected sample ends at 59 — the "
            "same value the static and interactive tiers answer for this stack"
        )

    def test_the_scan_is_bounded_by_the_shared_cap(self, monkeypatch):
        """No more frames are read than the shared cap allows.

        Args:
            monkeypatch: Records which members the scan actually read.

        Test scenario:
            The cap is what keeps a 500-member cube from paying 500 warps for a colour range; it has to still
            bound the scan now that the sampling spans the series.
        """
        scene = WebMap()
        seen = []

        def record(member, band=1):
            """Note the member, then hand back a one-value frame."""
            seen.append(member)
            return _Source(np.array([float(member)]))

        monkeypatch.setattr(scene, "_to_display_source", record)
        scene._global_clim(_Stack(60), 1)
        assert len(seen) <= DEFAULT_CLIM_SCAN_CAP, (
            f"the scan read {len(seen)} frames, over the cap of {DEFAULT_CLIM_SCAN_CAP}"
        )

    def test_a_masked_member_measures_as_its_nan_filled_twin(self, monkeypatch):
        """A member handed over masked answers the same range as the NaN-filled array pyramids returns.

        Args:
            monkeypatch: Replaces the display-CRS read with a masked frame.

        Test scenario:
            This tier used to fill the mask itself, immediately before measuring; the fill moved into
            :func:`~digitalearth.base.clim.measure_clim` with the rest of the reduction. If the move had
            dropped it, ``np.asarray`` would strip the mask and let the ``-9999`` sentinel through as a real
            minimum — the whole raster would then render in the top sliver of the colour ramp.
        """
        scene = WebMap()
        masked = np.ma.masked_array([1.0, -9999.0, 3.0], mask=[False, True, False])
        monkeypatch.setattr(
            scene, "_to_display_source", lambda member, band=1: _Source(masked)
        )
        assert scene._global_clim(_Stack(1), 1) == (1.0, 3.0), (
            f"the nodata sentinel reached the range: {scene._global_clim(_Stack(1), 1)}"
        )

    def test_a_stack_with_nothing_measurable_falls_back_to_zero_one(self, monkeypatch):
        """An all-nodata stack yields limits a colormap can take, rather than raising.

        Args:
            monkeypatch: Makes every member contribute no finite value.

        Test scenario:
            ``timeslider`` needs a range before it can draw anything, so the scan has to answer even for a
            cube it could measure nothing in. The public form of the rule is the one that falls back; the
            ``None`` form is reserved for callers that union several measurements.
        """
        scene = WebMap()
        monkeypatch.setattr(
            scene,
            "_to_display_source",
            lambda member, band=1: _Source(np.array([np.nan, np.nan])),
        )
        assert scene._global_clim(_Stack(3), 1) == (0.0, 1.0), (
            f"an unmeasurable stack must fall back, got {scene._global_clim(_Stack(3), 1)}"
        )

    def test_this_tier_reads_the_shared_cap(self):
        """The cap is imported, not spelled out again as 50.

        Test scenario:
            #174's root cause was three literals free to drift. Identity with the shared constant is what
            stops a fourth appearing here.
        """
        assert _CLIM_SCAN_CAP == DEFAULT_CLIM_SCAN_CAP, (
            f"the web tier must read the shared cap, got {_CLIM_SCAN_CAP}"
        )
