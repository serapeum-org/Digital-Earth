"""The interactive tier derives a stack colour range by the shared rule (#174 / DE-9).

This tier used to scan every member with no cap at all — correct, but it paid one warp per frame and gave a
different answer from the two capped tiers. It now reads the same evenly spaced sample they do.
"""

import numpy as np
import pytest

from digitalearth.base.clim import DEFAULT_CLIM_SCAN_CAP
from digitalearth.interactive import InteractiveMap


@pytest.fixture(autouse=True)
def _need_engine():
    """Skip the module when the interactive extra is absent."""
    pytest.importorskip("holoviews")


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
            the scan looked at. Uncapped this tier read all 60 and answered 59; under the shared stride it
            answers the same 57 the static and web tiers do.
        """
        scene = InteractiveMap()
        monkeypatch.setattr(
            scene,
            "_to_display_source",
            lambda member, band=1: _Source(np.array([float(member)])),
        )
        vmin, vmax = scene._global_clim(_Stack(60), 1)
        assert vmin == 0.0, f"the range must start at the first frame, got {vmin}"
        assert vmax == 57.0, (
            f"vmax={vmax} must match the static and web tiers' shared sample, which ends at 57"
        )

    def test_the_scan_is_bounded_by_the_shared_cap(self, monkeypatch):
        """No more frames are read than the shared cap allows.

        Args:
            monkeypatch: Records which members the scan actually read.

        Test scenario:
            The cap is what keeps a 500-member cube from paying 500 warps for a colour range; it has to still
            bound the scan now that the sampling spans the series.
        """
        scene = InteractiveMap()
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
