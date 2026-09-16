"""The three display helpers, lifted out of the tier base classes (DE-17, #277).

The roadmap called them "byte-identical in two tiers". Measured, only one of the three was: `_auto_cmap`
had three distinct bodies across three tiers, each claiming in its docstring to mirror the others.
"""

import numpy as np
import pytest

from digitalearth.base.display import (
    DEFAULT_CMAP,
    auto_cmap,
    needs_reproject,
    to_display_source,
)
from digitalearth.base.sources import DimensionInfo, Source


class _Placed:
    """A stand-in for a pyramids object that declares an EPSG code."""

    def __init__(self, epsg):
        self.epsg = epsg


class TestNeedsReproject:
    """Whether data has to be warped to reach the display CRS."""

    def test_a_matching_epsg_int_needs_no_warp(self):
        """The one case that answers False.

        Test scenario:
            The cheap comparison this exists for — data already in the display CRS must not pay for a warp.
        """
        assert needs_reproject(_Placed(4326), 4326) is False

    def test_a_different_epsg_int_warps(self):
        """Different codes mean different coordinates.

        Test scenario:
            The ordinary case, and the one whose absence would draw data in the wrong place.
        """
        assert needs_reproject(_Placed(4326), 3857) is True

    def test_a_proj4_display_crs_always_warps(self):
        """A CRS with no authority code cannot be compared, so it is never assumed equal.

        Test scenario:
            pyramids reports 4326 for a projection that names no authority, so comparing structurally would
            answer "already there" for data that is not — an orthographic globe drawn as plain lon/lat.
        """
        assert needs_reproject(_Placed(4326), "+proj=ortho +lat_0=53") is True

    def test_data_declaring_no_crs_is_answered_rather_than_raising(self):
        """The tolerant reading, which the static copy did not have.

        Test scenario:
            The static tier read `dataset.epsg` directly and raised AttributeError on an input declaring no
            CRS at all. The interactive and web copies used `getattr`; lifting the tolerant one is a superset,
            so no previously-returned answer changes.
        """
        assert needs_reproject(object(), 4326) is True


class TestAutoCmap:
    """Resolving a colormap from the variable, when the caller named none."""

    @staticmethod
    def _source(variable: str) -> Source:
        """Return a minimal Source carrying a variable name, which is what the lookup reads."""
        axis = DimensionInfo(np.array([0.0]), "x")
        return Source(None, axis, axis, crs=4326, metadata={"variable": variable})

    def test_a_caller_named_colormap_always_wins(self):
        """No lookup happens when the caller already chose.

        Test scenario:
            All three tier copies agreed on this, and it is the common path.
        """
        assert auto_cmap(self._source("rainfall"), "magma") == "magma"

    def test_an_unrecognised_variable_falls_back(self):
        """Something the lookup cannot place still gets a colormap.

        Test scenario:
            A builder needs a colormap unconditionally; returning None would fail inside matplotlib.
        """
        assert auto_cmap(self._source("not-a-known-variable"), None) == DEFAULT_CMAP

    def test_the_fallback_is_a_parameter(self, monkeypatch):
        """A tier with its own last resort can say so.

        Test scenario:
            The 3-D tier took `fallback` and the other two hard-coded "viridis" — the signature difference
            that stopped the three bodies being interchangeable, and the reason the general one was lifted.
            It has to be reached through a stubbed lookup: `auto_style` carries its own "viridis" default for
            an unrecognised variable, so in practice the fallback only covers a library answering with no
            colormap at all. The web tier's docstring said exactly that; this pins it.
        """
        import digitalearth.base.autostyle as autostyle

        monkeypatch.setattr(autostyle, "auto_style", lambda source: {})
        assert auto_cmap(self._source("anything"), None, "cividis") == "cividis"

    def test_a_supplied_lookup_is_consulted_instead_of_auto_style(self):
        """`lookup=` is how a tier keeps its colormap on the same lookup as its levels and units.

        Test scenario:
            The tiers' `_auto_style` / `_style_for` are documented as their single entry into the style
            table, and their `levels`/`units` readers go through them. Calling `auto_style` directly made the
            colormap a separate lookup and removed the one hook a subclass had for replacing all three.
        """
        consulted = []

        def lookup(source):
            consulted.append(source)
            return {"cmap": "from-the-tier"}

        source = self._source("t2m")
        assert auto_cmap(source, None, lookup=lookup) == "from-the-tier", (
            "the supplied lookup's answer must be used"
        )
        assert consulted == [source], (
            "and it must be asked exactly once, about this source"
        )

    def test_a_supplied_lookup_is_not_consulted_when_the_caller_named_a_colormap(self):
        """The caller's `cmap` still short-circuits, whichever lookup is in play.

        Test scenario:
            A lookup can be expensive (it may read CF attributes); a named colormap makes it unnecessary.
        """
        consulted = []
        assert auto_cmap(None, "magma", lookup=consulted.append) == "magma"
        assert consulted == [], (
            "the lookup must not run when there is nothing to resolve"
        )

    def test_a_supplied_lookup_replaces_auto_style_and_an_empty_answer_reaches_the_fallback(
        self, monkeypatch
    ):
        """With `lookup=` given, `auto_style` is never asked, and a lookup naming no colormap still falls back.

        Test scenario:
            The returned colormap alone cannot show that `auto_style` did not also run. Making it fail proves
            the tier's lookup is the only one consulted, and a lookup answering `{}` is handled exactly as an
            empty `auto_style` answer would be — by the `fallback` the caller passed.
        """
        import digitalearth.base.autostyle as autostyle

        monkeypatch.setattr(
            autostyle,
            "auto_style",
            lambda _: pytest.fail("auto_style must not run when a lookup is supplied"),
        )
        assert (
            auto_cmap(self._source("t2m"), None, "cividis", lookup=lambda _: {})
            == "cividis"
        ), (
            "an empty answer from the supplied lookup must fall back to the caller's fallback"
        )

    def test_a_style_entry_carrying_an_explicit_none_still_yields_a_colormap(
        self, monkeypatch
    ):
        """The defect the three spellings disagreed about.

        Test scenario:
            `.get("cmap", "viridis")` — the interactive and web spelling — returns None when the style entry
            carries `cmap=None` explicitly, because the key *is* present. `.get("cmap") or fallback`, the 3-D
            spelling, is the one that is right, and is what was lifted. Without this the difference between
            the three bodies would be invisible to the suite.
        """
        import digitalearth.base.autostyle as autostyle

        monkeypatch.setattr(autostyle, "auto_style", lambda source: {"cmap": None})
        assert auto_cmap(self._source("anything"), None) == DEFAULT_CMAP, (
            "an explicit cmap=None must fall back, not be returned as the colormap"
        )


class TestToDisplaySource:
    """The single display-CRS choke point every raster and vector builder calls."""

    @staticmethod
    def _raster():
        """Return a small pyramids Dataset, read from the repository's example data."""
        from pathlib import Path

        from pyramids.dataset import Dataset

        path = Path(__file__).resolve().parents[2] / "examples" / "data" / "acc4000.tif"
        return Dataset.read_file(str(path))

    def test_a_source_passes_straight_back(self):
        """Something already placed is not placed again.

        Test scenario:
            A `Source` has been through this once. Re-extracting it would at best waste the work and at
            worst reproject coordinates that are already in the display CRS.
        """
        axis = DimensionInfo(np.array([0.0]), "x")
        source = Source(None, axis, axis, crs=3857)
        assert to_display_source(source, 4326) is source, (
            "a Source must be returned unchanged, not re-extracted"
        )

    def test_a_plain_array_is_extracted_without_reprojection(self):
        """Input with no CRS to warp from goes straight to extraction.

        Test scenario:
            A raw numpy array declares no CRS and exposes no `to_crs`, so the reprojection branch must not
            be entered — attempting it would raise on an input the extractor handles perfectly well.
        """
        view = to_display_source(np.arange(12.0).reshape(3, 4), 4326)
        assert view.z.values.shape == (3, 4), "the array must be extracted as given"

    def test_data_already_in_the_display_crs_is_not_warped(self, monkeypatch):
        """A matching EPSG int skips the warp entirely.

        Test scenario:
            This is what `needs_reproject` is consulted for. Warping data that is already in the display CRS
            costs a full resample and can lose precision, for no change.
        """
        import digitalearth.base.display as display

        raster = self._raster()
        monkeypatch.setattr(
            display,
            "reproject",
            lambda *a, **k: pytest.fail(
                "data already in the display CRS must not be warped"
            ),
        )
        assert to_display_source(raster, raster.epsg).z is not None

    def test_data_in_another_crs_is_warped_through_pyramids(self, monkeypatch):
        """A differing CRS goes through the warp.

        Test scenario:
            The other half of the same decision, and the one that places data correctly. pyramids owns the
            warp; this only decides whether to ask for it.
        """
        import digitalearth.base.display as display

        raster = self._raster()
        asked = []
        monkeypatch.setattr(
            display,
            "reproject",
            lambda data, crs: asked.append(crs) or data,
        )
        to_display_source(raster, 3857)
        assert asked == [3857], f"expected one warp to 3857, got {asked}"
