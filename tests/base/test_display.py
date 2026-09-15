"""The three display helpers, lifted out of the tier base classes (DE-17, #277).

The roadmap called them "byte-identical in two tiers". Measured, only one of the three was: `_auto_cmap`
had three distinct bodies across three tiers, each claiming in its docstring to mirror the others.
"""

import numpy as np

from digitalearth.base.display import DEFAULT_CMAP, auto_cmap, needs_reproject
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
