"""DI.3 — time-slider datacubes (``timecube`` over a DatasetCollection).

``DynamicMap`` frames are lazy, so the tests materialise a frame (``dmap[key]``) before asserting on
it. Runs in the ``interactive`` pixi env; every test ``importorskip``s geoviews.
"""

import numpy as np
import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture()
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


@pytest.fixture(scope="module")
def cube():
    """A 3-member DatasetCollection (the acc4000 raster repeated) — distinct via scaling below."""
    from pyramids.dataset.collection import DatasetCollection

    return DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)


class TestTimecube:
    """``timecube`` — the DynamicMap time slider with a frozen colour range."""

    def test_is_dynamicmap_with_one_kdim_of_member_length(self, m, cube):
        m.timecube(cube)
        layer = m.layers[0]
        assert isinstance(
            layer, hv.DynamicMap
        ), f"expected hv.DynamicMap, got {type(layer)}"
        assert [d.name for d in layer.kdims] == ["time"], "one 'time' kdim expected"
        assert len(layer.kdims[0].values) == 3, "slider must span the three members"

    def test_frames_materialise_as_images(self, m, cube):
        m.timecube(cube)
        dmap = m.layers[0]
        first = dmap[0]
        assert isinstance(
            first, hv.Image
        ), f"a frame must be an hv.Image, got {type(first)}"

    def test_clim_is_frozen_across_frames(self, m, cube):
        """The first and last frame must carry an identical colour range (no per-frame jump)."""
        m.timecube(cube)
        dmap = m.layers[0]
        first_clim = hv.Store.lookup_options("bokeh", dmap[0], "plot").kwargs["clim"]
        last_clim = hv.Store.lookup_options("bokeh", dmap[2], "plot").kwargs["clim"]
        assert (
            first_clim == last_clim
        ), f"clim jumped between frames: {first_clim} vs {last_clim}"
        assert np.isfinite(first_clim).all(), f"clim must be finite, got {first_clim}"

    def test_explicit_clim_is_respected(self, m, cube):
        m.timecube(cube, clim=(0.0, 100.0))
        clim = hv.Store.lookup_options("bokeh", m.layers[0][0], "plot").kwargs["clim"]
        assert clim == (0.0, 100.0), f"explicit clim not honoured: {clim}"

    def test_datetime_labels_drive_the_slider(self, m, cube):
        import datetime as dt

        stamps = [dt.datetime(2020, 1, day) for day in (1, 2, 3)]
        m.timecube(cube, labels=stamps)
        dmap = m.layers[0]
        assert (
            list(dmap.kdims[0].values) == stamps
        ), "slider keys must be the supplied datetimes"
        frame = dmap[stamps[1]]
        assert isinstance(frame, hv.Image), "a label-keyed frame must materialise"

    def test_label_count_mismatch_raises(self, m, cube):
        with pytest.raises(ValueError, match="labels has 2 entries"):
            m.timecube(cube, labels=["a", "b"])

    def test_cmap_is_applied_to_frames(self, m, cube):
        m.timecube(cube, cmap="inferno")
        style = hv.Store.lookup_options("bokeh", m.layers[0][0], "style").kwargs
        assert style["cmap"] == "inferno", f"cmap not honoured: {style.get('cmap')}"

    def test_chains(self, m, cube):
        assert m.timecube(cube) is m


class TestGlobalClim:
    """``_global_clim`` — the frozen colour range over the whole stack."""

    def test_spans_all_members(self, m, cube):
        vmin, vmax = m._global_clim(cube, band=1)
        assert vmin <= vmax and np.isfinite([vmin, vmax]).all()

    def test_all_nodata_members_fall_back_to_unit_range(self, m, cube, monkeypatch):
        """When every member is all-NaN, the range falls back to (0.0, 1.0) instead of erroring."""
        from digitalearth.sources.dimension import DimensionInfo
        from digitalearth.sources.source import Source

        def _all_nan_source(self, data, *, band=1):
            grid = np.full((2, 2), np.nan)
            return Source(
                DimensionInfo(grid, "z"),
                DimensionInfo(np.arange(2.0), "x"),
                DimensionInfo(np.arange(2.0), "y"),
                crs=3857,
            )

        monkeypatch.setattr(InteractiveMap, "_to_display_source", _all_nan_source)
        assert m._global_clim(cube, band=1) == (0.0, 1.0)
