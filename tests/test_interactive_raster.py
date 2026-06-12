"""DI.1a — interactive raster builders (image / rgb / quadmesh / contours / spaghetti).

Element-type and styling assertions plus matplotlib-backend render smokes — no browser, no network.
Runs in the ``interactive`` pixi env (``pixi run -e interactive test-interactive``); every test
``importorskip``s geoviews so the lean ``dev`` env skips cleanly.
"""

import numpy as np
import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
pytest.importorskip("geoviews")


@pytest.fixture()
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


class TestImage:
    """``image`` — the I1 recipe (Dataset → display-CRS ``hv.Image``)."""

    def test_registers_hv_image_and_chains(self, m, dataset):
        out = m.image(dataset)
        assert out is m, "image() must return the map for chaining"
        assert len(m.layers) == 1
        assert isinstance(
            m.layers[0], hv.Image
        ), f"expected hv.Image, got {type(m.layers[0])}"

    def test_image_is_plain_hv_not_gv(self, m, dataset):
        """Option A: pre-reprojected coordinates must NOT carry a GeoViews CRS (no re-projection)."""
        import geoviews as gv

        m.image(dataset)
        assert not isinstance(
            m.layers[0], gv.element.geo._Element
        ), "raster elements must be plain hv.Image — a gv element would re-project 3857 coords"

    def test_coordinates_are_display_crs(self, m, dataset):
        """The element's x samples must be the reprojected (Web-Mercator) cell centres."""
        m.image(dataset)
        x_samples = m.layers[0].dimension_values("x", expanded=False)
        src = m._to_display_source(dataset)
        assert np.allclose(
            np.sort(x_samples), np.sort(src.x.values)
        ), "element x coordinates must match the pyramids-reprojected cell centres"

    def test_clim_and_cmap_are_recorded(self, m, dataset):
        m.image(dataset, cmap="magma", clim=(0.0, 50.0), alpha=0.5)
        opts = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert opts["cmap"] == "magma", f"cmap not honoured: {opts.get('cmap')}"
        assert opts["alpha"] == 0.5, f"alpha not honoured: {opts.get('alpha')}"
        plot = hv.Store.lookup_options("bokeh", m.layers[0], "plot").kwargs
        assert plot["clim"] == (0.0, 50.0), f"clim not honoured: {plot.get('clim')}"

    def test_bokeh_frame_and_hover_are_bokeh_only(self, m, dataset):
        m.image(dataset)
        plot = hv.Store.lookup_options("bokeh", m.layers[0], "plot").kwargs
        assert plot["width"] == m.width and plot["height"] == m.height
        assert "hover" in plot["tools"], f"hover tool missing: {plot.get('tools')}"

    def test_nodata_renders_as_nan(self, m, dataset):
        """Masked (NoData) cells must become NaN so Bokeh draws them transparent."""
        m.image(dataset)
        values = m.layers[0].dimension_values(2, flat=False)
        assert np.isnan(
            values
        ).any(), "fixture nodata cells should surface as NaN in the element"

    def test_mpl_backend_render_smoke(self, m, dataset, tmp_path):
        out = tmp_path / "image.png"
        m.image(dataset).save(str(out))
        assert out.exists() and out.stat().st_size > 0


class TestRgb:
    """``rgb`` — three-band composite with a percentile stretch."""

    def test_registers_hv_rgb(self, m, dataset):
        m.rgb(dataset, bands=(1, 1, 1))  # single-band fixture: grey composite
        assert isinstance(
            m.layers[0], hv.RGB
        ), f"expected hv.RGB, got {type(m.layers[0])}"

    def test_channels_are_stretched_to_unit_range(self, m, dataset):
        m.rgb(dataset, bands=(1, 1, 1))
        red = m.layers[0].dimension_values("R", flat=False)
        finite = red[np.isfinite(red)]
        assert (
            finite.min() >= 0.0 and finite.max() <= 1.0
        ), "channels must be clipped to [0, 1]"

    def test_wrong_band_count_raises(self, m, dataset):
        with pytest.raises(ValueError, match="three bands"):
            m.rgb(dataset, bands=(1, 2))

    def test_already_display_crs_skips_reproject(self, m, dataset, monkeypatch):
        """A dataset already in 3857 must not be warped again on the rgb path."""
        mercator = dataset.to_crs(3857)

        def _boom(*a, **k):  # pragma: no cover - only fires on regression
            raise AssertionError(
                "to_crs must not run for data already in the display CRS"
            )

        monkeypatch.setattr(type(mercator), "to_crs", _boom)
        m.rgb(mercator, bands=(1, 1, 1))
        assert isinstance(m.layers[0], hv.RGB)


class TestQuadmeshAndContours:
    """``quadmesh`` / ``contours`` / ``filled_contours``."""

    def test_quadmesh_type(self, m, dataset):
        m.quadmesh(dataset)
        assert isinstance(m.layers[0], hv.QuadMesh), f"got {type(m.layers[0])}"

    def test_contours_type_and_levels(self, m, dataset):
        m.contours(dataset, levels=5)
        assert isinstance(m.layers[0], hv.element.Contours), f"got {type(m.layers[0])}"

    def test_filled_contours_are_polygons(self, m, dataset):
        m.filled_contours(dataset, levels=5)
        assert isinstance(m.layers[0], hv.element.Polygons), f"got {type(m.layers[0])}"

    def test_contours_mpl_render_smoke(self, m, dataset, tmp_path):
        out = tmp_path / "contours.png"
        m.contours(dataset, levels=4).save(str(out))
        assert out.exists() and out.stat().st_size > 0


class TestSpaghetti:
    """``spaghetti`` — one contour layer per collection member."""

    def test_one_layer_per_member(self, m):
        from pyramids.dataset.collection import DatasetCollection

        dc = DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)
        m.spaghetti(dc, levels=4)
        assert len(m.layers) == 3, f"expected 3 contour layers, got {len(m.layers)}"
        assert all(isinstance(layer, hv.element.Contours) for layer in m.layers)
