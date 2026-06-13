"""DI.9 — projections beyond Web-Mercator via the matplotlib backend (projection / graticule).

Renders Orthographic/Robinson through the HoloViews matplotlib backend (no browser); asserts the
backend switch, the projection on the rendered object, the tiles-vs-projection guards, and a headless
PNG export. Runs in the ``interactive`` pixi env.
"""

import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture()
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


class TestProjection:
    """``projection`` — arbitrary display projections via the matplotlib backend."""

    def test_projection_makes_rasters_gv_with_crs(self, m, dataset):
        """Under a projection, image() emits a gv.Image (crs-aware) so GeoViews can reproject it."""
        m.projection("Robinson")
        m.image(dataset)
        assert isinstance(
            m.layers[0], gv.Image
        ), f"expected gv.Image under a projection, got {type(m.layers[0])}"

    def test_no_projection_keeps_plain_hv_image(self, m, dataset):
        m.image(dataset)
        assert isinstance(m.layers[0], hv.Image) and not isinstance(
            m.layers[0], gv.Image
        )

    def test_render_sets_projection_on_the_object(self, m, dataset):
        m.projection("Orthographic").image(dataset)
        obj = m.render()
        proj = hv.Store.lookup_options("matplotlib", obj, "plot").kwargs.get(
            "projection"
        )
        assert (
            proj is not None
        ), "render() must pass the projection through to the mpl backend"
        assert "Orthographic" in type(proj).__name__

    def test_orthographic_png_export(self, m, dataset, tmp_path):
        out = tmp_path / "globe.png"
        m.projection("Orthographic").image(dataset).save(str(out))
        assert out.exists() and out.stat().st_size > 0

    def test_projection_none_returns_to_bokeh_path(self, m, dataset):
        m.projection("Robinson")
        m.projection(None)
        m.image(dataset)
        assert isinstance(m.layers[0], hv.Image) and not isinstance(
            m.layers[0], gv.Image
        )

    def test_chains(self, m):
        assert m.projection("Robinson") is m

    def test_epsg_int_projection(self, m, dataset):
        """An EPSG int projection resolves via process_crs and renders through the mpl backend."""
        m.projection(3857).image(dataset)
        obj = m.render()
        proj = hv.Store.lookup_options("matplotlib", obj, "plot").kwargs.get(
            "projection"
        )
        assert proj is not None, "EPSG-int projection must reach the mpl backend"

    def test_cartopy_object_passes_through(self, m, dataset):
        """A pre-built cartopy projection object is used verbatim (no name resolution)."""
        import cartopy.crs as ccrs

        proj = ccrs.Mollweide()
        m.projection(proj).image(dataset)
        assert m._projection is proj, "a cartopy object must pass straight through"

    def test_unknown_projection_name_raises(self, m):
        with pytest.raises(ValueError, match="unknown projection"):
            m.projection("NotAProjection")

    def test_graticule_opts_forwarded(self, m):
        m.graticule(line_width=0.5)
        assert isinstance(m.layers[-1], gv.element.Feature)


class TestProjectionTileGuards:
    """tiles and a non-Mercator projection are mutually exclusive (both directions)."""

    def test_projection_after_tiles_raises(self, m, dataset):
        m.image(dataset).tiles("CartoLight")
        with pytest.raises(ValueError, match="Web-Mercator only"):
            m.projection("Robinson")

    def test_tiles_after_projection_raises(self, m, dataset):
        m.projection("Robinson").image(dataset)
        with pytest.raises(ValueError, match="Web-Mercator only"):
            m.tiles("CartoLight")

    def test_constructor_tiles_block_projection(self):
        m = InteractiveMap(tiles="CartoLight")
        with pytest.raises(ValueError, match="Web-Mercator only"):
            m.projection("Robinson")


class TestGraticule:
    """``graticule`` — lon/lat grid feature."""

    def test_graticule_adds_feature(self, m):
        m.graticule()
        assert isinstance(m.layers[0], gv.element.Feature), f"got {type(m.layers[0])}"

    def test_chains(self, m):
        assert m.graticule() is m
