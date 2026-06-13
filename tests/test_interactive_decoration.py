"""DI.1c — interactive decoration (tiles / coastlines / features / legend / colorbar).

Type-level and overlay-order assertions only — constructing tile/feature elements touches no
network (geometry/tiles are fetched by the renderer at display time, which these tests never do).
Runs in the ``interactive`` pixi env; every test ``importorskip``s geoviews.
"""

import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture()
def m() -> InteractiveMap:
    """A fresh Web-Mercator map for each test."""
    return InteractiveMap()


class TestTiles:
    """``tiles`` — web-tile basemaps."""

    def test_tiles_is_wmts_underlay(self, m, dataset):
        m.image(dataset).tiles("CartoLight")
        assert isinstance(m.layers[0], gv.element.WMTS), f"got {type(m.layers[0])}"
        assert isinstance(
            m.layers[1], hv.Image
        ), "tiles must insert beneath the data layers"

    def test_unknown_provider_raises_with_catalog(self, m):
        with pytest.raises(ValueError, match="unknown tile provider"):
            m.tiles("NotARealProvider")

    def test_non_mercator_map_raises(self):
        with pytest.raises(ValueError, match="crs=3857"):
            InteractiveMap(crs=4326).tiles()

    def test_chains(self, m):
        assert m.tiles() is m

    def test_constructor_tiles_apply_once_at_render(self, dataset):
        """``InteractiveMap(tiles=...)`` prepends the basemap on first render only."""
        m = InteractiveMap(tiles="CartoLight")
        m.image(dataset)
        first = m.render()
        assert isinstance(first, hv.Overlay) and len(first) == 2
        assert isinstance(m.layers[0], gv.element.WMTS)
        again = m.render()
        assert len(again) == 2, "re-rendering must not stack a second tile layer"

    def test_explicit_tiles_supersede_constructor_tiles(self, dataset):
        """A constructor provider + an explicit .tiles() must not stack two basemaps (L2)."""
        m = InteractiveMap(tiles="CartoLight").image(dataset).tiles("OSM")
        overlay = m.render()
        wmts = [layer for layer in overlay if isinstance(layer, gv.element.WMTS)]
        assert len(wmts) == 1, f"expected exactly one basemap, got {len(wmts)}"

    def test_custom_xyz_url_builds_wmts(self, m):
        """DI.10: a raw {Z}/{X}/{Y} URL template becomes a gv.WMTS without a catalog lookup."""
        m.tiles("https://a.tile.example/{Z}/{X}/{Y}.png")
        assert isinstance(m.layers[0], gv.element.WMTS), f"got {type(m.layers[0])}"

    def test_xyzservices_provider_builds_wmts(self, m):
        """DI.10: an xyzservices TileProvider object resolves through gv.WMTS."""
        xyz = pytest.importorskip("xyzservices")
        m.tiles(xyz.providers.OpenStreetMap.Mapnik)
        assert isinstance(m.layers[0], gv.element.WMTS), f"got {type(m.layers[0])}"

    def test_list_tile_providers_is_nonempty_sorted(self, m):
        providers = m.list_tile_providers()
        assert "CartoLight" in providers and "OSM" in providers
        assert providers == sorted(providers), "provider catalog must be sorted"

    def test_keyed_provider_without_key_raises(self, m):
        """A Stadia provider used without an api_key raises rather than rendering blank tiles."""
        providers = [p for p in m.list_tile_providers() if "stadia" in p.lower()]
        if not providers:  # pragma: no cover - GeoViews build without Stadia sources
            pytest.skip("no Stadia tile sources in this GeoViews build")
        with pytest.raises(ImportError, match="api_key"):
            m.tiles(providers[0])

    def test_overlay_level_puts_tiles_on_top(self, m, dataset):
        m.image(dataset).tiles("CartoLight", level="overlay")
        assert isinstance(
            m.layers[-1], gv.element.WMTS
        ), "overlay tiles must be the top layer"


class TestCoastlinesAndFeatures:
    """``coastlines`` / ``features`` — Natural-Earth context layers."""

    def test_coastline_is_feature_overlay(self, m, dataset):
        m.image(dataset).coastlines()
        assert isinstance(m.layers[-1], gv.element.Feature), f"got {type(m.layers[-1])}"

    def test_coastline_resolution_recorded(self, m):
        m.coastlines(resolution="50m")
        plot = hv.Store.lookup_options("bokeh", m.layers[-1], "plot").kwargs
        assert plot["scale"] == "50m", f"scale not honoured: {plot.get('scale')}"

    def test_features_underlay_vs_overlay_order(self, m, dataset):
        m.image(dataset).features(land=True, borders=True)
        assert isinstance(
            m.layers[0], gv.element.Feature
        ), "land must underlay the raster"
        assert isinstance(m.layers[1], hv.Image)
        assert isinstance(
            m.layers[2], gv.element.Feature
        ), "borders must overlay the raster"

    def test_features_none_requested_is_noop(self, m):
        m.features()
        assert m.layers == []

    def test_non_mercator_features_raise(self):
        with pytest.raises(ValueError, match="crs=3857"):
            InteractiveMap(crs=4326).coastlines()


class TestTogglesAndCompose:
    """``legend`` / ``colorbar`` toggles and the DI.1 acceptance overlay."""

    def test_colorbar_toggle_rewrites_last_layer(self, m, dataset):
        m.image(dataset).colorbar(False)
        plot = hv.Store.lookup_options("bokeh", m.layers[-1], "plot").kwargs
        assert plot["colorbar"] is False

    def test_colorbar_without_layers_raises(self, m):
        with pytest.raises(ValueError, match="at least one layer"):
            m.colorbar()

    def test_legend_toggle(self, m, dataset):
        m.contours(dataset, levels=4).legend(False)
        plot = hv.Store.lookup_options("bokeh", m.layers[-1], "plot").kwargs
        assert plot["show_legend"] is False

    def test_legend_without_layers_raises(self, m):
        with pytest.raises(ValueError, match="at least one layer"):
            m.legend()

    def test_tiles_style_opts_forwarded(self, m):
        """Extra opts on tiles() reach the cloned element (the shared singleton stays pristine)."""
        m.tiles("CartoLight", alpha=0.4)
        style = hv.Store.lookup_options("bokeh", m.layers[0], "style").kwargs
        assert style["alpha"] == 0.4, f"tile opts not applied: {style.get('alpha')}"

    def test_coastline_style_opts_forwarded(self, m):
        m.coastlines(line_width=2.0)
        style = hv.Store.lookup_options("bokeh", m.layers[-1], "style").kwargs
        assert (
            style["line_width"] == 2.0
        ), f"feature opts not applied: {style.get('line_width')}"

    def test_image_tiles_coastlines_compose(self, m, dataset):
        """The DI.1 acceptance chain: raster + basemap + coastline in one ordered overlay."""
        m.image(dataset).tiles().coastlines()
        overlay = m.render()
        assert isinstance(overlay, hv.Overlay)
        kinds = [type(layer) for layer in overlay]
        assert len(overlay) == 3, f"expected 3 layers, got {len(overlay)}"
        assert issubclass(kinds[0], gv.element.WMTS), "tiles must be the bottom layer"
        assert issubclass(
            kinds[2], gv.element.Feature
        ), "coastline must be the top layer"
