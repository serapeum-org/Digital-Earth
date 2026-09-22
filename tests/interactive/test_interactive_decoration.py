"""DI.1c — interactive decoration (tiles / coastlines / features / legend / colorbar).

Type-level and overlay-order assertions only — constructing tile/feature elements touches no
network (geometry/tiles are fetched by the renderer at display time, which these tests never do).
Runs in the ``interactive`` pixi env; every test ``importorskip``s geoviews.
"""

import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")


@pytest.fixture
def m():
    """Yield a fresh Web-Mercator map for each test, closing it on the way out.

    The object registry is process-global and holds strong references, so a map that is never closed
    keeps the data it drew for the rest of the session.

    Yields:
        The map.
    """
    interactive_map = InteractiveMap()
    yield interactive_map
    interactive_map.close()


def _element_of(interactive_map, element_type):
    """Return the one element of a type the map composes.

    Args:
        interactive_map: The map.
        element_type: The HoloViews element type wanted.

    Returns:
        The element — found by type rather than by position, because position is band order and that is
        exactly what the tests using this must not assume.
    """
    return next(
        element
        for element in interactive_map.layers
        if isinstance(element, element_type)
    )


class TestTiles:
    """``tiles`` — web-tile basemaps."""

    def test_tiles_is_wmts_underlay(self, m, dataset):
        m.image(dataset).tiles("CartoLight")
        assert isinstance(m.layers[0], gv.element.WMTS), f"got {type(m.layers[0])}"
        assert isinstance(m.layers[1], hv.Image), (
            "tiles must insert beneath the data layers"
        )

    def test_unknown_provider_raises_with_catalog(self, m):
        with pytest.raises(ValueError, match="unknown tile provider"):
            m.tiles("NotARealProvider")

    def test_non_mercator_map_raises(self):
        interactiveMap = InteractiveMap(crs=4326)
        with pytest.raises(ValueError, match="crs=3857"):
            interactiveMap.tiles()

    def test_chains(self, m):
        assert m.tiles() is m

    def test_constructor_tiles_apply_once_at_render(self, dataset):
        """``InteractiveMap(tiles=...)`` prepends the basemap on first render only."""
        m = InteractiveMap(tiles="CartoLight")
        m.image(dataset)
        first = m.render()
        assert isinstance(first, hv.Overlay)
        assert len(first) == 2
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
        assert "CartoLight" in providers
        assert "OSM" in providers
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
        assert isinstance(m.layers[-1], gv.element.WMTS), (
            "overlay tiles must be the top layer"
        )


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
        assert isinstance(m.layers[0], gv.element.Feature), (
            "land must underlay the raster"
        )
        assert isinstance(m.layers[1], hv.Image)
        assert isinstance(m.layers[2], gv.element.Feature), (
            "borders must overlay the raster"
        )

    def test_features_none_requested_is_noop(self, m):
        m.features()
        assert m.layers == []

    def test_feature_style_opts_reach_the_natural_earth_element(self, m):
        """A style recorded for a Natural-Earth layer has to end up on the element that is drawn.

        Args:
            m: A fresh Web-Mercator map.

        Test scenario:
            The five Natural-Earth kinds are now drawn from their description rather than built in the
            `features` loop, and `draw_natural_earth` clones the shared `gv.feature` singleton before
            styling it. A drawer that built the clone and dropped the recorded options would still
            register an element of the right type in the right band, so every other check in this class
            would pass while the caller's styling silently vanished. Read off the element, because that
            — not the record — is what a viewer sees.
        """
        m.features(land=True, alpha=0.3)
        style = hv.Store.lookup_options("bokeh", m.layers[-1], "style").kwargs
        assert style.get("alpha") == 0.3, (
            f"feature opts not applied: {style.get('alpha')}"
        )

    def test_non_mercator_features_raise(self):
        interactiveMap = InteractiveMap(crs=4326)
        with pytest.raises(ValueError, match="crs=3857"):
            interactiveMap.coastlines()


class TestTogglesAndCompose:
    """``legend`` / ``colorbar`` toggles and the DI.1 acceptance overlay."""

    def test_colorbar_toggle_rewrites_last_layer(self, m, dataset):
        m.image(dataset).colorbar(False)
        plot = hv.Store.lookup_options("bokeh", m.layers[-1], "plot").kwargs
        assert plot["colorbar"] is False

    def test_colorbar_acts_on_the_layer_just_added_when_it_is_not_on_top(
        self, m, dataset
    ):
        """A text label sits in the band over the data, so the raster added after it is drawn beneath it.

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            `colorbar()` read `self.layers[-1]`, and `layers` follows band order, so the toggle reached the
            label instead and HoloViews refused it: `Unexpected option 'colorbar' for Text type` (review H5).
        """
        m.text(4.0, 52.0, "label").image(dataset).colorbar(False)
        plot = hv.Store.lookup_options("bokeh", _element_of(m, hv.Image), "plot")
        assert plot.kwargs["colorbar"] is False, plot.kwargs

    def test_colorbar_does_not_restyle_a_layer_drawn_over_the_one_just_added(
        self, m, dataset
    ):
        """The quieter half of the same defect: an overlay accepts the option, so nothing raised.

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            `coastlines().image(dem).colorbar(False)` applied the option to the coastlines and left the
            raster's colorbar on — the call the caller made did nothing they could see.
        """
        m.coastlines().image(dataset).colorbar(False)
        plot = hv.Store.lookup_options("bokeh", _element_of(m, hv.Image), "plot")
        assert plot.kwargs["colorbar"] is False, plot.kwargs

    def test_an_underlay_added_after_the_data_does_not_take_the_toggle(
        self, m, dataset
    ):
        """A basemap is drawn beneath the data and has no colorbar, so the toggle stays with the raster.

        Args:
            m: The map.
            dataset: A small raster.

        Test scenario:
            Before the layer tree, an underlay was inserted at the front and never became the layer these
            toggles acted on; the web tier's `_last_layer_id` likewise counts data layers only. Tracking
            "the last layer added" without that exception would send this call to the tiles.
        """
        m.image(dataset).tiles("CartoLight").colorbar(False)
        plot = hv.Store.lookup_options("bokeh", _element_of(m, hv.Image), "plot")
        assert plot.kwargs["colorbar"] is False, plot.kwargs

    def test_legend_acts_on_the_layer_just_added(self, m, dataset):
        """`legend()` read the top-drawn layer too, so a contour under an overlay was never reached.

        Args:
            m: The map.
            dataset: A small raster to contour.
        """
        m.coastlines().contours(dataset, levels=4).legend(False)
        plot = hv.Store.lookup_options("bokeh", _element_of(m, hv.Contours), "plot")
        assert plot.kwargs["show_legend"] is False, plot.kwargs

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
        assert style["line_width"] == 2.0, (
            f"feature opts not applied: {style.get('line_width')}"
        )

    def test_image_tiles_coastlines_compose(self, m, dataset):
        """The DI.1 acceptance chain: raster + basemap + coastline in one ordered overlay."""
        m.image(dataset).tiles().coastlines()
        overlay = m.render()
        assert isinstance(overlay, hv.Overlay)
        kinds = [type(layer) for layer in overlay]
        assert len(overlay) == 3, f"expected 3 layers, got {len(overlay)}"
        assert issubclass(kinds[0], gv.element.WMTS), "tiles must be the bottom layer"
        assert issubclass(kinds[2], gv.element.Feature), (
            "coastline must be the top layer"
        )


class TestANamedLayerSaysWhereItIsDrawn:
    """A named builder's prose must name the band the registry files its kind in (round 2, M7).

    The six named Natural-Earth builders each delegate to `features()`, so which band they land in is the
    registry's answer, not theirs. Four of the six explain that answer in their own words, and the prose is
    what a caller reads — `lakes()` said "overlay" for a kind the registry files under `underlay`, which is
    the opposite of what the tier draws. The examples that would have shown it are `# doctest: +SKIP`
    blocks, so nothing executed them.
    """

    #: The named builders, each with the registered kind whose band decides where it draws.
    NAMED_LAYERS = [
        ("land", "land"),
        ("ocean", "ocean"),
        ("lakes", "lakes"),
        ("rivers", "rivers"),
        ("coastlines", "coastlines"),
        ("borders", "borders"),
    ]

    @pytest.mark.parametrize("builder, kind", NAMED_LAYERS)
    def test_the_prose_names_the_band_the_registry_files_it_in(self, builder, kind):
        """A docstring either says nothing about the band or says the one the kind is drawn in.

        Args:
            builder: The `InteractiveMap` method a caller reads.
            kind: The registered kind that builder adds, whose band the registry owns.

        Test scenario:
            Both band words are looked for, so a docstring cannot pass by naming neither the right one nor
            the wrong one while still claiming the opposite elsewhere in the same text.
        """
        from digitalearth.base.registry import band_of

        doc = (getattr(InteractiveMap, builder).__doc__ or "").lower()
        stated = tuple(word for word in ("underlay", "overlay") if word in doc)
        assert stated in ((), (band_of(kind),)), (
            f"{builder}() reads as {stated or 'nothing'}, but the registry draws {kind!r} "
            f"in the {band_of(kind)} band"
        )
