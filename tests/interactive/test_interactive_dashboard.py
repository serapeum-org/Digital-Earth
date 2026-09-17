"""DI.4 — Panel dashboards + HTML export (+ cross-tier compose).

Asserts the dashboard is a Panel Viewable with the requested widgets, that the bound view re-renders
with widget overrides, that ``save_app`` writes a non-trivial standalone file, and that the cross-tier
PyVista pane degrades cleanly without the ``[3d]`` extra. Runs in the ``interactive`` pixi env.
"""

import pytest

from digitalearth.interactive import InteractiveMap

hv = pytest.importorskip("holoviews")
gv = pytest.importorskip("geoviews")
pn = pytest.importorskip("panel")


def _resolved_style(element) -> dict:
    """Read back the options HoloViews resolved for ``element`` (plot options under style options).

    Args:
        element: A HoloViews element taken from a composed, restyled object.

    Returns:
        dict: the element's resolved ``plot`` and ``style`` options, merged — ``cmap``/``alpha`` land in
        ``style`` and ``clim`` in ``plot``, and an override has to be read back from wherever it landed.
    """
    style = hv.Store.lookup_options("bokeh", element, "style").kwargs
    plot = hv.Store.lookup_options("bokeh", element, "plot").kwargs
    return {**plot, **style}


@pytest.fixture()
def m(dataset) -> InteractiveMap:
    """A Web-Mercator map carrying one raster layer."""
    return InteractiveMap().image(dataset)


@pytest.fixture()
def point_fc():
    """The point fixture as a pyramids FeatureCollection (for the vector-only override test)."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


class TestDashboard:
    """``dashboard`` — the Panel layout + reactive widgets."""

    def test_is_panel_viewable(self, m):
        app = m.dashboard()
        assert isinstance(app, pn.viewable.Viewable), (
            f"not a Panel Viewable: {type(app)}"
        )

    def test_requested_widgets_present(self, m):
        app = m.dashboard(widgets=("cmap", "alpha", "basemap"))
        names = {w.name for w in app.select(pn.widgets.Widget)}
        assert {"Colormap", "Opacity", "Basemap"} <= names, f"missing widgets: {names}"

    def test_unknown_widget_raises(self, m):
        with pytest.raises(ValueError, match="unknown dashboard widget"):
            m.dashboard(widgets=("bogus",))

    def test_title_renders_as_heading(self, m):
        app = m.dashboard(title="Discharge")
        markdowns = [p for p in app.select(pn.pane.Markdown)]
        assert any("Discharge" in p.object for p in markdowns), "title heading missing"

    def test_sidebar_vs_row_layout(self, m):
        sidebar = m.dashboard(sidebar=True, widgets=("cmap",))
        inline = m.dashboard(sidebar=False, widgets=("cmap",))
        assert isinstance(sidebar, pn.viewable.Viewable)
        assert isinstance(inline, pn.viewable.Viewable)

    def test_bound_view_applies_widget_overrides(self, m):
        """The bound render function restyles the map from widget values."""
        out = m._render_with_overrides({"cmap": "magma", "alpha": 0.3})
        style = hv.Store.lookup_options("bokeh", out, "style").kwargs
        assert style.get("cmap") == "magma", (
            f"cmap override not applied: {style.get('cmap')}"
        )

    def test_override_with_no_values_returns_map_unchanged(self, m):
        out = m._render_with_overrides({})
        assert isinstance(out, hv.core.Dimensioned), (
            "no-override path must return the rendered map"
        )

    def test_override_on_vector_only_map_does_not_raise(self, point_fc):
        """cmap/alpha on a vector-only map must not raise — HoloViews applies them where they match,
        leaving a points-only map unchanged (M2: no broad except-swallow needed)."""
        vmap = InteractiveMap().points(point_fc)
        out = vmap._render_with_overrides({"cmap": "magma", "alpha": 0.3})
        assert isinstance(out, hv.core.Dimensioned), "must return the map, not raise"


class TestServeAndExport:
    """``serve`` / ``save_app`` — live and offline outputs."""

    def test_serve_returns_servable(self, m):
        app = m.serve(widgets=("cmap",))
        assert isinstance(app, pn.viewable.Viewable)

    def test_save_app_writes_standalone_file(self, m, tmp_path):
        out = tmp_path / "app.html"
        assert m.save_app(str(out), widgets=("cmap",)) == out
        assert out.stat().st_size > 1_000, (
            "exported app should be a non-trivial HTML page"
        )


class TestCrossTierPane:
    """``cross_tier_pane`` — embed an M1 Scene3D, degrade cleanly without [3d]/VTK."""

    def test_missing_vtk_degrades_with_actionable_error(self, m, monkeypatch):
        """With vtk unavailable (find_spec → None), a clear ImportError naming [3d] surfaces."""
        import digitalearth.interactive.dashboard as dash

        monkeypatch.setattr(dash, "find_spec", lambda name: None, raising=False)

        class _FakeScene:
            plotter = object()

        with pytest.raises(ImportError, match=r"digitalearth\[3d\]"):
            m.cross_tier_pane(_FakeScene())

    def test_pane_built_when_vtk_present(self, m, monkeypatch):
        """When vtk is importable, the helper hands the render window to panel.pane.VTK."""
        import digitalearth.interactive.dashboard as dash

        monkeypatch.setattr(dash, "find_spec", lambda name: object(), raising=False)
        captured = {}

        def _fake_vtk(window, **kwargs):
            captured["window"] = window
            return "vtk-pane"

        monkeypatch.setattr(pn.pane, "VTK", _fake_vtk)

        sentinel = object()

        class _FakePlotter:
            ren_win = sentinel

        class _FakeScene:
            plotter = _FakePlotter()

        assert m.cross_tier_pane(_FakeScene()) == "vtk-pane"
        assert captured["window"] is sentinel, (
            "the plotter render window must be forwarded"
        )


class TestLayerControlAndTable:
    """DI.13 — layer manager, attribute table, URL share."""

    @pytest.fixture()
    def multi(self, dataset):
        from pyramids.feature import FeatureCollection

        fc = FeatureCollection.read_file("tests/data/points.geojson")
        return InteractiveMap().image(dataset).points(fc)

    def test_layer_control_is_viewable_with_toggles(self, multi):
        panel_obj = multi.layer_control()
        assert isinstance(panel_obj, pn.viewable.Viewable)
        groups = panel_obj.select(pn.widgets.CheckBoxGroup)
        assert groups and len(groups[0].options) == 2, "one toggle per layer expected"

    def test_layer_control_has_opacity_and_basemap(self, multi):
        panel_obj = multi.layer_control(opacity=True, basemap_switch=True)
        assert panel_obj.select(pn.widgets.FloatSlider), "opacity slider missing"
        assert panel_obj.select(pn.widgets.Select), "basemap switch missing"

    def test_layer_control_without_layers_raises(self):
        with pytest.raises(ValueError, match="at least one layer"):
            InteractiveMap().layer_control()

    def test_compose_visible_all_hidden_is_blank_overlay(self, multi):
        """With nothing shown the layer-control view is a blank overlay (not an error)."""
        out = multi._compose_visible_layers([], op=1.0)
        assert isinstance(out, hv.Overlay) and len(out) == 0

    def test_compose_visible_single_layer(self, multi):
        out = multi._compose_visible_layers(["0: Image"], op=0.5)
        assert isinstance(out, hv.core.Dimensioned)

    def test_compose_visible_multiple_layers(self, multi):
        out = multi._compose_visible_layers(["0: Image", "1: Points"], op=0.7)
        assert isinstance(out, hv.Overlay) and len(out) == 2

    def test_attribute_table_is_tabulator_without_geometry(self):
        from pyramids.feature import FeatureCollection

        fc = FeatureCollection.read_file("tests/data/points.geojson")
        table = InteractiveMap().attribute_table(fc)
        assert isinstance(table, pn.widgets.Tabulator)
        assert "geometry" not in list(table.value.columns), (
            "geometry column must be dropped"
        )
        assert "fid" in list(table.value.columns)

    def test_share_off_server_returns_params(self, multi):
        """Off a running server (no session), pn.state.location is None, so share() returns the
        params it would sync rather than crashing."""
        assert pn.state.location is None, "test assumes no active Panel session"
        out = multi.share(params=("cmap", "extent"))
        assert out == ("cmap", "extent")


class TestBasemapWidgetIsWired:
    """#244 — the basemap widgets actually change the map (dashboard *and* layer control)."""

    @pytest.fixture
    def multi(self, dataset, point_fc):
        return InteractiveMap().image(dataset).points(point_fc)

    def test_dashboard_basemap_override_changes_the_tile_layer(self, m):
        """Picking a provider prepends that provider's tiles, not the same map twice."""
        dark = m._render_with_overrides({"basemap": "CartoDark"})
        osm = m._render_with_overrides({"basemap": "OSM"})
        dark_tiles = [e for e in dark if type(e).__name__ in ("WMTS", "Tiles")]
        osm_tiles = [e for e in osm if type(e).__name__ in ("WMTS", "Tiles")]
        assert dark_tiles, "the CartoDark value must add a tile layer"
        assert osm_tiles, "the OSM value must add a tile layer"
        assert dark_tiles[0].data != osm_tiles[0].data, (
            f"the provider must change the tile source: {dark_tiles[0].data}"
        )

    def test_dashboard_basemap_replaces_an_existing_basemap(self, dataset):
        """On a map that already called tiles(), the widget swaps the basemap instead of stacking."""
        built = InteractiveMap().image(dataset).tiles("CartoLight")
        out = built._render_with_overrides({"basemap": "OSM"})
        tiles = [e for e in out if type(e).__name__ in ("WMTS", "Tiles")]
        assert len(tiles) == 1, f"exactly one basemap expected, got {len(tiles)}"
        assert "openstreetmap" in tiles[0].data.lower(), (
            f"the chosen provider must win: {tiles[0].data}"
        )

    def test_dashboard_basemap_offers_the_shared_provider_list(self, m):
        """One provider list backs the dashboard widget and the layer-control switch."""
        from digitalearth.interactive.dashboard import _BASEMAP_CHOICES

        app = m.dashboard(widgets=("basemap",))
        selects = [w for w in app.select(pn.widgets.Select) if w.name == "Basemap"]
        assert selects, "the basemap Select is missing"
        assert list(selects[0].options) == list(_BASEMAP_CHOICES), (
            f"dashboard must use the shared provider list: {selects[0].options}"
        )

    def test_dashboard_basemap_refused_on_a_non_mercator_map(self, dataset):
        """Bokeh renders tiles in 3857 only, so the widget is refused rather than misaligned."""
        non_mercator = InteractiveMap(crs=4326).image(dataset)
        with pytest.raises(ValueError, match="Web-Mercator"):
            non_mercator.dashboard(widgets=("basemap",))

    def test_tiles_accept_web_mercator_spelled_as_a_string(self):
        """``crs="EPSG:3857"`` is Web Mercator, so tiles are drawn rather than refused (#287).

        Test scenario:
            The check compared ``self.crs != 3857``; the constructor keeps the string, so the map was refused
            for not being in the CRS it is in.
        """
        m = InteractiveMap(crs="EPSG:3857").tiles()
        assert len(m.layers) == 1, m.layers

    def test_the_basemap_select_is_built_for_web_mercator_spelled_as_a_string(self):
        """The layer-control basemap switch reads the display CRS by meaning too (#287)."""
        select = InteractiveMap(crs="EPSG:3857")._basemap_select(pn, requested=True)
        assert isinstance(select, pn.widgets.Select), type(select)

    def test_layer_control_binds_its_basemap_select(self, multi):
        """The layer-control Select is passed to pn.bind, not merely laid out."""
        panel_obj = multi.layer_control(basemap_switch=True)
        bound = panel_obj.select(pn.param.ParamFunction)[0].object
        assert "basemap" in getattr(bound, "_dinfo", {}).get("kw", {}), (
            "the basemap Select must be bound to the layer-control view"
        )

    def test_layer_control_compose_honours_the_basemap(self, multi):
        """The bound view really composes the chosen provider under the visible layers."""
        out = multi._compose_visible_layers(["0: Image"], op=0.5, basemap="OSM")
        tiles = [e for e in out if type(e).__name__ in ("WMTS", "Tiles")]
        assert tiles, (
            "the layer-control basemap must reach the overlay: "
            f"{[type(e).__name__ for e in out]}"
        )
        assert "openstreetmap" in tiles[0].data.lower(), (
            f"the chosen provider must be the one composed: {tiles[0].data}"
        )

    def test_unprompted_switch_follows_the_display_crs(self, dataset, point_fc):
        """Left at its default the switch is furniture: offered on 3857, dropped elsewhere.

        Args:
            dataset: The raster fixture.
            point_fc: The point fixture.

        Test scenario:
            Nobody asked for a basemap here — ``layer_control()`` offers one unprompted — so a map that
            cannot carry tiles drops the widget (and logs why) instead of refusing the whole control.
        """
        mercator = InteractiveMap().image(dataset).points(point_fc).layer_control()
        assert [w for w in mercator.select(pn.widgets.Select) if w.name == "Basemap"], (
            "a Web-Mercator map must still get the switch by default"
        )
        other = InteractiveMap(crs=4326).image(dataset).points(point_fc).layer_control()
        assert not [
            w for w in other.select(pn.widgets.Select) if w.name == "Basemap"
        ], "a non-Mercator map must not offer a basemap switch"

    def test_explicit_switch_is_refused_off_mercator(self, dataset, point_fc):
        """``basemap_switch=True`` is a request, and an impossible request is refused, not dropped.

        Args:
            dataset: The raster fixture.
            point_fc: The point fixture.

        Test scenario:
            The two halves of this module used to answer the same condition differently — ``dashboard``
            raised, ``layer_control`` logged and dropped — so a caller who asked outright got a control
            missing a widget and no error. The policy is now keyed on whether the basemap was asked for.
        """
        non_mercator = InteractiveMap(crs=4326).image(dataset).points(point_fc)
        with pytest.raises(ValueError, match="Web-Mercator") as excinfo:
            non_mercator.layer_control(basemap_switch=True)
        assert "4326" in str(excinfo.value), (
            f"the refusal must name the display CRS: {excinfo.value}"
        )

    def test_false_never_builds_the_switch(self, multi):
        """``False`` still means never, on a Web-Mercator map too.

        Args:
            multi: A Web-Mercator map carrying a raster and a point layer.

        Test scenario:
            The tri-state default must not turn the opt-out into "offer it anyway".
        """
        panel_obj = multi.layer_control(basemap_switch=False)
        assert not [
            w for w in panel_obj.select(pn.widgets.Select) if w.name == "Basemap"
        ], "basemap_switch=False must not build the switch"

    def test_both_entry_points_refuse_an_explicit_request_alike(self, dataset):
        """One condition, one policy: both entry points refuse, and both name the CRS.

        Args:
            dataset: The raster fixture.

        Test scenario:
            This is the finding itself — the same "not Web Mercator" condition had opposite answers in
            one module. Asserting the two messages agree is what keeps them from drifting apart again.
        """
        non_mercator = InteractiveMap(crs=4326).image(dataset)
        messages = []
        for call in (
            lambda: non_mercator.dashboard(widgets=("basemap",)),
            lambda: non_mercator.layer_control(basemap_switch=True),
        ):
            with pytest.raises(ValueError) as excinfo:
                call()
            messages.append(str(excinfo.value))
        assert all("crs=4326" in message for message in messages), messages
        assert all("EPSG:3857" in message for message in messages), messages


class TestBothOverridePathsRestyleTheSameElements:
    """M13 — the dashboard's overrides and the layer control's slider cover one element set, not two."""

    @pytest.fixture
    def mixed(self, dataset) -> InteractiveMap:
        """A map carrying one of each element type an override reaches: ``Image``, ``QuadMesh``, ``RGB``."""
        return (
            InteractiveMap()
            .image(dataset)
            .quadmesh(dataset)
            .rgb(dataset, bands=(1, 1, 1))
        )

    def test_dashboard_overrides_reach_quadmesh_and_rgb(self, mixed):
        """The cmap/alpha widgets restyle ``Image`` **and** ``QuadMesh``, and set ``RGB``'s alpha.

        Args:
            mixed: The three-element map.

        Test scenario:
            This path restyled ``Image``/``QuadMesh`` and left ``RGB`` alone, so an RGB layer sat at
            full opacity while the slider moved. ``cmap`` genuinely cannot apply to ``RGB`` — three
            colour channels, no scalar to map — so the assertion is alpha everywhere, cmap where it
            means something.
        """
        out = mixed._render_with_overrides({"cmap": "magma", "alpha": 0.3})
        styled = {type(element).__name__: _resolved_style(element) for element in out}
        assert set(styled) == {"Image", "QuadMesh", "RGB"}, sorted(styled)
        for name in ("Image", "QuadMesh"):
            assert styled[name].get("cmap") == "magma", (
                f"{name} ignored the colormap selector: {styled[name].get('cmap')}"
            )
        assert all(styled[name].get("alpha") == 0.3 for name in styled), {
            name: options.get("alpha") for name, options in styled.items()
        }
        assert styled["RGB"].get("cmap") is None, (
            f"an RGB composite has no scalar to colour-map: {styled['RGB'].get('cmap')}"
        )

    def test_layer_control_opacity_reaches_quadmesh_and_rgb(self, mixed):
        """The layer-control slider restyles the same three, ``QuadMesh`` included.

        Args:
            mixed: The three-element map.

        Test scenario:
            This path restyled ``Image`` and ``RGB``, so a ``QuadMesh`` layer ignored the opacity slider
            while the widget still moved — the mirror image of the dashboard's gap.
        """
        labels = [
            f"{index}: {type(layer).__name__}"
            for index, layer in enumerate(mixed.layers)
        ]
        out = mixed._compose_visible_layers(labels, op=0.25)
        styled = {type(element).__name__: _resolved_style(element) for element in out}
        assert set(styled) == {"Image", "QuadMesh", "RGB"}, sorted(styled)
        assert all(styled[name].get("alpha") == 0.25 for name in styled), {
            name: options.get("alpha") for name, options in styled.items()
        }

    def test_the_two_paths_restyle_the_same_element_set(self, mixed):
        """Whatever the element set is, both widgets must move it — that is the finding.

        Args:
            mixed: The three-element map.

        Test scenario:
            Comparing the two paths' restyled sets directly is what makes the drift impossible to
            reintroduce quietly: one path gaining or losing an element type now fails here, not in a
            widget that moves without changing the picture.
        """
        labels = [
            f"{index}: {type(layer).__name__}"
            for index, layer in enumerate(mixed.layers)
        ]
        dashboard_alphas = {
            type(element).__name__: _resolved_style(element).get("alpha")
            for element in mixed._render_with_overrides({"alpha": 0.4})
        }
        control_alphas = {
            type(element).__name__: _resolved_style(element).get("alpha")
            for element in mixed._compose_visible_layers(labels, op=0.4)
        }
        assert dashboard_alphas == control_alphas, (
            f"the two paths restyle different elements: {dashboard_alphas} vs {control_alphas}"
        )
        assert dashboard_alphas == {"Image": 0.4, "QuadMesh": 0.4, "RGB": 0.4}, (
            dashboard_alphas
        )


class TestInertFlagsAreRefused:
    """#242 / #243 — flags that are not implemented are refused, not silently ignored."""

    @pytest.fixture
    def multi(self, dataset, point_fc):
        return InteractiveMap().image(dataset).points(point_fc)

    def test_layer_control_reorder_true_raises(self, multi):
        """#242 — ``reorder=True`` is refused (reordering needs stable layer identity)."""
        with pytest.raises(NotImplementedError, match="reorder"):
            multi.layer_control(reorder=True)

    def test_layer_control_reorder_defaults_to_false(self, multi):
        """The default must not promise reordering."""
        import inspect

        default = inspect.signature(type(multi).layer_control).parameters["reorder"]
        assert default.default is False, (
            f"reorder must default to False, got {default.default!r}"
        )
        assert isinstance(multi.layer_control(), pn.viewable.Viewable), (
            "the default call must keep working unchanged"
        )

    def test_attribute_table_linked_true_raises(self, point_fc):
        """#243 — ``linked=True`` is refused (two-way linking needs link_selections)."""
        fresh_map = InteractiveMap()
        with pytest.raises(NotImplementedError, match="linked"):
            fresh_map.attribute_table(point_fc, linked=True)

    def test_attribute_table_linked_defaults_to_false(self, point_fc):
        """The default must not promise linking, and must still return the read-only table."""
        import inspect

        default = inspect.signature(InteractiveMap.attribute_table).parameters["linked"]
        assert default.default is False, (
            f"linked must default to False, got {default.default!r}"
        )
        table = InteractiveMap().attribute_table(point_fc)
        assert isinstance(table, pn.widgets.Tabulator), (
            f"the unlinked table must stay a Tabulator, got {type(table)}"
        )
        assert table.disabled, "the unlinked table must stay read-only"


class TestOverridesMergeOverRecordedStyle:
    """#241 — widget overrides are merged over the recorded style, not applied blind."""

    def test_recorded_style_is_read_back_for_the_colour_mapped_layers(
        self, dataset, point_fc
    ):
        """The override base comes from the map's own style record, raster layers only."""
        m = InteractiveMap().image(dataset, cmap="viridis", clim=(0.0, 10.0))
        m.points(point_fc, value_column="fid", cmap="magma")
        recorded = m._recorded_overridable_style()
        assert recorded["cmap"] == "viridis", (
            f"the raster layer's recorded cmap must drive the merge: {recorded}"
        )
        assert tuple(recorded["clim"]) == (0.0, 10.0), (
            f"the recorded clim must be part of the merge base: {recorded}"
        )

    def test_override_wins_over_the_recorded_style(self, dataset):
        """The widget value replaces the recorded entry; the untouched entries survive it."""
        m = InteractiveMap().image(dataset, cmap="viridis", clim=(0.0, 10.0))
        out = m._render_with_overrides({"cmap": "magma"})
        style = hv.Store.lookup_options("bokeh", out, "style").kwargs
        plot = hv.Store.lookup_options("bokeh", out, "plot").kwargs
        assert style.get("cmap") == "magma", "the override must win"
        assert tuple(plot.get("clim", ())) == (0.0, 10.0), (
            f"the recorded clim must survive the override: {plot.get('clim')}"
        )
