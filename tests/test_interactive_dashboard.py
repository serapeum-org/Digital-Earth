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
        assert isinstance(
            app, pn.viewable.Viewable
        ), f"not a Panel Viewable: {type(app)}"

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
        assert (
            style.get("cmap") == "magma"
        ), f"cmap override not applied: {style.get('cmap')}"

    def test_override_with_no_values_returns_map_unchanged(self, m):
        out = m._render_with_overrides({})
        assert isinstance(
            out, hv.core.Dimensioned
        ), "no-override path must return the rendered map"

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
        assert m.save_app(str(out), widgets=("cmap",)) == str(out)
        assert (
            out.stat().st_size > 1_000
        ), "exported app should be a non-trivial HTML page"


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
        assert (
            captured["window"] is sentinel
        ), "the plotter render window must be forwarded"


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
        assert "geometry" not in list(
            table.value.columns
        ), "geometry column must be dropped"
        assert "fid" in list(table.value.columns)

    def test_share_off_server_returns_params(self, multi):
        """Off a running server (no session), pn.state.location is None, so share() returns the
        params it would sync rather than crashing."""
        assert pn.state.location is None, "test assumes no active Panel session"
        out = multi.share(params=("cmap", "extent"))
        assert out == ("cmap", "extent")
