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
        assert out is not None


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
