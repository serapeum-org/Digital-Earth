"""DW.0 — packaging + ``WebMap`` skeleton + display-CRS plumbing.

Covers the web tier's foundation: the lazy engine import (the package imports without the ``web`` extra;
engine-touching methods raise an actionable ``ImportError``), the layer registry, the render/save lifecycle
over the ``maplibre`` widget, the ``_to_display_source`` reproject-through-pyramids choke point, and the proof
that the DX.3 import guard now covers ``src/digitalearth/web/``.

Engine-dependent tests ``importorskip`` maplibre — they run in the ``web`` pixi env
(``pixi run -e web test-web``); everything else runs in the lean ``dev`` env too.
"""

import importlib
import pathlib
import sys

import pytest

from digitalearth.sources.source import Source
from digitalearth.web import WebMap
from digitalearth.web.base import _require_maplibre, _resolve_style
from tests.test_no_competitor_imports import (
    FORBIDDEN,
    _top_level_imports,
    test_tiers_import_no_gis_competitor,
)

_WEB_ROOT = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "digitalearth" / "web"
)


class TestLazyImport:
    """The package and constructor work without the engine; builders fail actionably without it."""

    def test_package_imports_without_engine_loaded(self):
        """``import digitalearth.web`` must not itself import maplibre/lonboard.

        Only **module-top-level** statements count: the engine imports inside ``_require_maplibre``
        (and other methods) are exactly the lazy pattern the tier promises, so the guard's whole-tree
        AST walk would be the wrong tool here.
        """
        import ast

        engine = {"maplibre", "lonboard"}
        for mod in _WEB_ROOT.rglob("*.py"):
            tree = ast.parse(mod.read_text(encoding="utf-8"))
            top_level = set()
            for node in tree.body:  # module body only — function-local imports stay lazy
                if isinstance(node, ast.Import):
                    top_level.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    top_level.add(node.module.split(".")[0])
            assert not top_level & engine, (
                f"{mod.name} imports the MapLibre engine at module top — the tier promises a lazy "
                "engine import (engine imports belong inside methods via _require_maplibre)"
            )

    def test_constructing_needs_no_engine(self):
        m = WebMap(center=(8.0, 47.0), zoom=5, style="light", crs=4326, height=300)
        assert m.layers == []
        assert (m.center, m.zoom, m.style, m.crs, m.height) == (
            (8.0, 47.0),
            5,
            "light",
            4326,
            300,
        )

    def test_missing_engine_raises_actionable_error(self, monkeypatch):
        """With maplibre unimportable, the lazy import points at the install command."""
        monkeypatch.setitem(sys.modules, "maplibre", None)  # makes `import maplibre` raise
        with pytest.raises(ImportError, match=r"digitalearth\[web\]"):
            _require_maplibre()

    def test_render_without_engine_raises_actionable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "maplibre", None)
        with pytest.raises(ImportError, match=r"digitalearth\[web\]"):
            WebMap().render()

    def test_repr_mimebundle_degrades_gracefully_without_engine(self, monkeypatch):
        """A bare repr in a notebook must not raise when the extra is missing."""
        monkeypatch.setitem(sys.modules, "maplibre", None)
        assert WebMap()._repr_mimebundle_() == {}


class TestImportGuardCoversWeb:
    """DX.3 negative fixture: the guard now polices ``web/``."""

    def test_temp_forbidden_import_fails_the_guard(self):
        """Dropping an xarray-importing module under ``web/`` makes the guard test fail."""
        bad = _WEB_ROOT / "_tmp_guard_negative_fixture.py"
        bad.write_text("import xarray\n", encoding="utf-8")
        try:
            with pytest.raises(AssertionError, match="xarray"):
                test_tiers_import_no_gis_competitor()
        finally:
            bad.unlink()
        test_tiers_import_no_gis_competitor()  # tree is clean again

    def test_web_modules_are_clean(self):
        offenders = {
            mod.name: sorted(FORBIDDEN & set(_top_level_imports(mod)))
            for mod in _WEB_ROOT.rglob("*.py")
            if FORBIDDEN & set(_top_level_imports(mod))
        }
        assert offenders == {}


class TestDisplaySource:
    """``_to_display_source`` — the single reproject-through-pyramids choke point."""

    def test_reprojects_to_display_crs(self, dataset):
        m = WebMap(crs=3857)
        assert dataset.epsg != 3857, "fixture must start in a non-display CRS for this test"
        src = m._to_display_source(dataset)
        assert src.crs == 3857

    def test_same_crs_passes_through_without_warp(self, dataset, monkeypatch):
        m = WebMap(crs=dataset.epsg)

        def _boom(*a, **k):  # pragma: no cover - only fires on regression
            raise AssertionError(
                "to_crs must not be called when data is already in the display CRS"
            )

        monkeypatch.setattr(type(dataset), "to_crs", _boom)
        src = m._to_display_source(dataset)
        assert src.crs == dataset.epsg

    def test_source_passes_through_untouched(self):
        import numpy as np

        from digitalearth.sources.dimension import DimensionInfo

        src = Source(
            DimensionInfo(np.zeros((2, 2)), "z"),
            DimensionInfo(np.arange(2.0), "x"),
            DimensionInfo(np.arange(2.0), "y"),
            crs=4326,
        )
        assert WebMap(crs=3857)._to_display_source(src) is src


class TestRegistryAndRender:
    """The layer registry composes into one MapLibre widget (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_add_layer_chains(self):
        m = WebMap()
        out = m.add_layer("a").add_layer("b")
        assert out is m
        assert m.layers == ["a", "b"]

    def test_render_returns_maplibre_widget(self):
        from maplibre.ipywidget import MapWidget

        widget = WebMap(center=(0.0, 0.0), zoom=3).render()
        assert isinstance(widget, MapWidget)

    def test_empty_map_still_renders_a_widget(self):
        from maplibre.ipywidget import MapWidget

        assert isinstance(WebMap().render(), MapWidget)

    def test_height_lands_on_the_widget(self):
        widget = WebMap(height=321).render()
        # maplibre stores height as a CSS length string ("321px").
        assert str(widget.height).startswith("321")

    def test_save_html_writes_a_file(self, tmp_path):
        out = tmp_path / "m.html"
        assert WebMap(center=(0.0, 0.0), zoom=2).save(str(out)) == str(out)
        assert out.stat().st_size > 1_000
        assert "maplibre" in out.read_text(encoding="utf-8").lower()

    def test_show_returns_the_widget(self, monkeypatch):
        from maplibre.ipywidget import MapWidget

        shown = []
        import IPython.display

        monkeypatch.setattr(IPython.display, "display", shown.append)
        widget = WebMap().show()
        assert isinstance(widget, MapWidget)
        assert shown == [widget], "show() must push the widget through IPython display once"

    def test_repr_mimebundle_delegates_to_widget(self):
        bundle = WebMap()._repr_mimebundle_()
        # ipywidgets returns the (data, metadata) tuple form of the protocol; older hooks return a bare dict.
        data = bundle[0] if isinstance(bundle, tuple) else bundle
        assert isinstance(data, dict) and data, "expected the widget's non-empty mimebundle"
        assert "application/vnd.jupyter.widget-view+json" in data


class TestUtf8Shim:
    """The Windows-cp1252 UTF-8 workaround (`_patch_maplibre_html_encoding` + UTF-8 `save`)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("maplibre")

    def test_shim_rebinds_every_maplibre_submodule(self, monkeypatch):
        """On a platform that needs it, the shim rebinds the reader on *every* maplibre submodule.

        Simulates the cp1252 read failure so the patch installs regardless of the host OS, and injects a
        fake maplibre submodule that imported ``read_internal_file`` by name — it must be rebound too, so no
        stale bare-``open`` reader survives.
        """
        import builtins
        import types

        import maplibre._utils as _utils

        from digitalearth.web import base

        real_open = builtins.open

        def _sentinel(*args, **kwargs):  # a fresh, un-shimmed reader (no _digitalearth_utf8 flag)
            raise AssertionError("this reader should have been rebound by the shim")

        fake = types.ModuleType("maplibre._fake_reader_holder")
        fake.read_internal_file = _sentinel
        monkeypatch.setitem(sys.modules, "maplibre._fake_reader_holder", fake)
        # start from an un-patched reader (monkeypatch auto-restores after the test)
        monkeypatch.setattr(_utils, "read_internal_file", _sentinel)

        def _fake_open(file, *args, **kwargs):  # fail the probe like Windows cp1252 does
            if "pywidget.js" in str(file) and not kwargs.get("encoding"):
                raise UnicodeDecodeError("charmap", b"\x9d", 0, 1, "simulated cp1252")
            return real_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", _fake_open)

        base._patch_maplibre_html_encoding()

        shim = _utils.read_internal_file
        assert getattr(shim, "_digitalearth_utf8", False), "the canonical _utils reader must be shimmed"
        assert fake.read_internal_file is shim, "a submodule holding the reader by name must be rebound too"

    def test_save_writes_utf8_non_ascii_title(self, tmp_path):
        """`save` writes the HTML as UTF-8 so a non-ASCII title round-trips (the write-side cp1252 fix)."""
        out = tmp_path / "u.html"
        WebMap(center=(0.0, 0.0), zoom=2).save(str(out), title="façade ′ café —")
        assert "façade ′ café —" in out.read_text(encoding="utf-8"), "unicode title must survive the write"


class TestStyleResolution:
    """``_resolve_style`` maps aliases to CartoCDN URLs and passes URLs/dicts through (no engine)."""

    @pytest.mark.parametrize(
        "alias, slug",
        [("dark", "dark-matter"), ("light", "positron"), ("voyager", "voyager")],
    )
    def test_alias_resolves_to_cartocdn_url(self, alias, slug):
        url = _resolve_style(alias)
        assert url == f"https://basemaps.cartocdn.com/gl/{slug}-gl-style/style.json"

    def test_url_and_dict_pass_through(self):
        url = "https://example.com/style.json"
        assert _resolve_style(url) is url
        spec = {"version": 8, "layers": []}
        assert _resolve_style(spec) is spec


class TestConstructionDefaults:
    """Constructor defaults and the reproject predicate (no engine needed)."""

    def test_default_configuration(self):
        """A bare ``WebMap()`` defaults to EPSG:4326 (MapLibre lon/lat), zoom 2, dark style, height 500."""
        m = WebMap()
        assert (m.center, m.zoom, m.style, m.crs, m.height) == (None, 2, "dark", 4326, 500)
        assert m.layers == []

    @pytest.mark.parametrize(
        "crs, data_epsg, expected",
        [
            (3857, 3857, False),  # already in the display CRS
            (3857, 4326, True),  # differing EPSG -> reproject
            ("ESRI:54009", 3857, True),  # string display CRS -> always reproject
            (3857, None, True),  # no epsg attribute -> reproject path
        ],
    )
    def test_needs_reproject_matrix(self, crs, data_epsg, expected):
        """``_needs_reproject`` only short-circuits on an exact int-EPSG match."""

        class _Data:
            pass

        data = _Data()
        if data_epsg is not None:
            data.epsg = data_epsg
        assert WebMap(crs=crs)._needs_reproject(data) is expected

    def test_composition_includes_all_mixins(self):
        """``WebMap`` composes the base plus all seven capability mixins."""
        from digitalearth.web.base import WebMapBase
        from digitalearth.web.bigdata import BigDataMixin
        from digitalearth.web.decoration import DecorationMixin
        from digitalearth.web.export import ExportMixin
        from digitalearth.web.raster import RasterMixin
        from digitalearth.web.temporal import TemporalMixin
        from digitalearth.web.threed import ThreeDMixin
        from digitalearth.web.vector import VectorMixin

        mro = WebMap.__mro__
        for cls in (
            WebMapBase,
            RasterMixin,
            VectorMixin,
            BigDataMixin,
            ThreeDMixin,
            TemporalMixin,
            DecorationMixin,
            ExportMixin,
        ):
            assert cls in mro, f"{cls.__name__} missing from WebMap MRO"


def test_reimport_is_stable():
    """The package re-imports cleanly (no import-time engine side effects)."""
    mod = importlib.reload(importlib.import_module("digitalearth.web"))
    assert hasattr(mod, "WebMap")
