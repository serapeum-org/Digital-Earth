"""DI.0 — packaging + ``InteractiveMap`` skeleton + display-CRS plumbing.

Covers the tier's foundation: the lazy engine import (the package imports without the ``interactive``
extra; engine-touching methods raise an actionable ``ImportError``), the element/overlay registry, the
render/save lifecycle, the ``_to_display_source`` reproject-through-pyramids choke point, and the proof
that the DX.3 import guard already covers ``src/digitalearth/interactive/``.

Engine-dependent tests ``importorskip`` geoviews — they run in the ``interactive`` pixi env
(``pixi run -e interactive test-interactive``); everything else runs in the lean ``dev`` env too.
"""

import importlib
import pathlib
import sys

import pytest

from digitalearth.interactive import InteractiveMap
from digitalearth.interactive.base import _require_holoviz
from digitalearth.sources.source import Source
from tests.test_no_competitor_imports import (
    FORBIDDEN,
    _top_level_imports,
    test_tiers_import_no_gis_competitor,
)

_INTERACTIVE_ROOT = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "digitalearth" / "interactive"
)


class TestLazyImport:
    """The package and constructor work without the engine; builders fail actionably without it."""

    def test_package_imports_without_engine_loaded(self):
        """``import digitalearth.interactive`` must not itself import geoviews/holoviews.

        Only **module-top-level** statements count: the engine imports inside ``_require_holoviz``
        (and other methods) are exactly the lazy pattern the tier promises, so the guard's
        whole-tree AST walk would be the wrong tool here.
        """
        import ast

        engine = {"geoviews", "holoviews", "datashader", "panel", "hvplot"}
        for mod in _INTERACTIVE_ROOT.rglob("*.py"):
            tree = ast.parse(mod.read_text(encoding="utf-8"))
            top_level = set()
            for (
                node
            ) in tree.body:  # module body only — function-local imports stay lazy
                if isinstance(node, ast.Import):
                    top_level.update(alias.name.split(".")[0] for alias in node.names)
                elif (
                    isinstance(node, ast.ImportFrom) and node.module and node.level == 0
                ):
                    top_level.add(node.module.split(".")[0])
            assert not top_level & engine, (
                f"{mod.name} imports the HoloViz engine at module top — the tier promises a lazy "
                "engine import (engine imports belong inside methods via _require_holoviz)"
            )

    def test_constructing_needs_no_engine(self):
        m = InteractiveMap(crs=4326, width=300, height=200, title="t")
        assert m.layers == []
        assert (m.crs, m.width, m.height, m.title) == (4326, 300, 200, "t")

    def test_missing_engine_raises_actionable_error(self, monkeypatch):
        """With geoviews unimportable, the lazy import points at the install command."""
        monkeypatch.setitem(
            sys.modules, "geoviews", None
        )  # makes `import geoviews` raise
        with pytest.raises(ImportError, match=r"digitalearth\[interactive\]"):
            _require_holoviz()

    def test_render_without_engine_raises_actionable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "geoviews", None)
        with pytest.raises(ImportError, match=r"digitalearth\[interactive\]"):
            InteractiveMap().render()

    def test_repr_mimebundle_degrades_gracefully_without_engine(self, monkeypatch):
        """A bare repr in a notebook must not raise when the extra is missing."""
        monkeypatch.setitem(sys.modules, "geoviews", None)
        assert InteractiveMap()._repr_mimebundle_() == {}


class TestImportGuardCoversInteractive:
    """DX.3 negative fixture: the M1 guard already polices ``interactive/``."""

    def test_temp_forbidden_import_fails_the_guard(self):
        """Dropping an xarray-importing module under ``interactive/`` makes the guard test fail."""
        bad = _INTERACTIVE_ROOT / "_tmp_guard_negative_fixture.py"
        bad.write_text("import xarray\n", encoding="utf-8")
        try:
            with pytest.raises(AssertionError, match="xarray"):
                test_tiers_import_no_gis_competitor()
        finally:
            bad.unlink()
        test_tiers_import_no_gis_competitor()  # tree is clean again

    def test_interactive_modules_are_clean(self):
        offenders = {
            mod.name: sorted(FORBIDDEN & set(_top_level_imports(mod)))
            for mod in _INTERACTIVE_ROOT.rglob("*.py")
            if FORBIDDEN & set(_top_level_imports(mod))
        }
        assert offenders == {}


class TestDisplaySource:
    """``_to_display_source`` — the single reproject-through-pyramids choke point."""

    def test_reprojects_to_display_crs(self, dataset):
        m = InteractiveMap(crs=3857)
        assert (
            dataset.epsg != 3857
        ), "fixture must start in a non-display CRS for this test"
        src = m._to_display_source(dataset)
        assert src.crs == 3857

    def test_same_crs_passes_through_without_warp(self, dataset, monkeypatch):
        m = InteractiveMap(crs=dataset.epsg)

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
        assert InteractiveMap(crs=3857)._to_display_source(src) is src


class TestRegistryAndRender:
    """The element registry composes into one HoloViews object (engine required)."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("geoviews")

    def test_add_element_chains(self):
        import holoviews as hv

        m = InteractiveMap()
        out = m.add_element(hv.Points([])).add_element(hv.Points([]))
        assert out is m
        assert len(m.layers) == 2

    def test_empty_render_is_blank_overlay(self):
        import holoviews as hv

        obj = InteractiveMap().render()
        assert isinstance(obj, hv.core.Dimensioned)
        assert isinstance(obj, hv.Overlay) and len(obj) == 0

    def test_single_layer_renders_as_the_element(self):
        import holoviews as hv

        el = hv.Points([(0, 0)])
        assert InteractiveMap().add_element(el).render() is el

    def test_layers_overlay_in_add_order(self):
        import holoviews as hv

        first, second = hv.Points([(0, 0)]), hv.Points([(1, 1)])
        obj = InteractiveMap().add_element(first).add_element(second).render()
        assert isinstance(obj, hv.Overlay)
        assert list(obj) == [first, second]

    def test_save_html_writes_selfcontained_page(self, tmp_path):
        import holoviews as hv

        out = tmp_path / "m.html"
        m = InteractiveMap().add_element(hv.Points([(0.0, 0.0), (1.0, 1.0)]))
        assert m.save(str(out)) == str(out)
        assert out.stat().st_size > 1_000

    def test_save_png_via_matplotlib_backend(self, tmp_path):
        import holoviews as hv

        out = tmp_path / "m.png"
        InteractiveMap().add_element(hv.Points([(0.0, 0.0), (1.0, 1.0)])).save(str(out))
        assert out.exists() and out.stat().st_size > 0


class TestShowAndRepr:
    """``show()`` display behaviour and the notebook ``_repr_mimebundle_`` hook."""

    @pytest.fixture(autouse=True)
    def _need_engine(self):
        pytest.importorskip("geoviews")

    def test_show_returns_rendered_object_and_displays(self, monkeypatch):
        """``show()`` returns the composed object and pushes it through IPython display.

        Test scenario:
            With IPython importable, ``show()`` must call ``IPython.display.display`` exactly
            once with the rendered object, and return that object.
        """
        import holoviews as hv

        shown = []
        import IPython.display

        monkeypatch.setattr(IPython.display, "display", shown.append)
        el = hv.Points([(0, 0)])
        out = InteractiveMap().add_element(el).show()
        assert out is el, "show() must return the rendered object"
        assert shown == [
            el
        ], f"display() should receive the rendered object once, got {shown}"

    def test_show_without_ipython_still_returns_object(self, monkeypatch):
        """``show()`` degrades to returning the object when IPython is absent.

        Test scenario:
            With ``IPython.display`` unimportable (plain-script use), ``show()`` must not raise
            and must still return the composed HoloViews object.
        """
        import holoviews as hv

        monkeypatch.setitem(sys.modules, "IPython.display", None)
        monkeypatch.setitem(sys.modules, "IPython", None)
        el = hv.Points([(0, 0)])
        assert InteractiveMap().add_element(el).show() is el

    def test_repr_mimebundle_delegates_to_element_hook(self):
        """The map's mimebundle is the rendered element's mimebundle.

        Test scenario:
            A single registered element exposing ``_repr_mimebundle_`` is what ``render()``
            returns, so the map must delegate to that hook and return its result.
        """

        class _FakeElement:
            def _repr_mimebundle_(self, include=None, exclude=None):
                return {"text/plain": "fake"}

        bundle = InteractiveMap().add_element(_FakeElement())._repr_mimebundle_()
        assert bundle == {
            "text/plain": "fake"
        }, f"hook result not passed through: {bundle}"

    def test_repr_mimebundle_without_hook_is_empty(self):
        """An element without a mimebundle hook degrades to an empty bundle.

        Test scenario:
            ``render()`` returning an object with no ``_repr_mimebundle_`` attribute must
            yield ``{}`` rather than raising.
        """
        bundle = InteractiveMap().add_element(object())._repr_mimebundle_()
        assert bundle == {}, f"expected empty bundle for hookless element, got {bundle}"


class TestConstructionDefaults:
    """Constructor defaults and the reproject predicate (no engine needed)."""

    def test_default_configuration(self):
        """Defaults match the documented Web-Mercator-first contract.

        Test scenario:
            A bare ``InteractiveMap()`` must default to EPSG:3857, 700x500, no tiles, no title.
        """
        m = InteractiveMap()
        assert (m.crs, m.width, m.height, m.tiles, m.title) == (
            3857,
            700,
            500,
            None,
            "",
        )

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
        """``_needs_reproject`` only short-circuits on an exact int-EPSG match.

        Args:
            crs: Display CRS configured on the map.
            data_epsg: ``epsg`` the data object reports (``None`` = attribute absent).
            expected: Whether a reproject is required.
        """

        class _Data:
            pass

        data = _Data()
        if data_epsg is not None:
            data.epsg = data_epsg
        result = InteractiveMap(crs=crs)._needs_reproject(data)
        assert (
            result is expected
        ), f"crs={crs!r}, data_epsg={data_epsg!r}: expected {expected}, got {result}"

    def test_composition_includes_all_mixins(self):
        """``InteractiveMap`` composes the base plus all nine capability mixins.

        Test scenario:
            The MRO must contain every mixin the architecture promises, so later DI tasks
            can land methods on them without touching the composition.
        """
        from digitalearth.interactive.animation import AnimationMixin
        from digitalearth.interactive.base import InteractiveMapBase
        from digitalearth.interactive.bigdata import BigDataMixin
        from digitalearth.interactive.dashboard import DashboardMixin
        from digitalearth.interactive.decoration import DecorationMixin
        from digitalearth.interactive.interaction import InteractionMixin
        from digitalearth.interactive.projection import ProjectionMixin
        from digitalearth.interactive.raster import RasterMixin
        from digitalearth.interactive.temporal import TemporalMixin
        from digitalearth.interactive.vector import VectorMixin

        mro = InteractiveMap.__mro__
        for cls in (
            InteractiveMapBase,
            RasterMixin,
            VectorMixin,
            BigDataMixin,
            TemporalMixin,
            DecorationMixin,
            InteractionMixin,
            ProjectionMixin,
            AnimationMixin,
            DashboardMixin,
        ):
            assert cls in mro, f"{cls.__name__} missing from InteractiveMap MRO"


def test_reimport_is_stable():
    """The package re-imports cleanly (no import-time engine side effects)."""
    mod = importlib.reload(importlib.import_module("digitalearth.interactive"))
    assert hasattr(mod, "InteractiveMap")
