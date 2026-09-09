"""Tests for the StaticGlyph deprecation (PD-1 / L-2) and the digitalearth.static public surface.

``StaticGlyph`` moved to ``digitalearth/static/glyph.py`` and is re-exported from the package's
``__init__``. That re-export carries a contract worth pinning: ``from digitalearth.static import
StaticGlyph`` must keep working, and importing it must stay *silent* — the deprecation belongs on every
entry point of the class, not on the import, so a user who merely imports the backend is not warned about a
class they may never touch.
"""
import importlib
import sys
import warnings

import pytest

import digitalearth
from digitalearth.static import StaticGlyph

#: ``digitalearth.static.__all__``, and the submodule each name is defined in.
STATIC_EXPORTS = {
    "Scene": "digitalearth.static.scene",
    "Map": "digitalearth.static.map",
    "TexturedGlobe": "digitalearth.static.textured_globe",
    "StaticGlyph": "digitalearth.static.glyph",
    "grid": "digitalearth.static.figure",
    "shared_colorbar": "digitalearth.static.figure",
}


class TestStaticGlyphDeprecation:
    """StaticGlyph entry points emit a DeprecationWarning while still working."""

    def test_init_warns(self):
        """Constructing StaticGlyph emits a DeprecationWarning pointing at Map/quickmap.

        Test scenario:
            ``StaticGlyph()`` warns; the message names the modern replacement.
        """
        with pytest.warns(DeprecationWarning, match="StaticGlyph is deprecated"):
            StaticGlyph()

    def test_plot_warns_and_still_renders(self, dataset):
        """StaticGlyph.plot warns but still returns a figure/axes.

        Test scenario:
            Calling the legacy plot path emits the warning and produces a (fig, ax) pair.
        """
        with pytest.warns(DeprecationWarning, match="digitalearth.Map"):
            fig, ax = StaticGlyph.plot(dataset)
        assert fig is not None and ax is not None, "legacy plot should still render"

    def test_message_recommends_modern_api(self):
        """The deprecation message recommends quickmap / Map.

        Test scenario:
            The shared message mentions both ``quickmap`` and ``Map`` so users know where to go.
        """
        from digitalearth.static.glyph import _DEPRECATION_MSG

        assert "quickmap" in _DEPRECATION_MSG and "Map" in _DEPRECATION_MSG, (
            f"message should point to the modern API: {_DEPRECATION_MSG!r}"
        )


class TestStaticPackageSurface:
    """Tests for what digitalearth.static re-exports and how quietly it does it."""

    def test_import_emits_no_warning(self, monkeypatch):
        """Importing digitalearth.static is silent.

        Args:
            monkeypatch: Evicts the module-cache entry and the package attribute, restoring both on teardown.

        Test scenario:
            The package body is re-executed with the module cache evicted and every warning recorded; it
            must produce none. StaticGlyph is re-exported here, so a warning at import time would fire for
            everyone who touches the matplotlib backend at all, not just legacy users.
        """
        monkeypatch.setattr(digitalearth, "static", digitalearth.static, raising=False)
        monkeypatch.delitem(sys.modules, "digitalearth.static", raising=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module("digitalearth.static")
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]
        assert not messages, f"importing digitalearth.static should be silent, got: {messages}"

    def test_static_glyph_import_path_works(self):
        """``from digitalearth.static import StaticGlyph`` binds the class from static.glyph.

        Test scenario:
            The re-export must be the very class defined in ``static/glyph.py`` — not a subclass or an
            alias — so ``isinstance`` checks agree across both import paths.
        """
        from digitalearth.static.glyph import StaticGlyph as Defined

        assert StaticGlyph is Defined, "the re-export should be the class defined in static.glyph"

    def test_all_is_the_expected_surface(self):
        """``__all__`` advertises exactly the six documented names.

        Test scenario:
            The backend's public surface must not drift silently; additions and removals should be a
            deliberate edit to this list.
        """
        from digitalearth import static

        assert sorted(static.__all__) == sorted(STATIC_EXPORTS), (
            f"digitalearth.static.__all__ drifted: {sorted(static.__all__)}"
        )

    @pytest.mark.parametrize("name, module_path", sorted(STATIC_EXPORTS.items()))
    def test_export_is_the_object_from_its_submodule(self, name, module_path):
        """Each exported name is the identical object defined in its own submodule.

        Args:
            name: The exported attribute under test.
            module_path: The submodule that actually defines it.

        Test scenario:
            The ``__init__`` is a pure re-export layer, so every name must be the same object the defining
            module holds — no shadowing copy introduced by the package split.
        """
        from digitalearth import static

        defined = getattr(importlib.import_module(module_path), name)
        assert getattr(static, name) is defined, f"digitalearth.static.{name} is not {module_path}.{name}"

    @pytest.mark.parametrize("name", sorted(set(STATIC_EXPORTS) - {"StaticGlyph"}))
    def test_non_deprecated_exports_match_the_package_root(self, name):
        """The five non-deprecated names are the ones the package root exports.

        Args:
            name: The exported attribute under test.

        Test scenario:
            ``from digitalearth import Map`` and ``from digitalearth.static import Map`` must give the same
            object, so the root facade and the backend never diverge.
        """
        from digitalearth import static

        assert getattr(static, name) is getattr(digitalearth, name), (
            f"digitalearth.static.{name} is not digitalearth.{name}"
        )

    def test_static_glyph_is_not_re_exported_at_the_package_root(self):
        """The deprecated class stays off the package's public facade.

        Test scenario:
            ``StaticGlyph`` is reachable from the backend for backward compatibility but must not appear in
            ``digitalearth.__all__``, which advertises the API new code should use.
        """
        assert "StaticGlyph" not in digitalearth.__all__, (
            "a deprecated class should not be advertised on the package facade"
        )
