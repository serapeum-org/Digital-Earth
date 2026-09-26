"""The ``digitalearth.static`` public surface.

The backend's ``__init__`` is a pure re-export layer, and that carries a contract worth pinning: every name
it advertises is the object its own submodule defines, importing the package is *silent*, and a name it does
not export fails as an ordinary missing module attribute rather than resolving to something.
"""

import importlib
import sys
import warnings

import pytest

import digitalearth

#: ``digitalearth.static.__all__``, and the submodule each name is defined in.
STATIC_EXPORTS = {
    "Scene": "digitalearth.static.scene",
    "Map": "digitalearth.static.map",
    "TexturedGlobe": "digitalearth.static.textured_globe",
    "grid": "digitalearth.static.figure",
    "shared_colorbar": "digitalearth.static.figure",
}


class TestStaticPackageSurface:
    """Tests for what digitalearth.static re-exports and how quietly it does it."""

    def test_import_emits_no_warning(self, monkeypatch):
        """Importing digitalearth.static is silent.

        Args:
            monkeypatch: Evicts the module-cache entry and the package attribute, restoring both on teardown.

        Test scenario:
            The package body is re-executed with the module cache evicted and every warning recorded; it
            must produce none. A warning raised here would fire for everyone who touches the matplotlib
            backend at all, whatever they went on to draw.
        """
        monkeypatch.setattr(digitalearth, "static", digitalearth.static, raising=False)
        monkeypatch.delitem(sys.modules, "digitalearth.static", raising=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module("digitalearth.static")
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]
        assert not messages, (
            f"importing digitalearth.static should be silent, got: {messages}"
        )

    def test_all_is_the_expected_surface(self):
        """``__all__`` advertises exactly the five documented names.

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
        assert getattr(static, name) is defined, (
            f"digitalearth.static.{name} is not {module_path}.{name}"
        )

    @pytest.mark.parametrize("name", sorted(STATIC_EXPORTS))
    def test_exports_match_the_package_root(self, name):
        """Every name the backend exports is the one the package root exports.

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

    @pytest.mark.parametrize("name", ["geostatistics", "definitely_not_a_module"])
    def test_unknown_attribute_raises_attribute_error(self, name):
        """A name the backend does not export fails as a missing module attribute.

        Args:
            name: A name the backend does not define.

        Test scenario:
            A lookup that returned something for an unknown name would turn a typo into a silent ``None``.
            ``geostatistics`` is the case that matters: the module was deleted and moved upstream, so this
            is what a stale ``digitalearth.static.geostatistics`` reference lands on.
        """
        from digitalearth import static

        with pytest.raises(
            AttributeError,
            match=rf"module 'digitalearth\.static' has no attribute '{name}'",
        ):
            getattr(static, name)
