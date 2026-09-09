"""Tests for digitalearth.scene — the deprecation shim left behind by the scene -> static rename.

``digitalearth.scene`` was a package; it is now a single module that re-exports the five names the old
package exported and warns on import. The shim has exactly three obligations, and this file pins all three:

1. importing it emits a ``DeprecationWarning`` that names the replacement,
2. the forwarded names are the *same objects* as :mod:`digitalearth.static` (and the package root) exposes,
3. nothing more is forwarded — in particular the old submodule paths (``digitalearth.scene.maps.vector``)
   are deliberately **not** aliased, so a stale deep import fails loudly instead of silently working.

Every test imports the shim through :func:`importlib.import_module` after evicting it from ``sys.modules``,
because a module body — and therefore ``warnings.warn`` — runs only on the first import of a process.
"""

import importlib
import sys
import warnings

import pytest

import digitalearth
from digitalearth import static

SHIM = "digitalearth.scene"

#: The names the old ``digitalearth.scene`` package exported, and so the only ones the shim forwards.
FORWARDED = ["Scene", "Map", "TexturedGlobe", "grid", "shared_colorbar"]


@pytest.fixture
def evicted_shim():
    """Evict ``digitalearth.scene`` from the module cache so the next import re-runs its body.

    Yields:
        None: the fixture only manages ``sys.modules`` state.

    The originally-cached module (if any) is restored afterwards so a test that re-imports the shim cannot
    leak a second module object into later tests.
    """
    cached = sys.modules.pop(SHIM, None)
    try:
        yield
    finally:
        sys.modules.pop(SHIM, None)
        if cached is not None:
            sys.modules[SHIM] = cached
            digitalearth.scene = cached


@pytest.fixture
def shim(evicted_shim):
    """Import the shim freshly with its deprecation warning silenced.

    Args:
        evicted_shim: Fixture that clears the module cache first.

    Returns:
        module: the freshly-executed ``digitalearth.scene`` module.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return importlib.import_module(SHIM)


class TestSceneShimWarning:
    """Tests for the DeprecationWarning the shim emits on import."""

    def test_import_emits_deprecation_warning(self, evicted_shim):
        """Importing digitalearth.scene warns that the module is deprecated.

        Test scenario:
            A fresh import (module cache evicted) raises a DeprecationWarning whose text opens with the
            deprecated module's own name.
        """
        with pytest.warns(
            DeprecationWarning, match=r"digitalearth\.scene is deprecated"
        ):
            importlib.import_module(SHIM)

    def test_warning_names_the_replacement(self, evicted_shim):
        """The warning text points at digitalearth.static and at the package root.

        Test scenario:
            A deprecation is only actionable if it says where to go — the message must mention the new
            module, that it is the matplotlib backend, and the removal.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(SHIM)
        messages = [
            str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert messages, "importing the shim should record a DeprecationWarning"
        text = messages[0]
        for expected in (
            "digitalearth.static",
            "matplotlib backend",
            "removed in a future release",
        ):
            assert expected in text, (
                f"warning should mention {expected!r}, got: {text!r}"
            )

    def test_only_one_warning_per_import(self, evicted_shim):
        """The shim warns exactly once, not once per forwarded name.

        Test scenario:
            A single module-level ``warnings.warn`` must produce one record for one fresh import, so a user
            importing the shim is not spammed with five identical lines.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module(SHIM)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1, (
            f"expected exactly one DeprecationWarning, got {len(deprecations)}"
        )

    def test_importing_static_directly_does_not_warn(self, monkeypatch):
        """The replacement module is warning-free, so the warning really is about the old path.

        Args:
            monkeypatch: Evicts the module-cache entry and the package attribute, restoring both on teardown.

        Test scenario:
            Re-executing ``digitalearth.static``'s body records no DeprecationWarning — proof the shim's
            warning comes from ``scene.py`` and not from something it re-exports.
        """
        # Re-registering the current value is how monkeypatch is told to restore it after the re-import
        # rebinds the package attribute; delitem alone would only put sys.modules back.
        monkeypatch.setattr(digitalearth, "static", digitalearth.static, raising=False)
        monkeypatch.delitem(sys.modules, "digitalearth.static", raising=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.import_module("digitalearth.static")
        deprecations = [
            str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
        ]
        assert not deprecations, (
            f"digitalearth.static must not warn on import, got: {deprecations}"
        )


class TestSceneShimExports:
    """Tests for the names the shim forwards."""

    def test_all_is_exactly_the_forwarded_set(self, shim):
        """``__all__`` advertises precisely the five names the old package exported.

        Test scenario:
            The shim is a frozen compatibility surface — it must not grow new names, and must not have lost
            one of the five it exists to forward.
        """
        assert sorted(shim.__all__) == sorted(FORWARDED), (
            f"__all__ drifted: {sorted(shim.__all__)}"
        )

    @pytest.mark.parametrize("name", FORWARDED)
    def test_forwarded_name_is_the_static_object(self, shim, name):
        """Each forwarded name is the identical object exposed by digitalearth.static.

        Args:
            shim: The freshly-imported shim module.
            name: The forwarded attribute under test.

        Test scenario:
            Identity (``is``), not merely equality — a copy would mean ``isinstance``/``issubclass`` checks
            silently diverge between the old and new import paths.
        """
        assert getattr(shim, name) is getattr(static, name), (
            f"digitalearth.scene.{name} is not digitalearth.static.{name}"
        )

    @pytest.mark.parametrize("name", FORWARDED)
    def test_forwarded_name_matches_package_root(self, shim, name):
        """Each forwarded name is also the object the package root exports.

        Args:
            shim: The freshly-imported shim module.
            name: The forwarded attribute under test.

        Test scenario:
            The docstring recommends ``from digitalearth import Map`` as the durable path; that must resolve
            to the very same object the deprecated path hands back.
        """
        assert getattr(shim, name) is getattr(digitalearth, name), (
            f"digitalearth.scene.{name} is not digitalearth.{name}"
        )

    def test_from_import_still_works(self, evicted_shim):
        """``from digitalearth.scene import Map`` — the whole point of the shim — still binds Map.

        Test scenario:
            The documented legacy statement executes and yields the current ``Map`` class.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from digitalearth.scene import Map

        assert Map is static.Map, (
            "the legacy from-import should bind the current Map class"
        )

    def test_static_glyph_is_not_forwarded(self, shim):
        """The shim does not forward StaticGlyph, which the old scene package never exported.

        Test scenario:
            ``StaticGlyph`` lives on ``digitalearth.static``; forwarding it here would invent a legacy path
            that never existed.
        """
        assert not hasattr(shim, "StaticGlyph"), (
            "the shim should forward only what scene itself exported"
        )


class TestSceneShimSubmodules:
    """Tests for the deliberately un-aliased submodule paths."""

    @pytest.mark.parametrize(
        "path",
        [
            "digitalearth.scene.maps",
            "digitalearth.scene.maps.vector",
            "digitalearth.scene.map",
            "digitalearth.scene.domains",
        ],
    )
    def test_submodule_paths_are_not_aliased(self, evicted_shim, path):
        """Old deep import paths raise ModuleNotFoundError instead of resolving.

        Args:
            evicted_shim: Fixture that clears the module cache first.
            path: The legacy dotted path under test.

        Test scenario:
            The shim is a module, not a package, so every ``digitalearth.scene.<sub>`` import fails loudly —
            documented behaviour that pushes users to ``digitalearth.static.<sub>``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ModuleNotFoundError) as exc_info:
                importlib.import_module(path)

        assert "is not a package" in str(exc_info.value), (
            f"expected a 'not a package' failure for {path}, got: {exc_info.value}"
        )

    def test_static_submodule_path_is_the_working_replacement(self):
        """The replacement path named in the shim's docstring actually imports.

        Test scenario:
            ``digitalearth.static.maps.vector`` — the module the docstring tells users to switch to —
            resolves, so the migration advice is correct.
        """
        module = importlib.import_module("digitalearth.static.maps.vector")
        assert module is not None, (
            "digitalearth.static.maps.vector should be importable"
        )
