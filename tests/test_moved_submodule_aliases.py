"""The eleven submodules that moved in the backend restructure stay reachable, with a deprecation notice.

Before the restructure ``digitalearth.__init__`` imported from ``charts``, ``series``, ``scene``, ``batch`` and
the rest, which bound each of them — and everything they in turn imported — as an attribute of the package. So
``digitalearth.charts`` and ``from digitalearth import series`` both worked, whether or not anyone intended
them as API. Moving those modules under ``base/``, ``static/`` and ``ops/`` unbound all eleven at once:
attribute access raised a bare ``AttributeError`` and the ``from`` form an ``ImportError``, neither saying
where the module had gone.

:func:`digitalearth.__getattr__` (PEP 562) restores them as deprecated aliases. These tests pin that the alias
table is complete, that it warns, that it does not leak into the supported facade, and that an unknown name
still fails the normal way.

Each test that needs a first-touch warning drops the cached alias from ``digitalearth.__dict__`` first, because
``__getattr__`` caches on resolution so the warning fires once per process rather than once per access.
"""

import importlib
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

import digitalearth
from digitalearth import _MOVED_SUBMODULES

#: Every name that was reachable as ``digitalearth.<name>`` before the restructure.
MOVED = sorted(_MOVED_SUBMODULES)


@pytest.fixture
def uncached():
    """Drop the cached aliases so each test sees a first attribute access.

    Yields:
        A callable taking a name and removing it from the package namespace.

    The cache is restored afterwards so test order cannot matter.
    """
    saved = {n: digitalearth.__dict__[n] for n in MOVED if n in digitalearth.__dict__}

    def drop(name):
        digitalearth.__dict__.pop(name, None)

    yield drop
    digitalearth.__dict__.update(saved)


class TestMovedSubmoduleAliases:
    """Tests for the PEP 562 alias table on the package root."""

    @pytest.mark.parametrize("name", MOVED)
    def test_alias_resolves_to_the_new_location(self, name, uncached):
        """Each moved name resolves to the module now living at its new path.

        Args:
            name: A pre-restructure submodule name.
            uncached: Fixture clearing the resolution cache.
        """
        uncached(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module = getattr(digitalearth, name)
        expected = _MOVED_SUBMODULES[name]
        assert module.__name__ == expected, (
            f"{name} resolved to {module.__name__}, expected {expected}"
        )

    @pytest.mark.parametrize("name", MOVED)
    def test_alias_warns_on_first_access(self, name, uncached):
        """Resolving a moved name warns and names its new home.

        Args:
            name: A pre-restructure submodule name.
            uncached: Fixture clearing the resolution cache.
        """
        uncached(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            getattr(digitalearth, name)
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deprecations, (
            f"accessing digitalearth.{name} should emit a DeprecationWarning"
        )
        message = str(deprecations[0].message)
        assert _MOVED_SUBMODULES[name] in message, (
            f"warning should name the new path, got: {message}"
        )

    def test_from_import_form_works(self, uncached):
        """``from digitalearth import series`` — the form seven notebooks used — resolves again."""
        uncached("series")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from digitalearth import series
        assert series.__name__ == "digitalearth.static.series", f"got {series.__name__}"
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
            "the from-form should warn too"
        )

    def test_warning_is_not_repeated_after_resolution(self, uncached):
        """The alias caches, so a hot path does not warn on every attribute access."""
        uncached("charts")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            getattr(digitalearth, "charts")
        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            getattr(digitalearth, "charts")
        assert not [w for w in second if issubclass(w.category, DeprecationWarning)], (
            "the second access should come from the cache without re-warning"
        )

    def test_unknown_attribute_still_raises_attribute_error(self):
        """``__getattr__`` must not swallow genuine typos."""
        with pytest.raises(
            AttributeError, match="no attribute 'definitely_not_a_module'"
        ):
            digitalearth.definitely_not_a_module

    def test_dir_includes_the_aliases(self):
        """``dir()`` lists the aliases so tab-completion still finds them."""
        listed = dir(digitalearth)
        assert set(MOVED) <= set(listed), (
            f"missing from dir(): {sorted(set(MOVED) - set(listed))}"
        )

    @pytest.mark.parametrize("name", MOVED)
    def test_alias_is_not_in_the_supported_facade(self, name):
        """Deprecated aliases must stay out of ``__all__`` — they are back-compat, not API.

        Args:
            name: A pre-restructure submodule name.
        """
        assert name not in digitalearth.__all__, (
            f"{name} is a deprecated alias and must not be in __all__"
        )

    def test_every_target_actually_exists(self):
        """The table cannot point at a module that no longer exists."""
        for name, target in _MOVED_SUBMODULES.items():
            assert importlib.import_module(target), (
                f"{name} points at missing module {target}"
            )


def _fresh_interpreter(code: str) -> str:
    """Run ``code`` in a new interpreter with ``src/`` importable and return its stdout.

    Args:
        code: The program to execute.

    Returns:
        Whatever the program printed, stripped.

    Raises:
        AssertionError: if the program exits non-zero.
    """
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src"), "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo),
    )
    assert result.returncode == 0, (
        f"subprocess failed:\n{result.stdout}\n{result.stderr}"
    )
    return result.stdout.strip()


class TestFacadeStaysClean:
    """The supported surface must not pay for the back-compat shims.

    These run in a **subprocess** rather than by clearing ``sys.modules``. Purging ``digitalearth`` from a live
    interpreter and re-importing it rebinds every class to a fresh object while the already-imported test
    modules still hold the originals, so ``isinstance`` checks across the rest of the suite start failing. A
    fresh interpreter is the only honest way to observe a first import.
    """

    def test_importing_digitalearth_emits_no_deprecation_warning(self):
        """A plain ``import digitalearth`` is warning-free — only touching a moved name warns."""
        out = _fresh_interpreter(
            """
            import warnings

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                import digitalearth
            print([str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)])
            """
        )
        assert out == "[]", f"import digitalearth warned: {out}"

    def test_deprecated_glyph_module_is_not_loaded_eagerly(self):
        """``import digitalearth`` must not drag in the deprecated ``static.glyph`` module (L4)."""
        out = _fresh_interpreter(
            """
            import sys
            import digitalearth

            print("digitalearth.static.glyph" in sys.modules)
            """
        )
        assert out == "False", (
            "the facade should resolve StaticGlyph lazily, not load the deprecated module on every import"
        )

    def test_static_glyph_is_still_importable(self):
        """The lazy resolution must not break the documented back-compat path."""
        from digitalearth.static import StaticGlyph

        assert StaticGlyph.__module__ == "digitalearth.static.glyph", (
            f"got {StaticGlyph.__module__}"
        )
