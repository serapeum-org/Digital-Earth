"""Wave 8 step 1 (DE-25b / ST-21 load half): installed ``digitalearth.*`` plugins populate the registries.

``digitalearth.ops.plugins.load_plugins`` discovered and imported entry points, but nothing registered what it
returned, so a shipped source adapter or style library never reached the runtime registries. ``register_plugins``
and ``load_installed_plugins`` on the package root are that missing wiring: a ``digitalearth.sources`` plugin
loads to a ``(scheme, resolver)`` pair registered through ``base.registry.register_resolver``, and a
``digitalearth.styles`` plugin loads to a per-variable style mapping merged into the autostyle library through
``base.autostyle.register_style_library``.

Both groups travel the one wiring, so both are exercised here — a defect in the shared path would otherwise hide
behind whichever group a single-group test happened to cover. Fake entry points (``eps=`` to ``load_plugins``,
as its own doctest does) stand in for an installed package; no real plugin is installed for these tests.
"""

import numpy as np
import pytest

import digitalearth
from digitalearth.base import autostyle, registry
from digitalearth.base.autostyle import auto_style, load_library
from digitalearth.base.registry import resolve_uri, resolvers
from digitalearth.base.sources import DimensionInfo, Source
from digitalearth.ops.plugins import GROUPS, load_plugins


class _FakeEP:
    """A minimal EntryPoint stand-in: a name plus a ``load()`` returning a fixed object."""

    def __init__(self, name, target):
        """Store the entry-point name and the object its ``load()`` should return."""
        self.name = name
        self._target = target

    def load(self):
        """Return the pre-set target object (stands in for importing the entry point)."""
        return self._target


class _BrokenEP:
    """An entry point whose ``load()`` raises, so ``load_plugins`` skips it before the wiring sees it."""

    name = "broken"

    def load(self):
        """Raise, as a third-party plugin with a broken import would."""
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _restore_registries():
    """Snapshot and restore the resolver table and the plugin style library around each test.

    Both registries are process-global, and the wiring registers into them permanently, so a test that did not
    put them back would leak its fake scheme or style group into every later test in the session.
    """
    resolvers_before = dict(registry._RESOLVERS)
    styles_before = dict(autostyle._PLUGIN_LIBRARY)
    yield
    registry._RESOLVERS.clear()
    registry._RESOLVERS.update(resolvers_before)
    autostyle._PLUGIN_LIBRARY.clear()
    autostyle._PLUGIN_LIBRARY.update(styles_before)


def _source(variable):
    """Build a minimal raster Source carrying a variable name, for the auto-style path."""
    return Source(
        DimensionInfo(np.zeros((2, 2)), "z"),
        DimensionInfo(np.array([0.0, 1.0]), "x"),
        DimensionInfo(np.array([0.0, 1.0]), "y"),
        metadata={"variable": variable},
    )


class TestTheWiringCoversEveryGroup:
    """The shared wiring must dispatch every group ``load_plugins`` knows about, and no other."""

    def test_a_registrar_is_declared_for_exactly_the_known_groups(self):
        """A group in ``GROUPS`` with no registrar would be discovered and then silently dropped.

        Test scenario:
            ``GROUPS`` is the published set of extension points; ``_PLUGIN_REGISTRARS`` says how each is
            registered. If the two drift — a group added to one and not the other — a plugin would load and
            reach no registry, which is the exact failure this step exists to end.
        """
        assert set(digitalearth._PLUGIN_REGISTRARS) == set(GROUPS), (
            f"registrars {sorted(digitalearth._PLUGIN_REGISTRARS)} do not match groups {sorted(GROUPS)}"
        )


class TestASourcesPluginRegistersItsResolver:
    """A ``digitalearth.sources`` plugin's ``(scheme, resolver)`` pair reaches ``resolve_uri``."""

    def test_the_scheme_does_not_resolve_before_the_plugin_is_wired(self):
        """The before state: with no plugin wired, the scheme is unknown and ``resolve_uri`` refuses it.

        Test scenario:
            The counterpart to the after state below — a resolver that appears without a plugin having
            registered it would make the after check pass for the wrong reason.
        """
        with pytest.raises(KeyError, match="no resolver registered for scheme 'demo'"):
            resolve_uri("demo:x")

    def test_a_sources_plugin_registers_and_resolve_uri_finds_its_scheme(self):
        """The after state: once wired, ``resolve_uri`` opens the scheme through the plugin's resolver.

        Test scenario:
            ``load_plugins`` loads the fake entry point to its ``(scheme, resolver)`` pair; the wiring hands
            that to ``register_resolver``, so ``resolve_uri('demo:x')`` runs the plugin's resolver.
        """
        ep = _FakeEP("demo", ("demo", lambda uri: uri.upper()))
        digitalearth.register_plugins(
            "digitalearth.sources", load_plugins("digitalearth.sources", eps=[ep])
        )
        assert "demo" in resolvers(), "the plugin's scheme is registered after wiring"
        assert resolve_uri("demo:x") == "DEMO:X", (
            "resolve_uri reaches the plugin's resolver"
        )


class TestAStylesPluginExtendsTheLibrary:
    """A ``digitalearth.styles`` plugin's per-variable mapping merges into the autostyle library."""

    def test_the_group_is_absent_before_the_plugin_is_wired(self):
        """The before state: the plugin's style group is not in the library until it is wired."""
        assert "plasma_var" not in load_library(), (
            "the group must not be present before wiring"
        )

    def test_a_styles_plugin_merges_its_group_into_the_library(self):
        """The after state: the plugin's group is in the merged library.

        Test scenario:
            The fake entry point loads to a per-variable style mapping; the wiring merges it via
            ``register_style_library``, so ``load_library`` returns it beside the bundled groups.
        """
        ep = _FakeEP("extra", {"plasma_var": {"match": ["ohc"], "cmap": "plasma"}})
        digitalearth.register_plugins(
            "digitalearth.styles", load_plugins("digitalearth.styles", eps=[ep])
        )
        assert load_library()["plasma_var"]["cmap"] == "plasma"

    def test_auto_style_uses_a_plugin_contributed_group(self):
        """End to end: a variable a plugin group matches is styled by that group, not the default.

        Test scenario:
            The point of a styles plugin is that ``auto_style`` picks up its colormap. An unmatched variable
            would fall back to the ``viridis`` default, so a wired plugin changing the answer is what proves
            the merge reached the styling path.
        """
        ep = _FakeEP("extra", {"ocean_heat": {"match": ["ohc"], "cmap": "inferno"}})
        digitalearth.register_plugins(
            "digitalearth.styles", load_plugins("digitalearth.styles", eps=[ep])
        )
        assert auto_style(_source("ohc_2020"))["cmap"] == "inferno"


class TestABrokenPluginDoesNotAbortTheHealthyOnes:
    """One broken or malformed plugin must not stop the healthy ones registering — on either group."""

    def test_a_broken_sources_plugin_is_skipped_and_the_healthy_one_registers(self):
        """A ``load()`` that raises is skipped by ``load_plugins``; the healthy pair still reaches the registry."""
        good = _FakeEP("good", ("good", lambda uri: "opened"))
        digitalearth.register_plugins(
            "digitalearth.sources",
            load_plugins("digitalearth.sources", eps=[_BrokenEP(), good]),
        )
        assert "broken" not in resolvers(), "the broken plugin registered nothing"
        assert resolve_uri("good:x") == "opened", "the healthy plugin still registered"

    def test_a_broken_styles_plugin_is_skipped_and_the_healthy_one_merges(self):
        """The same tolerance on the styles group: a broken plugin is skipped, the healthy one merges."""
        good = _FakeEP("good", {"good_var": {"match": ["gv"], "cmap": "cividis"}})
        digitalearth.register_plugins(
            "digitalearth.styles",
            load_plugins("digitalearth.styles", eps=[_BrokenEP(), good]),
        )
        assert load_library()["good_var"]["cmap"] == "cividis"

    def test_a_malformed_sources_object_is_skipped_by_the_wiring(self):
        """A plugin that loads to something other than a ``(scheme, resolver)`` pair is skipped, not fatal.

        Test scenario:
            ``load_plugins`` cannot judge shape — it loaded the object fine — so the wiring is what tolerates a
            malformed one. A resolver-less object beside a healthy pair must leave the healthy one registered.
        """
        malformed = _FakeEP("malformed", object())
        good = _FakeEP("good", ("good", lambda uri: "opened"))
        digitalearth.register_plugins(
            "digitalearth.sources",
            load_plugins("digitalearth.sources", eps=[malformed, good]),
        )
        assert resolve_uri("good:x") == "opened", "the healthy plugin still registered"


class TestRegisteringIsIdempotentAndDuplicatesFollowTheExistingPolicy:
    """Re-wiring the same set changes nothing, and a duplicate name is the registry's own last-wins."""

    def test_registering_the_same_sources_plugin_twice_leaves_one_scheme(self):
        """Idempotence: wiring the same loaded set twice registers the scheme once, and it still resolves."""
        loaded = load_plugins(
            "digitalearth.sources",
            eps=[_FakeEP("demo", ("demo", lambda uri: uri.upper()))],
        )
        digitalearth.register_plugins("digitalearth.sources", loaded)
        digitalearth.register_plugins("digitalearth.sources", loaded)
        assert sorted(resolvers()) == ["demo", "file", "object"], sorted(resolvers())
        assert resolve_uri("demo:x") == "DEMO:X"

    def test_a_duplicate_scheme_follows_the_registry_last_wins_policy(self):
        """Two plugins claiming one scheme resolve to the last registered, as ``register_resolver`` already does."""
        first = _FakeEP("a", ("dup", lambda uri: "first"))
        second = _FakeEP("b", ("dup", lambda uri: "second"))
        digitalearth.register_plugins(
            "digitalearth.sources",
            load_plugins("digitalearth.sources", eps=[first, second]),
        )
        assert resolve_uri("dup:x") == "second"

    def test_a_duplicate_style_group_follows_the_library_last_wins_policy(self):
        """Two plugins defining one style group merge last-wins, as the bundled YAML files already do."""
        first = _FakeEP("a", {"dupvar": {"cmap": "first_map"}})
        second = _FakeEP("b", {"dupvar": {"cmap": "second_map"}})
        digitalearth.register_plugins(
            "digitalearth.styles",
            load_plugins("digitalearth.styles", eps=[first, second]),
        )
        assert load_library()["dupvar"]["cmap"] == "second_map"


class TestTheLoaderRunsWithNoPluginsInstalled:
    """``load_installed_plugins`` is the zero-argument entry point the package calls at import."""

    def test_load_installed_plugins_is_a_safe_no_op_with_nothing_installed(self):
        """With no third-party plugin installed, calling it registers nothing and keeps the built-ins.

        Test scenario:
            The import-time call must not disturb the built-in ``file``/``object`` resolvers, and calling it
            again is harmless — the package imports once, but the function must be safe to re-run.
        """
        before = sorted(resolvers())
        digitalearth.load_installed_plugins()
        assert sorted(resolvers()) == before, (
            "an empty environment must not change the resolver table"
        )
        assert {"file", "object"} <= set(resolvers()), (
            "the built-in schemes stay registered"
        )

    def test_an_unknown_group_registers_nothing(self):
        """A group with no registrar is a no-op rather than an error, so a stray group name cannot crash import."""
        digitalearth.register_plugins("digitalearth.nonexistent", {"x": object()})
        assert {"file", "object"} <= set(resolvers()), (
            "an unknown group touched no registry"
        )
