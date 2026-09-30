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

import logging

import numpy as np
import pytest

import digitalearth
from digitalearth.base import autostyle, registry
from digitalearth.base.autostyle import auto_style, load_library
from digitalearth.base.registry import resolve_uri, resolvers
from digitalearth.base.sources import DimensionInfo, Source
from digitalearth.ops.plugins import GROUPS, iter_plugins, load_plugins


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


class TestDiscoveryFailingDoesNotAbortImport:
    """entry_points() raising during discovery must not abort ``import digitalearth`` (M1)."""

    def test_entry_point_enumeration_raising_is_tolerated(self, monkeypatch, caplog):
        """A failed entry-point enumeration logs a WARNING and leaves the package importable.

        Test scenario:
            ``load_installed_plugins`` runs at ``import digitalearth``; driving ``iter_plugins`` calls
            ``entry_points()``, which a corrupt/partial environment can make raise (malformed distribution
            metadata, a broken ``RECORD``). The wave already tolerates a single plugin's ``load()`` raising, but
            the discovery call itself was unguarded, so the exception propagated out of module import and killed
            ``import digitalearth`` entirely. With discovery guarded, the call returns without raising, registers
            nothing, leaves the built-in resolvers untouched, and logs the failure — so the import that runs it
            still succeeds.
        """

        def _raise(*args, **kwargs):
            raise RuntimeError("corrupt distribution metadata")

        # Patch the name discovery looks it up under: iter_plugins imports entry_points into ops.plugins.
        monkeypatch.setattr("digitalearth.ops.plugins.entry_points", _raise)
        before = sorted(resolvers())
        with caplog.at_level(logging.WARNING, logger="digitalearth"):
            digitalearth.load_installed_plugins()  # must not raise
        assert sorted(resolvers()) == before, (
            "a failed enumeration must change no registry"
        )
        assert "corrupt distribution metadata" in caplog.text, (
            "the discovery failure is logged, not propagated"
        )


class TestTheExplicitDiscoveryApiPropagatesEnumerationFailures:
    """The public iter_plugins/load_plugins propagate a metadata-enumeration failure to a direct caller (N2).

    Deliberate asymmetry with ``load_installed_plugins``, the import-time convenience path (M1/L5): that path
    tolerates a corrupt environment so ``import digitalearth`` cannot die, but a direct programmatic call to the
    public helpers is an explicit request to enumerate the environment, so a failure there is the caller's to
    see and handle rather than one the library silently swallows into an empty result.
    """

    def test_iter_plugins_propagates_an_entry_points_failure(self, monkeypatch):
        """Iterating the public generator surfaces a raising ``entry_points()``, not an empty iterator.

        Test scenario:
            A corrupt/partial environment makes ``entry_points()`` raise. ``iter_plugins`` must not swallow it;
            the failure reaches the caller who asked to enumerate. ``entry_points()`` is evaluated lazily on the
            first advance, so the generator is built outside the ``raises`` block, leaving exactly one throwing
            call (``next``) inside it.
        """

        def _raise(*args, **kwargs):
            raise RuntimeError("corrupt distribution metadata")

        monkeypatch.setattr("digitalearth.ops.plugins.entry_points", _raise)
        iterator = iter_plugins("digitalearth.styles")
        with pytest.raises(RuntimeError, match="corrupt distribution metadata"):
            next(iterator)

    def test_load_plugins_propagates_an_entry_points_failure(self, monkeypatch):
        """``load_plugins`` lets the same enumeration failure propagate to its direct caller.

        Test scenario:
            ``load_plugins`` drives ``iter_plugins``; a raising ``entry_points()`` therefore propagates out of
            the call. Only a per-plugin ``load()`` is tolerated internally, not enumeration itself.
        """

        def _raise(*args, **kwargs):
            raise RuntimeError("corrupt distribution metadata")

        monkeypatch.setattr("digitalearth.ops.plugins.entry_points", _raise)
        with pytest.raises(RuntimeError, match="corrupt distribution metadata"):
            load_plugins("digitalearth.styles")


class TestShadowingABuiltInWarns:
    """A plugin that reuses a built-in scheme / bundled style group is loud about it (M3).

    This is a deliberate middle path. The override still happens — last-wins is kept, matching
    ``register_resolver``'s contract that :func:`temporary_resolver` relies on — but it no longer happens
    *silently*, so an installed dependency re-pointing the ``file:`` reader or the ``default`` palette for every
    consumer is at least visible. Contrast the plugin-vs-plugin case above, which stays quiet: shadowing another
    plugin is ordinary last-wins, shadowing a *built-in* is what warns. (The maintainer may choose to tighten
    this to a refusal, as ``register_kind``/``register_furniture`` do; that is their contract decision.)
    """

    def test_a_sources_plugin_shadowing_a_built_in_scheme_warns(self, caplog):
        """Reusing a built-in scheme ('file') logs a WARNING naming the plugin and the scheme; last-wins holds.

        Test scenario:
            ``file`` is a built-in resolver. A plugin registering it silently re-points every ``file:`` read;
            the warning makes that visible while still letting the override take effect.
        """
        ep = _FakeEP("rogue", ("file", lambda uri: "hijacked"))
        with caplog.at_level(logging.WARNING, logger="digitalearth"):
            digitalearth.register_plugins(
                "digitalearth.sources", load_plugins("digitalearth.sources", eps=[ep])
            )
        assert "rogue" in caplog.text and "file" in caplog.text, (
            f"the warning must name the plugin and the built-in scheme, got {caplog.text!r}"
        )
        assert resolve_uri("file:anything") == "hijacked", (
            "last-wins is kept: the plugin still overrides the built-in"
        )

    def test_a_sources_plugin_with_a_new_scheme_does_not_warn(self, caplog):
        """A brand-new scheme shadows no built-in, so it registers with no warning.

        Test scenario:
            The companion to the shadow case — proof the warning is scoped to built-ins, not fired for every
            plugin registration.
        """
        ep = _FakeEP("fresh", ("brandnew", lambda uri: uri))
        with caplog.at_level(logging.WARNING, logger="digitalearth"):
            digitalearth.register_plugins(
                "digitalearth.sources", load_plugins("digitalearth.sources", eps=[ep])
            )
        assert caplog.text == "", (
            f"a new scheme must register without a warning, got {caplog.text!r}"
        )

    def test_a_styles_plugin_shadowing_a_bundled_group_warns(self, caplog):
        """Redefining a bundled group ('default') logs a WARNING naming the plugin and the group; last-wins holds.

        Test scenario:
            ``default`` is a bundled style group. A plugin redefining it silently re-colours every fallback;
            the warning names it while the override still merges last-wins.
        """
        ep = _FakeEP("repaint", {"default": {"cmap": "plasma"}})
        with caplog.at_level(logging.WARNING, logger="digitalearth"):
            digitalearth.register_plugins(
                "digitalearth.styles", load_plugins("digitalearth.styles", eps=[ep])
            )
        assert "repaint" in caplog.text and "default" in caplog.text, (
            f"the warning must name the plugin and the bundled group, got {caplog.text!r}"
        )
        assert load_library()["default"]["cmap"] == "plasma", (
            "last-wins is kept: the plugin still overrides the bundled group"
        )

    def test_a_styles_plugin_with_a_new_group_does_not_warn(self, caplog):
        """A brand-new group shadows no bundled built-in, so it merges with no warning.

        Test scenario:
            The companion to the shadow case on the styles group — the warning is scoped to bundled groups.
        """
        ep = _FakeEP("adder", {"brand_new_var": {"cmap": "magma"}})
        with caplog.at_level(logging.WARNING, logger="digitalearth"):
            digitalearth.register_plugins(
                "digitalearth.styles", load_plugins("digitalearth.styles", eps=[ep])
            )
        assert caplog.text == "", (
            f"a new group must merge without a warning, got {caplog.text!r}"
        )


class TestAGenuineRegistrarBugIsNotMasked:
    """register_plugins tolerates a malformed plugin's contract error but lets a real registrar bug surface (L4).

    The broad catch exists so a plugin malformed *for its group* is skipped. But swallowing *every* Exception at
    WARNING also hid a genuine coding error inside a registrar (or a future one) behind a single log line, during
    import, invisible to anyone not watching logs. The catch is narrowed to the shapes a malformed plugin
    actually produces — ``TypeError``/``ValueError`` from the contract check — so a bug still surfaces.
    """

    def test_an_unexpected_registrar_error_propagates(self, monkeypatch):
        """A registrar raising something other than a contract error is not swallowed.

        Test scenario:
            A ``RuntimeError`` from inside a registrar is a wiring bug, not a bad plugin. Under the old broad
            ``except Exception`` it was logged at WARNING and swallowed; the narrowed catch lets it propagate so
            the defect is not invisible at import.
        """

        def _buggy_registrar(name, obj):
            raise RuntimeError("a real bug in the registrar")

        monkeypatch.setitem(
            digitalearth._PLUGIN_REGISTRARS, "digitalearth.sources", _buggy_registrar
        )
        # The `object()` built above the block, leaving only `register_plugins` inside it (tree guard
        # `test_refusal_blocks.py` counts any call in a `pytest.raises` block, so the loaded dict is hoisted).
        loaded = {"any": object()}
        with pytest.raises(RuntimeError, match="a real bug in the registrar"):
            digitalearth.register_plugins("digitalearth.sources", loaded)

    def test_a_malformed_plugin_contract_error_is_still_tolerated(self, caplog):
        """A ``ValueError`` from the contract check is still skipped at WARNING, so the narrowing kept tolerance.

        Test scenario:
            A sources object that is not a ``(scheme, resolver)`` pair raises when unpacked — here a 3-tuple
            gives ``ValueError`` (too many values). The narrowing must still catch that and skip the plugin, so a
            healthy plugin beside it registers and the skip is logged rather than raised.
        """
        good = _FakeEP("good", ("good", lambda uri: "opened"))
        bad = _FakeEP("bad", ("too", "many", "values"))  # unpacks to ValueError
        with caplog.at_level(logging.WARNING, logger="digitalearth"):
            digitalearth.register_plugins(
                "digitalearth.sources",
                load_plugins("digitalearth.sources", eps=[bad, good]),
            )
        assert resolve_uri("good:x") == "opened", "the healthy plugin still registered"
        assert "skipping plugin 'bad'" in caplog.text, (
            "the malformed plugin was skipped at WARNING, not raised"
        )
