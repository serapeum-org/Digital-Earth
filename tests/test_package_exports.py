"""Tests for the top-level digitalearth public surface (PC-4)."""

import ast
import importlib
import importlib.metadata
import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import digitalearth

EXPECTED = [
    "quickplot",
    "quickmap",
    "to_backend",
    "Map",
    "Scene",
    "TexturedGlobe",
    "facet",
    "grid",
    "shared_colorbar",
    "projections",
    "get_source",
    "Source",
    "DimensionInfo",
    "line",
    "bar",
    "histogram",
    "scatter",
    "bar_by",
    "line_by",
    "statistics",
    "envelope",
    "quantile_band",
    "boxplot",
    "multiboxplot",
    "stripes",
    "TimeSeries",
    "Climatology",
    "Batch",
    "gallery",
    "load_plugins",
]


class TestPackageExports:
    """Tests for digitalearth.__all__ and the re-exported names."""

    @pytest.mark.parametrize("name", EXPECTED)
    def test_name_is_exported(self, name):
        """Each documented public name is importable from the package root.

        Args:
            name: The attribute expected on the ``digitalearth`` package.

        Test scenario:
            ``from digitalearth import <name>`` works (the symbol is bound and in ``__all__``).
        """
        assert hasattr(digitalearth, name), f"{name} missing from digitalearth"
        assert name in digitalearth.__all__, f"{name} not in __all__"

    def test_all_matches_expected_set(self):
        """``__all__`` is exactly the curated public set (no drift).

        Test scenario:
            The package advertises precisely the expected names — additions/removals must be intentional.
        """
        assert sorted(set(digitalearth.__all__)) == sorted(set(EXPECTED)), (
            f"__all__ drifted: {sorted(digitalearth.__all__)}"
        )

    def test_temporal_classes_are_the_real_ones(self):
        """The re-exported temporal classes are the ones defined in the temporal package.

        Test scenario:
            ``digitalearth.TimeSeries`` is ``digitalearth.static.temporal.TimeSeries`` (no shadow/duplicate).
        """
        temporal = importlib.import_module("digitalearth.static.temporal")
        assert digitalearth.TimeSeries is temporal.TimeSeries, (
            "TimeSeries is not the temporal one"
        )
        assert digitalearth.Climatology is temporal.Climatology, (
            "Climatology is not the temporal one"
        )

    def test_projections_is_the_submodule(self):
        """The re-exported ``projections`` is the static.projections submodule.

        Test scenario:
            ``digitalearth.projections`` resolves a known projection factory (web_mercator -> 3857).
        """
        assert digitalearth.projections.get("web_mercator") == 3857, (
            "projections submodule not wired"
        )

    @pytest.mark.parametrize("name, home", sorted(digitalearth._LAZY_EXPORTS.items()))
    def test_a_public_name_is_imported_from_its_home_on_first_use_and_cached(
        self, name, home, monkeypatch
    ):
        """Each public name resolves to the object its home module defines, and stays bound once resolved.

        Args:
            name: A public name the package root resolves lazily.
            home: The module the name is imported from.
            monkeypatch: pytest's patcher, used to unbind the name so the access is a first one.

        Test scenario:
            The package root no longer imports these eagerly, so ``import digitalearth`` loads no renderer.
            Unbinding the name forces the ``__getattr__`` path; the object returned must be the home module's
            own, and it must then sit in the package namespace so later accesses skip the hook.
        """
        monkeypatch.delitem(vars(digitalearth), name, raising=False)
        resolved = getattr(digitalearth, name)
        expected = getattr(importlib.import_module(home), name)
        assert resolved is expected, f"digitalearth.{name} is not {home}.{name}"
        assert vars(digitalearth).get(name) is expected, (
            f"digitalearth.{name} was not cached after its first resolution"
        )

    def test_a_type_checker_sees_every_lazy_name_imported_from_its_home(self):
        """The root's `if TYPE_CHECKING:` block imports each lazy name, and each subpackage, from where it lives.

        Test scenario:
            A type checker does not run `__getattr__`; it reads the imports. Resolving the public names lazily
            left mypy and IDEs nothing but an unannotated `__getattr__`, so `from digitalearth import Map` was
            `Any` and `quickmap(backend=5)` passed mypy. The block this reads is what gives them the real
            objects back, and it has to name exactly what `_LAZY_EXPORTS` and `_LAZY_SUBPACKAGES` resolve at
            runtime — a name missing here is `Any` again, and one imported from the wrong module is a type the
            caller never gets.
        """
        tree = ast.parse(
            pathlib.Path(digitalearth.__file__).read_text(encoding="utf-8")
        )
        blocks = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ]
        assert len(blocks) == 1, (
            f"expected one top-level `if TYPE_CHECKING:` block, found {len(blocks)}"
        )
        imported = {
            alias.asname or alias.name: node.module
            for node in blocks[0].body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        subpackages = {
            name for name, module in imported.items() if module == "digitalearth"
        }
        names = {
            name: module
            for name, module in imported.items()
            if module != "digitalearth"
        }
        drift = sorted(set(names.items()) ^ set(digitalearth._LAZY_EXPORTS.items()))
        assert not drift, (
            f"the TYPE_CHECKING imports differ from _LAZY_EXPORTS by {drift}"
        )
        missing = sorted(subpackages ^ set(digitalearth._LAZY_SUBPACKAGES))
        assert not missing, (
            f"the TYPE_CHECKING subpackage imports differ from _LAZY_SUBPACKAGES by {missing}"
        )


#: The subpackages the backend-per-package layout introduced, and a module each must contain.
LAYOUT = [
    ("digitalearth.base", "digitalearth.base.arrays"),
    ("digitalearth.ops", "digitalearth.ops.batch"),
    ("digitalearth.static", "digitalearth.static.map"),
]


class TestSubpackageLayout:
    """Tests for the subpackage ``__init__`` modules the backend-per-package split added."""

    @pytest.mark.parametrize("package, member", LAYOUT, ids=lambda v: v.split(".")[-1])
    def test_subpackage_is_importable(self, package, member):
        """Each new subpackage imports and exposes the module it is supposed to hold.

        Args:
            package: The subpackage's dotted path.
            member: A module that must live inside it.

        Test scenario:
            ``base`` and ``ops`` are docstring-only ``__init__`` files, so the only thing that can break is
            the package itself being unimportable or a module having failed to move into it.
        """
        module = importlib.import_module(package)
        assert module.__doc__, (
            f"{package} should carry a module docstring explaining what it holds"
        )
        assert importlib.import_module(member) is not None, (
            f"{member} should live under {package}"
        )

    @pytest.mark.parametrize("package", [p for p, _ in LAYOUT])
    def test_subpackage_is_a_real_package(self, package):
        """Each subpackage is a package, not a module.

        Test scenario:
            ``__path__`` exists only on packages, so this is what refuses a subpackage that has become a
            plain module — which is what the restructure's own forwarding shim was.
        """
        module = importlib.import_module(package)
        assert hasattr(module, "__path__"), (
            f"{package} should be a package with submodules"
        )

    @pytest.mark.parametrize("name", digitalearth._LAZY_SUBPACKAGES)
    def test_a_subpackage_the_root_used_to_bind_resolves_on_attribute_access(
        self, name, monkeypatch
    ):
        """``import digitalearth`` then ``digitalearth.static.Map`` keeps working with the root's imports lazy.

        Args:
            name: A subpackage the eager imports used to leave bound on the package.
            monkeypatch: pytest's patcher, used to unbind the subpackage as a fresh ``import digitalearth`` would.

        Test scenario:
            A subpackage is bound on its parent only once something imports it, and the root no longer does.
            Attribute access on the unbound name must return the subpackage itself and bind it.
        """
        monkeypatch.delitem(vars(digitalearth), name, raising=False)
        resolved = getattr(digitalearth, name)
        expected = importlib.import_module(f"digitalearth.{name}")
        assert resolved is expected, (
            f"digitalearth.{name} resolved to {resolved!r}, not the subpackage"
        )
        assert vars(digitalearth).get(name) is expected, (
            f"digitalearth.{name} was not bound after its first resolution"
        )

    @pytest.mark.parametrize("name", ["interactive", "three_d", "web"])
    def test_an_optional_backend_is_not_resolved_on_attribute_access(
        self, name, monkeypatch
    ):
        """A backend behind an extra is not imported by touching the attribute.

        Args:
            name: A backend subpackage that needs an optional extra.
            monkeypatch: pytest's patcher, used to unbind the subpackage if an earlier test imported it.

        Test scenario:
            Resolving ``digitalearth.web`` on access would turn ``hasattr`` into an ImportError where the extra is
            not installed. Unbound, the attribute is absent until the caller imports the subpackage itself.
        """
        monkeypatch.delitem(vars(digitalearth), name, raising=False)
        assert not hasattr(digitalearth, name), (
            f"digitalearth.{name} was resolved on attribute access"
        )

    def test_old_flat_module_paths_are_gone(self):
        """The pre-split top-level module paths no longer resolve.

        Test scenario:
            ``digitalearth.batch``/``cli``/``browser``/``plugins`` moved under ``ops``; leaving a working
            alias behind would let stale imports silently keep the flat layout alive.
        """
        for stale in (
            "digitalearth.batch",
            "digitalearth.cli",
            "digitalearth.browser",
            "digitalearth.plugins",
            "digitalearth._arrays",
            "digitalearth._crs",
        ):
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(stale)


class TestPackageMetadata:
    """Tests for the dunder metadata ``digitalearth/__init__.py`` computes at import time."""

    def test_version_comes_from_the_installed_distribution(self):
        """``__version__`` is the version ``importlib.metadata`` reports for the installed package.

        Test scenario:
            The happy path of the version lookup. Nothing else in the suite reads ``__version__``, so a
            broken lookup would surface only in a built wheel or the rendered docs.
        """
        expected = importlib.metadata.version("digitalearth")
        assert digitalearth.__version__ == expected, (
            f"expected __version__ {expected!r}, got {digitalearth.__version__!r}"
        )
        assert digitalearth.__version__ != "unknown", (
            "the package is installed in this environment, so the fallback must not have been taken"
        )

    def test_version_falls_back_to_unknown_when_no_distribution_is_installed(self):
        """``__version__`` is ``"unknown"`` when the distribution metadata cannot be found.

        Test scenario:
            The ``PackageNotFoundError`` branch — the one a source checkout that was never installed takes,
            which is exactly how the docs build and ``pythonpath = ["src"]`` import the package. It is marked
            ``pragma: no cover`` because the test suite runs against an installed package, so it is exercised
            here by loading the same ``__init__.py`` under a module name no distribution provides: the lookup
            raises for real rather than through a patched ``version``. The module is deliberately not
            registered in ``sys.modules``, so the live ``digitalearth`` package is untouched.
        """
        source = pathlib.Path(digitalearth.__file__)
        spec = importlib.util.spec_from_file_location(
            "digitalearth_uninstalled_probe", source
        )
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        assert probe.__version__ == "unknown", (
            f"expected the fallback 'unknown', got {probe.__version__!r}"
        )

    def test_the_importlib_metadata_backport_fallback_is_gone(self):
        """``__init__.py`` reads the version from the stdlib only, with no ``importlib_metadata`` fallback.

        Test scenario:
            ``requires-python`` floors at 3.11 and ``importlib.metadata`` has been stdlib since 3.8, so the
            ``try/except ImportError`` around a backport import was unreachable code pretending to be a
            compatibility shim. Guard the deletion: re-adding it would put an undeclared dependency back on
            the import path of every ``import digitalearth``.
        """
        source = pathlib.Path(digitalearth.__file__).read_text(encoding="utf-8")
        offending = [
            line
            for line in source.splitlines()
            if "importlib_metadata" in line and not line.lstrip().startswith("#")
        ]
        assert not offending, (
            f"the importlib_metadata backport fallback should stay deleted, found: {offending}"
        )

    @pytest.mark.parametrize(
        "name, expected_type, expected_value",
        [
            ("hard_dependencies", tuple, ()),
            ("missing_dependencies", list, []),
        ],
    )
    def test_dependency_containers_keep_their_declared_types(
        self, name, expected_type, expected_value
    ):
        """The two module-level dependency containers are the annotated types, and both are empty.

        Args:
            name: The module-level container under test.
            expected_type: The concrete type its annotation promises.
            expected_value: The value it must hold in a correctly installed environment.

        Test scenario:
            ``hard_dependencies`` is an empty literal, which gives a type checker nothing to infer an element
            type from — hence the ``tuple[str, ...]`` / ``list[str]`` annotations #172 added. They are public
            module attributes, so pin the runtime shape alongside them. A non-empty ``missing_dependencies``
            would mean the import-time check found a missing hard dependency and the package only imported
            because the list is never actually populated.
        """
        value = getattr(digitalearth, name)
        assert isinstance(value, expected_type), (
            f"digitalearth.{name} should be a {expected_type.__name__}, got {type(value).__name__}"
        )
        assert value == expected_value, (
            f"digitalearth.{name} should be {expected_value!r}, got {value!r}"
        )


def _fresh_interpreter(code: str) -> str:
    """Run ``code`` in a new interpreter with ``src/`` importable and return its stdout.

    A first import can only be observed in a fresh process. Purging ``digitalearth`` from a live interpreter
    and re-importing it rebinds every class to a new object while the already-imported test modules hold the
    originals, after which ``isinstance`` checks elsewhere in the suite start failing.

    Args:
        code: The program to execute.

    Returns:
        Whatever the program printed, stripped.

    Raises:
        AssertionError: if the program exits non-zero.
    """
    repo = pathlib.Path(__file__).resolve().parents[1]
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


class TestTheRootsAttributeLookup:
    """``__getattr__``/``__dir__`` on the package root, and the guidance the removed names carry.

    Review M6: the alias proofs in ``tests/test_moved_submodule_aliases.py`` and ``tests/test_scene_shim.py``
    went with the deprecation machinery, correctly — but they were also the only tests of the **live** code in
    ``src/digitalearth/__init__.py`` that survived the deletion. The hook still resolves a public name
    lazily, still raises for a typo, still answers ``dir()`` and still carries the ``_REMOVED_GEOSTATISTICS``
    guidance, which was kept deliberately. None of that was asserted anywhere afterwards.
    """

    def test_a_typo_is_not_swallowed(self):
        """A name the package has never had raises ``AttributeError``, with the standard wording.

        Test scenario:
            The hook is consulted for every miss, so a ``__getattr__`` that fell through to ``None`` would
            turn every typo into a silent ``None`` — and the branches above it all return a value.
        """
        with pytest.raises(AttributeError) as excinfo:
            digitalearth.definitely_not_a_module
        assert "no attribute 'definitely_not_a_module'" in str(excinfo.value), (
            f"the miss must name the attribute, got: {excinfo.value}"
        )

    def test_dir_offers_every_name_the_hook_can_resolve(self):
        """``dir()`` lists the lazy names, so tab-completion finds one before its first use imports it.

        Test scenario:
            The listing is built from ``globals()`` plus the two lazy tables, and this branch stopped
            unioning ``_MOVED_SUBMODULES`` when that table went. A listing narrower than the hook hides half
            the package from tab-completion; one wider than it offers names that only raise.
        """
        listed = set(dir(digitalearth))
        resolvable = set(digitalearth._LAZY_EXPORTS) | set(
            digitalearth._LAZY_SUBPACKAGES
        )
        assert resolvable <= listed, (
            f"dir(digitalearth) omits resolvable names: {sorted(resolvable - listed)}"
        )

    @pytest.mark.parametrize("name", digitalearth._REMOVED_GEOSTATISTICS)
    def test_a_removed_geostatistics_name_is_absent(self, name):
        """The presets are removed, not deprecated, so ``hasattr`` must answer ``False``.

        Args:
            name: A removed geostatistics preset, or the removed submodule.

        Test scenario:
            The removal branch is tested **after** the two lazy tables, so binding one of these names into
            ``_LAZY_EXPORTS`` would make it resolve and leave the guidance below unreachable — with nothing
            failing. That is the scenario this asks about.
        """
        assert not hasattr(digitalearth, name), (
            f"digitalearth.{name} resolves, so it was deprecated rather than removed"
        )

    @pytest.mark.parametrize("name", digitalearth._REMOVED_GEOSTATISTICS)
    def test_a_removed_geostatistics_name_points_upstream(self, name):
        """The failure names where the preset went and what to draw instead of it.

        Args:
            name: A removed geostatistics preset, or the removed submodule.

        Test scenario:
            A bare "module 'digitalearth' has no attribute 'lisa_map'" sends a reader looking for a typo. The
            message was written to name geostatista and the ``choropleth(scheme="categorical")`` workaround,
            and the standard prefix was kept so ``hasattr`` still answers ``False``.
        """
        with pytest.raises(AttributeError) as excinfo:
            getattr(digitalearth, name)
        message = str(excinfo.value)
        missing = [
            wanted
            for wanted in (f"no attribute {name!r}", "geostatista", "choropleth")
            if wanted not in message
        ]
        assert missing == [], (
            f"the refusal for {name} names none of {missing}, got: {message}"
        )

    @pytest.mark.parametrize("name", digitalearth._REMOVED_GEOSTATISTICS)
    def test_a_removed_geostatistics_name_is_offered_by_neither_listing(self, name):
        """Neither ``__all__`` nor ``dir()`` may advertise a name that can only raise.

        Args:
            name: A removed geostatistics preset, or the removed submodule.
        """
        offered = [
            where
            for where, listing in (
                ("__all__", digitalearth.__all__),
                ("dir()", dir(digitalearth)),
            )
            if name in listing
        ]
        assert offered == [], (
            f"{name} is still offered by {offered}, and resolves to nothing"
        )

    def test_importing_the_package_is_silent(self):
        """A plain ``import digitalearth`` emits no ``DeprecationWarning``, in a fresh interpreter.

        Test scenario:
            With every deprecation path deleted there is nothing left to warn about, which is exactly the
            claim that would go unnoticed if a lazily resolved module started warning on import. Observed in
            a subprocess, because a first import cannot be observed twice in one.
        """
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


class TestTheDeletedShimsStayDeleted:
    """The modules this branch removed must not be importable by their full dotted path.

    Review M6: ``test_old_flat_module_paths_are_gone`` lists the six paths the ops split left behind and
    neither of these. An attribute hook on the package root says nothing about a module still importable by
    its dotted name, and a leftover ``static/geostatistics.py`` would keep importing geostatista — the
    dependency PR #180 dropped.
    """

    @pytest.mark.parametrize(
        "stale", ["digitalearth.scene", "digitalearth.static.geostatistics"]
    )
    def test_the_module_is_gone_from_disk(self, stale):
        """Importing the deleted module raises, rather than finding a file nothing references.

        Args:
            stale: The dotted path of a module this branch deleted.
        """
        with pytest.raises(ModuleNotFoundError) as excinfo:
            importlib.import_module(stale)
        assert stale in str(excinfo.value), (
            f"the failure must name {stale}, got: {excinfo.value}"
        )
