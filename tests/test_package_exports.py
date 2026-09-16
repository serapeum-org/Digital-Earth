"""Tests for the top-level digitalearth public surface (PC-4)."""

import importlib
import importlib.metadata
import importlib.util
import pathlib

import pytest

import digitalearth

EXPECTED = [
    "quickplot",
    "quickmap",
    "Map",
    "Scene",
    "TexturedGlobe",
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
            ``__path__`` exists only on packages — the check that distinguishes these from
            ``digitalearth.scene``, which the split deliberately turned into a plain shim module.
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
