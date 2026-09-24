"""Hold the `all` environment to the one thing it exists for: three rendering engines in one process.

Every other environment carries exactly one engine — `web` has MapLibre, `interactive` has HoloViz, `viz3d`
has PyVista, and none has another's. That is why
:mod:`tests.base.test_map_conformance` holds each tier against a shared **constant** rather than against the
other tier: a probe comparing two live tiers in one process had nowhere to run.

The `all` environment is that place. It resolves all three backend features in its own solve-group, so the
single-backend environments' versions cannot move when it does.

**A solve is not an import.** Three stacks can resolve to compatible version ranges and still fail to load
together — VTK, Bokeh and MapLibre each bring native libraries. So the checks here are deliberately ordered by
how expensive it is to be wrong about them: the engines import, then the facades construct, then the two 2-D
tiers are asked to describe the same figure and agree.

The declaration checks run **everywhere**, including the single-backend environments, so this module cannot
become another contract that passes because nothing collected it (#324). The live checks skip loudly where the
engines are absent.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Tuple

import pytest

from digitalearth.base.registry import band_of

#: The features the `all` environment must carry, one per rendering backend plus the test tooling.
REQUIRED_FEATURES = ("dev", "3d", "interactive", "web")

#: The manifest, read rather than imported so these checks need no environment of their own.
MANIFEST = Path(__file__).resolve().parents[2] / "pyproject.toml"

#: Each rendering engine, with the symbol the tier built on it reaches for first —
#: `three_d/base.py` builds a `pyvista.Plotter`, `interactive/renderer.py` reads `holoviews.Store`, and
#: `web/base.py` imports `maplibre.Layer`.
#:
#: The import alone was asserted by comparing each module's `__name__` against its own name, which is the
#: name it was imported under and can be nothing else (`R2-N3`). What that cannot tell apart from a working
#: install is a module that is there in name only: a same-named package earlier on the path, a namespace
#: package left by a half-removed install, or a stub some other test put in `sys.modules`. Naming a symbol
#: the tier really uses does tell them apart, and it is also the thing that breaks when an engine moves its
#: API — which, in the one environment that pins three engines together, is what a version bump looks like.
ENGINE_ENTRY_POINTS: Tuple[Tuple[str, str], ...] = (
    ("pyvista", "Plotter"),
    ("holoviews", "Store"),
    ("maplibre", "Layer"),
)

#: What both 2-D tiers must say about the seed figure: a graticule under points, both shown.
#:
#: Held as a constant rather than as one tier's live answer so a single wrong tier cannot make the pair agree.
#:
#: The band is the *resolved* one, not `LayerSpec.band`. Both tiers leave that field unset and let the kind
#: decide, so a constant expecting `None` there asserted nothing at all — the vacuous comparison
#: :func:`tests.base.test_map_conformance._described` already refuses to make, and for the reason it gives:
#: "the band is the layer's own override where it set one and its kind's band otherwise … comparing the raw
#: field would say they agree while saying nothing" (review R-L1).
EXPECTED_DESCRIPTION: Tuple[Tuple[str, Any, bool], ...] = (
    ("graticule", "reference", True),
    ("points", "data", True),
)


def _environments() -> dict:
    """Return the pixi environment table from the manifest.

    Returns:
        The `[tool.pixi.environments]` table.
    """
    with MANIFEST.open("rb") as handle:
        return tomllib.load(handle)["tool"]["pixi"]["environments"]


def _seed_points():
    """Build the small point layer both tiers are asked to describe.

    Returns:
        A four-row point `GeoDataFrame` in EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
        crs="EPSG:4326",
    )


def _described(tier) -> Tuple[Tuple[str, Any, bool], ...]:
    """Draw the seed figure on one tier and reduce it to what every tier should agree on.

    Args:
        tier: A constructed 2-D map.

    Returns:
        One `(kind, band, visible)` per described layer, in draw order. The band is the layer's own
        override where it set one and its kind's band otherwise — the same reduction
        :func:`tests.base.test_map_conformance._described` makes, and for the same reason: a tier that
        leaves `band` unset is not thereby drawing somewhere else, and both tiers leave it unset here, so
        comparing the raw field said the two agreed while saying nothing at all (review R-L1).
    """
    tier.points(_seed_points())
    tier.graticule()
    return tuple(
        (layer.kind, layer.band or band_of(layer.kind), layer.visible)
        for layer in tier.figure_spec.layers
    )


class TestTheEnvironmentIsDeclared:
    """The manifest half — these run in every environment, so the module is never silently uncollected."""

    def test_the_all_environment_exists(self):
        """Without it there is nowhere to compare two live tiers, and the suite falls back to constants."""
        assert "all" in _environments(), (
            "no `all` environment in [tool.pixi.environments]; cross-backend comparison has nowhere to run"
        )

    @pytest.mark.parametrize("feature", REQUIRED_FEATURES)
    def test_the_all_environment_carries_every_backend(self, feature):
        """A missing feature turns this into another single-backend environment.

        Args:
            feature: The pixi feature that installs one backend's engine.
        """
        carried = _environments()["all"]["features"]
        assert feature in carried, (
            f"`all` is missing the {feature!r} feature; it carries {carried}"
        )

    def test_the_all_environment_solves_on_its_own(self):
        """Its own solve-group is what stops it dragging the single-backend environments' versions.

        Test scenario:
            `all` resolves three stacks that no other environment resolves together. Sharing a solve-group
            would make every one of those version choices bind `web`, `interactive` and `viz3d` too, so a
            conflict in the combined environment would surface as a change in three unrelated ones.
        """
        environments = _environments()
        shared = [
            name
            for name, spec in environments.items()
            if name != "all"
            and spec.get("solve-group") == environments["all"].get("solve-group")
        ]
        assert shared == [], (
            f"`all` shares its solve-group with {shared}; it must resolve alone"
        )


class TestTheEnginesCoexist:
    """The runtime half — a solve proves versions, this proves imports."""

    def test_every_engine_imports_in_one_process(self):
        """Three native stacks loading together is the premise; nothing else here works without it.

        Test scenario:
            The import is the real check — a module whose native library fails to load raises
            `ImportError`, and `importorskip` turns that into a **failure** here rather than a skip, which
            was measured. What is asserted on top is the entry point each tier reaches for, because a name
            in `sys.modules` is not an engine: it can be a stub, a shadowing package or the remains of a
            half-removed install, and every one of those satisfied the `__name__` comparison this replaces
            (`R2-N3`).
        """
        loaded = {name: pytest.importorskip(name) for name, _ in ENGINE_ENTRY_POINTS}
        missing = sorted(
            f"{name}.{entry}"
            for name, entry in ENGINE_ENTRY_POINTS
            if not hasattr(loaded[name], entry)
        )
        assert missing == [], (
            f"these engines imported without the entry point their tier uses: {missing}; the module is on "
            "the path but it is not the engine"
        )

    def test_every_facade_constructs_in_one_process(self):
        """A facade that imports but cannot be built would fail a cross-tier probe for the wrong reason.

        Test scenario:
            Comparing each built object's class name against that class's own name says only that `Map()`
            returns a `Map` (`R2-N3`). What the probes below actually need is a facade that is *ready to be
            asked*, so each fresh map is asked the question they ask — how many layers it describes — and
            has to answer none rather than raise.
        """
        pytest.importorskip("pyvista")
        pytest.importorskip("geoviews")
        pytest.importorskip("maplibre")

        from digitalearth.interactive.map import InteractiveMap
        from digitalearth.static.map import Map
        from digitalearth.three_d import Scene3D
        from digitalearth.web.map import WebMap

        described = [
            len(build().figure_spec.layers)
            for build in (Map, InteractiveMap, WebMap, Scene3D)
        ]
        assert described == [0, 0, 0, 0], (
            f"a fresh map on each tier must describe no layers, and they described {described}"
        )


class TestTwoTiersDescribeOneFigureAlike:
    """The comparison the environment exists for, run live rather than against a recorded answer."""

    def test_the_web_tier_describes_the_seed_figure_as_expected(self):
        """One half of the pair, held to the constant so a wrong pair cannot agree its way to green."""
        pytest.importorskip("maplibre")
        from digitalearth.web.map import WebMap

        assert _described(WebMap()) == EXPECTED_DESCRIPTION

    def test_the_interactive_tier_describes_the_seed_figure_as_expected(self):
        """The other half, held to the same constant."""
        pytest.importorskip("geoviews")
        from digitalearth.interactive.map import InteractiveMap

        assert _described(InteractiveMap()) == EXPECTED_DESCRIPTION

    def test_the_two_tiers_agree_with_each_other_in_one_process(self):
        """The check no single-backend environment can run: two live tiers, one interpreter, one answer.

        Test scenario:
            `test_map_conformance.py` asks each tier this separately, in its own CI job, against a shared
            constant — because until the `all` environment existed no interpreter held both engines. This is
            the direct form, and it is the reason the environment is declared.
        """
        pytest.importorskip("maplibre")
        pytest.importorskip("geoviews")
        from digitalearth.interactive.map import InteractiveMap
        from digitalearth.web.map import WebMap

        assert _described(WebMap()) == _described(InteractiveMap())
