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

#: The features the `all` environment must carry, one per rendering backend plus the test tooling.
REQUIRED_FEATURES = ("dev", "3d", "interactive", "web")

#: The manifest, read rather than imported so these checks need no environment of their own.
MANIFEST = Path(__file__).resolve().parents[2] / "pyproject.toml"

#: What both 2-D tiers must say about the seed figure: a graticule under points, both shown.
#:
#: Held as a constant rather than as one tier's live answer so a single wrong tier cannot make the pair agree.
EXPECTED_DESCRIPTION: Tuple[Tuple[str, Any, bool], ...] = (
    ("graticule", None, True),
    ("points", None, True),
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
        One `(kind, band, visible)` per described layer, in draw order.
    """
    tier.points(_seed_points())
    tier.graticule()
    return tuple(
        (layer.kind, layer.band, layer.visible) for layer in tier.figure_spec.layers
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
        """Three native stacks loading together is the premise; nothing else here works without it."""
        pyvista = pytest.importorskip("pyvista")
        holoviews = pytest.importorskip("holoviews")
        maplibre = pytest.importorskip("maplibre")
        loaded = [pyvista.__name__, holoviews.__name__, maplibre.__name__]
        assert sorted(loaded) == ["holoviews", "maplibre", "pyvista"], (
            f"loaded {loaded}"
        )

    def test_every_facade_constructs_in_one_process(self):
        """A facade that imports but cannot be built would fail a cross-tier probe for the wrong reason."""
        pytest.importorskip("pyvista")
        pytest.importorskip("geoviews")
        pytest.importorskip("maplibre")

        from digitalearth.interactive.map import InteractiveMap
        from digitalearth.static.map import Map
        from digitalearth.three_d import Scene3D
        from digitalearth.web.map import WebMap

        built = [
            type(build()).__name__ for build in (Map, InteractiveMap, WebMap, Scene3D)
        ]
        assert built == ["Map", "InteractiveMap", "WebMap", "Scene3D"], f"built {built}"


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
