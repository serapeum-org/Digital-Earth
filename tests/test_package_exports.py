"""Tests for the top-level digitalearth public surface (PC-4)."""
import importlib

import pytest

import digitalearth


EXPECTED = [
    "quickplot", "quickmap", "Map", "Scene", "TexturedGlobe", "grid", "shared_colorbar", "projections",
    "get_source", "Source", "DimensionInfo",
    "line", "bar", "histogram", "scatter", "bar_by", "line_by", "statistics",
    "envelope", "quantile_band", "boxplot", "multiboxplot", "stripes",
    "TimeSeries", "Climatology",
    "Batch", "gallery", "load_plugins",
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
            ``digitalearth.TimeSeries`` is ``digitalearth.temporal.TimeSeries`` (no shadow/duplicate).
        """
        temporal = importlib.import_module("digitalearth.temporal")
        assert digitalearth.TimeSeries is temporal.TimeSeries, "TimeSeries is not the temporal one"
        assert digitalearth.Climatology is temporal.Climatology, "Climatology is not the temporal one"

    def test_projections_is_the_submodule(self):
        """The re-exported ``projections`` is the scene.projections submodule.

        Test scenario:
            ``digitalearth.projections`` resolves a known projection factory (web_mercator -> 3857).
        """
        assert digitalearth.projections.get("web_mercator") == 3857, "projections submodule not wired"
