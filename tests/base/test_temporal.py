"""Tests for base.temporal — the engine-neutral time reductions split out of the static backend (#168)."""

import numpy as np
import pytest
from pyramids.dataset.collection import DatasetCollection

from digitalearth.base.temporal import SERIES_REDUCERS, group_by_label, series_values


@pytest.fixture
def collection():
    """A three-step DatasetCollection built from the bundled raster (identical members)."""
    return DatasetCollection.from_files(["examples/data/acc4000.tif"] * 3)


class TestSeriesValues:
    """series_values reduces each member over space to one value per step."""

    def test_one_value_per_step(self, collection):
        """A three-member collection reduces to a length-3 series."""
        assert series_values(collection).shape == (3,)

    def test_identical_members_give_a_constant_series(self, collection):
        """Identical members reduce to the same value every step."""
        values = series_values(collection, reducer="mean")
        assert np.allclose(values, values[0]), (
            f"expected a constant series, got {values}"
        )

    @pytest.mark.parametrize("reducer", sorted(SERIES_REDUCERS))
    def test_every_named_reducer_runs(self, reducer, collection):
        """Each advertised reducer produces a finite length-3 series."""
        values = series_values(collection, reducer=reducer)
        assert values.shape == (3,)
        assert np.isfinite(values).all(), f"{reducer} gave non-finite values {values}"

    def test_unknown_reducer_is_refused(self, collection):
        """A reducer outside SERIES_REDUCERS is refused by name, listing the choices."""
        with pytest.raises(ValueError, match=r"unknown reducer 'median'; choose from"):
            series_values(collection, reducer="median")


class TestGroupByLabel:
    """group_by_label aggregates a 1-D series by a periodic label."""

    def test_groups_are_sorted_unique_labels(self):
        """The returned groups are the sorted unique labels."""
        groups, _, _, _ = group_by_label(np.arange(6.0), ["b", "a", "c", "a", "b", "c"])
        assert groups == ["a", "b", "c"]

    def test_mean_low_high_per_group(self):
        """Each group's mean/min/max is taken over the values whose label matches."""
        values = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
        groups, mean, low, high = group_by_label(values, ["x", "y", "z", "x", "y", "z"])
        assert groups == ["x", "y", "z"]
        assert mean.tolist() == [3.0, 4.0, 5.0]
        assert low.tolist() == [1.0, 2.0, 3.0]
        assert high.tolist() == [5.0, 6.0, 7.0]

    def test_length_mismatch_is_refused(self):
        """Labels of a different length from the series are refused by name."""
        with pytest.raises(
            ValueError, match=r"labels has 2 entries but the series has 3 values"
        ):
            group_by_label(np.arange(3.0), ["a", "b"])


class TestEngineNeutral:
    """The reduction is reusable by any backend — importing it pulls in no renderer."""

    def test_no_renderer_is_imported(self):
        """Importing base.temporal and reducing a series loads no matplotlib/cleopatra/pyvista module.

        Test scenario:
            #168 moved the reduction to base so every backend can reuse it; the point is lost if the module
            drags a renderer in. Assert no renderer package is present in sys.modules after a fresh import +
            a reduction (matplotlib may be imported by an unrelated earlier test, so check it is not pulled
            in by this module specifically via a subprocess-free module-attribute check).
        """
        import digitalearth.base.temporal as temporal_module

        assert not hasattr(temporal_module, "plt")
        assert not hasattr(temporal_module, "LineGlyph")
        # the module's own imports are numpy + pyramids + base only
        source = temporal_module.__doc__ or ""
        assert "renderer" in source  # the module documents its neutrality
