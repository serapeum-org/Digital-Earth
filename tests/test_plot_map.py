import numpy as np
import pytest
from geopandas.geodataframe import GeoDataFrame
from matplotlib.figure import Figure
from pyramids.dataset import Dataset
from digitalearth.static import StaticGlyph


class TestPlotArray:
    def test_plot(self, dataset: Dataset):
        fig, ax = StaticGlyph.plot(dataset, title="Flow Accumulation")
        assert isinstance(fig, Figure)

    def test_plot_with_points(
        self,
        dataset: Dataset,
        display_cellvalue: bool,
        points: GeoDataFrame,
        num_size: int,
        background_color_threshold,
        ticks_spacing: int,
        pid_size: int,
        pid_color: str,
        point_size: int,
        point_color: str,
    ):
        fig, ax = StaticGlyph.plot(
            dataset,
            point_color=point_color,
            point_size=point_size,
            pid_color=pid_color,
            pid_size=pid_size,
            points=points,
            display_cell_value=display_cellvalue,
            num_size=num_size,
            background_color_threshold=background_color_threshold,
            ticks_spacing=ticks_spacing,
        )

        assert isinstance(fig, Figure)


class TestStaticGlyphPlotEdges:
    """Edge cases of StaticGlyph (construction, ndarray input guard, id-labelled points)."""

    def test_init(self):
        """StaticGlyph is instantiable (covers the trivial constructor)."""
        assert isinstance(StaticGlyph(), StaticGlyph), "StaticGlyph() should construct"

    def test_plot_ndarray_requires_no_data_value(self):
        """A bare ndarray without the required ``no_data_value`` kwarg raises ValueError.

        Test scenario:
            Passing a numpy array as ``src`` without ``no_data_value`` must raise a clear ValueError
            naming the missing kwarg.
        """
        with pytest.raises(ValueError, match="no_data_value"):
            StaticGlyph.plot(np.ones((4, 4), dtype="float32"))

    def test_plot_ndarray_with_no_data_value(self):
        """A bare ndarray plots when ``no_data_value`` is supplied (and is not forwarded to ArrayGlyph).

        Test scenario:
            ``no_data_value`` is popped from kwargs, so the masked array renders without ArrayGlyph
            rejecting an unexpected keyword.
        """
        arr = np.ones((4, 4), dtype="float32")
        arr[0, 0] = -9999.0
        fig, ax = StaticGlyph.plot(arr, no_data_value=-9999.0)
        assert isinstance(fig, Figure), "expected a Figure from the ndarray path"

    def test_plot_points_with_id_column(self, dataset: Dataset, points: GeoDataFrame):
        """Points carrying an ``id`` column are labelled by that column, not the index.

        Test scenario:
            With an ``id`` column present the id-branch of the label lookup is taken; the call renders a figure.
        """
        pts = points.copy()
        pts["id"] = [f"g{i}" for i in range(len(pts))]
        fig, ax = StaticGlyph.plot(dataset, points=pts)
        assert isinstance(fig, Figure), "expected a Figure with id-labelled points"
