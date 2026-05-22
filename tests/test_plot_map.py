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
