"""Tests for T7.3 — ensemble/statistical series (spaghetti/envelope/quantiles/boxplot/multiboxplot/stripes)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from digitalearth import series
from digitalearth.scene import Map


def test_spaghetti_over_collection(dataset):
    """spaghetti contours every member of a DatasetCollection on one shared axes."""
    from pyramids.dataset.collection import DatasetCollection

    dc = DatasetCollection.from_files(["examples/data/acc4000.tif", "examples/data/acc4000.tif"])
    m = Map(crs=dataset.epsg)
    artists = m.spaghetti(dc)
    assert len(artists) == 2
    assert len(m.layers) == 2


class TestEnvelope:
    """Tests for series.envelope / quantile_band."""

    def test_envelope_fills_band(self):
        """envelope shades a PolyCollection between low and high."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        x = np.arange(10)
        _, _, pc = series.envelope(x, low=np.zeros(10), high=np.ones(10), ax=ax)
        assert pc in ax.collections

    def test_quantile_band_from_ensemble(self):
        """quantile_band shades the 10-90 band of a (members, points) ensemble."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        rng = np.random.default_rng(0)
        ensemble = rng.normal(size=(20, 12))
        _, _, pc = series.quantile_band(ensemble, lower=0.1, upper=0.9, ax=ax)
        assert pc in ax.collections


class TestStatistical:
    """Tests for series.boxplot / multiboxplot / stripes."""

    def test_boxplot(self):
        """boxplot renders a box-and-whisker from a 1-D sample."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        fig, out_ax, artists = series.boxplot(np.arange(50.0), ax=ax)
        assert out_ax is ax

    def test_multiboxplot(self):
        """multiboxplot renders one box per group."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        groups = [np.arange(10.0), np.arange(5.0, 15.0), np.arange(2.0, 12.0)]
        fig, out_ax, artists = series.multiboxplot(groups, ax=ax)
        assert out_ax is ax

    def test_stripes(self):
        """stripes renders a warming-stripes bar strip from a 1-D series."""
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        fig, out_ax, bars = series.stripes(np.linspace(-1.0, 1.0, 30), ax=ax)
        assert len(bars) == 30
