"""``scheme=``/``k=`` on a raster field: the classes are cut, drawn, published and keyed (ST-7, #222).

A classified raster used to be impossible — cleopatra's ``ArrayGlyph.plot`` took no ``classify`` group, so
``render_compat`` refused ``scheme``/``k`` on every raster layer. Since cleopatra 0.38 it does, and the static
tier needed no new code: ``field`` forwards the flat keys, ``render_compat`` folds them into the ``classify``
group, and the scene reads the class edges back off the drawn norm. Nothing pinned that path, though, so a
renamed upstream keyword would have turned every classified raster back into a refusal with no failing test.

These tests hold each link of the chain to edges computed **independently from the data** — never read back
from the figure under test — so they cannot agree with the drawing by construction.
"""

import numpy as np
import pytest
from matplotlib.colors import BoundaryNorm

from digitalearth.static import Map


@pytest.fixture
def values(dataset) -> np.ndarray:
    """Return the fixture raster's valid cell values (nodata masked out), as a flat float array.

    Args:
        dataset: The ``acc4000.tif`` raster fixture.

    Returns:
        The cell values the classifier sees.
    """
    band = dataset.read_array(masked=True)
    return np.ma.compressed(band).astype(float)


def _equal_interval_edges(values: np.ndarray, k: int) -> list:
    """Return the ``k`` equal-interval class edges over ``values``, computed straight from the data."""
    return list(np.linspace(values.min(), values.max(), k + 1))


def _drawn_norm(m: Map):
    """Return the norm of the last image drawn on ``m``'s axes."""
    return m.ax.images[-1].norm


class TestClassifiedRasterField:
    """``Map.field`` with ``scheme=``/``k=`` draws, describes and keys discrete classes."""

    def test_a_named_scheme_cuts_the_raster_into_classes(self, dataset, values):
        """``scheme="equal_interval", k=4`` draws the band with a ``BoundaryNorm`` on the data's own edges.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values.

        Test scenario:
            The mappable must carry discrete classes — a ``BoundaryNorm`` — whose boundaries are the four
            equal-width intervals over the data's range, computed here from the values themselves.
        """
        expected = _equal_interval_edges(values, 4)
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=4)
            norm = _drawn_norm(m)
            assert isinstance(norm, BoundaryNorm), (
                f"a classified raster should draw a BoundaryNorm, got {norm!r}"
            )
            assert np.allclose(norm.boundaries, expected), (
                f"class edges {list(norm.boundaries)} should be the equal intervals {expected}"
            )

    def test_explicit_edges_are_used_as_given(self, dataset):
        """A list passed as ``scheme`` is taken as the class edges themselves.

        Args:
            dataset: The raster fixture.

        Test scenario:
            ``scheme=[0, 10, 50, 88]`` must draw exactly those boundaries — no re-cutting.
        """
        edges = [0.0, 10.0, 50.0, 88.0]
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme=edges)
            boundaries = list(_drawn_norm(m).boundaries)
        assert boundaries == edges, (
            f"explicit edges {edges} were redrawn as {boundaries}"
        )

    def test_no_scheme_stays_a_continuous_ramp(self, dataset):
        """Without ``scheme`` the field is a continuous ramp, not discrete classes.

        Args:
            dataset: The raster fixture.

        Test scenario:
            The default must be unchanged by this feature: no ``BoundaryNorm``.
        """
        with Map(crs=dataset.epsg) as m:
            m.field(dataset)
            norm = _drawn_norm(m)
        assert not isinstance(norm, BoundaryNorm), (
            f"an unclassified raster should be continuous, got {norm!r}"
        )

    def test_the_classes_are_published_as_breaks(self, dataset, values):
        """The cut edges reach ``last_breaks`` and the layer's colour encoding.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values.

        Test scenario:
            Other tiers and the figure description read the classes from these two places, so both must
            carry the edges that were drawn.
        """
        expected = _equal_interval_edges(values, 4)
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=4)
            breaks = list(m.last_breaks)
            spec = m.figure_spec.layers.get(m.layer_ids[-1])
            encoded = list(spec.symbology.encoding("color").scale.breaks)
        assert np.allclose(breaks, expected), (
            f"last_breaks {breaks} should be {expected}"
        )
        assert np.allclose(encoded, expected), (
            f"the colour encoding's breaks {encoded} should be {expected}"
        )

    def test_the_colorbar_steps_at_the_class_edges(self, dataset, values):
        """``colorbar()`` on a classified raster puts its ticks on the class edges.

        Args:
            dataset: The raster fixture.
            values: The fixture's valid cell values.

        Test scenario:
            A stepped key, not a continuous ramp: one tick per class edge.
        """
        expected = _equal_interval_edges(values, 4)
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=4)
            m.colorbar()
            cax = [ax for ax in m.fig.axes if ax is not m.ax][-1]
            ticks = list(cax._colorbar.get_ticks())
        assert np.allclose(ticks, expected), (
            f"colorbar ticks {ticks} should sit on the class edges {expected}"
        )

    def test_the_legend_draws_one_swatch_per_class(self, dataset):
        """``legend()`` on a classified raster lists one swatch per class.

        Args:
            dataset: The raster fixture.

        Test scenario:
            ``k=4`` equal intervals give four classes, so the class legend has four entries.
        """
        with Map(crs=dataset.epsg) as m:
            m.field(dataset, scheme="equal_interval", k=4)
            m.legend()
            labels = [text.get_text() for text in m.ax.get_legend().get_texts()]
        assert len(labels) == 4, (
            f"four classes should give four legend entries, got {labels}"
        )
