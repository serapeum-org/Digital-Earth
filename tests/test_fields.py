"""Tests for T1.1 — Map field methods (contourf/contour/pcolormesh/imshow/block) over a pyramids raster."""

import pytest

from digitalearth.scene import Map


@pytest.mark.parametrize("kind", ["imshow", "contourf", "contour", "pcolormesh", "block"])
def test_field_methods_render(dataset, kind):
    """Each field method renders the raster on the shared axes and registers one layer."""
    m = Map(crs=dataset.epsg)  # same CRS -> no reprojection needed
    getattr(m, kind)(dataset)
    assert len(m.layers) == 1
    # something was drawn (an image for imshow, a collection for the mesh/contour kinds)
    assert m.ax.images or m.ax.collections


def test_field_then_colorbar(dataset):
    """A field layer can emit a matching aggregated colorbar via the Scene."""
    m = Map(crs=dataset.epsg)
    m.contourf(dataset)
    m.colorbar()
    assert len(m.fig.axes) == 2  # main axes + colorbar axes


def test_field_accepts_cmap_and_levels(dataset):
    """cmap and discrete levels are forwarded into the glyph without tripping strict kwarg validation."""
    m = Map(crs=dataset.epsg)
    m.contourf(dataset, cmap="viridis", levels=5)
    assert len(m.layers) == 1


def test_field_filters_unknown_kwargs(dataset):
    """Unknown styling kwargs are filtered out (ArrayGlyph.filter_kwargs) rather than raising."""
    m = Map(crs=dataset.epsg)
    m.imshow(dataset, not_a_real_option=123)
    assert len(m.layers) == 1
