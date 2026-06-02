"""Tests for T1.1 — Map field methods (contourf/contour/pcolormesh/imshow/block) over a pyramids raster."""

import pytest

from digitalearth.scene import Map
from digitalearth.sources import get_source


@pytest.mark.parametrize("kind", ["imshow", "contourf", "contour", "pcolormesh", "block"])
def test_field_extent_matches_data_bbox(dataset, kind):
    """Each field method places the data at its true coordinate bbox (cleopatra wants [xmin,ymin,xmax,ymax]).

    Guards the extent-ordering bug where passing matplotlib-order [xmin,xmax,ymin,ymax] put the raster at the
    wrong location/zoom (x-span exploding to thousands of km).
    """
    src = get_source(dataset)
    xmin, xmax = float(src.x.values.min()), float(src.x.values.max())
    ymin, ymax = float(src.y.values.min()), float(src.y.values.max())
    m = Map(crs=dataset.epsg)
    getattr(m, kind)(dataset)
    xlim, ylim = sorted(m.ax.get_xlim()), sorted(m.ax.get_ylim())
    # axes span should match the data span (within a cell), not be wildly larger
    assert abs((xlim[1] - xlim[0]) - (xmax - xmin)) < 2 * dataset.cell_size, f"x-span off: {xlim}"
    assert abs((ylim[1] - ylim[0]) - (ymax - ymin)) < 2 * dataset.cell_size, f"y-span off: {ylim}"


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


def test_field_rejects_unknown_kwargs(dataset):
    """An unknown styling kwarg is forwarded to cleopatra, which raises loudly (no silent drop)."""
    m = Map(crs=dataset.epsg)
    with pytest.raises(ValueError):
        m.imshow(dataset, not_a_real_option=123)


def test_contour_labels(dataset):
    """contour(labels=True) draws inline isoline labels (cleopatra 0.14.0 / cleopatra#148, RP.6)."""
    m = Map(crs=dataset.epsg)
    m.contour(dataset, labels=True)
    assert len(m.ax.texts) > 0, "labels=True should add inline contour-label Text artists"


def test_contour_no_labels_by_default(dataset):
    """contour without labels= draws no inline labels (no behaviour change)."""
    m = Map(crs=dataset.epsg)
    m.contour(dataset)
    assert len(m.ax.texts) == 0, "default contour must not add label artists"
