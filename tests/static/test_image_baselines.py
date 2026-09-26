"""Visual-regression baselines for the static (matplotlib) backend — the pixels, not the scaffolding.

Every other module under ``tests/static/`` asserts on the scaffolding a render leaves behind: artist counts,
norms, extents, axis limits. All of those pass just as happily when the figure comes out a grey rectangle.
This module is the other half — each test builds one representative figure and hands it back so
``pytest-mpl`` compares it, pixel for pixel, against a committed baseline PNG under ``tests/baseline/``.

The figures are chosen for what a refactor of this backend actually moves: the raster field methods, the
vector layers, the figure-level decoration (colorbar, categorical legend), the projection frame with its
graticule, the Natural-Earth overlays and the composites. When one of those stops looking the way it looks
today, one of these fails.

Running and regenerating:
    - Compare against the committed baselines: ``pixi run -e dev test-images``.
    - Regenerate them after a deliberate rendering change: ``pixi run -e dev test-images-generate``
      (``pytest -m mpl --mpl-generate-path=tests/baseline``), then review the PNG diff before committing.

Sensitivity:
    A baseline is only worth its bytes if a real change moves it further than the tolerance. Measured on the
    pinned stack (each figure re-rendered unchanged, then with one semantic change, RMS against the committed
    PNG): the unchanged re-render scores exactly **0.0** for all 18, and the weakest real change measured is
    **3.96** (one of the two shared-colorbar panels going from 6 contour levels to 7). Between them sit a
    moved point (4.4), a dropped graticule (4.4), a dropped border layer (5.3-5.6) and a dropped colorbar
    label (7.8-9.3); the colormap swaps run 9.7-68.2 and a reversed RGB band order 94.4. ``test_scatter_points``
    and ``test_grid_cells`` are the sparsest figures here, so a change confined to them scores lowest — keep
    that in mind before loosening ``mpl-default-tolerance`` (1, i.e. a quarter of the weakest of those), and
    prefer a per-figure ``@pytest.mark.mpl_image_compare(tolerance=...)`` to moving the default.

Determinism:
    Every figure is a fixed size in inches at 80 dpi (``_BASELINE``), rendered head-less on Agg
    (``MPLBACKEND=Agg``, set in ``[tool.pytest.ini_options].env``) under matplotlib's ``default`` style, so a
    contributor's ``matplotlibrc`` cannot move the output. Nothing here reads the clock, the locale, a random
    seed or the network: the rasters are the committed ``examples/data`` fixtures or small arrays built in
    the test, and the Natural-Earth overlays come from the committed ``tests/data/naturalearth`` assets that
    the session fixture in ``conftest.py`` copies into cleopatra's cache.
"""

import numpy as np
import pytest
from pyramids.dataset import Dataset, GeoReference
from pyramids.feature import FeatureCollection

from digitalearth.static import Map, grid, projections, shared_colorbar

#: Every test in this module is an image test, selected with ``-m mpl`` and compared with ``--mpl``.
pytestmark = pytest.mark.mpl

#: Shared ``mpl_image_compare`` kwargs: one fixed dpi for every baseline, and matplotlib's own defaults as
#: the style, so the committed PNGs depict what the backend really renders rather than the frozen
#: ``classic`` look. The tolerance comes from ``mpl-default-tolerance`` in pyproject.toml.
_BASELINE = {"savefig_kwargs": {"dpi": 80}, "style": "default"}

#: Figure size in inches for the single-panel maps (the multi-panel grid overrides it).
_FIGSIZE = (4.0, 4.0)


@pytest.fixture
def polygons():
    """Buffered polygons carrying a numeric ``fid`` and a nominal ``zone`` column.

    The buffer is wide enough (3 km against a ~50 km extent) that the fills cover a real share of the frame.
    At the 500 m other tests use, the polygons render as dots and a whole changed colormap moves the image by
    an RMS of ~3 — inside the noise a looser tolerance would have to allow.

    Returns:
        FeatureCollection: polygon geometries in the point fixture's CRS, with a numeric ``fid`` column for a
        continuous fill and a two-class ``zone`` column for a categorical one.
    """
    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc["geometry"] = fc.geometry.buffer(3000.0)
    fc["zone"] = ["urban", "rural"] * (len(fc) // 2) + ["urban"] * (len(fc) % 2)
    return fc


@pytest.fixture
def point_features():
    """The committed point fixture as a pyramids FeatureCollection.

    Returns:
        FeatureCollection: the ``tests/data/points.geojson`` points.
    """
    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def rgb_dataset(dataset):
    """A three-band raster whose bands carry three genuinely different pictures on one grid.

    ``rgb_composite`` stretches each band independently, so bands that are scalar multiples of one another
    — the ``[b, 0.5b, 0.25b]`` this fixture used to stack — normalise to the *same* channel and the
    composite comes out pure greyscale. A baseline built from such a stack scores RMS 0.0 against a
    reversed band order, an all-identical stack and a re-scaled stretch alike: against every regression the
    test claims to pin. These three share only the grid and the CRS — the catchment field itself, a
    west-east ramp and concentric rings about the grid centre — so the channels are independent and the
    band axis is legible in the pixels.

    The ``acc4000`` sentinel (half its 13x14 cells) is replaced with zero rather than left in: ``from_array``
    stamps its own nodata on the stack, so the sentinel would otherwise survive into the 2-98 percentile
    stretch and flatten the one band that carries real data.

    Args:
        dataset: The module's single-band ``acc4000`` raster — the grid, the CRS and the red channel.

    Returns:
        Dataset: a 3-band pyramids ``Dataset`` on the same grid and CRS.
    """
    base = dataset.read_array(band=0).astype("float32")
    base = np.where(base == np.float32(dataset.no_data_value[0]), np.float32(0.0), base)
    rows, cols = base.shape
    y = np.linspace(-1.0, 1.0, rows, dtype="float32")[:, None]
    x = np.linspace(-1.0, 1.0, cols, dtype="float32")[None, :]
    ramp = np.broadcast_to(x, base.shape).astype("float32")
    rings = np.sin(4.0 * np.pi * np.hypot(y, x)).astype("float32")
    stack = np.stack([base, ramp, rings]).astype("float32")
    return Dataset.from_array(
        arr=stack, geo_ref=GeoReference(geo=dataset.geotransform, epsg=dataset.epsg)
    )


@pytest.fixture
def uv():
    """A small, smooth (u, v) pair on an increasing-y EPSG:4326 grid.

    Returns:
        tuple[Dataset, Dataset]: the u and v components, both single-band.
    """
    ny, nx = 6, 8
    u = np.ones((ny, nx), dtype="float32")
    v = np.linspace(-1.0, 1.0, ny, dtype="float32")[:, None] * np.ones(
        (1, nx), "float32"
    )
    geo = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return (
        Dataset.from_array(arr=u, geo_ref=GeoReference(geo=geo, epsg=4326)),
        Dataset.from_array(arr=v, geo_ref=GeoReference(geo=geo, epsg=4326)),
    )


@pytest.fixture
def global_raster():
    """A coarse whole-world raster, so a globe render has something visible on it.

    The committed ``acc4000`` fixture covers a few tens of kilometres, which is sub-pixel once it is warped
    onto a whole-globe orthographic frame — a baseline built from it pins the frame and nothing else. This is
    a smooth analytic field on a 3-degree lon/lat grid instead, large enough that the warp itself is under
    the baseline.

    Returns:
        Dataset: a single-band EPSG:4326 raster spanning the full lon/lat domain.
    """
    lat = np.linspace(88.5, -88.5, 60, dtype="float32")[:, None]
    lon = np.linspace(-178.5, 178.5, 120, dtype="float32")[None, :]
    arr = (np.cos(np.radians(lat)) * np.sin(np.radians(2.0 * lon))).astype("float32")
    geo = (-180.0, 3.0, 0.0, 90.0, 0.0, -3.0)
    return Dataset.from_array(arr=arr, geo_ref=GeoReference(geo=geo, epsg=4326))


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_imshow_field(dataset):
    """A raster drawn as an image is placed and coloured as it is today.

    Test scenario:
        The cheapest field path — no reprojection, no resampling — so a change here is a change in how the
        backend hands an array and its extent to the glyph.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=dataset.epsg, figsize=_FIGSIZE)
    m.field(dataset, cmap="viridis")
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_contourf_levels(dataset):
    """A filled-contour field with discrete levels keeps its class boundaries.

    Test scenario:
        ``levels`` is the kwarg most likely to be re-plumbed when the field methods move, and a silently
        dropped one changes the picture without changing any artist count.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=dataset.epsg, figsize=_FIGSIZE)
    m.contours(dataset, cmap="terrain", levels=6, filled=True)
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_contour_lines(dataset):
    """Line contours keep their isolines where they are.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=dataset.epsg, figsize=_FIGSIZE)
    m.contours(dataset, cmap="plasma")
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_pcolormesh_field(dataset):
    """The quadrilateral-mesh field keeps its cell geometry.

    Test scenario:
        ``pcolormesh`` is the one field method that draws per-cell quads from the coordinate vectors rather
        than from an extent, so an off-by-one in the cell-edge maths shows up here and nowhere else.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=dataset.epsg, figsize=_FIGSIZE)
    m.pcolormesh(dataset, cmap="magma")
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_reprojected_field(dataset):
    """A raster reprojected to the display CRS lands where it lands today.

    Test scenario:
        Same data as :func:`test_imshow_field`, but drawn on a Web-Mercator map, so the warp itself is under
        the baseline rather than only the placement of an already-matching grid.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=3857, figsize=_FIGSIZE)
    m.field(dataset, cmap="viridis")
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_field_with_colorbar(dataset):
    """A field plus its aggregated colorbar, label and all.

    Test scenario:
        The colorbar is figure-level decoration that steals space from the axes, so it constrains both the
        bar and the layout the map is left with.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=dataset.epsg, figsize=_FIGSIZE)
    m.field(dataset, cmap="cividis")
    m.colorbar(label="flow accumulation")
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_categorical_legend(dataset):
    """A swatch legend attached over a field keeps its colours, order and placement.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=dataset.epsg, figsize=_FIGSIZE)
    m.field(dataset, cmap="Greys")
    m.legend(["#1f77b4", "#d62728", "#2ca02c"], ["river", "ridge", "plain"])
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_grid_cells(dataset):
    """Raster cells drawn as one polygon each keep their fill and outline.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=dataset.epsg, figsize=_FIGSIZE)
    m.grid_cells(dataset, cmap="YlGnBu")
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_scatter_points(point_features):
    """A point layer keeps its marker positions, size and colour scale.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=point_features.epsg, figsize=_FIGSIZE)
    m.points(point_features)
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_choropleth_continuous(polygons):
    """Polygons filled from a numeric column keep their continuous colour scale.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=polygons.epsg, figsize=_FIGSIZE)
    m.choropleth(polygons, column="fid", cmap="viridis")
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_choropleth_categorical(polygons):
    """A categorical choropleth keeps one colour per class and the swatch legend that keys them.

    Test scenario:
        ``scheme="categorical"`` swaps the continuous norm for a ``BoundaryNorm`` over class codes and draws
        its own legend instead of a colorbar — two pieces of styling a refactor can plausibly reorder, and
        neither of which any artist-count assertion would notice.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=polygons.epsg, figsize=_FIGSIZE)
    m.choropleth(polygons, column="zone", scheme="categorical", cmap="tab10")
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_kde_density(point_features):
    """The kernel-density surface over the point fixture keeps its shape and shading.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=point_features.epsg, figsize=_FIGSIZE)
    m.kde(point_features, cmap="magma")
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_quiver_field(uv):
    """A u/v vector field keeps its arrow placement, length and orientation.

    Returns:
        Figure: the rendered map.
    """
    u_ds, v_ds = uv
    m = Map(crs=4326, figsize=_FIGSIZE)
    m.quiver(u_ds, v_ds)
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_rgb_composite(rgb_dataset):
    """A three-band composite keeps its channel order and per-channel stretch.

    Test scenario:
        The composite is the one raster path whose output is an ``(rows, cols, 3)`` array rather than a
        scalar field through a colormap, so a transposed band axis or a moved stretch is visible only in the
        pixels. Measured against the committed baseline: reversing the band order to ``(3, 2, 1)`` scores
        RMS 94.4, swapping green and blue 72.5, collapsing the three bands to one repeated band 90.1, and
        freezing the 2-98 percentile stretch to fixed ``limits`` 7.2 — all far above the tolerance.

        Rescaling a band by a positive scalar is *not* among them, and cannot be: the per-band percentile
        stretch normalises it away by design, so ``[b, 0.9b, 0.1b]`` is the same picture. That is the
        contract, not a hole — which is why the fixture's bands are three different pictures rather than one
        picture scaled three ways.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=rgb_dataset.epsg, figsize=_FIGSIZE)
    m.rgb_composite(rgb_dataset)
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_globe_frame_with_graticule(global_raster):
    """An orthographic globe keeps its boundary, graticule and clipped data.

    Test scenario:
        The frame, the meridians/parallels and the clip are all applied at ``render()`` time from cached
        projected geometry — the part of the backend with the most moving pieces and the least to assert on.
        The field underneath is the whole-world raster, so the warp into the projection is pinned too and the
        clip has something to cut.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=projections.orthographic(lon=-9, lat=39), globe=True, figsize=_FIGSIZE)
    m.field(global_raster, cmap="viridis")
    m.graticule(lon_step=30, lat_step=30)
    m.render()
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_globe_coastlines_and_borders(global_raster):
    """A globe with Natural-Earth coastlines and borders keeps its limb-split polylines.

    Test scenario:
        On a globe the overlays are projected per line and split where they cross the limb, so this pins
        both the reference geometry and the splitting that keeps the far side off the figure. The field
        below them is drawn in grey so a shifted coastline stays legible against it.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=projections.orthographic(lon=10, lat=25), globe=True, figsize=_FIGSIZE)
    m.field(global_raster, cmap="Greys")
    m.coastlines(resolution="110m")
    m.borders(resolution="110m")
    m.render()
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_flat_coastlines_over_domain():
    """A flat lon/lat map keeps its reprojected land fill, coastlines and borders.

    Test scenario:
        The flat path goes through cleopatra's ``add_features`` rather than the globe's per-line projection,
        so it is a different code path over the same committed geometry.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=4326, domain="europe", figsize=_FIGSIZE)
    m.land(resolution="110m")
    m.coastlines(resolution="110m")
    m.borders(resolution="110m")
    m.set_domain()
    return m.fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_grid_panels_with_shared_colorbar(dataset):
    """Two panels sharing one figure and one colorbar keep their layout.

    Test scenario:
        ``grid`` + ``shared_colorbar`` is the only multi-panel product in this backend, and the colorbar
        steals space from both panels, so the baseline pins the whole figure layout rather than one axes.

    Returns:
        Figure: the shared figure.
    """
    fig, maps = grid(1, 2, crs=dataset.epsg, figsize=(6.0, 3.0))
    first = maps[0].field(dataset, cmap="viridis")
    maps[1].contours(dataset, cmap="viridis", levels=6, filled=True)
    shared_colorbar(fig, first, maps, label="accumulation")
    return fig


@pytest.mark.mpl_image_compare(**_BASELINE)
def test_composed_field_and_contours(dataset):
    """Two clearing glyph renders on one axes both survive into the picture.

    Test scenario:
        The other 18 baselines each put at most one clearing glyph on an axes — the multi-layer ones layer a
        graticule, a Natural-Earth overlay or a basemap over a single render, and the grid figure uses two
        axes. So none of them could catch a layer that silently replaced another, which is exactly what the
        tier did before #313: `field` then `contours` left the raster gone and the isolines on white. This
        is the case the visual suite could not see.

    Returns:
        Figure: the rendered map.
    """
    m = Map(crs=dataset.epsg, figsize=_FIGSIZE)
    m.field(dataset, cmap="viridis")
    m.contours(dataset, cmap="autumn", levels=6)
    return m.fig
