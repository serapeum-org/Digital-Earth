"""Tests for categorical ``scheme`` on the value-colored Map methods (cleopatra classify, #154).

``choropleth`` / ``voronoi`` / ``cartogram`` / ``quadtree`` forward ``scheme``/``k`` to ``PolygonGlyph``; a set
``scheme`` must bin the values into discrete classes — i.e. the mappable carries a ``BoundaryNorm`` — while the
default (no ``scheme``) stays a continuous norm.
"""

import pandas as pd
import pytest
from matplotlib.colors import BoundaryNorm, to_hex

from digitalearth._symbology import MISSING_COLOR, categorical_colors, resolve_categorical_cmap
from digitalearth.scene import Map


@pytest.fixture
def points_fc():
    """Point fixture (EPSG:32618, numeric 'fid')."""
    from pyramids.feature import FeatureCollection

    return FeatureCollection.read_file("tests/data/points.geojson")


@pytest.fixture
def polygons(points_fc):
    """Polygon fixture: buffered points with a numeric 'fid' column."""
    fc = points_fc.copy()
    fc["geometry"] = fc.geometry.buffer(500.0)
    return fc


@pytest.fixture
def zoned(polygons):
    """Polygon fixture carrying a nominal 'zone' column of three unordered classes."""
    fc = polygons.copy()
    labels = ["urban", "rural", "park"]
    fc["zone"] = [labels[i % len(labels)] for i in range(len(fc))]
    return fc


def rendered_colors(mappable, count):
    """Return the first ``count`` colours of a categorical mappable's colormap, as lower-case hex."""
    cmap = mappable.get_cmap()
    return [to_hex(cmap(i)) for i in range(count)]


@pytest.mark.parametrize(
    "cmap",
    [None, "viridis", "Set2"],
    ids=["default", "continuous-default", "qualitative"],
)
def test_categorical_cmap_agrees_with_the_sibling_tiers(zoned, cmap):
    """Static must resolve ``cmap`` through the same sentinel as web/interactive, else one cmap means two maps.

    cleopatra swaps its *own* continuous default (``coolwarm_r``) for a qualitative map, while the sibling tiers
    swap ``viridis`` — so without a shared sentinel ``cmap="viridis"`` renders viridis here and tab10 there (M1).
    """
    opts = {} if cmap is None else {"cmap": cmap}
    pc = Map(crs=zoned.epsg).choropleth(zoned, column="zone", scheme="categorical", **opts)
    _, expected = categorical_colors(zoned["zone"], resolve_categorical_cmap(cmap))
    assert rendered_colors(pc, len(expected)) == [c.lower() for c in expected]


def test_categorical_missing_values_are_drawn_neutral(polygons):
    """A missing category is drawn neutral, not invisible — matching the web/interactive tiers (M4).

    cleopatra codes a missing category as ``NaN``, whose default "bad" colour is fully transparent, so the
    feature would render as nothing and be indistinguishable from one that was never in the collection.
    """
    fc = polygons.copy()
    zones = [["urban", "rural"][i % 2] for i in range(len(fc))]
    zones[0] = None
    fc["zone"] = zones
    m = Map(crs=fc.epsg)
    pc = m.choropleth(fc, column="zone", scheme="categorical")
    pc.update_scalarmappable()  # resolve the value array into facecolors
    facecolors = pc.get_facecolors()
    assert to_hex(facecolors[0]) == MISSING_COLOR, "the null-attribute polygon must be drawn neutral"
    assert facecolors[0][3] > 0, "and must be visible at all (the default bad colour is fully transparent)"
    assert [t.get_text() for t in m.layers[-1][0].category_legend.get_texts()] == ["rural", "urban"]


@pytest.mark.parametrize("dtype", ["object", "string"], ids=["object-None", "nullable-pdNA"])
def test_categorical_nulls_never_become_a_category(polygons, dtype):
    """The same logical data must render the same whether its nulls are ``None`` or ``pd.NA`` (M5).

    A pandas nullable dtype spells its nulls ``pd.NA``, which survives cleopatra's ``is None``/float-NaN test
    and becomes a real ``<NA>`` class — taking a palette colour and shifting every other category's.
    """
    fc = polygons.copy()
    zones = [["urban", "rural"][i % 2] for i in range(len(fc))]
    zones[0] = None
    fc["zone"] = pd.array(zones, dtype=dtype)
    m = Map(crs=fc.epsg)
    pc = m.choropleth(fc, column="zone", scheme="categorical")
    labels = [t.get_text() for t in m.layers[-1][0].category_legend.get_texts()]
    assert labels == ["rural", "urban"], "a null must not become a category, whatever its dtype spelling"
    pc.update_scalarmappable()
    assert to_hex(pc.get_facecolors()[0]) == MISSING_COLOR, "the null row is missing data, not a class"


@pytest.mark.parametrize("spelling", ["categorical", "Categorical", "CATEGORICAL"])
def test_categorical_scheme_spelling_is_case_insensitive(zoned, spelling):
    """A case-variant ``scheme`` renders the same map, as it already does on the web tier (M3).

    cleopatra dispatches on an exact ``== "categorical"``, so an unnormalized case variant used to set up a
    categorical render here and then fall through to the continuous path there, dying in ``np.isfinite`` on a
    string column.
    """
    m = Map(crs=zoned.epsg)
    pc = m.choropleth(zoned, column="zone", scheme=spelling)
    assert isinstance(pc.norm, BoundaryNorm)
    assert [t.get_text() for t in m.layers[-1][0].category_legend.get_texts()] == ["park", "rural", "urban"]


def test_categorical_cmap_coolwarm_r_is_a_known_upstream_divergence(zoned):
    """``cmap="coolwarm_r"`` is cleopatra's own sentinel, so the static tier swaps it where web honours it.

    Pins the one divergence that cannot be resolved from this side (it needs an upstream change), so a future
    cleopatra release that changes the rule fails here loudly instead of drifting silently.
    """
    pc = Map(crs=zoned.epsg).choropleth(zoned, column="zone", scheme="categorical", cmap="coolwarm_r")
    _, web_colors = categorical_colors(zoned["zone"], resolve_categorical_cmap("coolwarm_r"))
    _, tab10 = categorical_colors(zoned["zone"], "tab10")
    assert rendered_colors(pc, 3) == [c.lower() for c in tab10], "static swaps cleopatra's own default sentinel"
    assert [c.lower() for c in web_colors] != [c.lower() for c in tab10], "web honours it — the divergence"


def test_choropleth_scheme_is_discrete(polygons):
    """A scheme bins choropleth fills into discrete classes (BoundaryNorm); default is continuous."""
    discrete = Map(crs=polygons.epsg).choropleth(polygons, column="fid", scheme="quantiles", k=3)
    continuous = Map(crs=polygons.epsg).choropleth(polygons, column="fid")
    assert isinstance(discrete.norm, BoundaryNorm)
    assert not isinstance(continuous.norm, BoundaryNorm)


def test_choropleth_categorical_colors_each_distinct_value(polygons):
    """scheme='categorical' gives every distinct value its own class code (cleopatra >=0.26, CAT-5)."""
    fc = polygons.copy()
    fc["zone"] = ["urban", "rural", "park"] * (len(fc) // 3) + ["urban"] * (len(fc) % 3)
    pc = Map(crs=fc.epsg).choropleth(fc, column="zone", scheme="categorical")
    assert isinstance(pc.norm, BoundaryNorm)
    # the mappable carries integer class codes (one per distinct label), not the labels themselves
    assert sorted(set(pc.get_array().tolist())) == [0.0, 1.0, 2.0]


def test_choropleth_categorical_draws_a_swatch_legend(polygons):
    """A categorical fill is keyed by a labelled swatch legend, not a colorbar (the codes would be opaque)."""
    fc = polygons.copy()
    fc["zone"] = ["urban", "rural"] * (len(fc) // 2) + ["urban"] * (len(fc) % 2)
    m = Map(crs=fc.epsg)
    m.choropleth(fc, column="zone", scheme="categorical")
    glyph = m.layers[-1][0]
    assert glyph.cbar is None
    assert [t.get_text() for t in glyph.category_legend.get_texts()] == ["rural", "urban"]


def test_choropleth_categorical_legend_can_be_suppressed(polygons):
    """An explicit add_colorbar=False keeps the glyph from drawing its own key (the Scene takes over)."""
    fc = polygons.copy()
    fc["zone"] = ["urban", "rural"] * (len(fc) // 2) + ["urban"] * (len(fc) % 2)
    m = Map(crs=fc.epsg)
    m.choropleth(fc, column="zone", scheme="categorical", add_colorbar=False)
    assert m.layers[-1][0].category_legend is None


def test_graduated_choropleth_still_defers_its_key_to_the_scene(polygons):
    """The categorical legend default must not leak into the graduated/continuous path (Scene owns that).

    The load-bearing assertion is on ``add_colorbar``: cleopatra resets ``category_legend`` to None on every
    ``plot()`` and only sets it on the categorical branch, so asserting it is None on a graduated fill can never
    fail — it would pass even if ``_polygon_layer`` wrongly defaulted ``add_colorbar=True`` for every scheme,
    which is exactly the leak this test guards. Assert the default the glyph actually received instead.
    """
    m = Map(crs=polygons.epsg)
    m.choropleth(polygons, column="fid", scheme="quantiles", k=3)
    glyph = m.layers[-1][0]
    assert glyph.default_options["add_colorbar"] is False, "the Scene, not the glyph, owns the graduated key"
    assert glyph.cbar is None


def test_voronoi_scheme_is_discrete(points_fc):
    """voronoi honours a categorical scheme on its filled cells."""
    pc = Map(crs=points_fc.epsg).voronoi(points_fc, column="fid", scheme="quantiles", k=3)
    assert isinstance(pc.norm, BoundaryNorm)


def test_cartogram_scheme_is_discrete(polygons):
    """cartogram honours a categorical scheme on its scaled polygons."""
    pc = Map(crs=polygons.epsg).cartogram(polygons, scale="fid", column="fid", scheme="quantiles", k=3)
    assert isinstance(pc.norm, BoundaryNorm)


def test_quadtree_scheme_is_discrete(points_fc):
    """quadtree honours a categorical scheme on its aggregate cells."""
    pc = Map(crs=points_fc.epsg).quadtree(points_fc, column="fid", nmax=1, scheme="quantiles", k=3)
    assert isinstance(pc.norm, BoundaryNorm)


def test_scheme_fisher_jenks(polygons):
    """The native Fisher-Jenks scheme is accepted (no mapclassify dependency)."""
    pc = Map(crs=polygons.epsg).choropleth(polygons, column="fid", scheme="fisher_jenks", k=3)
    assert isinstance(pc.norm, BoundaryNorm)


def test_categorical_scheme_on_outline_only_is_inert(polygons):
    """An outline-only draw with scheme='categorical' (no values) skips the categorical mapping harmlessly.

    ``shapes`` reaches ``_polygon_layer`` with ``values=None``, so the categorical value-mapping branch must be
    a no-op — the outlines still draw and nothing crashes on the absent values.
    """
    m = Map(crs=polygons.epsg)
    pc = m.shapes(polygons, scheme="categorical")
    assert len(pc.get_paths()) == len(polygons), "outlines must still be drawn"
    assert m.layers[-1][0].category_legend is None, "no values means no category legend"


@pytest.mark.parametrize(
    "fixture,call",
    [
        ("points_fc", lambda m, fc: m.voronoi(fc, column="fid", scheme="categorical")),
        ("polygons", lambda m, fc: m.cartogram(fc, scale="fid", column="fid", scheme="categorical")),
        ("points_fc", lambda m, fc: m.quadtree(fc, column="fid", nmax=1, scheme="categorical")),
    ],
    ids=["voronoi", "cartogram", "quadtree"],
)
def test_sibling_polygon_methods_honour_categorical(request, fixture, call):
    """voronoi/cartogram/quadtree inherit the categorical path through the shared helper (L4).

    They accept ``scheme`` and pass values, so the ``add_colorbar`` categorical default reaches them too — the
    swatch legend must be drawn (and no colorbar), exactly as for :meth:`choropleth`.
    """
    fc = request.getfixturevalue(fixture)
    m = Map(crs=fc.epsg)
    pc = call(m, fc)
    glyph = m.layers[-1][0]
    assert isinstance(pc.norm, BoundaryNorm)
    assert glyph.cbar is None
    assert glyph.category_legend is not None, "a categorical sibling fill must draw its own swatch legend"
