"""Tests for the D3.4 vector renderers (digitalearth.three_d.VectorMixin).

Gated on the optional ``3d`` extra (pyvista). Covers arrow glyphs from a (u, v, w) field, polygon extrusion
into 3-D prisms (uniform + per-feature height + colour-by-attribute), and the MultiPolygon ring reader.
"""

import logging

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import MultiPolygon, Polygon

from digitalearth.base.crs import OffLimbError
from digitalearth.three_d import Scene3D
from digitalearth.three_d.vector import MAGNITUDE, VALUE, _exterior_rings, _extrude_ring


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def _squares():
    """Two unit-square polygons with population values (a tiny extrusion fixture)."""
    return gpd.GeoDataFrame(
        {"pop": [10.0, 20.0]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
    )


def test_vectors_glyphs_render():
    """vectors() builds magnitude-coloured arrow glyphs and registers one layer."""
    ax = np.linspace(0, 1, 6)
    xx, yy = np.meshgrid(ax, ax)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    vec = np.column_stack(
        [np.ones(pts.shape[0]), np.zeros(pts.shape[0]), np.zeros(pts.shape[0])]
    )
    scene = Scene3D(off_screen=True)
    scene.vectors(pts, vec, factor=0.1)
    assert len(scene.layers) == 1
    assert MAGNITUDE in scene.layers[0][0].point_data
    assert bool(scene.screenshot().any())
    scene.close()


def test_extruded_polygons_uniform_height():
    """A uniform height extrudes every footprint to the same z; the merged mesh has cells."""
    scene = Scene3D(off_screen=True)
    scene.extruded_polygons(_squares(), height=2.0)
    mesh = scene.layers[0][0]
    assert mesh.n_cells > 0
    assert mesh.bounds[5] == pytest.approx(2.0)
    scene.close()


def test_extruded_polygons_height_from_column_and_colour():
    """height='pop' extrudes per-feature; column='pop' attaches the colour scalar."""
    scene = Scene3D(off_screen=True)
    scene.extruded_polygons(_squares(), height="pop", column="pop")
    mesh = scene.layers[0][0]
    assert mesh.bounds[5] == pytest.approx(20.0)  # tallest = max pop
    assert VALUE in mesh.cell_data
    scene.close()


def test_extrude_ring_height():
    """_extrude_ring lifts a flat ring into a prism of the requested height."""
    ring = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    prism = _extrude_ring(ring, 0.5)
    assert prism.n_cells > 0
    assert prism.bounds[5] == pytest.approx(0.5)


def test_multipolygon_yields_one_ring_per_part():
    """_exterior_rings walks every part of a MultiPolygon."""
    mp = MultiPolygon(
        [Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(5, 5), (6, 5), (6, 6)])]
    )
    rings = list(_exterior_rings(mp))
    assert len(rings) == 2
    assert all(r.shape[1] == 2 for r in rings)


def test_vectors_length_mismatch_raises():
    """vectors() rejects mismatched points/vectors shapes."""
    scene = Scene3D(off_screen=True)
    zeros = np.zeros((5, 3))
    zeros2 = np.zeros((4, 3))
    with pytest.raises(ValueError, match="same shape"):
        scene.vectors(zeros, zeros2)
    scene.close()


def test_extruded_polygons_empty_skips_and_warns(caplog):
    """C7: an empty geometry set is skipped with a warning, not raised, on a default scene.

    Args:
        caplog: Captures the warning the skipped layer logs.

    Test scenario:
        A 3-D scene is usually composed of several layers, so one of them having nothing to extrude must not
        take the others with it. The layer is dropped, nothing is registered, and a WARNING names both the
        layer and the reason so the absence is not silent.
    """
    empty = gpd.GeoDataFrame({"pop": []}, geometry=[])
    scene = Scene3D(off_screen=True)
    with caplog.at_level(logging.WARNING):
        assert scene.extruded_polygons(empty) is None, (
            "a skipped layer returns None rather than an actor"
        )
    scene.close()
    assert not scene.layers, "a skipped layer must not be registered"
    assert "extruded_polygons" in caplog.text, (
        f"the warning must name the layer, got {caplog.text!r}"
    )
    assert "no polygon" in caplog.text, (
        f"the warning must name the reason, got {caplog.text!r}"
    )


def test_extruded_polygons_empty_raises_under_strict():
    """C7: `strict=True` turns the skipped-layer warning back into an OffLimbError.

    Test scenario:
        A pipeline that would rather fail than ship a map with a missing layer opts in with `strict=True`,
        and gets the shared `OffLimbError` the 2-D tiers raise for "none of the data could be placed".
    """
    empty = gpd.GeoDataFrame({"pop": []}, geometry=[])
    scene = Scene3D(off_screen=True, strict=True)
    try:
        with pytest.raises(OffLimbError, match="no polygon"):
            scene.extruded_polygons(empty)
    finally:
        scene.close()


def test_extruded_polygons_rejects_non_polygon():
    """extruded_polygons() raises a clear TypeError on non-polygon geometry (e.g. a Point)."""
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({"pop": [1.0]}, geometry=[Point(0, 0)])
    scene = Scene3D(off_screen=True)
    with pytest.raises(TypeError, match="Polygon"):
        scene.extruded_polygons(gdf)
    scene.close()


class TestClassifiedExtrusionAnswersForItsOwnKeywords:
    """Round-2 review L1/L2/M5 — a classified extrusion must not fail through Python's call machinery."""

    @pytest.fixture
    def squares(self):
        """Two unit squares carrying a numeric column, the minimum a classified extrusion needs.

        Returns:
            A GeoDataFrame with a ``pop`` column and two polygons.
        """
        return gpd.GeoDataFrame(
            {"pop": [10.0, 20.0]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
        )

    @pytest.mark.parametrize("keyword", ["clim", "n_colors", "nan_color"])
    def test_a_colour_keyword_the_scheme_owns_is_named(self, squares, keyword):
        """L1 — ``scheme=`` plus ``clim=`` was a "got multiple values" ``TypeError`` about internals.

        Args:
            squares: Two polygons with a numeric column.
            keyword: A colour setting the classification derives.

        Test scenario:
            The style the classifier builds and the caller's ``**kwargs`` were splatted into one
            ``add_mesh`` call, so a caller pinning the colour range on a classified layer got a Python
            error naming no cause. The classification owns those keywords — the scalars are class indices —
            so the answer is a refusal that says which keyword and why.
        """
        values = {"clim": (0, 1), "n_colors": 3, "nan_color": "#000000"}[keyword]
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(TypeError) as excinfo:
                scene.extruded_polygons(
                    squares, column="pop", scheme="quantiles", k=2, **{keyword: values}
                )
        finally:
            scene.close()
        message = str(excinfo.value)
        assert f"{keyword}=" in message, (
            f"the refusal must name the keyword {keyword}=, got {message}"
        )
        assert "scheme='quantiles'" in message, (
            f"the refusal must name the scheme that owns it, got {message}"
        )

    def test_the_same_keyword_is_fine_without_a_scheme(self, squares):
        """An unclassified layer has no derived colour range, so the caller's ``clim`` is honoured.

        Args:
            squares: Two polygons with a numeric column.
        """
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.extruded_polygons(squares, column="pop", clim=(0.0, 30.0))
            assert actor is not None
        finally:
            scene.close()

    def test_a_nan_height_is_refused_by_row(self, squares):
        """L2 — a ``NaN`` height silently built prisms whose coordinates were all ``NaN``.

        Args:
            squares: Two polygons with a numeric column.

        Test scenario:
            The actor still reported cells, so nothing looked wrong: the buildings were simply not there.
            The colour column's ``NaN`` is handled deliberately by ``classified_scalars``; the height
            column's was not.
        """
        gdf = squares.copy()
        gdf["pop"] = [10.0, float("nan")]
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(ValueError) as excinfo:
                scene.extruded_polygons(gdf, height="pop")
        finally:
            scene.close()
        message = str(excinfo.value)
        assert "row 1" in message, (
            f"the refusal must name the offending row, got {message}"
        )
        assert "'pop'" in message, (
            f"the refusal must name the height column, got {message}"
        )

    def test_a_nan_uniform_height_is_refused_too(self, squares):
        """A scalar height gets the same check — it reaches the same extrusion.

        Args:
            squares: Two polygons with a numeric column.
        """
        scene = Scene3D(off_screen=True)
        try:
            not_a_number = float("nan")
            with pytest.raises(ValueError, match="finite extrusion height"):
                scene.extruded_polygons(squares, height=not_a_number)
        finally:
            scene.close()

    def test_a_finite_height_column_still_extrudes(self, squares):
        """The guard must not disturb the per-feature heights it protects.

        Args:
            squares: Two polygons with a numeric column.
        """
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.extruded_polygons(squares, height="pop")
            assert actor is not None, "a finite height column must return an actor"
            assert scene.layers[0][0].n_cells > 0, (
                f"the extrusion must build cells, got {scene.layers[0][0].n_cells}"
            )
            assert np.isfinite(scene.layers[0][0].points).all(), "no NaN coordinates"
        finally:
            scene.close()

    @pytest.mark.parametrize("colours", [["#ff0000", "#00ff00"], ["#ff0000"] * 7])
    def test_a_colour_list_that_does_not_match_the_class_count_is_refused(
        self, colours
    ):
        """M5 — a short colour list collapsed classes onto the last colour with no warning.

        Args:
            colours: An explicit colour sequence that does not carry one colour per class.

        Test scenario:
            ``sample_cmap`` takes a sequence as given, and ``_discrete_style`` then derived ``clim`` and
            ``n_colors`` from *its* length instead of from the class count — so classes 2, 3 and 4 of five
            all clamped onto the second colour and rendered identically.
        """
        from digitalearth.three_d.base import classified_scalars

        with pytest.raises(ValueError) as excinfo:
            classified_scalars([1, 2, 3, 4, 50], scheme="quantiles", k=5, cmap=colours)
        message = str(excinfo.value)
        assert "one colour per class" in message, message
        assert f"{len(colours)} colours for 5 classes" in message, message

    def test_a_colour_list_of_exactly_k_is_honoured(self):
        """The guard must not disturb the deliberate case it protects."""
        from digitalearth.three_d.base import classified_scalars

        colours = ["#ff0000", "#00ff00", "#0000ff"]
        style = classified_scalars(
            [1, 2, 3, 4, 50], scheme="quantiles", k=3, cmap=colours
        )
        assert style["cmap"] == colours, style["cmap"]
        assert style["n_colors"] == 3, (
            f"one colour slot per class, got {style['n_colors']}"
        )
        assert style["clim"] == (-0.5, 2.5), f"class-index range, got {style['clim']}"
