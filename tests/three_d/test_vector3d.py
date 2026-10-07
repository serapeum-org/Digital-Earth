"""Tests for the D3.4 vector renderers (digitalearth.three_d.VectorMixin).

Gated on the optional ``3d`` extra (pyvista). Covers arrow glyphs from a (u, v, w) field, polygon extrusion
into 3-D prisms (uniform + per-feature height + colour-by-attribute), and the MultiPolygon ring reader.
"""

import logging

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon

from digitalearth.base.crs import OffLimbError
from digitalearth.three_d import Scene3D
from digitalearth.three_d.vector import (
    MAGNITUDE,
    VALUE,
    _exterior_rings,
    _extrude_polygon,
    _extrude_ring,
    _polygon_parts,
)


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


def test_polygon_parts_reads_interior_rings():
    """_polygon_parts returns each part's exterior plus its hole rings (duck-typed, no shapely import)."""
    poly = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]]
    )
    parts = list(_polygon_parts(poly))
    assert len(parts) == 1
    exterior, interiors = parts[0]
    assert exterior.shape[1] == 2
    assert len(interiors) == 1 and interiors[0].shape[1] == 2


def test_extrude_polygon_carves_the_hole():
    """#199: a 4x4 square with a 2x2 hole extrudes to a prism with the hole carved, not filled.

    Cap area is 16 - 4 = 12, so a height-2 prism has volume 24; the old exterior-only path filled the hole and
    would give 32.
    """
    exterior = np.array([[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]], dtype=float)
    hole = np.array([[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]], dtype=float)
    prism = _extrude_polygon(exterior, [hole], 2.0)
    assert prism.volume == pytest.approx(24.0)


def test_extruded_polygons_public_path_carves_holes():
    """The public builder carves a footprint's holes through the merged prism mesh."""
    holed = gpd.GeoDataFrame(
        {"pop": [1.0]},
        geometry=[
            Polygon(
                [(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]]
            )
        ],
    )
    scene = Scene3D(off_screen=True)
    scene.extruded_polygons(holed, height=2.0)
    # The merged layer is a surface (2-D cells), so its enclosed volume is read off that surface — 24 with
    # the hole carved, 32 if it were filled.
    surface = scene.layers[0][0].extract_surface(algorithm="dataset_surface")
    assert surface.volume == pytest.approx(24.0)
    scene.close()


def _rivers():
    """Two line features with a flow value (a tiny lines fixture)."""
    return gpd.GeoDataFrame(
        {"flow": [3.0, 7.0]},
        geometry=[
            LineString([(0, 0), (1, 1), (2, 0)]),
            LineString([(0, 2), (2, 2)]),
        ],
    )


def test_lines_render_one_cell_per_part():
    """#201: lines() draws LineString features as polylines — one cell per part — and renders."""
    scene = Scene3D(off_screen=True)
    actor = scene.lines(_rivers(), width=4.0)
    assert actor is not None
    assert len(scene.layers) == 1
    assert scene.layers[0][0].n_cells == 2  # one polyline cell per feature
    assert bool(scene.screenshot().any())
    scene.close()


def test_lines_multilinestring_yields_a_cell_per_part():
    """A MultiLineString contributes one polyline cell per member part."""
    gdf = gpd.GeoDataFrame(
        {"flow": [1.0]},
        geometry=[MultiLineString([[(0, 0), (1, 0)], [(0, 1), (1, 1)]])],
    )
    scene = Scene3D(off_screen=True)
    scene.lines(gdf)
    assert scene.layers[0][0].n_cells == 2
    scene.close()


def test_lines_colour_by_column_classifies():
    """column= with a graduated scheme bins the lines into class codes on the mesh (like the 2-D tiers)."""
    scene = Scene3D(off_screen=True)
    scene.lines(_rivers(), column="flow", scheme="quantiles", k=2)
    mesh = scene.layers[0][0]
    assert VALUE in mesh.cell_data
    assert sorted({int(v) for v in mesh.cell_data[VALUE]}) == [0, 1]
    scene.close()


def test_lines_lift_a_2d_line_onto_z0():
    """A 2-D line is lifted onto z=0 so it has a valid 3-D coordinate for the scene."""
    scene = Scene3D(off_screen=True)
    scene.lines(_rivers())
    assert scene.layers[0][0].points[:, 2].tolist() == pytest.approx([0.0] * 5)
    scene.close()


def test_lines_empty_skips_and_warns(caplog):
    """An empty line collection is skipped with a warning, not a crash."""
    empty = gpd.GeoDataFrame({"flow": []}, geometry=[])
    scene = Scene3D(off_screen=True)
    with caplog.at_level(logging.WARNING):
        assert scene.lines(empty) is None
    assert not scene.layers
    assert "lines" in caplog.text
    scene.close()


def test_lines_reject_non_line_geometry():
    """lines() refuses a non-line geometry (e.g. a Polygon) with a clear TypeError."""
    gdf = gpd.GeoDataFrame(
        {"flow": [1.0]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1)])]
    )
    scene = Scene3D(off_screen=True)
    with pytest.raises(TypeError, match="LineString"):
        scene.lines(gdf)
    scene.close()


def test_lines_hidden_when_visible_false():
    """visible=False adds the layer described but drawn hidden, still addressable by id."""
    scene = Scene3D(off_screen=True)
    scene.lines(_rivers(), name="rivers", visible=False)
    assert "rivers" in scene.layer_ids
    assert scene.renderer.is_visible("rivers") is False
    scene.close()


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

    @pytest.mark.parametrize("keyword", ["clim", "n_colors"])
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
        values = {"clim": (0, 1), "n_colors": 3}[keyword]
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

    def test_the_missing_data_colour_is_the_callers_to_choose(self, squares):
        """One rule for the tier: `nan_color` is honoured here as it is on a coloured point cloud.

        Args:
            squares: Two polygons with a numeric column.

        Test scenario:
            It was refused beside `clim` and `n_colors`, which are the class-index range and the class count —
            consequences of the classification. `nan_color` is the colour missing data is drawn in, filled
            from the shared default, so a caller who names one is choosing rather than colliding. Refusing it
            on one builder while its neighbour honours it is the asymmetry (review R2-L2). Read off the
            actor's lookup table, which is what VTK paints with.
        """
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.extruded_polygons(
                squares, column="pop", scheme="quantiles", k=2, nan_color="red"
            )
            painted = tuple(actor.mapper.lookup_table.nan_color)
        finally:
            scene.close()
        assert painted == (1.0, 0.0, 0.0, 1.0), (
            f"the prisms paint missing data {painted} rather than the red that was asked for"
        )

    def test_the_default_missing_data_colour_is_the_shared_one(self, squares):
        """Built the other way round, so the test above is not passing on a coincidence.

        Args:
            squares: Two polygons with a numeric column.
        """
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.extruded_polygons(
                squares, column="pop", scheme="quantiles", k=2
            )
            painted = tuple(actor.mapper.lookup_table.nan_color)
        finally:
            scene.close()
        assert painted != (1.0, 0.0, 0.0, 1.0), (
            f"the default missing-data colour is {painted}, which is the red the other test asks for"
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


class TestTheExtrudedColumnIsClassifiedOnce:
    """The extrusion builder describes the classification and its drawer paints it — one computation.

    The point cloud's half of this claim is
    `tests/three_d/test_point_cloud3d.py::TestTheColumnIsClassifiedOnce`. These are the tier's only two
    classifying builders, and the reuse reached the cloud alone, leaving a claim documented as a property
    of the tier resting on one drawer — and leaving `replace_layer` behaving differently on the two
    (review R2-L11).
    """

    @staticmethod
    def _grid(n=8):
        """Build ``n`` unit squares in a row, with a skewed numeric column and a label column.

        Args:
            n: How many squares to build.

        Returns:
            geopandas.GeoDataFrame: the polygons, a ``pop`` column to classify and a ``band`` column to
            categorise.
        """
        return gpd.GeoDataFrame(
            {
                "pop": [float(i) ** 2 for i in range(n)],
                "band": ["low", "mid", "high"] * (n // 3) + ["low"] * (n % 3),
            },
            geometry=[
                Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)]) for i in range(n)
            ],
        )

    @staticmethod
    def _painted(actor):
        """Read the class colours back off the engine's own lookup table.

        Args:
            actor: The layer's drawn actor.

        Returns:
            list[str]: one ``#rrggbb`` per class, as VTK holds them.
        """
        from matplotlib.colors import to_hex

        rows = np.asarray(actor.mapper.lookup_table.values, dtype="float64") / 255.0
        return [to_hex(row[:3]) for row in rows]

    def test_a_graduated_column_reaches_the_classifier_once(self, monkeypatch):
        """Cutting an extrusion's classes twice is a second full pass for an identical result.

        Args:
            monkeypatch: Counts the calls that reach the shared classifier.

        Test scenario:
            `_color_encoding` cuts the column to describe the layer, and the drawer cut the same column
            again with the same scheme and k a moment later. The cloud stopped doing that in `dbe8ac07`;
            the extrusion went on doing it (review R2-L11).
        """
        from digitalearth.base.spec.scale import Scale

        scanned = []
        cut = Scale._breaks

        def counted(values, scheme, k):
            scanned.append(int(np.asarray(values).size))
            return cut(values, scheme, k)

        monkeypatch.setattr(Scale, "_breaks", staticmethod(counted))
        scene = Scene3D(off_screen=True)
        try:
            scene.extruded_polygons(
                self._grid(), height=1.0, column="pop", scheme="quantiles", k=4
            )
        finally:
            scene.close()
        assert scanned == [8], (
            f"the column must be cut once and the cut reused; it was scanned {scanned}"
        )

    def test_a_categorical_column_reaches_the_categoriser_once(self, monkeypatch):
        """The categorical half is the same shape: one scan for the distinct values, not two.

        Args:
            monkeypatch: Counts the calls that reach the shared categoriser.

        Test scenario:
            The second call is over the three categories the first one found — the key's length, not the
            data's — because the colours are always sampled from the drawing call's cmap (review R2-H3).
        """
        import digitalearth.base.symbology as symbology

        scanned = []
        categorise = symbology.categorical_colors

        def counted(values, **kwargs):
            scanned.append(int(np.asarray(values, dtype=object).size))
            return categorise(values, **kwargs)

        monkeypatch.setattr(symbology, "categorical_colors", counted)
        scene = Scene3D(off_screen=True)
        try:
            scene.extruded_polygons(
                self._grid(9),
                height=1.0,
                column="band",
                scheme="categorical",
                cmap="tab10",
            )
        finally:
            scene.close()
        assert scanned == [9, 3], (
            f"the column must be categorised once and the categories reused; it was scanned {scanned}"
        )

    @pytest.mark.parametrize(
        "changed, column, scheme, k, cmap",
        [
            ({"k": 6}, "pop", "quantiles", 3, "viridis"),
            ({"cmap": "Set1"}, "band", "categorical", 5, "tab10"),
        ],
    )
    def test_a_fresh_request_overrules_the_cut_the_description_carries(
        self, changed, column, scheme, k, cmap
    ):
        """Reusing the cut must not cost the extrusion what it costs the cloud nothing to keep.

        Args:
            changed: The symbology property `replace_layer` asks for, replacing the one it was built with.
            column: The attribute column the prisms are coloured by.
            scheme: The scheme the layer is first built with.
            k: The class count it is first built with.
            cmap: The colormap it is first built with.

        Test scenario:
            The two sides are built from **different** requests: the layer's description carries the cut its
            builder made, and `replace_layer` installs a symbology asking for another `k` or another `cmap`
            without republishing the encoding. Handing that stale cut over unconditionally is what made the
            cloud unable to be restyled (review R2-H3); passing a scale here must not import that.
        """
        from dataclasses import replace as with_fields

        from digitalearth.three_d.base import classified_scalars

        gdf = self._grid(9)
        asked = {"scheme": scheme, "k": k, "cmap": cmap, **changed}
        wanted = classified_scalars(gdf[column].to_numpy(), **asked)
        scene = Scene3D(off_screen=True)
        try:
            scene.extruded_polygons(
                gdf, height=1.0, column=column, scheme=scheme, k=k, cmap=cmap
            )
            held = scene.get_layer("extrusion-1")
            props = {**held.symbology.props, **changed}
            scene.replace_layer(
                with_fields(held, symbology=with_fields(held.symbology, props=props))
            )
            drawn = sorted(set(float(v) for v in scene.layers[0][0].cell_data[VALUE]))
            painted = self._painted(scene.layers[0][1])
        finally:
            scene.close()
        assert drawn == sorted(set(float(v) for v in wanted["scalars"])), (
            f"replace_layer({changed}) painted classes {drawn}, not the "
            f"{sorted(set(float(v) for v in wanted['scalars']))} it asked for"
        )
        assert painted == list(wanted["cmap"]), (
            f"replace_layer({changed}) painted {painted}, not the {list(wanted['cmap'])} it asked for"
        )
