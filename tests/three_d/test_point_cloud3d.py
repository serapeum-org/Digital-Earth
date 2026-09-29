"""Tests for the D3.2 point-cloud renderer (digitalearth.three_d.PointCloudMixin.point_cloud).

Gated on the optional ``3d`` extra (pyvista). Exercises the numpy xyz path (LiDAR-style), the 2-D lift-to-z=0
path, per-point colouring, the GeoDataFrame (``get_cell_points``-style) path read by duck-typing, and that the
guard module imports no GIS competitor.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D
from digitalearth.three_d.point_cloud import SCALAR, _coords_from_array


@pytest.fixture(autouse=True)
def _force_off_screen():
    """Render headless for every test."""
    prev = pv.OFF_SCREEN
    pv.OFF_SCREEN = True
    yield
    pv.OFF_SCREEN = prev


def test_point_cloud_from_xyz_array():
    """A coloured (N, 3) xyz table renders and registers one layer with the points and scalar."""
    pts = np.column_stack([np.arange(40.0), np.arange(40.0), np.linspace(0.0, 9.0, 40)])
    scene = Scene3D(off_screen=True)
    actor = scene.point_cloud(pts, values=pts[:, 2])
    assert actor is not None
    assert len(scene.layers) == 1
    cloud = scene.layers[0][0]
    assert cloud.n_points == 40
    assert SCALAR in cloud.point_data
    scene.close()


def test_two_column_table_is_lifted_to_z0():
    """A (N, 2) table is lifted to z=0 (so 2-D points still build a valid cloud)."""
    out = _coords_from_array(np.random.default_rng(0).random((25, 2)))
    assert out.shape == (25, 3)
    assert np.all(out[:, 2] == 0.0)


def test_bad_shape_raises():
    """A non-(N, 2)/(N, 3) array is rejected with a clear error."""
    zeros = np.zeros((5, 4))
    with pytest.raises(ValueError):
        _coords_from_array(zeros)


def test_uncoloured_cloud_has_no_scalar():
    """Without values/value_column the cloud carries no colour scalar."""
    scene = Scene3D(off_screen=True)
    scene.point_cloud(np.random.default_rng(1).random((30, 3)), eye_dome_lighting=False)
    assert SCALAR not in scene.layers[0][0].point_data
    scene.close()


def test_point_cloud_from_geodataframe(points):
    """A GeoDataFrame of points (e.g. get_cell_points) renders via duck-typed coordinate reads."""
    scene = Scene3D(off_screen=True)
    actor = scene.point_cloud(points, size=8.0)
    assert actor is not None
    assert scene.layers[0][0].n_points == len(points)
    scene.close()


def test_a_null_geometry_reads_as_nan_rather_than_raising():
    """A frame with one `null` geometry still builds a cloud, as it did before the PointArrays migration.

    `main` read `.x` straight off the series, and geopandas answers NaN there for a missing geometry.
    Routing the read through `PointArrays.from_features(..., centroids=False)` classified such a frame as
    "not all points" — a null's `geom_type` is NaN — and raised instead, turning a cloud that rendered into
    an exception on an ordinary GeoJSON feature.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    from digitalearth.three_d.point_cloud import _coords_from_geodataframe

    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), None, Point(2, 2)], crs="EPSG:3857")
    coords, values = _coords_from_geodataframe(gdf, None)
    assert coords.shape == (3, 3), f"every row survives the read, got {coords.shape}"
    assert np.isnan(coords[1, 0]), (
        "the missing row reads as NaN, the way geopandas answers for it"
    )
    assert values is None, "and no value column was asked for"


def test_renders_a_nonempty_frame():
    """The point cloud produces a real off-screen frame with content."""
    pts = np.random.default_rng(2).random((300, 3)) * 10
    scene = Scene3D(off_screen=True)
    scene.point_cloud(pts, values=pts[:, 2])
    img = scene.screenshot()
    assert img.ndim == 3 and bool(img.any())
    scene.close()


def test_values_length_mismatch_raises():
    """point_cloud() rejects a values array whose length does not match the points."""
    scene = Scene3D(off_screen=True)
    zeros = np.zeros((10, 3))
    zeros2 = np.zeros(7)
    with pytest.raises(ValueError, match="does not match"):
        scene.point_cloud(zeros, values=zeros2)
    scene.close()


class TestKeywordsTheDrawerWouldOverwrite:
    """H4 — `**kwargs` a caller passes must not be silently overwritten by the drawer.

    PyVista refuses a keyword it does not know (`_common_arg_parser` raises `TypeError`), so an arbitrary
    typo is already caught. What is not caught is a keyword PyVista *does* take and this drawer writes
    itself: `point_size`, the deleted spelling of `size` that the size fold overwrites, and the colour
    keywords the classification derives. Those drew a picture the caller did not ask for, with no
    diagnostic.
    """

    @staticmethod
    def _table() -> np.ndarray:
        """A small xyz point table.

        Returns:
            An ``(N, 3)`` float array of ascending points.
        """
        return np.column_stack([np.arange(12.0), np.arange(12.0), np.zeros(12)])

    def test_the_deleted_point_size_spelling_is_refused(self):
        """`point_size=` is the deleted spelling of `size=` and must raise, not draw at the default.

        Test scenario:
            The builder's `**kwargs` swallowed `point_size` into the layer's props, and the drawer then
            overwrote it from `size` — so `point_size=40` rendered at 5.0. Every other tier refuses the
            spelling because the parameter is gone from the signature; this one has to refuse it by name.
        """
        pts = self._table()
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(TypeError) as excinfo:
                scene.point_cloud(pts, point_size=40)
        finally:
            scene.close()
        message = str(excinfo.value)
        assert "point_size=" in message, (
            f"the refusal must name the keyword it refuses, got {message}"
        )
        assert "size=" in message, (
            f"the refusal must name the live spelling to use instead, got {message}"
        )

    def test_the_live_spelling_still_reaches_the_actor(self):
        """`size=` is the live spelling and the actor must draw at exactly what it was given."""
        asked = 40.0
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.point_cloud(self._table(), size=asked)
            drawn = float(actor.prop.point_size)
        finally:
            scene.close()
        assert drawn == asked, f"the cloud must draw at size={asked}, got {drawn}"

    @pytest.mark.parametrize("keyword", ["clim", "n_colors", "scalars"])
    def test_a_colour_keyword_the_classification_owns_is_named(self, keyword):
        """A pinned colour keyword on a coloured cloud is refused by name, not silently replaced.

        Args:
            keyword: A colour setting `classified_scalars` derives, or the scalar binding itself.

        Test scenario:
            `props.update(scalars=SCALAR, **style)` let the derived style win over whatever the caller
            pinned, so `clim=(0, 100)` on a classified cloud drew the class-index range instead and said
            nothing. The extruded-polygon drawer already refuses the same clash by name; this one did not.
        """
        values = {
            "clim": (0.0, 100.0),
            "n_colors": 3,
            "scalars": "elevation",
        }[keyword]
        pts = self._table()
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(TypeError) as excinfo:
                scene.point_cloud(
                    pts, values=pts[:, 0], scheme="quantiles", k=4, **{keyword: values}
                )
        finally:
            scene.close()
        message = str(excinfo.value)
        assert f"{keyword}=" in message, (
            f"the refusal must name the keyword {keyword}=, got {message}"
        )

    def test_the_missing_data_colour_is_the_callers_to_choose(self):
        """`nan_color` is a choice, not a consequence, so a coloured cloud takes the caller's own.

        Test scenario:
            It was refused with `clim` and `n_colors`, which is the wrong company: those are the class-index
            range and the class count, and overriding either re-colours the wrong classes, while `nan_color`
            is filled from the shared missing-data colour as a *default*. `point_cloud` has no `nan_color`
            parameter, so refusing it meant "give up the colouring" for anyone who wanted a different one
            (review R2-L2). Read off the actor's lookup table, which is what VTK actually paints with.
        """
        pts = self._table()
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.point_cloud(
                pts, values=pts[:, 0], scheme="quantiles", k=3, nan_color="red"
            )
            painted = tuple(actor.mapper.lookup_table.nan_color)
        finally:
            scene.close()
        assert painted == (1.0, 0.0, 0.0, 1.0), (
            f"the cloud paints missing data {painted} rather than the red that was asked for"
        )

    def test_the_default_missing_data_colour_is_the_shared_one(self):
        """Built the other way round: without a `nan_color` the shared neutral grey is what is painted.

        Test scenario:
            Without this the test above would pass for a cloud that happened to be red anyway, and the
            default is the thing the classifier is responsible for.
        """
        pts = self._table()
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.point_cloud(pts, values=pts[:, 0], scheme="quantiles", k=3)
            painted = tuple(actor.mapper.lookup_table.nan_color)
        finally:
            scene.close()
        assert painted != (1.0, 0.0, 0.0, 1.0), (
            f"the default missing-data colour is {painted}, which is the red the other test asks for"
        )

    def test_the_refusal_does_not_call_a_continuous_ramp_a_scheme(self):
        """On the continuous path there is no classification, so "scheme=None" named nothing.

        Test scenario:
            The message read "together with the values it colours by (scheme=None)", which reads as a scheme
            actually called None; the values reach the engine unchanged there and the ramp owns the range.
        """
        pts = self._table()
        scene = Scene3D(off_screen=True)
        try:
            with pytest.raises(TypeError) as excinfo:
                scene.point_cloud(pts, values=pts[:, 0], scalars="elevation")
        finally:
            scene.close()
        assert "scheme=None" not in str(excinfo.value), excinfo.value

    def test_the_same_keyword_is_fine_on_an_uncoloured_cloud(self):
        """An uncoloured cloud derives no colour style, so a pinned `clim` is the caller's to set."""
        scene = Scene3D(off_screen=True)
        try:
            actor = scene.point_cloud(self._table(), clim=(0.0, 30.0))
            assert actor is not None, (
                "an uncoloured cloud with a pinned clim still draws"
            )
        finally:
            scene.close()


class TestTheColumnIsClassifiedOnce:
    """The builder describes the classification and the drawer paints it — from one computation, not two.

    Every case here goes through `point_cloud`, which is one of the tier's **two** classifying builders.
    The extrusion's half of the same claim is
    `tests/three_d/test_vector3d.py::TestTheExtrudedColumnIsClassifiedOnce` — the reuse reached the cloud
    first and the class was named for the tier while resting on one drawer (review R2-L11).
    """

    @staticmethod
    def _table(n=200):
        """Build an ``(n, 3)`` point table whose x column doubles as a value column.

        Args:
            n: How many points to build.

        Returns:
            numpy.ndarray: the point table.
        """
        return np.column_stack([np.arange(float(n)), np.arange(float(n)), np.zeros(n)])

    def test_a_graduated_column_reaches_the_classifier_once(self, monkeypatch):
        """Cutting a large cloud's classes twice is a second full pass for an identical result.

        Args:
            monkeypatch: Counts the calls that reach the shared classifier.

        Test scenario:
            `_color_encoding` cuts the column to describe the layer, and the drawer cut the same column
            again with the same scheme and k a moment later — so a 50,000-point cloud scanned its values
            twice to draw once, whether or not a key was ever asked for (review N4).
        """
        from digitalearth.base.spec.scale import Scale

        scanned = []
        cut = Scale._breaks

        def counted(values, scheme, k):
            scanned.append(int(np.asarray(values).size))
            return cut(values, scheme, k)

        monkeypatch.setattr(Scale, "_breaks", staticmethod(counted))
        points = self._table()
        scene = Scene3D(off_screen=True)
        try:
            scene.point_cloud(points, values=points[:, 0], scheme="quantiles", k=4)
        finally:
            scene.close()
        assert scanned == [200], (
            f"the column must be cut once and the cut reused; it was scanned {scanned}"
        )

    def test_a_categorical_column_reaches_the_categoriser_once(self, monkeypatch):
        """The categorical half is the same shape: one scan for the distinct values, not two.

        Args:
            monkeypatch: Counts the calls that reach the shared categoriser.
        """
        import digitalearth.base.symbology as symbology

        scanned = []
        categorise = symbology.categorical_colors

        def counted(values, **kwargs):
            scanned.append(int(np.asarray(values, dtype=object).size))
            return categorise(values, **kwargs)

        monkeypatch.setattr(symbology, "categorical_colors", counted)
        points = self._table(60)
        labels = np.array(["a", "b", "c"] * 20, dtype=object)
        scene = Scene3D(off_screen=True)
        try:
            scene.point_cloud(points, values=labels, scheme="categorical", cmap="tab10")
        finally:
            scene.close()
        assert scanned == [60, 3], (
            "the column must be scanned once: the second call is over the three categories the first one "
            "found — the key's length, not the data's — because the colours are always sampled from this "
            f"call's cmap (review R2-H3). It was scanned {scanned}"
        )

    @staticmethod
    def _painted(actor):
        """Read the class colours back off the engine's own lookup table.

        Args:
            actor: The layer's drawn actor.

        Returns:
            list[str]: one ``#rrggbb`` per class, as VTK holds them. Read off the render rather than out of
            the style dict the drawer built, so a comparison against a computed colour list has two
            differently-built sides instead of one expression twice.
        """
        from matplotlib.colors import to_hex

        rows = np.asarray(actor.mapper.lookup_table.values, dtype="float64") / 255.0
        return [to_hex(row[:3]) for row in rows]

    @pytest.mark.parametrize(
        "values, scheme, k, cmap",
        [
            (None, "quantiles", 4, "magma"),
            (None, "equal_interval", 3, "cividis"),
            (["a", "b", "c", "a", "b", "c"], "categorical", 5, "Set1"),
        ],
    )
    def test_the_reused_cut_paints_exactly_what_a_fresh_one_would(
        self, values, scheme, k, cmap
    ):
        """Reusing the description's cut is a saving, not a second answer.

        Args:
            values: A categorical column, or `None` for the table's own x column.
            scheme: How the values are classified.
            k: How many classes a graduated scheme cuts.
            cmap: The colormap the classes are drawn from — a named one per case rather than the tier
                default, so a colour that came from somewhere other than this argument shows up.

        Test scenario:
            The saving is only sound if the two computations were identical to begin with, so **both**
            halves of the picture are checked: the class indices bound on the cloud, and the colours those
            indices are painted with.

            Neither expectation goes back through `classified_scalars`. The earlier version of this test
            called it for the comparison, which put both sides of the assertion through the one function
            under test — every mutation inside it moved them together, and it passed with the colours
            reversed (review R2-H3). The expectation is derived here from the shared primitives instead:
            the edges from :meth:`~digitalearth.base.spec.scale.Scale.breaks_of` and the class of a value
            counted off them, the categories and their colours from
            :func:`~digitalearth.base.symbology.categorical_colors`, the ramp's from
            :func:`~digitalearth.base.symbology.sample_cmap`. The drawn side is read off the engine — the
            array bound on the cloud and VTK's own lookup table.
        """
        from digitalearth.base.spec import Scale
        from digitalearth.base.symbology import (
            categorical_colors,
            resolve_categorical_cmap,
            sample_cmap,
        )

        points = self._table(6)
        column = points[:, 0] if values is None else np.array(values, dtype=object)
        if scheme == "categorical":
            names, wanted_colours = categorical_colors(
                column, cmap=resolve_categorical_cmap(cmap)
            )
            wanted_codes = [float(names.index(value)) for value in column]
        else:
            numbers = np.asarray(column, dtype="float64")
            edges = Scale.breaks_of(numbers, scheme, k)
            wanted_colours = sample_cmap(cmap, len(edges) - 1)
            # A value's class is how many interior edges it has passed — the same rule the drawer reaches
            # through `np.digitize`, counted rather than binned so the two sides are not one call twice.
            wanted_codes = [
                float(sum(value >= edge for edge in edges[1:-1])) for value in numbers
            ]
        scene = Scene3D(off_screen=True)
        try:
            scene.point_cloud(points, values=column, scheme=scheme, k=k, cmap=cmap)
            drawn = [float(code) for code in scene.layers[0][0][SCALAR]]
            painted = self._painted(scene.layers[0][1])
        finally:
            scene.close()
        assert drawn == wanted_codes, (
            f"the cloud was drawn with classes {drawn}, and the cut asks for {wanted_codes}"
        )
        assert painted == list(wanted_colours), (
            f"the cloud was painted {painted}, and the cut asks for {list(wanted_colours)}"
        )

    @pytest.mark.parametrize(
        "changed, values, scheme, k, cmap",
        [
            ({"k": 6}, None, "quantiles", 3, "viridis"),
            # A skewed column, because the two schemes cut an evenly-spaced one into the same classes and
            # the case would then pass with the scheme ignored.
            (
                {"scheme": "equal_interval"},
                [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
                "quantiles",
                2,
                "viridis",
            ),
            (
                {"cmap": "Set1"},
                ["a", "b", "c", "a", "b", "c"],
                "categorical",
                5,
                "tab10",
            ),
        ],
    )
    def test_a_fresh_request_overrules_the_cut_the_description_carries(
        self, changed, values, scheme, k, cmap
    ):
        """A reused cut is a saving on an unchanged request, never a veto over a changed one.

        Args:
            changed: The symbology property `replace_layer` asks for, replacing the one it was built with.
            values: A categorical column, or `None` for the table's own x column.
            scheme: The scheme the layer is first built with.
            k: The class count it is first built with.
            cmap: The colormap it is first built with.

        Test scenario:
            The two sides are built from **different** requests, which is what the previous version of the
            reuse test could not do: the layer's description carries the cut its builder made, and
            `replace_layer` then installs a symbology asking for another `k`, another `scheme` or another
            `cmap` without republishing the encoding. Handing that stale cut to the drawer unconditionally
            made it win outright — 6 classes asked for and 3 painted, `Set1` asked for and `tab10` painted
            (review R2-H3). The picture is compared against a fresh classification of the **new** request,
            computed with no description at all.
        """
        from dataclasses import replace as with_fields

        from digitalearth.three_d.base import classified_scalars

        points = self._table(6)
        column = points[:, 0] if values is None else np.array(values, dtype=object)
        asked = {"scheme": scheme, "k": k, "cmap": cmap, **changed}
        wanted = classified_scalars(column, **asked)
        scene = Scene3D(off_screen=True)
        try:
            scene.point_cloud(points, values=column, scheme=scheme, k=k, cmap=cmap)
            held = scene.get_layer("point_cloud-1")
            props = {**held.symbology.props, **changed}
            scene.replace_layer(
                with_fields(held, symbology=with_fields(held.symbology, props=props))
            )
            drawn = np.asarray(scene.layers[0][0][SCALAR], dtype="float64")
            painted = self._painted(scene.layers[0][1])
        finally:
            scene.close()
        assert np.array_equal(
            drawn, np.asarray(wanted["scalars"], dtype="float64"), equal_nan=True
        ), (
            f"replace_layer({changed}) painted {drawn}, not the {wanted['scalars']} it asked for"
        )
        assert painted == list(wanted["cmap"]), (
            f"replace_layer({changed}) painted {painted}, not the {list(wanted['cmap'])} it asked for"
        )

    def test_the_reused_categories_are_painted_in_this_calls_cmap(self):
        """A colour a description recorded cannot outvote the ``cmap`` the layer is drawn with.

        Test scenario:
            The reuse saves the pass that finds the distinct values; it settles nothing about what colour
            each of them gets. A scale carrying deliberately wrong colours is handed over with
            `cmap="Set1"`, and Set1 is what must reach the engine (review R2-H3).

            This is also what retires the shape the old read had to screen for —
            `Scale.categorical(["a", "b"], [None, None])`, which `Scale`'s one-colour-per-category
            constructor does let through and whose `color_for` answers `None`. No colour is read off a
            scale any more, so there is no `None` to bind where a colour belongs; the second case pins
            that it paints exactly like the first.

            Both sides are built differently from the expectation: the drawer samples the palette over the
            **categories the scale recorded**, while the expectation is taken from the categoriser over the
            **whole column** with no description in sight.
        """
        from digitalearth.base.spec import Scale
        from digitalearth.base.symbology import categorical_colors
        from digitalearth.three_d.base import classified_scalars

        column = np.array(["a", "b", "a"], dtype=object)
        asked = categorical_colors(column, cmap="Set1")[1]
        wrong = classified_scalars(
            column,
            scheme="categorical",
            k=2,
            cmap="Set1",
            scale=Scale.categorical(["a", "b"], ["#000000", "#000000"]),
        )
        colourless = classified_scalars(
            column,
            scheme="categorical",
            k=2,
            cmap="Set1",
            scale=Scale.categorical(["a", "b"], [None, None]),
        )
        assert wrong["cmap"] == asked, (
            f"the scale's own colours were painted instead of cmap='Set1': {wrong['cmap']} for {asked}"
        )
        assert colourless["cmap"] == asked, (
            f"a scale carrying no colour per category changed the picture: {colourless['cmap']}"
        )

    def test_a_recorded_category_set_from_another_column_is_not_painted_from(self):
        """Categories that cannot place this column's values are not this column's cut.

        Test scenario:
            The saving is the pass that finds the distinct values, so a recorded set is only a saving while
            it *is* this column's set. One from another column places nothing, and painting from it drew
            every feature as missing data — measured `scalars [nan nan]` for a two-value column, with no
            refusal (review R2-H3). Recognising that and categorising here is what keeps
            `replace_layer` able to re-point a cloud at another column.

            The second half pins the boundary: a value that is genuinely **missing** places nowhere either,
            and must not be read as a stale set — it is `NaN` by design, which is the rule
            :func:`~digitalearth.three_d.base._category_codes` exists to keep.
        """
        from digitalearth.base.spec import Scale
        from digitalearth.three_d.base import classified_scalars

        stale = Scale.categorical(["a", "b"], ["#ff0000", "#0000ff"])
        drawn = classified_scalars(
            np.array(["x", "y"], dtype=object),
            scheme="categorical",
            k=2,
            cmap="tab10",
            scale=stale,
        )
        assert [int(code) for code in drawn["scalars"]] == [0, 1], (
            f"a column the recorded categories do not cover was painted from them anyway: "
            f"{drawn['scalars']}"
        )
        with_missing = classified_scalars(
            np.array(["x", None], dtype=object),
            scheme="categorical",
            k=2,
            cmap="tab10",
            scale=Scale.categorical(["x"], ["#ff0000"]),
        )
        assert [float(code) for code in with_missing["scalars"]][0] == 0.0
        assert np.isnan(with_missing["scalars"][1]), (
            f"a missing value must stay missing, not re-categorise the column: {with_missing['scalars']}"
        )
