"""The 3-D labels verb: label features from a column, with halos and occlusion (#491).

`Scene3D.labels(features, column, ...)` is the tier's Core `labels` verb and the plural, styled companion to
`text`. These cover that it draws one layer for the collection, labels each feature from the column, that halo
and occlusion reach the engine, that a projected scene reprojects the features, the refusals, and the figure
round-trip.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

pv = pytest.importorskip("pyvista")

from digitalearth.base.crs import OffLimbError  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402


def _points() -> gpd.GeoDataFrame:
    """Return two point features with a name column.

    Returns:
        A GeoDataFrame of two EPSG:4326 points.
    """
    return gpd.GeoDataFrame(
        {"name": ["origin", "east"]},
        geometry=[Point(0.0, 0.0), Point(10.0, 5.0)],
        crs=4326,
    )


def _polygons() -> gpd.GeoDataFrame:
    """Return two polygon features with a value column.

    Returns:
        A GeoDataFrame of two EPSG:4326 polygons.
    """
    return gpd.GeoDataFrame(
        {"value": [1.0, 2.0]},
        geometry=[
            Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
            Polygon([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]),
        ],
        crs=4326,
    )


@pytest.fixture
def scene():
    """Yield an off-screen scene, closed on the way out.

    Yields:
        A scene under test.
    """
    built = Scene3D(off_screen=True)
    yield built
    built.close()


class TestLabels:
    """`Scene3D.labels` labels a feature collection from a column as one addressable layer."""

    def test_it_draws_one_labels_layer_for_the_collection(self, scene):
        """A collection of two features is one labels layer, not two.

        Args:
            scene: The scene under test.
        """
        scene.labels(_points(), "name")
        assert scene.layer_ids == ["labels-1"], scene.layer_ids

    def test_the_layer_kind_is_labels(self, scene):
        """The recorded kind is `labels`, distinct from `text`.

        Args:
            scene: The scene under test.
        """
        scene.labels(_points(), "name")
        assert scene.get_layer("labels-1").kind == "labels", scene.get_layer("labels-1")

    def test_it_returns_self_so_decoration_chains(self, scene):
        """The builder returns the scene.

        Args:
            scene: The scene under test.
        """
        assert scene.labels(_points(), "name") is scene, "labels must return self"

    def test_the_anchor_mesh_has_one_point_per_feature(self, scene):
        """The mesh the labels hang off carries one representative point per feature.

        Args:
            scene: The scene under test.
        """
        scene.labels(_polygons(), "value")
        assert scene.mesh_of("labels-1").n_points == 2, "one anchor per feature"

    def test_the_layer_is_removed_by_id(self, scene):
        """Removing the labels takes them off the figure.

        Args:
            scene: The scene under test.
        """
        scene.labels(_points(), "name")
        scene.remove_layer("labels-1")
        assert scene.layer_ids == [], scene.layer_ids

    def test_a_projected_scene_reprojects_the_features(self):
        """Over an EPSG:3857 scene the lon/lat features are reprojected to metres."""
        projected = Scene3D(off_screen=True, crs=3857)
        try:
            projected.labels(_points(), "name")
            x = projected.mesh_of("labels-1").points[1, 0]
            assert abs(float(x)) > 1000.0, "Web-Mercator metres, not degrees"
        finally:
            projected.close()


class TestLabelHaloAndOcclusion:
    """The halo setting reaches the PyVista label mapper's shape."""

    def test_a_halo_draws_a_rounded_background_shape(self, scene):
        """With `halo=True` the label mapper draws the rounded-rectangle background shape (VTK shape 2).

        Args:
            scene: The scene under test.
        """
        scene.labels(_points(), "name", halo=True)
        assert scene.actor_of("labels-1").GetMapper().GetShape() == 2, (
            "rounded-rect halo"
        )

    def test_no_halo_leaves_no_background_shape(self, scene):
        """With `halo=False` the mapper draws no background shape (VTK shape 0).

        Args:
            scene: The scene under test.
        """
        scene.labels(_points(), "name", halo=False)
        assert scene.actor_of("labels-1").GetMapper().GetShape() == 0, "no halo shape"


class TestLabelRefusals:
    """The refusals, each raising rather than drawing an empty or mislabelled layer."""

    def test_a_missing_column_is_refused(self, scene):
        """A column the features do not carry is named in the error.

        Args:
            scene: The scene under test.
        """
        with pytest.raises(KeyError, match="not a property"):
            scene.labels(_points(), "nope")

    def test_a_non_finite_text_size_is_refused(self, scene):
        """The text size has to be a real number.

        Args:
            scene: The scene under test.
        """
        with pytest.raises(ValueError, match="finite number"):
            scene.labels(_points(), "name", text_size=float("nan"))


class TestLabelsStrictAndRoundTrip:
    """Strict refusal when nothing labels, and the figure round-trip."""

    def test_a_strict_scene_refuses_an_empty_collection(self):
        """Under strict, a collection with no features to label raises."""
        empty = gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326)
        strict = Scene3D(off_screen=True, strict=True)
        try:
            with pytest.raises(OffLimbError):
                strict.labels(empty, "name")
        finally:
            strict.close()

    def test_an_empty_collection_is_skipped_without_strict(self, scene):
        """A non-strict scene skips an empty collection rather than drawing nothing.

        Args:
            scene: The scene under test.
        """
        empty = gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326)
        scene.labels(empty, "name")
        assert scene.layer_ids == [], "an empty collection draws no layer"

    def test_labels_are_redrawn_from_the_figure(self, scene):
        """`Scene3D.from_figure(scene.figure_spec)` redraws the same labels.

        Args:
            scene: The scene under test.

        Test scenario:
            A GeoDataFrame-backed layer references an in-process object, so — like `extruded_polygons` — it
            round-trips through the figure in memory rather than through `to_dict()`, which only stores
            path-backed sources.
        """
        scene.labels(_polygons(), "value")
        again = Scene3D.from_figure(scene.figure_spec, off_screen=True)
        try:
            assert again.get_layer("labels-1").kind == "labels", again.layer_ids
            assert again.mesh_of("labels-1").n_points == 2, "both features redrawn"
        finally:
            again.close()
