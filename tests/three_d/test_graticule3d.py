"""The 3-D graticule: a latitude/longitude grid drawn as a layer, reprojected through pyramids (#490).

`Scene3D.graticule(...)` draws meridians and parallels across the framed extent and labels them along the
edges. These cover that it is a drawn layer (id, visibility, removal, round-trip), that the step is chosen or
honoured, that the grid is reprojected into a projected display CRS rather than drawn in degrees there, and that
the argument refusals hold.
"""

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.crs import OffLimbError  # noqa: E402
from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.base.spec import FigureSpec  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402

BOX = (-10.0, 35.0, 30.0, 60.0)


@pytest.fixture
def scene():
    """Yield an off-screen geographic scene, closed on the way out.

    Yields:
        A scene whose display CRS is EPSG:4326, so a graticule in degrees is drawn without reprojection.
    """
    built = Scene3D(off_screen=True, crs=4326)
    yield built
    built.close()


class TestGraticule:
    """`Scene3D.graticule` draws a lon/lat grid as an addressable layer."""

    def test_it_draws_a_graticule_layer(self, scene):
        """The call records one layer of the `graticule` kind.

        Args:
            scene: The scene under test.
        """
        scene.graticule(bbox=BOX, spacing=10.0)
        assert scene.get_layer(scene.layer_ids[-1]).kind == "graticule", scene.layer_ids

    def test_an_explicit_step_sets_the_line_count(self, scene):
        """`spacing=10` over the box draws five meridians and three parallels — eight line cells.

        Args:
            scene: The scene under test.
        """
        scene.graticule(bbox=BOX, spacing=10.0, labels=False)
        assert scene.mesh_of(scene.layer_ids[-1]).n_cells == 8, (
            "five meridians + three parallels"
        )

    def test_an_automatic_step_still_draws_lines(self, scene):
        """With no step, a readable spacing is chosen and the grid is non-empty.

        Args:
            scene: The scene under test.
        """
        scene.graticule(bbox=BOX, labels=False)
        assert scene.mesh_of(scene.layer_ids[-1]).n_cells > 0, "a grid was drawn"

    def test_it_returns_self_so_decoration_chains(self, scene):
        """The builder returns the scene.

        Args:
            scene: The scene under test.
        """
        assert scene.graticule(bbox=BOX) is scene, "graticule must return self"

    def test_the_layer_is_removed_by_id(self, scene):
        """Removing the graticule takes it off the figure.

        Args:
            scene: The scene under test.
        """
        scene.graticule(bbox=BOX)
        scene.remove_layer(scene.layer_ids[-1])
        assert scene.layer_ids == [], scene.layer_ids

    def test_the_layer_is_hidden_and_shown(self, scene):
        """Visibility is recorded and applied to the actor.

        Args:
            scene: The scene under test.
        """
        scene.graticule(bbox=BOX)
        layer_id = scene.layer_ids[-1]
        scene.set_visible(layer_id, False)
        assert scene.renderer.is_visible(layer_id) is False, "hidden"

    def test_labels_add_an_actor(self):
        """A labelled grid draws more actors than an unlabelled one of the same extent."""
        labelled = Scene3D(off_screen=True, crs=4326)
        plain = Scene3D(off_screen=True, crs=4326)
        try:
            labelled.graticule(bbox=BOX, spacing=10.0, labels=True)
            plain.graticule(bbox=BOX, spacing=10.0, labels=False)
            assert len(labelled.plotter.actors) > len(plain.plotter.actors), (
                "labels draw an actor"
            )
        finally:
            labelled.close()
            plain.close()

    def test_a_projected_scene_reprojects_the_grid(self):
        """Over an EPSG:3857 scene the degrees are reprojected to metres, not drawn as degrees."""
        projected = Scene3D(off_screen=True, crs=3857)
        try:
            projected.graticule(bbox=BOX, spacing=10.0, crs=4326, labels=False)
            xs = projected.mesh_of(projected.layer_ids[-1]).points[:, 0]
            assert float(np.abs(xs).max()) > 1000.0, (
                "Web-Mercator metres, not lon/lat degrees"
            )
        finally:
            projected.close()


class TestGraticuleRefusals:
    """The argument refusals, each raising before anything is drawn."""

    def test_a_bbox_that_is_not_four_numbers_is_refused(self, scene):
        """A three-number box cannot be a (west, south, east, north).

        Args:
            scene: The scene under test.
        """
        with pytest.raises(ValueError, match="west, south, east, north"):
            scene.graticule(bbox=(1.0, 2.0, 3.0))

    def test_a_non_positive_step_is_refused(self, scene):
        """A grid step must be a positive number.

        Args:
            scene: The scene under test.
        """
        with pytest.raises(ValueError, match="must be positive"):
            scene.graticule(bbox=BOX, spacing=0.0)

    def test_a_non_finite_z_is_refused(self, scene):
        """A grid anchored at NaN would be drawn nowhere.

        Args:
            scene: The scene under test.
        """
        nan = float("nan")
        with pytest.raises(ValueError, match="finite number"):
            scene.graticule(bbox=BOX, z=nan)


class TestGraticuleFromTheDrawnExtent:
    """With no bbox, the grid is framed on the scene's drawn extent, reprojected as needed."""

    def test_it_frames_a_geographic_scene_on_its_drawn_extent(self, scene):
        """A graticule with no bbox grids the extent of the data already drawn.

        Args:
            scene: The scene under test (EPSG:4326).
        """
        lon = np.linspace(-10.0, 30.0, 20)
        lat = np.linspace(35.0, 60.0, 16)
        scene.terrain(
            get_source(
                np.add.outer(np.cos(np.radians(lat)), np.sin(np.radians(lon))),
                x=lon,
                y=lat,
            )
        )
        scene.graticule(spacing=10.0, labels=False)
        assert scene.mesh_of(scene.layer_ids[-1]).n_cells > 0, (
            "the grid framed the drawn extent"
        )

    def test_it_reprojects_the_extent_of_a_projected_scene(self):
        """Over an EPSG:3857 scene the drawn extent is reprojected into the grid's CRS to pick the lines."""
        mx = np.linspace(-1.0e6, 3.0e6, 20)
        my = np.linspace(4.0e6, 8.0e6, 16)
        projected = Scene3D(off_screen=True, crs=3857)
        try:
            projected.terrain(get_source(np.add.outer(my / 1e6, mx / 1e6), x=mx, y=my))
            projected.graticule(crs=4326, labels=False)
            assert projected.mesh_of(projected.layer_ids[-1]).n_cells > 0, (
                "grid framed the reprojected extent"
            )
        finally:
            projected.close()


class TestGraticuleProjectedAutoStep:
    """The auto step must scale to the grid CRS's units, not assume degrees."""

    def test_a_projected_crs_without_a_step_reads(self, scene):
        """Over a projected box (metres) with no step, the auto step picks a readable interval, not 30 units.

        Args:
            scene: The scene under test.

        Test scenario:
            A 4,000,000 m span with a degrees-only auto step would fall back to a 30 m grid and hit the
            1000-line safety cap, drawing a dense meaningless grid; a unit-agnostic step keeps it readable.
        """
        scene.graticule(
            bbox=(0.0, 0.0, 4_000_000.0, 4_000_000.0), crs=32618, labels=False
        )
        lines = scene.mesh_of(scene.layer_ids[-1]).n_cells
        assert lines < 100, (
            f"auto-step produced a degenerate {lines}-line grid over a projected span"
        )


class TestGraticuleUnitMismatchWarning:
    """A geographic grid CRS over clearly non-degree bounds (no display CRS) warns rather than failing silently."""

    def test_a_projected_bare_array_scene_warns_about_units(self, caplog):
        """A CRS-less scene whose data is in metres, gridded with the default crs=4326, warns about the units."""
        import logging

        axis = np.linspace(0.0, 4_000_000.0, 16)
        built = Scene3D(off_screen=True)
        built.terrain(get_source(np.add.outer(axis / 1e6, axis / 1e6), x=axis, y=axis))
        with caplog.at_level(logging.WARNING):
            built.graticule(labels=False)
        built.close()
        assert "far outside" in caplog.text, (
            f"no unit-mismatch warning emitted: {caplog.text!r}"
        )


class TestGraticuleWithoutAnExtent:
    """With no bbox and nothing drawn, there is no extent to frame."""

    def test_a_graticule_with_no_extent_is_skipped(self, scene):
        """A non-strict scene skips the layer rather than drawing an empty grid.

        Args:
            scene: The scene under test.
        """
        scene.graticule()
        assert scene.layer_ids == [], "nothing to frame, nothing drawn"

    def test_a_strict_scene_refuses_an_empty_graticule(self):
        """Under strict, the same case raises."""
        strict = Scene3D(off_screen=True, crs=4326, strict=True)
        try:
            with pytest.raises(OffLimbError):
                strict.graticule()
        finally:
            strict.close()


class TestGraticuleRoundTrip:
    """The grid is described, so it is drawn again from the figure."""

    def test_a_graticule_survives_the_figure_round_trip(self, scene):
        """`FigureSpec.from_dict(scene.figure_spec.to_dict())` redraws the same grid.

        Args:
            scene: The scene under test.
        """
        scene.graticule(bbox=BOX, spacing=10.0, labels=False)
        stored = scene.figure_spec.to_dict()
        again = Scene3D.from_figure(
            FigureSpec.from_dict(stored), off_screen=True, crs=4326
        )
        try:
            assert again.get_layer(again.layer_ids[-1]).kind == "graticule", (
                again.layer_ids
            )
            assert again.mesh_of(again.layer_ids[-1]).n_cells == 8, (
                "the same eight lines"
            )
        finally:
            again.close()
