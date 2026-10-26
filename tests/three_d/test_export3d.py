"""High-resolution raster masters and vector export for the 3-D tier (#494).

`screenshot(scale=N)` renders an N-times master; `export_vector(path)` / `save("scene.svg")` write a vector
graphic through VTK's GL2PS exporter. These cover the scale factor, the vector formats, the refusal of an
unsupported suffix, and the `save` dispatch.
"""

import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.three_d import Scene3D  # noqa: E402


@pytest.fixture
def scene():
    """Yield an off-screen scene with one mesh, closed on the way out.

    Yields:
        A scene carrying a sphere so there is something to render.
    """
    built = Scene3D(off_screen=True, window_size=(200, 150))
    built.add_mesh(pv.Sphere())
    yield built
    built.close()


class TestHighResolutionMaster:
    """`screenshot(scale=N)` renders at N times the window resolution."""

    def test_scale_multiplies_the_pixel_dimensions(self, scene):
        """A scale of 2 doubles the rendered frame's height and width.

        Args:
            scene: The scene under test.
        """
        base = scene.screenshot()
        master = scene.screenshot(scale=2)
        assert master.shape[0] == base.shape[0] * 2, (master.shape, base.shape)
        assert master.shape[1] == base.shape[1] * 2, (master.shape, base.shape)


class TestVectorExport:
    """`export_vector` writes a vector graphic through GL2PS, and `save` routes the vector suffixes to it."""

    @pytest.mark.parametrize("suffix", [".svg", ".pdf", ".eps"])
    def test_it_writes_a_non_empty_vector_file(self, scene, tmp_path, suffix):
        """Each supported vector suffix produces a non-empty file.

        Args:
            scene: The scene under test.
            tmp_path: Where the file is written.
            suffix: The vector suffix under test.
        """
        out = scene.export_vector(tmp_path / f"scene{suffix}")
        assert out.stat().st_size > 0, f"{suffix} export was empty"

    def test_it_returns_the_path_it_wrote(self, scene, tmp_path):
        """The written path is handed back, so a caller can chain on it.

        Args:
            scene: The scene under test.
            tmp_path: Where the file is written.
        """
        out = scene.export_vector(tmp_path / "scene.svg")
        assert out.suffix == ".svg", out

    def test_an_unsupported_suffix_is_refused_by_name(self, scene, tmp_path):
        """A raster suffix is not a vector format, and the refusal names the supported set.

        Args:
            scene: The scene under test.
            tmp_path: An unused path.
        """
        with pytest.raises(ValueError, match="not one it can write"):
            scene.export_vector(tmp_path / "scene.png")

    def test_kwargs_with_a_vector_suffix_are_refused(self, scene, tmp_path):
        """`save("x.svg", scale=2)` refuses the keyword rather than dropping it silently.

        Args:
            scene: The scene under test.
            tmp_path: An unused output path.
        """
        out = tmp_path / "scene.svg"
        with pytest.raises(ValueError, match="no keywords"):
            scene.save(out, scale=2)

    def test_save_routes_a_vector_suffix_to_export_vector(self, scene, tmp_path):
        """`save("scene.svg")` dispatches to the vector exporter and returns the path.

        Args:
            scene: The scene under test.
            tmp_path: Where the file is written.
        """
        out = scene.save(tmp_path / "via_save.svg")
        assert out.stat().st_size > 0, "save() did not write the vector file"
