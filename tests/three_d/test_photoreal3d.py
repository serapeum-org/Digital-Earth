"""Tests for the photorealistic path-traced renderer (digitalearth.three_d.PhotorealMixin.photoreal, #496 B).

Gated on the optional ``[photoreal]`` extra (mitsuba + oidn) and on pyvista; exercises the dependency guard, the
geometry/colour helpers, and a small end-to-end headless render of a terrain scene.
"""

import numpy as np
import pytest

pytest.importorskip("pyvista")
pytest.importorskip("mitsuba")
pytest.importorskip("oidn")

from digitalearth.base.sources import get_source
from digitalearth.three_d import Scene3D


def _small_terrain_scene():
    """Build an off-screen Scene3D with a tiny coloured terrain for a fast render."""
    rng = np.random.default_rng(0)
    z = rng.random((10, 10)) * 3.0
    scene = Scene3D(off_screen=True)
    scene.terrain(get_source(z, x=np.arange(10.0), y=np.arange(10.0)), cmap="viridis")
    scene.view_angles(azimuth=225.0, elevation=30.0)
    return scene


class TestRequirePhotoreal:
    """`_require_photoreal` guards the optional mitsuba/oidn dependencies."""

    def test_missing_mitsuba_raises_an_actionable_error(self, monkeypatch):
        """Without mitsuba installed the guard raises an ImportError naming the install command."""
        import sys

        from digitalearth.three_d.photoreal import _require_photoreal

        monkeypatch.setitem(sys.modules, "mitsuba", None)
        with pytest.raises(ImportError, match=r"digitalearth\[photoreal\]"):
            _require_photoreal()


class TestColoredPly:
    """`_write_colored_ply` writes a binary PLY with positions, normals and float colours."""

    def test_header_and_record_counts_round_trip(self, tmp_path):
        """The written header declares the vertex/face counts and the float colour properties."""
        from digitalearth.three_d.photoreal import _write_colored_ply

        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")
        faces = np.array([[0, 1, 2]], dtype="int32")
        normals = np.tile([0.0, 0.0, 1.0], (3, 1)).astype("float32")
        rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        out = tmp_path / "tri.ply"
        _write_colored_ply(str(out), verts, faces, normals, rgb)
        head = out.read_bytes().split(b"end_header")[0].decode("ascii")
        assert "element vertex 3" in head, head
        assert "element face 1" in head, head
        assert "property float red" in head, head


class TestVertexRgb:
    """`_vertex_rgb` maps scalars through the actor LUT, or falls back to a solid colour."""

    def test_scalar_colours_are_in_unit_range(self):
        """A terrain layer's elevation maps to per-vertex RGB within [0, 1]."""
        from digitalearth.three_d.photoreal import _triangulated_surface, _vertex_rgb

        scene = _small_terrain_scene()
        mesh, actor = scene.layers[0]
        rgb = _vertex_rgb(_triangulated_surface(mesh), actor)
        assert rgb.shape[1] == 3, rgb.shape
        assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0, (
            rgb.min(),
            rgb.max(),
        )
        scene.close()

    def test_no_scalars_falls_back_to_the_actor_solid_colour(self):
        """A surface with no active scalars is tiled with the actor's own flat colour."""
        from types import SimpleNamespace

        import pyvista as pv

        from digitalearth.three_d.photoreal import _triangulated_surface, _vertex_rgb

        surf = _triangulated_surface(pv.Plane(i_resolution=2, j_resolution=2))
        actor = SimpleNamespace(prop=SimpleNamespace(color=(0.2, 0.4, 0.6)))
        rgb = _vertex_rgb(surf, actor)
        assert rgb.shape == (surf.n_points, 3), rgb.shape
        assert np.allclose(rgb, [0.2, 0.4, 0.6]), rgb[0]

    def test_lut_failure_falls_back_to_the_actor_solid_colour(self):
        """A LUT that raises while mapping scalars is swallowed and the flat actor colour is used instead."""
        from types import SimpleNamespace

        import pyvista as pv

        from digitalearth.three_d.photoreal import _triangulated_surface, _vertex_rgb

        surf = _triangulated_surface(pv.Plane(i_resolution=2, j_resolution=2))
        surf.point_data["elev"] = np.linspace(0.0, 1.0, surf.n_points)
        surf.set_active_scalars("elev")

        def _raise(*_args, **_kwargs):
            raise RuntimeError("lut boom")

        actor = SimpleNamespace(
            mapper=SimpleNamespace(lookup_table=SimpleNamespace(MapScalars=_raise)),
            prop=SimpleNamespace(color=(0.1, 0.2, 0.3)),
        )
        rgb = _vertex_rgb(surf, actor)
        assert np.allclose(rgb, [0.1, 0.2, 0.3]), rgb[0]

    def test_missing_prop_falls_back_to_the_default_solid_colour(self):
        """When even the actor's colour is unreadable, the module default solid RGB is used."""
        from types import SimpleNamespace

        import pyvista as pv

        from digitalearth.three_d.photoreal import (
            _DEFAULT_SOLID_RGB,
            _triangulated_surface,
            _vertex_rgb,
        )

        surf = _triangulated_surface(pv.Plane(i_resolution=2, j_resolution=2))
        rgb = _vertex_rgb(surf, SimpleNamespace())
        assert np.allclose(rgb, _DEFAULT_SOLID_RGB), rgb[0]


class TestPhotorealMeshes:
    """`PhotorealMixin._photoreal_meshes` skips layers that carry no renderable triangles."""

    def test_skips_layers_without_geometry(self, tmp_path):
        """A ``(None, None)`` layer (or an empty mesh) contributes no Mitsuba shape."""
        from types import SimpleNamespace

        from digitalearth.three_d.photoreal import PhotorealMixin

        fake = SimpleNamespace(layers=[(None, None)])
        shapes = PhotorealMixin._photoreal_meshes(fake, (0.0, 0.0, 0.0), tmp_path)
        assert shapes == {}, shapes

    def test_skips_meshes_with_no_faces(self, tmp_path):
        """A mesh whose triangulated surface carries zero faces is skipped rather than written as a shape."""
        from types import SimpleNamespace

        from digitalearth.three_d.photoreal import PhotorealMixin

        surf = SimpleNamespace(faces=np.empty(0, dtype="int32"))
        tri = SimpleNamespace(compute_normals=lambda **_kwargs: surf)
        extract = SimpleNamespace(triangulate=lambda: tri)
        mesh = SimpleNamespace(n_points=5, extract_surface=lambda: extract)
        fake = SimpleNamespace(layers=[(mesh, None)])
        shapes = PhotorealMixin._photoreal_meshes(fake, (0.0, 0.0, 0.0), tmp_path)
        assert shapes == {}, shapes


class TestPhotorealRender:
    """`Scene3D.photoreal` renders a headless path-traced still of the scene's geometry."""

    def test_renders_a_non_blank_image_and_writes_the_file(self, tmp_path):
        """A terrain scene renders to a non-blank uint8 RGB image and is written to the given path."""
        out = tmp_path / "relief.png"
        scene = _small_terrain_scene()
        img = scene.photoreal(str(out), spp=4, width=96, height=64, denoise=True)
        assert img.shape == (64, 96, 3), img.shape
        assert img.dtype == np.uint8, img.dtype
        assert int(img.max()) > 0, "the render must not be entirely black"
        assert out.exists(), "the image should be written to the given path"
        scene.close()

    def test_without_denoise_also_renders(self):
        """denoise=False skips OIDN and still returns a non-blank frame."""
        scene = _small_terrain_scene()
        img = scene.photoreal(spp=4, width=80, height=60, denoise=False)
        assert img.shape == (60, 80, 3) and int(img.max()) > 0, img.shape
        scene.close()

    def test_an_empty_scene_is_refused(self):
        """photoreal() on a scene with no geometry raises a clear ValueError."""
        scene = Scene3D(off_screen=True)
        with pytest.raises(ValueError, match="no renderable geometry"):
            scene.photoreal(spp=2, width=32, height=32)
        scene.close()

    def test_the_temp_directory_is_cleaned_up(self):
        """photoreal() removes its per-call temp directory, leaking none (M1)."""
        import glob
        import tempfile
        from pathlib import Path

        pattern = str(Path(tempfile.gettempdir()) / "digitalearth-photoreal-*")
        before = set(glob.glob(pattern))
        scene = _small_terrain_scene()
        scene.photoreal(spp=2, width=48, height=32, denoise=False)
        scene.close()
        leaked = set(glob.glob(pattern)) - before
        assert leaked == set(), f"photoreal() leaked temp directories: {leaked}"
