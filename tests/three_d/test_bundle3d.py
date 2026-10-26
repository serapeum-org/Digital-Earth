"""Scene bundles — save a 3-D scene to a versioned zip and open it back (#495).

`Scene3D.save_bundle(path)` writes a zip of the figure description plus the drawn decoration;
`Scene3D.open_bundle(path)` reads it back into an equivalent scene. These cover the round trip for a
file-backed scene (layers, view and decoration preserved), the refusals for the data a description cannot
store, and the format-version check.
"""

import json
import zipfile

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from digitalearth.base.sources import get_source  # noqa: E402
from digitalearth.three_d import Scene3D  # noqa: E402

DEM_PATH = "examples/data/acc4000.tif"


class TestBundleRoundTrip:
    """A file-backed scene bundles and reopens to the same picture."""

    def test_the_layers_survive_the_round_trip(self, tmp_path):
        """A terrain drawn from a path is re-opened from it on load.

        Args:
            tmp_path: Where the bundle is written.
        """
        scene = Scene3D(off_screen=True)
        scene.terrain(DEM_PATH)
        out = scene.save_bundle(tmp_path / "scene.dez")
        scene.close()
        reopened = Scene3D.open_bundle(out, off_screen=True)
        try:
            assert reopened.layer_ids == ["terrain-1"], reopened.layer_ids
        finally:
            reopened.close()

    def test_the_title_survives_the_round_trip(self, tmp_path):
        """The figure's title is restored.

        Args:
            tmp_path: Where the bundle is written.
        """
        scene = Scene3D(off_screen=True)
        scene.terrain(DEM_PATH)
        scene.set_title("Relief")
        out = scene.save_bundle(tmp_path / "scene.dez")
        scene.close()
        reopened = Scene3D.open_bundle(out, off_screen=True)
        try:
            assert reopened.figure_spec.panels[0].title == "Relief", (
                reopened.figure_spec.panels[0]
            )
        finally:
            reopened.close()

    def test_the_decoration_survives_the_round_trip(self, tmp_path):
        """A corner attribution set before bundling is re-applied on open.

        Args:
            tmp_path: Where the bundle is written.
        """
        scene = Scene3D(off_screen=True)
        scene.terrain(DEM_PATH)
        scene.attribution("© data")
        out = scene.save_bundle(tmp_path / "scene.dez")
        scene.close()
        reopened = Scene3D.open_bundle(out, off_screen=True)
        try:
            assert (
                reopened._decoration.get("attribution", {}).get("text") == "© data"
            ), reopened._decoration
        finally:
            reopened.close()


class TestBundleRefusals:
    """The data a description cannot store is refused with a clear message."""

    def test_an_in_memory_source_is_refused(self, tmp_path):
        """A terrain drawn from an in-memory array has no path to point back at.

        Args:
            tmp_path: Where the bundle would be written.
        """
        scene = Scene3D(off_screen=True)
        scene.terrain(get_source(np.add.outer(np.arange(6.0), np.arange(6.0))))
        try:
            with pytest.raises(ValueError, match="path or URL"):
                scene.save_bundle(tmp_path / "scene.dez")
        finally:
            scene.close()

    def test_non_json_decoration_is_refused(self, tmp_path):
        """Decoration carrying a non-JSON value is refused with a clear message, not a cryptic TypeError.

        Args:
            tmp_path: Where the bundle would be written.
        """
        scene = Scene3D(off_screen=True)
        scene.terrain(DEM_PATH)
        scene._decoration["axes"] = {"bad": object()}
        try:
            with pytest.raises(ValueError, match="JSON-serialisable"):
                scene.save_bundle(tmp_path / "scene.dez")
        finally:
            scene.close()

    def test_a_caller_built_mesh_is_refused(self, tmp_path):
        """A scene holding an add_mesh object cannot be bundled.

        Args:
            tmp_path: Where the bundle would be written.
        """
        scene = Scene3D(off_screen=True)
        scene.add_mesh(pv.Sphere(), name="ball")
        try:
            with pytest.raises(ValueError, match="caller-built meshes"):
                scene.save_bundle(tmp_path / "scene.dez")
        finally:
            scene.close()


class TestBundleFormat:
    """The versioned container, and its refusals."""

    def test_the_bundle_is_a_zip_with_a_manifest(self, tmp_path):
        """A written bundle is a zip carrying a versioned manifest.json.

        Args:
            tmp_path: Where the bundle is written.
        """
        scene = Scene3D(off_screen=True)
        scene.terrain(DEM_PATH)
        out = scene.save_bundle(tmp_path / "scene.dez")
        scene.close()
        with zipfile.ZipFile(out) as bundle:
            manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["format_version"] == 1, manifest.get("format_version")

    def test_a_non_json_figure_is_refused_by_the_writer(self, tmp_path):
        """A non-JSON value anywhere in the manifest is refused with a clear ValueError, not a raw TypeError.

        Args:
            tmp_path: Where the bundle would be written.
        """
        from digitalearth.three_d.app.bundle import write_bundle

        bad_figure = {"panels": {1, 2, 3}}
        with pytest.raises(ValueError, match="not JSON-serialisable"):
            write_bundle(tmp_path / "scene.dez", figure_dict=bad_figure, decoration={})

    def test_a_v1_manifest_without_a_figure_is_refused(self, tmp_path):
        """A format-1 manifest missing its figure section is refused as corrupt, not a bare KeyError.

        Args:
            tmp_path: Where the file is written.
        """
        out = tmp_path / "nofigure.dez"
        with zipfile.ZipFile(out, "w") as bundle:
            bundle.writestr(
                "manifest.json", json.dumps({"format_version": 1, "decoration": {}})
            )
        with pytest.raises(ValueError, match="missing its 'figure'"):
            Scene3D.open_bundle(out, off_screen=True)

    def test_an_unknown_format_version_is_refused(self, tmp_path):
        """A bundle from a newer format is refused rather than mis-read.

        Args:
            tmp_path: Where the bundle is written.
        """
        out = tmp_path / "future.dez"
        with zipfile.ZipFile(out, "w") as bundle:
            bundle.writestr(
                "manifest.json", json.dumps({"format_version": 999, "figure": {}})
            )
        with pytest.raises(ValueError, match="format version"):
            Scene3D.open_bundle(out, off_screen=True)

    def test_a_file_without_a_manifest_is_refused(self, tmp_path):
        """A zip that is not a scene bundle is refused by name.

        Args:
            tmp_path: Where the file is written.
        """
        out = tmp_path / "notabundle.dez"
        with zipfile.ZipFile(out, "w") as bundle:
            bundle.writestr("other.txt", "hello")
        with pytest.raises(ValueError, match="not a digitalearth scene bundle"):
            Scene3D.open_bundle(out, off_screen=True)
