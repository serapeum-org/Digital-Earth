"""Tests for :class:`digitalearth.base.sources.mesh.MeshSource` — the unstructured-mesh value object.

The extraction that *builds* one from a pyramids ``UgridDataset`` is covered in ``test_sources.py``; this
file covers the container itself: what it validates, and what it hands back.
"""

import numpy as np
import pytest

from digitalearth.base.sources import MESH_LOCATIONS, DimensionInfo, MeshSource


def _axes(n: int = 4) -> tuple:
    """Build ``n`` node coordinates as the two axis dimensions a MeshSource takes.

    Args:
        n: How many nodes.

    Returns:
        tuple: the x and y :class:`DimensionInfo` pair.
    """
    return (
        DimensionInfo(np.linspace(0.0, 1.0, n), "x"),
        DimensionInfo(np.zeros(n), "y"),
    )


class TestMeshSource:
    """Tests for the MeshSource container."""

    def test_init_keeps_the_triangulation_and_the_location(self):
        """The triangles and the location are handed back exactly as given.

        Test scenario:
            A two-triangle mesh with one value per face. The container is a record, so it must not
            reorder, copy-and-alter or re-derive what it was given.
        """
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        x, y = _axes()
        src = MeshSource(
            DimensionInfo(np.array([10.0, 20.0]), "z"),
            x,
            y,
            triangles=faces,
            location="face",
            crs=4326,
            units="m",
        )
        assert src.triangles.tolist() == faces.tolist(), (
            f"triangles were altered: {src.triangles.tolist()}"
        )
        assert src.location == "face", f"expected 'face', got {src.location!r}"
        assert src.n_triangles == 2, f"expected 2 triangles, got {src.n_triangles}"
        assert src.units == "m", f"units not carried: {src.units!r}"
        assert src.crs == 4326, f"crs not carried: {src.crs!r}"

    @pytest.mark.parametrize("location", list(MESH_LOCATIONS))
    def test_init_accepts_every_declared_location(self, location):
        """Each location the module declares is accepted.

        Args:
            location: One of :data:`MESH_LOCATIONS`.

        Test scenario:
            Guards the pair: a location added to the tuple but rejected by the constructor — or the
            reverse — is a contradiction this catches.
        """
        x, y = _axes(3)
        src = MeshSource(None, x, y, triangles=np.array([[0, 1, 2]]), location=location)
        assert src.location == location

    @pytest.mark.parametrize("location", ["edge", "volume", "", "FACE", None])
    def test_init_refuses_a_location_no_surface_render_can_colour(self, location):
        """A location outside the declared pair is refused, rather than guessed at by a renderer.

        Args:
            location: A location a 2-D surface render cannot colour.

        Test scenario:
            CF-UGRID names more locations than a surface can shade. Accepting one would leave a renderer
            to pick which array to colour, which is exactly the guess this refusal prevents. ``"FACE"``
            is included because the comparison is deliberately case-sensitive.
        """
        x, y = _axes(3)
        with pytest.raises(ValueError, match="is not one of"):
            MeshSource(None, x, y, triangles=np.array([[0, 1, 2]]), location=location)

    def test_n_triangles_is_zero_when_there_is_no_triangulation(self):
        """An empty or non-2-D triangulation reports no triangles instead of raising on ``.shape[0]``.

        Test scenario:
            ``n_triangles`` is read for messages (an off-limb refusal names the count), so it has to
            answer for a degenerate mesh rather than raise while a refusal is being worded.
        """
        x, y = _axes(3)
        src = MeshSource(None, x, y, triangles=np.array([]), location="node")
        assert src.n_triangles == 0, f"expected 0, got {src.n_triangles}"

    def test_triangles_are_an_array_even_when_handed_a_list(self):
        """A nested list is normalised to an array, so a drawer can index it without converting first."""
        x, y = _axes(3)
        src = MeshSource(None, x, y, triangles=[[0, 1, 2]], location="node")
        assert isinstance(src.triangles, np.ndarray), (
            f"expected an ndarray, got {type(src.triangles).__name__}"
        )
        assert src.n_triangles == 1

    def test_it_is_a_source_so_the_shared_readers_still_work(self):
        """A MeshSource answers the Source contract — the axes, the CRS and the metadata fallback.

        Test scenario:
            Everything that already reads a ``Source`` (the style matcher, the field-name lookup) is
            handed these too, so the inherited half has to keep working rather than being shadowed.
        """
        x, y = _axes(3)
        src = MeshSource(
            DimensionInfo(np.arange(3.0), "z"),
            x,
            y,
            triangles=np.array([[0, 1, 2]]),
            location="node",
            metadata={"kind": "mesh"},
        )
        assert src.metadata("kind") == "mesh"
        assert src.metadata("variable", "FALLBACK") == "FALLBACK"
        assert src.x.values.size == 3 and src.y.values.size == 3
        assert src.crs is None and src.epsg is None
