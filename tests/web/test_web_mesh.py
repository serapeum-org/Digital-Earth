"""The web tier draws a UGRID mesh's faces as a deck.gl ``PolygonLayer`` (#547).

``quickmap(mesh, backend="web")`` classifies a ``MeshDataset`` as the ``mesh`` family, unwraps it to its bare
``Mesh2d`` (``api._mesh_of``) and routes it to ``WebMap.trimesh``, which fan-cuts the faces into triangles
(pyramids' own connectivity), reprojects the nodes to the tier's only CRS (EPSG:4326) and draws each triangle
as one polygon on the page's one GPU deck overlay — the web sibling of the static ``tripcolor`` and interactive
``trimesh`` renders. Runs in the ``web`` pixi env; every test ``importorskip``s maplibre.
"""

import math

import numpy as np
import pytest

from digitalearth import api, quickmap

pytest.importorskip("maplibre", reason="the web tier needs the web environment")

from digitalearth.web import WebMap  # noqa: E402


def _flat_coords(layer: dict) -> list:
    """Return every coordinate of every triangle a drawn mesh layer holds.

    Args:
        layer: The deck.gl ``PolygonLayer`` dict ``WebMap.layers`` reports for the mesh.

    Returns:
        A flat list of the lon/lat numbers across all the triangle rings.
    """
    return [
        value for row in layer["data"] for corner in row["polygon"] for value in corner
    ]


def test_quickmap_draws_a_ugrid_mesh_on_the_web_tier(ugrid_mesh):
    """``quickmap`` builds a ``WebMap`` whose one layer is the mesh's own faces as a deck polygon layer.

    Test scenario:
        The two-face fixture (a quad and a triangle) fan-cuts into three triangles, so the GPU layer carries
        three polygons — the mesh's own connectivity rather than a re-triangulation. The layer is addressable
        by id, and its coordinates are the finite lon/lat the EPSG:4326 tier draws in.
    """
    dataset = ugrid_mesh("face")
    web_map = quickmap(dataset, backend="web")

    assert isinstance(web_map, WebMap), type(web_map)
    assert web_map.figure_spec.layers.get(web_map.layer_ids[0]).kind == "mesh"

    layers = web_map.layers
    assert len(layers) == 1, f"expected one mesh layer, got {len(layers)}"
    layer = layers[0]
    assert layer["@@type"] == "PolygonLayer", layer["@@type"]
    # One polygon per fan triangle: the quad -> 2, the triangle -> 1, which is the mesh's own faces, unwrapped
    # from the MeshDataset to its Mesh2d rather than triangulated afresh.
    assert len(layer["data"]) == len(dataset.mesh.fan_triangles) == 3
    assert all(len(row["polygon"]) == 3 for row in layer["data"]), (
        "each face is a triangle"
    )

    coords = _flat_coords(layer)
    assert all(math.isfinite(value) for value in coords), coords
    # The fixture mesh declares no CRS, so its own node coordinates (0..2) come through as lon/lat — the tier
    # draws in EPSG:4326, and nothing was mis-placed as metres.
    assert max(coords) == 2.0, coords


def test_trimesh_reprojects_the_mesh_to_4326():
    """A mesh that declares another CRS is warped to the tier's EPSG:4326 before its faces are drawn.

    Test scenario:
        The nodes are hundreds of thousands of metres in EPSG:3857; after ``trimesh`` they must be small
        lon/lat degrees, which is the reprojection the EPSG:4326-only tier does through pyramids. A builder
        that drew the raw metres on a lon/lat canvas would land the mesh near null island at absurd scale.
    """
    from pyramids.base.crs import crs_from_user_input
    from pyramids.netcdf.ugrid.connectivity import Connectivity
    from pyramids.netcdf.ugrid.mesh import Mesh2d

    mesh = Mesh2d(
        node_x=np.array([500000.0, 600000.0, 600000.0, 500000.0, 700000.0]),
        node_y=np.array([6800000.0, 6800000.0, 6900000.0, 6900000.0, 6850000.0]),
        face_node_connectivity=Connectivity(
            data=np.array([[0, 1, 2, 3], [1, 4, 2, -1]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )
    mesh.crs = crs_from_user_input(3857)

    layer = WebMap().trimesh(mesh).layers[0]
    coords = _flat_coords(layer)
    # Metres became degrees: every coordinate is now inside the lon/lat range, nowhere near the input 5e5+.
    assert coords, "the mesh placed none of its faces"
    assert all(abs(value) <= 180.0 for value in coords), coords
    assert max(abs(value) for value in coords) < 100.0, coords


def test_the_mesh_layer_is_addressable_and_named():
    """The mesh joins the layer tree like any web layer: it carries the caller's name as its id."""
    from pyramids.netcdf.ugrid.connectivity import Connectivity
    from pyramids.netcdf.ugrid.mesh import Mesh2d

    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 1.0, 0.0, 2.0]),
        node_y=np.array([0.0, 0.0, 1.0, 1.0, 0.5]),
        face_node_connectivity=Connectivity(
            data=np.array([[0, 1, 2, 3], [1, 4, 2, -1]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )
    web_map = WebMap().trimesh(mesh, name="bed")
    assert web_map.layer_ids == ["bed"], web_map.layer_ids
    # Removable by id, like every other layer.
    web_map.remove_layer("bed")
    assert web_map.layer_ids == [], web_map.layer_ids


def test_quickmap_refuses_a_non_mesh_kind_on_the_web_tier(ugrid_mesh):
    """A ``kind`` other than the mesh render is refused rather than silently ignored.

    Test scenario:
        The public ``quickmap`` refuses a named renderer on this tier at the capability gate (the web tier has
        no ``raster_renderer``), and the ``_quickmap_web`` mesh branch's own guard refuses one too — both name
        the argument rather than drawing the wrong thing.
    """
    # Built above each block so only the refused call can raise inside the `with` (the refusal-block guard
    # in tests/base/test_refusal_blocks.py).
    dataset = ugrid_mesh("face")
    with pytest.raises(ValueError, match="kind"):
        quickmap(dataset, backend="web", kind="contourf")
    other = ugrid_mesh("face")
    with pytest.raises(ValueError, match="not a renderer for a mesh"):
        api._quickmap_web(other, kind="contourf")
