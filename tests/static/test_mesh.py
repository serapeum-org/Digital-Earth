"""Tests for T4.1 — Map unstructured methods (tricontourf/tricontour/tripcolor) via MeshGlyph."""

import pytest
from pyramids.base.crs import crs_from_user_input

from digitalearth.static import Map
from digitalearth.static.view import projections


@pytest.mark.parametrize("kind", ["tricontourf", "tricontour", "tripcolor"])
def test_tri_methods_from_dataset(dataset, kind):
    """Each tri method triangulates the raster's valid cell centres and renders one layer."""
    m = Map(crs=dataset.epsg)
    getattr(m, kind)(dataset)
    assert len(m.layers) == 1
    assert m.ax.collections


def test_tricontourf_from_features():
    """tricontourf works on a FeatureCollection of points with a numeric value column."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    m = Map(crs=fc.epsg)
    m.tricontourf(fc)
    assert len(m.layers) == 1


def test_tri_requires_value_column():
    """A FeatureCollection without a numeric column raises a clear error."""
    from pyramids.feature import FeatureCollection

    fc = FeatureCollection.read_file("tests/data/points.geojson")
    fc = fc.drop(columns=[c for c in fc.columns if c != fc.geometry.name])
    m = Map(crs=fc.epsg)
    with pytest.raises(ValueError, match="no numeric value column"):
        m.tricontourf(fc)


def test_tricontour_labels(dataset):
    """tricontour(labels=True) draws inline isoline labels (cleopatra 0.15.0 / cleopatra#151)."""
    m = Map(crs=dataset.epsg)
    m.tricontour(dataset, labels=True)
    assert len(m.ax.texts) > 0, (
        "labels=True should add inline tricontour-label Text artists"
    )


def test_tricontour_no_labels_by_default(dataset):
    """tricontour without labels= draws no inline labels (no behaviour change)."""
    m = Map(crs=dataset.epsg)
    m.tricontour(dataset)
    assert len(m.ax.texts) == 0, "default tricontour must not add label artists"


def test_tricontourf_labels_is_noop(dataset):
    """tricontourf(labels=True) is a no-op — filled tri-contours are not line-labelled."""
    m = Map(crs=dataset.epsg)
    m.tricontourf(dataset, labels=True)
    assert len(m.ax.texts) == 0, (
        "filled tricontourf must ignore labels= (no label artists)"
    )


@pytest.mark.parametrize("kind", ["tricontourf", "tricontour"])
def test_contour_refuses_a_mesh_whose_values_sit_on_the_faces(ugrid_mesh, kind):
    """A contour render refuses a face-located mesh and names the render that shades faces.

    Args:
        ugrid_mesh: Factory for the two-face mesh.
        kind: The contour verb under test.

    Test scenario:
        Contours interpolate between values at the nodes, which a face-located mesh does not have.
        Averaging the faces onto the nodes to make it work would invent data, so it is refused instead.
    """
    mesh = ugrid_mesh("face")
    verb = getattr(Map(), kind)
    with pytest.raises(ValueError, match="values sit on its"):
        verb(mesh)


def test_tripcolor_skips_a_mesh_no_projection_can_place(ugrid_mesh, caplog):
    """A mesh whose nodes are all unplaceable is skipped as off-limb, not drawn half-way.

    Args:
        ugrid_mesh: Factory for the two-face mesh.
        caplog: pytest log capture.

    Test scenario:
        Every node is non-finite, so no triangle keeps all three. That is the mesh equivalent of an
        off-limb raster: the verb records the skip and draws nothing, rather than handing matplotlib a
        triangulation full of infinities.
    """
    import logging

    import numpy as np

    unplaceable = ugrid_mesh("face", node_x=np.full(5, np.inf))
    scene = Map()
    with caplog.at_level(logging.WARNING):
        scene.tripcolor(unplaceable)
    assert not scene.ax.collections, "an unplaceable mesh should have drawn nothing"
    assert "tripcolor" in caplog.text, f"the skip should name the verb: {caplog.text!r}"
    assert "none of the data could be placed" in caplog.text, (
        f"the skip should be recorded, not silent: {caplog.text!r}"
    )
    assert "raster" not in caplog.text, (
        f"a mesh was skipped, so the message must not blame a raster: {caplog.text!r}"
    )


def test_tripcolor_keeps_a_finite_view_when_part_of_the_mesh_is_unplaceable(ugrid_mesh):
    """A partly unplaceable mesh still frames the triangles it did draw (#499 review H3).

    Test scenario:
        One node of the quad/triangle mesh is sent to infinity, which drops the one triangle leaning on it
        and keeps the other two. Leaving the infinite node in the coordinate arrays poisoned the extent, so
        the axes collapsed to about a tenth of a unit and the surviving triangles were drawn far outside the
        view — a blank figure with no warning. The view must enclose what was drawn.
    """
    import numpy as np

    partly = ugrid_mesh("face", node_x=np.array([0.0, 1.0, 1.0, 0.0, np.inf]))
    scene = Map()
    scene.tripcolor(partly)
    (x0, y0), (x1, y1) = scene.ax.dataLim.get_points()
    assert np.isfinite([x0, y0, x1, y1]).all(), (
        f"the extent is not finite: {(x0, y0, x1, y1)}"
    )
    assert x1 - x0 >= 0.9, (
        f"x extent collapsed to {x1 - x0}, so the drawn triangles are off-view"
    )
    assert y1 - y0 >= 0.9, (
        f"y extent collapsed to {y1 - y0}, so the drawn triangles are off-view"
    )


@pytest.mark.parametrize(
    "display",
    [
        pytest.param(3857, id="int-code"),
        pytest.param("EPSG:3857", id="epsg-string"),
        pytest.param("+proj=robin", id="proj4-robinson"),
        pytest.param(projections.orthographic(0.0, 0.0), id="orthographic"),
        pytest.param(crs_from_user_input("EPSG:3857").to_wkt(), id="wkt"),
        pytest.param("ESRI:54030", id="esri-code"),
    ],
)
def test_a_crs_carrying_mesh_draws_in_any_display_crs_spelling(ugrid_mesh, display):
    """A mesh that declares a CRS draws in any display CRS spelling pyproj accepts (#507).

    Args:
        ugrid_mesh: Factory for the two-face mesh.
        display: The display CRS in each spelling `pyproj.CRS.from_user_input` accepts — an EPSG int, the
            `"EPSG:<code>"` string, a `"+proj=..."` definition, a projection carrying no EPSG code, a full
            WKT string, and an `"ESRI:<code>"` code.

    Test scenario:
        pyramids-core >= 0.73.0 (serapeum-org/pyramids-core#1404) warps a mesh to a display CRS given in
        any form `pyproj.CRS.from_user_input` accepts, so an orthographic or Robinson projection, a WKT
        string or an ESRI code is drawn as readily as a coded one. No EPSG code is required and no refusal
        escapes a visualization call — the mesh is the last layer family that used to need special-casing
        here, and no longer does.
    """
    mesh = ugrid_mesh("face", epsg=4326)
    scene = Map(crs=display)
    scene.tripcolor(mesh)
    assert scene.ax.collections, "a CRS-carrying mesh drew nothing"


def test_a_crs_carrying_mesh_partly_beyond_the_limb_draws_the_survivors(
    ugrid_mesh, caplog
):
    """A mesh reprojected into an orthographic globe drops its off-limb triangles and frames the rest (#507).

    Args:
        ugrid_mesh: Factory for the two-face mesh.
        caplog: pytest log capture.

    Test scenario:
        The mesh declares EPSG:4326 and one node sits at lon 179 — the far side of an orthographic globe
        centred on (0, 0). That node warps to non-finite coordinates, so the one triangle leaning on it is
        dropped while the near-side quad survives. Until pyramids-core accepted a projection with no EPSG
        code, no test could reproject a mesh into such a CRS at all, so `_placeable_triangles` /
        `_compacted_nodes` had only ever been driven by synthetic infinities; this drives the off-limb path
        with a real warp. The survivors must be drawn inside a finite extent, and the partial drop must be
        recorded rather than silent.
    """
    import logging

    import numpy as np

    partly = ugrid_mesh("face", epsg=4326, node_x=np.array([0.0, 1.0, 1.0, 0.0, 179.0]))
    scene = Map(crs=projections.orthographic(0.0, 0.0))
    with caplog.at_level(logging.WARNING):
        scene.tripcolor(partly)
    assert scene.ax.collections, "the near-side triangles drew nothing"
    (x0, y0), (x1, y1) = scene.ax.dataLim.get_points()
    assert np.isfinite([x0, y0, x1, y1]).all(), (
        f"the extent is not finite: {(x0, y0, x1, y1)}"
    )
    assert x1 > x0, f"the x extent did not enclose the drawn triangles: {(x0, x1)}"
    assert y1 > y0, f"the y extent did not enclose the drawn triangles: {(y0, y1)}"
    assert "tripcolor" in caplog.text, f"the skip should name the verb: {caplog.text!r}"
    assert "fall outside" in caplog.text, (
        f"the partial drop should be recorded: {caplog.text!r}"
    )


def test_a_crs_carrying_mesh_wholly_beyond_the_limb_is_skipped_as_off_limb(
    ugrid_mesh, caplog
):
    """A CRS-carrying mesh whose every node warps off the limb draws nothing, and the skip is logged (#507).

    Args:
        ugrid_mesh: Factory for the two-face mesh.
        caplog: pytest log capture.

    Test scenario:
        Every node sits near lon 178 — the far side of an orthographic globe centred on (0, 0) — so the warp
        sends all of them non-finite and no triangle keeps its three nodes. This drives the fully off-limb
        path through a *real* reprojection: `_placed_mesh` reaches `reproject` (the mesh declares EPSG:4326),
        an `OffLimbError` is raised, and `_tri` answers it with an empty frame. The synthetic-infinity sibling
        (`test_tripcolor_skips_a_mesh_no_projection_can_place`) returns early via the no-CRS guard and never
        warps, so this is the only coverage of the CRS-carrying skip. The assertion pins the user-visible
        contract — nothing drawn and a verb-named skip — rather than *where* the error was raised, so it
        stays valid if a future pyramids moves the off-limb detection into the warp itself.
    """
    import logging

    import numpy as np

    off_limb = ugrid_mesh(
        "face", epsg=4326, node_x=np.array([179.0, 178.0, 178.0, 179.0, 177.0])
    )
    scene = Map(crs=projections.orthographic(0.0, 0.0))
    with caplog.at_level(logging.WARNING):
        scene.tripcolor(off_limb)
    assert not scene.ax.collections, "a wholly off-limb mesh should have drawn nothing"
    assert "tripcolor" in caplog.text, f"the skip should name the verb: {caplog.text!r}"
    assert "none of the data could be placed" in caplog.text, (
        f"the off-limb skip should be recorded: {caplog.text!r}"
    )


def _two_triangle_mesh(location="node", node_x=None):
    """Build a mesh of two triangles sharing one node, with values on ``location``.

    Faces ``[0, 1, 2]`` and ``[2, 3, 4]`` share only node 2, so making node 1 unplaceable drops the first
    face and keeps the second — which is what makes a *shifted* renumbering observable.

    Args:
        location: ``"node"`` or ``"face"``.
        node_x: Overrides the node x coordinates.

    Returns:
        The pyramids ``UgridDataset``.
    """
    import numpy as np
    from pyramids.netcdf.ugrid.connectivity import Connectivity
    from pyramids.netcdf.ugrid.dataset import UgridDataset
    from pyramids.netcdf.ugrid.mesh import Mesh2d
    from pyramids.netcdf.ugrid.models import MeshVariable

    mesh = Mesh2d(
        node_x=np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        if node_x is None
        else np.asarray(node_x),
        node_y=np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        face_node_connectivity=Connectivity(
            data=np.array([[0, 1, 2], [2, 3, 4]], dtype=np.intp),
            fill_value=-1,
            cf_role="face_node_connectivity",
            original_start_index=0,
        ),
    )
    values = (
        np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        if location == "node"
        else np.array([10.0, 20.0])
    )
    name = "zs" if location == "node" else "depth"
    return UgridDataset(
        mesh=mesh,
        data_variables={
            name: MeshVariable(
                name=name,
                location=location,
                mesh_name="m",
                shape=values.shape,
                _data=values,
            )
        },
        global_attributes={},
    )


def test_node_located_values_follow_their_nodes_through_the_compaction():
    """A node-located mesh keeps each node's own value when unplaceable nodes are dropped.

    Test scenario:
        Node 1 is unplaceable, so the face [0, 1, 2] goes and [2, 3, 4] stays. The surviving nodes are
        2, 3 and 4, whose values are 30, 40 and 50 — mean 40. Compacting the coordinates without
        compacting the values left the renumbered indices pointing at the *original* array, which drew
        mean(10, 20, 30) = 20 instead: wrong colours, silently (round-2 H1).
    """
    import numpy as np

    partly = _two_triangle_mesh("node", node_x=np.array([0.0, np.inf, 2.0, 3.0, 4.0]))
    scene = Map()
    scene.tripcolor(partly)
    drawn = np.asarray(scene.ax.collections[-1].get_array()).tolist()
    assert drawn == [40.0], f"expected the surviving nodes' mean 40.0, drew {drawn}"


@pytest.mark.parametrize("kind", ["tricontourf", "tricontour"])
def test_node_located_contours_survive_the_compaction(kind):
    """A contour render of a partly unplaceable node-located mesh draws instead of failing raw.

    Args:
        kind: The contour verb under test.

    Test scenario:
        Leaving the values uncompacted handed matplotlib five values for three nodes, which it answered
        with `ValueError: data length (5) does not match n_nodes (3)` — a message about an array the
        caller never built.
    """
    import numpy as np

    partly = _two_triangle_mesh("node", node_x=np.array([0.0, np.inf, 2.0, 3.0, 4.0]))
    scene = Map()
    getattr(scene, kind)(partly)
    assert scene.ax.collections, (
        f"{kind} drew nothing for a partly unplaceable node-located mesh"
    )


def test_a_mesh_already_in_the_display_crs_needs_no_epsg_code(ugrid_mesh):
    """A mesh already in the display CRS is drawn, even where that CRS has no EPSG code (round-2 M3).

    Test scenario:
        The upstream restriction is about *warping* a mesh, so it cannot apply when no warp is required.
        A mesh whose own CRS is the display CRS was nevertheless refused, because the guard resolved the
        code before asking whether a reprojection was needed at all.
    """
    from pyramids.base.crs import crs_from_user_input

    robinson = crs_from_user_input("+proj=robin").to_wkt()
    mesh = ugrid_mesh("face")
    mesh = mesh.__class__(
        mesh=mesh.mesh,
        data_variables=dict(mesh.data_vars),
        global_attributes={},
        crs_wkt=robinson,
    )
    scene = Map(crs="+proj=robin")
    scene.tripcolor(mesh)
    assert scene.ax.collections, "a mesh already in the display CRS should draw"
